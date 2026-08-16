import io
import json
import time
import torch
import requests
from urllib.parse import quote
from abc import ABC, abstractmethod
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from huggingface_hub import HfFileSystem

class Source(ABC):
    @abstractmethod
    def iter_files(self):
        """Yields identifiers for the chunk files lazily to strictly bound memory."""
        pass

    @abstractmethod
    def download_chunk(self, identifier):
        """Downloads a chunk and returns a PyTorch tensor directly in memory."""
        pass


class GoogleDriveSource(Source):
    def __init__(self, folder_id, credentials_json):
        self.folder_id = folder_id
        
        if isinstance(credentials_json, str):
            with open(credentials_json, 'r') as f:
                self.creds_dict = json.load(f)
        else:
            self.creds_dict = credentials_json
            
        # Narrows scope to readonly as requested by peer review
        creds = Credentials.from_authorized_user_info(self.creds_dict, scopes=['https://www.googleapis.com/auth/drive.readonly'])
        self.drive_service = build('drive', 'v3', credentials=creds)

    def iter_files(self):
        query = f"'{self.folder_id}' in parents and name contains 'checkpoint_batch_' and trashed = false"
        page_token = None
        while True:
            results = self.drive_service.files().list(
                q=query, 
                fields="nextPageToken, files(id, name)", 
                orderBy="name_natural",
                pageToken=page_token
            ).execute()
            
            for file_info in results.get('files', []):
                yield file_info
                
            page_token = results.get('nextPageToken')
            if not page_token:
                break

    def download_chunk(self, identifier):
        file_id = identifier['id']
        request = self.drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        fh.seek(0)
        return torch.load(fh, map_location='cpu', weights_only=False)


from huggingface_hub import HfFileSystem, HfApi

class HuggingFaceSource(Source):
    def __init__(
        self,
        repo_id,
        folder_path="",
        token=None,
        revision=None,
        transfer_observer=None,
    ):
        self.repo_id = repo_id
        self.folder_path = folder_path.strip("/")
        self.token = token
        self.revision = revision
        self.transfer_observer = transfer_observer
        self.fs = HfFileSystem(token=token)
        self.api = HfApi(token=token)

    def iter_files(self):
        # Generate items completely lazily directly from the Hugging Face API
        path_in_repo = self.folder_path if self.folder_path else None
        tree_generator = self.api.list_repo_tree(
            repo_id=self.repo_id,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            revision=self.revision,
            recursive=False,
            expand=False
        )
        for item in tree_generator:
            if "checkpoint_batch_" in item.rfilename:
                yield item.rfilename

    def download_chunk(self, identifier):
        file_path = identifier
        # We strip the leading "datasets/" prefix from HfFileSystem if it exists
        if file_path.startswith(f"datasets/{self.repo_id}/"):
            file_path = file_path[len(f"datasets/{self.repo_id}/"):]
            
        revision = quote(self.revision or "main", safe="")
        url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/{revision}/{file_path}"
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        event = {
            "identifier": identifier,
            "download_start": time.perf_counter(),
            "download_finish": None,
            "deserialize_finish": None,
            "bytes_downloaded": 0,
        }

        try:
            # Stream directly to RAM via requests, avoiding local disk cache and
            # double-materialization. Count payload bytes for benchmark tracing.
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()

            fh = io.BytesIO()
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    event["bytes_downloaded"] += len(block)
                    fh.write(block)

            event["download_finish"] = time.perf_counter()
            fh.seek(0)
            chunk = torch.load(fh, map_location='cpu', weights_only=False)
            event["deserialize_finish"] = time.perf_counter()
            return chunk
        except Exception as exc:
            event["error"] = type(exc).__name__
            raise
        finally:
            if self.transfer_observer is not None:
                self.transfer_observer(event.copy())

class SyntheticLatencySource(Source):
    """
    A controlled source that allows configuring exact artificial network latency
    and returns fixed-size dummy tensors. Crucial for isolated mechanism experiments.
    """
    def __init__(self, num_chunks=15, latency_ms=100, chunk_size=20000):
        self.num_chunks = num_chunks
        self.latency_ms = latency_ms
        self.chunk_size = chunk_size
        
    def iter_files(self):
        for i in range(self.num_chunks):
            yield f"synthetic_chunk_{i}.pt"
            
    def download_chunk(self, identifier):
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)
        return torch.randn(self.chunk_size, 3)

class PreenumeratedSource(Source):
    """
    A wrapper source that takes a pre-fetched list of file identifiers and an underlying source.
    This eliminates remote API discovery overhead (e.g. list_repo_tree) during benchmarks,
    ensuring that timing strictly measures the orchestration pipeline.
    """
    def __init__(self, identifiers, underlying_source):
        self.identifiers = identifiers
        self.underlying_source = underlying_source

    def iter_files(self):
        for identifier in self.identifiers:
            yield identifier

    def download_chunk(self, identifier):
        return self.underlying_source.download_chunk(identifier)
