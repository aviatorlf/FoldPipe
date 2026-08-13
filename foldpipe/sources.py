import io
import json
import torch
import requests
from abc import ABC, abstractmethod
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from huggingface_hub import HfFileSystem

class Source(ABC):
    @abstractmethod
    def get_files(self):
        """Returns a list of identifiers for the chunk files."""
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

    def get_files(self):
        files = []
        query = f"'{self.folder_id}' in parents and name contains 'checkpoint_batch_' and trashed = false"
        page_token = None
        while True:
            results = self.drive_service.files().list(
                q=query, 
                fields="nextPageToken, files(id, name)", 
                orderBy="name_natural",
                pageToken=page_token
            ).execute()
            files.extend(results.get('files', []))
            page_token = results.get('nextPageToken')
            if not page_token:
                break
        return files

    def download_chunk(self, identifier):
        file_id = identifier['id']
        request = self.drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        fh.seek(0)
        return torch.load(fh, map_location='cpu')


class HuggingFaceSource(Source):
    def __init__(self, repo_id, folder_path="", token=None):
        self.repo_id = repo_id
        self.folder_path = folder_path.strip("/")
        self.token = token
        self.fs = HfFileSystem(token=token)

    def get_files(self):
        path = f"datasets/{self.repo_id}/{self.folder_path}".strip("/")
        files_info = self.fs.ls(path)
        # Filter for checkpoint files and return the full paths
        files = [f["name"] for f in files_info if "checkpoint_batch_" in f["name"]]
        
        # Sort naturally (PyTorch style)
        import re
        def natural_keys(text):
            return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]
        files.sort(key=natural_keys)
        
        return files

    def download_chunk(self, identifier):
        file_path = identifier
        # We strip the leading "datasets/" prefix from HfFileSystem if it exists
        if file_path.startswith(f"datasets/{self.repo_id}/"):
            file_path = file_path[len(f"datasets/{self.repo_id}/"):]
            
        url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{file_path}"
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
            
        # Stream directly to RAM via requests, completely avoiding local disk cache
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        fh = io.BytesIO(response.content)
        fh.seek(0)
        return torch.load(fh, map_location='cpu')
