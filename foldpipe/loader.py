import io
import gc
import torch
import concurrent.futures
from torch.utils.data import IterableDataset
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials

class AsyncFoldPipeLoader(IterableDataset):
    """
    True O(1) bounded-memory asynchronous streaming dataloader.
    Downloads native PyTorch .pt chunk files from Google Drive in the background.
    Hides network I/O latency behind GPU computation.
    """
    def __init__(self, drive_folder_id, credentials_json, batch_size=128):
        self.drive_folder_id = drive_folder_id
        self.batch_size = batch_size
        self.creds_json = credentials_json
        
        # Build Drive Service
        creds = Credentials.from_authorized_user_info(self.creds_json, scopes=['https://www.googleapis.com/auth/drive'])
        self.drive_service = build('drive', 'v3', credentials=creds)
        
        # Get list of files
        query = f"'{self.drive_folder_id}' in parents and name contains 'checkpoint_batch_' and trashed = false"
        results = self.drive_service.files().list(q=query, fields="files(id, name)").execute()
        self.files = results.get('files', [])
        
    def _download_chunk(self, file_id):
        """Producer function: executed in background thread."""
        request = self.drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        fh.seek(0)
        return torch.load(fh, map_location='cpu')

    def __iter__(self):
        """Consumer pipeline."""
        if not self.files:
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Kick off the prefetch for Chunk 0
            future_chunk = executor.submit(self._download_chunk, self.files[0]['id'])
            
            for i in range(len(self.files)):
                # Block only if GPU is faster than network
                chunk_tensor = future_chunk.result()
                
                # Kick off prefetch for Chunk N+1 immediately
                if i + 1 < len(self.files):
                    future_chunk = executor.submit(self._download_chunk, self.files[i+1]['id'])
                
                # Yield batches to the GPU
                for b in range(0, chunk_tensor.size(0), self.batch_size):
                    yield chunk_tensor[b:b+self.batch_size]
                
                # Explicitly delete the chunk and garbage collect to guarantee O(1) memory bound
                del chunk_tensor
                gc.collect()
