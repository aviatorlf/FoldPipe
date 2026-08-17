import os
import json
import tempfile
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaIoBaseDownload
from huggingface_hub import HfApi

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Please set the HF_TOKEN environment variable")
HF_REPO = "aviatorlf/prion-dataset"

# Google Drive configuration
GDRIVE_FOLDER_ID = "1Few5wzRuuhlwbj4DJD9nkOP98t_QqZcz"
GDRIVE_SECRET_PATH = os.environ.get(
    "GDRIVE_SECRET_PATH",
    "/kaggle/input/gcp-secret-dataset/token.json",
)

def migrate_gdrive_to_hf():
    print("Authenticating with Google Drive...")
    if not os.path.exists(GDRIVE_SECRET_PATH):
        raise FileNotFoundError(f"Missing {GDRIVE_SECRET_PATH}")
        
    with open(GDRIVE_SECRET_PATH, 'r') as f:
        creds_json = json.load(f)
        
    creds = Credentials.from_authorized_user_info(creds_json, scopes=['https://www.googleapis.com/auth/drive'])
    drive_service = build('drive', 'v3', credentials=creds)
    
    print("Authenticating with HuggingFace Hub...")
    hf_api = HfApi(token=HF_TOKEN)
    
    # 1. Fetch all checkpoint files from the Drive folder
    print("Paginating Google Drive files...")
    query = f"'{GDRIVE_FOLDER_ID}' in parents and name contains 'checkpoint_batch_' and trashed = false"
    files = []
    page_token = None
    
    while True:
        results = drive_service.files().list(
            q=query, 
            fields="nextPageToken, files(id, name)", 
            orderBy="name_natural",
            pageToken=page_token
        ).execute()
        files.extend(results.get('files', []))
        page_token = results.get('nextPageToken')
        if not page_token:
            break
            
    print(f"Found {len(files)} files to migrate.")
    
    # 2. Sequentially stream from Drive to HF using local temporary storage
    # Kaggle instances have fast ephemeral disk and massive datacenter bandwidth
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, f in enumerate(files):
            file_id = f['id']
            file_name = f['name']
            local_path = os.path.join(temp_dir, file_name)
            
            print(f"[{i+1}/{len(files)}] Downloading {file_name} from Google Drive...")
            request = drive_service.files().get_media(fileId=file_id)
            
            with open(local_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        print(f"  Drive Download: {int(status.progress() * 100)}%", end='\r')
            
            print(f"\n[{i+1}/{len(files)}] Uploading {file_name} to HuggingFace...")
            hf_api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"trajectories/{file_name}",
                repo_id=HF_REPO,
                repo_type="dataset"
            )
            
            # Explicitly delete the 650MB+ file to maintain Kaggle disk space limits
            os.remove(local_path)
            
    print("\nMigration Complete! All files are now hosted on HuggingFace.")

if __name__ == "__main__":
    migrate_gdrive_to_hf()
