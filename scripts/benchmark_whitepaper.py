import os
import time
import json
import psutil
import torch
import torch.nn as nn
import threading
import subprocess
import matplotlib.pyplot as plt
from foldpipe import AsyncFoldPipeLoader
from foldpipe.sources import GoogleDriveSource
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import glob
import io
from googleapiclient.http import MediaIoBaseDownload

DRIVE_FOLDER_ID = "1Few5wzRuuhlwbj4DJD9nkOP98t_QqZcz"
MAX_CHUNKS = 15  # Limit to avoid running all day
BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('results', exist_ok=True)

# ---------------------------------------------------------
# PROFILER
# ---------------------------------------------------------
class Profiler:
    def __init__(self):
        self.running = False
        self.ram_history = []
        self.gpu_history = []
        self.time_history = []
        self.start_time = 0
        self.process = psutil.Process(os.getpid())
        
    def _poll(self):
        while self.running:
            self.time_history.append(time.time() - self.start_time)
            ram_gb = self.process.memory_info().rss / (1024 ** 3)
            self.ram_history.append(ram_gb)
            
            util = 0.0
            if torch.cuda.is_available():
                try:
                    res = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                        encoding='utf-8'
                    )
                    util = float(res.strip().split('\n')[0])
                except Exception:
                    pass
            self.gpu_history.append(util)
            time.sleep(0.5)

    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._poll)
        self.thread.start()
        
    def stop(self):
        self.running = False
        self.thread.join()

# ---------------------------------------------------------
# MODELS (SYMMETRIC WORKLOADS)
# ---------------------------------------------------------
def get_tiny_model():
    """Compute < Network Fetch (I/O Bound)"""
    return nn.Sequential(nn.Linear(3, 16), nn.Linear(16, 3)).to(device)

def get_deep_model():
    """Compute > Network Fetch (Compute Bound)"""
    layers = [nn.Linear(3, 256), nn.ReLU()]
    for _ in range(60):  # Massively deep to guarantee compute saturation in a single pass
        layers.extend([nn.Linear(256, 256), nn.ReLU()])
    layers.append(nn.Linear(256, 3))
    return nn.Sequential(*layers).to(device)

def train_batch(model, optimizer, criterion, mini_batch):
    """Exact identical training step run by both pipelines."""
    mini_batch = mini_batch.to(device, non_blocking=True)
    optimizer.zero_grad()
    out = model(mini_batch)
    loss = criterion(out, torch.zeros_like(out))
    loss.backward()
    optimizer.step()
    
    # Artificial sleep to simulate heavy CUDA kernels if running on fast M-series CPUs locally
    if not torch.cuda.is_available():
        time.sleep(0.01)

# ---------------------------------------------------------
# PHASE 1: BASELINE (IN MEMORY) OOM TEST
# ---------------------------------------------------------
def run_baseline_in_memory(drive_service, files, model):
    print("\n--- BASELINE: PyG InMemoryDataset ---")
    profiler = Profiler()
    profiler.start()
    
    dataset_in_memory = []
    crashed = False
    
    # Symmetrical optimizer instantiated ONCE
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    try:
        for i, f in enumerate(files[:MAX_CHUNKS]):
            print(f"Baseline: Loading and Computing Chunk {i+1} / {MAX_CHUNKS}...")
            request = drive_service.files().get_media(fileId=f['id'])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            fh.seek(0)
            tensor = torch.load(fh, map_location='cpu')
            
            # PROTECTIVE CUTOFF: If RAM hits 90% of system total, we trigger natural MemoryError 
            # to protect the machine from actually freezing.
            if psutil.virtual_memory().percent > 90.0:
                raise MemoryError("System RAM exhausted.")
                
            dataset_in_memory.append(tensor) # HOG RAM
            
            # Single-pass execution of the chunk
            for b in range(0, tensor.size(0), BATCH_SIZE):
                train_batch(model, optimizer, criterion, tensor[b:b+BATCH_SIZE])
            
    except MemoryError as e:
        print(f"[!] Memory Safety Threshold Reached: {e}")
        crashed = True
        
    profiler.stop()
    del dataset_in_memory
    import gc
    gc.collect()
    return profiler, crashed

# ---------------------------------------------------------
# PHASE 2: FOLDPIPE (ASYNC STREAMING) TEST
# ---------------------------------------------------------
def run_foldpipe_stream(creds_json, files, model):
    print(f"\n--- FOLDPIPE ASYNC STREAM ---")
    profiler = Profiler()
    profiler.start()
    
    # Symmetrical optimizer instantiated ONCE
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Initialize the True AsyncFoldPipeLoader with GoogleDriveSource
    source = GoogleDriveSource(folder_id=DRIVE_FOLDER_ID, credentials_json=creds_json)
    loader = AsyncFoldPipeLoader(source=source, batch_size=BATCH_SIZE)
    # Hack the loader to only fetch MAX_CHUNKS for the benchmark
    loader.files = files[:MAX_CHUNKS]
    
    # The loader is a drop-in generator yielding continuous batches
    for batch_idx, mini_batch in enumerate(loader):
        if batch_idx % 200 == 0:
            print(f"GPU Computing Batch {batch_idx+1}...")
        train_batch(model, optimizer, criterion, mini_batch)

    profiler.stop()
    return profiler

# ---------------------------------------------------------
# EXECUTION & PLOTTING
# ---------------------------------------------------------
if __name__ == "__main__":
    import sys
    secret_path = "/kaggle/input/gcp-secret-dataset/token.json"
    if not os.path.exists(secret_path):
        possible_paths = glob.glob('/kaggle/input/**/token.json', recursive=True)
        if not possible_paths:
            possible_paths = glob.glob('**/token.json', recursive=True)
        secret_path = possible_paths[0] if possible_paths else None
        
    if not secret_path:
        print("Requires token.json for Google Drive OAuth.")
        sys.exit(1)
        
    with open(secret_path, 'r') as f:
        creds_json = json.load(f)
        
    creds = Credentials.from_authorized_user_info(creds_json, scopes=['https://www.googleapis.com/auth/drive'])
    drive_service = build('drive', 'v3', credentials=creds)
    
    # Pagination
    query = f"'{DRIVE_FOLDER_ID}' in parents and name contains 'checkpoint_batch_' and trashed = false"
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
    
    # 1. Baseline Test (Heavy Compute)
    print("Running Baseline...")
    base_prof, crashed = run_baseline_in_memory(drive_service, files, get_deep_model())
    
    # 2. FoldPipe (Compute > I/O) -> Saturation
    print("Running FoldPipe (Compute Bound)...")
    fp_deep_prof = run_foldpipe_stream(creds_json, files, get_deep_model())
    
    # 3. FoldPipe (Compute < I/O) -> Starvation
    print("Running FoldPipe (I/O Bound)...")
    fp_tiny_prof = run_foldpipe_stream(creds_json, files, get_tiny_model())
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: RAM Leak vs Stable
    # Maintained pedantic accuracy label requested previously, and corrected PyG name
    ax1.plot(base_prof.time_history, base_prof.ram_history, color='red', label='Eager in-memory accumulation')
    ax1.plot(fp_deep_prof.time_history, fp_deep_prof.ram_history, color='green', label='FoldPipe (Bounded)')
    if crashed:
        ax1.scatter([base_prof.time_history[-1]], [base_prof.ram_history[-1]], color='darkred', s=150, marker='X', label='Memory Safety Cutoff')
    ax1.set_title("RAM Footprint")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("RAM (GB)")
    ax1.legend()
    
    # Right: Compute-to-I/O Crossover
    ax2.plot(fp_deep_prof.time_history, fp_deep_prof.gpu_history, color='green', alpha=0.8, label='Compute > I/O (Saturation)')
    ax2.plot(fp_tiny_prof.time_history, fp_tiny_prof.gpu_history, color='orange', alpha=0.8, label='Compute < I/O (Starvation)')
    ax2.set_title("FoldPipe Latency Masking (Compute-to-I/O Ratio)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Compute Utilization (%)")
    ax2.legend()
    
    plt.savefig('results/benchmark_comparison.png', dpi=300)
    print("Saved results/benchmark_comparison.png")
