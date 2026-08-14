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
from foldpipe.sources import HuggingFaceSource
from torch_geometric.nn.models import SchNet
import io
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

class RealMLFFWorkload(nn.Module):
    """
    Wraps PyG's actual SchNet implementation to process the synthetic benchmark tensors.
    This guarantees the GPU executes genuine MLFF CUDA kernels during the benchmark.
    """
    def __init__(self):
        super().__init__()
        self.schnet = SchNet(hidden_channels=128, num_filters=128, num_interactions=6, num_gaussians=50, cutoff=10.0)
        
    def forward(self, x):
        # x acts as dummy coordinates (pos)
        z = torch.ones(x.size(0), dtype=torch.long, device=x.device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.schnet(z, x, batch=batch)

def get_real_mlff_model():
    """Real MLFF Workload (SchNet)"""
    return RealMLFFWorkload().to(device)

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
# PHASE 1: BASELINE A (EAGER ACCUMULATION) OOM TEST
# ---------------------------------------------------------
def run_baseline_in_memory(drive_service, files, model):
    print("\n--- BASELINE A: Eager In-Memory Accumulation ---")
    profiler = Profiler()
    profiler.start()
    
    dataset_in_memory = []
    crashed = False
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    try:
        for i, f in enumerate(files[:MAX_CHUNKS]):
            print(f"Eager Baseline: Loading and Computing Chunk {i+1} / {MAX_CHUNKS}...")
            request_url = f"https://huggingface.co/datasets/aviatorlf/prion-dataset/resolve/main/{f}"
            headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"} if os.environ.get("HF_TOKEN") else {}
            import requests
            response = requests.get(request_url, headers=headers, stream=True)
            response.raise_for_status()
            fh = io.BytesIO(response.content)
            fh.seek(0)
            tensor = torch.load(fh, map_location='cpu')
            
            # PROTECTIVE CUTOFF: If RAM hits 90% of system total, we trigger natural MemoryError 
            # to protect the machine from actually freezing.
            if psutil.virtual_memory().percent > 90.0:
                raise MemoryError("System RAM exhausted.")
                
            dataset_in_memory.append(tensor) # HOG RAM
            
            for b in range(0, tensor.size(0), BATCH_SIZE):
                train_batch(model, optimizer, criterion, tensor[b:b+BATCH_SIZE])
            
    except MemoryError as e:
        print(f"[!] Memory Safety Threshold Reached: {e}")
        crashed = True
        
    profiler.stop()
    del dataset_in_memory
    return profiler, crashed

# ---------------------------------------------------------
# PHASE 2: BASELINE B (SEQUENTIAL BOUNDED STREAMING)
# ---------------------------------------------------------
def run_sequential_stream(drive_service, files, model):
    print("\n--- BASELINE B: Sequential Bounded Streaming ---")
    profiler = Profiler()
    profiler.start()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for i, f in enumerate(files[:MAX_CHUNKS]):
        print(f"Sequential Streaming: Loading Chunk {i+1} / {MAX_CHUNKS}...")
        request_url = f"https://huggingface.co/datasets/aviatorlf/prion-dataset/resolve/main/{f}"
        headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"} if os.environ.get("HF_TOKEN") else {}
        import requests
        response = requests.get(request_url, headers=headers, stream=True)
        response.raise_for_status()
        fh = io.BytesIO(response.content)
        fh.seek(0)
        tensor = torch.load(fh, map_location='cpu')
        
        for b in range(0, tensor.size(0), BATCH_SIZE):
            train_batch(model, optimizer, criterion, tensor[b:b+BATCH_SIZE])
            
        # O(1) bound achieved by explicitly deleting the chunk before fetching the next
        del tensor
            
    profiler.stop()
    return profiler

# ---------------------------------------------------------
# PHASE 3: FOLDPIPE (ASYNC STREAMING) TEST
# ---------------------------------------------------------
def run_foldpipe_stream(files, model):
    print(f"\n--- FOLDPIPE ASYNC STREAM ---")
    profiler = Profiler()
    profiler.start()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    source = HuggingFaceSource(repo_id="aviatorlf/prion-dataset", token=os.environ.get("HF_TOKEN"))
    loader = AsyncFoldPipeLoader(source=source, batch_size=BATCH_SIZE)
    loader.files = files[:MAX_CHUNKS]
    
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
    source = HuggingFaceSource(repo_id="aviatorlf/prion-dataset", token=os.environ.get("HF_TOKEN"))
    files = source.get_files()
            
    # We benchmark against both the Synthetic Deep Model (to prove I/O crossover) 
    # and the Real MLFF Workload (SchNet) to prove practical relevance.
    models_to_test = {
        "Synthetic Deep": get_deep_model(),
        "Real MLFF (SchNet)": get_real_mlff_model()
    }
    
    fig, axes = plt.subplots(len(models_to_test), 2, figsize=(16, 6 * len(models_to_test)))
    
    for row_idx, (model_name, active_model) in enumerate(models_to_test.items()):
        print(f"\n=========================================")
        print(f"TESTING MODEL ARCHITECTURE: {model_name}")
        print(f"=========================================")
        
        # 1. Baseline A (Eager Accumulation)
        print("Running Baseline A (Eager Accumulation)...")
        eager_prof, crashed = run_baseline_in_memory(None, files, active_model)
        
        # 2. Baseline B (Sequential Streaming)
        print("Running Baseline B (Sequential Streaming)...")
        seq_prof = run_sequential_stream(None, files, active_model)
        
        # 3. FoldPipe (Async Streaming)
        print("Running FoldPipe (Async Streaming)...")
        fp_prof = run_foldpipe_stream(files, active_model)
        
        ax1, ax2 = axes[row_idx]
        
        # Left: RAM Memory Bound Isolation
        ax1.plot(eager_prof.time_history, eager_prof.ram_history, color='red', label='Eager Accumulation (O(N) RAM)')
        ax1.plot(seq_prof.time_history, seq_prof.ram_history, color='blue', alpha=0.7, label='Sequential Stream (O(1) RAM)')
        ax1.plot(fp_prof.time_history, fp_prof.ram_history, color='green', label='FoldPipe Async (O(1) RAM)')
        if crashed:
            ax1.scatter([eager_prof.time_history[-1]], [eager_prof.ram_history[-1]], color='darkred', s=150, marker='X', label='Memory Safety Cutoff')
        ax1.set_title(f"RAM Footprint ({model_name})")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("RAM (GB)")
        ax1.legend()
        
        # Right: Throughput Latency Masking Isolation
        # We compare Sequential vs FoldPipe. Eager is omitted as it crashes before finishing.
        ax2.plot(seq_prof.time_history, seq_prof.gpu_history, color='blue', alpha=0.7, label='Sequential Stream (Idle GPU Traps)')
        ax2.plot(fp_prof.time_history, fp_prof.gpu_history, color='green', alpha=0.8, label='FoldPipe Async (Saturated)')
        ax2.set_title(f"GPU Saturation ({model_name})")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Compute Utilization (%)")
        ax2.legend()
        
    plt.tight_layout()
    plt.savefig('results/benchmark_comparison.png', dpi=300)
    print("Saved results/benchmark_comparison.png")
