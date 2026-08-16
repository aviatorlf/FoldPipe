import os
import time
import json
import psutil
import torch
import torch.nn as nn
import threading
import subprocess
import copy
import itertools
import matplotlib.pyplot as plt
import numpy as np

# PyG imports
from torch_geometric.data import Batch
from torch_geometric.nn.models import SchNet

from foldpipe import AsyncFoldPipeLoader
from foldpipe.sources import HuggingFaceSource
import io

MAX_CHUNKS = 5  # Smaller limit for MD17 (since chunks might be large)
NUM_RUNS = 3
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
        self.peak_rss = 0
        
    def _poll(self):
        while self.running:
            self.time_history.append(time.time() - self.start_time)
            rss = self.process.memory_info().rss
            self.peak_rss = max(self.peak_rss, rss)
            ram_gb = rss / (1024 ** 3)
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
# BATCHING ABSTRACTION & MODELS
# ---------------------------------------------------------
class LimitedSource:
    def __init__(self, source, max_chunks):
        self.source = source
        self.max_chunks = max_chunks

    def iter_files(self):
        return itertools.islice(self.source.iter_files(), self.max_chunks)

    def download_chunk(self, identifier):
        return self.source.download_chunk(identifier)

def pyg_batch_fn(chunk_list, batch_size=32):
    """Batches a list of PyG Data objects into Batch objects."""
    for i in range(0, len(chunk_list), batch_size):
        yield Batch.from_data_list(chunk_list[i:i+batch_size])

def get_real_mlff_model():
    """Real MLFF Workload (SchNet)"""
    return SchNet(hidden_channels=128, num_filters=128, num_interactions=6, num_gaussians=50, cutoff=10.0).to(device)

def train_batch(model, optimizer, criterion, mini_batch):
    """Genuine Molecular MLFF Optimization Step."""
    mini_batch = mini_batch.to(device)
    optimizer.zero_grad()
    
    # We must require grad on pos to compute forces (dE/dPos)
    mini_batch.pos.requires_grad_(True)
    
    # Forward pass predicts energy
    pred_energy = model(mini_batch.z, mini_batch.pos, batch=mini_batch.batch)
    
    # Target energy might be scalar or batched
    target_energy = mini_batch.energy.view_as(pred_energy) if hasattr(mini_batch, 'energy') else torch.zeros_like(pred_energy)
    
    # Compute forces via autograd derivative (dE/dPos)
    pred_force = -torch.autograd.grad(
        [pred_energy], [mini_batch.pos], 
        grad_outputs=torch.ones_like(pred_energy),
        create_graph=True, retain_graph=True
    )[0]
    
    target_force = mini_batch.force if hasattr(mini_batch, 'force') else torch.zeros_like(pred_force)
    
    # Combined Loss: Energy MSE + Force MSE
    loss_energy = criterion(pred_energy, target_energy)
    loss_force = criterion(pred_force, target_force)
    loss = loss_energy + 10.0 * loss_force
    
    loss.backward()
    optimizer.step()
    
    if not torch.cuda.is_available():
        time.sleep(0.01)

# ---------------------------------------------------------
# PHASE 1: SEQUENTIAL BOUNDED STREAMING
# ---------------------------------------------------------
def run_sequential_stream(files, model, initial_state_dict):
    print("      --- BASELINE: Sequential Bounded Streaming ---")
    torch.manual_seed(42)
    model.load_state_dict(initial_state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    profiler = Profiler()
    profiler.start()
    start_t = time.time()
    
    for i, f in enumerate(files[:MAX_CHUNKS]):
        request_url = f"https://huggingface.co/datasets/aviatorlf/md17-shards/resolve/main/{f}"
        headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"} if os.environ.get("HF_TOKEN") else {}
        import requests
        response = requests.get(request_url, headers=headers, stream=True)
        response.raise_for_status()
        fh = io.BytesIO(response.content)
        fh.seek(0)
        chunk_list = torch.load(fh, map_location='cpu')
        
        for mini_batch in pyg_batch_fn(chunk_list, batch_size=32):
            train_batch(model, optimizer, criterion, mini_batch)
            
        del chunk_list
            
    profiler.stop()
    return profiler, False, time.time() - start_t

# ---------------------------------------------------------
# PHASE 2: FOLDPIPE (ASYNC STREAMING) TEST
# ---------------------------------------------------------
def run_foldpipe_stream(model, initial_state_dict):
    print(f"      --- FOLDPIPE ASYNC STREAM ---")
    torch.manual_seed(42)
    model.load_state_dict(initial_state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    profiler = Profiler()
    profiler.start()
    start_t = time.time()
    
    source = HuggingFaceSource(repo_id="aviatorlf/md17-shards", token=os.environ.get("HF_TOKEN"))
    limited_source = LimitedSource(source, MAX_CHUNKS)
    
    # Inject our custom PyG batching function (we partially apply batch_size)
    loader = AsyncFoldPipeLoader(
        source=limited_source, 
        batch_size=32, 
        batch_fn=lambda chunk: pyg_batch_fn(chunk, batch_size=32)
    )
    
    for batch_idx, mini_batch in enumerate(loader):
        train_batch(model, optimizer, criterion, mini_batch)

    profiler.stop()
    return profiler, False, time.time() - start_t

# ---------------------------------------------------------
# EXECUTION & PLOTTING
# ---------------------------------------------------------
if __name__ == "__main__":
    source = HuggingFaceSource(repo_id="aviatorlf/md17-shards", token=os.environ.get("HF_TOKEN"))
    all_files = list(itertools.islice(source.iter_files(), MAX_CHUNKS))
            
    print(f"\n=========================================")
    print(f"TESTING MODEL ARCHITECTURE: Real SchNet on MD17")
    print(f"=========================================")
    
    active_model = get_real_mlff_model()
    initial_state_dict = copy.deepcopy(active_model.state_dict())
    
    metrics = {
        "sequential": {"time": [], "peak_rss": [], "avg_gpu": [], "crashed": []},
        "foldpipe": {"time": [], "peak_rss": [], "avg_gpu": [], "crashed": []},
    }
    
    reference_traces = {}
    
    for run_idx in range(NUM_RUNS):
        print(f"  --- RUN {run_idx+1}/{NUM_RUNS} ---")
        
        seq_prof, seq_crashed, seq_time = run_sequential_stream(all_files, active_model, initial_state_dict)
        fp_prof, fp_crashed, fp_time = run_foldpipe_stream(active_model, initial_state_dict)
        
        metrics["sequential"]["time"].append(seq_time)
        metrics["sequential"]["peak_rss"].append(seq_prof.peak_rss / (1024**3))
        metrics["sequential"]["avg_gpu"].append(np.mean(seq_prof.gpu_history) if seq_prof.gpu_history else 0)
        metrics["sequential"]["crashed"].append(seq_crashed)
        
        metrics["foldpipe"]["time"].append(fp_time)
        metrics["foldpipe"]["peak_rss"].append(fp_prof.peak_rss / (1024**3))
        metrics["foldpipe"]["avg_gpu"].append(np.mean(fp_prof.gpu_history) if fp_prof.gpu_history else 0)
        metrics["foldpipe"]["crashed"].append(fp_crashed)
        
        if run_idx == 0:
            reference_traces = {
                "sequential": seq_prof,
                "foldpipe": fp_prof
            }
            
    # Aggregate statistics
    def aggregate(data):
        return {
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
            "raw": data
        }
        
    experiment_results = {
        "sequential": {
            "time": aggregate(metrics["sequential"]["time"]),
            "peak_rss_gb": aggregate(metrics["sequential"]["peak_rss"]),
            "avg_gpu_util": aggregate(metrics["sequential"]["avg_gpu"])
        },
        "foldpipe": {
            "time": aggregate(metrics["foldpipe"]["time"]),
            "peak_rss_gb": aggregate(metrics["foldpipe"]["peak_rss"]),
            "avg_gpu_util": aggregate(metrics["foldpipe"]["avg_gpu"])
        }
    }
    
    with open(f"results/benchmark_stats_md17.json", "w") as f:
        json.dump(experiment_results, f, indent=4)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    seq_prof = reference_traces["sequential"]
    fp_prof = reference_traces["foldpipe"]
    
    ax1.plot(seq_prof.time_history, seq_prof.ram_history, color='blue', alpha=0.7, label='Sequential Stream (O(1) RAM)')
    ax1.plot(fp_prof.time_history, fp_prof.ram_history, color='green', label='FoldPipe Async (O(1) RAM)')
    ax1.set_title(f"RAM Footprint (MD17 SchNet)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("RAM (GB)")
    ax1.legend()
    
    ax2.plot(seq_prof.time_history, seq_prof.gpu_history, color='blue', alpha=0.7, label='Sequential Stream (Idle GPU Traps)')
    ax2.plot(fp_prof.time_history, fp_prof.gpu_history, color='green', alpha=0.8, label='FoldPipe Async (Saturated)')
    ax2.set_title(f"GPU Saturation (MD17 SchNet)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Compute Utilization (%)")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/benchmark_comparison_md17.png', dpi=300)
    print(f"Saved results/benchmark_comparison_md17.png")
