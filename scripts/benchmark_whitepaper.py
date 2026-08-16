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
from foldpipe import AsyncFoldPipeLoader
from foldpipe.sources import HuggingFaceSource
from torch_geometric.nn.models import SchNet
import io

MAX_CHUNKS = 15  # Limit to avoid running all day
BATCH_SIZE = 128
NUM_RUNS = 5
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
# MODELS (SYMMETRIC WORKLOADS)
# ---------------------------------------------------------
class LimitedSource:
    """Wraps any source to artificially limit the number of shards yielded (useful for bounded benchmarks)."""
    def __init__(self, source, max_chunks):
        self.source = source
        self.max_chunks = max_chunks

    def iter_files(self):
        return itertools.islice(self.source.iter_files(), self.max_chunks)

    def download_chunk(self, identifier):
        return self.source.download_chunk(identifier)

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
    """
    def __init__(self):
        super().__init__()
        self.schnet = SchNet(hidden_channels=128, num_filters=128, num_interactions=6, num_gaussians=50, cutoff=10.0)
        
    def forward(self, x):
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
    
    if not torch.cuda.is_available():
        time.sleep(0.01)

# ---------------------------------------------------------
# PHASE 1: BASELINE A (EAGER ACCUMULATION) OOM TEST
# ---------------------------------------------------------
def run_baseline_in_memory(files, model, initial_state_dict):
    print("      --- BASELINE A: Eager In-Memory Accumulation ---")
    torch.manual_seed(42)
    model.load_state_dict(initial_state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    profiler = Profiler()
    profiler.start()
    
    dataset_in_memory = []
    crashed = False
    start_t = time.time()
    
    try:
        for i, f in enumerate(files[:MAX_CHUNKS]):
            request_url = f"https://huggingface.co/datasets/aviatorlf/prion-dataset/resolve/main/{f}"
            headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"} if os.environ.get("HF_TOKEN") else {}
            import requests
            response = requests.get(request_url, headers=headers, stream=True)
            response.raise_for_status()
            fh = io.BytesIO(response.content)
            fh.seek(0)
            tensor = torch.load(fh, map_location='cpu')
            
            if psutil.virtual_memory().percent > 90.0:
                raise MemoryError("System RAM exhausted.")
                
            dataset_in_memory.append(tensor)
            
            for b in range(0, tensor.size(0), BATCH_SIZE):
                train_batch(model, optimizer, criterion, tensor[b:b+BATCH_SIZE])
            
    except MemoryError as e:
        print(f"      [!] Memory Safety Threshold Reached: {e}")
        crashed = True
        
    profiler.stop()
    del dataset_in_memory
    return profiler, crashed, time.time() - start_t

# ---------------------------------------------------------
# PHASE 2: BASELINE B (SEQUENTIAL BOUNDED STREAMING)
# ---------------------------------------------------------
def run_sequential_stream(files, model, initial_state_dict):
    print("      --- BASELINE B: Sequential Bounded Streaming ---")
    torch.manual_seed(42)
    model.load_state_dict(initial_state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    profiler = Profiler()
    profiler.start()
    start_t = time.time()
    
    for i, f in enumerate(files[:MAX_CHUNKS]):
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
            
        del tensor
            
    profiler.stop()
    return profiler, False, time.time() - start_t

# ---------------------------------------------------------
# PHASE 3: FOLDPIPE (ASYNC STREAMING) TEST
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
    
    source = HuggingFaceSource(repo_id="aviatorlf/prion-dataset", token=os.environ.get("HF_TOKEN"))
    limited_source = LimitedSource(source, MAX_CHUNKS)
    loader = AsyncFoldPipeLoader(source=limited_source, batch_size=BATCH_SIZE)
    
    for batch_idx, mini_batch in enumerate(loader):
        train_batch(model, optimizer, criterion, mini_batch)

    profiler.stop()
    return profiler, False, time.time() - start_t

# ---------------------------------------------------------
# EXECUTION & PLOTTING
# ---------------------------------------------------------
if __name__ == "__main__":
    source = HuggingFaceSource(repo_id="aviatorlf/prion-dataset", token=os.environ.get("HF_TOKEN"))
    
    # Materialize file names ONLY for the baselines (which don't strictly use iter_files yet internally in their loop)
    # FoldPipe strictly uses iter_files internally.
    all_files = list(itertools.islice(source.iter_files(), MAX_CHUNKS))
            
    models_to_test = {
        "Synthetic Deep": get_deep_model(),
        "Real MLFF (SchNet)": get_real_mlff_model()
    }
    
    # Store overall stats
    experiment_results = {}
    
    for model_name, active_model in models_to_test.items():
        print(f"\n=========================================")
        print(f"TESTING MODEL ARCHITECTURE: {model_name}")
        print(f"=========================================")
        
        initial_state_dict = copy.deepcopy(active_model.state_dict())
        
        # To aggregate metrics across runs
        metrics = {
            "eager": {"time": [], "peak_rss": [], "avg_gpu": [], "crashed": []},
            "sequential": {"time": [], "peak_rss": [], "avg_gpu": [], "crashed": []},
            "foldpipe": {"time": [], "peak_rss": [], "avg_gpu": [], "crashed": []},
        }
        
        # Save a reference trace for plotting
        reference_traces = {}
        
        for run_idx in range(NUM_RUNS):
            print(f"  --- RUN {run_idx+1}/{NUM_RUNS} ---")
            
            eager_prof, eager_crashed, eager_time = run_baseline_in_memory(all_files, active_model, initial_state_dict)
            seq_prof, seq_crashed, seq_time = run_sequential_stream(all_files, active_model, initial_state_dict)
            fp_prof, fp_crashed, fp_time = run_foldpipe_stream(active_model, initial_state_dict)
            
            # Record metrics
            metrics["eager"]["time"].append(eager_time)
            metrics["eager"]["peak_rss"].append(eager_prof.peak_rss / (1024**3))
            metrics["eager"]["avg_gpu"].append(np.mean(eager_prof.gpu_history) if eager_prof.gpu_history else 0)
            metrics["eager"]["crashed"].append(eager_crashed)
            
            metrics["sequential"]["time"].append(seq_time)
            metrics["sequential"]["peak_rss"].append(seq_prof.peak_rss / (1024**3))
            metrics["sequential"]["avg_gpu"].append(np.mean(seq_prof.gpu_history) if seq_prof.gpu_history else 0)
            metrics["sequential"]["crashed"].append(seq_crashed)
            
            metrics["foldpipe"]["time"].append(fp_time)
            metrics["foldpipe"]["peak_rss"].append(fp_prof.peak_rss / (1024**3))
            metrics["foldpipe"]["avg_gpu"].append(np.mean(fp_prof.gpu_history) if fp_prof.gpu_history else 0)
            metrics["foldpipe"]["crashed"].append(fp_crashed)
            
            # Save the first run's trace for the visual graphs
            if run_idx == 0:
                reference_traces = {
                    "eager": eager_prof,
                    "eager_crashed": eager_crashed,
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
            
        experiment_results[model_name] = {
            "eager": {
                "time": aggregate(metrics["eager"]["time"]),
                "peak_rss_gb": aggregate(metrics["eager"]["peak_rss"]),
                "crashed": any(metrics["eager"]["crashed"])
            },
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
        
        # Save raw JSON for this model
        with open(f"results/benchmark_stats_{model_name.replace(' ', '_')}.json", "w") as f:
            json.dump(experiment_results[model_name], f, indent=4)
        
        # Plotting the Reference Trace (Run 0)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        eager_prof = reference_traces["eager"]
        seq_prof = reference_traces["sequential"]
        fp_prof = reference_traces["foldpipe"]
        
        # Left: RAM Memory Bound Isolation
        ax1.plot(eager_prof.time_history, eager_prof.ram_history, color='red', label='Eager Accumulation (O(N) RAM)')
        ax1.plot(seq_prof.time_history, seq_prof.ram_history, color='blue', alpha=0.7, label='Sequential Stream (O(1) RAM)')
        ax1.plot(fp_prof.time_history, fp_prof.ram_history, color='green', label='FoldPipe Async (O(1) RAM)')
        if reference_traces["eager_crashed"]:
            ax1.scatter([eager_prof.time_history[-1]], [eager_prof.ram_history[-1]], color='darkred', s=150, marker='X', label='Memory Safety Cutoff')
        ax1.set_title(f"RAM Footprint ({model_name})")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("RAM (GB)")
        ax1.legend()
        
        # Right: Throughput Latency Masking Isolation
        ax2.plot(seq_prof.time_history, seq_prof.gpu_history, color='blue', alpha=0.7, label='Sequential Stream (Idle GPU Traps)')
        ax2.plot(fp_prof.time_history, fp_prof.gpu_history, color='green', alpha=0.8, label='FoldPipe Async (Saturated)')
        ax2.set_title(f"GPU Saturation ({model_name})")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Compute Utilization (%)")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/benchmark_comparison_{model_name.replace(" ", "_")}.png', dpi=300)
        print(f"Saved results/benchmark_comparison_{model_name.replace(' ', '_')}.png")

    print("\nBenchmark completed. Raw aggregated stats saved in results/.")
