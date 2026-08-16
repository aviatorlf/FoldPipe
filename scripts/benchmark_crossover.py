import os
import time
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import itertools
from foldpipe import AsyncFoldPipeLoader
from foldpipe.sources import SyntheticLatencySource

os.makedirs('results', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_CHUNKS = 10
BATCH_SIZE = 128
CONSUMER_DELAY_MS = 100  # Fixed synthetic controlled consumer processing delay

def get_tiny_model():
    return nn.Sequential(nn.Linear(3, 16), nn.Linear(16, 3)).to(device)

def train_batch(model, optimizer, criterion, mini_batch):
    mini_batch = mini_batch.to(device, non_blocking=True)
    optimizer.zero_grad()
    out = model(mini_batch)
    loss = criterion(out, torch.zeros_like(out))
    loss.backward()
    optimizer.step()
    
    # Force fixed consumer delay to isolate the crossover mechanism precisely
    time.sleep(CONSUMER_DELAY_MS / 1000.0)

def run_sequential_sweep(source, model, initial_state_dict):
    torch.manual_seed(42)
    model.load_state_dict(initial_state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    start_t = time.time()
    for f in itertools.islice(source.iter_files(), MAX_CHUNKS):
        chunk = source.download_chunk(f)
        for b in range(0, chunk.size(0), BATCH_SIZE):
            train_batch(model, optimizer, criterion, chunk[b:b+BATCH_SIZE])
        del chunk
    return time.time() - start_t

def run_async_sweep(source, model, initial_state_dict):
    torch.manual_seed(42)
    model.load_state_dict(initial_state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    start_t = time.time()
    loader = AsyncFoldPipeLoader(source=source, batch_size=BATCH_SIZE)
    
    for mini_batch in loader:
        train_batch(model, optimizer, criterion, mini_batch)
    return time.time() - start_t

if __name__ == "__main__":
    latencies_ms = [10, 50, 100, 150, 200, 300, 400]
    
    active_model = get_tiny_model()
    import copy
    initial_state_dict = copy.deepcopy(active_model.state_dict())
    
    results = {
        "D_C_ratios": [],
        "measured_speedup": [],
        "theoretical_speedup": []
    }
    
    print("=========================================")
    print("RUNNING ISOLATED MECHANISM SWEEP")
    print(f"Fixed Consumer Delay: {CONSUMER_DELAY_MS}ms")
    print("=========================================")
    
    for lat in latencies_ms:
        print(f"Sweeping Network Latency: {lat}ms")
        source = SyntheticLatencySource(num_chunks=MAX_CHUNKS, latency_ms=lat, chunk_size=BATCH_SIZE*4)
        
        t_seq = run_sequential_sweep(source, active_model, initial_state_dict)
        t_async = run_async_sweep(source, active_model, initial_state_dict)
        
        speedup = t_seq / t_async
        
        # Calculate theoretical values per batch
        C = CONSUMER_DELAY_MS / 1000.0 * 4 # 4 batches per chunk
        D = lat / 1000.0
        ratio = D / C
        theoretical = (D + C) / max(D, C)
        
        results["D_C_ratios"].append(ratio)
        results["measured_speedup"].append(speedup)
        results["theoretical_speedup"].append(theoretical)
        print(f"  Ratio (D/C): {ratio:.2f} | Measured Speedup: {speedup:.2f}x | Theoretical: {theoretical:.2f}x")

    with open("results/crossover_stats.json", "w") as f:
        json.dump(results, f, indent=4)
        
    plt.figure(figsize=(10, 6))
    plt.plot(results["D_C_ratios"], results["theoretical_speedup"], label="Theoretical Ideal", linestyle="--", color="gray", linewidth=2)
    plt.plot(results["D_C_ratios"], results["measured_speedup"], label="Measured FoldPipe", marker="o", color="green", linewidth=2)
    
    plt.axvline(x=1.0, color='red', linestyle=':', label="Consumer = I/O (Optimal Overlap)")
    
    plt.title("Mechanism Crossover: Measured vs Theoretical Speedup")
    plt.xlabel("D/C Ratio (Download Time / Consumer Delay)")
    plt.ylabel("Speedup (T_seq / T_async)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/crossover_mechanism.png', dpi=300)
    print("Saved results/crossover_mechanism.png")
