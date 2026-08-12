import time
import torch
import psutil
from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader

# 1. Hardware Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Benchmarking on: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 2. Acquire the MD17 Dataset (Aspirin trajectory is the standard benchmark)
print("\nDownloading MD17 (Aspirin)...")
dataset = MD17(root='./data', name='aspirin')
print(f"Dataset loaded: {len(dataset)} molecular conformations.")

# 3. The Baseline PyG DataLoader (This is what we are going to optimize)
batch_size = 32
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 4. Dummy Model Training Loop for Profiling
print(f"\nStarting Baseline I/O Profiling (Batch Size: {batch_size})...")

num_batches_to_profile = 200
data_load_times = []
compute_times = []

# Warm-up GPU
if torch.cuda.is_available():
    torch.cuda.synchronize()
start_profiling = time.time()

epoch_start = time.time()
for idx, batch in enumerate(loader):
    if idx >= num_batches_to_profile:
        break
        
    # --- Profile Data Loading ---
    load_end = time.time()
    data_load_times.append(load_end - epoch_start)
    
    # Push to GPU
    batch = batch.to(device)
    
    # --- Dummy GPU Compute (Simulating a TorchMD-Net forward/backward pass) ---
    # We do a heavy matrix multiplication to simulate Equivariant Transformer load
    dummy_tensor = torch.randn((batch_size, 1024, 1024), device=device)
    dummy_out = torch.bmm(dummy_tensor, dummy_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # ----------------------------------------------------------------------
    
    compute_end = time.time()
    compute_times.append(compute_end - load_end)
    
    # Reset timer for next load
    epoch_start = time.time()

# 5. Extract Metrics
total_load_time = sum(data_load_times)
total_compute_time = sum(compute_times)
total_time = total_load_time + total_compute_time

# 6. The Verdict
print("\n" + "="*50)
print("BASELINE I/O PROFILING RESULTS")
print("="*50)
print(f"Total Batches Profiled : {num_batches_to_profile}")
print(f"Total Time             : {total_time:.2f} seconds")
print(f"Data Loading Time (CPU): {total_load_time:.2f} seconds ({ (total_load_time/total_time)*100:.1f}% of total time)")
print(f"GPU Compute Time       : {total_compute_time:.2f} seconds ({ (total_compute_time/total_time)*100:.1f}% of total time)")
print(f"System RAM Used        : {psutil.virtual_memory().percent}%")
if torch.cuda.is_available():
    print(f"GPU VRAM Peak          : {torch.cuda.max_memory_allocated() / (1024 ** 2):.1f} MB")
print("="*50)

print("\nCONCLUSION: If Data Loading Time is > 20%, your GPU is starving.")
