# FoldPipe: PyTorch Molecular Dynamics Pipeline

FoldPipe is an optimized pipeline for running AI-based Molecular Dynamics (MD) simulations using `TorchMD-Net`. It focuses on eliminating I/O bottlenecks and maximizing GPU throughput on constrained hardware environments like Kaggle's dual Nvidia T4 GPUs.

## Overview
AI physics and molecular dynamics (MD) simulations often suffer from massive data-loading bottlenecks. The complex 3D atom coordinate graphs can starve the GPU if not pre-fetched correctly. 

FoldPipe implements:
- **Hardware-Aware Data Loading:** Uses PyTorch Geometric (`torch_geometric`) with multi-worker prefetching and pinned memory.
- **Mixed Precision (AMP):** Utilizes Automatic Mixed Precision (FP16/FP32) to reduce VRAM footprint and increase batch sizes by 4x.
- **Memory Fragmentation Management:** Aggressively clears GPU caches to prevent Out of Memory (OOM) errors on 16GB T4 GPUs.

## Profiling Results (Nvidia T4 x2)
On the MD17 dataset (Aspirin):
- **Naive Pipeline:** ~6,800 samples/sec (No training loop, severely I/O bound).
- **FoldPipe Optimized:** ~8,000 samples/sec (Executing dummy forward/backward passes and AMP scaling!).

## Directory Structure
- `src/`
  - `dataset.py`: The optimized PyTorch Geometric dataloader for the MD17 dataset.
  - `train.py`: The core training loop demonstrating AMP and memory clearing.
- `notebooks/`: Kaggle-ready Jupyter Notebooks for immediate execution.
  - `baseline_training.ipynb`: The naive pipeline for establishing baseline metrics.
  - `optimized_training.ipynb`: The FoldPipe optimized pipeline.
- `examples/`
  - `local_training.py`: A quickstart example for running the pipeline locally.

## Getting Started

### Local Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the example training script:
```bash
python -m examples.local_training
```

### Kaggle Execution
The `notebooks/` directory contains pre-configured Kaggle notebooks. You can push them directly using the Kaggle API:
```bash
# Push baseline
kaggle kernels push -p notebooks/ --accelerator NvidiaTeslaT4

# Note: Update kernel-metadata.json's `code_file` to point to the desired notebook before pushing!
```
