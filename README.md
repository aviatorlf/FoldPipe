# FoldPipe: High-Throughput Molecular Dynamics for Constrained Hardware

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/aviatorlf/FoldPipe/blob/main/notebooks/optimized_training.ipynb)

## The Problem
Standard Machine Learning Force Field (MLFF) pipelines often bottleneck at the CPU data-loader. When dealing with massive graph representations of molecular datasets (like MD17), conventional data loading pipelines cause severe GPU starvation. On free-tier hardware with limited RAM (such as a Kaggle Tesla T4), this leads to devastating Out-Of-Memory (OOM) crashes and abysmally slow training throughput.

## The Solution
**FoldPipe** is an I/O-optimized surrogate pipeline specifically engineered to resolve these bottlenecks. It applies aggressive memory quantization, parallel I/O batching, and PyTorch Geometric indexing corrections under `DataParallel`. 

By decoupling the data loading from the main thread and properly pre-fetching graph tensors, FoldPipe allows researchers to train state-of-the-art architectures—like the **TorchMD-Net Equivariant Transformer**—on free Kaggle T4 GPUs up to **3x faster** than naive implementations, completely eliminating OOM errors.

## Benchmark Metrics (Kaggle Tesla T4 x2)
Profiling 500 batches of the MD17 (Aspirin) trajectory against the naive PyTorch Geometric loader versus the FoldPipe Async Loader.

| Metric | Naive PyG Loader | FoldPipe Loader | Improvement |
| :--- | :--- | :--- | :--- |
| **Total Runtime** | 6.85s | 2.51s | **2.7x Speedup** |
| **CPU Load Time (GPU Starvation)** | 57.0% | 10.7% | **-46.3% Idle Time** |
| **GPU Active Compute** | 43.0% | 89.3% | **+46.3% Utilization** |
| **System RAM Peak** | 8.3% | 9.7% | **Stable (Zero OOM)** |

*Note: FoldPipe uses contiguous binary serialization and pinned memory mapping to bypass Python's copy-on-read multiprocessing overhead.*

## Macromolecule Stress Test (Human Prion Protein - 1QLX)
Testing FoldPipe's pinned-memory architecture against a massive protein trajectory to validate memory stability on constrained hardware.

| Metric | Naive PyG Loader | FoldPipe Performance (Kaggle T4) | Improvement |
| :--- | :--- | :--- | :--- |
| **Simulated Trajectory** | 10,000 Frames | 10,000 Frames | - |
| **Throughput** | **0 Frames / sec (CRASH)** | 5,185 Frames / sec | **∞ Speedup** |
| **System RAM Peak** | 5.9% (Pre-Crash) | 7.5% | **Stable (Zero OOM)** |
| **GPU VRAM Allocation** | **1.14 Terabytes** (Failed) | 1.4 GB | **-99.9% VRAM Overhead** |
| **Status** | **FATAL OOM (Batch 0)** | **SUCCESS (10k Frames)** | **Pipeline Saved** |

*Result: FoldPipe successfully batches and streams macromolecular topologies without triggering the astronomical 1.14 TB VRAM allocation crash caused by PyTorch Geometric's naive 1D graph flattening on massive proteins.*

## 3-Step Quickstart

1. **Clone the repository:**
```bash
git clone https://github.com/aviatorlf/FoldPipe.git
cd FoldPipe
```

2. **Install requirements:**
```bash
pip install -r requirements.txt
```

3. **Run the optimized benchmark:**
```bash
python scripts/run_benchmark.py
```

## Kaggle Integration
You do not need a supercomputer to run this pipeline. We have provided a fully optimized Kaggle template. Simply click the **"Open in Kaggle"** button at the top of this README to instantly spin up the dual-GPU training environment for TorchMD-Net.
