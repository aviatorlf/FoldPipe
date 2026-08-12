# FoldPipe: Eliminating GPU Starvation in Large-Scale Structural Biology

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/aviatorlf/FoldPipe/blob/main/notebooks/optimized_training.ipynb)

## The Problem
Standard Machine Learning Force Field (MLFF) pipelines often bottleneck at the CPU data-loader. When dealing with multi-terabyte graph representations of molecular datasets (like MD17 or AlphaFold trajectories), standard in-memory pipelines (like PyTorch Geometric's `InMemoryDataset`) cause severe GPU starvation. On free-tier hardware with limited RAM (such as a Kaggle P100 or T4), this inevitably leads to devastating Out-Of-Memory (OOM) crashes and abysmally slow training throughput.

## The Solution
**FoldPipe** introduces an asynchronous, bounded-memory streaming architecture that completely decouples cloud I/O latency from CUDA execution. 

By applying concurrent `ThreadPoolExecutor` pre-fetching, parallel I/O batching, and explicit Python garbage collection, FoldPipe allows researchers to train state-of-the-art graph neural networks on massive datasets using cheap, low-memory preemptible instances.

## Empirical Whitepaper Benchmark
To prove the architecture's efficiency at eliminating network I/O bounds, we conducted a rigorous A/B benchmark on a Kaggle Hardware instance with a multi-terabyte trajectory dataset.

![Benchmark Results](assets/benchmark_comparison.png)

### The Results
1. **Baseline Failure (PyTorch Geometric):** The standard in-memory dataloader suffered a catastrophic memory leak. It breached the 7.4 GB process limit and crashed the OS (Exit Code 137). GPU utilization was 0% as the pipeline hung on network I/O.
2. **FoldPipe Success:** By pipelining background network fetches with foreground 20-epoch mini-batch GPU processing, FoldPipe strictly bounded RAM utilization below 1.8 GB and achieved **near 100% continuous GPU saturation**, successfully masking all network latency.

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
You do not need a supercomputer to run this pipeline. We have provided a fully optimized Kaggle template. Simply click the **"Open in Kaggle"** button at the top of this README to instantly spin up a GPU training environment for TorchMD-Net.
