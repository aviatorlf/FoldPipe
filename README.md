# FoldPipe: Eliminating GPU Starvation in Large-Scale Structural Biology

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/aviatorlf/FoldPipe/blob/main/notebooks/optimized_training.ipynb)

## The Problem

Standard Machine Learning Force Field (MLFF) pipelines often bottleneck at the CPU data loader. When large molecular trajectories do not fit in memory, eager or in-memory representations can exhaust RAM and leave accelerators waiting on data, especially on constrained instances such as Kaggle P100 or T4 runners.

## The Solution

**FoldPipe** is a domain-specific integration of bounded-buffer asynchronous prefetching tailored for native `.pt` molecular trajectories. It bridges the gap between PyG's in-memory/on-disk abstractions and format-converting streaming systems such as WebDataset, particularly for memory-constrained ephemeral compute.

FoldPipe provides an out-of-core orchestration layer for datasets that exceed local memory constraints. A single background prefetch worker overlaps retrieval of shard *N+1* with training on shard *N*, while a bounded shard lifecycle keeps the working set independent of total dataset size.

## Honest Positioning & Prior Art

To be 100% factual, transparent, and defensible in peer review, here is how FoldPipe positions itself against the prior art:

*   **"We do not replace LMDB for local high-performance computing clusters."** Existing out-of-core streaming systems like WebDataset and LMDB provide petascale throughput. However, they require researchers to convert their native PyTorch/PyG outputs into POSIX `.tar` archives or binary databases. FoldPipe sacrifices absolute peak cluster throughput to eliminate this conversion step, allowing researchers to train directly on native `.pt` shards.
*   **"We extend the PyTorch Geometric ecosystem."** PyG's `InMemoryDataset` provides excellent throughput for datasets that fit in RAM. FoldPipe provides an outer orchestration layer for datasets that exceed local RAM limits, with bounded $O(1)$ working-set memory with respect to total dataset size.

The utilization measurements belong to different experiments and should not be conflated:

| Experiment | Workload | FoldPipe mean GPU utilization | Interpretation |
| :--- | :--- | ---: | :--- |
| Controlled crossover | Deliberately deep synthetic compute | >90% in the compute-dominant regime | Demonstrates that prefetch can mask I/O when compute time is long enough. |
| Real MD17 benchmark | SchNet energy-and-force training, ten paired order-alternating runs | 37.65% | Real-workload mean across FoldPipe passes; high run-to-run variance remains. |

## Empirical Whitepaper Benchmark

To evaluate when asynchronous prefetch masks network I/O, we benchmarked FoldPipe against shards drawn from a large trajectory corpus on Kaggle hardware.

![Benchmark Results](assets/benchmark_comparison.png)

### The Results

1. **RAM Scaling ($O(N)$ vs $O(1)$):** The eager-accumulation baseline demonstrated linear $O(N)$ payload-memory growth, hoarding every downloaded tensor in RAM. FoldPipe successfully demonstrated strict $O(1)$ memory bounding by yielding and aggressively releasing each chunk after GPU computation.
2. **Controlled Hardware Crossover:** FoldPipe can fully mask shard-fetch latency when current-shard processing time is at least as long as retrieval of the next shard. In a deliberately deep synthetic-compute benchmark, sampled GPU utilization exceeded 90% in this compute-dominant regime. This result characterizes the mechanism; it is not the MD17 SchNet utilization result.

### Kaggle T4 MD17 Benchmark

The final private Kaggle kernel completed ten paired, order-alternating measurements of a real SchNet energy-and-force workload streaming revision-pinned MD17 shards from Hugging Face. Each timed measurement is a **5-shard benchmark pass**, not a full-dataset epoch. With 5,000 structures per shard, each pass processes 25,000 structures.

| Metric | Sequential | FoldPipe |
| :--- | ---: | ---: |
| Mean pass time | 155.52 s | 107.59 s |
| 95% bootstrap CI for mean time | 113.64–203.16 s | 65.44–156.03 s |
| Median pass time | 138.71 s | 89.47 s |
| Time standard deviation | 76.99 s | 77.50 s |
| Mean sampled GPU utilization | 20.43% | 37.65% |
| Mean peak RSS | 1.938 GiB | 2.037 GiB |
| Mean I/O/compute overlap | 0.00 s | 15.26 s |
| Mean GPU wait time | 128.61 s | 79.60 s |

FoldPipe was faster in 7 of 10 pairs. The arithmetic mean of paired speedup ratios was 2.312× with a 95% percentile-bootstrap interval of 1.284×–3.465×. The additive paired effect was 47.93 seconds saved on average, but its interval was −12.22 to 103.44 seconds and therefore included zero. The ratio and additive estimands disagree under substantial run-to-run network variability; the defensible conclusion is that FoldPipe improved the observed means and demonstrated real I/O/compute overlap, not that this run establishes a universal speedup.

The benchmark pinned `aviatorlf/md17-shards` to revision `f779686deb9217877dd7ddde99b2522bd441492a` and embedded a source bundle with SHA-256 `0ed7e4c583621edc793234e51dafc3756221f7e2337e1d86708c1d5f25fcad39`. The raw JSON preserves all paired durations plus per-shard download, deserialization, training, payload-byte, overlap, and wait-time traces.

![MD17 SchNet benchmark](results/benchmark_comparison_md17.png)

Artifacts: [`benchmark report`](results/benchmark_report_md17.md), [`raw statistics and traces`](results/benchmark_stats_md17.json), [`source manifest`](results/benchmark_source_manifest_md17.json), [`Kaggle execution log`](results/foldpipe-md17-rigorous-benchmark-v6.log), and the [Kaggle version 6 kernel](https://www.kaggle.com/code/dhirenkhatri/foldpipe-md17-rigorous-benchmark).

## Quickstart & Usage

### 1. Installation

Install directly from PyPI (Recommended):
```bash
pip install foldpipe
```

*(Alternative)* Install from source:
```bash
git clone https://github.com/aviatorlf/FoldPipe.git
cd FoldPipe
pip install -e .
```

### 2. Standard PyTorch Training Loop

FoldPipe operates as a drop-in iterator. It handles the background threading and bounded-memory shard lifecycle automatically.

```python
from foldpipe import AsyncFoldPipeLoader
from foldpipe.sources import GoogleDriveSource, HuggingFaceSource

# Option A: Stream from Google Drive
source = GoogleDriveSource(
    folder_id="1Few5wzRuuhlwbj4DJD9nkOP98t_QqZcz",
    credentials_json="path/to/token.json"  # Handled automatically
)

# Option B: Stream from HuggingFace (Zero disk caching)
# source = HuggingFaceSource(
#     repo_id="username/prion-dataset",
#     folder_path="trajectories"
# )

loader = AsyncFoldPipeLoader(source=source, batch_size=128)

model = MyEquivariantNetwork().to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for batch in loader:
    optimizer.zero_grad()
    
    # Hardware Masking: chunk N+1 is fetched while GPU computes chunk N
    out = model(batch.to('cuda', non_blocking=True))
    loss = criterion(out)
    
    loss.backward()
    optimizer.step()
```

## Kaggle Integration

You do not need a supercomputer to run this pipeline. We have provided a fully optimized Kaggle template. Simply click the **"Open in Kaggle"** button at the top of this README to instantly spin up a GPU training environment for TorchMD-Net.
