# FoldPipe: Bounded Remote Streaming for Native Molecular-Data Shards

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/aviatorlf/FoldPipe/blob/main/notebooks/Quickstart.ipynb)

FoldPipe is a small loader for streaming trusted, already-sharded PyTorch/PyG
molecular data without first converting it into a different storage format.

## Start Here

Most users only need the library and one tutorial:

```bash
pip install "foldpipe @ git+https://github.com/aviatorlf/FoldPipe.git@v0.3.0"
```

- Open [`Quickstart.ipynb`](notebooks/Quickstart.ipynb) for a five-minute,
  credential-free introduction.
- Open [`Prion_Case_Study.ipynb`](notebooks/Prion_Case_Study.ipynb) for an
  educational end-to-end example using the bundled 1QLX structure.

## Repository Structure

| Path | Audience | Purpose |
| :--- | :--- | :--- |
| [`foldpipe/`](foldpipe/) | Users | The installed core library and public API. Start here if you want to inspect the implementation. |
| [`notebooks/`](notebooks/) | Users | Two maintained tutorials. Historical prototypes are clearly separated under `notebooks/archive/`. |
| [`scripts/`](scripts/) and [`results/`](results/) | Peer reviewers | Reproducibility code, raw traces, reports, and benchmark packaging. Normal users can safely ignore these. |
| [`tests/`](tests/) | Contributors | Automated checks that protect the core loader and benchmark logic. |
| [`data/`](data/) | Tutorial users | Small bundled inputs and data-location notes; large trajectory shards remain remote. |
| [`paper/`](paper/) | Readers and reviewers | Manuscript draft, references, and the plagiarism/novelty audit. |

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
| Real MD17 benchmark | SchNet energy-and-force training, 20 paired order-alternating runs | 37.70% | Real-workload mean across FoldPipe passes; high run-to-run variance remains. |

## Empirical Whitepaper Benchmark

To evaluate when asynchronous prefetch masks network I/O, we benchmarked FoldPipe against shards drawn from a large trajectory corpus on Kaggle hardware.

![Benchmark Results](assets/benchmark_comparison.png)

### The Results

1. **RAM Scaling ($O(N)$ vs $O(1)$):** The eager-accumulation baseline demonstrated linear $O(N)$ payload-memory growth, hoarding every downloaded tensor in RAM. FoldPipe successfully demonstrated strict $O(1)$ memory bounding by yielding and aggressively releasing each chunk after GPU computation.
2. **Controlled Hardware Crossover:** FoldPipe can fully mask shard-fetch latency when current-shard processing time is at least as long as retrieval of the next shard. In a deliberately deep synthetic-compute benchmark, sampled GPU utilization exceeded 90% in this compute-dominant regime. This result characterizes the mechanism; it is not the MD17 SchNet utilization result.

### Kaggle T4 MD17 Benchmark

The final private Kaggle kernel completed 20 paired, order-alternating measurements of a real SchNet energy-and-force workload streaming revision-pinned MD17 shards from Hugging Face. Each timed measurement is a **5-shard benchmark pass**, not a full-dataset epoch. With 5,000 structures per shard, each pass processes 25,000 structures. Across both pipelines, the protocol recorded 40 passes, 200 shard traces, and 1,000,000 repeated structure-visits.

| Metric | Sequential | FoldPipe |
| :--- | ---: | ---: |
| Mean pass time | 83.37 s | 76.78 s |
| 95% bootstrap CI for mean time | 67.87–101.06 s | 64.86–89.44 s |
| Median pass time | 75.27 s | 75.97 s |
| Time standard deviation | 39.86 s | 28.46 s |
| Mean sampled GPU utilization | 36.17% | 37.70% |
| Mean peak RSS | 1.935 GiB | 2.064 GiB |
| Mean I/O/compute overlap | 0.00 s | 16.33 s |
| Mean GPU wait time | 56.50 s | 48.65 s |

FoldPipe was faster in 11 of 20 pairs. The primary multiplicative estimand, the geometric mean paired speedup, was 1.059× with a 95% paired bootstrap interval of 0.878×–1.288×. Mean paired time saved was 6.59 seconds (95% CI −7.69 to 21.11 seconds), and median paired time saved was 3.45 seconds (95% CI −13.48 to 22.89 seconds). All three intervals include their no-effect values, so this run is **inconclusive about a speed advantage**. The positive 16.33-second mean overlap directly confirms that the asynchronous mechanism overlaps I/O and compute, but that mechanism measurement must not be promoted into a universal throughput claim under the observed network variability.

The benchmark pinned `aviatorlf/md17-shards` to revision `f779686deb9217877dd7ddde99b2522bd441492a`, executed clean Git commit `16fdbb26b00f9721ce4034335ce0ee12bda77720`, and embedded a source bundle with SHA-256 `4fc3f6d712557822efc0d5d71aaa616b0fe30f578631726ef60d6a0fe73b32c0`. The raw JSON preserves all paired durations plus per-shard download, deserialization, training, payload-byte, overlap, and wait-time traces.

![MD17 SchNet benchmark](results/benchmark_comparison_md17.png)

Artifacts: [`benchmark report`](results/benchmark_report_md17.md), [`raw statistics and traces`](results/benchmark_stats_md17.json), [`source manifest`](results/benchmark_source_manifest_md17.json), [`Kaggle execution log`](results/foldpipe-md17-rigorous-benchmark-v8.log), and the [Kaggle version 8 kernel](https://www.kaggle.com/code/dhirenkhatri/foldpipe-md17-rigorous-benchmark).

## Quickstart & Usage

### 1. Installation

Install the tagged GitHub release:
```bash
pip install "foldpipe @ git+https://github.com/aviatorlf/FoldPipe.git@v0.3.0"
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

The **Open in Kaggle** badge at the top opens the maintained Quickstart notebook.
Its first example runs without credentials or a GPU; connect a remote source only
after supplying your own revision-pinned dataset and secret through the platform's
secret manager.
