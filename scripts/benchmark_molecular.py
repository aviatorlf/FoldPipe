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
import datetime
import platform
import statistics
import matplotlib.pyplot as plt
import numpy as np

# PyG imports
from torch_geometric.data import Batch
from torch_geometric.nn.models import SchNet

from foldpipe import AsyncFoldPipeLoader
from foldpipe.sources import HuggingFaceSource, PreenumeratedSource

MAX_CHUNKS = int(os.environ.get("FOLDPIPE_MAX_CHUNKS", "5"))
NUM_RUNS = int(os.environ.get("FOLDPIPE_NUM_RUNS", "10"))
BOOTSTRAP_SAMPLES = int(os.environ.get("FOLDPIPE_BOOTSTRAP_SAMPLES", "20000"))
HF_REPO_ID = os.environ.get("FOLDPIPE_HF_REPO_ID", "aviatorlf/md17-shards")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('results', exist_ok=True)

if MAX_CHUNKS < 1:
    raise ValueError("FOLDPIPE_MAX_CHUNKS must be at least 1")
if NUM_RUNS < 2:
    raise ValueError("FOLDPIPE_NUM_RUNS must be at least 2")
if BOOTSTRAP_SAMPLES < 1000:
    raise ValueError("FOLDPIPE_BOOTSTRAP_SAMPLES must be at least 1000")

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
            self.time_history.append(time.perf_counter() - self.start_time)
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
        self.start_time = time.perf_counter()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        self.thread.join()


class RunTrace:
    """Thread-safe per-shard transfer and training timeline for one pipeline run."""

    def __init__(self, pipeline, run_index, identifiers):
        self.pipeline = pipeline
        self.run_index = run_index
        self.origin = time.perf_counter()
        self.finish = None
        self._lock = threading.Lock()
        self.records = [
            {
                "shard_index": index,
                "identifier": identifier,
                "download_start_s": None,
                "download_finish_s": None,
                "deserialize_finish_s": None,
                "training_start_s": None,
                "training_finish_s": None,
                "bytes_downloaded": 0,
                "structures": None,
            }
            for index, identifier in enumerate(identifiers)
        ]
        self._indices_by_identifier = {
            identifier: index for index, identifier in enumerate(identifiers)
        }

    def on_transfer(self, event):
        index = self._indices_by_identifier[event["identifier"]]
        with self._lock:
            record = self.records[index]
            record["download_start_s"] = event["download_start"] - self.origin
            if event["download_finish"] is not None:
                record["download_finish_s"] = event["download_finish"] - self.origin
            if event["deserialize_finish"] is not None:
                record["deserialize_finish_s"] = event["deserialize_finish"] - self.origin
            record["bytes_downloaded"] = event["bytes_downloaded"]
            if "error" in event:
                record["error"] = event["error"]

    def start(self):
        self.origin = time.perf_counter()
        self.finish = None

    def training_started(self, shard_index, structures=None):
        with self._lock:
            record = self.records[shard_index]
            if record["training_start_s"] is None:
                record["training_start_s"] = time.perf_counter() - self.origin
            if structures is not None:
                record["structures"] = structures

    def training_finished(self, shard_index):
        with self._lock:
            self.records[shard_index]["training_finish_s"] = time.perf_counter() - self.origin

    def stop(self):
        self.finish = time.perf_counter()

    @property
    def wall_time(self):
        finish = self.finish if self.finish is not None else time.perf_counter()
        return finish - self.origin

    @staticmethod
    def _interval_overlap(left, right):
        return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))

    def summary(self):
        download_intervals = []
        training_intervals = []
        gpu_wait_s = 0.0
        previous_training_finish = 0.0

        for record in self.records:
            download_start = record["download_start_s"]
            download_finish = record["download_finish_s"]
            training_start = record["training_start_s"]
            training_finish = record["training_finish_s"]
            if download_start is not None and download_finish is not None:
                download_intervals.append((download_start, download_finish))
            if training_start is not None and training_finish is not None:
                training_intervals.append((training_start, training_finish))
                gpu_wait_s += max(0.0, training_start - previous_training_finish)
                previous_training_finish = training_finish

        overlap_s = sum(
            self._interval_overlap(download, training)
            for download in download_intervals
            for training in training_intervals
        )
        io_s = sum(end - start for start, end in download_intervals)
        compute_s = sum(end - start for start, end in training_intervals)
        deserialize_s = sum(
            max(0.0, record["deserialize_finish_s"] - record["download_finish_s"])
            for record in self.records
            if record["deserialize_finish_s"] is not None
            and record["download_finish_s"] is not None
        )
        return {
            "wall_time_s": self.wall_time,
            "io_time_s": io_s,
            "deserialize_time_s": deserialize_s,
            "compute_time_s": compute_s,
            "overlap_time_s": overlap_s,
            "gpu_wait_time_s": gpu_wait_s,
            "bytes_downloaded": sum(record["bytes_downloaded"] for record in self.records),
            "structures": sum(record["structures"] or 0 for record in self.records),
        }

    def as_dict(self):
        return {
            "pipeline": self.pipeline,
            "run_index": self.run_index,
            "summary": self.summary(),
            "shards": self.records,
        }

# ---------------------------------------------------------
# BATCHING ABSTRACTION & MODELS
# ---------------------------------------------------------
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


def warm_up_model(model, initial_state_dict, chunk):
    """Run one untimed batch so one-off CUDA initialization is not assigned to a pipeline."""
    print("Running one untimed SchNet warm-up batch...")
    torch.manual_seed(42)
    model.load_state_dict(initial_state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    first_batch = next(pyg_batch_fn(chunk, batch_size=32))
    train_batch(model, optimizer, criterion, first_batch)
    synchronize_device()
    model.load_state_dict(initial_state_dict)

# ---------------------------------------------------------
# PHASE 1: SEQUENTIAL BOUNDED STREAMING
# ---------------------------------------------------------
def synchronize_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_sequential_stream(source, model, initial_state_dict, trace):
    print("      --- BASELINE: Sequential Bounded Streaming ---")
    torch.manual_seed(42)
    model.load_state_dict(initial_state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    synchronize_device()
    trace.start()
    profiler = Profiler()
    profiler.start()

    for i, f in enumerate(source.iter_files()):
        chunk_list = source.download_chunk(f)
        trace.training_started(i, structures=len(chunk_list))
        for mini_batch in pyg_batch_fn(chunk_list, batch_size=32):
            train_batch(model, optimizer, criterion, mini_batch)
        synchronize_device()
        trace.training_finished(i)
        del chunk_list

    synchronize_device()
    trace.stop()
    profiler.stop()
    return profiler, trace.wall_time

# ---------------------------------------------------------
# PHASE 2: FOLDPIPE (ASYNC STREAMING) TEST
# ---------------------------------------------------------
def run_foldpipe_stream(source, model, initial_state_dict, trace):
    print(f"      --- FOLDPIPE ASYNC STREAM ---")
    torch.manual_seed(42)
    model.load_state_dict(initial_state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    shard_counter = itertools.count()

    def traced_batch_fn(chunk):
        shard_index = next(shard_counter)
        structures = len(chunk)
        total_batches = (structures + 31) // 32
        for batch_index, mini_batch in enumerate(pyg_batch_fn(chunk, batch_size=32)):
            yield shard_index, structures, batch_index == total_batches - 1, mini_batch

    synchronize_device()
    trace.start()
    profiler = Profiler()
    profiler.start()

    # Inject our custom PyG batching function (we partially apply batch_size)
    loader = AsyncFoldPipeLoader(
        source=source,
        batch_size=32,
        batch_fn=traced_batch_fn,
    )

    for shard_index, structures, is_last_batch, mini_batch in loader:
        trace.training_started(shard_index, structures=structures)
        train_batch(model, optimizer, criterion, mini_batch)
        if is_last_batch:
            synchronize_device()
            trace.training_finished(shard_index)

    synchronize_device()
    trace.stop()
    profiler.stop()
    return profiler, trace.wall_time


def bootstrap_ci(data, samples=BOOTSTRAP_SAMPLES, seed=20260817):
    """Deterministic nonparametric percentile-bootstrap confidence interval."""
    values = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)
    resampled = rng.choice(values, size=(samples, len(values)), replace=True)
    estimates = np.mean(resampled, axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {
        "low": float(low),
        "high": float(high),
        "method": "percentile bootstrap",
        "resamples": samples,
    }


def aggregate(data, seed=20260817):
    values = [float(value) for value in data]
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "sample_std": float(statistics.stdev(values)),
        "ci_95": bootstrap_ci(values, seed=seed),
        "raw": values,
    }


def pipeline_order(run_index):
    """Balance time-varying network conditions across the paired comparison."""
    return (
        ["sequential", "foldpipe"]
        if run_index % 2 == 0
        else ["foldpipe", "sequential"]
    )


def git_metadata():
    def git_output(*args):
        try:
            return subprocess.check_output(
                ["git", *args], encoding="utf-8", stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = git_output("rev-parse", "HEAD")
    dirty = bool(git_output("status", "--porcelain")) if commit else None
    metadata = {"commit": commit, "dirty": dirty}
    manifest_path = os.environ.get("FOLDPIPE_SOURCE_MANIFEST")
    if manifest_path:
        with open(manifest_path, encoding="utf-8") as manifest_file:
            metadata["source_bundle"] = json.load(manifest_file)
    return metadata

# ---------------------------------------------------------
# EXECUTION & PLOTTING
# ---------------------------------------------------------
if __name__ == "__main__":
    hf_source = HuggingFaceSource(repo_id=HF_REPO_ID, token=os.environ.get("HF_TOKEN"))
    dataset_revision = hf_source.api.dataset_info(HF_REPO_ID).sha
    hf_source.revision = dataset_revision
    all_files = list(itertools.islice(hf_source.iter_files(), MAX_CHUNKS))
    if len(all_files) != MAX_CHUNKS:
        raise RuntimeError(
            f"Requested {MAX_CHUNKS} shards, but only discovered {len(all_files)} in {HF_REPO_ID}"
        )
    preenum_source = PreenumeratedSource(all_files, hf_source)

    print(f"\n=========================================")
    print(f"REAL SCHNET ON MD17: {NUM_RUNS} PAIRED, ORDER-BALANCED RUNS")
    print(f"SHARDS PER PASS: {MAX_CHUNKS}")
    print(f"=========================================")

    active_model = get_real_mlff_model()
    initial_state_dict = copy.deepcopy(active_model.state_dict())
    hf_source.transfer_observer = None
    warmup_chunk = hf_source.download_chunk(all_files[0])
    warm_up_model(active_model, initial_state_dict, warmup_chunk)
    del warmup_chunk
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = {
        "sequential": {"time": [], "peak_rss": [], "avg_gpu": [], "trace": []},
        "foldpipe": {"time": [], "peak_rss": [], "avg_gpu": [], "trace": []},
    }
    reference_traces = {}
    run_records = []
    runners = {
        "sequential": run_sequential_stream,
        "foldpipe": run_foldpipe_stream,
    }

    for run_idx in range(NUM_RUNS):
        order = pipeline_order(run_idx)
        print(f"  --- PAIRED RUN {run_idx + 1}/{NUM_RUNS}: {' -> '.join(order)} ---")
        run_record = {"run_index": run_idx, "order": order, "pipelines": {}}

        for pipeline in order:
            trace = RunTrace(pipeline, run_idx, all_files)
            hf_source.transfer_observer = trace.on_transfer
            profiler, elapsed = runners[pipeline](
                preenum_source, active_model, initial_state_dict, trace
            )
            trace_dict = trace.as_dict()

            metrics[pipeline]["time"].append(elapsed)
            metrics[pipeline]["peak_rss"].append(profiler.peak_rss / (1024**3))
            metrics[pipeline]["avg_gpu"].append(
                float(np.mean(profiler.gpu_history)) if profiler.gpu_history else 0.0
            )
            metrics[pipeline]["trace"].append(trace_dict)
            run_record["pipelines"][pipeline] = trace_dict
            reference_traces.setdefault(pipeline, profiler)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        run_records.append(run_record)

    hf_source.transfer_observer = None

    experiment_results = {
        "schema_version": 2,
        "metadata": {
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "code": git_metadata(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dataset_repo": HF_REPO_ID,
            "dataset_revision": dataset_revision,
            "shard_identifiers": all_files,
            "shards_per_pass": MAX_CHUNKS,
            "paired_runs": NUM_RUNS,
            "batch_size": 32,
            "warmup_protocol": "one untimed training batch from the first pinned shard",
            "order_protocol": "alternating paired order",
            "confidence_interval": "95% percentile bootstrap",
        },
        "runs": run_records,
    }

    for pipeline_index, pipeline in enumerate(("sequential", "foldpipe")):
        trace_summaries = [item["summary"] for item in metrics[pipeline]["trace"]]
        experiment_results[pipeline] = {
            "time_s": aggregate(metrics[pipeline]["time"], seed=20260817 + pipeline_index),
            "throughput_shards_per_s": aggregate(
                [MAX_CHUNKS / value for value in metrics[pipeline]["time"]],
                seed=20260827 + pipeline_index,
            ),
            "peak_rss_gb": aggregate(
                metrics[pipeline]["peak_rss"], seed=20260837 + pipeline_index
            ),
            "avg_gpu_util_percent": aggregate(
                metrics[pipeline]["avg_gpu"], seed=20260847 + pipeline_index
            ),
            "io_time_s": aggregate(
                [item["io_time_s"] for item in trace_summaries],
                seed=20260857 + pipeline_index,
            ),
            "compute_time_s": aggregate(
                [item["compute_time_s"] for item in trace_summaries],
                seed=20260867 + pipeline_index,
            ),
            "overlap_time_s": aggregate(
                [item["overlap_time_s"] for item in trace_summaries],
                seed=20260877 + pipeline_index,
            ),
            "gpu_wait_time_s": aggregate(
                [item["gpu_wait_time_s"] for item in trace_summaries],
                seed=20260887 + pipeline_index,
            ),
        }

    paired_speedups = [
        sequential / foldpipe
        for sequential, foldpipe in zip(
            metrics["sequential"]["time"], metrics["foldpipe"]["time"]
        )
    ]
    paired_time_saved = [
        sequential - foldpipe
        for sequential, foldpipe in zip(
            metrics["sequential"]["time"], metrics["foldpipe"]["time"]
        )
    ]
    experiment_results["paired_effect"] = {
        "speedup_ratio": aggregate(paired_speedups, seed=20260897),
        "time_saved_s": aggregate(paired_time_saved, seed=20260907),
        "foldpipe_faster_fraction": float(
            np.mean(np.asarray(paired_time_saved, dtype=float) > 0)
        ),
    }

    with open(f"results/benchmark_stats_md17.json", "w") as f:
        json.dump(experiment_results, f, indent=4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    seq_prof = reference_traces["sequential"]
    fp_prof = reference_traces["foldpipe"]

    ax1.plot(seq_prof.time_history, seq_prof.ram_history, color='blue', alpha=0.7, label='Sequential Stream (O(1) RAM)')
    ax1.plot(fp_prof.time_history, fp_prof.ram_history, color='green', label='FoldPipe Async (O(1) RAM)')
    ax1.set_title(f"RAM Footprint (MD17 SchNet, Representative Pass)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("RAM (GB)")
    ax1.legend()

    ax2.plot(seq_prof.time_history, seq_prof.gpu_history, color='blue', alpha=0.7, label='Sequential Stream')
    ax2.plot(fp_prof.time_history, fp_prof.gpu_history, color='green', alpha=0.8, label='FoldPipe Async')
    ax2.set_title(f"Sampled GPU Utilization (MD17 SchNet, Representative Pass)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Compute Utilization (%)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'results/benchmark_comparison_md17.png', dpi=300)
    print(f"Saved results/benchmark_comparison_md17.png")
