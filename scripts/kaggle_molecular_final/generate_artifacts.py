"""Generate a self-contained, reproducible Kaggle benchmark notebook."""

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
PAIRED_RUNS = 20
DATASET_REVISION = "f779686deb9217877dd7ddde99b2522bd441492a"
FOLDPIPE_VERSION = "0.3.1"
SOURCE_PATHS = (
    "scripts/benchmark_molecular.py",
)


def git_output(*args):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, encoding="utf-8"
    ).strip()


def source_bundle():
    payloads = {}
    files = {}
    bundle_hash = hashlib.sha256()
    for relative_path in SOURCE_PATHS:
        payload = (ROOT / relative_path).read_bytes()
        payloads[relative_path] = base64.b64encode(payload).decode("ascii")
        files[relative_path] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        bundle_hash.update(relative_path.encode("utf-8"))
        bundle_hash.update(b"\0")
        bundle_hash.update(payload)
        bundle_hash.update(b"\0")

    manifest = {
        "format_version": 1,
        "base_git_commit": git_output("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git_output("status", "--porcelain")),
        "bundle_sha256": bundle_hash.hexdigest(),
        "foldpipe_distribution": {
            "name": "foldpipe",
            "version": FOLDPIPE_VERSION,
            "index_url": f"https://pypi.org/project/foldpipe/{FOLDPIPE_VERSION}/",
        },
        "files": files,
    }
    return payloads, manifest


def build_notebook(payloads, manifest):
    install_code = """import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "foldpipe==__FOLDPIPE_VERSION__",
        "matplotlib",
        "psutil",
    ],
    check=True,
)

from importlib.metadata import version
assert version("foldpipe") == "__FOLDPIPE_VERSION__"

import torch

if not torch.cuda.is_available():
    raise RuntimeError("The benchmark requires the requested NVIDIA T4 GPU")

torch_version = torch.__version__.split("+")[0]
cuda_tag = torch.version.cuda.replace(".", "")
pyg_wheels = f"https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_tag}.html"
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "pyg_lib",
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
        "torch_spline_conv",
        "-f",
        pyg_wheels,
    ],
    check=True,
)

print({
    "foldpipe": version("foldpipe"),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
})
"""
    install_code = install_code.replace("__FOLDPIPE_VERSION__", FOLDPIPE_VERSION)

    materialize_code = f"""import base64
import json
import os
from pathlib import Path

WORK_ROOT = Path("/kaggle/working/foldpipe-benchmark")
PAYLOADS = {json.dumps(payloads, sort_keys=True)}
SOURCE_MANIFEST = json.loads({json.dumps(json.dumps(manifest, sort_keys=True))})

for relative_path, encoded_payload in PAYLOADS.items():
    destination = WORK_ROOT / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(encoded_payload))

manifest_path = WORK_ROOT / "source_manifest.json"
manifest_path.write_text(
    json.dumps(SOURCE_MANIFEST, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
os.chdir(WORK_ROOT)
print({{
    "source_bundle_sha256": SOURCE_MANIFEST["bundle_sha256"],
    "base_git_commit": SOURCE_MANIFEST["base_git_commit"],
    "embedded_files": len(SOURCE_MANIFEST["files"]),
}})
"""

    run_code = '''import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from kaggle_secrets import UserSecretsClient

secret_token = UserSecretsClient().get_secret("HF_TOKEN")
run_environment = os.environ.copy()
run_environment.update({
    "HF_TOKEN": secret_token,
    "FOLDPIPE_HF_REPO_ID": "aviatorlf/md17-shards",
    "FOLDPIPE_HF_REVISION": "__DATASET_REVISION__",
    "FOLDPIPE_MAX_CHUNKS": "5",
    "FOLDPIPE_NUM_RUNS": "20",
    "FOLDPIPE_BOOTSTRAP_SAMPLES": "20000",
    "FOLDPIPE_SOURCE_MANIFEST": str(Path.cwd() / "source_manifest.json"),
    "PYTHONUNBUFFERED": "1",
})
subprocess.run(
    [sys.executable, "scripts/benchmark_molecular.py"],
    env=run_environment,
    check=True,
)
del secret_token
run_environment.pop("HF_TOKEN", None)

results_path = Path("results/benchmark_stats_md17.json")
plot_path = Path("results/benchmark_comparison_md17.png")
results = json.loads(results_path.read_text(encoding="utf-8"))

seq = results["sequential"]
fold = results["foldpipe"]
effect = results["paired_effect"]
geometric = effect["geometric_mean_speedup"]
geometric_ci = geometric["ci_95"]
mean_saved = effect["time_saved_s"]
mean_saved_ci = mean_saved["ci_95"]
median_saved = effect["median_time_saved_s"]
median_saved_ci = median_saved["ci_95"]
geometric_includes_null = geometric_ci["low"] <= 1.0 <= geometric_ci["high"]
mean_saved_includes_null = mean_saved_ci["low"] <= 0.0 <= mean_saved_ci["high"]
median_saved_includes_null = median_saved_ci["low"] <= 0.0 <= median_saved_ci["high"]
if geometric_includes_null and mean_saved_includes_null and median_saved_includes_null:
    interpretation = "All three paired intervals include their no-effect values; this run is inconclusive about a speed advantage."
elif (
    not geometric_includes_null
    and not mean_saved_includes_null
    and not median_saved_includes_null
    and geometric_ci["low"] > 1.0
    and mean_saved_ci["low"] > 0.0
    and median_saved_ci["low"] > 0.0
):
    interpretation = (
        "All three paired intervals exclude their no-effect values in FoldPipe's favor for this protocol. "
        "This supports a conditional benefit under the measured environment, not a universal speedup claim."
    )
else:
    interpretation = (
        "The multiplicative and additive paired estimands do not give uniformly decisive intervals. "
        "Under high run-to-run network variability, all summaries and raw runs should be reported rather than presenting a universal speedup."
    )

report = f"""# FoldPipe MD17 + SchNet benchmark

- Generated: {results['metadata']['generated_at_utc']}
- Hardware: {results['metadata']['gpu']}
- Dataset: `{results['metadata']['dataset_repo']}@{results['metadata']['dataset_revision']}`
- FoldPipe distribution: `{results['metadata']['code']['source_bundle']['foldpipe_distribution']['name']}=={results['metadata']['code']['source_bundle']['foldpipe_distribution']['version']}` from PyPI
- Source bundle: `{results['metadata']['code']['source_bundle']['bundle_sha256']}`
- Benchmark-driver Git commit: `{results['metadata']['code']['source_bundle']['base_git_commit']}`
- Protocol: {results['metadata']['paired_runs']} paired, order-alternating passes; {results['metadata']['shards_per_pass']} pinned shards per pass; batch size {results['metadata']['batch_size']}
- Warm-up: {results['metadata']['warmup_protocol']}

| Metric | Sequential | FoldPipe |
| --- | ---: | ---: |
| Mean time (s) | {seq['time_s']['mean']:.3f} | {fold['time_s']['mean']:.3f} |
| 95% bootstrap CI, mean time (s) | [{seq['time_s']['ci_95']['low']:.3f}, {seq['time_s']['ci_95']['high']:.3f}] | [{fold['time_s']['ci_95']['low']:.3f}, {fold['time_s']['ci_95']['high']:.3f}] |
| Mean peak RSS (GiB) | {seq['peak_rss_gb']['mean']:.3f} | {fold['peak_rss_gb']['mean']:.3f} |
| Mean sampled GPU utilization (%) | {seq['avg_gpu_util_percent']['mean']:.3f} | {fold['avg_gpu_util_percent']['mean']:.3f} |
| Mean I/O/compute overlap (s) | {seq['overlap_time_s']['mean']:.3f} | {fold['overlap_time_s']['mean']:.3f} |
| Mean GPU wait time (s) | {seq['gpu_wait_time_s']['mean']:.3f} | {fold['gpu_wait_time_s']['mean']:.3f} |

Geometric mean paired speedup: **{geometric['estimate']:.4f}x** (95% paired bootstrap CI in log-ratio space [{geometric_ci['low']:.4f}, {geometric_ci['high']:.4f}]).

Mean paired time saved: **{mean_saved['mean']:.3f} s** (95% paired bootstrap CI [{mean_saved_ci['low']:.3f}, {mean_saved_ci['high']:.3f}]).

Median paired time saved: **{median_saved['median']:.3f} s** (95% paired bootstrap CI [{median_saved_ci['low']:.3f}, {median_saved_ci['high']:.3f}]). FoldPipe was faster in {effect['foldpipe_faster_fraction']:.0%} of pairs.

For continuity with the earlier artifact, the arithmetic mean of paired speedup ratios was **{effect['speedup_ratio']['mean']:.4f}x**; it is retained as a supplementary, skew-sensitive summary rather than the headline ratio.

{interpretation}

The JSON artifact contains every raw paired duration and per-shard download, deserialization, training, payload-byte, overlap, and wait-time trace.
"""

output_root = Path("/kaggle/working")
shutil.copy2(results_path, output_root / "benchmark_stats_md17.json")
shutil.copy2(plot_path, output_root / "benchmark_comparison_md17.png")
shutil.copy2(
    "source_manifest.json",
    output_root / "benchmark_source_manifest_md17.json",
)
(output_root / "benchmark_report_md17.md").write_text(report, encoding="utf-8")

print(report)
'''
    run_code = run_code.replace("__DATASET_REVISION__", DATASET_REVISION)

    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# FoldPipe MD17 + SchNet rigorous benchmark\n\n"
            f"{PAIRED_RUNS} paired, order-alternating passes on five revision-pinned private MD17 shards. "
            f"FoldPipe {FOLDPIPE_VERSION} is installed from PyPI; only the benchmark driver is embedded for provenance. "
            "The notebook emits raw traces, bootstrap intervals, a plot, a source manifest, and a Markdown report."
        ),
        nbf.v4.new_code_cell(install_code),
        nbf.v4.new_code_cell(materialize_code),
        nbf.v4.new_code_cell(run_code),
    ]
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    return notebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for the generated Kaggle notebook and metadata",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads, manifest = source_bundle()
    notebook = build_notebook(payloads, manifest)
    nbf.write(notebook, output_dir / "benchmark.ipynb")

    metadata = {
        "id": "dhirenkhatri/foldpipe-md17-rigorous-benchmark",
        "title": "foldpipe-md17-rigorous-benchmark",
        "code_file": "benchmark.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
