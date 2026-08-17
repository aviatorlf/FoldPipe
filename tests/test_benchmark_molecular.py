import copy
import json
import subprocess

import pytest
import torch

import scripts.benchmark_molecular as benchmark
from scripts.benchmark_molecular import (
    RunTrace,
    aggregate,
    geometric_mean_speedup,
    median_paired_difference,
    pipeline_order,
)


def test_pipeline_order_is_balanced_and_alternating():
    assert pipeline_order(0) == ["sequential", "foldpipe"]
    assert pipeline_order(1) == ["foldpipe", "sequential"]
    assert pipeline_order(2) == ["sequential", "foldpipe"]


def test_warm_up_resets_model_state(monkeypatch):
    model = torch.nn.Linear(1, 1)
    initial_state = copy.deepcopy(model.state_dict())

    monkeypatch.setattr(
        benchmark, "pyg_batch_fn", lambda *_args, **_kwargs: iter([object()])
    )

    def mutate_model(active_model, *_args):
        with torch.no_grad():
            active_model.weight.fill_(99.0)

    monkeypatch.setattr(benchmark, "train_batch", mutate_model)
    monkeypatch.setattr(benchmark, "synchronize_device", lambda: None)

    benchmark.warm_up_model(model, initial_state, [object()])

    assert torch.equal(model.weight, initial_state["weight"])
    assert torch.equal(model.bias, initial_state["bias"])


def test_aggregate_uses_sample_std_and_bootstrap_interval():
    result = aggregate([1.0, 2.0, 3.0], seed=7)

    assert result["mean"] == 2.0
    assert result["sample_std"] == 1.0
    assert result["ci_95"]["method"] == "percentile bootstrap"
    assert result["ci_95"]["low"] <= result["mean"] <= result["ci_95"]["high"]


def test_geometric_mean_speedup_bootstraps_in_log_space():
    result = geometric_mean_speedup([1.0, 2.0, 4.0], seed=7)

    assert result["estimate"] == pytest.approx(2.0)
    assert result["ci_95"]["low"] <= result["estimate"]
    assert result["ci_95"]["high"] >= result["estimate"]
    assert "log-ratio space" in result["ci_95"]["method"]


def test_median_paired_difference_has_its_own_bootstrap_interval():
    result = median_paired_difference([-1.0, 3.0, 5.0], seed=7)

    assert result["median"] == 3.0
    assert result["ci_95"]["low"] <= result["median"]
    assert result["ci_95"]["high"] >= result["median"]
    assert "median" in result["ci_95"]["method"]


def test_git_metadata_falls_back_to_clean_source_manifest(monkeypatch, tmp_path):
    manifest = {
        "base_git_commit": "abc123",
        "working_tree_dirty": False,
        "bundle_sha256": "def456",
    }
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def no_git(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(benchmark.subprocess, "check_output", no_git)
    monkeypatch.setenv("FOLDPIPE_SOURCE_MANIFEST", str(manifest_path))

    metadata = benchmark.git_metadata()

    assert metadata["commit"] == "abc123"
    assert metadata["dirty"] is False
    assert metadata["source_bundle"] == manifest


def test_run_trace_calculates_io_compute_overlap_and_bytes():
    trace = RunTrace("foldpipe", 0, ["shard-0", "shard-1"])
    trace.origin = 100.0
    trace.finish = 108.0
    trace.records = [
        {
            "shard_index": 0,
            "identifier": "shard-0",
            "download_start_s": 0.0,
            "download_finish_s": 2.0,
            "deserialize_finish_s": 2.5,
            "training_start_s": 2.5,
            "training_finish_s": 5.0,
            "bytes_downloaded": 10,
            "structures": 5,
        },
        {
            "shard_index": 1,
            "identifier": "shard-1",
            "download_start_s": 2.5,
            "download_finish_s": 4.5,
            "deserialize_finish_s": 4.75,
            "training_start_s": 5.0,
            "training_finish_s": 8.0,
            "bytes_downloaded": 20,
            "structures": 7,
        },
    ]

    summary = trace.summary()

    assert summary["wall_time_s"] == pytest.approx(8.0)
    assert summary["io_time_s"] == pytest.approx(4.0)
    assert summary["deserialize_time_s"] == pytest.approx(0.75)
    assert summary["compute_time_s"] == pytest.approx(5.5)
    assert summary["overlap_time_s"] == pytest.approx(2.0)
    assert summary["gpu_wait_time_s"] == pytest.approx(2.5)
    assert summary["bytes_downloaded"] == 30
    assert summary["structures"] == 12
