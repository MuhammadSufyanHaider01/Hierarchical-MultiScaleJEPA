from __future__ import annotations

import sys
import json
import hashlib
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from masking_head_seed_jobs import _job_script  # noqa: E402
from masking_head_multiseed_report import benchmark_packet, load_packet, _json_ready  # noqa: E402
from phase5_exits import _batch_progress  # noqa: E402
from masking_head_study import (  # noqa: E402
    EXITS,
    METHODS,
    SEEDS,
    TASKS,
    aggregate_method,
    derive_b0_from_p0,
)


def packet(method: str, seed: int) -> dict:
    tasks = {}
    for task_index, task_name in enumerate(TASKS):
        metric = "mean_distance_error" if task_name == "nrpos-outdoor" else "macro_accuracy"
        higher = task_name != "nrpos-outdoor"
        exits = {}
        for exit_layer in EXITS:
            value = float(seed + task_index + exit_layer / 10)
            if not higher:
                value = float(20 - seed - exit_layer / 10)
            exits[str(exit_layer)] = {"test": {metric: value}}
        tasks[task_name] = {
            "spec": {"metric_name": metric, "higher_is_better": higher},
            "exits": exits,
            "selected_exit_best_validation": EXITS[seed % len(EXITS)],
        }
    return {
        "method": method,
        "pretrain_seed": 0,
        "head_seed": seed,
        "split_seed": 2025,
        "protocol": {"test_evaluation": True},
        "tasks": tasks,
    }


def test_manifest_json_normalizes_mixed_mapping_keys():
    ready = _json_ready({"exits": {2: {"mean": 1.0}, "method": "p0"}})
    encoded = json.dumps(ready, sort_keys=True)
    assert json.loads(encoded) == {"exits": {"2": {"mean": 1.0}, "method": "p0"}}


def test_complete_aggregation_includes_per_seed_sd_and_exit_frequency():
    rows = {seed: packet("p1", seed) for seed in SEEDS}
    result = aggregate_method(rows, method="p1")
    rfs = result["tasks"]["rfs"]
    assert rfs["fixed"][8]["n"] == 7
    assert set(rfs["fixed"][8]["per_seed"]) == set(SEEDS)
    assert rfs["selected"]["sd"] > 0
    assert sum(rfs["selected"]["selection_frequency"].values()) == 7
    assert set(rfs["selected"]["selected_exit_by_seed"]) == set(SEEDS)


def test_missing_seed_and_method_or_seed_mismatch_fail():
    rows = {seed: packet("p1", seed) for seed in SEEDS[:-1]}
    with pytest.raises(ValueError, match="seed coverage"):
        aggregate_method(rows, method="p1")
    rows = {seed: packet("p1", seed) for seed in SEEDS}
    rows[3]["head_seed"] = 99
    with pytest.raises(ValueError, match="head seed mismatch"):
        aggregate_method(rows, method="p1")
    rows[3] = packet("p2", 3)
    with pytest.raises(ValueError, match="method mismatch"):
        aggregate_method(rows, method="p1")


def test_split_and_incomplete_task_or_exit_coverage_fail():
    rows = {seed: packet("p1", seed) for seed in SEEDS}
    rows[0]["split_seed"] = 12
    with pytest.raises(ValueError, match="split seed mismatch"):
        aggregate_method(rows, method="p1")
    rows[0] = packet("p1", 0)
    del rows[0]["tasks"]["rfs"]["exits"]["8"]
    with pytest.raises(ValueError, match="exit coverage"):
        aggregate_method(rows, method="p1")


def test_nr_position_error_keeps_lower_is_better_metric_and_b0_uses_p0_l8():
    rows = {seed: packet("p0", seed) for seed in SEEDS}
    result = aggregate_method(rows, method="p0")
    nrpos = result["tasks"]["nrpos-outdoor"]
    assert nrpos["higher_is_better"] is False
    assert nrpos["fixed"][8]["mean"] < nrpos["fixed"][2]["mean"]
    b0 = derive_b0_from_p0(result)
    assert b0["tasks"]["rfs"]["fixed_l8"] == result["tasks"]["rfs"]["fixed"][8]
    assert b0["tasks"]["nrpos-outdoor"]["per_seed"][0] == rows[0]["tasks"]["nrpos-outdoor"]["exits"]["8"]["test"]["mean_distance_error"]


def test_imported_p2_reverse_seed_2025_is_not_seed_zero():
    rows = {seed: packet("p2_reverse", seed) for seed in SEEDS}
    rows[0]["head_seed"] = 2025
    with pytest.raises(ValueError, match="head seed mismatch"):
        aggregate_method(rows, method="p2_reverse")


def test_batch_progress_changes_and_is_log_line_friendly():
    assert "001/004" in _batch_progress("rfs L2 epoch 001", 1, 4)
    assert "002/004" in _batch_progress("rfs L2 epoch 001", 2, 4)

def test_head_job_output_roots_are_seed_isolated_and_never_pretrain():
    entry = {"mask": "frequency_n6", "method": "p0", "checkpoint_path": "/fixed/p0.ckpt"}
    run_root = Path("/tmp/study")
    job0 = _job_script(entry, 0, run_root, benchmark=False, partition="gpu-a100")
    job1 = _job_script(entry, 1, run_root, benchmark=False, partition="gpu-a100")
    assert "--head-seed 0" in job0
    assert "--head-seed 1" in job1
    assert str(run_root / "frequency_n6/head_seed_000") in job0
    assert str(run_root / "frequency_n6/head_seed_001") in job1
    assert "phase5_pretrain.py" not in job0
    assert "phase5_exits.py" in job0
    assert "--split-seed 2025" in job0

def test_missing_raw_metric_artifact_fails_closed(tmp_path):
    entry = {"mask": "frequency_n6", "method": "p0", "seed0_reusable": False}
    with pytest.raises(FileNotFoundError, match="required raw artifact is missing"):
        load_packet({}, entry, 0, tmp_path)


def test_new_head_packet_rejects_runtime_profile_mismatch(tmp_path):
    mask, method, seed = "frequency_n6", "p1", 0
    run_root = tmp_path / "runs"
    checkpoint = tmp_path / "fixed_checkpoint.ckpt"
    checkpoint.write_bytes(b"fixed checkpoint")
    split = run_root / mask / "head_seed_000" / "splits_seed_2025_primary_v3.json"
    split.parent.mkdir(parents=True)
    split.write_text(json.dumps({"seed": 2025}))

    tasks = {}
    for task_name in TASKS:
        exits = {}
        for layer in EXITS:
            head = run_root / mask / "head_seed_000" / method / "seed_000" / "downstream" / "heads" / task_name / f"L{layer}.pt"
            head.parent.mkdir(parents=True, exist_ok=True)
            head.write_bytes(f"{task_name}-L{layer}".encode())
            exits[str(layer)] = {"head_path": str(head)}
        tasks[task_name] = {"exits": exits}

    metrics = run_root / mask / "head_seed_000" / method / "seed_000" / "downstream" / "exit_metrics.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "split_manifest": str(split),
        "tasks": tasks,
    }))
    runtime = run_root / "runtime" / mask / method / "head_seed_000" / "runtime.json"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(json.dumps({"runtime_profile": "source_faithful_p0"}))
    entry = {
        "mask": mask,
        "method": method,
        "seed0_reusable": False,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }

    with pytest.raises(ValueError, match="runtime profile mismatch"):
        load_packet({}, entry, seed, run_root)


def test_imported_p2_reverse_benchmark_keeps_imported_provenance(tmp_path):
    checkpoint = tmp_path / "p2_reverse.ckpt"
    checkpoint.write_bytes(b"fixture checkpoint")
    benchmark_path = tmp_path / "benchmarks.json"
    benchmark = {
        "method": "p2_reverse",
        "pretrain_seed": 0,
        "batch_size": 1,
        "checkpoint": str(checkpoint),
        "entries": [
            {"task": task, "exit_layer": layer}
            for task in TASKS for layer in EXITS
        ],
    }
    benchmark_path.write_text(json.dumps(benchmark))
    entry = {
        "mask": "frequency_n6", "method": "p2_reverse",
        "checkpoint_path": str(checkpoint), "benchmark_reusable": True,
        "benchmark_source_path": str(benchmark_path),
    }
    _, source = benchmark_packet(entry, tmp_path)
