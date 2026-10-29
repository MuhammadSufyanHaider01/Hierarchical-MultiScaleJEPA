from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import random_encoder_ablation as control
import random_encoder_ablation_multiseed_jobs as jobs
import random_encoder_ablation_multiseed_report as report
import random_encoder_fixed_head_report as fixed_head_report


def test_encoder_seed_and_script_isolation(tmp_path):
    records = []
    for seed in (1, 2):
        checkpoint, metadata = tmp_path / f"{seed}.pt", tmp_path / f"{seed}.json"
        records.append(control.generate(checkpoint, metadata, seed))
        with pytest.raises(ValueError, match="provenance"):
            control.validate(checkpoint, metadata, 0)
    assert records[0]["checkpoint_sha256"] != records[1]["checkpoint_sha256"]
    for seed in range(1, 7):
        for algorithm in report.ALGORITHMS:
            script = jobs.job_script(algorithm, seed, tmp_path)
            assert f"--head-seed {seed}" in script
            assert str(tmp_path / f"seed_{seed:03d}" / f"random_encoder_seed_{seed:03d}.pt") in script
            assert "--encoder-kind random" in script and "#SBATCH --gres=gpu:l40:1" in script
            assert "--epochs 200" in script and "--head-early-stopping-patience 20" in script
            if algorithm == "single_gelu":
                assert "--adapter-variant single_gelu" in script


def test_report_dependency_contains_all_jobs(tmp_path):
    items = [{"algorithm": algorithm, "seed": seed, "script": f"{algorithm}-{seed}"}
             for seed in range(1, 7) for algorithm in report.ALGORITHMS]
    prepared = {"jobs": items, "report_script": "report"}
    with patch.object(jobs.subprocess, "check_output", side_effect=[f"{x}\n" for x in range(101, 114)]) as calls:
        result = jobs.submit(prepared, tmp_path)
    assert result["report_job"]["dependency"] == "afterok:" + ":".join(str(x) for x in range(101, 113))
    assert calls.call_count == 13
    with pytest.raises(FileExistsError):
        jobs.submit(prepared, tmp_path)


def test_summary_requires_seven_seeds_and_counts_selections():
    rows = [{"encoder": "random", "seed": seed, "selected_exit": 4 if seed < 5 else 8, "value": seed} for seed in report.SEEDS]
    summary, spread = report.summarize(rows, ("encoder",), "value")
    assert summary[0]["mean"] == 3 and json.loads(summary[0]["selection_frequency"]) == {"4": 5, "8": 2}
    assert spread[0]["sample_variance"] == pytest.approx(14 / 3)
    with pytest.raises(ValueError, match="coverage"):
        report.summarize(rows[:-1], ("encoder",), "value")


def test_report_builds_complete_paired_comparison(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"; checkpoint.write_text("checkpoint")
    split = tmp_path / "split"; split.write_text("split")
    sources = {}
    for condition in report.CONDITIONS:
        for algorithm in report.ALGORITHMS:
            for seed in report.SEEDS:
                path = tmp_path / f"{condition}_{algorithm}_{seed}.json"; path.write_text("{}")
                tasks = {}
                for task in report.TASKS:
                    higher = task != "nrpos-outdoor"
                    metric = "macro_accuracy" if higher else "mean_distance_error"
                    value = {"random": .1 if higher else 10., "p0": .2 if higher else 8., "p2": .3 if higher else 6.}[condition]
                    entries = {str(layer): {"validation": {metric: value}, "test": {metric: value},
                        "fusion_diagnostics": {str(source): {"gate_mean": .5} for source in report.EXITS if source < layer}}
                        for layer in report.EXITS}
                    tasks[task] = {"spec": {"metric_name": metric, "higher_is_better": higher}, "exits": entries,
                        "selected_exit_best_validation": 4}
                sources[condition, algorithm, seed] = (path, {"checkpoint": str(checkpoint), "split_manifest": str(split), "tasks": tasks})
    monkeypatch.setattr(report, "read_sources", lambda *args: sources)
    (tmp_path / "registry.json").write_text("{}")
    (tmp_path / "submissions.json").write_text("{}")
    monkeypatch.setattr(report, "seed_root", lambda seed, root: tmp_path / f"seed_{seed:03d}")
    for seed in report.SEEDS:
        runtime = tmp_path / f"seed_{seed:03d}" / "runtime"; runtime.mkdir(parents=True)
        for algorithm in report.ALGORITHMS:
            (runtime / f"{algorithm}.json").write_text("{}")
    output = tmp_path / "report"
    result = report.build_report(tmp_path, output)
    assert result["sources"] == 42 and len(result["figures"]) == 4
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["sources"]) == 42
    with (output / "direction_aware_gain_summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r["comparison_type"] == "pretraining" and r["encoder"] == "p2" and r["algorithm"] == "current"
               and r["task"] == "nrpos-outdoor" and r["comparison"] == "fixed_L2")
    assert float(row["mean"]) == 4.0
    with pytest.raises(FileExistsError):
        report.build_report(tmp_path, output)


def test_fixed_head_report_accepts_legacy_current_ee_split_hash():
    assert fixed_head_report.split_hash_matches({}, "expected")
    assert fixed_head_report.split_hash_matches({"split_manifest_sha256": None}, "expected")
    assert fixed_head_report.split_hash_matches({"split_manifest_sha256": "expected"}, "expected")
    assert not fixed_head_report.split_hash_matches({"split_manifest_sha256": "wrong"}, "expected")
    assert fixed_head_report.current_head_identity_matches({})
    assert fixed_head_report.current_head_identity_matches({"head_algorithm": "current_ee"})
    assert not fixed_head_report.current_head_identity_matches({"head_algorithm": "other"})
