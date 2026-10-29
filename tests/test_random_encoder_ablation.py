from __future__ import annotations

import json
import csv
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import random_encoder_ablation as control  # noqa: E402
import random_encoder_ablation_jobs as jobs  # noqa: E402
import random_encoder_ablation_report as report  # noqa: E402
from models import build_encoder, load_context_encoder_checkpoint  # noqa: E402
from configs import init_jepa_config  # noqa: E402


def test_random_checkpoint_deterministic_and_tamper_rejected(tmp_path):
    first, second = tmp_path / "a.pt", tmp_path / "b.pt"
    control.generate(first, tmp_path / "a.json")
    control.generate(second, tmp_path / "b.json")
    a, b = torch.load(first, map_location="cpu", weights_only=False), torch.load(second, map_location="cpu", weights_only=False)
    assert all(torch.equal(a["state_dict"][key], b["state_dict"][key]) for key in a["state_dict"])
    model = build_encoder(init_jepa_config(str(control.CONFIG)).encoder)
    load_context_encoder_checkpoint(model, first)
    metadata = json.loads((tmp_path / "a.json").read_text()); metadata["initialization_seed"] = 2
    (tmp_path / "a.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="provenance"):
        control.validate(first, tmp_path / "a.json")


def test_jobs_are_isolated_and_report_depends_on_both(tmp_path):
    scripts = [jobs.job_script(algorithm, tmp_path) for algorithm in jobs.ALGORITHMS]
    assert all("--encoder-kind random" in script and "--head-seed 0" in script and "#SBATCH --gres=gpu:l40:1" in script for script in scripts)
    assert all(str(tmp_path / "random_encoder_seed_000.pt") in script for script in scripts)
    assert "--adapter-variant single_gelu" in scripts[1]
    prepared = {"jobs": [{"algorithm": name, "script": name} for name in jobs.ALGORITHMS], "report_script": "report"}
    with patch.object(jobs.subprocess, "check_output", side_effect=["101\n", "102\n", "103\n"]) as calls:
        result = jobs.submit(prepared, tmp_path)
    assert result["report_job"]["dependency"] == "afterok:101:102"
    assert "--dependency=afterok:101:102" in calls.call_args.args[0]
    with pytest.raises(FileExistsError):
        jobs.submit(prepared, tmp_path)


def test_report_rejects_absent_and_incompatible_random_packet(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="packet missing"):
        report.load_random("current", tmp_path)
    path = report.random_path("current", tmp_path); path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"encoder_kind": "pretrained"}))
    monkeypatch.setattr(control, "validate", lambda *args: {"checkpoint": "unused", "checkpoint_sha256": "bad"})
    with pytest.raises(ValueError, match="provenance"):
        report.load_random("current", tmp_path)


def test_report_rejects_missing_exit_and_fusion_diagnostics(tmp_path, monkeypatch):
    checkpoint = tmp_path / "random_encoder_seed_000.pt"; checkpoint.write_text("random")
    metadata = tmp_path / "random_encoder_seed_000.json"; metadata.write_text("{}")
    split = tmp_path / "split.json"; split.write_text("{}")
    head = tmp_path / "head.pt"; head.write_text("head")
    monkeypatch.setattr(control, "SPLIT", split)
    monkeypatch.setattr(control, "validate", lambda *args: {"checkpoint": str(checkpoint), "checkpoint_sha256": control.sha(checkpoint)})
    entries = {str(layer): {"head_path": str(head), "checkpoint_sha256": control.sha(head),
               "validation": {"accuracy": .5}, "test": {"accuracy": .5},
               "fusion_diagnostics": {str(source): {"gate_mean": .5} for source in report.EXITS if source < layer}}
               for layer in report.EXITS}
    packet = {"encoder_kind": "random", "method": "p0", "head_seed": 0, "pretrain_seed": None, "split_seed": 2025,
              "head_algorithm": "residual_fusion_adapter_ablation", "adapter_variant": "single_gelu",
              "checkpoint": str(checkpoint), "checkpoint_sha256": control.sha(checkpoint),
              "random_encoder_metadata": str(metadata), "split_manifest": str(split), "split_manifest_sha256": control.sha(split),
              "protocol": {"head_max_epochs": 200, "batch_size": 32, "learning_rate": .001, "weight_decay": .05,
                           "tasks": list(report.TASKS), "exit_layers": list(report.EXITS), "test_evaluation": True,
                           "selection_uses": "validation_only", "head_early_stopping": {"enabled": True, "patience": 20}},
              "tasks": {task: {"spec": {"metric_name": "accuracy"}, "selected_exit_best_validation": 2,
                               "exits": json.loads(json.dumps(entries))} for task in report.TASKS}}
    path = report.random_path("single_gelu", tmp_path); path.parent.mkdir(parents=True)
    del packet["tasks"]["rfs"]["exits"]["8"]
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="exit coverage"):
        report.load_random("single_gelu", tmp_path)
    packet["tasks"]["rfs"]["exits"]["8"] = entries["8"]
    packet["tasks"]["rfs"]["exits"]["8"]["fusion_diagnostics"] = {}
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="diagnostics"):
        report.load_random("single_gelu", tmp_path)


def test_pretraining_and_fusion_direction_for_positioning():
    assert report.prior.gain(3.0, 5.0, False) == 2.0
    assert report.prior.gain(0.8, 0.6, True) == pytest.approx(0.2)


def test_report_generates_six_sources_and_complete_figures(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"; checkpoint.write_text("checkpoint")
    split = tmp_path / "split"; split.write_text("split")
    sources = {}
    for condition in report.CONDITIONS:
        for algorithm in report.ALGORITHMS:
            path = tmp_path / f"{condition}_{algorithm}.json"; path.write_text("{}")
            tasks = {}
            for task in report.TASKS:
                higher = task != "nrpos-outdoor"
                metric = "macro_accuracy" if higher else "mean_distance_error"
                value = {"random": 0.1 if higher else 10.0, "p0": 0.2 if higher else 8.0, "p2": 0.3 if higher else 6.0}[condition]
                if algorithm == "single_gelu":
                    value += 0.01 if higher else -1.0
                exits = {str(layer): {"validation": {metric: value}, "test": {metric: value},
                         "fusion_diagnostics": {str(source): {"gate_mean": 0.5} for source in report.EXITS if source < layer}}
                         for layer in report.EXITS}
                tasks[task] = {"spec": {"metric_name": metric, "higher_is_better": higher}, "exits": exits,
                               "selected_exit_best_validation": 4}
            sources[condition, algorithm] = (path, {"checkpoint": str(checkpoint), "split_manifest": str(split), "tasks": tasks})
    monkeypatch.setattr(report, "read_sources", lambda *args: sources)
    monkeypatch.setattr(control, "validate", lambda *args: {"encoder_kind": "random", "initialization_seed": 0})
    runtime = tmp_path / "runtime"; runtime.mkdir()
    for algorithm in report.ALGORITHMS:
        (runtime / f"{algorithm}.json").write_text("{}")
    output = tmp_path / "report"
    result = report.build_report(tmp_path, output)
    assert result["sources"] == 6 and result["fixed_rows"] == 144 and result["selected_rows"] == 36
    assert len(result["figures"]) == 4 and all((output / name).is_file() for name in result["figures"])
    with (output / "direction_aware_gains.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    p2 = next(row for row in rows if row["comparison_type"] == "pretraining" and row["encoder"] == "p2"
              and row["algorithm"] == "current" and row["task"] == "nrpos-outdoor" and row["comparison"] == "fixed_L2")
    assert float(p2["direction_aware_gain"]) == 4.0
    with pytest.raises(FileExistsError):
        report.build_report(tmp_path, output)
