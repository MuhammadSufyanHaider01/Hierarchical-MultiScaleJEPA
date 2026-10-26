from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]

import single_gelu_fusion_multiseed_jobs as jobs  # noqa: E402
import single_gelu_fusion_multiseed_report as report  # noqa: E402


def test_audit_reuses_only_seed_zero_and_covers_current_baselines():
    result = jobs.audit()
    assert result["passed"], result["failures"]
    assert result["missing_single_gelu_seeds"] == (1, 2, 3, 4, 5, 6)
    assert {(row["method"], row["head_seed"]) for row in result["inputs"] if row["algorithm"] == "current"} == {
        (method, seed) for method in report.METHODS for seed in report.SEEDS
    }
    assert {(row["method"], row["head_seed"]) for row in result["inputs"] if row["algorithm"] == "single_gelu"} == {
        (method, 0) for method in report.METHODS
    }


@pytest.mark.parametrize("method", report.METHODS)
def test_training_script_is_seed_isolated_and_protocol_locked(method, tmp_path):
    script = jobs.job_script(method, 3, "gpu-l40", tmp_path)
    assert "--adapter-variant single_gelu" in script
    assert "--head-seed 3" in script and "--split-seed 2025" in script
    assert "--epochs 200" in script and "--batch-size 32" in script
    assert "--learning-rate 0.001" in script and "--weight-decay 0.05" in script
    assert "--head-early-stopping-patience 20" in script
    assert "#SBATCH --gres=gpu:l40:1" in script
    assert "#SBATCH --cpus-per-task=8" in script and "#SBATCH --mem=32G" in script
    assert str(tmp_path / "runtime_single_gelu_multiseed" / method / "head_seed_003.json") in script
    with pytest.raises(ValueError, match="seeds 1–6"):
        jobs.job_script(method, 0, "gpu-l40", tmp_path)


def test_report_dependency_lists_all_twelve_job_ids(monkeypatch, tmp_path):
    prepared = {"prepared_jobs": [
        {"kind": "single_gelu_fusion_train", "method": method, "head_seed": seed, "script": f"{method}-{seed}"}
        for method in report.METHODS for seed in jobs.MISSING_SEEDS
    ], "report_script": "report.sbatch"}
    calls = []

    class Response:
        def __init__(self, value):
            self.stdout = value

    def fake_run(command, **_kwargs):
        calls.append(command)
        return Response(f"{900 + len(calls)};cluster\n")

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)
    result = jobs.submit(prepared, tmp_path)
    assert len(result["training_jobs"]) == 12
    assert len(calls) == 13
    assert calls[-1][2].startswith("--dependency=afterok:")
    assert len(calls[-1][2].split(":", 1)[1].split(":")) == 12


def test_report_rejects_missing_or_wrong_identity_packet(tmp_path):
    with pytest.raises(FileNotFoundError, match="required single_gelu packet missing"):
        report.load_packet("p0", "single_gelu", 1, tmp_path, tmp_path)
    with pytest.raises(ValueError, match="unsupported algorithm"):
        report.packet_path("p0", "residual_projected_dynamic_fusion", 0, tmp_path, tmp_path)


def _fake_packet(method: str, algorithm: str, seed: int) -> dict:
    tasks = {}
    for task in report.TASKS:
        metric = "mean_distance_error" if task == "nrpos-outdoor" else "macro_accuracy"
        higher = task != "nrpos-outdoor"
        exits = {}
        for layer in report.EXITS:
            value = (3.0 - seed * .01 - layer * .001 if not higher else .5 + seed * .01 + layer * .001)
            entry = {"validation": {metric: value}, "test": {metric: value}, "head_path": "unused"}
            if algorithm == "single_gelu":
                entry["fusion_diagnostics"] = {str(source): {field: .5 for field in (
                    "gate_mean", "gate_median", "gate_p10", "gate_p90", "gate_sd",
                    "relative_residual_norm_mean", "residual_norm_mean")}
                    for source in report.EXITS if source < layer}
            exits[str(layer)] = entry
        tasks[task] = {"spec": {"metric_name": metric, "higher_is_better": higher},
                       "selected_exit_best_validation": 4, "exits": exits}
    return {"tasks": tasks, "checkpoint": str(report.CHECKPOINT_PATHS[method]),
            "_checkpoint_sha256": report.CHECKPOINT_HASHES[method], "_split_sha256": "shared-split"}


def test_report_aggregates_exact_sources_and_replaces_stale_output(monkeypatch, tmp_path):
    packets = {}
    for method in report.METHODS:
        for algorithm in report.ALGORITHMS:
            for seed in report.SEEDS:
                path = tmp_path / f"{method}-{algorithm}-{seed}.json"
                path.write_text("{}\n")
                packets[method, algorithm, seed] = (path, _fake_packet(method, algorithm, seed))
    monkeypatch.setattr(report, "load_packet", lambda method, algorithm, seed, *_args: packets[method, algorithm, seed])
    monkeypatch.setattr(report, "plot_figures", lambda _summary, _output: ["fixture.png"])
    output = tmp_path / "report"
    output.mkdir()
    (output / "p1_legacy_file.txt").write_text("stale\n")
    result = report.build_report(output=output)
    assert result["sources"] == 28
    assert result["fixed_rows"] == 672
    assert result["selected_rows"] == 168
    assert result["gain_rows"] == 420
    assert not (output / "p1_legacy_file.txt").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert tuple(manifest["methods"]) == report.METHODS
    assert tuple(manifest["algorithms"]) == report.ALGORITHMS
    assert len(manifest["sources"]) == 28


def test_figure_styling_uses_sd_only_for_individual_tasks(monkeypatch, tmp_path):
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    summary = [{"method": method, "algorithm": algorithm, "task": task, "exit": layer,
                "mean": .5, "sample_sd": .1, "n": 7}
               for method in report.METHODS for algorithm in report.ALGORITHMS
               for task in report.TASKS for layer in report.EXITS]
    errorbars, saved = [], []
    original_errorbar, original_savefig = Axes.errorbar, Figure.savefig

    def track_errorbar(self, *args, **kwargs):
        errorbars.append((args, kwargs))
        return original_errorbar(self, *args, **kwargs)

    def track_savefig(self, *args, **kwargs):
        saved.append(self)
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", track_errorbar)
    monkeypatch.setattr(Figure, "savefig", track_savefig)
    names = report.plot_figures(summary, tmp_path)
    assert len(errorbars) == 24  # P0/P2 × six tasks × Current/Fusion.
    assert {name for name in names if name.endswith(".png")} == {
        f"{method}_{task}_exit_performance.png" for method in report.METHODS for task in report.TASKS
    } | {"p0_p2_six_task_single_gelu_comparison.png"}
    compact = saved[-2]
    assert all(len(axis.lines) == 4 for axis in compact.axes)
    for axis in compact.axes:
        assert [line.get_linestyle() for line in axis.lines] == [":", "-", ":", "-"]
        assert axis.lines[0].get_color() == axis.lines[1].get_color()
        assert axis.lines[2].get_color() == axis.lines[3].get_color()
        assert axis.lines[0].get_color() != axis.lines[2].get_color()
