from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]

import residual_fusion_v2_multiseed_jobs as jobs  # noqa: E402
import residual_fusion_v2_multiseed_report as report  # noqa: E402
import residual_fusion_v2_additional_multiseed_jobs as additional_jobs  # noqa: E402
import phase5_residual_fusion_exits as fusion_runner  # noqa: E402


def test_v2_job_script_is_seed_isolated_and_locked_to_protocol(tmp_path):
    script = jobs.job_script("p0", 3, "gpu-l40", tmp_path)
    assert "#SBATCH --partition=gpu-l40" in script
    assert "#SBATCH --gres=gpu:l40:1" in script
    assert "#SBATCH --cpus-per-task=8" in script and "#SBATCH --mem=32G" in script
    assert "--head-seed 3" in script and "--split-seed 2025" in script
    assert "--epochs 200" in script and "--batch-size 32" in script
    assert "--head-early-stopping-patience 20" in script
    assert str(tmp_path.parent) in script
    with pytest.raises(ValueError, match="seeds 1–6"):
        jobs.job_script("p0", 0, "gpu-l40", tmp_path)


def test_report_script_is_cpu_only_and_uses_multiseed_report(tmp_path):
    script = jobs.report_script("cpu2023", tmp_path)
    assert "#SBATCH --partition=cpu2023" in script
    assert "#SBATCH --cpus-per-task=2" in script and "#SBATCH --mem=8G" in script
    assert "residual_fusion_v2_multiseed_report.py" in script
    assert "--gres" not in script


def test_report_uses_exact_seven_values_and_excludes_v1(tmp_path):
    values = [float(seed) for seed in report.SEEDS]
    result = report.stats(values)
    assert result["n"] == 7 and result["mean"] == pytest.approx(3.0)
    assert result["sample_sd"] > 0
    with pytest.raises(ValueError, match="seven values"):
        report.stats(values[:-1])
    assert report.ALGORITHMS == ("current", "residual_projected_dynamic_fusion")
    with pytest.raises(FileNotFoundError, match="required current packet missing"):
        report.load_packet("p0", "current", 0, tmp_path, tmp_path)


def test_gain_aggregation_keeps_seven_seeds_and_is_direction_aware():
    assert report.direction_aware_gain(1.2, 3.0, higher_is_better=False) == pytest.approx(1.8)
    assert report.direction_aware_gain(0.8, 0.7, higher_is_better=True) == pytest.approx(0.1)
    rows = [{"head_seed": seed, "direction_aware_gain": seed / 10, "metric": "mean_distance_error",
             "v2_exit": 2 + 2 * (seed % 4), "reference_exit": 2 + 2 * ((seed + 1) % 4)}
            for seed in report.SEEDS]
    summary = report.summarize_gains({("p0", "nrpos-outdoor", "selected_vs_current"): rows})
    assert len(summary) == 1 and summary[0]["n"] == 7
    assert summary[0]["mean"] == pytest.approx(0.3)
    assert len(json.loads(summary[0]["per_seed"])) == 7
    assert len(json.loads(summary[0]["v2_exit_by_seed"])) == 7
    assert len(json.loads(summary[0]["reference_exit_by_seed"])) == 7


def test_six_task_figures_use_solid_mean_lines_without_sd(monkeypatch, tmp_path):
    from matplotlib.axes import Axes

    summary = [
        {"method": method, "algorithm": algorithm, "task": task, "exit": layer,
         "mean": 0.5, "sample_sd": 0.1}
        for method in report.METHODS for algorithm in report.ALGORITHMS
        for task in report.TASKS for layer in report.EXITS
    ]
    errorbar_calls = []
    original_errorbar = Axes.errorbar

    def track_errorbar(self, *args, **kwargs):
        errorbar_calls.append((args, kwargs))
        return original_errorbar(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", track_errorbar)
    figures = report.plot_figures(summary, tmp_path)
    assert len(errorbar_calls) == 48  # 4 methods × 6 tasks × 2 algorithms; composites have no SD bars.
    assert {name for name in figures if name.endswith(".png")} == {
        f"{method}_{task}_exit_performance.png" for method in report.METHODS for task in report.TASKS
    } | {"algorithm_layerwise_performance.png", "p0_algorithm_layerwise_performance.png",
         "p1_algorithm_layerwise_performance.png", "p2_reverse_algorithm_layerwise_performance.png"}


def test_report_dependency_lists_all_twelve_job_ids(monkeypatch, tmp_path):
    registry = {"passed": True, "prepared_jobs": [
        {"kind": "v2_train", "method": method, "head_seed": seed, "script": f"{method}-{seed}"}
        for method in jobs.PILOT_METHODS for seed in jobs.MISSING_SEEDS
    ], "report_script": "report.sbatch"}
    calls = []

    class Response:
        def __init__(self, value): self.stdout = value

    def fake_run(command, **_kwargs):
        calls.append(command)
        return Response(f"{700 + len(calls)};cluster\n")

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)
    result = jobs.submit(registry, tmp_path)
    assert len(result["training_jobs"]) == 12
    assert len(calls) == 13
    assert calls[-1][2].startswith("--dependency=afterok:")
    assert len(calls[-1][2].split(":", 1)[1].split(":")) == 12


@pytest.mark.parametrize("method", additional_jobs.METHODS)
def test_additional_method_job_script_is_seed_isolated_and_checkpoint_locked(method, tmp_path):
    script = additional_jobs.job_script(method, 0, "gpu-l40", tmp_path)
    assert f"--method {method}" in script
    assert "--head-seed 0" in script and "--split-seed 2025" in script
    assert f"--checkpoint {report.CHECKPOINT_PATHS[method]}" in script
    assert "--epochs 200" in script and "--batch-size 32" in script
    assert "--learning-rate 0.001" in script and "--weight-decay 0.05" in script
    assert "--head-early-stopping-patience 20" in script
    assert "#SBATCH --partition=gpu-l40" in script and "#SBATCH --gres=gpu:l40:1" in script
    assert "#SBATCH --cpus-per-task=8" in script and "#SBATCH --mem=32G" in script


def test_p1_and_reverse_encoder_configs_are_method_correct(tmp_path):
    config_root = tmp_path / "configs"
    assert fusion_runner.config_path_for_method("p1", config_root) == config_root / "p1.yml"
    assert fusion_runner.config_path_for_method("p2_reverse", config_root) == config_root / "p2.yml"
    with pytest.raises(ValueError, match="unsupported Fusion V2 method"):
        fusion_runner.config_path_for_method("p3", config_root)


def test_additional_method_audit_covers_all_baseline_sources():
    result = additional_jobs.audit()
    assert result["passed"], result["failures"]
    assert len(result["inputs"]) == 14
    assert {(row["method"], row["head_seed"]) for row in result["inputs"]} == {
        (method, seed) for method in additional_jobs.METHODS for seed in report.SEEDS
    }


def test_additional_report_dependency_lists_all_fourteen_jobs(monkeypatch, tmp_path):
    registry = {"passed": True, "prepared_jobs": [
        {"kind": "v2_train", "method": method, "head_seed": seed, "script": f"{method}-{seed}"}
        for method in additional_jobs.METHODS for seed in report.SEEDS
    ], "report_script": "report.sbatch"}
    calls = []

    class Response:
        def __init__(self, value): self.stdout = value

    def fake_run(command, **_kwargs):
        calls.append(command)
        return Response(f"{800 + len(calls)};cluster\n")

    monkeypatch.setattr(additional_jobs.subprocess, "run", fake_run)
    monkeypatch.setattr(additional_jobs, "SUBMISSION_PATH", tmp_path / "submissions.json")
    result = additional_jobs.submit(registry)
    assert len(result["training_jobs"]) == 14
    assert len(calls) == 15
    assert calls[-1][2].startswith("--dependency=afterok:")
    assert len(calls[-1][2].split(":", 1)[1].split(":")) == 14


def test_multimethod_report_aggregates_all_four_methods_and_publishes_complete_figures(monkeypatch, tmp_path):
    packets = {}
    split_hash = "split-hash"
    checkpoint_hashes = {method: report.CHECKPOINT_HASHES[method] for method in report.METHODS}
    for method in report.METHODS:
        for algorithm in report.ALGORITHMS:
            for seed in report.SEEDS:
                path = tmp_path / f"{method}-{algorithm}-{seed}.json"
                path.write_text("{}\n")
                tasks = {}
                for task in report.TASKS:
                    metric = "mean_distance_error" if task == "nrpos-outdoor" else "macro_accuracy"
                    exits = {}
                    for layer in report.EXITS:
                        value = 3.0 - seed * 0.01 - layer * 0.001 if task == "nrpos-outdoor" else 0.5 + seed * 0.01 + layer * 0.001
                        entry = {"validation": {metric: value}, "test": {metric: value}, "head_path": f"{task}-L{layer}"}
                        if algorithm == "residual_projected_dynamic_fusion":
                            entry["fusion_diagnostics"] = {str(source): {field: 0.5 for field in (
                                "gate_mean", "gate_median", "gate_p10", "gate_p90", "gate_sd",
                                "relative_residual_norm_mean", "residual_norm_mean")}
                                for source in report.EXITS if source < layer}
                        exits[str(layer)] = entry
                    tasks[task] = {"spec": {"metric_name": metric, "higher_is_better": task != "nrpos-outdoor"},
                                   "selected_exit_best_validation": 4, "exits": exits}
                packets[method, algorithm, seed] = (path, {"tasks": tasks, "_split_sha256": split_hash,
                    "checkpoint": str(report.CHECKPOINT_PATHS[method]),
                    "_checkpoint_sha256": checkpoint_hashes[method]})

    monkeypatch.setattr(report, "load_packet", lambda method, algorithm, seed, *_args: packets[method, algorithm, seed])
    expected_figures = [f"{method}_{task}_exit_performance.{suffix}"
                        for method in report.METHODS for task in report.TASKS for suffix in ("png", "pdf")]
    expected_figures.extend(f"{stem}_algorithm_layerwise_performance.{suffix}"
                            for stem in ("p0", "p1", "p2_reverse") for suffix in ("png", "pdf"))
    expected_figures.extend(f"algorithm_layerwise_performance.{suffix}" for suffix in ("png", "pdf"))
    monkeypatch.setattr(report, "plot_figures", lambda _summary, _output: expected_figures)
    result = report.build_report(output=tmp_path / "report")
    assert result["sources"] == 56
    assert result["fixed_rows"] == 1344
    assert result["selected_rows"] == 336
    assert result["gain_rows"] == 840
    assert result["figures"] == 56
    manifest = json.loads((tmp_path / "report" / "manifest.json").read_text())
    assert tuple(manifest["methods"]) == ("p0", "p1", "p2", "p2_reverse")
    assert len(manifest["sources"]) == 56
