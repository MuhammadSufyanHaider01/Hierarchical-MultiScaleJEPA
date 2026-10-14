from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]

import residual_fusion_v2_multiseed_jobs as jobs  # noqa: E402
import residual_fusion_v2_multiseed_report as report  # noqa: E402


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


def test_report_dependency_lists_all_twelve_job_ids(monkeypatch, tmp_path):
    registry = {"passed": True, "prepared_jobs": [
        {"kind": "v2_train", "method": method, "head_seed": seed, "script": f"{method}-{seed}"}
        for method in report.METHODS for seed in jobs.MISSING_SEEDS
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
