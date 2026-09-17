from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "pareto_knee_ablation.py"


def _artifacts(root: Path) -> tuple[dict, dict]:
    checkpoint = root / "adaptive-p2.ckpt"
    exits = {}
    for layer, validation in {2: 0.840, 4: 0.920, 6: 0.935, 8: 0.940}.items():
        record = {"validation": {"macro_accuracy": validation}}
        if layer == 4:
            record["test"] = {"macro_accuracy": 0.91}
        exits[str(layer)] = record
    exit_metrics = {
        "method": "p2",
        "pretrain_seed": 0,
        "dataset_suite": "primary",
        "checkpoint": str(checkpoint),
        "tasks": {
            "rfs": {
                "spec": {"metric_name": "macro_accuracy", "higher_is_better": True},
                "exits": exits,
                "selected_exit_best_validation": 8,
                "selected_exit_earliest_within_tolerance": 4,
            }
        },
    }
    benchmarks = {
        "method": "p2",
        "pretrain_seed": 0,
        "dataset_suite": "primary",
        "checkpoint": str(checkpoint),
        "entries": [
            {
                "task": "rfs",
                "exit_layer": layer,
                "flops": layer * 1_000_000,
                "latency": {"median_ms": float(layer), "mean_ms": float(layer) + 0.1},
            }
            for layer in (2, 4, 6, 8)
        ],
    }
    return exit_metrics, benchmarks


class ParetoKneeAblationScriptTest(unittest.TestCase):
    def _write_artifacts(self, directory: Path) -> tuple[Path, Path]:
        exit_metrics, benchmarks = _artifacts(directory)
        exit_path = directory / "exit_metrics.json"
        benchmark_path = directory / "benchmarks.json"
        exit_path.write_text(json.dumps(exit_metrics, indent=2))
        benchmark_path.write_text(json.dumps(benchmarks, indent=2))
        return exit_path, benchmark_path

    def test_cli_writes_detached_ablation_and_leaves_sources_unchanged(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            exit_path, benchmark_path = self._write_artifacts(directory)
            exit_before = exit_path.read_bytes()
            benchmark_before = benchmark_path.read_bytes()
            output = directory / "result.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--exit-metrics",
                    str(exit_path),
                    "--benchmarks",
                    str(benchmark_path),
                    "--cost",
                    "flops",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("ablation", completed.stdout)
            self.assertEqual(exit_path.read_bytes(), exit_before)
            self.assertEqual(benchmark_path.read_bytes(), benchmark_before)
            result = json.loads(output.read_text())
            task = result["tasks"]["rfs"]
            self.assertEqual(task["selected_exit_pareto_knee"], 4)
            self.assertEqual(task["selected_exit_tolerance"], 4)
            self.assertEqual(task["selected_test_pareto_knee"], {"macro_accuracy": 0.91})
            self.assertEqual(task["selected_test_tolerance"], {"macro_accuracy": 0.91})

    def test_provenance_and_benchmark_shape_fail_loudly(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            exit_path, benchmark_path = self._write_artifacts(directory)
            benchmarks = json.loads(benchmark_path.read_text())
            benchmarks["method"] = "p0"
            benchmark_path.write_text(json.dumps(benchmarks))
            failed = subprocess.run(
                [sys.executable, str(SCRIPT), "--exit-metrics", str(exit_path), "--benchmarks", str(benchmark_path)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("provenance mismatch", failed.stderr)

            benchmarks = deepcopy(_artifacts(directory)[1])
            benchmarks["entries"].pop()
            benchmark_path.write_text(json.dumps(benchmarks))
            failed = subprocess.run(
                [sys.executable, str(SCRIPT), "--exit-metrics", str(exit_path), "--benchmarks", str(benchmark_path)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("must contain exactly", failed.stderr)

            benchmarks = deepcopy(_artifacts(directory)[1])
            benchmarks["entries"].append(deepcopy(benchmarks["entries"][0]))
            benchmark_path.write_text(json.dumps(benchmarks))
            failed = subprocess.run(
                [sys.executable, str(SCRIPT), "--exit-metrics", str(exit_path), "--benchmarks", str(benchmark_path)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("duplicate benchmark", failed.stderr)


if __name__ == "__main__":
    unittest.main()
