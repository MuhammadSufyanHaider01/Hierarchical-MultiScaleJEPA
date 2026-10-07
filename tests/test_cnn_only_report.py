import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import phase5_primary_report as report  # noqa: E402


class CNNOnlyReportTest(unittest.TestCase):
    def _split_manifest(self, path: Path, *, marker: int = 0) -> None:
        path.write_text(json.dumps({
            "seed": 2025,
            "marker": marker,
            "tasks": {task: {"spec": {"path": f"host-specific/{task}.h5"}} for task in report.TASK_ORDER[:6]},
        }))

    def _packet(self, root: Path, method: str, manifest: Path) -> None:
        seed_root = root / method / "seed_000"
        checkpoint = seed_root / "checkpoints" / "best.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"{method}-checkpoint".encode())
        (seed_root / "experiment.json").write_text(json.dumps({"metadata": {"phase5": {"method": method}}}))
        tasks = {}
        entries = []
        for task in report.TASK_ORDER[:6]:
            metric, _, _ = report.TASK_METRICS[task]
            exits = {}
            for layer in report.EXIT_LAYERS:
                value = 0.70 + layer / 1000 if task != "nrpos-outdoor" else 2.0 - layer / 100
                exits[str(layer)] = {
                    "test": {metric: value},
                    "validation": {metric: value},
                    "best_epoch": 3,
                    "epochs_completed": 4,
                    "stopped_early": False,
                }
                entries.append({
                    "task": task,
                    "exit_layer": layer,
                    "flops": layer * 1000,
                    "sample_shape": [1, 1, 224, 224],
                    "latency": {"mean_ms": float(layer), "median_ms": float(layer), "p95_ms": float(layer + 0.1), "standard_deviation_ms": 0.01},
                })
            chosen_layer = 8
            selected = exits[str(chosen_layer)]["test"]
            tasks[task] = {
                "spec": {"metric_name": metric},
                "exits": exits,
                "selected_exit_best_validation": chosen_layer,
                "selected_test_best_validation": selected,
                "selected_exit_earliest_within_tolerance": 2,
                "selected_test_earliest_within_tolerance": exits["2"]["test"],
            }
        downstream = seed_root / "downstream"
        downstream.mkdir()
        (downstream / "exit_metrics.json").write_text(json.dumps({
            "method": method,
            "pretrain_seed": 0,
            "dataset_suite": "primary",
            "checkpoint": str(checkpoint),
            "split_manifest": str(manifest),
            "protocol": {"test_evaluation": True},
            "tasks": tasks,
        }))
        (seed_root / "benchmarks.json").write_text(json.dumps({"entries": entries}))

    def test_cnn_only_packet_ignores_stale_methods_and_omits_vit_artifacts(self):
        original_tasks, original_methods = report.TASKS, report.METHODS
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "runs"
                manifest = Path(directory) / "split.json"
                self._split_manifest(manifest)
                self._packet(root, "cnn_p0", manifest)
                self._packet(root, "cnn_p2", manifest)
                stale = root / "cnn_p1" / "seed_000" / "downstream"
                stale.mkdir(parents=True)
                (stale / "exit_metrics.json").write_text('{"intentionally": "ignored"}')
                output = Path(directory) / "report"
                output.mkdir()
                args = argparse.Namespace(
                    cnn_methods=["cnn_p0", "cnn_p2"],
                    masking_label="M-block / span-time, N_m=6",
                    report_title="Synthetic CNN-only packet",
                    provenance_note="test fixture",
                )
                report._run_cnn_only_report(args, root, output)
                self.assertTrue((output / "README.md").is_file())
                self.assertTrue((output / "table_all_methods_downstream.csv").is_file())
                self.assertTrue((output / "early_exit_performance.png").is_file())
                self.assertFalse((output / "table_b0_context_l8_reference.md").exists())
                readme = (output / "README.md").read_text()
                self.assertIn("CNN-P0", readme)
                self.assertIn("CNN-P2", readme)
                self.assertNotIn("Original LatentWave", readme)
        finally:
            report.TASKS, report.METHODS = original_tasks, original_methods

    def test_incompatible_split_packets_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            first, second = directory / "first.json", directory / "second.json"
            self._split_manifest(first, marker=0)
            self._split_manifest(second, marker=1)
            packets = {
                "cnn_p0": {"tasks": {task: {} for task in report.TASK_ORDER[:6]}, "split_manifest": str(first)},
                "cnn_p2": {"tasks": {task: {} for task in report.TASK_ORDER[:6]}, "split_manifest": str(second)},
            }
            with self.assertRaisesRegex(ValueError, "different persisted split assignments"):
                report._discover_tasks_from_packets(packets)


if __name__ == "__main__":
    unittest.main()
