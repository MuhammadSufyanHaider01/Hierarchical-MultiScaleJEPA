from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "masking_study_available_methods_report",
    ROOT / "scripts/masking_study_available_methods_report.py",
)
REPORT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REPORT)


def reverse_packet() -> dict:
    tasks = {}
    for index, name in enumerate(REPORT.TASK_LABELS):
        metric = "mean_distance_error" if name == "nrpos-outdoor" else "macro_accuracy"
        value = float(index + 1)
        tasks[name] = {
            "spec": {"metric_name": metric, "higher_is_better": metric != "mean_distance_error"},
            "exits": {
                str(layer): {
                    "test": {metric: value},
                    "validation": {metric: value + .1},
                }
                for layer in REPORT.EXIT_LAYERS
            },
            "selected_exit_best_validation": 4,
            "selected_test_best_validation": {metric: value},
            "selected_exit_earliest_within_tolerance": 2,
            "selected_test_earliest_within_tolerance": {metric: value + .1},
        }
    return {
        "method": "p2_reverse",
        "dataset_suite": "primary",
        "protocol": {"test_evaluation": True},
        "tasks": tasks,
    }


class AvailableMethodsReportTests(unittest.TestCase):
    def test_uses_only_complete_historical_methods_and_p2_reverse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            metrics = temp / "exit_metrics.json"
            metrics.write_text(json.dumps(reverse_packet()))
            output = temp / "output"

            methods = REPORT.build(
                "time_n6",
                metrics,
                ROOT / "reports/masking_study/time_n6",
                output,
            )

            self.assertEqual(methods, ["p0", "p2", "p2_reverse"])
            with (output / "table_headline.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["method"] for row in rows}, {"p0", "p2", "p2_reverse"})
            self.assertEqual(len(rows), 3 * len(REPORT.TASK_LABELS))
            self.assertTrue(all(int(row["flops"]) > 0 for row in rows))

    def test_omits_historical_methods_when_no_complete_packet_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            metrics = temp / "exit_metrics.json"
            metrics.write_text(json.dumps(reverse_packet()))

            methods = REPORT.build("frequency_n6", metrics, temp / "empty", temp / "out")

            self.assertEqual(methods, ["p2_reverse"])
            manifest = json.loads((temp / "out/manifest.json").read_text())
            self.assertTrue(all(row["flops"] for row in REPORT.read_csv(temp / "out/table_fixed_exits.csv")))
            self.assertEqual(manifest["methods_omitted_for_missing_complete_downstream_results"], ["p0", "p2", "p1", "b1"])


if __name__ == "__main__":
    unittest.main()
