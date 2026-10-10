import sys
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cnn_head_multiseed_report as report  # noqa: E402


class CNNHeadMultiseedReportTest(unittest.TestCase):
    def test_report_uses_the_existing_vit_multiseed_split_checksum(self):
        source_report = ROOT / "reports/masking_study/multi_block_n6"
        manifest = json.loads((source_report / "manifest.json").read_text())
        expected = manifest["sources"]["p0"]["0"]["split_sha256"]
        self.assertEqual(report._baseline_split_sha(source_report), expected)

    def test_requires_exactly_seven_finite_head_seed_values(self):
        mean, sd = report._mean_sd([1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(mean, 4.0)
        self.assertGreater(sd, 0.0)
        with self.assertRaises(ValueError):
            report._mean_sd([1.0] * 6)
        with self.assertRaises(ValueError):
            report._mean_sd([1.0] * 6 + [float("nan")])

    def test_cnn_aggregation_keeps_validation_exit_per_seed(self):
        packets = {}
        for seed in report.SEEDS:
            tasks = {}
            for task in report.TASKS:
                tasks[task] = {
                    "spec": {"metric_name": "macro_accuracy", "higher_is_better": True},
                    "selected_exit_best_validation": report.EXITS[seed % len(report.EXITS)],
                    "exits": {
                        str(layer): {"test": {"macro_accuracy": 0.5 + seed / 100 + layer / 1000}}
                        for layer in report.EXITS
                    },
                }
            packets[seed] = {"tasks": tasks}
        fixed, selected, exits = report._add_cnn_rows("cnn_p0", packets)
        self.assertEqual(len(fixed), len(report.TASKS) * len(report.EXITS))
        self.assertEqual(len(selected), len(report.TASKS))
        self.assertEqual(exits["rfs"], {seed: report.EXITS[seed % 4] for seed in report.SEEDS})
        self.assertEqual(selected[0]["selection_frequency"], '{"2": 2, "4": 2, "6": 2, "8": 1}')


if __name__ == "__main__":
    unittest.main()
