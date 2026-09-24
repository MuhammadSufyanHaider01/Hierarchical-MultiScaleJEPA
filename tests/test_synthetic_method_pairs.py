from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_synthetic_method_pairs as pair_audit


TASK_SPECS = {
    "nrpos-outdoor": {
        "name": "nrpos-outdoor", "target_type": "position", "num_outputs": 3,
        "metric_name": "mean_distance_error", "higher_is_better": False,
        "sample_shape": [4, 224, 224], "coord_min": [0, 0, 0], "coord_max": [80, 60, 40],
    },
    "interference": {
        "name": "interference", "target_type": "classification", "num_outputs": 2,
        "metric_name": "macro_accuracy", "higher_is_better": True,
        "sample_shape": [1, 224, 224], "coord_min": None, "coord_max": None,
    },
    "jamming": {
        "name": "jamming", "target_type": "classification", "num_outputs": 6,
        "metric_name": "macro_accuracy", "higher_is_better": True,
        "sample_shape": [1, 224, 224], "coord_min": None, "coord_max": None,
    },
}


class PairedSourceAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        split_ids = {"train": [0, 1], "validation": [2], "test": [3]}
        split = {
            "schema_version": 3,
            "tasks": {
                task: {"indices": split_ids}
                for task in TASK_SPECS
            },
        }
        (self.root / "runs").mkdir()
        for method in ("p0", "p2"):
            checkpoint = self.root / "runs" / f"{method}.ckpt"
            checkpoint.write_bytes(method.encode())
            manifest = self.root / "runs" / f"{method}_split.json"
            manifest.write_text(json.dumps(split))
            tasks = {}
            for task, spec in TASK_SPECS.items():
                exits = {}
                for layer in pair_audit.LAYERS:
                    head = self.root / "runs" / f"{method}_{task}_L{layer}.pt"
                    head.write_bytes(f"{method}-{task}-{layer}".encode())
                    exits[str(layer)] = {"head_path": str(head)}
                tasks[task] = {"spec": spec, "exits": exits}
            metrics = {
                "method": method,
                "protocol": {"tolerance": 0.05},
                "checkpoint": str(checkpoint),
                "split_manifest": str(manifest),
                "tasks": tasks,
            }
            (self.root / "runs" / f"{method}_metrics.json").write_text(json.dumps(metrics))
        self.p0 = self.root / "runs" / "p0_metrics.json"
        self.p2 = self.root / "runs" / "p2_metrics.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_accepts_matching_frozen_sources(self) -> None:
        result = pair_audit.audit(self.p0, self.p2)
        self.assertEqual(result["status"], "ready_for_synthetic_adapter_and_simulator_gates")
        self.assertEqual(set(result["tasks"]), set(TASK_SPECS))
        self.assertEqual(result["tasks"]["jamming"]["frozen_layers"], [2, 4, 6, 8])

    def test_rejects_different_source_split_indices(self) -> None:
        manifest = self.root / "runs" / "p2_split.json"
        split = json.loads(manifest.read_text())
        split["tasks"]["interference"]["indices"]["test"] = [99]
        manifest.write_text(json.dumps(split))
        with self.assertRaisesRegex(ValueError, "split indices differ"):
            pair_audit.audit(self.p0, self.p2)

    def test_rejects_missing_frozen_exit_head(self) -> None:
        metrics = json.loads(self.p2.read_text())
        metrics["tasks"]["jamming"]["exits"].pop("6")
        self.p2.write_text(json.dumps(metrics))
        with self.assertRaisesRegex(ValueError, "missing jamming L6 exit"):
            pair_audit.audit(self.p0, self.p2)


if __name__ == "__main__":
    unittest.main()
