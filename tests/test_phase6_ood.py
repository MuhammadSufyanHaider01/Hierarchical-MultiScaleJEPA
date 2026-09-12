from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from phase6 import (
    audit_ood_datasets,
    build_ood_task_dataset,
    create_ood_split_manifest,
)


class Phase6OODTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> None:
        records = (
            ("jamming.h5", "normalized_spectrograms", "labels", [0, 1] * 10),
            ("interference.h5", "cls_spectrograms", "cls_labels", [1, 2] * 10),
            ("has.h5", "csi", "label", [0, 1] * 10),
        )
        for filename, sample_key, label_key, labels in records:
            with h5py.File(root / filename, "w") as h5:
                h5.create_dataset(sample_key, data=np.ones((20, 1, 4, 4), dtype=np.float32))
                h5.create_dataset(label_key, data=np.asarray(labels, dtype=np.int64))

    def test_audit_manifest_and_in_memory_label_remapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            audit = audit_ood_datasets(root)
            self.assertTrue(audit["valid"])
            self.assertEqual(audit["tasks"]["interference"]["label_mapping"], {"1": 0, "2": 1})

            dataset, spec = build_ood_task_dataset("interference", root)
            self.assertEqual(spec.num_outputs, 2)
            self.assertEqual(int(dataset[0][1]), 0)
            self.assertEqual(int(dataset[1][1]), 1)

            manifest = create_ood_split_manifest(root, seed=13)
            for task in manifest["tasks"].values():
                self.assertEqual(task["counts"], {"train": 16, "validation": 2, "test": 2})
                indices = task["indices"]
                self.assertTrue(set(indices["train"]).isdisjoint(indices["validation"]))
                self.assertTrue(set(indices["train"]).isdisjoint(indices["test"]))


if __name__ == "__main__":
    unittest.main()
