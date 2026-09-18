from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from phase5 import PRIMARY_TASKS, SPLIT_FRACTIONS, build_task_dataset, split_indices


class Phase5SplitTest(unittest.TestCase):
    def test_classification_split_is_reproducible_disjoint_and_stratified(self):
        labels = [0] * 20 + [1] * 30
        first = split_indices(len(labels), seed=101, labels=labels)
        second = split_indices(len(labels), seed=101, labels=labels)
        self.assertEqual(first, second)
        self.assertEqual(sum(len(values) for values in first.values()), len(labels))
        self.assertTrue(set(first["train"]).isdisjoint(first["validation"]))
        self.assertTrue(set(first["train"]).isdisjoint(first["test"]))
        self.assertTrue(set(first["validation"]).isdisjoint(first["test"]))
        for split in first.values():
            self.assertEqual({labels[index] for index in split}, {0, 1})

    def test_position_split_is_reproducible_and_has_expected_proportions(self):
        split = split_indices(100, seed=7)
        self.assertEqual({name: len(indices) for name, indices in split.items()}, {
            "train": 80,
            "validation": 10,
            "test": 10,
        })
        self.assertEqual(SPLIT_FRACTIONS, {"train": 0.80, "validation": 0.10, "test": 0.10})

    def test_promoted_classification_caches_use_contiguous_labels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            for name, sample_key, label_key, labels in (
                ("jamming.h5", "normalized_spectrograms", "labels", [0, 1, 2, 0]),
                ("interference.h5", "cls_spectrograms", "cls_labels", [1, 2, 1, 2]),
            ):
                with h5py.File(root.parent / name, "w") as h5:
                    h5.create_dataset(sample_key, data=np.zeros((4, 1, 8, 8), dtype=np.float32))
                    h5.create_dataset(label_key, data=np.asarray(labels, dtype=np.int64))
            for task, expected_outputs, expected_labels in (
                ("jamming", 3, {0, 1, 2}),
                ("interference", 2, {0, 1}),
            ):
                dataset, spec = build_task_dataset(task, root)
                self.assertEqual(spec.num_outputs, expected_outputs)
                self.assertEqual({int(dataset[index][1]) for index in range(len(dataset))}, expected_labels)
        self.assertEqual(len(PRIMARY_TASKS), 6)


if __name__ == "__main__":
    unittest.main()
