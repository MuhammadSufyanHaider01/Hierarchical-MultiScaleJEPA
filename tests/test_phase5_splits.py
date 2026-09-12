from __future__ import annotations

import unittest

from phase5 import SPLIT_FRACTIONS, split_indices


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


if __name__ == "__main__":
    unittest.main()
