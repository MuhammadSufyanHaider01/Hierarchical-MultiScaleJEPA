import sys
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from phase5_pretrain import _split_pretrain_indices  # noqa: E402


class PretrainValidationSplitTest(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_complete(self):
        train_a, validation_a = _split_pretrain_indices(101, 0.1, 2025)
        train_b, validation_b = _split_pretrain_indices(101, 0.1, 2025)

        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(validation_a, validation_b)
        self.assertEqual(len(validation_a), 11)
        self.assertEqual(len(train_a), 90)
        self.assertEqual(set(train_a).isdisjoint(set(validation_a)), True)
        self.assertEqual(set(train_a) | set(validation_a), set(range(101)))


if __name__ == "__main__":
    unittest.main()
