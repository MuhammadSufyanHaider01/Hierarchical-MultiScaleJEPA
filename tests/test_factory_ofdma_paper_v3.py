from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from factory_ofdma_paper_v3_exits import classification_anchor_gate


class ClassificationAnchorGateTests(unittest.TestCase):
    def test_gate_uses_macro_accuracy_not_overall_accuracy(self) -> None:
        gate = classification_anchor_gate(2, {
            "accuracy": {"lower": 0.9},
            "macro_accuracy": {"lower": 0.49},
        })
        self.assertFalse(gate["passes_chance_gate"])
        self.assertEqual(gate["macro_accuracy_ci"]["lower"], 0.49)

    def test_macro_accuracy_lower_bound_above_chance_passes(self) -> None:
        gate = classification_anchor_gate(6, {
            "accuracy": {"lower": 0.99},
            "macro_accuracy": {"lower": 0.18},
        })
        self.assertTrue(gate["passes_chance_gate"])
