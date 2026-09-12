from __future__ import annotations

import unittest

import torch

from dynamic_exits import choose_entropy_threshold, route_logits


class DynamicExitTest(unittest.TestCase):
    def test_entropy_routing_stops_at_the_earliest_confident_exit(self):
        logits = {
            2: torch.tensor([[8.0, 0.0], [0.0, 0.0]]),
            4: torch.tensor([[0.0, 8.0], [0.0, 8.0]]),
            8: torch.tensor([[0.0, 8.0], [0.0, 8.0]]),
        }
        routed, layers = route_logits(logits, threshold=0.1)
        self.assertEqual(layers.tolist(), [2, 4])
        torch.testing.assert_close(routed[0], logits[2][0])
        torch.testing.assert_close(routed[1], logits[4][1])

    def test_threshold_selection_uses_validation_accuracy_constraint(self):
        labels = torch.tensor([0, 1, 0, 1])
        logits = {
            2: torch.tensor([[6.0, 0.0], [6.0, 0.0], [6.0, 0.0], [6.0, 0.0]]),
            8: torch.tensor([[6.0, 0.0], [0.0, 6.0], [6.0, 0.0], [0.0, 6.0]]),
        }
        selected = choose_entropy_threshold(
            logits,
            labels,
            num_classes=2,
            tolerance=0.0,
            thresholds=(0.0, 1.0),
        )
        self.assertEqual(selected["threshold"], 0.0)
        self.assertEqual(selected["validation_mean_exit_layer"], 8.0)


if __name__ == "__main__":
    unittest.main()
