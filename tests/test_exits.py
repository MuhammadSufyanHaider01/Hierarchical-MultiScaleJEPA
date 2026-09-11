from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from evaluation import select_best_exit, select_earliest_within_tolerance
from models import Encoder, FrozenEncoderExit, load_target_encoder_checkpoint


class ExitTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.encoder = Encoder(
            img_size=(32, 32),
            patch_size=8,
            embed_dim=16,
            depth=8,
            num_heads=2,
        ).eval()
        self.inputs = torch.randn(2, 1, 32, 32)

    def test_exit_shapes_and_l8_default_equivalence(self):
        outputs = self.encoder(self.inputs, return_layers=[2, 4, 6, 8])
        self.assertEqual(set(outputs), {2, 4, 6, 8})
        for output in outputs.values():
            self.assertEqual(tuple(output.shape), (2, 16, 16))

        default_output = self.encoder(self.inputs)
        torch.testing.assert_close(outputs[8], default_output, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            outputs[2], self.encoder.forward_to_layer(self.inputs, 2), rtol=0.0, atol=0.0
        )

    def test_invalid_exit_layers_fail(self):
        with self.assertRaises(ValueError):
            self.encoder(self.inputs, return_layers=[0])
        with self.assertRaises(ValueError):
            self.encoder.forward_to_layer(self.inputs, 9)

    def test_only_head_receives_gradients(self):
        model = FrozenEncoderExit(self.encoder, exit_layer=4, num_outputs=3)
        logits = model(self.inputs)
        logits.square().mean().backward()

        self.assertTrue(all(not parameter.requires_grad for parameter in self.encoder.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in self.encoder.parameters()))
        self.assertTrue(all(parameter.grad is not None for parameter in model.head.parameters()))

    def test_target_encoder_checkpoint_prefix_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.ckpt"
            checkpoint = {
                "state_dict": {
                    f"target_encoder.{key}": value.detach().clone()
                    for key, value in self.encoder.state_dict().items()
                }
            }
            torch.save(checkpoint, path)
            restored = Encoder(
                img_size=(32, 32),
                patch_size=8,
                embed_dim=16,
                depth=8,
                num_heads=2,
            )
            load_target_encoder_checkpoint(restored, path)
            for expected, actual in zip(self.encoder.parameters(), restored.parameters()):
                torch.testing.assert_close(expected, actual)

    def test_validation_exit_selection(self):
        metrics = {2: 0.80, 4: 0.90, 6: 0.90, 8: 0.89}
        self.assertEqual(select_best_exit(metrics, higher_is_better=True), 4)
        self.assertEqual(
            select_earliest_within_tolerance(metrics, higher_is_better=True, tolerance=0.02), 4
        )
        position = {2: 1.20, 4: 1.00, 6: 1.04, 8: 1.01}
        self.assertEqual(select_best_exit(position, higher_is_better=False), 4)
        self.assertEqual(
            select_earliest_within_tolerance(position, higher_is_better=False, tolerance=0.05), 4
        )
        with self.assertRaises(ValueError):
            select_best_exit({2: float("nan")}, higher_is_better=True)


if __name__ == "__main__":
    unittest.main()
