from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from models.exit_heads import load_context_encoder_checkpoint
from models.convnext_v2_hier import ConvNeXtV2HierEncoder


class ConvNeXtV2HierEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.encoder = ConvNeXtV2HierEncoder(
            img_size=(224, 224),
            patch_size=16,
            embed_dim=16,
            stage_depths=(1, 1, 1, 1),
            stage_channels=(8, 16, 32, 64),
        )

    def test_all_exits_have_canonical_token_shapes_for_variable_channel_counts(self):
        self.encoder.eval()
        for channels in (1, 2, 4):
            images = torch.randn(1, channels, 224, 224)
            exits = self.encoder(images, return_layers=(2, 4, 6, 8))
            for layer in (2, 4, 6, 8):
                self.assertEqual(tuple(exits[layer].shape), (1, channels * 196, 16))
            self.assertTrue(torch.equal(self.encoder.forward_to_layer(images, 8), exits[8]))

    def test_context_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cnn_context.ckpt"
            expected = {key: value.clone() for key, value in self.encoder.state_dict().items()}
            torch.save(
                {"state_dict": {f"context_encoder.{key}": value for key, value in expected.items()}},
                checkpoint,
            )
            restored = ConvNeXtV2HierEncoder(
                img_size=(224, 224), patch_size=16, embed_dim=16,
                stage_depths=(1, 1, 1, 1), stage_channels=(8, 16, 32, 64),
            )
            load_context_encoder_checkpoint(restored, checkpoint)
            for key, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]), key)

    def test_masked_input_values_cannot_leak_into_context_tokens(self):
        self.encoder.eval()
        visible = torch.arange(98, dtype=torch.long).reshape(1, -1)
        mask = [visible]
        first = torch.randn(1, 1, 224, 224)
        second = first.clone()
        patches = second.reshape(1, 1, 14 * 14, 16, 16)
        hidden = torch.ones(14 * 14, dtype=torch.bool)
        hidden[visible.reshape(-1)] = False
        patches[:, :, hidden, :, :] = 0.0
        second = patches.reshape_as(second)
        first_tokens = self.encoder(first, mask)
        second_tokens = self.encoder(second, mask)
        self.assertTrue(torch.equal(first_tokens, second_tokens))

    def test_local_auxiliary_gradients_are_isolated_to_their_own_stage(self):
        self.encoder.train()
        self.encoder.gradient_checkpointing = True
        images = torch.randn(1, 1, 224, 224)
        visible = torch.arange(98, dtype=torch.long).reshape(1, -1)
        for layer, stage_index in ((2, 0), (4, 1), (6, 2)):
            self.encoder.zero_grad(set_to_none=True)
            tokens = self.encoder.forward_local_layers(images, [visible], (layer,))[layer]
            tokens.square().mean().backward()
            active_grad = sum(
                float(parameter.grad.abs().sum())
                for parameter in self.encoder.stages[stage_index].parameters()
                if parameter.grad is not None
            )
            self.assertGreater(active_grad, 0.0, f"L{layer} did not train its corresponding stage")
            for earlier_stage in range(stage_index):
                self.assertTrue(
                    all(parameter.grad is None for parameter in self.encoder.stages[earlier_stage].parameters()),
                    f"L{layer} leaked auxiliary gradients into stage {earlier_stage + 1}",
                )
            if stage_index:
                self.assertTrue(all(parameter.grad is None for parameter in self.encoder.stem.parameters()))

    def test_malformed_masks_fail_loudly(self):
        images = torch.randn(1, 1, 224, 224)
        with self.assertRaises(ValueError):
            self.encoder(images, [torch.tensor([[0, 196]], dtype=torch.long)])
        with self.assertRaises(TypeError):
            self.encoder(images, [torch.tensor([[0, 1]], dtype=torch.float32)])


if __name__ == "__main__":
    unittest.main()
