from __future__ import annotations

import unittest

from models import LatentWFM


def tiny_model() -> LatentWFM:
    return LatentWFM(
        num_outputs=3,
        img_size=(32, 32),
        patch_size=8,
        embed_dim=16,
        depth=4,
        num_heads=2,
    )


class LatentWFMFreezingTest(unittest.TestCase):
    def test_default_freeze_encoder_keeps_only_head_trainable(self):
        model = tiny_model()
        model.freeze_encoder()

        frozen_prefixes = ("patch_embed.", "blocks.", "norm.", "pos_embed")
        for name, parameter in model.named_parameters():
            if name.startswith(frozen_prefixes):
                self.assertFalse(parameter.requires_grad, name)
            elif name.startswith("head."):
                self.assertTrue(parameter.requires_grad, name)

    def test_partial_block_freeze_does_not_implicitly_freeze_tokenizer(self):
        model = tiny_model()
        model.freeze_encoder(num_blocks=2)

        self.assertTrue(all(parameter.requires_grad for parameter in model.patch_embed.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.blocks[:2].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.blocks[2:].parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.norm.parameters()))


if __name__ == "__main__":
    unittest.main()
