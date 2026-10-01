from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from models import JEPA


def tiny_cnn_config(mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        encoder=SimpleNamespace(
            img_size=(224, 224),
            patch_size=16,
            embed_dim=16,
            depth=8,
            num_heads=4,
            backbone="convnext_v2_hier",
            stage_depths=(1, 1, 1, 1),
            stage_channels=(8, 16, 32, 64),
            mask_value=0.0,
        ),
        predictor=SimpleNamespace(embed_dim=16, depth=1, num_heads=4),
        optimization=SimpleNamespace(
            ema=[0.996, 1.0],
            epochs=60,
            warmup=12,
            start_lr=1.0e-6,
            ref_lr=1.0e-3,
            final_lr=1.0e-6,
            ref_weight_decay=0.04,
            final_weight_decay=0.04,
            grad_clip=None,
        ),
        data=SimpleNamespace(
            pretrain_datasets=[],
            rfs_path="missing-rfs.h5",
            pos_path="missing-position.h5",
            beam_path="missing-beam.h5",
        ),
        hierarchy=SimpleNamespace(
            enabled=True,
            mode=mode,
            layers=[2, 4, 6],
            lambda_h=1.0,
            warmup_epochs=12,
            include_l8_coarse=False,
        ),
        seed=0,
    )


class CNNHierJEPAObjectiveTest(unittest.TestCase):
    def test_p1_p2_and_reverse_have_finite_losses_and_frozen_targets(self):
        torch.manual_seed(11)
        prediction_mask = torch.arange(28, dtype=torch.long).reshape(1, -1)
        context_mask = torch.arange(28, 196, dtype=torch.long).reshape(1, -1)
        samples = torch.randn(1, 1, 224, 224)

        for mode in (
            "same_scale",
            "mask_relative_fine_to_coarse",
            "mask_relative_coarse_to_fine",
        ):
            with self.subTest(mode=mode):
                model = JEPA(
                    tiny_cnn_config(mode),
                    iterations_per_epoch=1,
                    eval_fns=[],
                    metrics=[],
                )
                outputs = model(samples, [context_mask], [prediction_mask], return_hierarchy=True)
                anchor = F.mse_loss(outputs[0], outputs[1])
                auxiliary = [
                    F.mse_loss(outputs[3][layer], outputs[4][layer])
                    for layer in (2, 4, 6)
                ]
                total = anchor + torch.stack(auxiliary).mean()
                self.assertTrue(torch.isfinite(total))
                total.backward()
                self.assertTrue(
                    all(parameter.grad is None for parameter in model.target_encoder.parameters())
                )
                self.assertTrue(
                    any(parameter.grad is not None for parameter in model.context_encoder.parameters())
                )


if __name__ == "__main__":
    unittest.main()
