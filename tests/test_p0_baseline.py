from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from models import JEPA
from models.utils import apply_masks, repeat_interleave_batch


def tiny_config() -> SimpleNamespace:
    return SimpleNamespace(
        encoder=SimpleNamespace(img_size=(32, 32), patch_size=8, embed_dim=16, depth=2, num_heads=2),
        predictor=SimpleNamespace(embed_dim=8, depth=1, num_heads=2),
        optimization=SimpleNamespace(
            ema=[0.996, 1.0],
            epochs=4,
            warmup=1,
            start_lr=1.0e-6,
            ref_lr=1.0e-3,
            final_lr=1.0e-6,
            ref_weight_decay=0.04,
            final_weight_decay=0.04,
        ),
        data=SimpleNamespace(
            pretrain_datasets=[],
            rfs_path="missing-rfs.h5",
            pos_path="missing-pos.h5",
            beam_path="missing-beam.h5",
        ),
        seed=7,
    )


def reference_p0_forward(model: JEPA, x: torch.Tensor, masks_enc: torch.Tensor, masks_pred: torch.Tensor):
    """Independent spelling of the pinned P0 forward/loss target path."""
    channels = x.size(1)
    context = model.context_encoder(x, masks_enc)
    prediction = model.predictor(context, channels, masks_enc, masks_pred)
    with torch.no_grad():
        target = model.target_encoder(x)
        target = F.layer_norm(target, (target.size(-1),))
        batch, _, dim = target.shape
        target = target.reshape(batch, channels, -1, dim)
        target = apply_masks(target, masks_pred)
        target = repeat_interleave_batch(target, B=batch, repeat=1)
        target = target.reshape(target.size(0), -1, dim)
    return prediction, target


class P0BaselineTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = JEPA(tiny_config(), iterations_per_epoch=2, eval_fns=[lambda *_: 0.0], metrics=["smoke"])
        self.model.eval()
        self.x = torch.randn(2, 1, 32, 32)
        self.masks_enc = [torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)]
        self.masks_pred = [torch.tensor(
            [[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]],
            dtype=torch.long,
        )]

    def test_hierarchy_disabled_forward_matches_p0_reference(self):
        prediction, target, _ = self.model(self.x, self.masks_enc, self.masks_pred)
        reference_prediction, reference_target = reference_p0_forward(
            self.model, self.x, self.masks_enc, self.masks_pred
        )
        torch.testing.assert_close(prediction, reference_prediction, rtol=0.0, atol=0.0)
        torch.testing.assert_close(target, reference_target, rtol=0.0, atol=0.0)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(target).all())

    def test_target_is_frozen_and_optimizer_excludes_it(self):
        self.assertTrue(all(not parameter.requires_grad for parameter in self.model.target_encoder.parameters()))
        optimizers, _ = self.model.configure_optimizers()
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizers[0].param_groups
            for parameter in group["params"]
        }
        target_parameter_ids = {id(parameter) for parameter in self.model.target_encoder.parameters()}
        self.assertTrue(optimizer_parameter_ids.isdisjoint(target_parameter_ids))

    def test_ema_update_matches_original_formula(self):
        context_parameter = next(self.model.context_encoder.parameters())
        target_parameter = next(self.model.target_encoder.parameters())
        with torch.no_grad():
            context_parameter.fill_(1.0)
            target_parameter.zero_()
        self.model.update_target_encoder(momentum=0.25)
        self.assertTrue(torch.equal(target_parameter, torch.full_like(target_parameter, 0.75)))

    def test_target_has_no_gradient_and_checkpoint_metadata(self):
        self.model.train()
        prediction, target, _ = self.model(self.x, self.masks_enc, self.masks_pred)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.target_encoder.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.context_encoder.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.predictor.parameters()))

        checkpoint = {}
        self.model.on_save_checkpoint(checkpoint)
        self.assertEqual(checkpoint["metadata"]["source"]["commit"], "330e9ef5f6eb69e728ba43b3755036cffdd086cd")
        self.assertIn("environment", checkpoint["metadata"])
        self.assertIn("dataset_manifest", checkpoint["metadata"])


if __name__ == "__main__":
    unittest.main()
