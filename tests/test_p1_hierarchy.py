from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from models import JEPA


def tiny_hierarchy_config(enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        encoder=SimpleNamespace(img_size=(32, 32), patch_size=8, embed_dim=16, depth=8, num_heads=2),
        predictor=SimpleNamespace(embed_dim=8, depth=1, num_heads=2),
        optimization=SimpleNamespace(
            ema=[0.996, 1.0],
            epochs=12,
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
        hierarchy=SimpleNamespace(
            enabled=enabled,
            mode="same_scale",
            layers=[2, 4, 6],
            lambda_h=1.0,
            warmup_epochs=12,
        ),
        seed=19,
    )


class P1HierarchyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.model = JEPA(
            tiny_hierarchy_config(),
            iterations_per_epoch=2,
            eval_fns=[lambda *_: 0.0],
            metrics=["smoke"],
        )
        self.model.eval()
        self.x = torch.randn(2, 1, 32, 32)
        self.masks_enc = [torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)]
        self.masks_pred = [torch.tensor(
            [[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]],
            dtype=torch.long,
        )]

    def test_p1_outputs_match_at_fine_mask_resolution(self):
        output = self.model(self.x, self.masks_enc, self.masks_pred, return_hierarchy=True)
        _, anchor_target, _, auxiliary_predictions, auxiliary_targets = output
        self.assertEqual(tuple(anchor_target.shape), (2, 12, 16))
        self.assertEqual(set(auxiliary_predictions), {2, 4, 6})
        self.assertEqual(set(auxiliary_targets), {2, 4, 6})
        for layer in (2, 4, 6):
            self.assertEqual(tuple(auxiliary_predictions[layer].shape), (2, 12, 16))
            self.assertEqual(tuple(auxiliary_targets[layer].shape), (2, 12, 16))
            self.assertFalse(auxiliary_targets[layer].requires_grad)

    def test_auxiliary_predictors_receive_gradients_but_targets_do_not(self):
        _, anchor_target, _, auxiliary_predictions, auxiliary_targets = self.model(
            self.x, self.masks_enc, self.masks_pred, return_hierarchy=True
        )
        loss = torch.nn.functional.mse_loss(
            auxiliary_predictions[2], auxiliary_targets[2]
        ) + torch.nn.functional.mse_loss(
            auxiliary_predictions[4], auxiliary_targets[4]
        ) + torch.nn.functional.mse_loss(
            auxiliary_predictions[6], auxiliary_targets[6]
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.target_encoder.parameters()))
        for predictor in self.model.aux_predictors.values():
            self.assertTrue(any(parameter.grad is not None for parameter in predictor.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.context_encoder.parameters()))

    def test_warmup_and_optimizer_include_auxiliary_predictors(self):
        self.assertAlmostEqual(self.model.hierarchy_weight(epoch=0), 1.0 / 12.0)
        self.assertAlmostEqual(self.model.hierarchy_weight(epoch=5), 0.5)
        self.assertAlmostEqual(self.model.hierarchy_weight(epoch=12), 1.0)

        optimizers, _ = self.model.configure_optimizers()
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizers[0].param_groups
            for parameter in group["params"]
        }
        auxiliary_parameter_ids = {
            id(parameter)
            for parameter in self.model.aux_predictors.parameters()
        }
        self.assertTrue(auxiliary_parameter_ids.issubset(optimizer_parameter_ids))

    def test_p1_total_loss_averages_all_auxiliary_layers(self):
        anchor_loss = torch.tensor(2.0)
        auxiliary_losses = {
            2: torch.tensor(3.0),
            4: torch.tensor(6.0),
            6: torch.tensor(12.0),
        }

        loss = self.model._p1_total_loss(
            anchor_loss,
            auxiliary_losses,
            hierarchy_weight=0.5,
        )

        self.assertEqual(loss.item(), 5.5)

    def test_disabled_hierarchy_does_not_add_auxiliary_predictors(self):
        model = JEPA(
            tiny_hierarchy_config(enabled=False),
            iterations_per_epoch=2,
            eval_fns=[lambda *_: 0.0],
            metrics=["smoke"],
        )
        self.assertEqual(len(model.aux_predictors), 0)
        self.assertEqual(model.hierarchy_weight(epoch=10), 0.0)


if __name__ == "__main__":
    unittest.main()
