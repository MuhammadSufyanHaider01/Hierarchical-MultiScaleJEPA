from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from models import JEPA, Predictor
from models.regions import (
    build_mask_relative_region_layout,
    gather_target_regions,
    pool_target_regions,
)


def tiny_p2_config() -> SimpleNamespace:
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
            enabled=True,
            mode="mask_relative_fine_to_coarse",
            layers=[2, 4, 6],
            lambda_h=1.0,
            warmup_epochs=12,
        ),
        seed=29,
    )


class MaskRelativeRegionsTest(unittest.TestCase):
    def test_hand_computed_pooling_and_geometry(self):
        mask = torch.arange(16, dtype=torch.long).reshape(1, -1)
        layout = build_mask_relative_region_layout([mask], grid_size=4)

        self.assertEqual(tuple(layout.for_layer(2).shape), (1, 1, 16, 1))
        self.assertEqual(tuple(layout.for_layer(4).shape), (1, 1, 4, 4))
        self.assertEqual(tuple(layout.for_layer(6).shape), (1, 1, 1, 16))
        self.assertTrue(torch.equal(layout.for_layer(4)[0, 0, 0], torch.tensor([0, 1, 4, 5])))

        target = torch.stack(
            [torch.arange(16, dtype=torch.float32), torch.arange(100, 116, dtype=torch.float32)],
            dim=-1,
        ).unsqueeze(0)
        gathered = gather_target_regions(target, channels=1, layout=layout, layer=4)
        expected_members = torch.tensor(
            [[0.0, 100.0], [1.0, 101.0], [4.0, 104.0], [5.0, 105.0]]
        )
        torch.testing.assert_close(gathered[0, 0, 0, 0], expected_members)

        pooled = pool_target_regions(target, channels=1, layout=layout, layer=4)
        expected_pooled = F.layer_norm(expected_members.mean(dim=0), (2,))
        torch.testing.assert_close(pooled[0, 0, 0, 0], expected_pooled)

    def test_span_time_geometry_and_no_leakage(self):
        mask = torch.arange(28, dtype=torch.long).reshape(1, -1)
        layout = build_mask_relative_region_layout(mask, grid_size=14)
        self.assertEqual(tuple(layout.for_layer(2).shape), (1, 1, 28, 1))
        self.assertEqual(tuple(layout.for_layer(4).shape), (1, 1, 7, 4))
        self.assertEqual(tuple(layout.for_layer(6).shape), (1, 1, 1, 28))
        for layer in (2, 4, 6):
            self.assertTrue(torch.isin(layout.for_layer(layer), mask).all())

    def test_odd_geometry_uses_padded_regions_and_valid_member_counts(self):
        rows = torch.arange(7, dtype=torch.long)[:, None]
        cols = torch.arange(5, dtype=torch.long)[None, :]
        mask = (rows * 8 + cols).reshape(1, -1)
        layout = build_mask_relative_region_layout(mask, grid_size=8)

        self.assertEqual(tuple(layout.for_layer(4).shape), (1, 1, 4, 12))
        self.assertEqual(layout.sizes_for_layer(4).tolist(), [[[12, 8, 9, 6]]])
        for layer in (2, 4, 6):
            regions = layout.for_layer(layer)
            sizes = layout.sizes_for_layer(layer)
            for region_index, size in enumerate(sizes[0, 0].tolist()):
                self.assertTrue(torch.isin(regions[0, 0, region_index, :size], mask).all())

    def test_malformed_masks_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "complete rectangular"):
            build_mask_relative_region_layout([torch.tensor([[0, 1, 2, 4]])], grid_size=4)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_mask_relative_region_layout([torch.tensor([[0, 1, 4, 4]])], grid_size=4)

    def test_region_query_uses_mean_fine_position(self):
        predictor = Predictor(
            num_patches=16,
            encoder_embed_dim=4,
            output_dim=4,
            predictor_embed_dim=4,
            depth=1,
            num_heads=2,
        )
        with torch.no_grad():
            predictor.pos_embed.copy_(torch.arange(64, dtype=torch.float32).reshape(1, 1, 16, 4))
        indices = torch.tensor([[[[0, 1, 4, 5]]]], dtype=torch.long)
        positions = predictor._region_positional_embeddings(indices)
        expected = predictor.pos_embed[0, 0, torch.tensor([0, 1, 4, 5])].mean(dim=0)
        torch.testing.assert_close(positions[0, 0, 0], expected)


class P2HierarchyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.model = JEPA(
            tiny_p2_config(),
            iterations_per_epoch=2,
            eval_fns=[lambda *_: 0.0],
            metrics=["smoke"],
        )
        self.model.eval()
        self.x = torch.randn(2, 1, 32, 32)
        prediction = torch.tensor([0, 1, 4, 5, 8, 9, 12, 13], dtype=torch.long)
        visible = torch.tensor([2, 3, 6, 7, 10, 11, 14, 15], dtype=torch.long)
        self.masks_enc = [visible.repeat(2, 1)]
        self.masks_pred = [prediction.repeat(2, 1)]

    def test_p2_predictions_match_target_regions_and_target_is_frozen(self):
        output = self.model(self.x, self.masks_enc, self.masks_pred, return_hierarchy=True)
        _, anchor_target, _, predictions, targets = output

        self.assertEqual(tuple(anchor_target.shape), (2, 8, 16))
        expected_regions = {2: 8, 4: 2, 6: 1}
        for layer, regions in expected_regions.items():
            self.assertEqual(tuple(predictions[layer].shape), (1, 1, 2, 1, regions, 16))
            self.assertEqual(tuple(targets[layer].shape), (1, 1, 2, 1, regions, 16))
            self.assertFalse(targets[layer].requires_grad)

        anchor_prediction = self.model(self.x, self.masks_enc, self.masks_pred)[0]
        loss = F.mse_loss(anchor_prediction, anchor_target)
        auxiliary_losses = {
            layer: F.mse_loss(predictions[layer], targets[layer]) for layer in (2, 4, 6)
        }
        loss = self.model._hierarchy_total_loss(loss, auxiliary_losses, hierarchy_weight=1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.target_encoder.parameters()))
        for predictor in self.model.aux_predictors.values():
            self.assertTrue(any(parameter.grad is not None for parameter in predictor.parameters()))

    def test_p2_odd_mask_geometry_matches_prediction_and_target_axes(self):
        prediction = torch.tensor([0, 1, 4, 5, 8, 9], dtype=torch.long)
        visible = torch.tensor([2, 3, 6, 7, 10, 11, 12, 13, 14, 15], dtype=torch.long)
        masks_enc = [visible.repeat(2, 1)]
        masks_pred = [prediction.repeat(2, 1)]
        layout = build_mask_relative_region_layout(masks_pred, grid_size=4)

        self.assertEqual(layout.sizes_for_layer(4).tolist(), [[[2, 2, 1, 1], [2, 2, 1, 1]]])
        _, _, _, predictions, targets = self.model(
            self.x,
            masks_enc,
            masks_pred,
            return_hierarchy=True,
        )
        expected_shapes = {
            2: (1, 1, 2, 1, 6, 16),
            4: (1, 1, 2, 1, 4, 16),
            6: (1, 1, 2, 1, 1, 16),
        }
        for layer, shape in expected_shapes.items():
            self.assertEqual(tuple(predictions[layer].shape), shape)
            self.assertEqual(tuple(targets[layer].shape), shape)
            self.assertTrue(torch.isfinite(predictions[layer]).all())
            self.assertTrue(torch.isfinite(targets[layer]).all())

    def test_p2_target_taps_still_run_the_full_ema_encoder(self):
        full_depth_calls = []
        handle = self.model.target_encoder.blocks[-1].register_forward_hook(
            lambda *_: full_depth_calls.append(True)
        )
        try:
            self.model(self.x, self.masks_enc, self.masks_pred, return_hierarchy=True)
        finally:
            handle.remove()

        # One full target pass builds the unchanged P0 anchor; the second pass
        # exposes L2/L4/L6 raw taps and must still execute through L8.
        self.assertEqual(len(full_depth_calls), 2)

    def test_optional_l8_coarse_ablation_uses_full_block_regions(self):
        config = tiny_p2_config()
        config.hierarchy.include_l8_coarse = True
        model = JEPA(
            config,
            iterations_per_epoch=2,
            eval_fns=[lambda *_: 0.0],
            metrics=["smoke"],
        ).eval()
        output = model(self.x, self.masks_enc, self.masks_pred, return_hierarchy=True)
        _, anchor_target, _, predictions, targets = output
        self.assertEqual(set(predictions), {2, 4, 6, 8})
        self.assertEqual(tuple(predictions[8].shape), (1, 1, 2, 1, 1, 16))
        self.assertEqual(tuple(targets[8].shape), (1, 1, 2, 1, 1, 16))
        loss = self.model.loss(predictions[8], targets[8]) + self.model.loss(
            self.model(self.x, self.masks_enc, self.masks_pred)[0], anchor_target
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(targets[8].requires_grad)


if __name__ == "__main__":
    unittest.main()
