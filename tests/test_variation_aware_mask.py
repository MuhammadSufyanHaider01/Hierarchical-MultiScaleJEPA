from __future__ import annotations

import unittest

import torch

from collators import VariationAwareMultiBlockMask
from configs import init_jepa_config
from masking_profiles import apply_mask_profile
from models.regions import build_mask_relative_region_layout


def make_collator(**overrides):
    values = {
        "input_size": (224, 224),
        "patch_size": 16,
        "aspect_ratio_type": "span_time",
        "aspect_ratio": [0.75, 1.5],
        "enc_mask_scale": [1.0, 1.0],
        "pred_mask_scale": [0.15, 0.2],
        "num_enc_masks": 1,
        "num_pred_masks": 6,
        "min_keep": 10,
        "allow_overlap": False,
        "alpha": 0.5,
        "exploration_eta": 0.25,
        "epsilon": 1.0e-6,
    }
    values.update(overrides)
    return VariationAwareMultiBlockMask(**values)


class VariationAwareMaskMathTest(unittest.TestCase):
    def test_constant_input_is_uniform(self):
        collator = make_collator()
        samples = torch.ones(2, 3, 224, 224)
        temporal, frequency = collator._variation_maps(samples)
        self.assertTrue(torch.equal(temporal, torch.zeros_like(temporal)))
        self.assertTrue(torch.equal(frequency, torch.zeros_like(frequency)))
        probabilities = collator._candidate_probabilities(torch.zeros(2, 13))
        torch.testing.assert_close(probabilities, torch.full((2, 13), 1.0 / 13.0))

    def test_temporal_variation_has_zero_boundary(self):
        collator = make_collator()
        samples = torch.zeros(1, 2, 4, 5)
        samples[..., 2:] = 3.0
        temporal, frequency = collator._variation_maps(samples)
        expected_temporal = torch.zeros(1, 4, 5)
        expected_temporal[..., 2] = 3.0
        torch.testing.assert_close(temporal, expected_temporal)
        self.assertTrue(torch.equal(frequency, torch.zeros_like(frequency)))

    def test_frequency_variation_has_zero_boundary(self):
        collator = make_collator()
        samples = torch.zeros(1, 2, 5, 4)
        samples[:, :, 2:, :] = 4.0
        temporal, frequency = collator._variation_maps(samples)
        expected_frequency = torch.zeros(1, 5, 4)
        expected_frequency[:, 2, :] = 4.0
        self.assertTrue(torch.equal(temporal, torch.zeros_like(temporal)))
        torch.testing.assert_close(frequency, expected_frequency)

    def test_patch_pooling_is_exact(self):
        collator = make_collator(alpha=1.0, exploration_eta=0.0)
        samples = torch.zeros(1, 1, 224, 224)
        samples[..., 16:] = 2.0
        patch_importance = collator._patch_importance(samples)
        expected = torch.zeros(1, 14, 14)
        expected[:, :, 1] = 2.0 / 16.0
        torch.testing.assert_close(patch_importance, expected)

    def test_high_frequency_band_receives_higher_score(self):
        collator = make_collator(alpha=0.0, exploration_eta=0.0)
        samples = torch.zeros(1, 1, 224, 224)
        samples[:, :, 32:, :] = 1.0
        patch_importance = collator._patch_importance(samples)
        scores = collator._candidate_scores(patch_importance, (2, 14))
        self.assertGreater(float(scores[0, 1]), float(scores[0, 0]))
        self.assertGreater(float(scores[0, 1]), float(scores[0, 3]))

    def test_alpha_endpoints_select_one_variation_map(self):
        samples = torch.zeros(1, 1, 224, 224)
        samples[..., 1:] = torch.arange(223, dtype=torch.float32)
        samples[:, :, 100:, :] += 1.0

        temporal_collator = make_collator(alpha=1.0)
        frequency_collator = make_collator(alpha=0.0)
        temporal, frequency = temporal_collator._variation_maps(samples)
        expected_temporal = torch.nn.functional.avg_pool2d(
            temporal.unsqueeze(1), kernel_size=16, stride=16
        ).squeeze(1)
        expected_frequency = torch.nn.functional.avg_pool2d(
            frequency.unsqueeze(1), kernel_size=16, stride=16
        ).squeeze(1)
        torch.testing.assert_close(temporal_collator._patch_importance(samples), expected_temporal)
        torch.testing.assert_close(frequency_collator._patch_importance(samples), expected_frequency)

    def test_probability_support_and_uniform_exploration(self):
        scores = torch.tensor([[0.0, 1.0, 0.0]])
        exploratory = make_collator(exploration_eta=0.25)._candidate_probabilities(scores)
        self.assertTrue(torch.all(exploratory > 0))
        torch.testing.assert_close(exploratory.sum(dim=1), torch.ones(1))

        uniform = make_collator(exploration_eta=1.0)._candidate_probabilities(scores)
        torch.testing.assert_close(uniform, torch.full((1, 3), 1.0 / 3.0))


class VariationAwareMaskContractTest(unittest.TestCase):
    @staticmethod
    def _assert_two_row_full_width(mask):
        rows = torch.div(mask, 14, rounding_mode="floor")
        columns = mask.remainder(14)
        if len(torch.unique(rows)) != 2:
            raise AssertionError(f"expected two rows, got {torch.unique(rows).tolist()}")
        if not torch.equal(torch.unique(columns), torch.arange(14)):
            raise AssertionError(f"expected all time columns, got {torch.unique(columns).tolist()}")
        if not torch.equal(torch.sort(mask).values, mask):
            raise AssertionError("prediction mask indices must be sorted")
        if int(rows.max() - rows.min()) != 1:
            raise AssertionError("prediction rows must be contiguous")

    def test_output_geometry_and_no_context_leakage(self):
        collator = make_collator()
        batch, masks_enc, masks_pred = collator([torch.randn(1, 224, 224) for _ in range(2)])
        self.assertEqual(tuple(batch.shape), (2, 1, 224, 224))
        self.assertEqual(len(masks_pred), 6)
        self.assertEqual(tuple(masks_pred[0].shape), (2, 28))
        self.assertEqual(len(masks_enc), 1)
        self.assertEqual(tuple(masks_enc[0].shape[:1]), (2,))
        for batch_index in range(2):
            prediction_masks = [mask[batch_index] for mask in masks_pred]
            for mask in prediction_masks:
                self._assert_two_row_full_width(mask)
            for context_mask in (mask[batch_index] for mask in masks_enc):
                self.assertFalse(torch.isin(context_mask, torch.cat(prediction_masks)).any())

    def test_deterministic_replay(self):
        batch = [torch.randn(1, 224, 224) for _ in range(2)]
        first = make_collator()(batch)
        second = make_collator()(batch)
        torch.testing.assert_close(first[0], second[0])
        for first_masks, second_masks in zip(first[1:], second[1:]):
            self.assertEqual(len(first_masks), len(second_masks))
            for first_mask, second_mask in zip(first_masks, second_masks):
                torch.testing.assert_close(first_mask, second_mask)

    def test_p2_region_counts_for_adaptive_masks(self):
        collator = make_collator()
        _, _, masks_pred = collator([torch.randn(1, 224, 224)])
        for prediction_mask in masks_pred:
            layout = build_mask_relative_region_layout([prediction_mask], grid_size=14)
            self.assertEqual(layout.for_layer(2).shape[-2:], (28, 1))
            self.assertEqual(layout.for_layer(4).shape[-2:], (7, 4))
            self.assertEqual(layout.for_layer(6).shape[-2:], (1, 28))

    def test_profile_resolves_to_adaptive_collator_config(self):
        config = init_jepa_config("configs/p2.yml")
        record = apply_mask_profile(config, "adaptive_frequency_n6")
        self.assertEqual(config.masks.type, "variation_aware_multi_block")
        self.assertEqual(config.masks.variation_aware_multi_block.num_pred_masks, 6)
        self.assertEqual(record["name"], "adaptive_frequency_n6")
        self.assertEqual(record["resolved"]["type"], "variation_aware_multi_block")


if __name__ == "__main__":
    unittest.main()
