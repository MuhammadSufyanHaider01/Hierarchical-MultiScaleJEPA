from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

from evaluation.exits import select_earliest_within_tolerance
from models import Encoder, FrozenEncoderExit
from synthetic.array_subsets import (
    center_out_nested_indices,
    fixed_aperture_indices,
    site_balanced_nested_indices,
)
from synthetic.dataset import SyntheticAvailabilityDataset
from synthetic.generator import expected_master_shape, sionna_operational_realization
from synthetic.manifests import create_realization_manifest, validate_realization_manifest
from synthetic.preprocessing import crop_temporal_patches
from synthetic.schema import load_task_profile, require_verified_semantics
from synthetic.statistics import (
    classification_anchor_passes,
    position_anchor_passes,
)
from synthetic_availability_exits import _load_exit_models, _test


class IndexedMaster(Dataset):
    def __init__(self, axis: int = 2, shape: tuple[int, int, int] = (16, 64, 64)):
        self.axis = axis
        self.shape = shape
        self.labels = torch.tensor([3, 7])
        self.meta = [{"row": 0}, {"row": 1}]
        base = torch.arange(np.prod(shape), dtype=torch.int64).reshape(shape)
        self.samples = (base, base + 1)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return self.samples[index], self.labels[index], self.meta[index]


class NoReadMaster(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise AssertionError("view construction should not fetch an HDF5 sample")


class LargeShapeMaster(Dataset):
    def __init__(self, shape):
        self.shape = shape

    def __len__(self):
        return 1

    def __getitem__(self, index):
        sample = torch.zeros((1, 1, 1)).expand(self.shape)
        return sample, torch.tensor(0)


class SyntheticAvailabilityTests(unittest.TestCase):
    def test_fixed_aperture_subsets_are_valid_and_deterministic(self):
        self.assertEqual(fixed_aperture_indices(1, 16), (7,))
        self.assertEqual(fixed_aperture_indices(4, 16), (0, 5, 10, 15))
        self.assertEqual(fixed_aperture_indices(16, 16), tuple(range(16)))
        with self.assertRaises(ValueError):
            fixed_aperture_indices(17, 16)

        self.assertEqual(site_balanced_nested_indices(1), (0,))
        self.assertEqual(site_balanced_nested_indices(4), (0, 8, 4, 12))
        self.assertEqual(site_balanced_nested_indices(8), (0, 8, 4, 12, 1, 9, 5, 13))
        self.assertEqual(site_balanced_nested_indices(16), (0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15))
        with self.assertRaises(ValueError):
            site_balanced_nested_indices(17)

        self.assertEqual(center_out_nested_indices(1), (7,))
        self.assertEqual(center_out_nested_indices(2), (7, 8))
        self.assertEqual(center_out_nested_indices(4), (7, 8, 6, 9))
        self.assertEqual(center_out_nested_indices(16), (7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2, 13, 1, 14, 0, 15))
        with self.assertRaises(ValueError):
            center_out_nested_indices(17)

    def test_antenna_view_crops_fixed_time_and_preserves_items(self):
        base = IndexedMaster()
        view = SyntheticAvailabilityDataset(
            base,
            mode="antenna",
            level=4,
            temporal_axis=1,
            reference_channels=4,
            reference_time_patches=2,
            patch_size=16,
        )
        sample, label, meta = view[1]
        indices = fixed_aperture_indices(4, 16)
        self.assertEqual(tuple(sample.shape), (4, 32, 64))
        self.assertTrue(torch.equal(sample, base.samples[1][list(indices), :32, :]))
        self.assertEqual(label.item(), 7)
        self.assertEqual(meta, {"row": 1})

    def test_time_view_selects_fixed_channels_and_leading_window(self):
        base = IndexedMaster()
        view = SyntheticAvailabilityDataset(
            base,
            mode="time",
            level=3,
            temporal_axis=2,
            reference_channels=4,
            reference_time_patches=14,
            patch_size=16,
        )
        sample, label, meta = view[0]
        indices = fixed_aperture_indices(4, 16)
        self.assertEqual(tuple(sample.shape), (4, 64, 48))
        self.assertTrue(torch.equal(sample, base.samples[0][list(indices), :, :48]))
        self.assertEqual(label.item(), 3)
        self.assertEqual(meta, {"row": 0})

    def test_patch_crop_rejects_unaligned_or_oversized_windows(self):
        sample = torch.zeros((1, 32, 65))
        with self.assertRaises(ValueError):
            crop_temporal_patches(sample, patch_count=2, temporal_axis=2)
        with self.assertRaises(ValueError):
            crop_temporal_patches(sample[:, :, :64], patch_count=5, temporal_axis=2)
        with self.assertRaises(ValueError):
            SyntheticAvailabilityDataset(
                IndexedMaster(shape=(4, 64, 64)),
                mode="antenna",
                level=4,
                temporal_axis=2,
                reference_channels=4,
                reference_time_patches=2,
            )

    def test_shape_metadata_avoids_reading_a_sample_during_selection_setup(self):
        view = SyntheticAvailabilityDataset(
            NoReadMaster(),
            mode="time",
            level=2,
            temporal_axis=2,
            reference_channels=1,
            reference_time_patches=14,
            source_sample_shape=(16, 64, 64),
        )
        self.assertEqual(view.sample_shape, (1, 64, 32))

    def test_max_information_views_have_expected_token_counts(self):
        for task, filename, axis in (
            ("nrpos-outdoor", "factory_ofdma_nrpos_paper_v3.yml", 2),
            ("interference", "factory_ofdma_interference_paper_v3.yml", 2),
            ("jamming", "factory_ofdma_jamming_paper_v3.yml", 2),
        ):
            profile, _ = load_task_profile(
                Path(__file__).resolve().parents[1] / "configs" / "sionna" / filename
            )
            self.assertEqual(profile.temporal_axis, axis)
            self.assertEqual(expected_master_shape(profile)[0], 16)
        source = LargeShapeMaster((16, 224, 896))
        antenna = SyntheticAvailabilityDataset(
            source, mode="antenna", level=16, temporal_axis=2,
            reference_channels=1, reference_time_patches=14
        )
        time = SyntheticAvailabilityDataset(
            source, mode="time", level=56, temporal_axis=2,
            reference_channels=1, reference_time_patches=14
        )
        self.assertEqual(int(np.prod(antenna.sample_shape) // 256), 3136)
        self.assertEqual(int(np.prod(time.sample_shape) // 256), 784)

    def test_manifest_splits_are_stable_and_disjoint(self):
        ids = [f"id-{index}" for index in range(100)]
        labels = np.asarray([index % 4 for index in range(100)])
        first = create_realization_manifest(
            ids, labels, task="demo", target_type="classification", seed=11
        )
        second = create_realization_manifest(
            ids, labels, task="demo", target_type="classification", seed=11
        )
        self.assertEqual(first, second)
        validate_realization_manifest(first)
        self.assertEqual(len(first["splits"]["calibration"]), 20)
        self.assertEqual(len(first["splits"]["selection"]), 20)
        self.assertEqual(len(first["splits"]["test"]), 60)
        broken = dict(first)
        broken["splits"] = dict(first["splits"])
        broken["splits"]["test"] = first["splits"]["test"][:-1] + first["splits"]["selection"][:1]
        with self.assertRaises(ValueError):
            validate_realization_manifest(broken)

    def test_earliest_within_five_percent_for_both_metric_directions(self):
        self.assertEqual(
            select_earliest_within_tolerance(
                {2: 0.94, 4: 0.96, 8: 0.97},
                higher_is_better=True,
                tolerance=0.05,
            ),
            2,
        )
        self.assertEqual(
            select_earliest_within_tolerance(
                {2: 10.4, 4: 10.0, 8: 9.8},
                higher_is_better=False,
                tolerance=0.05,
            ),
            4,
        )
        self.assertTrue(classification_anchor_passes(0.51, 2))
        self.assertFalse(classification_anchor_passes(0.50, 2))
        self.assertTrue(position_anchor_passes(0.01))
        self.assertFalse(position_anchor_passes(0.0))

    def test_test_stage_refuses_incomplete_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            selection = Path(directory) / "selection.json"
            selection.write_text('{"stage": "selection_pending"}')
            args = SimpleNamespace(task="jamming", sweep="time")
            with self.assertRaisesRegex(ValueError, "completed validation-only"):
                _test(selection, Path(directory) / "test.json", args, torch.device("cpu"))

    def test_paper_v3_profiles_are_explicit_assumption_based_stress_tests(self):
        config_root = Path(__file__).resolve().parents[1] / "configs" / "sionna"
        profiles = (
            ("factory_ofdma_nrpos_paper_v3.yml", 3),
            ("factory_ofdma_interference_paper_v3.yml", 2),
            ("factory_ofdma_jamming_paper_v3.yml", 6),
        )
        for filename, output_count in profiles:
            profile, _ = load_task_profile(config_root / filename)
            self.assertEqual(profile.num_outputs, output_count)
            self.assertEqual(profile.specification_status, "assumption_based_synthetic_domain")
            self.assertFalse(profile.semantics_verified)
            self.assertTrue(profile.assumptions_approved)
            self.assertFalse(profile.requires_source_verified_semantics)
            self.assertTrue(profile.assumptions)
            require_verified_semantics(profile)

    def test_loaded_encoder_and_heads_are_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = Encoder(img_size=(16, 16), patch_size=16, embed_dim=8, depth=8, num_heads=2)
            paths = {}
            for layer in (2, 4, 6, 8):
                frozen = FrozenEncoderExit(encoder, exit_layer=layer, num_outputs=2)
                path = Path(directory) / f"head_l{layer}.pt"
                torch.save({"head": frozen.head.state_dict()}, path)
                paths[layer] = path
            models = _load_exit_models(encoder, paths, SimpleNamespace(num_outputs=2), torch.device("cpu"))
            for model in models.values():
                self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
                with torch.inference_mode():
                    model(torch.zeros((1, 1, 16, 16)))
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
