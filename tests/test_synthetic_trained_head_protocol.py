"""Protocol checks that do not require Sionna or model checkpoints."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from synthetic_study.ofdma import (
    NFFT, N_RX_ANT, N_SYMBOLS, CP_LEN,
    _make_jammer_grid, _make_legitimate_grids, _power_normalize_grid,
    balanced_jammer_labels, capture_to_feature,
)
from synthetic_study.protocol import (
    ANTENNA_LEVELS,
    ARRAY_SUBSET_ORDER,
    TIME_PATCH_LEVELS,
    availability_view,
    make_split_manifest,
    token_count,
)


class AvailabilityProtocolTests(unittest.TestCase):
    def setUp(self):
        self.features = np.broadcast_to(
            np.arange(16, dtype=np.float32)[:, None, None], (16, 224, 896)
        ).copy()

    def test_all_antenna_views_are_nested_and_aligned(self):
        sizes = []
        for channels in ANTENNA_LEVELS:
            view = availability_view(
                self.features, task="nrpos-outdoor", axis="antenna", level=channels
            )
            self.assertEqual(view.shape, (channels, 224, 224))
            self.assertEqual(view[:, 0, 0].tolist(), list(ARRAY_SUBSET_ORDER[:channels]))
            sizes.append(token_count(channels, 14))
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(sizes[-1], 3136)

    def test_all_time_views_crop_exact_leading_windows(self):
        for task, channels in (("nrpos-outdoor", 4), ("interference", 1), ("jamming", 1)):
            for patches in TIME_PATCH_LEVELS:
                view = availability_view(
                    self.features, task=task, axis="time", level=patches
                )
                self.assertEqual(view.shape, (channels, 224, patches * 16))
                self.assertTrue(np.array_equal(view, self.features[list(ARRAY_SUBSET_ORDER[:channels]), :, : patches * 16]))
        self.assertEqual(token_count(4, 56), 3136)

    def test_invalid_views_fail(self):
        with self.assertRaises(ValueError):
            availability_view(self.features, task="jamming", axis="antenna", level=3)
        with self.assertRaises(ValueError):
            availability_view(self.features, task="jamming", axis="time", level=57)
        with self.assertRaises(ValueError):
            availability_view(self.features[:15], task="jamming", axis="time", level=2)

    def test_split_manifest_is_disjoint_stratified_and_reproducible(self):
        ids = [f"r{i:04d}" for i in range(120)]
        labels = np.repeat(np.arange(6), 20)
        a = make_split_manifest(ids, labels, seed=20260925)
        b = make_split_manifest(ids, labels, seed=20260925)
        self.assertEqual(a, b)
        split_ids = a["realization_ids"]
        self.assertEqual(set(split_ids), {"head_train", "head_validation", "exit_selection", "test"})
        all_values = [value for values in split_ids.values() for value in values]
        self.assertEqual(len(all_values), len(set(all_values)))
        self.assertEqual(set(all_values), set(ids))
        for counts in a["class_counts"].values():
            self.assertTrue(all(int(value) > 0 for value in counts.values()))

    def test_pilot_manifest_has_no_test_members(self):
        ids = [f"r{i:03d}" for i in range(60)]
        labels = np.repeat(np.arange(6), 10)
        manifest = make_split_manifest(ids, labels, seed=4, pilot=True)
        self.assertTrue(manifest["pilot"])
        self.assertNotIn("test", manifest["realization_ids"])
        self.assertEqual(sum(manifest["counts"].values()), len(ids))


class SimulatorSignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "configs/synthetic_trained_heads/factory_ofdma_v1.yml").read_text()
        )

    def test_balanced_labels_are_reproducible_and_cover_all_modes(self):
        first = balanced_jammer_labels(120, np.random.default_rng(12))
        second = balanced_jammer_labels(120, np.random.default_rng(12))
        self.assertTrue(np.array_equal(first, second))
        counts = np.bincount(first, minlength=6)
        self.assertEqual(counts[0], 60)
        self.assertTrue(np.all(counts[1:] == 12))

    def test_waveforms_keep_symbol_frequency_axes_for_every_jammer(self):
        rng = np.random.default_rng(91)
        grids, allocations, _ = _make_legitimate_grids(n_tx=4, rng=rng, config=self.profile)
        self.assertEqual(grids.shape, (4, N_SYMBOLS, NFFT))
        self.assertEqual(allocations[0][0], 0)
        self.assertEqual(allocations[-1][1], 110)
        self.assertTrue(np.isfinite(grids).all())
        powered = _power_normalize_grid(grids, 10.0)
        for mode in range(1, 6):
            jammer = _make_jammer_grid(
                mode=mode, power_dbm=10.0, legitimate_grids=powered, rng=rng
            )
            self.assertEqual(jammer.shape, (N_SYMBOLS, NFFT))
            self.assertGreater(np.count_nonzero(jammer), 0)
            self.assertTrue(np.isfinite(jammer).all())

    def test_feature_transform_emits_fixed_finite_master_shape(self):
        n_samples = N_SYMBOLS * (NFFT + CP_LEN)
        capture = np.zeros((N_RX_ANT, n_samples), dtype=np.complex64)
        result = capture_to_feature(capture)
        self.assertEqual(result.shape, (16, 224, 896))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())

    def test_label_generator_rejects_small_unbalanced_design(self):
        with self.assertRaises(ValueError):
            balanced_jammer_labels(5, np.random.default_rng(0))


if __name__ == "__main__":
    unittest.main()
