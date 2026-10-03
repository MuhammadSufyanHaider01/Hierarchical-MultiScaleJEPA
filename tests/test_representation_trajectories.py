from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from models import Encoder
from representation_trajectories import (
    CHECKPOINT_HORIZONS, COMPARISONS, HIERARCHY, LAYERS, METHODS, TEMPORAL_LEVELS, fit_pca, load_split_and_labels,
    paired_bootstrap, temporal_token_indices, transform_pca, trajectory_metrics,
    validate_checkpoint_metadata, validate_source_metrics,
)


def _metadata(method: str) -> dict:
    return {
        "seed": 0,
        "phase5": {"method": method, "seed": 0, "epoch": CHECKPOINT_HORIZONS[method][0],
                   "max_epochs": 50 if method == "p2_reverse" else 60},
        "config": {
            "masks": {"type": "multi_block", "multi_block": {
                "allow_overlap": False, "aspect_ratio_type": "span_time", "aspect_ratio": [0.75, 1.5],
                "enc_mask_scale": [1.0, 1.0], "min_keep": 10, "num_enc_masks": 1,
                "num_pred_masks": 6, "pred_mask_scale": [0.15, 0.2],
            }},
            "hierarchy": {
                "enabled": HIERARCHY[method][0],
                "mode": HIERARCHY[method][1] or "same_scale",
                "layers": [2, 4, 6],
            },
        },
    }


class RepresentationTrajectoryTests(unittest.TestCase):
    def test_checkpoint_validation_checks_method_and_mask_metadata(self):
        for method in METHODS:
            result = validate_checkpoint_metadata(_metadata(method), method=method)
            self.assertEqual(result["global_step"], CHECKPOINT_HORIZONS[method][1])
        with self.assertRaisesRegex(ValueError, "span_time"):
            metadata = _metadata("p0")
            metadata["config"]["masks"]["multi_block"]["aspect_ratio_type"] = "random"
            validate_checkpoint_metadata(metadata, method="p0")
        with self.assertRaisesRegex(ValueError, "hierarchy"):
            metadata = _metadata("p2_reverse")
            metadata["config"]["hierarchy"]["mode"] = "mask_relative_fine_to_coarse"
            validate_checkpoint_metadata(metadata, method="p2_reverse")
        with self.assertRaisesRegex(ValueError, "mask signature"):
            metadata = _metadata("p1")
            metadata["config"]["masks"]["multi_block"]["pred_mask_scale"] = [0.2, 0.3]
            validate_checkpoint_metadata(metadata, method="p1")
        with self.assertRaisesRegex(ValueError, "hierarchy"):
            validate_checkpoint_metadata(_metadata("p0"), method="p2")

    def test_source_metrics_validate_task_method_seed_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "splits.json"
            split_payload = {"schema_version": 3, "seed": 2025, "tasks": {"interference": {
                "indices": {"validation": [0, 1], "test": [2, 3], "train": [4, 5]}}}}
            manifest.write_text(json.dumps(split_payload))
            source_manifest = root / "different-location" / "source-splits.json"
            source_manifest.parent.mkdir()
            source_manifest.write_text(json.dumps(split_payload))
            metrics = root / "runs" / "p0" / "exit_metrics.json"
            metrics.parent.mkdir(parents=True)
            metrics.write_text(json.dumps({
                "method": "p0", "pretrain_seed": 0,
                "split_manifest": str(source_manifest), "tasks": {"interference": {}},
            }))
            self.assertEqual(validate_source_metrics(metrics, method="p0", manifest_path=manifest)["method"], "p0")
            with self.assertRaisesRegex(ValueError, "method"):
                validate_source_metrics(metrics, method="p2", manifest_path=manifest)
            reverse_metrics = root / "p2_reverse_metrics.json"
            reverse_metrics.write_text(json.dumps({
                "method": "p2_reverse", "pretrain_seed": 0,
                "split_manifest": str(root / "missing-split.json"), "tasks": {"interference": {}},
            }))
            with self.assertRaisesRegex(ValueError, "manifest is unavailable"):
                validate_source_metrics(reverse_metrics, method="p2_reverse", manifest_path=manifest)
            accepted = validate_source_metrics(reverse_metrics, method="p2_reverse", manifest_path=manifest,
                                               allow_unverifiable_manifest=True)
            self.assertEqual(accepted["_split_verification"]["status"], "unverifiable")

    def test_temporal_token_indices_keep_prefix_columns_on_fixed_grid(self):
        expected_counts = (28, 56, 112, 196)
        for columns, count in zip(TEMPORAL_LEVELS, expected_counts):
            indices = temporal_token_indices(columns)
            self.assertEqual(indices.numel(), count)
            self.assertEqual(indices.unique().numel(), count)
            self.assertTrue(torch.equal(indices[:columns], torch.arange(columns)))
        self.assertTrue(torch.equal(temporal_token_indices(14), torch.arange(196)))

    def test_full_grid_mask_is_equivalent_to_unmasked_encoder(self):
        torch.manual_seed(11)
        encoder = Encoder(img_size=(224, 224), patch_size=16, embed_dim=256, depth=8, num_heads=8).eval()
        sample = torch.randn(1, 1, 224, 224)
        full_grid = temporal_token_indices(14).unsqueeze(0)
        with torch.inference_mode():
            reference = encoder(sample, return_layers=LAYERS)
            selected = encoder(sample, masks=[full_grid], return_layers=LAYERS)
        for layer in LAYERS:
            torch.testing.assert_close(reference[layer], selected[layer], rtol=0, atol=0)

    def test_split_indices_label_alignment_and_class_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            val = list(range(1400))
            test = list(range(1400, 2800))
            train = list(range(2800, 4200))
            path.write_text(json.dumps({
                "schema_version": 3, "seed": 2025,
                "tasks": {"interference": {"counts": {"validation": 1400, "test": 1400, "train": 1400},
                    "indices": {"validation": val, "test": test, "train": train}}},
            }))
            raw_labels = np.tile(np.array([1, 2]), 2100)
            indices, mapped = load_split_and_labels(path, raw_labels)
            self.assertEqual(mapped[indices["test"]].tolist()[:4], [0, 1, 0, 1])
            self.assertEqual(np.bincount(mapped[indices["validation"]]).tolist(), [700, 700])

    def test_validation_fit_pca_is_deterministic_and_transforms_expected_shape(self):
        rng = np.random.default_rng(2)
        values = rng.normal(size=(10, 4, 4, 256)).astype(np.float32)
        pca_a = fit_pca(values)
        pca_b = fit_pca(values)
        coords_a = transform_pca(pca_a, values)
        coords_b = transform_pca(pca_b, values)
        self.assertEqual(coords_a.shape, (10, 4, 4, 2))
        np.testing.assert_allclose(coords_a, coords_b)

    def test_full_input_metrics_and_separation_invariants(self):
        labels = np.array([0, 0, 1, 1])
        features = np.zeros((4, 4, 4, 256), dtype=np.float32)
        features[labels == 0, :, :, 0] = 1.0
        features[labels == 1, :, :, 1] = 1.0
        all_features = {method: features.copy() for method in METHODS}
        rows = trajectory_metrics(all_features, labels)
        full = [row for row in rows if row["method"] == "p0" and row["layer"] == 2 and row["temporal_columns"] == 14][0]
        self.assertAlmostEqual(full["normalized_distance"], 0.0)
        self.assertAlmostEqual(full["cosine_similarity"], 1.0)
        self.assertGreater(full["class_separation_ratio"], 1.0)
        collapsed = np.ones_like(features)
        collapsed_rows = trajectory_metrics({method: collapsed for method in METHODS}, labels)
        collapsed_full = [row for row in collapsed_rows if row["method"] == "p0" and row["layer"] == 2 and row["temporal_columns"] == 14][0]
        self.assertAlmostEqual(collapsed_full["class_separation_ratio"], 0.0)

    def test_bootstrap_is_deterministic_and_returns_paired_intervals(self):
        rng = np.random.default_rng(3)
        labels = np.array([0] * 8 + [1] * 8)
        p0 = rng.normal(size=(16, 4, 4, 256)).astype(np.float32)
        feature_map = {method: (p0 if i == 0 else rng.normal(scale=.4 + .2 * i, size=p0.shape).astype(np.float32))
                       for i, method in enumerate(METHODS)}
        first = paired_bootstrap(feature_map, labels, replicates=5, seed=7)
        second = paired_bootstrap(feature_map, labels, replicates=5, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1008)
        self.assertEqual(sum(r["record_type"] == "method" for r in first), 448)
        self.assertEqual(sum(r["record_type"] == "comparison" for r in first), 560)
        self.assertEqual({(r["method"], r["reference_method"]) for r in first if r["record_type"] == "comparison"}, set(COMPARISONS))
        method_rows = [r for r in first if r["record_type"] == "method"]
        contrast_rows = [r for r in first if r["record_type"] == "comparison"]
        self.assertTrue(any(r["class_id"] == 0 and r["metric"] == "normalized_distance" for r in method_rows))
        self.assertTrue(all(r["reference_method"] for r in contrast_rows))
        p2_contrasts = [r for r in contrast_rows if r["method"] == "p2" and r["layer"] == 2
                        and r["temporal_columns"] == 2 and r["metric"] == "class_separation_ratio"]
        self.assertEqual(len(p2_contrasts), 3)
        self.assertGreater(len({(r["difference_low"], r["difference_high"]) for r in p2_contrasts}), 1)


if __name__ == "__main__":
    unittest.main()
