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
    LAYERS, TEMPORAL_LEVELS, fit_pca, load_split_and_labels,
    paired_bootstrap, temporal_token_indices, transform_pca, trajectory_metrics,
    validate_checkpoint_metadata, validate_source_metrics,
)


def _metadata(method: str) -> dict:
    return {
        "seed": 0,
        "phase5": {"method": method, "seed": 0},
        "config": {
            "masks": {"type": "multi_block", "multi_block": {
                "aspect_ratio_type": "span_time", "num_pred_masks": 6,
            }},
            "hierarchy": {
                "enabled": method == "p2",
                "mode": "mask_relative_fine_to_coarse" if method == "p2" else "same_scale",
                "layers": [2, 4, 6],
            },
        },
    }


class RepresentationTrajectoryTests(unittest.TestCase):
    def test_checkpoint_validation_checks_method_and_mask_metadata(self):
        for method in ("p0", "p2"):
            result = validate_checkpoint_metadata(_metadata(method), method=method)
            self.assertEqual(result["global_step"], 13560)
        with self.assertRaisesRegex(ValueError, "span_time"):
            metadata = _metadata("p0")
            metadata["config"]["masks"]["multi_block"]["aspect_ratio_type"] = "random"
            validate_checkpoint_metadata(metadata, method="p0")
        with self.assertRaisesRegex(ValueError, "hierarchy"):
            validate_checkpoint_metadata(_metadata("p0"), method="p2")

    def test_source_metrics_validate_task_method_seed_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "splits.json"
            manifest.write_text("{}")
            metrics = root / "runs" / "p0" / "exit_metrics.json"
            metrics.parent.mkdir(parents=True)
            metrics.write_text(json.dumps({
                "method": "p0", "pretrain_seed": 0,
                "split_manifest": str(manifest), "tasks": {"interference": {}},
            }))
            self.assertEqual(validate_source_metrics(metrics, method="p0", manifest_path=manifest)["method"], "p0")
            with self.assertRaisesRegex(ValueError, "method"):
                validate_source_metrics(metrics, method="p2", manifest_path=manifest)

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
        rows = trajectory_metrics({"p0": features, "p2": features.copy()}, labels)
        full = [row for row in rows if row["method"] == "p0" and row["layer"] == 2 and row["temporal_columns"] == 14][0]
        self.assertAlmostEqual(full["normalized_distance"], 0.0)
        self.assertAlmostEqual(full["cosine_similarity"], 1.0)
        self.assertGreater(full["class_separation_ratio"], 1.0)
        collapsed = np.ones_like(features)
        collapsed_rows = trajectory_metrics({"p0": collapsed, "p2": collapsed}, labels)
        collapsed_full = [row for row in collapsed_rows if row["method"] == "p0" and row["layer"] == 2 and row["temporal_columns"] == 14][0]
        self.assertAlmostEqual(collapsed_full["class_separation_ratio"], 0.0)

    def test_bootstrap_is_deterministic_and_returns_paired_intervals(self):
        rng = np.random.default_rng(3)
        labels = np.array([0] * 8 + [1] * 8)
        p0 = rng.normal(size=(16, 4, 4, 256)).astype(np.float32)
        p2 = p0 + .1
        first = paired_bootstrap({"p0": p0, "p2": p2}, labels, replicates=5, seed=7)
        second = paired_bootstrap({"p0": p0, "p2": p2}, labels, replicates=5, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(LAYERS) * len(TEMPORAL_LEVELS) * 7)


if __name__ == "__main__":
    unittest.main()
