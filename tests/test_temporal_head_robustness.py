from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import ExitHead
from representation_trajectories import LAYERS, METHODS, TEMPORAL_LEVELS, temporal_token_indices
from temporal_head_robustness import (
    assert_cached_feature_parity,
    assert_full_input_logit_parity,
    build_metric_tables,
    classification_metrics,
    evaluate_temporal_prefixes,
    hierarchical_paired_bootstrap,
    sha256_file,
    validate_head_packet,
)


class ToyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("positions", torch.arange(196, dtype=torch.float32))

    def _tokens(self, samples, keep, layer):
        base = samples.mean(dim=(1, 2, 3)).view(-1, 1, 1)
        tokens = base + self.positions.view(1, -1, 1) * 1e-3
        tokens = tokens.expand(-1, -1, 256) + layer * 1e-2
        return tokens.index_select(1, keep[0]) if torch.all(keep == keep[0]) else torch.stack(
            [tokens[i].index_select(0, keep[i]) for i in range(len(samples))])

    def forward(self, samples, masks=None, return_layers=None, normalize_exits=True):
        keep = masks[0]
        return {layer: self._tokens(samples, keep, layer) for layer in return_layers}

    def forward_to_layer(self, samples, layer):
        full = torch.arange(196, device=samples.device).unsqueeze(0).expand(len(samples), -1)
        return self._tokens(samples, full, layer)


class TemporalHeadRobustnessTests(unittest.TestCase):
    def test_temporal_prefix_counts_and_full_grid(self):
        self.assertEqual([temporal_token_indices(t).numel() for t in TEMPORAL_LEVELS],
                         [28, 56, 112, 196])
        torch.testing.assert_close(temporal_token_indices(14), torch.arange(196))

    def test_classification_metrics_macro_accuracy(self):
        labels = np.asarray([0, 0, 1, 1])
        pred = np.asarray([0, 1, 1, 1])
        scores = classification_metrics(pred, labels)
        self.assertAlmostEqual(scores["accuracy"], .75)
        self.assertAlmostEqual(scores["class0_accuracy"], .5)
        self.assertAlmostEqual(scores["class1_accuracy"], 1.0)
        self.assertAlmostEqual(scores["macro_accuracy"], .75)

    def test_head_packet_validation_hash_split_seed_and_state_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "encoder.ckpt"
            checkpoint.write_bytes(b"encoder checkpoint bytes")
            split = root / "split.json"
            split.write_text('{"schema_version":3,"seed":2025}')
            sha = sha256_file(checkpoint)
            exits = {}
            for layer in LAYERS:
                head_path = root / f"L{layer}.pt"
                torch.save({"head": ExitHead(256, 2).state_dict(), "epoch": 1}, head_path)
                exits[str(layer)] = {"head_path": str(head_path), "best_epoch": 1}
            packet = root / "exit_metrics.json"
            packet.write_text(json.dumps({
                "method": "p0", "pretrain_seed": 0, "head_seed": 2025,
                "checkpoint": str(checkpoint), "split_manifest": str(split),
                "dataset_suite": "primary",
                "protocol": {"selection_uses": "validation_only", "test_evaluation": False, "head_max_epochs": 200,
                             "head_early_stopping": {"enabled": True, "patience": 20, "min_delta": 0.0},
                             "batch_limits": {"train": None, "validation": None, "test": None}},
                "tasks": {"interference": {"spec": {
                    "target_type": "classification", "num_outputs": 2}, "exits": exits}},
            }))
            paths = validate_head_packet(packet, method="p0", head_seed=2025,
                                          checkpoint_path=checkpoint, checkpoint_sha256=sha,
                                          split_manifest=split, split_sha256=sha256_file(split))
            self.assertEqual(set(paths), set(LAYERS))
            with self.assertRaisesRegex(ValueError, "head seed"):
                validate_head_packet(packet, method="p0", head_seed=2026,
                                     checkpoint_path=checkpoint, checkpoint_sha256=sha,
                                     split_manifest=split, split_sha256=sha256_file(split))
            with self.assertRaisesRegex(ValueError, "trajectory checkpoint"):
                validate_head_packet(packet, method="p0", head_seed=2025,
                                     checkpoint_path=checkpoint, checkpoint_sha256="0" * 64,
                                     split_manifest=split, split_sha256=sha256_file(split))

    def test_head_packet_rejects_wrong_output_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "encoder.ckpt"
            checkpoint.write_bytes(b"encoder")
            split = root / "split.json"
            split.write_text("{}")
            exits = {}
            for layer in LAYERS:
                path = root / f"L{layer}.pt"
                head = ExitHead(256, 3 if layer == 2 else 2)
                torch.save({"head": head.state_dict()}, path)
                exits[str(layer)] = {"head_path": str(path)}
            packet = root / "packet.json"
            packet.write_text(json.dumps({
                "method": "p0", "pretrain_seed": 0, "head_seed": 5,
                "checkpoint": str(checkpoint), "split_manifest": str(split),
                "dataset_suite": "primary",
                "protocol": {"selection_uses": "validation_only", "test_evaluation": False, "head_max_epochs": 200,
                             "head_early_stopping": {"enabled": True, "patience": 20, "min_delta": 0.0},
                             "batch_limits": {"train": None, "validation": None, "test": None}},
                "tasks": {"interference": {"spec": {"target_type": "classification", "num_outputs": 2},
                                            "exits": exits}},
            }))
            with self.assertRaisesRegex(ValueError, "LayerNorm"):
                validate_head_packet(packet, method="p0", head_seed=5, checkpoint_path=checkpoint,
                                     checkpoint_sha256=sha256_file(checkpoint), split_manifest=split,
                                     split_sha256=sha256_file(split))

    def test_eval_matches_cached_features_and_full_input_logits(self):
        torch.manual_seed(7)
        n = 4
        samples = torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1).expand(n, 1, 224, 224) / 4
        loader = DataLoader(TensorDataset(samples, torch.arange(n)), batch_size=2, shuffle=False)
        encoder = ToyEncoder().eval()
        heads = {layer: {11: ExitHead(256, 2).eval()} for layer in LAYERS}
        indices = np.arange(n, dtype=np.int64)
        cached = np.empty((n, len(TEMPORAL_LEVELS), len(LAYERS), 256), dtype=np.float32)
        with torch.inference_mode():
            for ti, columns in enumerate(TEMPORAL_LEVELS):
                keep = temporal_token_indices(columns).unsqueeze(0).expand(n, -1)
                outputs = encoder(samples, masks=[keep], return_layers=LAYERS)
                for li, layer in enumerate(LAYERS):
                    cached[:, ti, li] = outputs[layer].mean(dim=1).numpy()
        probs, preds = evaluate_temporal_prefixes(encoder, heads, loader, indices=indices,
                                                   cached_features=cached, device=torch.device("cpu"),
                                                   inference_repeats=3, parity_check=True)
        self.assertEqual(probs.shape, (1, 3, 4, 4, n, 2))
        self.assertEqual(preds.shape, (1, 3, 4, 4, n))
        np.testing.assert_array_equal(preds[:, 0], preds[:, 1])
        assert_cached_feature_parity(cached, cached.copy())
        baseline = torch.randn(4, 256)
        assert_cached_feature_parity(baseline + torch.randn_like(baseline) * 1e-4, baseline)
        with self.assertRaises(AssertionError):
            assert_cached_feature_parity(baseline + torch.randn_like(baseline) * 0.1, baseline)
        logits = heads[2][11](encoder.forward_to_layer(samples[:2], 2))
        masked = encoder(samples[:2], masks=[temporal_token_indices(14).unsqueeze(0).expand(2, -1)],
                         return_layers=[2])[2]
        assert_full_input_logit_parity(heads[2][11](masked), logits)

    def test_metric_tables_degradation_and_expected_row_counts(self):
        labels = np.asarray([0, 0, 1, 1])
        m, h, r, l, t, n = 4, 2, 2, 4, 4, 4
        predictions = np.zeros((m, h, r, l, t, n), dtype=np.int64)
        for mi in range(m):
            for hi in range(h):
                predictions[mi, hi, :, :, :, :] = np.asarray([0, 0, 1, 1])
                predictions[mi, hi, :, :, 0, 0] = 1
        probabilities = np.eye(2, dtype=np.float32)[predictions]
        raw, per_seed, aggregate = build_metric_tables(probabilities, predictions, labels, (2025, 2026))
        self.assertEqual(len(raw), 4 * 2 * 2 * 4 * 4)
        self.assertEqual(len(per_seed), 4 * 2 * 4 * 4)
        self.assertEqual(len(aggregate), 64)
        full = next(row for row in aggregate if row["method"] == "p0" and row["layer"] == 2
                    and row["temporal_columns"] == 14)
        self.assertEqual(full["mean_absolute_degradation"], 0.0)
        partial = next(row for row in aggregate if row["method"] == "p0" and row["layer"] == 2
                       and row["temporal_columns"] == 2)
        self.assertGreater(partial["mean_absolute_degradation"], 0.0)

    def test_hierarchical_bootstrap_determinism_pairing_and_full_drop(self):
        labels = np.asarray([0] * 8 + [1] * 8)
        predictions = np.empty((4, 2, 2, 4, 4, len(labels)), dtype=np.int64)
        for mi in range(4):
            predictions[mi] = labels[None, None, None, None, :]
        # At Tp2, P1 is perfect and P0 predicts the opposite class.
        predictions[0, :, :, :, 0] = 1 - labels
        first = hierarchical_paired_bootstrap(predictions, labels, (2025, 2026), replicates=25, seed=9)
        second = hierarchical_paired_bootstrap(predictions, labels, (2025, 2026), replicates=25, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 288)
        p1 = next(row for row in first if row["record_type"] == "comparison"
                  and row["method"] == "p1" and row["reference_method"] == "p0"
                  and row["layer"] == 2 and row["temporal_columns"] == 2
                  and row["metric"] == "macro_accuracy")
        self.assertGreater(p1["difference"], 0)
        full_drop = next(row for row in first if row["record_type"] == "method"
                         and row["method"] == "p1" and row["layer"] == 2
                         and row["temporal_columns"] == 14
                         and row["metric"] == "absolute_degradation")
        self.assertEqual(full_drop["estimate"], 0.0)
        self.assertEqual(full_drop["ci_low"], 0.0)
        self.assertEqual(full_drop["ci_high"], 0.0)

    def test_multiclass_accuracy_and_metric_tables(self):
        labels = np.asarray([0, 1, 2, 0, 1, 2])
        pred = np.asarray([0, 2, 2, 0, 1, 1])
        scores = classification_metrics(pred, labels, num_classes=3)
        self.assertAlmostEqual(scores["macro_accuracy"], (1.0 + 0.5 + 0.5) / 3)
        self.assertAlmostEqual(scores["class2_accuracy"], 0.5)
        m, h, r, l, t, n, k = 4, 2, 1, 4, 4, len(labels), 3
        predictions = np.broadcast_to(labels, (m, h, r, l, t, n)).copy()
        probabilities = np.eye(k, dtype=np.float32)[predictions]
        raw, per_seed, aggregate = build_metric_tables(
            probabilities, predictions, labels, (2025, 2026), num_classes=k)
        self.assertEqual(len(raw), m * h * r * l * t)
        self.assertEqual(len(per_seed), m * h * l * t)
        self.assertEqual(len(aggregate), m * l * t)
        self.assertAlmostEqual(aggregate[0]["mean_class2_accuracy"], 1.0)

    def test_multiclass_hierarchical_bootstrap_is_paired_and_deterministic(self):
        labels = np.asarray([0] * 4 + [1] * 5 + [2] * 6)
        predictions = np.broadcast_to(labels, (4, 2, 1, 4, 4, len(labels))).copy()
        first = hierarchical_paired_bootstrap(predictions, labels, (2025, 2026),
                                               replicates=12, seed=23, num_classes=3)
        second = hierarchical_paired_bootstrap(predictions, labels, (2025, 2026),
                                                replicates=12, seed=23, num_classes=3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 288)
        self.assertTrue(all(row["estimate"] == 1.0 for row in first
                            if row["metric"] == "macro_accuracy"))

    def test_multiclass_head_packet_validation_checks_task_and_output_dimension(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "encoder.ckpt"; checkpoint.write_bytes(b"encoder")
            split = root / "splits.json"; split.write_text('{"schema_version":3,"seed":2025}')
            exits = {}
            for layer in LAYERS:
                head_path = root / f"rfs_L{layer}.pt"
                torch.save({"head": ExitHead(256, 3).state_dict()}, head_path)
                exits[str(layer)] = {"head_path": str(head_path)}
            packet = root / "metrics.json"
            packet.write_text(json.dumps({
                "method": "p0", "pretrain_seed": 0, "head_seed": 2025,
                "checkpoint": str(checkpoint), "split_manifest": str(split), "dataset_suite": "primary",
                "protocol": {"selection_uses": "validation_only", "test_evaluation": False,
                             "head_max_epochs": 200,
                             "head_early_stopping": {"enabled": True, "patience": 20, "min_delta": 0.0},
                             "batch_limits": {"train": None, "validation": None, "test": None}},
                "tasks": {"rfs": {"spec": {"target_type": "classification", "num_outputs": 3},
                                     "exits": exits}},
            }))
            paths = validate_head_packet(packet, method="p0", head_seed=2025,
                                         checkpoint_path=checkpoint, checkpoint_sha256=sha256_file(checkpoint),
                                         split_manifest=split, split_sha256=sha256_file(split),
                                         task="rfs", num_classes=3)
            self.assertEqual(set(paths), set(LAYERS))
            with self.assertRaisesRegex(ValueError, "20 outputs"):
                validate_head_packet(packet, method="p0", head_seed=2025,
                                     checkpoint_path=checkpoint, checkpoint_sha256=sha256_file(checkpoint),
                                     split_manifest=split, split_sha256=sha256_file(split),
                                     task="rfs", num_classes=20)


if __name__ == "__main__":
    unittest.main()
