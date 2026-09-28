import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from phase5_pretrain import (  # noqa: E402
    FORMAL_EPOCHS,
    _split_pretrain_indices,
    _uses_expandable_segments,
    _validate_run_arguments,
    _worker_allocation,
)


class PretrainValidationSplitTest(unittest.TestCase):
    @staticmethod
    def _arguments(**overrides):
        values = {
            "method": "p0",
            "run_kind": "formal",
            "max_epochs": FORMAL_EPOCHS,
            "batch_size": 16,
            "accumulate_grad_batches": 4,
            "num_workers": 0,
            "hdf5_worker_budget": None,
            "prefetch_factor": 2,
            "log_every_n_steps": 1,
            "progress_every_n_batches": 25,
            "epoch_metrics_only": False,
            "attention_backend": "math",
            "checkpoint_every": 5,
            "validate_every": 0,
            "limit_train_batches": 1.0,
            "limit_val_batches": 1.0,
            "validation_fraction": 0.1,
            "validation_seed": 2025,
            "early_stopping": False,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1.0e-4,
            "hierarchy_lambda": None,
            "include_l8_coarse": False,
            "device": "cpu",
            "memory_safe": False,
            "throughput_safe": False,
            "performance_safe": False,
            "precision": "32-true",
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_phase5_pilot_horizon_is_60_epochs(self):
        self.assertEqual(FORMAL_EPOCHS, 60)

    def test_split_is_deterministic_disjoint_and_complete(self):
        train_a, validation_a = _split_pretrain_indices(101, 0.1, 2025)
        train_b, validation_b = _split_pretrain_indices(101, 0.1, 2025)

        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(validation_a, validation_b)
        self.assertEqual(len(validation_a), 11)
        self.assertEqual(len(train_a), 90)
        self.assertEqual(set(train_a).isdisjoint(set(validation_a)), True)
        self.assertEqual(set(train_a) | set(validation_a), set(range(101)))

    def test_formal_protocol_requires_the_pinned_horizon_and_no_latent_early_stopping(self):
        _validate_run_arguments(self._arguments())
        with self.assertRaisesRegex(ValueError, "max-epochs"):
            _validate_run_arguments(self._arguments(max_epochs=12))
        with self.assertRaisesRegex(ValueError, "cannot use latent-loss early stopping"):
            _validate_run_arguments(self._arguments(early_stopping=True))

    def test_diagnostic_protocol_can_explicitly_enable_latent_early_stopping(self):
        _validate_run_arguments(
            self._arguments(
                run_kind="diagnostic",
                max_epochs=12,
                limit_train_batches=1,
                early_stopping=True,
            )
        )

    def test_expandable_allocator_setting_is_detected(self):
        self.assertTrue(_uses_expandable_segments("expandable_segments:True"))
        self.assertTrue(_uses_expandable_segments("max_split_size_mb:128, expandable_segments:true"))
        self.assertFalse(_uses_expandable_segments("max_split_size_mb:128"))

    def test_progress_interval_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "progress interval"):
            _validate_run_arguments(self._arguments(progress_every_n_batches=0))

    def test_hdf5_worker_budget_is_split_deterministically_across_caches(self):
        self.assertEqual(
            _worker_allocation(cache_count=4, workers_per_cache=12, worker_budget=12),
            (3, 3, 3, 3),
        )
        self.assertEqual(
            _worker_allocation(cache_count=4, workers_per_cache=12, worker_budget=10),
            (3, 3, 2, 2),
        )
        self.assertEqual(
            _worker_allocation(cache_count=4, workers_per_cache=3, worker_budget=None),
            (3, 3, 3, 3),
        )

    def test_throughput_profile_requires_matched_batch_and_workers(self):
        arguments = self._arguments(
            method="p2",
            batch_size=8,
            accumulate_grad_batches=8,
            num_workers=8,
            throughput_safe=True,
            device="cuda",
        )
        with patch("phase5_pretrain.torch.cuda.is_available", return_value=True), patch.dict(
            "phase5_pretrain.os.environ",
            {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
            clear=False,
        ):
            _validate_run_arguments(arguments)

        with self.assertRaisesRegex(ValueError, "at least 8 data-loader workers"):
            with patch("phase5_pretrain.torch.cuda.is_available", return_value=True), patch.dict(
                "phase5_pretrain.os.environ",
                {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
                clear=False,
            ):
                _validate_run_arguments(
                    self._arguments(
                        method="p2",
                        batch_size=8,
                        accumulate_grad_batches=8,
                        num_workers=4,
                        throughput_safe=True,
                        device="cuda",
                    )
                )

    def test_performance_profile_requires_bf16_and_bounded_loader_settings(self):
        arguments = self._arguments(
            method="p2",
            batch_size=16,
            accumulate_grad_batches=4,
            num_workers=3,
            hdf5_worker_budget=12,
            prefetch_factor=4,
            epoch_metrics_only=True,
            attention_backend="fused",
            performance_safe=True,
            device="cuda",
            precision="bf16-mixed",
        )
        with patch("phase5_pretrain.torch.cuda.is_available", return_value=True), patch(
            "phase5_pretrain.torch.cuda.is_bf16_supported", return_value=True
        ), patch.dict(
            "phase5_pretrain.os.environ",
            {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
            clear=False,
        ):
            _validate_run_arguments(arguments)

        legacy_arguments = self._arguments(
            method="p2",
            batch_size=16,
            accumulate_grad_batches=4,
            num_workers=12,
            hdf5_worker_budget=None,
            prefetch_factor=4,
            epoch_metrics_only=True,
            attention_backend="fused",
            performance_safe=True,
            device="cuda",
            precision="bf16-mixed",
        )
        with patch("phase5_pretrain.torch.cuda.is_available", return_value=True), patch(
            "phase5_pretrain.torch.cuda.is_bf16_supported", return_value=True
        ), patch.dict("phase5_pretrain.os.environ", {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}, clear=False):
            _validate_run_arguments(legacy_arguments)

        with self.assertRaisesRegex(ValueError, "either the balanced"):
            with patch("phase5_pretrain.torch.cuda.is_available", return_value=True), patch(
                "phase5_pretrain.torch.cuda.is_bf16_supported", return_value=True
            ), patch.dict(
                "phase5_pretrain.os.environ",
                {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
                clear=False,
            ):
                _validate_run_arguments(
                    self._arguments(
                        method="p2",
                        batch_size=16,
                        accumulate_grad_batches=4,
                        num_workers=8,
                        hdf5_worker_budget=12,
                        prefetch_factor=4,
                        epoch_metrics_only=True,
                        attention_backend="fused",
                        performance_safe=True,
                        device="cuda",
                        precision="bf16-mixed",
                    )
                )


if __name__ == "__main__":
    unittest.main()
