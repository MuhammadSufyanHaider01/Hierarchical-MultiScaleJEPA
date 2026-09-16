#!/usr/bin/env python3
"""Audit adaptive-mask geometry and the four cached pretraining datasets.

This script is read-only and never constructs or trains a JEPA model.  The
adaptive protocol explicitly treats cache tensors as [channels, frequency,
time], so the audit records that convention and rejects incompatible spatial
shapes rather than silently transposing data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
os.environ.setdefault("WANDB_KEY", "offline")

from collators import get_collator  # noqa: E402
from configs import init_jepa_config  # noqa: E402
from masking_profiles import apply_mask_profile  # noqa: E402
from models.regions import build_mask_relative_region_layout  # noqa: E402


EXPECTED_SPATIAL_SHAPE = (224, 224)
AXIS_CONVENTION = {"height": "frequency", "width": "time"}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _collator(config):
    return get_collator(config.masks.type)(
        input_size=tuple(config.encoder.img_size),
        patch_size=config.encoder.patch_size,
        **getattr(config.masks, config.masks.type).model_dump(),
    )


def _assert_adaptive_geometry(config) -> dict[str, object]:
    collator = _collator(config)
    samples = torch.zeros(2, 1, 224, 224)
    patch_importance = collator._patch_importance(samples)
    candidate_scores = collator._candidate_scores(patch_importance, (2, 14))
    probabilities = collator._candidate_probabilities(candidate_scores)
    if tuple(patch_importance.shape) != (2, 14, 14):
        raise AssertionError(f"unexpected patch score shape: {tuple(patch_importance.shape)}")
    if tuple(candidate_scores.shape) != (2, 13):
        raise AssertionError(f"unexpected candidate score shape: {tuple(candidate_scores.shape)}")
    if not torch.allclose(probabilities, torch.full_like(probabilities, 1.0 / 13.0)):
        raise AssertionError("constant input did not produce a uniform candidate distribution")
    if not torch.all(probabilities > 0):
        raise AssertionError("candidate probabilities do not have full exploration support")

    high_band = torch.zeros(1, 1, 224, 224)
    high_band[:, :, 32:, :] = 1.0
    high_scores = collator._candidate_scores(collator._patch_importance(high_band), (2, 14))
    if float(high_scores[0, 1]) <= float(high_scores[0, 0]):
        raise AssertionError("high-variation frequency band did not receive a higher score")

    _, masks_enc, masks_pred = collator([torch.randn(1, 224, 224) for _ in range(2)])
    if len(masks_pred) != 6 or any(tuple(mask.shape) != (2, 28) for mask in masks_pred):
        raise AssertionError(
            "unexpected prediction mask shape: "
            f"{[tuple(mask.shape) for mask in masks_pred]}"
        )
    if len(masks_enc) != 1 or tuple(masks_enc[0].shape[:1]) != (2,):
        raise AssertionError(f"unexpected context mask shape: {[tuple(mask.shape) for mask in masks_enc]}")

    unique_counts = []
    for batch_index in range(2):
        prediction_masks = [mask[batch_index] for mask in masks_pred]
        unique_counts.append(len({tuple(mask.tolist()) for mask in prediction_masks}))
        for mask_batch, mask in zip(masks_pred, prediction_masks):
            rows = torch.div(mask, 14, rounding_mode="floor")
            columns = mask.remainder(14)
            if len(mask) != 28 or len(torch.unique(rows)) != 2:
                raise AssertionError("prediction mask is not a 2-row block")
            if int(rows.max() - rows.min()) != 1 or not torch.equal(
                torch.unique(columns), torch.arange(14)
            ):
                raise AssertionError("prediction mask is not a contiguous 2x14 block")
            layout = build_mask_relative_region_layout([mask_batch], grid_size=14)
            if tuple(layout.for_layer(2).shape[-2:]) != (28, 1):
                raise AssertionError("P2 L2 region count is not 28")
            if tuple(layout.for_layer(4).shape[-2:]) != (7, 4):
                raise AssertionError("P2 L4 region count is not 7")
            if tuple(layout.for_layer(6).shape[-2:]) != (1, 28):
                raise AssertionError("P2 L6 region count is not 1")
        for context_mask in (mask[batch_index] for mask in masks_enc):
            if torch.isin(context_mask, torch.cat(prediction_masks)).any():
                raise AssertionError("context mask overlaps a prediction mask")

    return {
        "grid": [14, 14],
        "block": [2, 14],
        "candidate_locations": 13,
        "prediction_masks": 6,
        "p2_regions_per_mask": {"L2": 28, "L4": 7, "L6": 1},
        "constant_probability_min": float(probabilities.min()),
        "constant_probability_max": float(probabilities.max()),
        "unique_selected_masks": unique_counts,
        "axis_convention": AXIS_CONVENTION,
    }


def _audit_cache(root: Path, dataset_config, sample_count: int, scan_finite: bool) -> dict[str, object]:
    path = _resolve(root, dataset_config.path)
    result: dict[str, object] = {
        "name": dataset_config.name,
        "path": str(path),
        "key": dataset_config.key,
        "axis_convention": AXIS_CONVENTION,
        "errors": [],
    }
    if not path.is_file():
        result["errors"].append("missing HDF5 file")
        result["valid"] = False
        return result

    with h5py.File(path, "r") as h5:
        if dataset_config.key not in h5:
            result["errors"].append("missing configured dataset key")
            result["valid"] = False
            return result
        dataset = h5[dataset_config.key]
        result["shape"] = list(dataset.shape)
        result["dtype"] = str(dataset.dtype)
        if len(dataset.shape) != 4 or tuple(dataset.shape[-2:]) != EXPECTED_SPATIAL_SHAPE:
            result["errors"].append(
                f"expected [N, C, 224, 224], observed {tuple(dataset.shape)}"
            )
            result["valid"] = False
            return result

        indices = sorted({0, max(0, dataset.shape[0] - 1), *range(min(sample_count, dataset.shape[0]))})
        values = np.asarray(dataset[indices], dtype=np.float32)
        finite = bool(np.isfinite(values).all())
        if not finite:
            result["errors"].append("sampled values are non-finite")
        if scan_finite:
            chunk = max(1, min(dataset.shape[0], 32))
            finite = True
            for start in range(0, dataset.shape[0], chunk):
                if not np.isfinite(dataset[start : start + chunk]).all():
                    finite = False
                    break
            if not finite:
                result["errors"].append("full dataset contains non-finite values")

        height_difference = np.abs(values[:, :, 1:, :] - values[:, :, :-1, :]).mean()
        width_difference = np.abs(values[:, :, :, 1:] - values[:, :, :, :-1]).mean()
        result.update(
            {
                "sample_indices": indices,
                "finite": finite,
                "mean_absolute_height_difference": float(height_difference),
                "mean_absolute_width_difference": float(width_difference),
                "height_axis": "frequency",
                "width_axis": "time",
            }
        )
    result["valid"] = not result["errors"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/p2.yml")
    parser.add_argument("--profile", default="adaptive_frequency_n6")
    parser.add_argument("--samples-per-cache", type=int, default=4)
    parser.add_argument("--scan-finite", action="store_true")
    args = parser.parse_args()
    if args.samples_per_cache < 1:
        raise ValueError("--samples-per-cache must be positive")

    config = init_jepa_config(str(args.config))
    profile = apply_mask_profile(config, args.profile)
    if config.masks.type != "variation_aware_multi_block":
        raise ValueError("adaptive audit requires the variation-aware mask profile")

    caches = [
        _audit_cache(args.root, dataset_config, args.samples_per_cache, args.scan_finite)
        for dataset_config in config.data.pretrain_datasets
    ]
    geometry = _assert_adaptive_geometry(config)
    report = {
        "profile": profile,
        "geometry": geometry,
        "caches": caches,
        "valid": all(cache["valid"] for cache in caches),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
