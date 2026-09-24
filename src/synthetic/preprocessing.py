"""Task-aware, patch-aligned availability operations."""

from __future__ import annotations

import torch


def crop_temporal_patches(
    sample: torch.Tensor,
    *,
    patch_count: int,
    patch_size: int = 16,
    temporal_axis: int,
) -> torch.Tensor:
    """Keep a leading nested time window without resizing or renormalizing."""

    if sample.ndim != 3:
        raise ValueError(f"expected [C,H,W] sample, got {tuple(sample.shape)}")
    if temporal_axis not in (1, 2):
        raise ValueError(f"temporal_axis must be 1 or 2, got {temporal_axis}")
    if patch_count < 1 or patch_size < 1:
        raise ValueError("patch_count and patch_size must be positive")
    extent_available = int(sample.shape[temporal_axis])
    extent_requested = int(patch_count) * int(patch_size)
    if extent_available % patch_size:
        raise ValueError(
            f"available temporal extent {extent_available} is not patch aligned to {patch_size}"
        )
    if extent_requested > extent_available:
        raise ValueError(
            f"requested {patch_count} temporal patches ({extent_requested} samples) "
            f"from extent {extent_available}"
        )
    slices = [slice(None)] * 3
    slices[temporal_axis] = slice(0, extent_requested)
    return sample[tuple(slices)]
