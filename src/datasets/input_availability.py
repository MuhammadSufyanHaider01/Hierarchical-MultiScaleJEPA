"""Read-only downstream views for input-information early-exit studies."""

from __future__ import annotations

from typing import Literal

import torch
from torch.utils.data import Dataset


AvailabilityMode = Literal["antenna_prefix", "leading_temporal_patches"]


class InputAvailabilityDataset(Dataset):
    """Expose a fixed information subset without changing the wrapped dataset.

    ``antenna_prefix`` keeps channels ``[0:C)``.  ``leading_temporal_patches``
    keeps the leading ``T_p`` patch columns, without resizing or normalizing.
    The wrapper preserves every non-sample item returned by the base dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        mode: AvailabilityMode,
        level: int,
        patch_size: int = 16,
    ):
        self.dataset = dataset
        self.mode = mode
        self.level = int(level)
        self.patch_size = int(patch_size)
        if len(dataset) < 1:
            raise ValueError("input-availability dataset cannot wrap an empty dataset")
        if self.patch_size < 1:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        sample_shape = getattr(dataset, "sample_shape", None)
        if sample_shape is None:
            first = dataset[0]
            sample = first[0] if isinstance(first, tuple) else first
            sample_shape = tuple(sample.shape)
        self.source_sample_shape = tuple(int(value) for value in sample_shape)
        if len(self.source_sample_shape) != 3:
            raise ValueError(
                "input-availability views require samples with shape [C, H, W], "
                f"got {self.source_sample_shape}"
            )
        channels, height, width = self.source_sample_shape

        if mode == "antenna_prefix":
            if not 1 <= self.level <= channels:
                raise ValueError(f"antenna level must be in [1, {channels}], got {self.level}")
            self.sample_shape = (self.level, height, width)
        elif mode == "leading_temporal_patches":
            if width % self.patch_size:
                raise ValueError(
                    f"source width {width} is not divisible by patch_size={self.patch_size}"
                )
            maximum_columns = width // self.patch_size
            if not 1 <= self.level <= maximum_columns:
                raise ValueError(
                    f"temporal patch columns must be in [1, {maximum_columns}], got {self.level}"
                )
            self.sample_shape = (channels, height, self.level * self.patch_size)
        else:
            raise ValueError(f"unsupported input-availability mode: {mode!r}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        if not isinstance(item, tuple) or not item:
            raise TypeError("wrapped dataset items must be non-empty tuples beginning with a sample tensor")
        sample = item[0]
        if not torch.is_tensor(sample) or tuple(sample.shape) != self.source_sample_shape:
            raise ValueError(
                f"wrapped sample at index {index} must have shape {self.source_sample_shape}, "
                f"got {tuple(sample.shape) if torch.is_tensor(sample) else type(sample).__name__}"
            )
        if self.mode == "antenna_prefix":
            available = sample[:self.level]
        else:
            available = sample[..., :self.level * self.patch_size]
        return (available, *item[1:])
