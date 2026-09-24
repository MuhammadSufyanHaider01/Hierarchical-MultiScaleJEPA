"""Read-only synthetic availability views for fixed master HDF5 data."""

from __future__ import annotations

from torch.utils.data import Dataset

from .array_subsets import availability_indices
from .preprocessing import crop_temporal_patches

class SyntheticAvailabilityDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        *,
        mode: str,
        level: int,
        temporal_axis: int,
        reference_channels: int,
        reference_time_patches: int,
        source_sample_shape: tuple[int, ...] | None = None,
        patch_size: int = 16,
        max_channels: int = 16,
        array_subset_policy: str = "fixed_aperture_uniform",
    ):
        if mode not in {"antenna", "time"}:
            raise ValueError(f"unsupported synthetic availability mode: {mode}")
        if len(dataset) < 1:
            raise ValueError("synthetic availability view cannot wrap an empty dataset")
        self.dataset = dataset
        self.mode = mode
        self.level = int(level)
        self.temporal_axis = int(temporal_axis)
        self.reference_channels = int(reference_channels)
        self.reference_time_patches = int(reference_time_patches)
        self.patch_size = int(patch_size)
        self.max_channels = int(max_channels)
        self.array_subset_policy = str(array_subset_policy)
        if source_sample_shape is None:
            item = dataset[0]
            sample = item[0] if isinstance(item, (tuple, list)) else item
            source_sample_shape = tuple(int(v) for v in sample.shape)
        if len(source_sample_shape) != 3:
            raise ValueError("synthetic master samples must have shape [C,H,W]")
        self.source_sample_shape = tuple(int(v) for v in source_sample_shape)
        channels, height, width = self.source_sample_shape
        if self.temporal_axis not in (1, 2):
            raise ValueError("temporal_axis must be 1 or 2")
        if mode == "antenna":
            if channels != self.max_channels:
                raise ValueError(f"antenna master requires C={self.max_channels}, got {channels}")
            self.antenna_indices = availability_indices(
                self.level, policy=self.array_subset_policy, max_channels=self.max_channels
            )
            if self.reference_time_patches < 1:
                raise ValueError("reference_time_patches must be positive")
            self.sample_shape = self._cropped_shape(self.level, self.reference_time_patches)
        else:
            if not 1 <= self.reference_channels <= self.max_channels:
                raise ValueError("reference_channels must be within the master array size")
            if height % self.patch_size or width % self.patch_size:
                raise ValueError("time master H and W must be patch aligned")
            extent = self.source_sample_shape[self.temporal_axis]
            if extent % self.patch_size or self.level < 1:
                raise ValueError("time level and source temporal extent must be patch aligned")
            if self.level * self.patch_size > extent:
                raise ValueError(f"time level {self.level} exceeds source extent {extent}")
            self.antenna_indices = availability_indices(
                self.reference_channels,
                policy=self.array_subset_policy,
                max_channels=self.max_channels,
            )
            self.sample_shape = self._cropped_shape(self.reference_channels, self.level)

    def _cropped_shape(self, channels: int, temporal_patches: int) -> tuple[int, int, int]:
        shape = [channels, *self.source_sample_shape[1:]]
        shape[self.temporal_axis] = temporal_patches * self.patch_size
        if shape[1] % self.patch_size or shape[2] % self.patch_size:
            raise ValueError("availability output must remain patch aligned")
        return tuple(shape)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        if not isinstance(item, (tuple, list)) or not item:
            raise TypeError("master dataset items must be tuples beginning with a sample")
        sample = item[0]
        if tuple(sample.shape) != self.source_sample_shape:
            raise ValueError(
                f"master sample {index} shape {tuple(sample.shape)} != {self.source_sample_shape}"
            )
        if self.mode == "antenna":
            sample = sample[list(self.antenna_indices)]
            sample = crop_temporal_patches(
                sample,
                patch_count=self.reference_time_patches,
                patch_size=self.patch_size,
                temporal_axis=self.temporal_axis,
            )
        else:
            sample = sample[list(self.antenna_indices)]
            sample = crop_temporal_patches(
                sample,
                patch_count=self.level,
                patch_size=self.patch_size,
                temporal_axis=self.temporal_axis,
            )
        return (sample, *item[1:])
