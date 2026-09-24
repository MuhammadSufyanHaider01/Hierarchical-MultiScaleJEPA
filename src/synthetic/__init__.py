"""Synthetic wireless availability data helpers."""

from .array_subsets import (
    availability_indices,
    center_out_nested_indices,
    fixed_aperture_indices,
    site_balanced_nested_indices,
)
from .dataset import SyntheticAvailabilityDataset
from .manifests import create_realization_manifest, write_manifest
from .preprocessing import crop_temporal_patches
from .schema import SyntheticTaskProfile, load_task_profile

__all__ = [
    "SyntheticAvailabilityDataset",
    "SyntheticTaskProfile",
    "create_realization_manifest",
    "crop_temporal_patches",
    "availability_indices",
    "center_out_nested_indices",
    "fixed_aperture_indices",
    "load_task_profile",
    "site_balanced_nested_indices",
    "write_manifest",
]
