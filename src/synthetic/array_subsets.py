"""Deterministic receive-element subsets at fixed maximum array aperture."""

from __future__ import annotations

import numpy as np


def fixed_aperture_indices(channels: int, max_channels: int = 16) -> tuple[int, ...]:
    """Select a deterministic, approximately uniform subset of a fixed ULA."""

    channels = int(channels)
    max_channels = int(max_channels)
    if max_channels < 1 or not 1 <= channels <= max_channels:
        raise ValueError(f"channels must be in [1, {max_channels}], got {channels}")
    if channels == 1:
        return (max(0, max_channels // 2 - 1),)
    indices = np.rint(np.linspace(0, max_channels - 1, channels)).astype(np.int64)
    if np.unique(indices).size != channels:
        raise RuntimeError(f"fixed-aperture selection produced duplicate elements for C={channels}")
    return tuple(int(v) for v in indices)


def site_balanced_nested_indices(
    channels: int, *, sites: int = 4, elements_per_site: int = 4
) -> tuple[int, ...]:
    """Nested 4-site receive selection for the synthetic NR configuration.

    The ordering exposes one reference element at each virtual site before
    adding a second, third, then fourth element. Every supported availability
    level is therefore a fixed prefix of the same 16-element aperture.
    """

    channels = int(channels)
    if sites < 1 or elements_per_site < 1:
        raise ValueError("sites and elements_per_site must be positive")
    max_channels = sites * elements_per_site
    if not 1 <= channels <= max_channels:
        raise ValueError(f"channels must be in [1, {max_channels}], got {channels}")
    site_order = (0, 2, 1, 3) if sites == 4 else tuple(range(sites))
    ordering = tuple(
        site * elements_per_site + element
        for element in range(elements_per_site)
        for site in site_order
    )
    return ordering[:channels]

def center_out_nested_indices(channels: int, *, max_channels: int = 16) -> tuple[int, ...]:
    """Select a fixed-aperture ULA prefix by expanding symmetrically from center."""
    channels = int(channels)
    max_channels = int(max_channels)
    if max_channels != 16:
        raise ValueError("center_out_nested is defined for the fixed 16-element aperture")
    if not 1 <= channels <= max_channels:
        raise ValueError(f"channels must be in [1, {max_channels}], got {channels}")
    ordering = (7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2, 13, 1, 14, 0, 15)
    return ordering[:channels]


def availability_indices(
    channels: int, *, policy: str, max_channels: int = 16
) -> tuple[int, ...]:
    if policy == "fixed_aperture_uniform":
        return fixed_aperture_indices(channels, max_channels)
    if policy == "site_balanced_nested":
        if max_channels != 16:
            raise ValueError("site_balanced_nested is defined for the fixed 16-element aperture")
        return site_balanced_nested_indices(channels)
    if policy == "fixed_aperture_center_out_nested":
        return center_out_nested_indices(channels, max_channels=max_channels)
    raise ValueError(f"unknown array subset policy: {policy}")

