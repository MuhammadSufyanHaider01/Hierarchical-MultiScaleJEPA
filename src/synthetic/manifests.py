"""Immutable realization-level calibration/selection/test split manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


SPLIT_FRACTIONS = {"calibration": 0.20, "selection": 0.20, "test": 0.60}
SPLIT_NAMES = tuple(SPLIT_FRACTIONS)


def _allocate(group: np.ndarray, rng: np.random.Generator) -> dict[str, list[int]]:
    shuffled = group[rng.permutation(len(group))]
    exact = np.asarray(list(SPLIT_FRACTIONS.values()), dtype=np.float64) * len(group)
    counts = np.floor(exact).astype(int)
    remainder = len(group) - int(counts.sum())
    if remainder:
        for index in np.argsort(-(exact - counts))[:remainder]:
            counts[index] += 1
    result: dict[str, list[int]] = {}
    start = 0
    for name, count in zip(SPLIT_NAMES, counts):
        result[name] = [int(v) for v in shuffled[start : start + count]]
        start += int(count)
    return result


def create_realization_manifest(
    realization_ids: Sequence[str | int],
    labels: Sequence[int] | np.ndarray,
    *,
    task: str,
    target_type: str,
    seed: int = 2026,
    coordinate_min: Sequence[float] | None = None,
    coordinate_max: Sequence[float] | None = None,
) -> dict[str, object]:
    """Create deterministic split IDs, stratified by class or coarse 3-D cell."""

    ids = [str(value) for value in realization_ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("realization IDs must be non-empty and unique")
    values = np.asarray(labels)
    if values.shape[0] != len(ids):
        raise ValueError("labels and realization IDs must have equal length")
    rng = np.random.default_rng(int(seed))

    groups: dict[tuple[int, ...], list[int]] = {}
    if target_type == "classification":
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise TypeError("classification labels must be a one-dimensional integer array")
        for index, label in enumerate(values.tolist()):
            groups.setdefault((int(label),), []).append(index)
    elif target_type == "position":
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("position labels must have shape [N,3]")
        if coordinate_min is None or coordinate_max is None:
            raise ValueError("coordinate bounds are required for position stratification")
        lo = np.asarray(coordinate_min, dtype=np.float64)
        hi = np.asarray(coordinate_max, dtype=np.float64)
        if lo.shape != (3,) or hi.shape != (3,) or np.any(hi <= lo):
            raise ValueError("position coordinate bounds must be ordered 3-vectors")
        bins = np.clip(np.floor((values - lo) / (hi - lo) * 4), 0, 3).astype(int)
        for index, cell in enumerate(bins.tolist()):
            groups.setdefault(tuple(int(v) for v in cell), []).append(index)
    else:
        raise ValueError(f"unsupported target_type: {target_type}")

    split_indices = {name: [] for name in SPLIT_NAMES}
    for key in sorted(groups):
        allocated = _allocate(np.asarray(groups[key], dtype=np.int64), rng)
        for name in SPLIT_NAMES:
            split_indices[name].extend(allocated[name])
    index_to_id = {index: value for index, value in enumerate(ids)}
    splits = {
        name: [index_to_id[index] for index in sorted(indices)]
        for name, indices in split_indices.items()
    }
    manifest = {
        "schema_version": 1,
        "task": task,
        "seed": int(seed),
        "fractions": SPLIT_FRACTIONS,
        "target_type": target_type,
        "realization_ids": ids,
        "splits": splits,
    }
    validate_realization_manifest(manifest)
    return manifest


def validate_realization_manifest(manifest: dict[str, object]) -> None:
    ids = manifest.get("realization_ids")
    splits = manifest.get("splits")
    if not isinstance(ids, list) or not isinstance(splits, dict):
        raise ValueError("manifest must contain realization_ids and splits")
    if len(set(ids)) != len(ids):
        raise ValueError("manifest realization IDs are not unique")
    seen: list[str] = []
    for name in SPLIT_NAMES:
        values = splits.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"manifest split {name} is missing or empty")
        if len(set(values)) != len(values):
            raise ValueError(f"manifest split {name} contains duplicate IDs")
        seen.extend(values)
    if len(seen) != len(set(seen)) or set(seen) != set(ids):
        raise ValueError("calibration, selection, and test IDs must partition the master IDs")


def write_manifest(path: str | Path, manifest: dict[str, object]) -> str:
    """Write once and return the content SHA-256; never replace an existing split."""

    path = Path(path)
    validate_realization_manifest(manifest)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable synthetic manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    path.write_text(payload)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
