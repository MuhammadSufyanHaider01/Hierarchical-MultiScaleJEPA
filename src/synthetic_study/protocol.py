"""Paired sample splits and nested availability views for the synthetic study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


ANTENNA_LEVELS = (1, 2, 4, 6, 8, 10, 12, 16)
TIME_PATCH_LEVELS = (2, 4, 6, 8, 10, 12, 14, 18, 22, 28, 36, 44, 56)
EXIT_LAYERS = (2, 4, 6, 8)
TASKS = ("nrpos-outdoor", "interference", "jamming")
PATCH_SIZE = 16
MASTER_SHAPE = (16, 224, 896)
ARRAY_SUBSET_ORDER = (7, 8, 3, 12, 5, 10, 1, 14, 6, 9, 4, 11, 2, 13, 0, 15)
TIME_SWEEP_CHANNELS = {"nrpos-outdoor": 4, "interference": 1, "jamming": 1}


def token_count(channels: int, temporal_patches: int) -> int:
    """Return encoder tokens for a task view under the 16-pixel patch size."""

    if channels not in ANTENNA_LEVELS:
        raise ValueError(f"unsupported channel count C={channels}")
    if temporal_patches not in TIME_PATCH_LEVELS:
        raise ValueError(f"unsupported temporal patch count T_p={temporal_patches}")
    if channels > MASTER_SHAPE[0] or temporal_patches * PATCH_SIZE > MASTER_SHAPE[2]:
        raise ValueError("availability view exceeds the maximum-information master")
    if MASTER_SHAPE[1] % PATCH_SIZE:
        raise ValueError("master frequency dimension is not patch aligned")
    count = channels * (MASTER_SHAPE[1] // PATCH_SIZE) * temporal_patches
    if count > 3136:
        raise ValueError(f"view has {count} tokens, exceeding the locked 3136-token cap")
    return count


def availability_view(
    features: np.ndarray,
    *,
    task: str,
    axis: str,
    level: int,
) -> np.ndarray:
    """Return a non-mutating antenna/time crop from one master feature tensor.

    Antenna subsets use the same center-out ordering at every level. Temporal
    crops always take the leading `T_p * 16` columns. No interpolation or
    condition-specific normalization is performed here.
    """

    if task not in TASKS:
        raise ValueError(f"unsupported task {task!r}")
    x = np.asarray(features)
    if x.shape != MASTER_SHAPE:
        raise ValueError(f"master sample must have shape {MASTER_SHAPE}, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("master sample contains non-finite values")

    if axis == "antenna":
        if level not in ANTENNA_LEVELS:
            raise ValueError(f"unsupported antenna count C={level}")
        channels = ARRAY_SUBSET_ORDER[:level]
        temporal_patches = 14
    elif axis == "time":
        if level not in TIME_PATCH_LEVELS:
            raise ValueError(f"unsupported temporal patch count T_p={level}")
        channels = ARRAY_SUBSET_ORDER[: TIME_SWEEP_CHANNELS[task]]
        temporal_patches = level
    else:
        raise ValueError(f"axis must be 'antenna' or 'time', got {axis!r}")

    token_count(len(channels), temporal_patches)
    width = temporal_patches * PATCH_SIZE
    view = x[np.asarray(channels), :, :width]
    if view.shape != (len(channels), MASTER_SHAPE[1], width):
        raise AssertionError("availability crop produced an unexpected shape")
    return view


def _largest_remainder_counts(size: int, fractions: Mapping[str, float]) -> dict[str, int]:
    names = tuple(fractions)
    raw = np.asarray([fractions[name] * size for name in names], dtype=np.float64)
    counts = np.floor(raw).astype(int)
    remainder = size - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:remainder]] += 1
    return dict(zip(names, counts.tolist(), strict=True))


def make_split_manifest(
    realization_ids: Sequence[str],
    jammer_mode_labels: Sequence[int],
    *,
    seed: int,
    pilot: bool = False,
) -> dict[str, object]:
    """Create deterministic stratified splits shared by all tasks and models.

    Formal data use four disjoint ID sets. Pilots intentionally omit test IDs;
    every pilot sample belongs to head-train, head-validation, or exit-selection.
    Stratification is by jammer mode, which also balances the binary interference
    target. Localization uses the same realization split without task-specific
    reassignment.
    """

    ids = tuple(str(value) for value in realization_ids)
    labels = np.asarray(jammer_mode_labels, dtype=np.int64).reshape(-1)
    if len(ids) != labels.size:
        raise ValueError("realization IDs and jammer labels must have equal length")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("realization IDs must be non-empty and unique")
    if np.any((labels < 0) | (labels > 5)):
        raise ValueError("jammer-mode labels must be in [0,5]")

    fractions = (
        {"head_train": 0.70, "head_validation": 0.15, "exit_selection": 0.15}
        if pilot
        else {
            "head_train": 0.60,
            "head_validation": 0.15,
            "exit_selection": 0.10,
            "test": 0.15,
        }
    )
    rng = np.random.default_rng(seed)
    split_ids = {name: [] for name in fractions}
    for label in sorted(np.unique(labels).tolist()):
        class_ids = np.asarray([ids[i] for i in np.flatnonzero(labels == label)], dtype=object)
        class_ids = class_ids[rng.permutation(class_ids.size)]
        counts = _largest_remainder_counts(class_ids.size, fractions)
        cursor = 0
        for name, count in counts.items():
            split_ids[name].extend(class_ids[cursor : cursor + count].tolist())
            cursor += count

    # Deterministically shuffle within split so the stored manifest does not
    # group samples by class while retaining class stratification.
    for name in split_ids:
        values = np.asarray(split_ids[name], dtype=object)
        split_ids[name] = values[rng.permutation(values.size)].tolist()

    expected = set(ids)
    all_split_ids = [value for values in split_ids.values() for value in values]
    if len(all_split_ids) != len(expected) or set(all_split_ids) != expected:
        raise AssertionError("split construction lost or duplicated realization IDs")
    counts_by_split = {name: len(values) for name, values in split_ids.items()}
    label_by_id = dict(zip(ids, labels.tolist(), strict=True))
    return {
        "schema_version": 1,
        "seed": int(seed),
        "pilot": bool(pilot),
        "stratification": "jammer_mode_6class",
        "fractions": fractions,
        "counts": counts_by_split,
        "realization_ids": {name: split_ids[name] for name in fractions},
        "class_counts": {
            name: {
                str(label): int(sum(label_by_id[value] == label for value in values))
                for label in range(6)
            }
            for name, values in split_ids.items()
        },
    }
