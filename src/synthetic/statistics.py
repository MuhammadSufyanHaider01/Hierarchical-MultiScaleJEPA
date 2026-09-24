"""Deterministic paired bootstrap utilities for synthetic studies."""

from __future__ import annotations

import numpy as np


def classification_anchor_passes(lower_bound: float, class_count: int) -> bool:
    if class_count < 2 or not np.isfinite(lower_bound):
        raise ValueError("classification anchor requires finite bound and at least two classes")
    return float(lower_bound) > 1.0 / int(class_count)


def position_anchor_passes(lower_bound_m: float) -> bool:
    if not np.isfinite(lower_bound_m):
        raise ValueError("position anchor bound must be finite")
    return float(lower_bound_m) > 0.0


def bootstrap_interval(
    values: np.ndarray,
    *,
    statistic,
    seed: int = 2026,
    resamples: int = 1000,
    confidence: float = 0.95,
    strata: np.ndarray | None = None,
) -> dict[str, float | int]:
    values = np.asarray(values)
    if values.shape[0] < 2:
        raise ValueError("bootstrap intervals require at least two realizations")
    if resamples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap parameters")
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    if strata is None:
        groups = [np.arange(len(values))]
    else:
        strata = np.asarray(strata)
        if len(strata) != len(values):
            raise ValueError("strata length must match values")
        groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    for draw in range(resamples):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        draws[draw] = float(statistic(values[indices]))
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    estimate = float(statistic(values))
    return {
        "estimate": estimate,
        "lower": float(low),
        "upper": float(high),
        "half_width": float((high - low) / 2.0),
        "confidence": float(confidence),
        "resamples": int(resamples),
    }
