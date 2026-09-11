"""Validation-only fixed-exit selection utilities."""

from __future__ import annotations

import math
from collections.abc import Mapping


def _validate_metrics(metrics: Mapping[int, float], tolerance: float | None = None) -> None:
    if not metrics:
        raise ValueError("exit metrics cannot be empty")
    if tolerance is not None and (tolerance < 0 or not math.isfinite(tolerance)):
        raise ValueError(f"tolerance must be finite and non-negative, got {tolerance}")
    for layer, value in metrics.items():
        if int(layer) < 1:
            raise ValueError(f"exit layers must be positive, got {layer}")
        if not math.isfinite(float(value)):
            raise ValueError(f"exit metric for layer {layer} is not finite: {value}")


def select_best_exit(metrics: Mapping[int, float], *, higher_is_better: bool) -> int:
    """Select the best validation exit, breaking ties toward the earliest layer."""

    _validate_metrics(metrics)
    ordered = sorted((int(layer), float(value)) for layer, value in metrics.items())
    key = (lambda item: (-item[1], item[0])) if higher_is_better else (lambda item: (item[1], item[0]))
    return min(ordered, key=key)[0]


def select_earliest_within_tolerance(
    metrics: Mapping[int, float],
    *,
    higher_is_better: bool,
    tolerance: float = 0.05,
) -> int:
    """Select earliest validation exit within a relative tolerance of the best."""

    _validate_metrics(metrics, tolerance=tolerance)
    best_layer = select_best_exit(metrics, higher_is_better=higher_is_better)
    best_value = float(metrics[best_layer])
    if higher_is_better:
        threshold = best_value * (1.0 - tolerance)
        eligible = [int(layer) for layer, value in metrics.items() if float(value) >= threshold]
    else:
        threshold = best_value * (1.0 + tolerance)
        eligible = [int(layer) for layer, value in metrics.items() if float(value) <= threshold]
    return min(eligible)
