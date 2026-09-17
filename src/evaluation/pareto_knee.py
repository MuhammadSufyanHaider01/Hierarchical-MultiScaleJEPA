"""Deterministic, validation-only Pareto-knee fixed-exit selection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class ParetoKneeResult:
    """Intermediate values needed to reproduce one Pareto-knee decision."""

    selected_exit: int
    pareto_exits: tuple[int, ...]
    dominated_exits: tuple[int, ...]
    cheap_endpoint: int
    performance_endpoint: int
    oriented_utility: dict[int, float]
    normalized_compute: dict[int, float]
    normalized_utility: dict[int, float]
    knee_scores: dict[int, float]
    used_fallback: bool
    fallback_reason: str | None


def _canonical_values(values: Mapping[int, float], *, name: str, positive: bool) -> dict[int, float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{name} must be a non-empty mapping")

    canonical: dict[int, float] = {}
    for layer, value in values.items():
        if isinstance(layer, bool) or not isinstance(layer, Integral) or int(layer) < 1:
            raise ValueError(f"{name} layers must be positive integers, got {layer!r}")
        layer = int(layer)
        if layer in canonical:
            raise ValueError(f"{name} contains duplicate layer {layer}")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} value for layer {layer} is not numeric: {value!r}") from error
        if not math.isfinite(numeric):
            raise ValueError(f"{name} value for layer {layer} is not finite: {value!r}")
        if positive and numeric <= 0.0:
            raise ValueError(f"{name} value for layer {layer} must be positive: {value!r}")
        canonical[layer] = numeric
    return canonical


def select_pareto_knee_exit(
    metrics: Mapping[int, float],
    costs: Mapping[int, float],
    *,
    higher_is_better: bool,
) -> ParetoKneeResult:
    """Select a fixed exit from validation utility and inference cost.

    The function has no file I/O and never receives test metrics.  It first
    removes dominated exits, then selects the intermediate Pareto point with
    maximum normalized utility minus normalized cost.  Explicit fallbacks
    retain the best-performance endpoint where a geometric knee is absent.
    """

    metric_values = _canonical_values(metrics, name="metrics", positive=False)
    cost_values = _canonical_values(costs, name="costs", positive=True)
    if metric_values.keys() != cost_values.keys():
        raise ValueError(
            "metrics and costs must contain exactly the same exit layers: "
            f"metrics={sorted(metric_values)}, costs={sorted(cost_values)}"
        )

    utility = {
        layer: value if higher_is_better else -value
        for layer, value in metric_values.items()
    }
    layers = tuple(sorted(metric_values))
    pareto: list[int] = []
    for layer in layers:
        dominated = any(
            other != layer
            and cost_values[other] <= cost_values[layer]
            and utility[other] >= utility[layer]
            and (cost_values[other] < cost_values[layer] or utility[other] > utility[layer])
            for other in layers
        )
        if not dominated:
            pareto.append(layer)
    pareto.sort(key=lambda layer: (cost_values[layer], layer))
    pareto_exits = tuple(pareto)
    dominated_exits = tuple(layer for layer in layers if layer not in pareto_exits)

    if len(pareto_exits) == 1:
        layer = pareto_exits[0]
        return ParetoKneeResult(
            selected_exit=layer,
            pareto_exits=pareto_exits,
            dominated_exits=dominated_exits,
            cheap_endpoint=layer,
            performance_endpoint=layer,
            oriented_utility={layer: utility[layer]},
            normalized_compute={layer: 0.0},
            normalized_utility={layer: 0.0},
            knee_scores={layer: 0.0},
            used_fallback=True,
            fallback_reason="single_pareto_exit",
        )

    cheap = min(pareto_exits, key=lambda layer: (cost_values[layer], layer))
    performance = max(
        pareto_exits,
        key=lambda layer: (utility[layer], -cost_values[layer], -layer),
    )
    cost_range = cost_values[performance] - cost_values[cheap]
    utility_range = utility[performance] - utility[cheap]
    if cost_range <= 0.0:
        raise ValueError("Pareto frontier has zero compute range between its endpoints")
    if utility_range <= 0.0:
        raise ValueError("Pareto frontier has zero utility range between its endpoints")

    normalized_compute = {
        layer: (cost_values[layer] - cost_values[cheap]) / cost_range
        for layer in pareto_exits
    }
    normalized_utility = {
        layer: (utility[layer] - utility[cheap]) / utility_range
        for layer in pareto_exits
    }
    knee_scores = {
        layer: normalized_utility[layer] - normalized_compute[layer]
        for layer in pareto_exits
    }
    intermediate = tuple(
        layer for layer in pareto_exits if layer not in {cheap, performance}
    )
    if not intermediate:
        return ParetoKneeResult(
            selected_exit=performance,
            pareto_exits=pareto_exits,
            dominated_exits=dominated_exits,
            cheap_endpoint=cheap,
            performance_endpoint=performance,
            oriented_utility={layer: utility[layer] for layer in pareto_exits},
            normalized_compute=normalized_compute,
            normalized_utility=normalized_utility,
            knee_scores=knee_scores,
            used_fallback=True,
            fallback_reason="no_intermediate_pareto_point",
        )

    selected = max(intermediate, key=lambda layer: (knee_scores[layer], -layer))
    if knee_scores[selected] <= 0.0:
        return ParetoKneeResult(
            selected_exit=performance,
            pareto_exits=pareto_exits,
            dominated_exits=dominated_exits,
            cheap_endpoint=cheap,
            performance_endpoint=performance,
            oriented_utility={layer: utility[layer] for layer in pareto_exits},
            normalized_compute=normalized_compute,
            normalized_utility=normalized_utility,
            knee_scores=knee_scores,
            used_fallback=True,
            fallback_reason="no_positive_knee",
        )

    return ParetoKneeResult(
        selected_exit=selected,
        pareto_exits=pareto_exits,
        dominated_exits=dominated_exits,
        cheap_endpoint=cheap,
        performance_endpoint=performance,
        oriented_utility={layer: utility[layer] for layer in pareto_exits},
        normalized_compute=normalized_compute,
        normalized_utility=normalized_utility,
        knee_scores=knee_scores,
        used_fallback=False,
        fallback_reason=None,
    )
