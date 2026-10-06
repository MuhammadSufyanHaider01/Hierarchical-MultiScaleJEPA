"""Validation and aggregation helpers for seven-seed frozen-head masking reports."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Mapping

MASKS = ("frequency_n6", "time_n6", "region_n6", "multi_block_n6")
METHODS = ("p0", "p1", "p2", "p2_reverse")
TASKS = ("rfs", "jamming", "human-activity", "interference", "nrpos-outdoor", "deepmimo-beam")
EXITS = (2, 4, 6, 8)
SEEDS = tuple(range(7))
SPLIT_SEED = 2025
METHOD_LABELS = {
    "b0": "B0 — Original LatentWave",
    "p0": "P0 — LatentWave with Post-hoc Frozen Early Exits",
    "p1": "P1 — Same-scale Multi-depth JEPA",
    "p2": "P2 — Proposed Hierarchical Multi-Scale JEPA",
    "p2_reverse": "P2-Reverse — Coarse-to-fine Hierarchical JEPA",
}
TASK_LABELS = {
    "rfs": "RF sensing",
    "jamming": "Jamming classification",
    "human-activity": "Human activity sensing",
    "interference": "Interference classification",
    "nrpos-outdoor": "NR positioning",
    "deepmimo-beam": "DeepMIMO beam",
}


def finite_metric(value: Any, *, context: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite metric at {context}: {value!r}")
    return number


def mean_sd(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty seed set")
    vals = [finite_metric(value, context="summary") for value in values]
    return {
        "mean": statistics.fmean(vals),
        "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def packet_metrics(
    packet: Mapping[str, Any],
    *,
    method: str,
    head_seed: int,
    expected_tasks: tuple[str, ...] = TASKS,
    split_seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Validate one complete run and return per-exit and selected test values."""

    if packet.get("method") != method:
        raise ValueError(f"method mismatch: expected {method}, got {packet.get('method')}")
    if int(packet.get("pretrain_seed", -1)) != 0:
        raise ValueError("head study must use the fixed seed-0 pretrained checkpoint")
    if int(packet.get("head_seed", -1)) != head_seed:
        raise ValueError(f"head seed mismatch: expected {head_seed}, got {packet.get('head_seed')}")
    recorded_split = packet.get("split_seed")
    if recorded_split is not None and int(recorded_split) != split_seed:
        raise ValueError(f"split seed mismatch: expected {split_seed}, got {recorded_split}")
    if not packet.get("protocol", {}).get("test_evaluation", False):
        raise ValueError("packet has no held-out test evaluation")
    tasks = packet.get("tasks", {})
    if set(tasks) != set(expected_tasks):
        missing = sorted(set(expected_tasks) - set(tasks))
        extra = sorted(set(tasks) - set(expected_tasks))
        raise ValueError(f"task coverage mismatch; missing={missing}, extra={extra}")

    result: dict[str, Any] = {"method": method, "head_seed": head_seed, "tasks": {}}
    for task_name in expected_tasks:
        task = tasks[task_name]
        spec = task.get("spec", {})
        metric = spec.get("metric_name")
        if not metric:
            raise ValueError(f"missing metric name for {task_name}")
        higher_is_better = bool(spec.get("higher_is_better"))
        exits = task.get("exits", {})
        expected_exits = {str(exit_layer) for exit_layer in EXITS}
        if set(exits) != expected_exits:
            raise ValueError(f"exit coverage mismatch for {task_name}")
        fixed = {}
        for exit_layer in EXITS:
            record = exits[str(exit_layer)]
            test = record.get("test")
            if not isinstance(test, Mapping) or metric not in test:
                raise ValueError(f"missing test metric {metric} for {task_name} L{exit_layer}")
            fixed[exit_layer] = finite_metric(
                test[metric], context=f"{method}/seed{head_seed}/{task_name}/L{exit_layer}"
            )
        selected = int(task.get("selected_exit_best_validation", -1))
        if selected not in EXITS:
            raise ValueError(f"invalid validation-selected exit for {task_name}: {selected}")
        result["tasks"][task_name] = {
            "metric": metric,
            "higher_is_better": higher_is_better,
            "fixed": fixed,
            "selected_exit": selected,
            "selected_test": fixed[selected],
        }
    return result


def aggregate_method(
    packets_by_seed: Mapping[int, Mapping[str, Any]],
    *,
    method: str,
    expected_seeds: tuple[int, ...] = SEEDS,
    expected_tasks: tuple[str, ...] = TASKS,
) -> dict[str, Any]:
    if set(map(int, packets_by_seed)) != set(expected_seeds):
        raise ValueError(
            f"seed coverage mismatch for {method}: expected {list(expected_seeds)}, "
            f"got {sorted(map(int, packets_by_seed))}"
        )
    per_seed = {
        seed: packet_metrics(
            packets_by_seed[seed], method=method, head_seed=seed, expected_tasks=expected_tasks
        )
        for seed in expected_seeds
    }
    aggregated: dict[str, Any] = {"method": method, "tasks": {}, "seed_values": per_seed}
    for task_name in expected_tasks:
        task_seed_rows = {seed: per_seed[seed]["tasks"][task_name] for seed in expected_seeds}
        metric_names = {row["metric"] for row in task_seed_rows.values()}
        directions = {row["higher_is_better"] for row in task_seed_rows.values()}
        if len(metric_names) != 1 or len(directions) != 1:
            raise ValueError(f"metric definition mismatch across seeds for {method}/{task_name}")
        fixed = {}
        for exit_layer in EXITS:
            values = [task_seed_rows[seed]["fixed"][exit_layer] for seed in expected_seeds]
            fixed[exit_layer] = {**mean_sd(values), "per_seed": dict(zip(expected_seeds, values))}
        selected_values = [task_seed_rows[seed]["selected_test"] for seed in expected_seeds]
        selected_layers = [task_seed_rows[seed]["selected_exit"] for seed in expected_seeds]
        aggregated["tasks"][task_name] = {
            "metric": next(iter(metric_names)),
            "higher_is_better": next(iter(directions)),
            "fixed": fixed,
            "selected": {
                **mean_sd(selected_values),
                "per_seed": dict(zip(expected_seeds, selected_values)),
                "selected_exit_by_seed": dict(zip(expected_seeds, selected_layers)),
                "selection_frequency": dict(Counter(selected_layers)),
            },
        }
    return aggregated


def derive_b0_from_p0(p0_aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """B0 is the P0 full-depth L8 result in the existing report convention."""

    result = {"method": "b0", "tasks": {}}
    for task_name, task in p0_aggregate["tasks"].items():
        l8 = task["fixed"][8]
        per_seed = {
            seed: rows["tasks"][task_name]["fixed"][8]
            for seed, rows in p0_aggregate["seed_values"].items()
        }
        result["tasks"][task_name] = {
            "metric": task["metric"],
            "higher_is_better": task["higher_is_better"],
            "fixed_l8": l8,
            "fixed": {8: l8},
            "selected": {
                **{key: l8[key] for key in ("mean", "sd", "min", "max", "n")},
                "per_seed": per_seed,
                "selected_exit_by_seed": {seed: 8 for seed in per_seed},
                "selection_frequency": {8: len(per_seed)},
            },
            "per_seed": per_seed,
        }
    return result
