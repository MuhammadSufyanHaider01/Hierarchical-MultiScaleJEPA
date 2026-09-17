#!/usr/bin/env python3
"""Compare tolerance and Pareto-knee fixed exits from existing JSON artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from evaluation import select_pareto_knee_exit  # noqa: E402


EXIT_LAYERS = (2, 4, 6, 8)
COST_CHOICES = ("flops", "latency-median", "latency-mean")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit-metrics", type=Path, required=True)
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--cost", choices=COST_CHOICES, default="flops")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path; defaults to downstream/ablations/pareto_knee_<cost>.json.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing ablation JSON.")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected in {path}")
    return value


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _canonical_layer(value: Any, *, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source} exit layer must be an integer, got {value!r}")
    try:
        layer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source} exit layer must be an integer, got {value!r}") from error
    if str(layer) != str(value) and not isinstance(value, int):
        raise ValueError(f"{source} exit layer must be canonical, got {value!r}")
    return layer


def _artifact_checkpoint(value: Mapping[str, Any], *, artifact_path: Path) -> Path:
    checkpoint = value.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"{artifact_path} has no checkpoint provenance")
    path = Path(checkpoint)
    return (path if path.is_absolute() else artifact_path.parent / path).resolve()


def _validate_provenance(
    exit_metrics: Mapping[str, Any],
    benchmarks: Mapping[str, Any],
    *,
    exit_metrics_path: Path,
    benchmarks_path: Path,
) -> dict[str, Any]:
    fields = ("method", "pretrain_seed", "dataset_suite")
    for field in fields:
        if field not in exit_metrics or field not in benchmarks:
            raise ValueError(f"artifact provenance is missing {field!r}")
        if exit_metrics[field] != benchmarks[field]:
            raise ValueError(
                f"artifact provenance mismatch for {field}: "
                f"exit_metrics={exit_metrics[field]!r}, benchmarks={benchmarks[field]!r}"
            )
    exit_checkpoint = _artifact_checkpoint(exit_metrics, artifact_path=exit_metrics_path)
    benchmark_checkpoint = _artifact_checkpoint(benchmarks, artifact_path=benchmarks_path)
    if exit_checkpoint != benchmark_checkpoint:
        raise ValueError(
            "artifact provenance mismatch for checkpoint: "
            f"exit_metrics={exit_checkpoint}, benchmarks={benchmark_checkpoint}"
        )
    return {
        "method": exit_metrics["method"],
        "pretrain_seed": int(exit_metrics["pretrain_seed"]),
        "dataset_suite": exit_metrics["dataset_suite"],
        "checkpoint": str(exit_checkpoint),
    }


def _task_costs(benchmarks: Mapping[str, Any], *, cost_type: str) -> dict[str, dict[int, float]]:
    entries = benchmarks.get("entries")
    if not isinstance(entries, list):
        raise ValueError("benchmarks.entries must be a list")
    costs: dict[str, dict[int, float]] = {}
    for entry_index, entry in enumerate(entries):
        entry = _require_mapping(entry, name=f"benchmarks.entries[{entry_index}]")
        task = entry.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError(f"benchmarks.entries[{entry_index}].task must be a non-empty string")
        layer = _canonical_layer(entry.get("exit_layer"), source=f"benchmarks.entries[{entry_index}]")
        if cost_type == "flops":
            raw_cost = entry.get("flops")
        else:
            latency = _require_mapping(entry.get("latency"), name=f"benchmark latency for {task}/L{layer}")
            latency_key = "median_ms" if cost_type == "latency-median" else "mean_ms"
            if latency_key not in latency:
                raise ValueError(f"benchmark latency for {task}/L{layer} is missing {latency_key}")
            raw_cost = latency[latency_key]
        try:
            cost = float(raw_cost)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid {cost_type} cost for {task}/L{layer}: {raw_cost!r}") from error
        task_costs = costs.setdefault(task, {})
        if layer in task_costs:
            raise ValueError(f"duplicate benchmark entry for {task}/L{layer}")
        task_costs[layer] = cost
    return costs


def _task_metrics(task: str, record: Mapping[str, Any]) -> tuple[str, bool, dict[int, float], Mapping[str, Any]]:
    spec = _require_mapping(record.get("spec"), name=f"exit metrics spec for {task}")
    metric_name = spec.get("metric_name")
    higher_is_better = spec.get("higher_is_better")
    if not isinstance(metric_name, str) or not metric_name:
        raise ValueError(f"exit metrics spec for {task} has no metric_name")
    if not isinstance(higher_is_better, bool):
        raise ValueError(f"exit metrics spec for {task} has no boolean higher_is_better")
    exits = _require_mapping(record.get("exits"), name=f"exit metrics exits for {task}")
    layers = {_canonical_layer(layer, source=f"exit metrics for {task}") for layer in exits}
    if layers != set(EXIT_LAYERS):
        raise ValueError(f"exit metrics for {task} must contain exactly {EXIT_LAYERS}, got {sorted(layers)}")
    metrics: dict[int, float] = {}
    for layer in EXIT_LAYERS:
        exit_record = _require_mapping(exits[str(layer)], name=f"exit metrics for {task}/L{layer}")
        validation = _require_mapping(exit_record.get("validation"), name=f"validation metrics for {task}/L{layer}")
        try:
            metrics[layer] = float(validation[metric_name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"validation metric {metric_name!r} missing for {task}/L{layer}") from error
    return metric_name, higher_is_better, metrics, exits


def _selected_exit(record: Mapping[str, Any], *, field: str, task: str) -> int:
    layer = _canonical_layer(record.get(field), source=f"{task}.{field}")
    if layer not in EXIT_LAYERS:
        raise ValueError(f"{task}.{field} must be one of {EXIT_LAYERS}, got L{layer}")
    return layer


def _test_metrics(exits: Mapping[str, Any], *, layer: int, task: str) -> dict[str, Any]:
    record = _require_mapping(exits[str(layer)], name=f"exit metrics for {task}/L{layer}")
    test = _require_mapping(record.get("test"), name=f"test metrics for {task}/L{layer}")
    return dict(test)


def _as_json_keyed(values: Mapping[int, float]) -> dict[str, float]:
    return {str(layer): float(values[layer]) for layer in sorted(values)}


def build_ablation(
    exit_metrics: Mapping[str, Any],
    benchmarks: Mapping[str, Any],
    *,
    exit_metrics_path: Path,
    benchmarks_path: Path,
    cost_type: str,
) -> dict[str, Any]:
    """Build a fully reproducible ablation record without writing files."""

    if cost_type not in COST_CHOICES:
        raise ValueError(f"unsupported cost type {cost_type!r}")
    provenance = _validate_provenance(
        exit_metrics,
        benchmarks,
        exit_metrics_path=exit_metrics_path,
        benchmarks_path=benchmarks_path,
    )
    tasks = _require_mapping(exit_metrics.get("tasks"), name="exit_metrics.tasks")
    if not tasks:
        raise ValueError("exit_metrics.tasks must not be empty")
    task_costs = _task_costs(benchmarks, cost_type=cost_type)
    if set(tasks) != set(task_costs):
        raise ValueError(
            "task mismatch between exit metrics and benchmarks: "
            f"exit_metrics={sorted(tasks)}, benchmarks={sorted(task_costs)}"
        )

    task_results: dict[str, Any] = {}
    for task in sorted(tasks):
        record = _require_mapping(tasks[task], name=f"exit_metrics.tasks.{task}")
        metric_name, higher_is_better, validation_metrics, exits = _task_metrics(task, record)
        costs = task_costs[task]
        if set(costs) != set(EXIT_LAYERS):
            raise ValueError(f"benchmarks for {task} must contain exactly {EXIT_LAYERS}, got {sorted(costs)}")
        result = select_pareto_knee_exit(
            validation_metrics,
            costs,
            higher_is_better=higher_is_better,
        )
        tolerance_exit = _selected_exit(
            record,
            field="selected_exit_earliest_within_tolerance",
            task=task,
        )
        best_exit = _selected_exit(record, field="selected_exit_best_validation", task=task)
        pareto_test = _test_metrics(exits, layer=result.selected_exit, task=task)
        tolerance_test = _test_metrics(exits, layer=tolerance_exit, task=task)
        task_results[task] = {
            "metric_name": metric_name,
            "higher_is_better": higher_is_better,
            "validation_metrics": _as_json_keyed(validation_metrics),
            "oriented_utility": _as_json_keyed(result.oriented_utility),
            "costs": _as_json_keyed(costs),
            "pareto_exits": list(result.pareto_exits),
            "dominated_exits": list(result.dominated_exits),
            "cheap_endpoint": result.cheap_endpoint,
            "performance_endpoint": result.performance_endpoint,
            "normalized_compute": _as_json_keyed(result.normalized_compute),
            "normalized_utility": _as_json_keyed(result.normalized_utility),
            "knee_scores": _as_json_keyed(result.knee_scores),
            "selected_exit_pareto_knee": result.selected_exit,
            "selected_exit_tolerance": tolerance_exit,
            "selected_exit_best_validation": best_exit,
            "selected_test_pareto_knee": pareto_test,
            "selected_test_tolerance": tolerance_test,
            "used_fallback": result.used_fallback,
            "fallback_reason": result.fallback_reason,
        }
    return {
        "schema_version": 1,
        "ablation": "pareto_knee_exit_selection",
        "cost_type": cost_type,
        "selection_uses": "validation_metrics_and_costs_only",
        "source_exit_metrics": str(exit_metrics_path.resolve()),
        "source_benchmarks": str(benchmarks_path.resolve()),
        "provenance": provenance,
        "tasks": task_results,
    }


def _default_output(exit_metrics: Path, cost_type: str) -> Path:
    suffix = cost_type.replace("-", "_")
    return exit_metrics.parent / "ablations" / f"pareto_knee_{suffix}.json"


def main() -> int:
    args = _arguments()
    exit_metrics_path = args.exit_metrics.resolve()
    benchmarks_path = args.benchmarks.resolve()
    output = (args.output.resolve() if args.output is not None else _default_output(exit_metrics_path, args.cost))
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}; pass --overwrite to replace it")
    exit_metrics = _read_json(exit_metrics_path)
    benchmarks = _read_json(benchmarks_path)
    result = build_ablation(
        exit_metrics,
        benchmarks,
        exit_metrics_path=exit_metrics_path,
        benchmarks_path=benchmarks_path,
        cost_type=args.cost,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ablation": str(output), "tasks": sorted(result["tasks"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
