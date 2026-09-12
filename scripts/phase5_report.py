#!/usr/bin/env python3
"""Aggregate Phase 5 per-seed exit and inference measurements without claims."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


METHOD_LABELS = {"p0": "P0 / B0-B1", "p1": "P1 / B2", "p2": "P2 / B3"}
EXIT_LAYERS = (2, 4, 6, 8)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs/phase5")
    parser.add_argument("--methods", choices=tuple(METHOD_LABELS), nargs="+", default=tuple(METHOD_LABELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        raise ValueError("cannot summarize an empty metric series")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else None,
        "minimum": min(values),
        "maximum": max(values),
    }


def _relative(full: float, partial: float) -> dict[str, float]:
    return {
        "relative_to_l8": partial / full if full else math.nan,
        "savings_relative_to_l8": 1.0 - partial / full if full else math.nan,
    }


def _read(path: Path) -> dict[str, object] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _metric_rows(metrics: list[tuple[int, dict[str, object]]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    selected_layers: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for seed, result in metrics:
        for task_name, task in result["tasks"].items():
            spec = task["spec"]
            metric_name = spec["metric_name"]
            for exit_layer in EXIT_LAYERS:
                value = task["exits"][str(exit_layer)]["test"][metric_name]
                grouped[(task_name, metric_name, f"L{exit_layer}", "fixed_exit")].append(float(value))
            for policy, record_name, layer_name in (
                ("best_validation", "selected_test_best_validation", "selected_exit_best_validation"),
                ("earliest_within_tolerance", "selected_test_earliest_within_tolerance", "selected_exit_earliest_within_tolerance"),
            ):
                grouped[(task_name, metric_name, policy, "validation_selected")].append(
                    float(task[record_name][metric_name])
                )
                selected_layers[(task_name, policy, metric_name)].append(float(task[layer_name]))

    rows = []
    for (task, metric, label, kind), values in sorted(grouped.items()):
        row = {"task": task, "metric": metric, "evaluation": label, "kind": kind, **_stats(values)}
        if kind == "validation_selected":
            row["selected_exit_layers"] = _stats(selected_layers[(task, label, metric)])
        rows.append(row)
    return rows


def _compute_rows(benchmarks: list[tuple[int, dict[str, object]]]) -> list[dict[str, object]]:
    series: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for seed, result in benchmarks:
        for entry in result["entries"]:
            task = str(entry["task"])
            layer = int(entry["exit_layer"])
            series[(task, layer, "flops")].append(float(entry["flops"]))
            series[(task, layer, "latency_mean_ms")].append(float(entry["latency"]["mean_ms"]))

    rows = []
    for task in sorted({key[0] for key in series}):
        for measure in ("flops", "latency_mean_ms"):
            l8 = _stats(series[(task, 8, measure)])["mean"] if (task, 8, measure) in series else None
            for layer in EXIT_LAYERS:
                values = series.get((task, layer, measure))
                if not values:
                    continue
                row = {"task": task, "exit_layer": layer, "measure": measure, **_stats(values)}
                if isinstance(l8, (int, float)):
                    row.update(_relative(float(l8), float(row["mean"])))
                rows.append(row)
    return rows


def _pretraining_rows(summaries: list[tuple[int, dict[str, object]]]) -> list[dict[str, object]]:
    series: dict[str, list[float]] = defaultdict(list)
    for _, summary in summaries:
        overhead = summary.get("pretraining_overhead")
        if not isinstance(overhead, dict):
            continue
        for source, label in (
            ("wall_time_seconds", "wall_time_seconds"),
            ("optimizer_steps_per_second", "optimizer_steps_per_second"),
            ("peak_memory_allocated_bytes", "peak_memory_allocated_bytes"),
        ):
            value = overhead.get(source)
            if isinstance(value, (int, float)) and math.isfinite(value):
                series[label].append(float(value))
    return [{"measure": measure, **_stats(values)} for measure, values in sorted(series.items())]


def _format(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Phase 5 experiment summary",
        "",
        "This is an aggregation of recorded artifacts, not an inferential claim. "
        "A complete main result requires all three seeds for P0/B0-B1, P1/B2, and P2/B3.",
        "",
        f"Requested methods: {', '.join(summary['requested_methods'])}; seeds: {', '.join(map(str, summary['requested_seeds']))}.",
        "",
    ]
    missing = summary["missing_artifacts"]
    if missing:
        lines.extend(["## Missing artifacts", "", "```text", *missing, "```", ""])
    lines.extend([
        "## Downstream test metrics",
        "",
        "Selection policies use validation records only; test values are reported after selection.",
        "",
        "| Method | Task | Evaluation | Metric | n | Mean | SD |",
        "|---|---|---|---|---:|---:|---:|",
    ])
    for method, rows in summary["metrics"].items():
        for row in rows:
            lines.append(
                f"| {METHOD_LABELS[method]} | {row['task']} | {row['evaluation']} | {row['metric']} | "
                f"{row['count']} | {_format(row['mean'])} | {_format(row['standard_deviation'])} |"
            )
    lines.extend(["", "## Batch-1 frozen-exit inference", "", "| Method | Task | Exit | Measure | Mean | SD | Saving vs L8 |", "|---|---|---:|---|---:|---:|---:|"])
    for method, rows in summary["compute"].items():
        for row in rows:
            lines.append(
                f"| {METHOD_LABELS[method]} | {row['task']} | L{row['exit_layer']} | {row['measure']} | "
                f"{_format(row['mean'])} | {_format(row['standard_deviation'])} | {_format(row.get('savings_relative_to_l8'))} |"
            )
    lines.extend(["", "## Pretraining overhead", "", "These measurements are reported separately and are not treated as inference savings.", "", "| Method | Measure | n | Mean | SD |", "|---|---|---:|---:|---:|"])
    for method, rows in summary["pretraining_overhead"].items():
        for row in rows:
            lines.append(
                f"| {METHOD_LABELS[method]} | {row['measure']} | {row['count']} | "
                f"{_format(row['mean'])} | {_format(row['standard_deviation'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def _plot(summary: dict[str, object], output: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    selected = []
    for method, rows in summary["metrics"].items():
        for row in rows:
            if row["evaluation"] == "earliest_within_tolerance":
                selected.append((method, row))
    if not selected:
        return None
    tasks = sorted({row["task"] for _, row in selected})
    methods = [method for method in summary["requested_methods"] if any(item[0] == method for item in selected)]
    figure, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)
    for axis, task in zip(axes[0], tasks):
        task_values = {method: row for method, row in selected if row["task"] == task}
        labels = [METHOD_LABELS[method] for method in methods if method in task_values]
        means = [task_values[method]["mean"] for method in methods if method in task_values]
        errors = [task_values[method]["standard_deviation"] or 0.0 for method in methods if method in task_values]
        axis.bar(labels, means, yerr=errors, capsize=4)
        axis.set_title(f"{task}: earliest within tolerance")
        axis.set_ylabel(next(iter(task_values.values()))["metric"])
        axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return str(output)


def main() -> int:
    args = _arguments()
    output_dir = args.output_dir or args.run_root
    output_dir.mkdir(parents=True, exist_ok=True)
    # The workspace sandbox does not expose a writable user config directory.
    # Keep Matplotlib's cache alongside the gitignored report artifacts.
    os.environ["MPLCONFIGDIR"] = str(output_dir / ".matplotlib")
    metrics_by_method: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
    benchmarks_by_method: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
    pretraining_by_method: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
    missing = []
    for method in args.methods:
        for seed in args.seeds:
            base = args.run_root / method / f"seed_{seed:03d}"
            metrics = _read(base / "downstream/exit_metrics.json")
            benchmarks = _read(base / "benchmarks.json")
            pretraining = _read(base / "summary.json")
            if metrics is None:
                missing.append(f"missing downstream metrics: {base / 'downstream/exit_metrics.json'}")
            else:
                metrics_by_method[method].append((seed, metrics))
            if benchmarks is None:
                missing.append(f"missing benchmark: {base / 'benchmarks.json'}")
            else:
                benchmarks_by_method[method].append((seed, benchmarks))
            if pretraining is None:
                missing.append(f"missing pretraining summary: {base / 'summary.json'}")
            else:
                pretraining_by_method[method].append((seed, pretraining))
    summary = {
        "requested_methods": list(args.methods),
        "requested_seeds": list(args.seeds),
        "missing_artifacts": missing,
        "metrics": {method: _metric_rows(records) for method, records in metrics_by_method.items()},
        "compute": {method: _compute_rows(records) for method, records in benchmarks_by_method.items()},
        "pretraining_overhead": {
            method: _pretraining_rows(records) for method, records in pretraining_by_method.items()
        },
    }
    (output_dir / "phase5_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (output_dir / "phase5_summary.md").write_text(_markdown(summary))
    csv_rows = []
    for method, rows in summary["metrics"].items():
        csv_rows.extend({"category": "metric", "method": method, **row} for row in rows)
    for method, rows in summary["compute"].items():
        csv_rows.extend({"category": "compute", "method": method, **row} for row in rows)
    for method, rows in summary["pretraining_overhead"].items():
        csv_rows.extend({"category": "pretraining_overhead", "method": method, **row} for row in rows)
    fields = sorted({field for row in csv_rows for field in row})
    with (output_dir / "phase5_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    plot = _plot(summary, output_dir / "phase5_selected_exits.png")
    print(json.dumps({"summary": str(output_dir / "phase5_summary.md"), "plot": plot, "missing_artifacts": len(missing)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
