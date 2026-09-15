#!/usr/bin/env python3
"""Create the single-seed, paper-comparable Phase 5 result packet.

The source-faithful P0 probe and the context-encoder frozen-exit protocol are
intentionally reported in separate sections.  The former uses the EMA target
encoder and the original LatentWave 80/20 scikit-learn probe; the latter is
the comparable B1/P2 80/10/10 frozen-head evaluation.  Combining those two
protocols into one metric table would make the result misleading.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXIT_LAYERS = (2, 4, 6, 8)
TASKS = ("rfs", "nrpos-outdoor", "deepmimo-beam", "deepmimo-los")
METHODS = ("b1", "p2")
METHOD_LABELS = {
    "b0": "Original LatentWave",
    "b1": "LatentWave with Post-hoc Frozen Early Exits",
    "p2": "Proposed Hierarchical Multi-Scale JEPA",
}
FIGURE_METHOD_LABELS = {
    "b0": "Original LatentWave (full-depth)",
    "b1": "LatentWave with Post-hoc Frozen Early Exits",
    "p2": "Proposed Hierarchical Multi-Scale JEPA",
}
FIGURE_METHOD_SHORT = {
    "b0": "Original\nLatentWave",
    "b1": "LatentWave\nfrozen early exits",
    "p2": "Proposed\nhierarchical JEPA",
}
TASK_LABELS = {
    "rfs": "RF sensing",
    "nrpos-outdoor": "NR positioning",
    "deepmimo-beam": "DeepMIMO beam",
    "deepmimo-los": "DeepMIMO LoS/NLoS",
}
TASK_METRICS = {
    "rfs": ("macro_accuracy", "Macro accuracy", True),
    "nrpos-outdoor": ("mean_distance_error", "Mean distance error (m)", False),
    "deepmimo-beam": ("macro_accuracy", "Macro accuracy", True),
    "deepmimo-los": ("macro_accuracy", "Macro accuracy", True),
}
SOURCE_METRICS = {
    "rfs": ("rfs_pca", "Balanced accuracy (source probe)", True),
    "nrpos-outdoor": ("pos_error", "Mean distance error (m)", False),
    "deepmimo-beam": ("beam_pca", "Balanced accuracy (source probe)", True),
    "deepmimo-los": ("los_pca", "Balanced accuracy (source probe)", True),
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs/phase5_formal")
    parser.add_argument("--source-root", type=Path, default=REPOSITORY_ROOT / "runs/phase5_pilot")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "reports/phase5_primary")
    return parser.parse_args()


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_metric(task: str, value: float) -> str:
    if TASK_METRICS[task][2]:
        return f"{value * 100:.2f}%"
    return f"{value:.3f} m"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_source_reference(source_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = [source_root / "p0_source_probe.json", source_root / "p0_source_probe_los.json"]
    records = [_read(path) for path in paths]
    tasks: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for record in records:
        if record.get("protocol", {}).get("name") != "pinned_latentwave_source_style_l8_probe":
            raise ValueError("source probe does not identify the pinned LatentWave protocol")
        for task, value in record.get("tasks", {}).items():
            if task in tasks:
                raise ValueError(f"duplicate source-probe task: {task}")
            if task not in TASKS:
                raise ValueError(f"unexpected source-probe task: {task}")
            tasks[task] = value
    missing = [task for task in TASKS if task not in tasks]
    if missing:
        raise ValueError(f"source-probe results are incomplete: {missing}")
    first = records[0]
    reference = {
        "method": "b0",
        "label": METHOD_LABELS["b0"],
        "checkpoint": first["checkpoint"],
        "config": first["config"],
        "device": first["device"],
        "protocol": first["protocol"],
        "tasks": tasks,
    }
    rows = []
    for task in TASKS:
        result = tasks[task]
        metric_key, metric_label, higher_is_better = SOURCE_METRICS[task]
        if result["metric_name"] != metric_key:
            raise ValueError(
                f"source metric mismatch for {task}: expected {metric_key}, got {result['metric_name']}"
            )
        rows.append(
            {
                "method": "b0",
                "method_label": METHOD_LABELS["b0"],
                "task": task,
                "task_label": TASK_LABELS[task],
                "metric": metric_key,
                "metric_label": metric_label,
                "value": float(result["value"]),
                "display_value": _fmt_metric(task, float(result["value"])),
                "higher_is_better": higher_is_better,
                "examples": int(result["examples"]),
                "protocol": "EMA target encoder, full-input L8, source-style 80/20 probe",
            }
        )
    return reference, rows


def _load_exit_results(run_root: Path, method: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_method = "p0" if method == "b1" else "p2"
    base = run_root / source_method / "seed_000"
    metrics_path = base / "downstream/exit_metrics.json"
    benchmark_path = base / "benchmarks.json"
    metrics = _read(metrics_path)
    benchmarks = _read(benchmark_path)
    rows = []
    for task in TASKS:
        if task not in metrics.get("tasks", {}):
            raise ValueError(f"{method} downstream metrics missing task: {task}")
        task_record = metrics["tasks"][task]
        metric_key, metric_label, higher_is_better = TASK_METRICS[task]
        if task_record["spec"]["metric_name"] != metric_key:
            raise ValueError(
                f"{method} metric mismatch for {task}: expected {metric_key}, "
                f"got {task_record['spec']['metric_name']}"
            )
        for layer in EXIT_LAYERS:
            exit_record = task_record["exits"][str(layer)]
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "exit_layer": layer,
                    "evaluation": "fixed_exit",
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "value": float(exit_record["test"][metric_key]),
                    "display_value": _fmt_metric(task, float(exit_record["test"][metric_key])),
                    "validation_value": float(exit_record["validation"][metric_key]),
                    "higher_is_better": higher_is_better,
                    "best_epoch": int(exit_record["best_epoch"]),
                    "head_epochs_completed": int(exit_record["epochs_completed"]),
                    "head_stopped_early": bool(exit_record["stopped_early"]),
                    "protocol": "frozen context encoder, independent head, shared 80/10/10 split",
                }
            )
        for policy, value_key, layer_key in (
            ("best_validation", "selected_test_best_validation", "selected_exit_best_validation"),
            ("earliest_within_tolerance", "selected_test_earliest_within_tolerance", "selected_exit_earliest_within_tolerance"),
        ):
            selected = task_record[value_key]
            layer = int(task_record[layer_key])
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "exit_layer": layer,
                    "evaluation": policy,
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "value": float(selected[metric_key]),
                    "display_value": _fmt_metric(task, float(selected[metric_key])),
                    "validation_value": None,
                    "higher_is_better": higher_is_better,
                    "protocol": "validation-selected from the same frozen-head evaluation",
                }
            )

    benchmark_rows = []
    entries = {(entry["task"], int(entry["exit_layer"])): entry for entry in benchmarks.get("entries", [])}
    for task in TASKS:
        for layer in EXIT_LAYERS:
            key = (task, layer)
            if key not in entries:
                raise ValueError(f"{method} benchmark missing task/exit: {task}/L{layer}")
            entry = entries[key]
            benchmark_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "exit_layer": layer,
                    "flops": int(entry["flops"]),
                    "latency_mean_ms": float(entry["latency"]["mean_ms"]),
                    "latency_median_ms": float(entry["latency"]["median_ms"]),
                    "latency_p95_ms": float(entry["latency"]["p95_ms"]),
                    "latency_std_ms": float(entry["latency"]["standard_deviation_ms"]),
                    "sample_shape": entry["sample_shape"],
                    "protocol": "batch-1 synchronized CUDA timing after warm-up",
                }
            )
    for task in TASKS:
        l8 = next(row["latency_mean_ms"] for row in benchmark_rows if row["task"] == task and row["exit_layer"] == 8)
        l8_flops = next(row["flops"] for row in benchmark_rows if row["task"] == task and row["exit_layer"] == 8)
        for row in benchmark_rows:
            if row["task"] == task:
                row["latency_savings_vs_l8"] = 1.0 - row["latency_mean_ms"] / l8
                row["flops_savings_vs_l8"] = 1.0 - row["flops"] / l8_flops

    packet = {
        "method": method,
        "label": METHOD_LABELS[method],
        "pretraining_method": source_method,
        "checkpoint": metrics["checkpoint"],
        "metrics_path": str(metrics_path),
        "benchmark_path": str(benchmark_path),
        "split_manifest": metrics["split_manifest"],
        "metrics_protocol": metrics["protocol"],
        "metric_rows": rows,
        "benchmark_rows": benchmark_rows,
    }
    return packet, rows + benchmark_rows


def _plot_performance(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = {"b1": "#0072B2", "p2": "#D55E00"}
    paths = []
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), squeeze=False)
    for axis, task in zip(axes.flat, TASKS):
        metric_key, metric_label, higher_is_better = TASK_METRICS[task]
        for method in METHODS:
            values = [
                next(row["value"] for row in rows if row["method"] == method and row["task"] == task and row.get("exit_layer") == layer and row["evaluation"] == "fixed_exit")
                for layer in EXIT_LAYERS
            ]
            axis.plot(
                EXIT_LAYERS,
                values,
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                color=colors[method],
                label=FIGURE_METHOD_LABELS[method],
            )
        axis.set_title(TASK_LABELS[task], fontsize=11, weight="bold")
        axis.set_xlabel("Encoder exit layer")
        axis.set_ylabel(metric_label)
        axis.set_xticks(EXIT_LAYERS)
        axis.grid(axis="y", alpha=0.25, linewidth=0.8)
        if higher_is_better:
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        else:
            axis.text(
                0.98,
                0.05,
                "lower is better",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#555555",
            )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Frozen-head downstream performance by encoder depth", fontsize=14, weight="bold", y=1.045)
    fig.text(0.5, -0.015, "Single-seed pilot; test metrics are reported after validation-only head/exit selection.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"early_exit_performance.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _plot_selected(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = {"b1": "#0072B2", "p2": "#D55E00"}
    hatches = {"best_validation": "", "earliest_within_tolerance": "//"}
    labels = {"best_validation": "Best validation", "earliest_within_tolerance": "Earliest within 5%"}
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), squeeze=False)
    for axis, task in zip(axes.flat, TASKS):
        _, metric_label, higher_is_better = TASK_METRICS[task]
        positions = []
        heights = []
        bar_colors = []
        bar_hatches = []
        bar_labels = []
        for method in METHODS:
            for policy in ("best_validation", "earliest_within_tolerance"):
                row = next(row for row in rows if row["method"] == method and row["task"] == task and row["evaluation"] == policy)
                positions.append(len(positions))
                heights.append(row["value"])
                bar_colors.append(colors[method])
                bar_hatches.append(hatches[policy])
                bar_labels.append(f"{FIGURE_METHOD_SHORT[method]}\nL{row['exit_layer']}")
        bars = axis.bar(positions, heights, color=bar_colors, edgecolor="#222222", linewidth=0.7)
        for bar, hatch in zip(bars, bar_hatches):
            bar.set_hatch(hatch)
        axis.set_xticks(positions, bar_labels)
        axis.set_title(TASK_LABELS[task], fontsize=11, weight="bold")
        axis.set_ylabel(metric_label)
        axis.grid(axis="y", alpha=0.25, linewidth=0.8)
        if higher_is_better:
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        else:
            axis.text(0.98, 0.05, "lower is better", transform=axis.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")
    method_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[method], label=FIGURE_METHOD_LABELS[method])
        for method in METHODS
    ]
    policy_handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#222222", hatch=hatches[policy], label=labels[policy]) for policy in hatches]
    fig.legend(handles=method_handles + policy_handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Validation-selected exits and held-out test performance", fontsize=14, weight="bold", y=1.045)
    fig.text(0.5, -0.015, "Exit labels show the selected encoder depth; selection never uses the test split.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"selected_exit_performance.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _plot_efficiency(output_dir: Path, benchmark_rows: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"b1": "#0072B2", "p2": "#D55E00"}
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.3), squeeze=False)
    for column, task in enumerate(TASKS):
        for method in METHODS:
            task_rows = [row for row in benchmark_rows if row["method"] == method and row["task"] == task]
            task_rows.sort(key=lambda row: row["exit_layer"])
            axes[0, column].plot(
                [row["exit_layer"] for row in task_rows],
                [row["flops"] / 1e9 for row in task_rows],
                marker="o",
                linewidth=2,
                color=colors[method],
                linestyle="-" if method == "b1" else "--",
                label=FIGURE_METHOD_LABELS[method],
            )
            axes[1, column].plot(
                [row["exit_layer"] for row in task_rows],
                [row["latency_mean_ms"] for row in task_rows],
                marker="o",
                linewidth=2,
                color=colors[method],
                label=FIGURE_METHOD_LABELS[method],
            )
        axes[0, column].set_title(TASK_LABELS[task], fontsize=10, weight="bold")
        axes[1, column].set_title(TASK_LABELS[task], fontsize=10, weight="bold")
        for axis in (axes[0, column], axes[1, column]):
            axis.set_xticks(EXIT_LAYERS)
            axis.set_xlabel("Exit layer")
            axis.grid(axis="y", alpha=0.25, linewidth=0.8)
        axes[0, column].set_ylabel("FLOPs (billions)" if column == 0 else "")
        axes[1, column].set_ylabel("Latency (ms)" if column == 0 else "")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.text(0.5, 0.03, "Batch-1 synchronized CUDA timing; lower is better.", ha="center", fontsize=9)
    fig.suptitle("Inference compute and latency by exit depth", fontsize=14, weight="bold", y=1.065)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96), h_pad=2.2, w_pad=1.3)
    paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"inference_efficiency.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _plot_source_reference(output_dir: Path, source_rows: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    classification = [row for row in source_rows if row["higher_is_better"]]
    position = next(row for row in source_rows if not row["higher_is_better"])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), gridspec_kw={"width_ratios": [3, 1]})
    axes[0].bar(
        [row["task_label"] for row in classification],
        [row["value"] for row in classification],
        color="#009E73",
        edgecolor="#222222",
        linewidth=0.7,
    )
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].set_title("Classification", weight="bold")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar([position["task_label"]], [position["value"]], color="#009E73", edgecolor="#222222", linewidth=0.7)
    axes[1].set_ylabel("Mean distance error (m)")
    axes[1].set_title("Positioning", weight="bold")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    fig.suptitle("Original LatentWave full-depth reference", fontsize=14, weight="bold", y=1.04)
    fig.text(0.5, -0.025, "EMA target encoder at L8; source-style 80/20 probes. Not directly comparable to frozen-head metrics.", ha="center", fontsize=8.5)
    fig.tight_layout()
    paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"b0_source_reference.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _build_headline_rows(
    source_rows: list[dict[str, Any]],
    comparable_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one deployment-oriented row per method and downstream task."""

    rows = []
    for task in TASKS:
        source = next(row for row in source_rows if row["task"] == task)
        source_benchmark = next(
            row
            for row in benchmark_rows
            if row["method"] == "b0" and row["task"] == task and row["exit_layer"] == 8
        )
        rows.append(
            {
                "method": "b0",
                "method_label": METHOD_LABELS["b0"],
                "task": task,
                "task_label": TASK_LABELS[task],
                "metric": source["metric"],
                "metric_label": source["metric_label"],
                "value": source["value"],
                "display_value": source["display_value"],
                "exit_layer": 8,
                "selection": "full-depth source reference",
                "protocol": "EMA target encoder, source-style 80/20 probe",
                "flops": source_benchmark["flops"],
                "latency_mean_ms": source_benchmark["latency_mean_ms"],
                "latency_p95_ms": source_benchmark["latency_p95_ms"],
            }
        )
        for method in METHODS:
            selected = next(
                row
                for row in comparable_rows
                if row["method"] == method
                and row["task"] == task
                and row["evaluation"] == "earliest_within_tolerance"
            )
            benchmark = next(
                row
                for row in benchmark_rows
                if row["method"] == method
                and row["task"] == task
                and row["exit_layer"] == selected["exit_layer"]
            )
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "metric": selected["metric"],
                    "metric_label": selected["metric_label"],
                    "value": selected["value"],
                    "display_value": selected["display_value"],
                    "exit_layer": selected["exit_layer"],
                    "selection": "earliest within 5% of validation best",
                    "protocol": "frozen context encoder, shared 80/10/10 split",
                    "flops": benchmark["flops"],
                    "latency_mean_ms": benchmark["latency_mean_ms"],
                    "latency_p95_ms": benchmark["latency_p95_ms"],
                }
            )
    return rows


def _plot_all_method_performance(
    output_dir: Path,
    source_rows: list[dict[str, Any]],
    comparable_rows: list[dict[str, Any]],
) -> list[str]:
    """Plot all three approaches in one downstream-performance figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = {"b0": "#009E73", "b1": "#0072B2", "p2": "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8), squeeze=False)
    for axis, task in zip(axes.flat, TASKS):
        _, _, higher_is_better = TASK_METRICS[task]
        source = next(row for row in source_rows if row["task"] == task)
        axis.plot(
            [8],
            [source["value"]],
            marker="D",
            markersize=7,
            linestyle="None",
            color=colors["b0"],
            label=FIGURE_METHOD_LABELS["b0"],
        )
        for method in METHODS:
            values = [
                next(
                    row["value"]
                    for row in comparable_rows
                    if row["method"] == method
                    and row["task"] == task
                    and row["evaluation"] == "fixed_exit"
                    and row["exit_layer"] == layer
                )
                for layer in EXIT_LAYERS
            ]
            axis.plot(
                EXIT_LAYERS,
                values,
                marker="o",
                linewidth=2.1,
                markersize=5,
                color=colors[method],
                label=FIGURE_METHOD_LABELS[method],
            )
        axis.set_title(TASK_LABELS[task], fontsize=11, weight="bold")
        axis.set_xlabel("Encoder exit layer")
        axis.set_ylabel("Accuracy" if higher_is_better else "Mean distance error (m)")
        axis.set_xticks(EXIT_LAYERS)
        axis.grid(axis="y", alpha=0.25, linewidth=0.8)
        if higher_is_better:
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        else:
            axis.text(0.98, 0.05, "lower is better", transform=axis.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.suptitle("Downstream performance across the three approaches", fontsize=14, weight="bold", y=1.045)
    fig.text(
        0.5,
        -0.012,
        "Diamond: Original LatentWave source-style full-depth reference. Circles: frozen-head exits. The full-depth reference uses a separate 80/20 probe protocol.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.93))
    paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"all_methods_downstream_performance.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _plot_all_method_selected(output_dir: Path, headline_rows: list[dict[str, Any]]) -> list[str]:
    """Plot the headline deployment choice for all three approaches."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = {"b0": "#009E73", "b1": "#0072B2", "p2": "#D55E00"}
    methods = ("b0", "b1", "p2")
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8), squeeze=False)
    for axis, task in zip(axes.flat, TASKS):
        _, _, higher_is_better = TASK_METRICS[task]
        task_rows = [row for row in headline_rows if row["task"] == task]
        bars = axis.bar(
            range(len(methods)),
            [next(row["value"] for row in task_rows if row["method"] == method) for method in methods],
            color=[colors[method] for method in methods],
            edgecolor="#222222",
            linewidth=0.7,
        )
        for bar, method in zip(bars, methods):
            row = next(row for row in task_rows if row["method"] == method)
            bar.set_label(FIGURE_METHOD_LABELS[method])
            axis.annotate(
                f"L{row['exit_layer']}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3 if higher_is_better else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if higher_is_better else "top",
                fontsize=8,
            )
        axis.set_xticks(range(len(methods)), [FIGURE_METHOD_SHORT[method] for method in methods])
        axis.set_title(TASK_LABELS[task], fontsize=11, weight="bold")
        axis.set_ylabel("Accuracy" if higher_is_better else "Mean distance error (m)")
        axis.grid(axis="y", alpha=0.25, linewidth=0.8)
        if higher_is_better:
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        else:
            axis.text(0.98, 0.05, "lower is better", transform=axis.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[method], label=FIGURE_METHOD_LABELS[method]) for method in methods]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.suptitle("Headline downstream result at each approach's deployment choice", fontsize=14, weight="bold", y=1.045)
    fig.text(0.5, -0.012, "Layer labels show full-depth L8 for Original LatentWave and the earliest exit within 5% of validation-best for the two exit methods.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.025, 1, 0.93))
    paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"all_methods_selected_performance.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _plot_all_method_inference(output_dir: Path, headline_rows: list[dict[str, Any]]) -> list[str]:
    """Plot one inference-cost comparison containing all three approaches."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"b0": "#009E73", "b1": "#0072B2", "p2": "#D55E00"}
    methods = ("b0", "b1", "p2")
    fig, axes = plt.subplots(2, 4, figsize=(16.4, 7.1), squeeze=False)
    for column, task in enumerate(TASKS):
        task_rows = [row for row in headline_rows if row["task"] == task]
        labels = [f"{FIGURE_METHOD_SHORT[method]}\nL{next(row['exit_layer'] for row in task_rows if row['method'] == method)}" for method in methods]
        for axis, measure, ylabel in (
            (axes[0, column], "flops", "FLOPs (billions)"),
            (axes[1, column], "latency_mean_ms", "Latency (ms)"),
        ):
            values = []
            for method in methods:
                row = next(row for row in task_rows if row["method"] == method)
                if method == "b0":
                    b0_benchmark = row.get("flops") if measure == "flops" else row.get("latency_mean_ms")
                    values.append(float(b0_benchmark))
                else:
                    values.append(float(row[measure]))
            if measure == "flops":
                values = [value / 1e9 for value in values]
            bars = axis.bar(range(len(methods)), values, color=[colors[method] for method in methods], edgecolor="#222222", linewidth=0.7)
            axis.set_xticks(range(len(methods)), labels)
            axis.tick_params(axis="x", labelsize=7.2)
            axis.set_title(TASK_LABELS[task], fontsize=10, weight="bold")
            axis.set_ylabel(ylabel if column == 0 else "")
            axis.grid(axis="y", alpha=0.25, linewidth=0.8)
            for bar, value in zip(bars, values):
                axis.annotate(
                    f"{value:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[method], label=FIGURE_METHOD_LABELS[method]) for method in methods]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=9)
    fig.suptitle("Inference cost comparison at each approach's deployment choice", fontsize=14, weight="bold", y=1.065)
    fig.text(0.5, 0.03, "Batch-1 synchronized CUDA timing on the RTX 5070 Ti; lower is better. Layer labels are printed below each method.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96), h_pad=2.2, w_pad=1.3)
    paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"all_methods_inference_cost.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _build_report(
    output_dir: Path,
    source_reference: dict[str, Any],
    source_rows: list[dict[str, Any]],
    b1_packet: dict[str, Any],
    p2_packet: dict[str, Any],
    headline_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    figure_paths: list[str],
) -> None:
    comparable_rows = b1_packet["metric_rows"] + p2_packet["metric_rows"]
    detailed_benchmark_rows = [row for row in benchmark_rows if row["method"] in METHODS]
    fixed_rows = [row for row in comparable_rows if row["evaluation"] == "fixed_exit"]
    selected_rows = [row for row in comparable_rows if row["evaluation"] != "fixed_exit"]
    fixed_table = []
    for row in fixed_rows:
        fixed_table.append(
            [
                row["method_label"],
                row["task_label"],
                f"L{row['exit_layer']}",
                row["metric_label"],
                row["display_value"],
                _fmt_metric(row["task"], row["validation_value"]),
                row["best_epoch"],
            ]
        )
    selected_table = []
    for row in selected_rows:
        selected_table.append(
            [
                row["method_label"],
                row["task_label"],
                row["evaluation"].replace("_", " "),
                f"L{row['exit_layer']}",
                row["display_value"],
            ]
        )
    source_table = [[row["task_label"], row["metric_label"], row["display_value"], row["examples"]] for row in source_rows]
    headline_table = []
    for row in headline_rows:
        headline_table.append(
            [
                row["method_label"],
                row["task_label"],
                f"L{row['exit_layer']}",
                row["selection"],
                row["display_value"],
            ]
        )
    headline_inference_table = []
    for row in headline_rows:
        headline_inference_table.append(
            [
                row["method_label"],
                row["task_label"],
                f"L{row['exit_layer']}",
                f"{row['flops'] / 1e9:.3f}",
                f"{row['latency_mean_ms']:.4f}",
                f"{row['latency_p95_ms']:.4f}",
            ]
        )
    compute_table = []
    for row in detailed_benchmark_rows:
        compute_table.append(
            [
                row["method_label"],
                row["task_label"],
                f"L{row['exit_layer']}",
                f"{row['flops'] / 1e9:.3f}",
                f"{row['latency_mean_ms']:.4f}",
                f"{row['latency_p95_ms']:.4f}",
                f"{row['flops_savings_vs_l8'] * 100:.1f}%",
                f"{row['latency_savings_vs_l8'] * 100:.1f}%",
            ]
        )
    report = f"""# Phase 5 primary single-seed pilot packet

This directory contains the submission-oriented result tables and figures for
the completed single-seed pilot. It is an aggregation of recorded
artifacts, not a statistical claim. The report was generated from repository
revision `{_git_revision()}`.

## Scope and comparability

| Approach | Pretraining | Downstream representation | Role |
|---|---|---|---|
| Original LatentWave | Original LatentWave training | EMA target encoder, full-input L8 | Source-faithful full-depth reference |
| LatentWave with Post-hoc Frozen Early Exits | Reuses the Original LatentWave checkpoint | Frozen context encoder at L2/L4/L6/L8 | Early-exit baseline |
| Proposed Hierarchical Multi-Scale JEPA | Separately pretrained hierarchical JEPA | Frozen context encoder at L2/L4/L6/L8 | Proposed method |

The Original LatentWave reference uses the pinned source-style 80/20
scikit-learn probe. The two exit approaches use independent frozen heads, a
shared persisted 80/10/10 split, and validation-only exit selection. The
single-figure all-approach plots show this protocol distinction explicitly;
the Original LatentWave point is a full-depth source reference, not a claim
that its metric was obtained with the frozen-head split.

All values are single-seed pilot point estimates.

## All-approach headline downstream comparison

The deployment rows use the full-depth Original LatentWave reference and the
earliest exit within 5% of validation-best for the two exit approaches.

{_markdown_table(['Approach', 'Task', 'Exit', 'Selection', 'Test value'], headline_table)}

## All-approach headline inference comparison

{_markdown_table(['Approach', 'Task', 'Exit', 'FLOPs (B)', 'Mean ms', 'P95 ms'], headline_inference_table)}

## Original LatentWave source-faithful reference

{_markdown_table(['Task', 'Metric', 'Test value', 'Examples'], source_table)}

## Detailed frozen-exit downstream metrics

Test values are shown after each head's validation-selected checkpoint. The
validation column is included for auditability; it is not a test result.

{_markdown_table(['Method', 'Task', 'Exit', 'Metric', 'Test value', 'Validation value', 'Head epoch'], fixed_table)}

## Detailed selected exits

"Earliest within 5%" is selected from validation only. For positioning,
lower mean distance error is better; for classification, higher macro
accuracy is better.

{_markdown_table(['Method', 'Task', 'Policy', 'Exit', 'Test value'], selected_table)}

## Detailed batch-1 inference cost

Latency is the synchronized CUDA mean/p95 measurement after warm-up on the
RTX 5070 Ti. Savings are relative to the same method/task at L8.

{_markdown_table(['Method', 'Task', 'Exit', 'FLOPs (B)', 'Mean ms', 'P95 ms', 'FLOPs saving', 'Latency saving'], compute_table)}

## Figures

The PNG files are 300-DPI raster exports and the PDF files are vector exports.

{chr(10).join(f'- `{Path(path).name}`' for path in figure_paths)}

## Exclusions

Diagnostic early-stopped pretraining runs, the same-scale auxiliary JEPA
ablation, OOD transfer results, extra seeds, and alternate masking regimes are
excluded.
They require separate approval and are not part of this primary pilot packet.
"""
    (output_dir / "README.md").write_text(report)

    (output_dir / "table_b0_source_reference.md").write_text(
        "# Original LatentWave source-faithful reference\n\n" + _markdown_table(["Task", "Metric", "Test value", "Examples"], source_table) + "\n"
    )
    (output_dir / "table_primary_fixed_exits.md").write_text(
        "# Detailed frozen-exit downstream metrics\n\n" + _markdown_table(["Approach", "Task", "Exit", "Metric", "Test value", "Validation value", "Head epoch"], fixed_table) + "\n"
    )
    (output_dir / "table_selected_exits.md").write_text(
        "# Detailed selected exits\n\n" + _markdown_table(["Approach", "Task", "Policy", "Exit", "Test value"], selected_table) + "\n"
    )
    (output_dir / "table_inference_cost.md").write_text(
        "# Detailed batch-1 inference cost\n\n" + _markdown_table(["Approach", "Task", "Exit", "FLOPs (B)", "Mean ms", "P95 ms", "FLOPs saving", "Latency saving"], compute_table) + "\n"
    )
    (output_dir / "table_all_methods_downstream.md").write_text(
        "# All-approach headline downstream comparison\n\n"
        + _markdown_table(["Approach", "Task", "Exit", "Selection", "Test value"], headline_table)
        + "\n"
    )
    (output_dir / "table_all_methods_inference.md").write_text(
        "# All-approach headline inference comparison\n\n"
        + _markdown_table(["Approach", "Task", "Exit", "FLOPs (B)", "Mean ms", "P95 ms"], headline_inference_table)
        + "\n"
    )

    import csv

    with (output_dir / "table_primary_fixed_exits.csv").open("w", newline="") as handle:
        fields = ["method", "task", "exit_layer", "metric", "value", "validation_value", "best_epoch", "head_epochs_completed", "head_stopped_early"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in fixed_rows)
    with (output_dir / "table_selected_exits.csv").open("w", newline="") as handle:
        fields = ["method", "task", "evaluation", "exit_layer", "metric", "value", "higher_is_better"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in selected_rows)
    with (output_dir / "table_b0_source_reference.csv").open("w", newline="") as handle:
        fields = ["method", "task", "metric", "value", "examples", "protocol"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in source_rows)
    with (output_dir / "table_inference_cost.csv").open("w", newline="") as handle:
        fields = ["method", "task", "exit_layer", "flops", "latency_mean_ms", "latency_median_ms", "latency_p95_ms", "latency_std_ms", "flops_savings_vs_l8", "latency_savings_vs_l8"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in detailed_benchmark_rows)
    with (output_dir / "table_all_methods_downstream.csv").open("w", newline="") as handle:
        fields = ["method", "method_label", "task", "exit_layer", "selection", "metric", "value", "protocol"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in headline_rows)
    with (output_dir / "table_all_methods_inference.csv").open("w", newline="") as handle:
        fields = ["method", "method_label", "task", "exit_layer", "flops", "latency_mean_ms", "latency_p95_ms", "protocol"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in headline_rows)


def main() -> int:
    args = _arguments()
    run_root = _repo_path(args.run_root)
    source_root = _repo_path(args.source_root)
    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(output_dir / ".matplotlib")

    source_reference, source_rows = _load_source_reference(source_root)
    b1_packet, _ = _load_exit_results(run_root, "b1")
    p2_packet, _ = _load_exit_results(run_root, "p2")
    figure_paths = []
    comparable_rows = b1_packet["metric_rows"] + p2_packet["metric_rows"]
    b0_benchmark_rows = []
    for row in b1_packet["benchmark_rows"]:
        if row["exit_layer"] != 8:
            continue
        b0_row = dict(row)
        b0_row["method"] = "b0"
        b0_row["method_label"] = METHOD_LABELS["b0"]
        b0_row["protocol"] = "Original LatentWave full-depth batch-1 benchmark"
        b0_benchmark_rows.append(b0_row)
    benchmark_rows = b0_benchmark_rows + b1_packet["benchmark_rows"] + p2_packet["benchmark_rows"]
    headline_rows = _build_headline_rows(source_rows, comparable_rows, benchmark_rows)
    figure_paths.extend(_plot_all_method_performance(output_dir, source_rows, comparable_rows))
    figure_paths.extend(_plot_all_method_selected(output_dir, headline_rows))
    figure_paths.extend(_plot_all_method_inference(output_dir, headline_rows))
    figure_paths.extend(_plot_performance(output_dir, comparable_rows))
    figure_paths.extend(_plot_selected(output_dir, comparable_rows))
    figure_paths.extend(_plot_efficiency(output_dir, [row for row in benchmark_rows if row["method"] in METHODS]))
    figure_paths.extend(_plot_source_reference(output_dir, source_rows))
    _build_report(output_dir, source_reference, source_rows, b1_packet, p2_packet, headline_rows, benchmark_rows, figure_paths)
    manifest = {
        "report": "phase5_primary_single_seed",
        "git_revision": _git_revision(),
        "run_root": str(run_root),
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "methods": [METHOD_LABELS[method] for method in ("b0", "b1", "p2")],
        "seed": 0,
        "masking": "multi_block/span_time",
        "included_inputs": {
            "b0_source_probe": [str(source_root / "p0_source_probe.json"), str(source_root / "p0_source_probe_los.json")],
            "b1_metrics": str(run_root / "p0/seed_000/downstream/exit_metrics.json"),
            "b1_benchmark": str(run_root / "p0/seed_000/benchmarks.json"),
            "p2_metrics": str(run_root / "p2/seed_000/downstream/exit_metrics.json"),
            "p2_benchmark": str(run_root / "p2/seed_000/benchmarks.json"),
        },
        "excluded": ["diagnostic early-stopped runs", "same-scale auxiliary JEPA control ablation", "OOD", "seeds 1 and 2", "alternate masking regimes"],
        "figures": [Path(path).name for path in figure_paths],
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "figures": len(figure_paths), "methods": manifest["methods"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
