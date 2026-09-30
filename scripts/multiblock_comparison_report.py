#!/usr/bin/env python3
"""Render an immutable multi-block P1/P2/P2-Reverse comparison packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("rfs", "jamming", "human-activity", "interference", "nrpos-outdoor", "deepmimo-beam")
EXIT_LAYERS = (2, 4, 6, 8)
NEW_METHODS = ("p1", "p2", "p2_reverse")
METHOD_LABELS = {
    "b0": "B0 Original LatentWave (L8)",
    "p0": "P0 Post-hoc Frozen Exits",
    "p1": "P1 Same-scale Intermediate Supervision",
    "p2": "P2 Fine-to-Coarse JEPA",
    "p2_reverse": "P2-Reverse Coarse-to-Fine JEPA",
}
COLORS = {"b0": "#009E73", "p0": "#0072B2", "p1": "#CC79A7", "p2": "#D55E00", "p2_reverse": "#E69F00"}
TASK_LABELS = {
    "rfs": "RF sensing", "jamming": "Jamming", "human-activity": "Human activity",
    "interference": "Interference", "nrpos-outdoor": "NR positioning", "deepmimo-beam": "DeepMIMO beam",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"expected finite number, got {value!r}")
    return result


def _task_metric(task: dict[str, Any]) -> tuple[str, bool]:
    spec = task.get("spec", {})
    return str(spec["metric_name"]), bool(spec["higher_is_better"])


def _load_new_downstream(path: Path, method: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    packet = _read_json(path)
    if packet.get("method") != method or packet.get("dataset_suite") != "primary":
        raise ValueError(f"{path} is not a completed {method} primary exit packet")
    if not packet.get("protocol", {}).get("test_evaluation"):
        raise ValueError(f"{path} lacks post-selection held-out test results")
    tasks = packet.get("tasks", {})
    if set(tasks) != set(TASKS):
        raise ValueError(f"{path} must contain exactly {TASKS}, found {tuple(sorted(tasks))}")
    split_manifest = str(packet.get("split_manifest", ""))
    if not split_manifest:
        raise ValueError(f"{path} does not record a split manifest")
    fixed, selected, headline = [], [], []
    for task_name in TASKS:
        task = tasks[task_name]
        metric, higher = _task_metric(task)
        exits = task.get("exits", {})
        if set(exits) != {str(layer) for layer in EXIT_LAYERS}:
            raise ValueError(f"{path} lacks one or more exits for {task_name}")
        for layer in EXIT_LAYERS:
            record = exits[str(layer)]
            fixed.append({
                "method": method, "task": task_name, "exit_layer": layer, "metric": metric,
                "value": _float(record["test"][metric]),
                "validation_value": _float(record["validation"][metric]),
                "best_epoch": record.get("best_epoch"),
                "head_epochs_completed": record.get("epochs_completed"),
                "head_stopped_early": record.get("stopped_early"),
            })
        for policy, exit_key, test_key in (
            ("best_validation", "selected_exit_best_validation", "selected_test_best_validation"),
            ("earliest_within_tolerance", "selected_exit_earliest_within_tolerance", "selected_test_earliest_within_tolerance"),
        ):
            selected.append({
                "method": method, "task": task_name, "evaluation": policy,
                "exit_layer": int(task[exit_key]), "metric": metric,
                "value": _float(task[test_key][metric]), "higher_is_better": higher,
            })
        row = next(item for item in selected if item["task"] == task_name and item["evaluation"] == "earliest_within_tolerance")
        headline.append({
            "method": method, "method_label": METHOD_LABELS[method], "task": task_name,
            "exit_layer": row["exit_layer"], "selection": "earliest within 5% of validation best",
            "metric": metric, "value": row["value"],
            "protocol": "fresh frozen-context heads; seed-2025 validation-selected split",
        })
    return fixed, selected, headline, split_manifest


def _load_benchmark(path: Path, method: str) -> list[dict[str, Any]]:
    packet = _read_json(path)
    entries = packet.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{path} lacks benchmark entries")
    if len(entries) != len(TASKS) * len(EXIT_LAYERS):
        raise ValueError(f"{path} must contain exactly 24 benchmark entries")
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in entries:
        key = (str(entry["task"]), int(entry["exit_layer"]))
        lookup[key] = entry
    required = {(task, layer) for task in TASKS for layer in EXIT_LAYERS}
    if set(lookup) != required:
        raise ValueError(f"{path} must contain exactly 24 six-task fixed-exit entries")
    rows = []
    for task in TASKS:
        l8 = lookup[(task, 8)]
        l8_flops, l8_latency = _float(l8["flops"]), _float(l8["latency"]["mean_ms"])
        for layer in EXIT_LAYERS:
            entry = lookup[(task, layer)]
            latency = entry["latency"]
            flops, mean = _float(entry["flops"]), _float(latency["mean_ms"])
            rows.append({
                "method": method, "task": task, "exit_layer": layer, "flops": int(flops),
                "latency_mean_ms": mean, "latency_median_ms": _float(latency["median_ms"]),
                "latency_p95_ms": _float(latency["p95_ms"]), "latency_std_ms": _float(latency["standard_deviation_ms"]),
                "flops_savings_vs_l8": 1.0 - flops / l8_flops,
                "latency_savings_vs_l8": 1.0 - mean / l8_latency,
            })
    return rows


def _baseline_rows(report: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    files = {
        "b0": report / "table_b0_context_l8_reference.csv",
        "fixed": report / "table_primary_fixed_exits.csv",
        "selected": report / "table_selected_exits.csv",
        "headline": report / "table_all_methods_downstream.csv",
        "cost": report / "table_inference_cost.csv",
        "inference": report / "table_all_methods_inference.csv",
        "manifest": report / "manifest.json",
    }
    hashes = {name: _sha256(path) for name, path in files.items()}
    b0 = [{**row, "method": "b0", "value": _float(row["value"])} for row in _read_csv(files["b0"])]
    fixed = [{**row, "method": "p0", "exit_layer": int(row["exit_layer"]), "value": _float(row["value"]), "validation_value": _float(row["validation_value"])} for row in _read_csv(files["fixed"]) if row["method"] == "b1"]
    selected = [{**row, "method": "p0", "exit_layer": int(row["exit_layer"]), "value": _float(row["value"]), "higher_is_better": row["higher_is_better"] == "True"} for row in _read_csv(files["selected"]) if row["method"] == "b1"]
    headline = [{**row, "method": "b0" if row["method"] == "b0" else "p0", "method_label": METHOD_LABELS["b0" if row["method"] == "b0" else "p0"], "exit_layer": int(row["exit_layer"]), "value": _float(row["value"])} for row in _read_csv(files["headline"]) if row["method"] in {"b0", "b1"}]
    cost = [{
        **row, "method": "p0", "exit_layer": int(row["exit_layer"]), "flops": int(row["flops"]),
        "latency_mean_ms": _float(row["latency_mean_ms"]), "latency_median_ms": _float(row["latency_median_ms"]),
        "latency_p95_ms": _float(row["latency_p95_ms"]), "latency_std_ms": _float(row["latency_std_ms"]),
        "flops_savings_vs_l8": _float(row["flops_savings_vs_l8"]),
        "latency_savings_vs_l8": _float(row["latency_savings_vs_l8"]),
    } for row in _read_csv(files["cost"]) if row["method"] == "b1"]
    inference = [{**row, "method": "b0" if row["method"] == "b0" else "p0", "method_label": METHOD_LABELS["b0" if row["method"] == "b0" else "p0"], "exit_layer": int(row["exit_layer"]), "flops": int(row["flops"]), "latency_mean_ms": _float(row["latency_mean_ms"]), "latency_p95_ms": _float(row["latency_p95_ms"])} for row in _read_csv(files["inference"]) if row["method"] in {"b0", "b1"}]
    if {row["task"] for row in b0} != set(TASKS) or {row["task"] for row in fixed} != set(TASKS):
        raise ValueError("historical baseline packet does not contain the six-task m-block suite")
    return b0, fixed, selected, headline, cost, {**hashes, "manifest_contents": json.dumps(_read_json(files["manifest"]), sort_keys=True)}


def _markdown_table(headers: tuple[str, ...], rows: list[dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")).replace("|", "\\|") for header in headers) + " |")
    return "\n".join(lines) + "\n"


def _plots(output: Path, fixed: list[dict[str, Any]], selected: list[dict[str, Any]], headline: list[dict[str, Any]], cost: list[dict[str, Any]], inference: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    def save(figure, name: str) -> None:
        figure.tight_layout()
        figure.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
        figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
        plt.close(figure)

    def axes(title: str, value_key: str, records: list[dict[str, Any]], methods: tuple[str, ...], name: str, y_label: str | None = None) -> None:
        fig, grid = plt.subplots(2, 3, figsize=(15, 7.5), squeeze=False)
        for axis, task in zip(grid.flat, TASKS):
            higher = next(row["metric"] != "mean_distance_error" for row in fixed if row["task"] == task)
            for method in methods:
                values = sorted((row for row in records if row["method"] == method and row["task"] == task), key=lambda row: row["exit_layer"])
                if not values:
                    continue
                axis.plot([row["exit_layer"] for row in values], [row[value_key] for row in values], marker="o", label=METHOD_LABELS[method], color=COLORS[method])
            axis.set_title(TASK_LABELS[task]); axis.set_xticks(EXIT_LAYERS); axis.grid(axis="y", alpha=.25)
            axis.set_xlabel("Encoder exit layer")
            axis.set_ylabel(y_label or ("Macro accuracy" if higher else "Mean distance error (m)"))
            if higher and y_label is None:
                axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        handles, labels = grid.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
        fig.suptitle(title, y=1.02)
        save(fig, name)

    b0_fixed = [{"method": "b0", "task": row["task"], "exit_layer": 8, "metric": row["metric"], "value": row["value"]} for row in headline if row["method"] == "b0"]
    axes("Frozen-head downstream performance by encoder depth", "value", b0_fixed + fixed, ("b0", "p0", *NEW_METHODS), "early_exit_performance")
    axes("Batch-1 frozen-exit inference latency", "latency_mean_ms", inference + cost, ("b0", "p0", *NEW_METHODS), "all_methods_inference_cost", "Latency (ms)")

    fig, grid = plt.subplots(2, 3, figsize=(15, 7.5), squeeze=False)
    methods = ("b0", "p0", *NEW_METHODS)
    for axis, task in zip(grid.flat, TASKS):
        rows = [row for row in headline if row["task"] == task]
        axis.bar(range(len(methods)), [next(row["value"] for row in rows if row["method"] == method) for method in methods], color=[COLORS[method] for method in methods])
        axis.set_xticks(range(len(methods)), [method.replace("_", "\n") for method in methods], fontsize=8)
        if rows[0]["metric"] != "mean_distance_error":
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_title(TASK_LABELS[task]); axis.grid(axis="y", alpha=.25)
    fig.suptitle("Validation-selected held-out downstream comparison", y=1.02)
    save(fig, "selected_exit_performance")


    fig, grid = plt.subplots(2, 3, figsize=(15, 7.5), squeeze=False)
    methods = ("p0", *NEW_METHODS)
    for axis, task in zip(grid.flat, TASKS):
        for method in methods:
            rows = sorted((row for row in cost if row["method"] == method and row["task"] == task), key=lambda row: row["exit_layer"])
            axis.plot([row["exit_layer"] for row in rows], [row["flops_savings_vs_l8"] for row in rows], marker="o", label=METHOD_LABELS[method], color=COLORS[method])
        axis.set_title(TASK_LABELS[task]); axis.set_xticks(EXIT_LAYERS); axis.grid(axis="y", alpha=.25)
        axis.set_xlabel("Encoder exit layer"); axis.set_ylabel("FLOPs saving vs L8")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    handles, labels = grid.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("Early-exit FLOPs efficiency", y=1.02)
    save(fig, "inference_efficiency")

def build_packet(baseline_report: Path, run_root: Path, output_dir: Path, *, render_figures: bool = True) -> dict[str, Any]:
    baseline_report, run_root, output_dir = Path(baseline_report), Path(run_root), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory already contains artifacts: {output_dir}")
    source_before = {path: _sha256(path) for path in baseline_report.glob("*") if path.is_file()}
    b0, fixed, selected, headline, cost, source = _baseline_rows(baseline_report)
    split_manifests = set()
    for method in NEW_METHODS:
        base = run_root / method / "seed_000"
        new_fixed, new_selected, new_headline, split = _load_new_downstream(base / "downstream/exit_metrics.json", method)
        fixed.extend(new_fixed); selected.extend(new_selected); headline.extend(new_headline); split_manifests.add(split)
        cost.extend(_load_benchmark(base / "benchmarks.json", method))
    if len(split_manifests) != 1:
        raise ValueError(f"fresh methods must share one split manifest, got {sorted(split_manifests)}")
    for task in TASKS:
        for layer in EXIT_LAYERS:
            fresh_flops = {row["flops"] for row in cost if row["task"] == task and row["exit_layer"] == layer and row["method"] in NEW_METHODS}
            if len(fresh_flops) != 1:
                raise ValueError(f"P1/P2/P2-Reverse FLOPs must match for {task} L{layer}: {sorted(fresh_flops)}")
    for row in cost:
        row["method_label"] = METHOD_LABELS[row["method"]]
    inference = [row for row in _read_csv(baseline_report / "table_all_methods_inference.csv") if row["method"] in {"b0", "b1"}]
    inference = [{**row, "method": "b0" if row["method"] == "b0" else "p0", "method_label": METHOD_LABELS["b0" if row["method"] == "b0" else "p0"], "exit_layer": int(row["exit_layer"]), "flops": int(row["flops"]), "latency_mean_ms": _float(row["latency_mean_ms"]), "latency_p95_ms": _float(row["latency_p95_ms"])} for row in inference]
    cost_lookup = {(row["method"], row["task"], row["exit_layer"]): row for row in cost}
    for row in headline:
        if row["method"] in NEW_METHODS:
            benchmark = cost_lookup[(row["method"], row["task"], row["exit_layer"])]
            inference.append({"method": row["method"], "method_label": METHOD_LABELS[row["method"]], "task": row["task"], "exit_layer": row["exit_layer"], "flops": benchmark["flops"], "latency_mean_ms": benchmark["latency_mean_ms"], "latency_p95_ms": benchmark["latency_p95_ms"], "protocol": "fresh synchronized CUDA benchmark"})
    output_dir.mkdir(parents=True)
    _write_csv(output_dir / "table_b0_context_l8_reference.csv", b0, ("method", "task", "metric", "value", "protocol"))
    _write_csv(output_dir / "table_primary_fixed_exits.csv", fixed, ("method", "task", "exit_layer", "metric", "value", "validation_value", "best_epoch", "head_epochs_completed", "head_stopped_early"))
    _write_csv(output_dir / "table_selected_exits.csv", selected, ("method", "task", "evaluation", "exit_layer", "metric", "value", "higher_is_better"))
    _write_csv(output_dir / "table_all_methods_downstream.csv", headline, ("method", "method_label", "task", "exit_layer", "selection", "metric", "value", "protocol"))
    _write_csv(output_dir / "table_inference_cost.csv", cost, ("method", "task", "exit_layer", "flops", "latency_mean_ms", "latency_median_ms", "latency_p95_ms", "latency_std_ms", "flops_savings_vs_l8", "latency_savings_vs_l8"))
    _write_csv(output_dir / "table_all_methods_inference.csv", inference, ("method", "method_label", "task", "exit_layer", "flops", "latency_mean_ms", "latency_p95_ms", "protocol"))
    markdown_tables = (
        ("table_b0_context_l8_reference", b0, ("method", "task", "metric", "value", "protocol")),
        ("table_primary_fixed_exits", fixed, ("method", "task", "exit_layer", "metric", "value", "validation_value", "best_epoch", "head_epochs_completed", "head_stopped_early")),
        ("table_selected_exits", selected, ("method", "task", "evaluation", "exit_layer", "metric", "value", "higher_is_better")),
        ("table_all_methods_downstream", headline, ("method", "method_label", "task", "exit_layer", "selection", "metric", "value", "protocol")),
        ("table_inference_cost", cost, ("method", "task", "exit_layer", "flops", "latency_mean_ms", "latency_median_ms", "latency_p95_ms", "latency_std_ms", "flops_savings_vs_l8", "latency_savings_vs_l8")),
        ("table_all_methods_inference", inference, ("method", "method_label", "task", "exit_layer", "flops", "latency_mean_ms", "latency_p95_ms", "protocol")),
    )
    for name, rows, fields in markdown_tables:
        (output_dir / f"{name}.md").write_text(_markdown_table(fields, rows))
    (output_dir / "README.md").write_text("# M-block P1/P2/P2-Reverse comparison packet\n\nSingle-seed screening packet. B0 L8 and P0 frozen-exit rows are reused from the historical m-block report; their split hash cannot be verified against the newly regenerated shared seed-2025 split. P1, P2, and P2-Reverse use fresh checkpoints, frozen heads, and synchronized CUDA benchmarks.\n")
    manifest = {"methods": ["b0", "p0", *NEW_METHODS], "baseline_report": str(baseline_report), "baseline_sha256": source, "fresh_run_root": str(run_root), "fresh_split_manifest": next(iter(split_manifests)), "historical_split_hash_verified": False, "task_suite": list(TASKS), "single_seed": 0}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if render_figures:
        _plots(output_dir, fixed, selected, headline, cost, inference)
    source_after = {path: _sha256(path) for path in baseline_report.glob("*") if path.is_file()}
    if source_before != source_after:
        raise RuntimeError("historical baseline packet changed while rendering")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, default=ROOT / "reports/masking_study/multi_block_n6")
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/masking_study/multi_block_n6/p1_p2_p2_reverse_seed0")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/masking_study/multi_block_n6/p1_p2_p2_reverse_seed0")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))
    manifest = build_packet(args.baseline_report, args.run_root, args.output_dir, render_figures=not args.skip_figures)
    print(json.dumps({"output_dir": str(args.output_dir), "methods": manifest["methods"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
