#!/usr/bin/env python3
"""Build a downstream-only per-mask packet from complete, available methods.

Historical P0/P2 results are reused only from the checked-in per-mask report
tables. P2-Reverse is added from its validated frozen-head test packet. P1 and
any method without a complete downstream packet are omitted automatically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FLOPS_FILE = ROOT / "reports/masking_study/flops_primary.json"
MASKS = ("frequency_n6", "time_n6", "region_n6")
TASK_LABELS = {
    "rfs": "RF sensing",
    "jamming": "Jamming classification",
    "human-activity": "Human activity sensing",
    "interference": "Interference classification",
    "nrpos-outdoor": "NR positioning",
    "deepmimo-beam": "DeepMIMO beam",
}
METHOD_LABELS = {
    "p0": "P0 — Original LatentWave (L8)",
    "p2": "P2 — Hierarchical Multi-Scale JEPA",
    "p2_reverse": "P2-Reverse — Coarse-to-Fine JEPA",
}
EXIT_LAYERS = (2, 4, 6, 8)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reverse(path: Path) -> dict[str, Any]:
    packet = json.loads(path.read_text())
    if packet.get("method") != "p2_reverse" or packet.get("dataset_suite") != "primary":
        raise ValueError(f"{path} is not a P2-Reverse primary downstream packet")
    if not packet.get("protocol", {}).get("test_evaluation"):
        raise ValueError(f"{path} does not contain post-selection test evaluation")
    tasks = packet.get("tasks", {})
    if set(tasks) != set(TASK_LABELS):
        raise ValueError(f"{path} must contain all six primary tasks; found {sorted(tasks)}")
    for name, task in tasks.items():
        if set(task.get("exits", {})) != {str(layer) for layer in EXIT_LAYERS}:
            raise ValueError(f"{path} is missing fixed exits for {name}")
        for field in (
            "selected_exit_best_validation",
            "selected_test_best_validation",
            "selected_exit_earliest_within_tolerance",
            "selected_test_earliest_within_tolerance",
        ):
            if field not in task:
                raise ValueError(f"{path} is missing {field} for {name}")
    return packet


def reverse_rows(packet: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fixed: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    headline: list[dict[str, Any]] = []
    for name, task in packet["tasks"].items():
        spec = task["spec"]
        metric = spec["metric_name"]
        higher = bool(spec["higher_is_better"])
        for layer in EXIT_LAYERS:
            result = task["exits"][str(layer)]
            fixed.append({
                "method": "p2_reverse", "method_label": METHOD_LABELS["p2_reverse"],
                "task": name, "exit_layer": layer, "metric": metric,
                "value": result["test"][metric],
                "validation_value": result["validation"][metric],
            })
        for policy, exit_key, value_key in (
            ("best_validation", "selected_exit_best_validation", "selected_test_best_validation"),
            (
                "earliest_within_tolerance",
                "selected_exit_earliest_within_tolerance",
                "selected_test_earliest_within_tolerance",
            ),
        ):
            selected.append({
                "method": "p2_reverse", "task": name, "evaluation": policy,
                "exit_layer": int(task[exit_key]), "metric": metric,
                "value": task[value_key][metric], "higher_is_better": higher,
            })
        layer = int(task["selected_exit_earliest_within_tolerance"])
        headline.append({
            "method": "p2_reverse", "method_label": METHOD_LABELS["p2_reverse"],
            "task": name, "exit_layer": layer,
            "selection": "earliest within 5% of validation best",
            "metric": metric,
            "value": task["selected_test_earliest_within_tolerance"][metric],
            "protocol": "frozen context; validation-only selection; held-out test after selection",
        })
    return fixed, selected, headline


def history_rows(report_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    """Load complete historical P0/P2 rows from a prior per-mask packet."""
    source_files = [
        report_dir / "table_b0_context_l8_reference.csv",
        report_dir / "table_primary_fixed_exits.csv",
        report_dir / "table_selected_exits.csv",
        report_dir / "table_all_methods_downstream.csv",
    ]
    source_ref, old_fixed, old_selected, old_headline = [read_csv(path) for path in source_files]
    fixed: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    headline: list[dict[str, Any]] = []

    # P0 is the original context encoder at full-depth L8, not a trained exit
    # selection method. Its historical source-reference table is authoritative.
    p0_by_task = {row["task"]: row for row in source_ref if row.get("method") in {"b0", "p0"}}
    if set(p0_by_task) == set(TASK_LABELS):
        for name, row in p0_by_task.items():
            fixed.append({
                "method": "p0", "method_label": METHOD_LABELS["p0"], "task": name,
                "exit_layer": 8, "metric": row["metric"], "value": row["value"],
                "validation_value": "",
            })
            headline.append({
                "method": "p0", "method_label": METHOD_LABELS["p0"], "task": name,
                "exit_layer": 8, "selection": "full-depth L8",
                "metric": row["metric"], "value": row["value"],
                "protocol": row.get("protocol", "historical P0 L8 reference"),
            })

    p2_fixed = [row for row in old_fixed if row.get("method") == "p2"]
    p2_selected = [row for row in old_selected if row.get("method") == "p2"]
    p2_headline = [row for row in old_headline if row.get("method") == "p2"]
    p2_complete = (
        len(p2_fixed) == len(TASK_LABELS) * len(EXIT_LAYERS)
        and len(p2_selected) == len(TASK_LABELS) * 2
        and len(p2_headline) == len(TASK_LABELS)
        and {row["task"] for row in p2_fixed} == set(TASK_LABELS)
    )
    if p2_complete:
        for row in p2_fixed:
            fixed.append({**row, "method_label": METHOD_LABELS["p2"]})
        for row in p2_selected:
            selected.append(row)
        for row in p2_headline:
            headline.append({**row, "method_label": METHOD_LABELS["p2"]})
    return fixed, selected, headline, [path for path in source_files if path.is_file()]


def plot_packet(output_dir: Path, fixed: list[dict[str, Any]], methods: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = {"p0": "#009E73", "p2": "#D55E00", "p2_reverse": "#E69F00"}
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.2), squeeze=False)
    for axis, task in zip(axes.flat, TASK_LABELS):
        rows = [row for row in fixed if row["task"] == task]
        higher = rows[0]["metric"] != "mean_distance_error"
        for method in methods:
            values = sorted(
                (row for row in rows if row["method"] == method),
                key=lambda row: int(row["exit_layer"]),
            )
            if values:
                axis.plot(
                    [int(row["exit_layer"]) for row in values],
                    [float(row["value"]) for row in values],
                    marker="o", color=colors[method], label=METHOD_LABELS[method],
                )
        axis.set_title(TASK_LABELS[task], fontsize=10, weight="bold")
        axis.set_xticks(EXIT_LAYERS)
        axis.set_xlabel("Encoder exit layer")
        axis.set_ylabel("Macro accuracy" if higher else "Mean distance error (m)")
        if higher:
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="y", alpha=.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), frameon=False)
    fig.suptitle("Available-method downstream performance by encoder depth", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, .92))
    for suffix, options in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(output_dir / f"downstream_exits.{suffix}", bbox_inches="tight", **options)
    plt.close(fig)


def build(mask: str, reverse_path: Path, report_dir: Path, output_dir: Path) -> list[str]:
    flop_data = json.loads(FLOPS_FILE.read_text())
    if not flop_data.get("operation_count_only") or flop_data.get("dataset_suite") != "primary":
        raise ValueError(f"{FLOPS_FILE} is not a primary FLOPs-only result")
    flop_lookup = {(row["task"], int(row["exit_layer"])): int(row["flops"]) for row in flop_data["entries"]}
    if len(flop_lookup) != len(TASK_LABELS) * len(EXIT_LAYERS):
        raise ValueError("FLOPs file must contain all six tasks at L2/L4/L6/L8")
    reverse = load_reverse(reverse_path)
    historical_fixed, historical_selected, historical_headline, historical_sources = history_rows(report_dir)
    reverse_fixed, reverse_selected, reverse_headline = reverse_rows(reverse)
    fixed = historical_fixed + reverse_fixed
    selected = historical_selected + reverse_selected
    headline = historical_headline + reverse_headline
    for rows in (fixed, selected, headline):
        for row in rows:
            row["flops"] = flop_lookup[(row["task"], int(row["exit_layer"]))]

    methods = [method for method in ("p0", "p2", "p2_reverse") if any(r["method"] == method for r in headline)]
    if "p2_reverse" not in methods:
        raise ValueError("P2-Reverse must be present in the report")
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_fields = ["method", "method_label", "task", "exit_layer", "metric", "value", "validation_value", "flops"]
    selected_fields = ["method", "task", "evaluation", "exit_layer", "metric", "value", "higher_is_better", "flops"]
    headline_fields = ["method", "method_label", "task", "exit_layer", "selection", "metric", "value", "protocol", "flops"]
    write_csv(output_dir / "table_fixed_exits.csv", fixed, fixed_fields)
    write_csv(output_dir / "table_selected_exits.csv", selected, selected_fields)
    write_csv(output_dir / "table_headline.csv", headline, headline_fields)
    (output_dir / "table_fixed_exits.md").write_text(
        "# Fixed-exit downstream results\n\nValidation results are shown only where available; test values are from the held-out split. FLOPs are per example through the listed exit and task head.\n\n"
        + markdown_table(
            ["Method", "Task", "Exit", "Metric", "Test", "Validation", "FLOPs/example"],
            [[r["method_label"], TASK_LABELS[r["task"]], f'L{r["exit_layer"]}', r["metric"], r["value"], r.get("validation_value", ""), r["flops"]] for r in fixed],
        ) + "\n"
    )
    (output_dir / "table_selected_exits.md").write_text(
        "# Validation-selected exits\n\nExit choice uses validation only; reported values are held-out test metrics. FLOPs are per example.\n\n"
        + markdown_table(
            ["Method", "Task", "Policy", "Exit", "Test value", "FLOPs/example"],
            [[METHOD_LABELS[r["method"]], TASK_LABELS[r["task"]], r["evaluation"], f'L{r["exit_layer"]}', r["value"], r["flops"]] for r in selected],
        ) + "\n"
    )
    (output_dir / "table_headline.md").write_text(
        "# Available-method headline comparison\n\nFLOPs are hardware-independent per-example operation counts; latency is hardware-dependent and is not reported.\n\n"
        + markdown_table(
            ["Method", "Task", "Exit", "Selection", "Test value", "FLOPs/example"],
            [[r["method_label"], TASK_LABELS[r["task"]], f'L{r["exit_layer"]}', r["selection"], r["value"], r["flops"]] for r in headline],
        ) + "\n"
    )
    plot_packet(output_dir, fixed, methods)

    missing = [method for method in ("p0", "p2") if method not in methods]
    manifest = {
        "report": "masking_study_available_methods_downstream",
        "masking": mask,
        "methods_included": methods,
        "method_labels": {method: METHOD_LABELS[method] for method in methods},
        "methods_omitted_for_missing_complete_downstream_results": missing + ["p1", "b1"],
        "p2_reverse_metrics": str(reverse_path),
        "p2_reverse_metrics_sha256": sha256(reverse_path),
        "historical_report_dir": str(report_dir),
        "historical_sources_used": [str(path) for path in historical_sources],
        "flops_source": str(FLOPS_FILE),
        "flops_source_sha256": sha256(FLOPS_FILE),
        "flops_protocol": flop_data["flops_protocol"],
        "historical_source_sha256": {path.name: sha256(path) for path in historical_sources},
        "protocol": "single-seed downstream comparison; validation-only selection; held-out test after selection",
        "inference_benchmarks": "not included; no P2-Reverse inference run is part of this packet",
        "output_dir": str(output_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "README.md").write_text(
        f"# {mask} available-method downstream comparison\n\n"
        "This downstream-only packet includes only methods with complete comparable results: "
        + ", ".join(METHOD_LABELS[method] for method in methods)
        + ". P1 and the post-hoc B1 approach are excluded. P0/P2 are reused only when a complete historical per-mask result packet is present; P2-Reverse is loaded from its validated frozen-head test packet. FLOPs/example count the matched encoder through each exit and task head; counts use `torch.utils.flop_counter.FlopCounterMode` supported operations only and do not indicate wall-clock speed.\n\n"
        "For P2-Reverse, head checkpoints and exits are selected on validation only; held-out test metrics are reported after selection. Consult `manifest.json` for exact input files and hashes.\n"
    )
    return methods


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", choices=MASKS, required=True)
    parser.add_argument("--p2-reverse-metrics", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    report_dir = args.report_dir or ROOT / "reports/masking_study" / args.mask
    output_dir = args.output_dir or report_dir / "available_methods_p2_reverse"
    methods = build(args.mask, args.p2_reverse_metrics, report_dir, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "methods": methods}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
