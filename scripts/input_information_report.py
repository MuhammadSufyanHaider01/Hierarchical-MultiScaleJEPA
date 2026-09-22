#!/usr/bin/env python3
"""Render compact CSV and Markdown summaries for an input-information study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


TASK_TITLES = {
    "nrpos-outdoor": "5G NR outdoor positioning",
    "human-activity": "WiFi human activity recognition",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="input_information_exit_metrics.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _format(value: object) -> str:
    return "" if value is None else (f"{value:.6f}" if isinstance(value, float) else str(value))


def _rows(study_name: str, study: dict) -> list[dict[str, object]]:
    spec = study["task_spec"]
    metric = spec["metric_name"]
    level_name = (
        "available_channels"
        if study.get("availability_mode") == "antenna_prefix" or study_name == "antenna"
        else "temporal_patch_columns"
    )
    rows = []
    for condition in study["conditions"]:
        row: dict[str, object] = {
            "study": study_name,
            "task": study["task"],
            level_name: condition[level_name],
            "condition": condition["condition"],
            "input_shape": "x".join(str(value) for value in condition["input_shape"]),
            "token_count": condition["token_count"],
        }
        for layer in (2, 4, 6, 8):
            row[f"validation_L{layer}_{metric}"] = condition["validation"][str(layer)][metric]
            row[f"l8_relative_gap_L{layer}"] = condition["validation_l8_relative_gap"][str(layer)]
            if "test" in condition:
                row[f"test_L{layer}_{metric}"] = condition["test"][str(layer)][metric]
        selected = int(condition["selected_exit_earliest_within_tolerance"])
        row["selected_exit"] = selected
        row["selected_validation_metric"] = condition["validation"][str(selected)][metric]
        row["selected_test_metric"] = (
            condition.get("selected_test_earliest_within_tolerance", {}).get(metric)
        )
        compute = condition["selected_exit_compute"]
        row["selected_flops"] = compute["flops"]
        row["selected_latency_ms_median"] = compute["latency_ms"]["median"]
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w") as handle:
        handle.write("| " + " | ".join(fields) + " |\n")
        handle.write("| " + " | ".join("---" for _ in fields) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(_format(row[field]) for field in fields) + " |\n")


def _write_antenna_exit_plot(output_dir: Path, study: dict, *, stem: str) -> None:
    """Plot validation-selected exit depth against available antenna channels."""

    metric = study["task_spec"]["metric_name"]
    conditions = sorted(study["conditions"], key=lambda condition: int(condition["available_channels"]))
    channels = [int(condition["available_channels"]) for condition in conditions]
    exits = [int(condition["selected_exit_earliest_within_tolerance"]) for condition in conditions]
    errors = [float(condition["validation"][str(exit)][metric]) for condition, exit in zip(conditions, exits)]
    suffix = " m" if metric == "mean_distance_error" else ""
    metric_label = "mean distance error (m)" if metric == "mean_distance_error" else metric.replace("_", " ")

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.plot(channels, errors, marker="o", linewidth=2.2, markersize=7, color="#1f77b4")
    for channel, exit_layer, error in zip(channels, exits, errors):
        axis.annotate(
            f"L{exit_layer}",
            xy=(channel, error),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#1f77b4", "alpha": 0.9},
        )
    axis.set_xticks(channels)
    margin = max(0.1, (max(errors) - min(errors)) * 0.15)
    axis.set_ylim(min(errors) - margin, max(errors) + margin * 1.8)
    axis.set_xlabel("Available antenna channels (C)")
    axis.set_ylabel(metric_label.capitalize())
    axis.set_title(TASK_TITLES.get(study["task"], str(study["task"])))
    axis.grid(axis="both", color="#d9d9d9", linewidth=0.8)
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = _arguments()
    artifact = json.loads(args.input.read_text())
    studies = artifact.get("studies", {})
    if not studies:
        raise ValueError(f"study artifact has no studies: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for study_name in ("antenna", "antenna-human-activity", "temporal"):
        study = studies.get(study_name)
        if study is None:
            continue
        if not isinstance(study, dict):
            raise ValueError(f"study artifact has malformed {study_name} data")
        rows = _rows(study_name, study)
        all_rows.extend(rows)
        _write_csv(args.output_dir / f"input_information_{study_name}.csv", rows)
        _write_markdown(args.output_dir / f"input_information_{study_name}.md", rows)
        if study.get("availability_mode") == "antenna_prefix" or study_name == "antenna":
            stem = "antenna_exit_vs_channels" if study["task"] == "nrpos-outdoor" else f"{study['task']}_antenna_exit_vs_channels"
            _write_antenna_exit_plot(args.output_dir, study, stem=stem)
    if not all_rows:
        raise ValueError(f"study artifact has no recognized study results: {args.input}")
    selected = [
        {
            key: row[key]
            for key in (
                "study", "task", "condition", "token_count", "selected_exit",
                "selected_validation_metric", "selected_test_metric", "selected_flops",
                "selected_latency_ms_median",
            )
        }
        for row in all_rows
    ]
    _write_csv(args.output_dir / "input_information_selected_exits.csv", selected)
    _write_markdown(args.output_dir / "input_information_selected_exits.md", selected)
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(all_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
