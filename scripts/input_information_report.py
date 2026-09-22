#!/usr/bin/env python3
"""Render compact CSV and Markdown summaries for an input-information study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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
    level_name = "available_channels" if study_name == "antenna" else "temporal_patch_columns"
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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


def main() -> int:
    args = _arguments()
    artifact = json.loads(args.input.read_text())
    studies = artifact.get("studies", {})
    if not studies:
        raise ValueError(f"study artifact has no studies: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for study_name in ("antenna", "temporal"):
        study = studies.get(study_name)
        if not isinstance(study, dict):
            raise ValueError(f"study artifact lacks {study_name}")
        rows = _rows(study_name, study)
        all_rows.extend(rows)
        _write_csv(args.output_dir / f"input_information_{study_name}.csv", rows)
        _write_markdown(args.output_dir / f"input_information_{study_name}.md", rows)
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
