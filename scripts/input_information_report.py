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
    "interference": "Wireless interference classification",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="One input_information_exit_metrics.json")
    inputs.add_argument("--baseline-input", type=Path, help="P0 LatentWave + frozen early-exit selection artifact for a paired report")
    parser.add_argument("--hierarchical-input", type=Path, help="P2 proposed-hierarchical selection artifact for a paired report")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--write-readme",
        action="store_true",
        help="Write a self-contained README for a single-method packet; valid only with --input.",
    )
    args = parser.parse_args()
    if (args.baseline_input is None) != (args.hierarchical_input is None):
        parser.error("--baseline-input and --hierarchical-input must be supplied together")
    if args.write_readme and args.input is None:
        parser.error("--write-readme is valid only with --input")
    return args


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


def _write_temporal_exit_plot(output_dir: Path, study: dict, *, stem: str) -> None:
    """Plot validation-selected exit depth against available temporal patches."""

    metric = study["task_spec"]["metric_name"]
    conditions = sorted(study["conditions"], key=lambda condition: int(condition["temporal_patch_columns"]))
    patches = [int(condition["temporal_patch_columns"]) for condition in conditions]
    exits = [int(condition["selected_exit_earliest_within_tolerance"]) for condition in conditions]
    errors = [float(condition["validation"][str(exit)][metric]) for condition, exit in zip(conditions, exits)]
    metric_label = "mean distance error (m)" if metric == "mean_distance_error" else metric.replace("_", " ")

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.plot(patches, errors, marker="o", linewidth=2.2, markersize=7, color="#1f77b4")
    for patches_count, exit_layer, error in zip(patches, exits, errors):
        axis.annotate(
            f"L{exit_layer}",
            xy=(patches_count, error),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#1f77b4", "alpha": 0.9},
        )
    axis.set_xticks(patches)
    margin = max(0.1, (max(errors) - min(errors)) * 0.15)
    axis.set_ylim(min(errors) - margin, max(errors) + margin * 1.8)
    axis.set_xlabel("Available time patches (Tₚ)")
    axis.set_ylabel(metric_label.capitalize())
    axis.set_title(TASK_TITLES.get(study["task"], str(study["task"])))
    axis.grid(axis="both", color="#d9d9d9", linewidth=0.8)
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _artifact_study(artifact: dict, path: Path) -> tuple[str, dict]:
    if artifact.get("stage") != "selection_complete" or artifact.get("test_evaluation") is not False:
        raise ValueError(f"paired report requires a validation-only completed selection artifact: {path}")
    studies = artifact.get("studies")
    if not isinstance(studies, dict) or len(studies) != 1:
        raise ValueError(f"paired report requires exactly one study in each artifact: {path}")
    name, study = next(iter(studies.items()))
    if not isinstance(study, dict):
        raise ValueError(f"paired report has malformed study data: {path}")
    return name, study


def _manifest_hash(artifact: dict, path: Path) -> str:
    provenance = artifact.get("provenance", {})
    record = provenance.get("split_manifest", {}) if isinstance(provenance, dict) else {}
    value = record.get("sha256") if isinstance(record, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError(f"paired report requires split-manifest SHA-256 provenance: {path}")
    return value


def _axis(study_name: str, study: dict) -> tuple[str, str, str]:
    mode = study.get("availability_mode")
    if mode == "antenna_prefix" or study_name == "antenna":
        return "available_channels", "Available antenna channels (C)", "antenna_exit_vs_channels"
    if mode == "leading_temporal_patches":
        return "temporal_patch_columns", "Available time patches (Tₚ)", "temporal_exit_vs_time_patches"
    raise ValueError(f"paired report does not recognize availability mode {mode!r}")


def _write_single_method_readme(output_dir: Path, artifact: dict, study_name: str, study: dict) -> None:
    """Write an index for one immutable P0 or P2 method packet."""

    provenance = artifact.get("provenance", {})
    source_method = provenance.get("source_method")
    method_label = {
        "p0": "LatentWave + EE",
        "p2": "Proposed hierarchical approach",
    }.get(source_method, str(source_method or "Frozen early-exit method"))
    _, axis_label, stem = _axis(study_name, study)
    if study["task"] != "nrpos-outdoor":
        stem = f"{study['task']}_antenna_exit_vs_channels" if study.get("availability_mode") == "antenna_prefix" else f"{study['task']}_temporal_exit_vs_time_patches"
    metric = study["task_spec"]["metric_name"].replace("_", " ")
    output_study_name = "temporal" if study_name.startswith("temporal") else study_name
    full_shape = " × ".join(str(value) for value in study["task_spec"].get("sample_shape", ()))
    source_metrics = provenance.get("source_metrics", {}).get("path", "unavailable")
    checkpoint = provenance.get("checkpoint", {}).get("path", "unavailable")
    manifest = provenance.get("split_manifest", {}).get("path", "unavailable")
    precision = artifact.get("protocol", {}).get("matmul_precision", "source run setting")
    tf32 = artifact.get("protocol", {}).get("tf32_enabled")
    tf32_line = "" if tf32 is None else f"- TF32 enabled: `{tf32}`\n"
    output_dir.joinpath("README.md").write_text(
        f"# {method_label} — {TASK_TITLES.get(study['task'], study['task'])}\n\n"
        "This is an immutable, validation-only input-availability selection packet. "
        "The foundation checkpoint, frozen exit heads, source HDF5 files, and split manifest were not modified; the test split was not evaluated.\n\n"
        "## Run definition\n\n"
        f"- Availability axis: {axis_label}\n"
        f"- Levels: `{', '.join(str(value) for value in study['levels'])}`\n"
        f"- Full input shape: `{full_shape}`\n"
        f"- Primary validation metric: {metric}\n"
        "- Candidate exits: L2, L4, L6, L8; selection is earliest within the source 5% validation tolerance.\n"
        f"- Matmul precision: `{precision}`\n"
        f"{tf32_line}\n"
        "## Results files\n\n"
        f"- `input_information_{output_study_name}.md` / `.csv`: every fixed-exit validation metric, L8-relative gap, input shape, token count, selected exit, FLOPs, and latency.\n"
        "- `input_information_selected_exits.md` / `.csv`: compact selected-exit summary.\n"
        f"- `{stem}.png` / `.pdf`: this method's selected-metric curve; annotations identify the selected exit layer.\n"
        "- `input_information_exit_metrics.json`: complete machine-readable metrics and immutable provenance.\n\n"
        "## Provenance\n\n"
        f"- Source full-information metrics: `{source_metrics}`\n"
        f"- Frozen checkpoint: `{checkpoint}`\n"
        f"- Split manifest: `{manifest}`\n"
    )


def _selected_error(condition: dict, metric: str) -> tuple[int, float]:
    selected = int(condition["selected_exit_earliest_within_tolerance"])
    return selected, float(condition["validation"][str(selected)][metric])


def _validate_paired_studies(
    baseline_artifact: dict,
    baseline_path: Path,
    hierarchical_artifact: dict,
    hierarchical_path: Path,
) -> tuple[str, dict, dict, str, str, str]:
    baseline_name, baseline = _artifact_study(baseline_artifact, baseline_path)
    hierarchical_name, hierarchical = _artifact_study(hierarchical_artifact, hierarchical_path)
    baseline_method = baseline_artifact.get("provenance", {}).get("source_method")
    hierarchical_method = hierarchical_artifact.get("provenance", {}).get("source_method")
    if baseline_method != "p0" or hierarchical_method != "p2":
        raise ValueError("paired report requires P0 baseline and P2 hierarchical source artifacts")
    if _manifest_hash(baseline_artifact, baseline_path) != _manifest_hash(hierarchical_artifact, hierarchical_path):
        raise ValueError("paired report artifacts use different split manifests")
    if baseline.get("task") != hierarchical.get("task"):
        raise ValueError("paired report artifacts describe different downstream tasks")
    if baseline.get("task_spec") != hierarchical.get("task_spec"):
        raise ValueError("paired report artifacts have incompatible task specifications")
    if baseline.get("levels") != hierarchical.get("levels"):
        raise ValueError("paired report artifacts have different declared availability levels")
    axis_key, axis_label, stem = _axis(baseline_name, baseline)
    other_axis_key, _, _ = _axis(hierarchical_name, hierarchical)
    if axis_key != other_axis_key:
        raise ValueError("paired report artifacts use different availability axes")
    baseline_conditions = baseline.get("conditions")
    hierarchical_conditions = hierarchical.get("conditions")
    if not isinstance(baseline_conditions, list) or not isinstance(hierarchical_conditions, list):
        raise ValueError("paired report artifact has malformed conditions")
    if len(baseline_conditions) != len(hierarchical_conditions) or not baseline_conditions:
        raise ValueError("paired report artifacts have different condition counts")
    for index, (p0, p2) in enumerate(zip(baseline_conditions, hierarchical_conditions)):
        if p0.get("condition") != p2.get("condition") or p0.get(axis_key) != p2.get(axis_key):
            raise ValueError(f"paired report condition mismatch at ordered index {index}")
        if p0.get("input_shape") != p2.get("input_shape") or p0.get("token_count") != p2.get("token_count"):
            raise ValueError(f"paired report input-control mismatch at ordered index {index}")
    expected_full_level = max(int(condition[axis_key]) for condition in baseline_conditions)
    if int(baseline_conditions[-1][axis_key]) != expected_full_level or int(hierarchical_conditions[-1][axis_key]) != expected_full_level:
        raise ValueError("paired report requires matching ordered full-information controls")
    full_shape = list(baseline["task_spec"].get("sample_shape", ()))
    if full_shape and (baseline_conditions[-1].get("input_shape") != full_shape or hierarchical_conditions[-1].get("input_shape") != full_shape):
        raise ValueError("paired report full-information control does not retain the complete input shape")
    return baseline_name, baseline, hierarchical, axis_key, axis_label, stem


def _comparison_rows(study_name: str, study: dict, *, method_label: str) -> list[dict[str, object]]:
    rows = _rows(study_name, study)
    for row in rows:
        row["method"] = method_label
    fields = ["method", *[field for field in rows[0] if field != "method"]] if rows else []
    return [{field: row[field] for field in fields} for row in rows]


def _write_paired_exit_plot(
    output_dir: Path,
    baseline: dict,
    hierarchical: dict,
    *,
    axis_key: str,
    axis_label: str,
    stem: str,
) -> None:
    metric = baseline["task_spec"]["metric_name"]
    p0 = sorted(baseline["conditions"], key=lambda condition: int(condition[axis_key]))
    p2 = sorted(hierarchical["conditions"], key=lambda condition: int(condition[axis_key]))
    levels = [int(condition[axis_key]) for condition in p0]
    p0_selected = [_selected_error(condition, metric) for condition in p0]
    p2_selected = [_selected_error(condition, metric) for condition in p2]
    p0_errors = [value for _, value in p0_selected]
    p2_errors = [value for _, value in p2_selected]
    metric_label = "Mean distance error (m)" if metric == "mean_distance_error" else metric.replace("_", " ").capitalize()

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.plot(levels, p0_errors, marker="o", linewidth=2.2, markersize=7, color="#1f77b4", label="LatentWave + EE")
    axis.plot(levels, p2_errors, marker="s", linewidth=2.2, markersize=7, color="#d62728", label="Proposed hierarchical approach")
    for level, (exit_layer, error) in zip(levels, p0_selected):
        axis.annotate(f"P0 L{exit_layer}", xy=(level, error), xytext=(0, 12), textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#1f77b4")
    for level, (exit_layer, error) in zip(levels, p2_selected):
        axis.annotate(f"P2 L{exit_layer}", xy=(level, error), xytext=(0, -17), textcoords="offset points", ha="center", va="top", fontsize=8, color="#d62728")
    errors = [*p0_errors, *p2_errors]
    margin = max(0.1, (max(errors) - min(errors)) * 0.15)
    axis.set_xticks(levels)
    axis.set_ylim(min(errors) - margin * 1.8, max(errors) + margin * 1.8)
    axis.set_xlabel(axis_label)
    axis.set_ylabel(metric_label)
    axis.set_title(TASK_TITLES.get(baseline["task"], str(baseline["task"])))
    axis.grid(axis="both", color="#d9d9d9", linewidth=0.8)
    axis.legend()
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _write_comparison_readme(
    output_dir: Path,
    baseline_path: Path,
    hierarchical_path: Path,
    study: dict,
    axis_label: str,
) -> None:
    metric = study["task_spec"]["metric_name"].replace("_", " ")
    output_dir.joinpath("README.md").write_text(
        f"# Paired availability study — {TASK_TITLES.get(study['task'], study['task'])}\n\n"
        "This packet compares validation-only, frozen-head early-exit selection for the "
        "LatentWave + EE P0 baseline and the proposed hierarchical P2 approach. "
        "No foundation weights, downstream heads, manifest, or test split were changed.\n\n"
        f"- Availability axis: {axis_label}\n"
        f"- Primary validation metric: {metric}\n"
        f"- P0 source artifact: `{baseline_path}`\n"
        f"- P2 source artifact: `{hierarchical_path}`\n\n"
        "`latentwave_ee/` and `hierarchical/` contain immutable per-method metric artifacts; "
        "the root PNG/PDF is the single two-line comparison figure.\n"
    )


def _write_paired_report(baseline_path: Path, hierarchical_path: Path, output_dir: Path) -> None:
    baseline_artifact = json.loads(baseline_path.read_text())
    hierarchical_artifact = json.loads(hierarchical_path.read_text())
    study_name, baseline, hierarchical, axis_key, axis_label, stem = _validate_paired_studies(
        baseline_artifact, baseline_path, hierarchical_artifact, hierarchical_path
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        *_comparison_rows(study_name, baseline, method_label="LatentWave + EE"),
        *_comparison_rows(study_name, hierarchical, method_label="Proposed hierarchical approach"),
    ]
    _write_csv(output_dir / "availability_comparison.csv", rows)
    _write_markdown(output_dir / "availability_comparison.md", rows)
    selected_fields = ("method", "study", "task", "condition", "token_count", "selected_exit", "selected_validation_metric", "selected_flops", "selected_latency_ms_median")
    selected = [{field: row[field] for field in selected_fields} for row in rows]
    _write_csv(output_dir / "availability_comparison_selected_exits.csv", selected)
    _write_markdown(output_dir / "availability_comparison_selected_exits.md", selected)
    _write_paired_exit_plot(output_dir, baseline, hierarchical, axis_key=axis_key, axis_label=axis_label, stem=stem)
    _write_comparison_readme(output_dir, baseline_path, hierarchical_path, baseline, axis_label)
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows), "mode": "paired"}, indent=2))

def main() -> int:
    args = _arguments()
    if args.baseline_input is not None:
        _write_paired_report(args.baseline_input, args.hierarchical_input, args.output_dir)
        return 0
    artifact = json.loads(args.input.read_text())
    studies = artifact.get("studies", {})
    if not studies:
        raise ValueError(f"study artifact has no studies: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for study_name, study in studies.items():
        if not isinstance(study, dict):
            raise ValueError(f"study artifact has malformed {study_name} data")
        rows = _rows(study_name, study)
        all_rows.extend(rows)
        output_study_name = "temporal" if study_name.startswith("temporal") else study_name
        _write_csv(args.output_dir / f"input_information_{output_study_name}.csv", rows)
        _write_markdown(args.output_dir / f"input_information_{output_study_name}.md", rows)
        if study.get("availability_mode") == "antenna_prefix" or study_name == "antenna":
            stem = "antenna_exit_vs_channels" if study["task"] == "nrpos-outdoor" else f"{study['task']}_antenna_exit_vs_channels"
            _write_antenna_exit_plot(args.output_dir, study, stem=stem)
        elif study.get("availability_mode") == "leading_temporal_patches":
            stem = "temporal_exit_vs_time_patches" if study["task"] == "nrpos-outdoor" else f"{study['task']}_temporal_exit_vs_time_patches"
            _write_temporal_exit_plot(args.output_dir, study, stem=stem)
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
    if args.write_readme:
        if len(studies) != 1:
            raise ValueError("--write-readme requires exactly one study")
        study_name, study = next(iter(studies.items()))
        _write_single_method_readme(args.output_dir, artifact, study_name, study)
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(all_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
