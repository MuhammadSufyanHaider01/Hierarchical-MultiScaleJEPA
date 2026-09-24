#!/usr/bin/env python3
"""Render one paired P0/P2 factory-OFDMA availability figure and method packets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("p0", "p2")
DISPLAY = {"p0": "LatentWave + EE", "p2": "Proposed hierarchical approach"}
FOLDER = {"antenna": "vary_antenna", "time": "vary_time_samples"}
TITLE = {"nrpos-outdoor": "NR positioning", "interference": "Interference", "jamming": "Jamming"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_packet(path: Path, method: str) -> dict[str, Any]:
    packet = json.loads(path.read_text())
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().strip() != sha256(path):
        raise ValueError(f"selection packet missing or mismatched digest sidecar: {path}")
    if (packet.get("stage") != "selection_complete" or packet.get("test_evaluation") is not False
            or packet.get("method") != method):
        raise ValueError(f"{method} input is not a matching validation-only selection artifact")
    return packet


def read_test(path: Path | None, selection_path: Path, selection: dict[str, Any]):
    if path is None:
        return None
    packet = json.loads(path.read_text())
    if (packet.get("stage") != "test_complete" or packet.get("test_evaluation") is not True
            or packet.get("task") != selection["task"] or packet.get("sweep") != selection["sweep"]
            or packet.get("method") != selection["method"]
            or packet.get("selection_sha256") != sha256(selection_path)):
        raise ValueError("test result does not match the selected model/task/sweep")
    return packet


def packet_rows(packet: dict[str, Any], test: dict[str, Any] | None):
    metric = packet["task_spec"]["metric_name"]
    tests = {int(row["resource_level"]): row for row in test["conditions"]} if test else {}
    rows = []
    for condition in packet["conditions"]:
        level = int(condition["resource_level"])
        layer = int(condition["selected_exit_earliest_within_tolerance"])
        item = {
            "method": packet["method"], "display_method": packet["display_method"],
            "task": packet["task"], "sweep": packet["sweep"],
            "availability_level": level,
            "selected_exit_layer": layer,
            "selected_validation_metric": condition["validation_by_exit"][str(layer)][metric],
            "best_validation_exit": condition["best_exit_validation"],
            "l8_validation_metric": condition["validation_by_exit"]["8"][metric],
            "l8_relative_gap": condition["validation_l8_relative_gap"].get(str(layer), 0.0),
            "input_shape": "x".join(map(str, condition["input_shape"])),
            "token_count": condition["token_count"],
            "antenna_indices": json.dumps(condition["antenna_indices"]),
            "selected_flops": condition["selected_exit_compute"]["flops"],
            "selected_latency_ms_median": condition["selected_exit_compute"]["latency_ms"]["median"],
            "validation_by_exit_json": json.dumps(condition["validation_by_exit"], sort_keys=True),
            "source_metrics": packet["provenance"]["source"]["metrics"]["path"],
            "source_checkpoint": packet["provenance"]["source"]["checkpoint"]["path"],
            "source_split_manifest": packet["provenance"]["source"]["real_split_manifest"]["path"],
            "synthetic_manifest": packet["provenance"]["formal_manifest"]["path"],
            "synthetic_manifest_sha256": packet["provenance"]["formal_manifest"]["sha256"],
            "formal_calibration_gate_passed": packet["protocol"]["formal_calibration_gate_passed"],
            "scope": packet["protocol"]["scientific_scope"],
        }
        if level in tests:
            item["test_selected_exit"] = tests[level]["selected_exit"]
            item["test_selected_metrics_json"] = json.dumps(tests[level]["selected_metrics"], sort_keys=True)
            item["test_fixed_exits_json"] = json.dumps(tests[level]["fixed_exits"], sort_keys=True)
        else:
            item["test_selected_exit"] = ""
            item["test_selected_metrics_json"] = ""
            item["test_fixed_exits_json"] = ""
        rows.append(item)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        raise ValueError("cannot write empty comparison")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, title: str, rows: list[dict[str, Any]], metric: str, packets):
    columns = ["method", "availability_level", "selected_exit_layer", "selected_validation_metric",
               "l8_validation_metric", "l8_relative_gap", "token_count", "selected_flops",
               "selected_latency_ms_median", "formal_calibration_gate_passed", "test_selected_metrics_json"]
    lines = [f"# {title}", "", f"Metric: `{metric}`. Selection used the synthetic selection split only. "
             "The calibration gates did not pass; interpret results only as synthetic-domain stress-test behavior.", "",
             f"- Synthetic scenario: `{packets['p0']['provenance']['formal_master']['path']}`",
             f"- Synthetic manifest SHA-256: `{packets['p0']['provenance']['formal_manifest']['sha256']}`",
             f"- P0 source metrics: `{packets['p0']['provenance']['source']['metrics']['path']}`",
             f"- P2 source metrics: `{packets['p2']['provenance']['source']['metrics']['path']}`", "",
             "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def plot_packet(path: Path, packet: dict[str, Any], rows: list[dict[str, Any]]):
    metric = packet["task_spec"]["metric_name"]
    x = [row["availability_level"] for row in rows]
    y = [row["selected_validation_metric"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(x, y, marker="o", linewidth=2, label=packet["display_method"])
    for row in rows:
        ax.annotate(f"L{row['selected_exit_layer']}", (row["availability_level"], row["selected_validation_metric"]),
                    xytext=(4, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Antenna channels C" if packet["sweep"] == "antenna" else "Time patches Tₚ")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{TITLE[packet['task']]}: {packet['display_method']} ({packet['sweep']})")
    ax.grid(True, alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0", type=Path, required=True)
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--p0-test", type=Path)
    parser.add_argument("--p2-test", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    paths = {"p0": resolve(args.p0), "p2": resolve(args.p2)}
    packets = {method: read_packet(paths[method], method) for method in METHODS}
    p0, p2 = packets["p0"], packets["p2"]
    for key in ("task", "sweep", "levels"):
        if p0.get(key) != p2.get(key):
            raise ValueError(f"P0/P2 artifacts mismatch in {key}")
    if p0["task_spec"] != p2["task_spec"]:
        raise ValueError("P0/P2 task metrics/specifications differ")
    for field in ("formal_manifest",):
        if p0["provenance"][field]["sha256"] != p2["provenance"][field]["sha256"]:
            raise ValueError("P0/P2 runs used different synthetic realization manifests")
    source_split_p0 = p0["provenance"]["source"]["real_split_manifest"]["sha256"]
    source_split_p2 = p2["provenance"]["source"]["real_split_manifest"]["sha256"]
    if source_split_p0 != source_split_p2:
        raise ValueError("P0/P2 source models used different measured-data split manifests")
    tests = {
        "p0": read_test(resolve(args.p0_test), paths["p0"], p0) if args.p0_test else None,
        "p2": read_test(resolve(args.p2_test), paths["p2"], p2) if args.p2_test else None,
    }
    rows = packet_rows(p0, tests["p0"]) + packet_rows(p2, tests["p2"])
    task, sweep = p0["task"], p0["sweep"]
    default_root = ROOT / "experiments/synthetic_data" / FOLDER[sweep] / task
    root = resolve(args.output_root) if args.output_root else default_root
    outdirs = {method: root / ("latentwave_ee" if method == "p0" else "hierarchical") for method in METHODS}
    targets = [root / name for name in ("availability_comparison.csv", "availability_comparison.md",
                                         "availability_comparison.png", "availability_comparison.pdf", "README.md")]
    targets += [folder / name for folder in outdirs.values() for name in
                ("availability.csv", "availability.md", "availability.png", "availability.pdf")]
    if any(path.exists() for path in targets):
        raise FileExistsError(f"report outputs already exist under {root}")
    root.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        outdirs[method].mkdir(parents=True, exist_ok=True)
        method_rows = [row for row in rows if row["method"] == method]
        write_csv(outdirs[method] / "availability.csv", method_rows)
        write_markdown(outdirs[method] / "availability.md", f"{TITLE[task]} — {DISPLAY[method]} synthetic {sweep}",
                       method_rows, p0["task_spec"]["metric_name"], packets)
        plot_packet(outdirs[method] / "availability.png", packets[method], method_rows)
        plot_packet(outdirs[method] / "availability.pdf", packets[method], method_rows)
    write_csv(root / "availability_comparison.csv", rows)
    write_markdown(root / "availability_comparison.md", f"{TITLE[task]} — paired synthetic {sweep}",
                   rows, p0["task_spec"]["metric_name"], packets)
    metric = p0["task_spec"]["metric_name"]
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    colors = {"p0": "#145ea8", "p2": "#d26a00"}
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        ax.plot([r["availability_level"] for r in group],
                [r["selected_validation_metric"] for r in group],
                marker="o", linewidth=2, color=colors[method], label=DISPLAY[method])
        for row in group:
            ax.annotate(f"{method.upper()} L{row['selected_exit_layer']}",
                        (row["availability_level"], row["selected_validation_metric"]),
                        xytext=(4, 6 if method == "p0" else -13), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Antenna channels C" if sweep == "antenna" else "Time patches Tₚ")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{TITLE[task]} — {sweep.replace('_', ' ')} (synthetic factory-OFDMA stress test)")
    ax.grid(True, alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "availability_comparison.png", dpi=180)
    fig.savefig(root / "availability_comparison.pdf")
    plt.close(fig)
    readme = [f"# {TITLE[task]} synthetic {sweep} availability", "",
              "Paired zero-shot synthetic-domain stress test using the published factory-OFDMA configuration. "
              "The calibration anchors failed; these are not source-task accuracy or a reproduction of measured data.", "",
              "- LatentWave + EE source: " + p0["provenance"]["source"]["metrics"]["path"],
              "- Hierarchical source: " + p2["provenance"]["source"]["metrics"]["path"],
              "- Synthetic master: " + p0["provenance"]["formal_master"]["path"],
              "- Synthetic manifest SHA-256: " + p0["provenance"]["formal_manifest"]["sha256"],
              "- Separate immutable numeric packets and individual plots are in `latentwave_ee/` and `hierarchical/`.",
              "- Test data were not evaluated unless matching post-review test JSON files were supplied.", ""]
    (root / "README.md").write_text("\n".join(readme))
    print(json.dumps({"comparison_root": str(root), "figure": str(root / "availability_comparison.png"),
                      "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
