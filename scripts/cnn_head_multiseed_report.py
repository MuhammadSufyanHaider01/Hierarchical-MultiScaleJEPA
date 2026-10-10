#!/usr/bin/env python3
"""Combine seven-seed C0/C2 head results with the existing ViT M-block report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("rfs", "jamming", "human-activity", "interference", "nrpos-outdoor", "deepmimo-beam")
EXITS = (2, 4, 6, 8)
SEEDS = tuple(range(7))
CNN_METHODS = ("cnn_p0", "cnn_p2")
METHODS = ("b0", "p0", "p2", *CNN_METHODS)
METHOD_LABELS = {
    "b0": "B0 — Original LatentWave",
    "p0": "P0 — LatentWave with Post-hoc Frozen Early Exits",
    "p2": "P2 — Proposed Hierarchical Multi-Scale JEPA",
    "cnn_p0": "C0 — CNN Original L8 JEPA Objective",
    "cnn_p2": "C2 — CNN Mask-relative Fine-to-coarse JEPA",
}
TASK_LABELS = {
    "rfs": "RF sensing", "jamming": "Jamming classification",
    "human-activity": "Human activity sensing", "interference": "Interference classification",
    "nrpos-outdoor": "NR positioning", "deepmimo-beam": "DeepMIMO beam prediction",
}
COLORS = {"b0": "#009E73", "p0": "#0072B2", "p2": "#D55E00", "cnn_p0": "#44AA99", "cnn_p2": "#E69F00"}
SHORT_LABELS = {"b0": "B0", "p0": "P0", "p2": "P2", "cnn_p0": "C0", "cnn_p2": "C2"}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact is missing: {path}")
    return json.loads(path.read_text())


def _mean_sd(values: list[float]) -> tuple[float, float]:
    if len(values) != len(SEEDS) or not all(np.isfinite(values)):
        raise ValueError(f"expected seven finite values, got {values}")
    return statistics.fmean(values), statistics.stdev(values)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _load_cnn_method(method: str, run_root: Path, expected_split_sha: str):
    packets: dict[int, dict[str, Any]] = {}
    provenance = []
    checkpoint_digests = set()
    for seed in SEEDS:
        seed_name = f"{seed:03d}"
        run = run_root / f"head_seed_{seed_name}" / method / "seed_000" / "downstream"
        metrics_path = run / "exit_metrics.json"
        packet = _json(metrics_path)
        if packet.get("method") != method or int(packet.get("pretrain_seed", -1)) != 0:
            raise ValueError(f"wrong method/pretraining seed in {metrics_path}")
        if int(packet.get("head_seed", -1)) != seed:
            raise ValueError(f"wrong head seed in {metrics_path}: expected {seed}")
        if not packet.get("protocol", {}).get("test_evaluation"):
            raise ValueError(f"test evaluation is absent from {metrics_path}")
        if set(packet.get("tasks", {})) != set(TASKS):
            raise ValueError(f"task suite mismatch in {metrics_path}")
        split_path = Path(packet["split_manifest"])
        if not split_path.is_absolute():
            split_path = ROOT / split_path
        split_sha = _sha(split_path)
        if split_sha != expected_split_sha:
            raise ValueError(f"split manifest differs from the established report: {split_path}")
        checkpoint = Path(packet["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint missing for {metrics_path}: {checkpoint}")
        checkpoint_sha = _sha(checkpoint)
        checkpoint_digests.add(checkpoint_sha)
        if len(checkpoint_digests) > 1:
            raise ValueError(f"pretraining checkpoint changed across {method} head seeds")
        for task in TASKS:
            record = packet["tasks"][task]
            exits = record.get("exits", {})
            if set(exits) != {str(layer) for layer in EXITS}:
                raise ValueError(f"missing exits for {method}/seed{seed}/{task}")
            metric = record.get("spec", {}).get("metric_name")
            if not metric:
                raise ValueError(f"missing metric name in {method}/seed{seed}/{task}")
            for layer in EXITS:
                exit_record = exits[str(layer)]
                if metric not in exit_record.get("test", {}):
                    raise ValueError(f"missing held-out metric for {method}/seed{seed}/{task}/L{layer}")
                head_path = Path(exit_record.get("head_path", ""))
                if not head_path.is_file():
                    raise FileNotFoundError(f"head weights missing: {head_path}")
            selected = int(record.get("selected_exit_best_validation", -1))
            if selected not in EXITS:
                raise ValueError(f"invalid validation-selected exit in {method}/seed{seed}/{task}")
        packets[seed] = packet
        provenance.append({
            "head_seed": seed,
            "metrics_path": str(metrics_path),
            "metrics_sha256": _sha(metrics_path),
            "split_manifest": str(split_path),
            "split_sha256": split_sha,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "heads": [
                {
                    "task": task, "exit": layer,
                    "path": packet["tasks"][task]["exits"][str(layer)]["head_path"],
                    "sha256": _sha(Path(packet["tasks"][task]["exits"][str(layer)]["head_path"])),
                }
                for task in TASKS for layer in EXITS
            ],
        })
    return packets, provenance


def _add_cnn_rows(method: str, packets: dict[int, dict[str, Any]]):
    fixed_rows, selected_rows = [], []
    selected_exits: dict[str, dict[int, int]] = {}
    for task in TASKS:
        spec = packets[0]["tasks"][task]["spec"]
        metric = spec["metric_name"]
        higher = bool(spec["higher_is_better"])
        for layer in EXITS:
            values = [float(packets[seed]["tasks"][task]["exits"][str(layer)]["test"][metric]) for seed in SEEDS]
            mean, sd = _mean_sd(values)
            fixed_rows.append({
                "method": method, "method_label": METHOD_LABELS[method], "task": task,
                "task_label": TASK_LABELS[task], "metric": metric, "higher_is_better": higher,
                "exit_layer": layer, "mean": mean, "sample_sd": sd,
                "min": min(values), "max": max(values),
                "per_seed": json.dumps(dict(zip(SEEDS, values)), sort_keys=True),
            })
        values, selected_by_seed = [], {}
        for seed in SEEDS:
            task_packet = packets[seed]["tasks"][task]
            layer = int(task_packet["selected_exit_best_validation"])
            selected_by_seed[seed] = layer
            values.append(float(task_packet["exits"][str(layer)]["test"][metric]))
        mean, sd = _mean_sd(values)
        counts = {str(layer): sum(value == layer for value in selected_by_seed.values()) for layer in EXITS}
        selected_rows.append({
            "method": method, "method_label": METHOD_LABELS[method], "task": task,
            "task_label": TASK_LABELS[task], "metric": metric, "higher_is_better": higher,
            "mean": mean, "sample_sd": sd, "min": min(values), "max": max(values),
            "per_seed": json.dumps(dict(zip(SEEDS, values)), sort_keys=True),
            "selected_exit_by_seed": json.dumps(selected_by_seed, sort_keys=True),
            "selection_frequency": json.dumps(counts, sort_keys=True),
        })
        selected_exits[task] = selected_by_seed
    return fixed_rows, selected_rows, selected_exits


def _cnn_benchmarks(method: str, report_manifest: dict[str, Any], existing_report_dir: Path):
    source = report_manifest["method_provenance"][method]["benchmark_path"]
    benchmark_path = Path(source)
    if not benchmark_path.is_absolute():
        benchmark_path = ROOT / benchmark_path
    packet = _json(benchmark_path)
    entries = {(entry["task"], int(entry["exit_layer"])): entry for entry in packet["entries"]}
    if set(entries) != {(task, layer) for task in TASKS for layer in EXITS}:
        raise ValueError(f"benchmark coverage mismatch: {benchmark_path}")
    return entries, benchmark_path


def _baseline_selected_rows(report_dir: Path):
    rows = _read_csv(report_dir / "multiseed_validation_selected.csv")
    selected = {(row["method"], row["task"]): row for row in rows if row["method"] in ("b0", "p0", "p2")}
    if set(selected) != {(method, task) for method in ("b0", "p0", "p2") for task in TASKS}:
        raise ValueError("existing multiseed report lacks one or more B0/P0/P2 task rows")
    return selected


def _baseline_split_sha(report_dir: Path) -> str:
    manifest = _json(report_dir / "manifest.json")
    try:
        sources = manifest["sources"]["p0"]
        record = sources.get("0", sources.get(0))
        split_sha = record["split_sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("existing ViT multiseed manifest does not record the P0 split checksum") from error
    if not isinstance(split_sha, str) or len(split_sha) != 64:
        raise ValueError("invalid P0 split checksum in existing ViT multiseed manifest")
    return split_sha


def _baseline_fixed_rows(report_dir: Path):
    rows = _read_csv(report_dir / "multiseed_fixed_exits.csv")
    selected = {(row["method"], row["task"], int(row["exit"])): row for row in rows if row["method"] in ("p0", "p2")}
    if set(selected) != {(method, task, layer) for method in ("p0", "p2") for task in TASKS for layer in EXITS}:
        raise ValueError("existing multiseed report lacks full P0/P2 fixed-exit coverage")
    b0_rows = {(row["method"], row["task"], int(row["exit"])): row for row in rows if row["method"] == "b0"}
    if set(b0_rows) != {("b0", task, 8) for task in TASKS}:
        raise ValueError("existing multiseed report lacks B0 L8 values")
    return selected | b0_rows


def _baseline_inference(report_dir: Path):
    rows = _read_csv(report_dir / "multiseed_inference_cost.csv")
    indexed = {(row["method"], row["task"], int(row["exit"])): row for row in rows if row["method"] in ("b0", "p0", "p2")}
    if set(indexed) != ({("b0", task, 8) for task in TASKS} | {(method, task, layer) for method in ("p0", "p2") for task in TASKS for layer in EXITS}):
        raise ValueError("existing multiseed report lacks B0/P0/P2 inference coverage")
    return indexed


def _make_plots(output: Path, selected_rows: list[dict[str, Any]], fixed_rows: list[dict[str, Any]], inference_rows: list[dict[str, Any]]):
    paths = []
    selected_by = {(row["method"], row["task"]): row for row in selected_rows}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for ax, task in zip(axes.flat, TASKS):
        rows = [selected_by[(method, task)] for method in METHODS]
        x = np.arange(len(METHODS))
        ax.bar(x, [row["mean"] for row in rows], yerr=[row["sample_sd"] for row in rows], color=[COLORS[m] for m in METHODS], capsize=3)
        ax.set_title(TASK_LABELS[task])
        ax.set_xticks(x, ["B0", "P0", "P2", "C0", "C2"], rotation=0)
        ax.grid(axis="y", alpha=.25)
        ax.set_ylabel("Mean distance error (m)" if task == "nrpos-outdoor" else "Macro accuracy")
    for suffix, options in (("png", {"dpi": 300}), ("pdf", {})):
        path = output / f"multiseed_selected_performance.{suffix}"
        fig.savefig(path, bbox_inches="tight", **options)
        paths.append(path.name)
    plt.close(fig)

    fixed_by = {(row["method"], row["task"], int(row["exit_layer"])): row for row in fixed_rows}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for ax, task in zip(axes.flat, TASKS):
        for method in METHODS:
            exits = (8,) if method == "b0" else EXITS
            rows = [fixed_by[(method, task, layer)] for layer in exits]
            ax.errorbar(exits, [row["mean"] for row in rows], yerr=[row["sample_sd"] for row in rows], marker="o", capsize=2, label=SHORT_LABELS[method], color=COLORS[method])
        ax.set_title(TASK_LABELS[task])
        ax.set_xticks(EXITS, [f"L{x}" for x in EXITS])
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    for suffix, options in (("png", {"dpi": 300}), ("pdf", {})):
        path = output / f"multiseed_fixed_exits.{suffix}"
        fig.savefig(path, bbox_inches="tight", **options)
        paths.append(path.name)
    plt.close(fig)

    grouped = {(row["method"], row["task"]): row for row in inference_rows}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    x = np.arange(len(TASKS))
    width = .15
    for index, method in enumerate(METHODS):
        rows = [grouped[(method, task)] for task in TASKS]
        positions = x + (index - 2) * width
        axes[0].bar(positions, [row["mean_flops"] / 1e9 for row in rows], width, label=SHORT_LABELS[method], color=COLORS[method])
        axes[1].bar(positions, [row["mean_latency_ms"] for row in rows], width, label=SHORT_LABELS[method], color=COLORS[method])
    axes[0].set_ylabel("FLOPs (G)")
    axes[1].set_ylabel("Synchronized batch-1 latency (ms)")
    for ax in axes:
        ax.set_xticks(x, [TASK_LABELS[t] for t in TASKS], rotation=25, ha="right")
        ax.grid(axis="y", alpha=.25)
    axes[0].legend(fontsize=8)
    for suffix, options in (("png", {"dpi": 300}), ("pdf", {})):
        path = output / f"multiseed_selected_inference.{suffix}"
        fig.savefig(path, bbox_inches="tight", **options)
        paths.append(path.name)
    plt.close(fig)
    return paths


def build(run_root: Path, report_dir: Path, output: Path) -> None:
    current_manifest = _json(output / "manifest.json")
    baseline_split_sha = _baseline_split_sha(report_dir)
    cnn_split_path = ROOT / "runs/masking_study/multi_block_n6/splits_seed_2025_primary_v3.json"
    expected_split_sha = _sha(cnn_split_path)
    baseline_selected = _baseline_selected_rows(report_dir)
    baseline_fixed = _baseline_fixed_rows(report_dir)
    baseline_inference = _baseline_inference(report_dir)
    fixed_rows, selected_rows, inference_rows = [], [], []
    sources: dict[str, Any] = {}

    for method in ("b0", "p0", "p2"):
        for task in TASKS:
            source = baseline_fixed.get((method, task, 8)) if method == "b0" else None
            for layer in ((8,) if method == "b0" else EXITS):
                record = source or baseline_fixed[(method, task, layer)]
                metric = record["metric"]
                fixed_rows.append({
                    "method": method, "method_label": METHOD_LABELS[method], "task": task,
                    "task_label": TASK_LABELS[task], "metric": metric,
                    "higher_is_better": record["higher_is_better"], "exit_layer": layer,
                    "mean": float(record["mean"]), "sample_sd": float(record["sample_sd"]),
                    "min": float(record["min"]), "max": float(record["max"]),
                    "per_seed": record["per_seed"],
                })
            row = baseline_selected[(method, task)]
            selected_rows.append({
                "method": method, "method_label": METHOD_LABELS[method], "task": task,
                "task_label": TASK_LABELS[task], "metric": row["metric"],
                "higher_is_better": row["higher_is_better"], "mean": float(row["mean"]),
                "sample_sd": float(row["sample_sd"]), "min": float(row["min"]), "max": float(row["max"]),
                "per_seed": row["per_seed"], "selected_exit_by_seed": row["selected_exit_by_seed"],
                "selection_frequency": row["selection_frequency"],
            })
        sources[method] = {"source_report": str(report_dir), "source_manifest_sha256": _sha(report_dir / "manifest.json")}

    cnn_exits: dict[str, dict[str, dict[int, Any]]] = {}
    for method in CNN_METHODS:
        packets, provenance = _load_cnn_method(method, run_root, expected_split_sha)
        method_fixed, method_selected, selected_by_seed = _add_cnn_rows(method, packets)
        fixed_rows.extend(method_fixed)
        selected_rows.extend(method_selected)
        cnn_exits[method] = selected_by_seed
        sources[method] = provenance
        benchmark, benchmark_path = _cnn_benchmarks(method, current_manifest, output)
        sources[method + "_benchmark"] = {"path": str(benchmark_path), "sha256": _sha(benchmark_path)}
        for task in TASKS:
            per_seed = []
            selected_layers = selected_by_seed[task]
            for seed in SEEDS:
                layer = selected_layers[seed]
                row = benchmark[(task, layer)]
                per_seed.append({
                    "head_seed": seed, "selected_exit": layer,
                    "flops": float(row["flops"]),
                    "latency_mean_ms": float(row["latency"]["mean_ms"]),
                    "latency_p95_ms": float(row["latency"]["p95_ms"]),
                })
            fmean, fsd = _mean_sd([row["flops"] for row in per_seed])
            lmean, lsd = _mean_sd([row["latency_mean_ms"] for row in per_seed])
            pmean, psd = _mean_sd([row["latency_p95_ms"] for row in per_seed])
            inference_rows.append({
                "method": method, "method_label": METHOD_LABELS[method], "task": task,
                "task_label": TASK_LABELS[task], "selected_exit_by_seed": json.dumps({r["head_seed"]: r["selected_exit"] for r in per_seed}, sort_keys=True),
                "mean_flops": fmean, "sample_sd_flops": fsd,
                "mean_latency_ms": lmean, "sample_sd_latency_ms": lsd,
                "mean_p95_latency_ms": pmean, "sample_sd_p95_latency_ms": psd,
                "per_seed": json.dumps(per_seed, sort_keys=True),
            })

    for method in ("b0", "p0", "p2"):
        for task in TASKS:
            selected_exits = json.loads(baseline_selected[(method, task)]["selected_exit_by_seed"])
            per_seed = []
            for seed in SEEDS:
                layer = int(selected_exits[str(seed)] if str(seed) in selected_exits else selected_exits[seed])
                if method == "b0": layer = 8
                row = baseline_inference[(method, task, layer)]
                per_seed.append({
                    "head_seed": seed, "selected_exit": layer,
                    "flops": float(row["flops"]),
                    "latency_mean_ms": float(row["latency_mean_ms"]),
                    "latency_p95_ms": float(row["latency_p95_ms"]),
                })
            fmean, fsd = _mean_sd([r["flops"] for r in per_seed])
            lmean, lsd = _mean_sd([r["latency_mean_ms"] for r in per_seed])
            pmean, psd = _mean_sd([r["latency_p95_ms"] for r in per_seed])
            inference_rows.append({
                "method": method, "method_label": METHOD_LABELS[method], "task": task,
                "task_label": TASK_LABELS[task], "selected_exit_by_seed": json.dumps({r["head_seed"]: r["selected_exit"] for r in per_seed}, sort_keys=True),
                "mean_flops": fmean, "sample_sd_flops": fsd,
                "mean_latency_ms": lmean, "sample_sd_latency_ms": lsd,
                "mean_p95_latency_ms": pmean, "sample_sd_p95_latency_ms": psd,
                "per_seed": json.dumps(per_seed, sort_keys=True),
            })

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "multiseed_fixed_exits.csv", fixed_rows)
    _write_csv(output / "multiseed_validation_selected.csv", selected_rows)
    _write_csv(output / "multiseed_selected_inference.csv", inference_rows)
    figures = _make_plots(output, selected_rows, fixed_rows, inference_rows)
    split_note = (
        "The existing ViT multiseed report and the CNN head runs record different split-manifest SHA-256 values "
        f"({baseline_split_sha} vs {expected_split_sha}). Both use split seed 2025 and the same six task files, "
        "but exact sample-level split identity is not verified; treat cross-backbone comparisons as descriptive, "
        "not paired."
    )
    readme = f"""# Seven-head-seed M-block comparison: CNN C0 and C2\n\nThis supplement adds seven independently initialized frozen-head runs (head seeds 0–6) for C0 and C2 to the existing M-block CNN report. The pretrained CNN checkpoint is held fixed at pretraining seed 0 for each method. All six downstream tasks use the persisted CNN split manifest (seed 2025); exits are selected by validation performance separately for each task and head seed, and each saved selection is evaluated once on the held-out test split.\n\nB0, P0, and P2 are reused from `{report_dir.relative_to(ROOT)}`; their seven-head-seed outcomes are not retrained here. **Split comparability caveat:** {split_note}\n\nThe CNN inference summary uses the already-recorded per-exit FLOPs and synchronized batch-1 latency, weighted over each head seed's validation-selected exit. Thus the selected-inference spread reflects exit choice across head seeds, not repeated timing runs.\n\n## Seven-seed validation-selected held-out results\n\nSee `multiseed_validation_selected.csv` for mean, sample standard deviation, min/max, all per-seed test results, chosen exits, and exit-selection frequency.\n\n## Fixed-exit results\n\nSee `multiseed_fixed_exits.csv` for mean ± sample standard deviation at L2/L4/L6/L8 (B0 is full-depth L8 only).\n\n## Selected-exit inference\n\nSee `multiseed_selected_inference.csv` for per-task FLOPs and latency averaged over the seven validation-selected exits. Latency timing itself is reused from the existing benchmark packets, not remeasured for each head seed.\n\n## Figures\n\n{chr(10).join('- `' + path + '`' for path in figures)}\n"""
    (output / "README_multiseed.md").write_text(readme)
    index_readme = output / "README.md"
    seed0_readme = output / "README_seed000.md"
    if index_readme.is_file() and not seed0_readme.exists():
        shutil.copy2(index_readme, seed0_readme)
    index_readme.write_text(
        "# M-block CNN comparison — seven-head-seed update\n\n"
        "The authoritative multi-seed comparison for CNN C0 and C2 is now in "
        "[README_multiseed.md](README_multiseed.md). It reports seven independently "
        "initialized downstream-head seeds (0–6), reuses the existing seven-seed "
        "B0/P0/P2 results, and includes validation-selected held-out performance, "
        "fixed-exit results, inference summaries, provenance, and split-comparability caveats.\n\n"
        "Open `multiseed_validation_selected.csv`, `multiseed_fixed_exits.csv`, and "
        "`multiseed_selected_inference.csv` for the tables; the corresponding "
        "`multiseed_*.png` and PDF files are the updated comparison figures.\n\n"
        "The original single-seed packet is retained for traceability in "
        "[README_seed000.md](README_seed000.md) and its original unprefixed files; "
        "do not use those single-seed files as the multi-seed result.\n"
    )
    _write_json(output / "multiseed_manifest.json", {
        "study": "CNN M-block seven-head-seed downstream comparison",
        "head_seeds": list(SEEDS), "pretrain_seed": 0, "split_seed": 2025,
        "cnn_split_manifest": str(cnn_split_path),
        "cnn_split_manifest_sha256": expected_split_sha,
        "reused_vit_multiseed_split_sha256": baseline_split_sha,
        "split_hashes_match": expected_split_sha == baseline_split_sha,
        "comparability_note": split_note,
        "task_suite": list(TASKS),
        "methods": list(METHODS), "reused_baseline_report": str(report_dir),
        "sources": sources, "figures": figures,
        "inference_note": "Per-exit timing is reused; selected inference is averaged over validation-selected exit choices across head seeds.",
    })
    print(json.dumps({"report_dir": str(output), "head_seeds": list(SEEDS), "methods": list(METHODS), "figures": figures}, indent=2))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/cnn_head_multiseed/multi_block_n6_cnn_split_v1")
    parser.add_argument("--existing-report", type=Path, default=ROOT / "reports/masking_study/multi_block_n6")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/cnn_backbone/multi_block_n6/cnn_p0_vs_cnn_p2")
    args = parser.parse_args()
    build(args.run_root.resolve(), args.existing_report.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
