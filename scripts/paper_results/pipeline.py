"""Build paper figures from versioned report artifacts; never train or benchmark."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from style import WIDTH, COLORS, METHODS, TASKS, TASK_ORDER, configure

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/paper_results"
MASK = ROOT / "reports/masking_study"
FUSION = ROOT / "reports/early_exit_optimization/multi_block_n6/residual_dynamic_fusion_multiseed"
ABLATION = ROOT / "reports/early_exit_optimization/multi_block_n6/residual_fusion_adapter_ablation_single_seed"
REPS = ROOT / "reports/representation_trajectories/multi_block_n6"
CONDITIONS = ("time_n6", "frequency_n6", "region_n6", "multi_block_n6")
METHOD_ORDER = tuple(METHODS)
EXITS = (2, 4, 6, 8)
SEEDS = {str(i) for i in range(7)}
ALGORITHMS = ("current", "single_gelu")
EXPECTED_REVISION = "69b3c8746043314dffd45a39a66678d81212d3e2"
INPUTS = set()
BENCHMARK = ROOT / "reports/matched_exit_benchmark/benchmark.json"


def source(path):
    path = Path(path)
    INPUTS.add(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing required input: {path}")
    return path


def read_csv(path, required):
    path = source(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing columns {sorted(set(required) - set(reader.fieldnames or []))}")
        return list(reader)


def read_json(path):
    return json.loads(source(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows, fields=None):
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def index_unique(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in result:
            raise ValueError(f"Duplicate key {key} for {keys}")
        result[key] = row
    return result


def require_keys(actual, expected, name):
    missing, extra = expected - set(actual), set(actual) - expected
    if missing or extra:
        raise ValueError(f"{name}: missing={sorted(missing)}, unexpected={sorted(extra)}")


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite number: {value}")
    return result


def check_close(a, b, name, tolerance=1e-7):
    if not math.isclose(number(a), number(b), rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{name}: {a} differs from {b}")


def task_metric(task):
    return ("mean_distance_error", False, "m", 1.0) if task == "nrpos-outdoor" else ("macro_accuracy", True, "%", 100.0)


def validate_metric(row, task):
    metric, higher, _, _ = task_metric(task)
    if row["metric"] != metric or ("higher_is_better" in row and row["higher_is_better"].lower() != str(higher).lower()):
        raise ValueError(f"Metric/direction mismatch for {task}: {row}")


def seed_values(row):
    values = json.loads(row["per_seed"])
    if set(values) != SEEDS:
        raise ValueError(f"Expected head seeds 0-6, found {sorted(values)}")
    values = {str(k): number(v) for k, v in values.items()}
    check_close(row["mean"], statistics.mean(values.values()), "mean")
    if "sample_sd" in row:
        check_close(row["sample_sd"], statistics.stdev(values.values()), "sample SD")
    return values


def save_fig(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches=None)
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches=None)
    plt.close(fig)


def latex_escape(value):
    s = str(value)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&")):
        s = s.replace(a, b)
    return s


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fusion_provenance():
    data = read_json(FUSION / "manifest.json")
    if data.get("test_used_for_selection") is not False or data.get("head_seeds") != list(range(7)):
        raise ValueError("Fusion manifest does not confirm the expected split and seeds")
    rows = data["sources"]
    expected = {(m, a, s) for m in ("p0", "p2") for a in ALGORITHMS for s in range(7)}
    entries = {}
    for row in rows:
        key = (row["method"], row["algorithm"], row["head_seed"])
        if key in entries:
            raise ValueError(f"Duplicate fusion provenance {key}")
        entries[key] = row
    require_keys(entries, expected, "fusion provenance")
    splits = {r.get("split_manifest_sha256") for r in rows}
    if len(splits) != 1 or None in splits:
        raise ValueError("Fusion methods do not share a verified split hash")
    for method in ("p0", "p2"):
        hashes = {r.get("checkpoint_sha256") for r in rows if r["method"] == method}
        if len(hashes) != 1 or None in hashes:
            raise ValueError(f"Fusion checkpoint mismatch for {method}")
    return entries


def load_fusion():
    fusion_sources = fusion_provenance()
    provenance = mask_provenance("multi_block_n6")
    for method in ("p0", "p2"):
        for algorithm in ALGORITHMS:
            item = fusion_sources[(method, algorithm, 0)]
            if item["checkpoint_sha256"] != provenance[method]["checkpoint_sha256"] or item["split_manifest_sha256"] != provenance[method]["split_manifest_sha256"]:
                raise ValueError(f"Fusion and masking reports use different checkpoint/split for {method}")
    summary = read_csv(FUSION / "fixed_exit_summary.csv", ("method", "algorithm", "task", "exit", "metric", "n", "mean", "per_seed", "higher_is_better"))
    variance = read_csv(FUSION / "fixed_exit_variance_sd.csv", ("method", "algorithm", "task", "exit", "metric", "n", "sample_sd", "sample_variance"))
    raw = read_csv(FUSION / "per_seed_fixed_exit_results.csv", ("method", "algorithm", "task", "exit", "head_seed", "test_value", "metric"))
    keys = ("method", "algorithm", "task", "exit")
    means = index_unique(summary, keys)
    sds = index_unique(variance, keys)
    rawmap = index_unique(raw, (*keys, "head_seed"))
    expected = {(m, a, t, str(e)) for m in ("p0", "p2") for a in ALGORITHMS for t in TASK_ORDER for e in EXITS}
    require_keys(means, expected, "fusion means")
    require_keys(sds, expected, "fusion SDs")
    require_keys(rawmap, {(*k, s) for k in expected for s in SEEDS}, "fusion per-seed")
    result = []
    for key in sorted(expected):
        m, a, task, layer = key
        row, sd = means[key], sds[key]
        validate_metric(row, task)
        validate_metric(sd, task)
        if row["n"] != "7" or sd["n"] != "7":
            raise ValueError(f"Expected seven heads for {key}")
        values = seed_values(row)
        check_close(sd["sample_sd"], statistics.stdev(values.values()), f"SD {key}")
        check_close(sd["sample_variance"], statistics.variance(values.values()), f"variance {key}")
        for seed, value in values.items():
            item = rawmap[(*key, seed)]
            validate_metric(item, task)
            check_close(item["test_value"], value, f"per seed {key}, {seed}")
        _, _, unit, factor = task_metric(task)
        result.append({"method": m, "algorithm": a, "task": task, "exit": int(layer), "metric": row["metric"],
                       "unit": unit, "head_seed_count": 7, "mean": number(row["mean"]) * factor,
                       "sample_sd": number(sd["sample_sd"]) * factor, "per_seed": json.dumps({k: v * factor for k, v in values.items()}, sort_keys=True)})
    return result


def plot_fusion(rows, *, uncertainty):
    configure()
    lookup = {(r["method"], r["algorithm"], r["task"], r["exit"]): r for r in rows}
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 5.85))
    handles = []
    for panel, (task, ax) in enumerate(zip(TASK_ORDER, axes.flat)):
        for method in ("p0", "p2"):
            for alg in ALGORITHMS:
                points = [lookup[(method, alg, task, e)] for e in EXITS]
                marker = {("p0", "current"): "o", ("p0", "single_gelu"): "^", ("p2", "current"): "s", ("p2", "single_gelu"): "D"}[(method, alg)]
                kw = dict(color=COLORS[method], marker=marker, linestyle="-" if alg == "current" else "--", markersize=3.2, linewidth=1.0)
                y = [r["mean"] for r in points]
                if uncertainty:
                    ax.errorbar(EXITS, y, yerr=[r["sample_sd"] for r in points], capsize=2, elinewidth=.7, **kw)
                else:
                    ax.plot(EXITS, y, **kw)
                if panel == 0:
                    handles.append(Line2D([], [], label=f"{METHODS[method]} + {'linear EE' if alg == 'current' else 'fusion EE'}", **kw))
        ax.set_title(f"({chr(97 + panel)}) {TASKS[task]}", pad=4)
        ax.set_xticks(EXITS, [f"L{e}" for e in EXITS])
        ax.set_ylabel("Error (m) ↓" if task == "nrpos-outdoor" else "Macro accuracy (%) ↑")
        ax.grid(axis="y", color="0.88", linewidth=.5)
        ax.tick_params(length=2, pad=2)
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, .995), columnspacing=1.5, handlelength=1.7)
    fig.text(.5, .014, "Encoder exit", ha="center", va="bottom")
    fig.subplots_adjust(left=.075, right=.99, bottom=.085, top=.84, wspace=.40, hspace=.55)
    return fig


def make_fusion():
    rows = load_fusion()
    write_csv(OUT / "data/ee_pretraining_fusion_six_tasks.csv", rows)
    save_fig(plot_fusion(rows, uncertainty=False), OUT / "figures/main/ee_pretraining_fusion_six_tasks")
    save_fig(plot_fusion(rows, uncertainty=True), OUT / "figures/supplementary/ee_pretraining_fusion_uncertainty")
    return "partial", "Means validated; uncertainty is supplementary; some checkpoint schedule metadata are unavailable."


def mask_provenance(condition):
    data = read_json(MASK / condition / "manifest.json")
    if data.get("head_seeds") != list(range(7)) or data.get("split_seed") != 2025:
        raise ValueError(f"Unexpected seed metadata for {condition}")
    output = {}
    all_splits = set()
    for method in METHOD_ORDER:
        entries = data["sources"][method]
        records = [entries[str(i)] for i in range(7)]
        hashes = {r.get("checkpoint_sha256") for r in records}
        splits = {r.get("split_sha256") for r in records}
        if len(hashes) != 1 or None in hashes or len(splits) != 1 or None in splits:
            raise ValueError(f"Unverified checkpoint or split in {condition}/{method}")
        all_splits.update(splits)
        output[method] = {"checkpoint_path": records[0].get("checkpoint"), "checkpoint_sha256": next(iter(hashes)),
                          "checkpoint_source_note": records[0].get("checkpoint_source_note"),
                          "split_manifest_sha256": next(iter(splits)), "pretrain_seed": data.get("pretrain_seed"),
                          "actual_epoch": 64 if "epoch 64 of configured 240" in (records[0].get("checkpoint_source_note") or "") else None,
                          "configured_epochs": 240 if "epoch 64 of configured 240" in (records[0].get("checkpoint_source_note") or "") else None,
                          "mask_geometry": "span_time (nominal, as stated in report)" if condition in ("frequency_n6", "multi_block_n6") else "not verified by checked-in report",
                          "metadata_status": "epoch/schedule and geometry not fully recorded in checked-in reports"}
    if len(all_splits) != 1:
        raise ValueError(f"Split mismatch across methods for {condition}")
    source(MASK / condition / "README.md")
    return output


def make_selected():
    results, provenance = [], {}
    for condition in CONDITIONS:
        provenance[condition] = mask_provenance(condition)
        raw = read_csv(MASK / condition / "multiseed_validation_selected.csv",
                       ("method", "task", "metric", "higher_is_better", "mean", "sample_sd", "per_seed", "selected_exit_by_seed", "selection_frequency"))
        relevant = [r for r in raw if r["method"] in METHOD_ORDER]
        indexed = index_unique(relevant, ("method", "task"))
        expected = {(m, t) for m in METHOD_ORDER for t in TASK_ORDER}
        require_keys(indexed, expected, f"selected {condition}")
        for method in METHOD_ORDER:
            for task in TASK_ORDER:
                row = indexed[(method, task)]
                validate_metric(row, task)
                values = seed_values(row)
                selected = json.loads(row["selected_exit_by_seed"])
                frequency = json.loads(row["selection_frequency"])
                if set(selected) != SEEDS or any(int(v) not in EXITS for v in selected.values()):
                    raise ValueError(f"Invalid selected exits: {condition}/{method}/{task}")
                counts = {str(e): sum(int(x) == e for x in selected.values()) for e in EXITS}
                if {str(k): int(v) for k, v in frequency.items() if int(v)} != {k: v for k, v in counts.items() if v}:
                    raise ValueError(f"Invalid selection frequency: {condition}/{method}/{task}")
                _, higher, unit, factor = task_metric(task)
                results.append({"condition": condition, "method": method, "task": task, "metric": row["metric"],
                                "higher_is_better": higher, "unit": unit, "head_seed_count": 7,
                                "mean": number(row["mean"]) * factor, "sample_sd": number(row["sample_sd"]) * factor,
                                "per_seed": json.dumps({k: v * factor for k, v in values.items()}, sort_keys=True),
                                "selected_exit_by_seed": json.dumps(selected, sort_keys=True), "selection_frequency": json.dumps(counts, sort_keys=True)})
    write_csv(OUT / "tables/pretraining_comparison.csv", results)
    write_csv(OUT / "figures/supplementary/selected_exit_frequencies.csv", [
        {k: r[k] for k in ("condition", "method", "task", "selected_exit_by_seed", "selection_frequency")} for r in results])
    write_text(OUT / "data/checkpoint_provenance.json", json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    lines = []
    for conditions in (CONDITIONS[:2], CONDITIONS[2:]):
        lines += [r"\begin{table*}[t]", r"\centering", r"\caption{Downstream test performance at validation-selected exits. Mean $\pm$ sample SD across seven head seeds on fixed pretrained checkpoints. Accuracy is macro accuracy (\%, higher is better); positioning is distance error (m, lower is better). Bold marks the best observed mean, not statistical significance.}",
                  r"\begin{tabular}{llrrrr}", r"\hline", r"Condition & Task & P0 & P1 & P2 & P2-Reverse \\", r"\hline"]
        for condition in conditions:
            for task in TASK_ORDER:
                group = [r for r in results if r["condition"] == condition and r["task"] == task]
                higher = task_metric(task)[1]
                best = (max if higher else min)(r["mean"] for r in group)
                entries = []
                for method in METHOD_ORDER:
                    r = next(r for r in group if r["method"] == method)
                    precision = 2 if r["unit"] == "%" else 3
                    cell = f"{r['mean']:.{precision}f} $\\pm$ {r['sample_sd']:.{precision}f}"
                    if math.isclose(r["mean"], best, abs_tol=1e-12):
                        cell = r"\textbf{" + cell + "}"
                    entries.append(cell)
                lines.append(" & ".join([latex_escape(condition), latex_escape(TASKS[task]), *entries]) + r" \\")
            lines.append(r"\hline")
        lines += [r"\end{tabular}", r"\end{table*}", ""]
    write_text(OUT / "tables/pretraining_comparison.tex", "\n".join(lines))
    return "partial", "Values validated; some checkpoint epoch, schedule, and mask geometry metadata are unavailable in checked-in reports."


def make_representations(task, main=False):
    directory = REPS / task / "seed_000"
    summary_text = source(directory / "summary.md").read_text(encoding="utf-8")
    if "test" not in summary_text.lower() or "seed" not in summary_text.lower():
        raise ValueError(f"Representation summary does not establish held-out test/seed provenance for {task}")
    metrics = read_csv(directory / "metrics.csv", ("method", "layer", "temporal_columns", "class_separation_ratio"))
    intervals = read_csv(directory / "bootstrap_intervals.csv", ("record_type", "method", "layer", "temporal_columns", "metric", "estimate", "ci_low", "ci_high", "replicates", "seed"))
    selected_columns = 14 if main else 4
    selected = [r for r in metrics if r["temporal_columns"] == str(selected_columns)]
    points = index_unique(selected, ("method", "layer"))
    ci = index_unique([r for r in intervals if r["record_type"] == "method" and r["metric"] == "class_separation_ratio" and r["temporal_columns"] == str(selected_columns)], ("method", "layer"))
    expected = {(m, str(e)) for m in METHOD_ORDER for e in EXITS}
    require_keys(points, expected, f"{task} separation")
    require_keys(ci, expected, f"{task} separation CI")
    rows = []
    for method in METHOD_ORDER:
        for e in EXITS:
            key = (method, str(e))
            row, interval = points[key], ci[key]
            estimate, low, high = map(number, (interval["estimate"], interval["ci_low"], interval["ci_high"]))
            check_close(row["class_separation_ratio"], estimate, f"representation estimate {task}/{key}")
            if low > high or interval["replicates"] != "2000" or interval["seed"] != "2025":
                raise ValueError(f"Invalid bootstrap interval {task}/{key}")
            rows.append({"task": task, "method": method, "exit": e, "temporal_patch_columns": selected_columns,
                         "class_separation_ratio": estimate, "ci_low": low, "ci_high": high,
                         "bootstrap_replicates": 2000, "bootstrap_seed": 2025, "split": "test"})
    if main:
        write_csv(OUT / "data/layerwise_representation_analysis.csv", rows)
    else:
        write_csv(OUT / f"data/representations_{task}.csv", rows)
    configure()
    fig, ax = plt.subplots(figsize=(WIDTH, 4.1))
    styles = [("o", "-"), ("s", "--"), ("^", "-."), ("D", ":")]
    for i, method in enumerate(METHOD_ORDER):
        group = [r for r in rows if r["method"] == method]
        means = [r["class_separation_ratio"] for r in group]
        x = np.arange(4) + (i - 1.5) * .045
        ax.plot(x, means, color=COLORS[method], marker=styles[i][0], linestyle=styles[i][1], label=METHODS[method])
        for xval, r in zip(x, group):
            ax.vlines(xval, r["ci_low"], r["ci_high"], color=COLORS[method], linewidth=1)
            ax.hlines([r["ci_low"], r["ci_high"]], xval - .018, xval + .018, color=COLORS[method], linewidth=1)
    ax.set_xticks(range(4), [f"L{e}" for e in EXITS])
    ax.set_xlabel("Encoder exit")
    ax.set_ylabel("Class separation (higher = more separated)\ncentroid distance / within-class RMS spread")
    ax.grid(axis="y", color="0.88", linewidth=.6)
    ax.set_axisbelow(True)
    axis_label = "time-patch" if task in ("interference", "human-activity") else "input-patch"
    context = f"{TASKS[task]}: {selected_columns} of 14 {axis_label} columns"
    fig.text(.5, .98, context, ha="center", va="top", fontsize=9)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2, frameon=False, columnspacing=1.3)
    fig.subplots_adjust(left=.14, right=.98, bottom=.16, top=.69)
    stem = OUT / ("figures/main/layerwise_representation_analysis" if main else f"figures/supplementary/representations_{task}")
    save_fig(fig, stem)
    return "partial", "Quantitative test-set analysis rendered; PCA has a separate optional artifact status."



def make_interference_trajectory():
    """Show how original-space separation changes with visible input length."""
    directory = REPS / "interference" / "seed_000"
    source(directory / "summary.md")
    metrics = read_csv(directory / "metrics.csv", ("method", "layer", "temporal_columns", "class_separation_ratio"))
    intervals = read_csv(directory / "bootstrap_intervals.csv", ("record_type", "method", "layer", "temporal_columns", "metric", "estimate", "ci_low", "ci_high", "replicates", "seed"))
    points = index_unique(metrics, ("method", "layer", "temporal_columns"))
    ci = index_unique([r for r in intervals if r["record_type"] == "method" and r["metric"] == "class_separation_ratio"], ("method", "layer", "temporal_columns"))
    columns = (2, 4, 8, 14)
    expected = {(m, str(e), str(t)) for m in METHOD_ORDER for e in EXITS for t in columns}
    require_keys(points, expected, "interference trajectory metrics")
    require_keys(ci, expected, "interference trajectory intervals")
    rows = []
    for layer in EXITS:
        for method in METHOD_ORDER:
            for column in columns:
                key = (method, str(layer), str(column))
                interval = ci[key]
                estimate, low, high = map(number, (interval["estimate"], interval["ci_low"], interval["ci_high"]))
                check_close(points[key]["class_separation_ratio"], estimate, f"trajectory {key}")
                if low > high or interval["replicates"] != "2000" or interval["seed"] != "2025":
                    raise ValueError(f"Invalid trajectory interval {key}")
                rows.append({"task": "interference", "method": method, "exit": layer,
                             "temporal_patch_columns": column, "class_separation_ratio": estimate,
                             "ci_low": low, "ci_high": high, "bootstrap_replicates": 2000,
                             "bootstrap_seed": 2025, "split": "test"})
    write_csv(OUT / "data/interference_separation_vs_input.csv", rows)
    configure()
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.9), sharex=True, sharey=True)
    styles = [("o", "-"), ("s", "--"), ("^", "-."), ("D", ":")]
    for panel, (layer, ax) in enumerate(zip(EXITS, axes.flat)):
        for i, method in enumerate(METHOD_ORDER):
            group = [r for r in rows if r["method"] == method and r["exit"] == layer]
            ax.plot(columns, [r["class_separation_ratio"] for r in group], color=COLORS[method],
                    marker=styles[i][0], linestyle=styles[i][1], label=METHODS[method])
        ax.set_title(f"({chr(97 + panel)}) Exit L{layer}", pad=4)
        ax.set_xticks(columns)
        ax.grid(axis="y", color="0.88", linewidth=.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, .99), columnspacing=1.3)
    fig.text(.5, .02, "Visible temporal patch columns (of 14)", ha="center", va="bottom")
    fig.text(.015, .48, "Class centroid distance / within-class RMS spread", rotation="vertical", ha="left", va="center")
    fig.subplots_adjust(left=.16, right=.98, bottom=.13, top=.78, wspace=.18, hspace=.40)
    save_fig(fig, OUT / "figures/supplementary/interference_separation_vs_input")
    return "complete", "Original-space separation versus input length; bootstrap intervals included in plotted-data CSV."




def make_temporal_performance(task="interference"):
    """Plot interference accuracy with full-input-trained frozen heads."""
    if task not in ("interference", "human-activity", "rfs", "jamming"):
        raise ValueError(f"Unsupported temporal task: {task}")
    directory = REPS / task / "seed_000"
    stem = task.replace("-", "_")
    classes = {"interference": 2, "human-activity": 6, "rfs": 20, "jamming": 6}[task]
    temporal = task in ("interference", "human-activity")
    summary = source(directory / "temporal_head_robustness_summary.md").read_text(encoding="utf-8").lower()
    required_notes = (("heads were trained on tp14 only", "same saved head", "five independent head-seed", "2025", "2029", "1,400")
                      if task == "interference" else
                      ("same full-information", "temporal prefixes", "five head seeds", "class 5: 20") if task == "human-activity" else
                      ("same full-information", "input-column prefixes", "five head seeds", "class 19: 29") if task == "rfs" else
                      ("same full-information", "input-column prefixes", "five head seeds", "class 5: 125"))
    if any(note not in summary for note in required_notes):
        raise ValueError("Temporal robustness summary does not establish the expected head and evaluation protocol")
    raw = read_csv(directory / "temporal_head_robustness.csv", ("method", "layer", "temporal_columns", "mean_macro_accuracy", "std_macro_accuracy_across_head_seeds", "min_macro_accuracy", "max_macro_accuracy", *(f"mean_class{i}_accuracy" for i in range(classes)), "max_repeat_prediction_disagreement"))
    source_rows = index_unique(raw, ("method", "layer", "temporal_columns"))
    columns = (2, 4, 8, 14)
    expected = {(m, str(e), str(t)) for m in METHOD_ORDER for e in EXITS for t in columns}
    require_keys(source_rows, expected, "temporal head robustness")
    rows = []
    for layer in EXITS:
        for method in METHOD_ORDER:
            for column in columns:
                item = source_rows[(method, str(layer), str(column))]
                mean = number(item["mean_macro_accuracy"])
                sd = number(item["std_macro_accuracy_across_head_seeds"])
                lower, upper = number(item["min_macro_accuracy"]), number(item["max_macro_accuracy"])
                check_close(mean, sum(number(item[f"mean_class{i}_accuracy"]) for i in range(classes)) / classes, "macro accuracy")
                if not (0 <= lower <= mean + 1e-9 <= upper + 1e-9 <= 1 + 1e-9) or sd < 0:
                    raise ValueError(f"Invalid accuracy or standard deviation for {method}, L{layer}, Tp{column}")
                if number(item["max_repeat_prediction_disagreement"]) != 0:
                    raise ValueError(f"Repeated inference disagreement for {method}, L{layer}, Tp{column}")
                rows.append({"task": task, "method": method, "exit": layer,
                             "temporal_patch_columns": column, "metric": "macro_accuracy", "unit": "%",
                             "mean": mean * 100, "sample_sd": sd * 100, "min": lower * 100, "max": upper * 100,
                             "head_seed_count": 5, "head_seeds": "2025-2029" if task == "interference" else "not recorded in checked-in summary", "head_training_columns": 14,
                             "split": "test", "checkpoint_seed": 0})
    if not temporal:
        for row in rows:
            row["input_patch_columns"] = row.pop("temporal_patch_columns")
    axis = "time" if temporal else "input"
    write_csv(OUT / f"data/{stem}_accuracy_vs_{axis}_patches.csv", rows)
    configure()
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.1), sharex=True, sharey=True)
    styles = [("o", "-"), ("s", "--"), ("^", "-."), ("D", ":")]
    for panel, (layer, ax) in enumerate(zip(EXITS, axes.flat)):
        ax.axhline(100 / classes, color="0.55", linewidth=.8, linestyle=(0, (2, 3)), zorder=0)
        for i, method in enumerate(METHOD_ORDER):
            group = [r for r in rows if r["method"] == method and r["exit"] == layer]
            ax.errorbar(columns, [r["mean"] for r in group], yerr=[r["sample_sd"] for r in group],
                        color=COLORS[method], marker=styles[i][0], linestyle=styles[i][1],
                        linewidth=1.1, markersize=3.7, elinewidth=.7, capsize=2, label=METHODS[method])
        ax.set_title(f"({chr(97 + panel)}) Exit L{layer}", pad=4)
        ax.set_xticks(columns)
        ax.set_ylim(0, max(100, max(r["mean"] + r["sample_sd"] for r in rows) + 1))
        ax.set_yticks((0, 25, 50, 75, 100))
        ax.grid(axis="y", color="0.88", linewidth=.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, .99), columnspacing=1.3)
    fig.text(.5, .025, f"Visible {'temporal' if temporal else 'input'} patch columns (of 14)", ha="center", va="bottom")
    fig.text(.025, .46, f"{TASKS[task]} macro accuracy (%)", rotation="vertical", ha="left", va="center")
    fig.subplots_adjust(left=.13, right=.98, bottom=.13, top=.79, wspace=.18, hspace=.36)
    save_fig(fig, OUT / f"figures/main/{stem}_accuracy_vs_{axis}_patches")
    return "complete", "Five full-input-trained head seeds; mean +/- sample SD across head seeds at each input length."


def make_paired_representation(task="interference"):
    """Show original-space separation and paired differences from LatentWave."""
    if task not in ("interference", "human-activity", "rfs", "jamming"):
        raise ValueError(f"Unsupported representation task: {task}")
    directory = REPS / task / "seed_000"
    stem = task.replace("-", "_")
    source(directory / "summary.md")
    metrics = read_csv(directory / "metrics.csv", ("method", "layer", "temporal_columns", "class_separation_ratio"))
    intervals = read_csv(directory / "bootstrap_intervals.csv", ("record_type", "method", "reference_method", "layer", "temporal_columns", "metric", "estimate", "ci_low", "ci_high", "difference", "difference_low", "difference_high", "replicates", "seed"))
    points = index_unique([r for r in metrics if r["temporal_columns"] == "14"], ("method", "layer"))
    absolute = index_unique([r for r in intervals if r["record_type"] == "method" and r["metric"] == "class_separation_ratio" and r["temporal_columns"] == "14"], ("method", "layer"))
    comparisons = index_unique([r for r in intervals if r["record_type"] == "comparison" and r["metric"] == "class_separation_ratio" and r["temporal_columns"] == "14" and r["reference_method"] == "p0"], ("method", "layer"))
    expected = {(m, str(e)) for m in METHOD_ORDER for e in EXITS}
    require_keys(points, expected, "paired representation point estimates")
    require_keys(absolute, expected, "paired representation absolute intervals")
    require_keys(comparisons, {(m, str(e)) for m in METHOD_ORDER if m != "p0" for e in EXITS}, "paired representation differences")
    rows = []
    for method in METHOD_ORDER:
        for layer in EXITS:
            key = (method, str(layer))
            row = absolute[key]
            estimate, low, high = map(number, (row["estimate"], row["ci_low"], row["ci_high"]))
            check_close(points[key]["class_separation_ratio"], estimate, f"absolute separation {key}")
            if low > high or row["replicates"] != "2000" or row["seed"] != "2025":
                raise ValueError(f"Invalid bootstrap interval {key}")
            rows.append({"task": task, "panel": "absolute", "method": method, "reference_method": "", "layer": layer,
                         "temporal_patch_columns": 14, "estimate": estimate, "ci_low": low, "ci_high": high,
                         "interval": "95% class-stratified paired bootstrap", "bootstrap_replicates": 2000,
                         "bootstrap_seed": 2025, "split": "test"})
            if method != "p0":
                comparison = comparisons[key]
                delta, delta_low, delta_high = map(number, (comparison["difference"], comparison["difference_low"], comparison["difference_high"]))
                check_close(comparison["estimate"], estimate, f"comparison method estimate {key}")
                check_close(delta, estimate - number(points[("p0", str(layer))]["class_separation_ratio"]), f"paired difference {key}")
                if delta_low > delta_high or comparison["replicates"] != "2000" or comparison["seed"] != "2025":
                    raise ValueError(f"Invalid paired interval {key}")
                rows.append({"task": task, "panel": "difference", "method": method, "reference_method": "p0", "layer": layer,
                             "temporal_patch_columns": 14, "estimate": delta, "ci_low": delta_low, "ci_high": delta_high,
                             "interval": "95% class-stratified paired bootstrap", "bootstrap_replicates": 2000,
                             "bootstrap_seed": 2025, "split": "test"})
    if task in ("rfs", "jamming"):
        for row in rows:
            row["input_patch_columns"] = row.pop("temporal_patch_columns")
    write_csv(OUT / f"data/{stem}_paired_separation.csv", rows)
    configure()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 4.25), sharey=True, gridspec_kw={"width_ratios": [1, 1]})
    markers = {"p0": "o", "p1": "s", "p2": "^", "p2_reverse": "D"}
    offsets = {"p0": .24, "p1": .08, "p2": -.08, "p2_reverse": -.24}
    for ax, panel in zip(axes, ("absolute", "difference")):
        for i in range(4):
            if i % 2 == 0:
                ax.axhspan(i-.48, i+.48, color="0.965", zorder=0)
        if panel == "difference":
            ax.axvline(0, color="0.25", linestyle="--", linewidth=.8, zorder=1)
        for row in rows:
            if row["panel"] != panel:
                continue
            method = row["method"]
            y = (3 - EXITS.index(row["layer"])) + offsets[method]
            ax.hlines(y, row["ci_low"], row["ci_high"], color=COLORS[method], linewidth=1.2, zorder=2)
            ax.plot([row["ci_low"], row["ci_high"]], [y, y], linestyle="none", marker="|", color=COLORS[method], markersize=5, zorder=2)
            ax.plot(row["estimate"], y, linestyle="none", marker=markers[method], color=COLORS[method], markersize=4, zorder=3)
        ax.set_ylim(-.55, 3.55)
        ax.set_yticks(range(4), ["L8", "L6", "L4", "L2"])
        ax.grid(axis="x", color="0.87", linewidth=.5)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
    axes[0].set_title("(a) Class separation", fontsize=9)
    axes[1].set_title("(b) Difference from LatentWave", fontsize=9)
    axes[0].set_xlabel("Centroid distance / class spread")
    axes[1].set_xlabel("Change in class-separation ratio")
    axes[0].set_ylabel("Encoder exit")
    handles = [Line2D([0], [0], color=COLORS[m], marker=markers[m], linestyle="none", label=METHODS[m], markersize=5) for m in METHOD_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, .99), columnspacing=1.3)
    fig.text(.5, .845, f"{TASKS[task]}: full input (14 {'temporal' if task in ('interference', 'human-activity') else 'input'} patch columns)", ha="center", va="center", fontsize=9)
    if task == "rfs":
        fig.text(.5, .79, "Bootstrap intervals may exclude point estimates", ha="center", va="center", fontsize=8)
    fig.subplots_adjust(left=.09, right=.98, bottom=.16, top=.70 if task == "rfs" else .76, wspace=.16)
    save_fig(fig, OUT / f"figures/main/{stem}_paired_separation")
    return "complete", f"Original-space class separation and paired 95% bootstrap differences at all 14 {'input patch' if task in ('rfs', 'jamming') else 'temporal'} columns."



def make_nrpos_depth():
    """Plot fixed-depth held-out positioning performance from M-block reports."""
    provenance = mask_provenance("multi_block_n6")
    raw = read_csv(MASK / "multi_block_n6/multiseed_fixed_exits.csv", ("method", "task", "metric", "higher_is_better", "exit", "mean", "sample_sd", "per_seed"))
    selected = [r for r in raw if r["task"] == "nrpos-outdoor" and r["method"] in METHOD_ORDER]
    indexed = index_unique(selected, ("method", "exit"))
    require_keys(indexed, {(m, str(e)) for m in METHOD_ORDER for e in EXITS}, "NR positioning fixed exits")
    rows = []
    for method in METHOD_ORDER:
        for layer in EXITS:
            item = indexed[(method, str(layer))]
            validate_metric(item, "nrpos-outdoor")
            values = seed_values(item)
            rows.append({"task": "nrpos-outdoor", "method": method, "exit": layer,
                         "mean_distance_error_m": number(item["mean"]), "sample_sd_m": number(item["sample_sd"]),
                         "head_seed_count": 7, "head_seeds": "0-6", "per_seed": json.dumps(values, sort_keys=True),
                         "split": "test", "mask_condition": "multi_block_n6", "checkpoint_sha256": provenance[method]["checkpoint_sha256"]})
    write_csv(OUT / "data/nrpos_full_input_error_vs_exit.csv", rows)
    configure()
    fig, ax = plt.subplots(figsize=(WIDTH, 3.8))
    styles = [("o", "-"), ("s", "--"), ("^", "-."), ("D", ":")]
    for i, method in enumerate(METHOD_ORDER):
        group = [r for r in rows if r["method"] == method]
        ax.errorbar(EXITS, [r["mean_distance_error_m"] for r in group], yerr=[r["sample_sd_m"] for r in group],
                    color=COLORS[method], marker=styles[i][0], linestyle=styles[i][1],
                    linewidth=1.1, markersize=4, capsize=2, label=METHODS[method])
    ax.set_xticks(EXITS, [f"L{e}" for e in EXITS])
    ax.set_xlabel("Encoder exit")
    ax.set_ylabel("Mean position error (m); lower is better")
    ax.grid(axis="y", color="0.88", linewidth=.5)
    fig.text(.5, .98, "NR outdoor positioning: full input, held-out test", ha="center", va="top", fontsize=9)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2, frameon=False)
    fig.subplots_adjust(left=.12, right=.98, bottom=.16, top=.67)
    save_fig(fig, OUT / "figures/main/nrpos_full_input_error_vs_exit")
    return "partial", "Held-out full-input M-block results, seven head seeds; some checkpoint schedule metadata unavailable."


def make_nrpos_time():
    """Plot validation-only NR fixed-exit results from the time_n6 packet."""
    base = ROOT / "experiments/original_data/vary_time_samples/nrpos-outdoor"
    source(base / "README.md")
    packets = {}
    source_rows = {}
    columns = (2, 4, 8, 14)
    for method, folder in (("p0", "latentwave_ee"), ("p2", "hierarchical")):
        packet = read_json(base / folder / "input_information_exit_metrics.json")
        if (packet.get("stage") != "selection_complete" or packet.get("test_evaluation") is not False
                or packet["protocol"].get("selection_uses") != "validation_only"
                or packet["protocol"].get("foundation_weights_updated") is not False
                or packet["protocol"].get("downstream_head_weights_updated") is not False
                or packet["provenance"].get("source_method") != method):
            raise ValueError(f"NR packet does not establish frozen validation-only protocol for {method}")
        study = packet["studies"]["temporal-nrpos-outdoor"]
        if study.get("availability_mode") != "leading_temporal_patches" or study.get("task") != "nrpos-outdoor":
            raise ValueError(f"NR input study differs for {method}")
        conditions = {int(r["temporal_patch_columns"]): r for r in study["conditions"]}
        if len(conditions) != len(study["conditions"]) or set(conditions) != set(columns):
            raise ValueError(f"NR packet missing or duplicates input conditions for {method}")
        csv_rows = read_csv(base / folder / "input_information_temporal.csv", ("task", "temporal_patch_columns", "input_shape", "token_count", *(f"validation_L{e}_mean_distance_error" for e in EXITS)))
        indexed = index_unique(csv_rows, ("task", "temporal_patch_columns"))
        require_keys(indexed, {("nrpos-outdoor", str(t)) for t in columns}, "NR time CSV")
        for column in columns:
            condition, row = conditions[column], indexed[("nrpos-outdoor", str(column))]
            if condition["input_shape"] != [4, 224, 16 * column] or int(row["token_count"]) != int(condition["token_count"]):
                raise ValueError(f"NR input shape or token count mismatch for {method}/Tp{column}")
            for layer in EXITS:
                value = number(condition["validation"][str(layer)]["mean_distance_error"])
                check_close(row[f"validation_L{layer}_mean_distance_error"], value, "NR CSV/JSON validation metric")
                if value < 0:
                    raise ValueError("Position error must be non-negative")
                source_rows[(method, layer, column)] = {"task": "nrpos-outdoor", "method": method, "exit": layer,
                    "temporal_patch_columns": column, "mean_distance_error_m": value, "split": "validation",
                    "head_seed_count": 1, "head_seed": "not recorded", "mask_condition": "time_n6",
                    "input_shape": row["input_shape"], "token_count": int(row["token_count"]),
                    "checkpoint_path": packet["provenance"]["checkpoint"]["path"],
                    "checkpoint_sha256": packet["provenance"]["checkpoint"].get("sha256", "not recorded"),
                    "split_manifest_sha256": packet["provenance"]["split_manifest"].get("sha256", "")}
        packets[method] = packet
    splits = {p["provenance"]["split_manifest"].get("sha256") for p in packets.values()}
    if len(splits) != 1 or not next(iter(splits)):
        raise ValueError("NR time packets do not share a verified split hash")
    rows = [source_rows[(m, e, t)] for e in EXITS for m in ("p0", "p2") for t in columns]
    write_csv(OUT / "data/nrpos_validation_error_vs_time_patches.csv", rows)
    configure()
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.0), sharex=True, sharey=True)
    for panel, (layer, ax) in enumerate(zip(EXITS, axes.flat)):
        for method, marker, ls in (("p0", "o", "-"), ("p2", "^", "--")):
            group = [r for r in rows if r["method"] == method and r["exit"] == layer]
            ax.plot(columns, [r["mean_distance_error_m"] for r in group], color=COLORS[method],
                    marker=marker, linestyle=ls, label=METHODS[method])
        ax.set_title(f"({chr(97 + panel)}) Exit L{layer}", pad=4)
        ax.set_xticks(columns)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="0.88", linewidth=.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, .99))
    fig.text(.5, .89, "NR outdoor positioning: validation set", ha="center", va="center", fontsize=9)
    fig.text(.5, .025, "Visible temporal patch columns (of 14)", ha="center", va="bottom")
    fig.text(.025, .46, "Mean position error (m); lower is better", rotation="vertical", ha="left", va="center")
    fig.subplots_adjust(left=.13, right=.98, bottom=.13, top=.79, wspace=.18, hspace=.36)
    save_fig(fig, OUT / "figures/main/nrpos_validation_error_vs_time_patches")
    return "partial", "Validation only; one saved head per exit and method, fine-to-coarse epoch 29, LatentWave horizon and checkpoint hashes unverified."


def load_parameters():
    rows = read_csv(ABLATION / "parameter_counts.csv", ("method", "adapter_variant", "task", "exit", "total", "adapter", "gate", "output"))
    result = index_unique(rows, ("method", "adapter_variant", "task", "exit"))
    expected = {(m, a, t, str(e)) for m in ("p0", "p2") for a in ("current", "bottleneck_gelu", "bottleneck_linear", "bottleneck_sigmoid", "single_linear", "single_gelu", "single_sigmoid") for t in TASK_ORDER for e in EXITS}
    require_keys(result, expected, "parameter counts")
    for key, row in result.items():
        if int(row["total"]) != sum(int(row[k]) for k in ("adapter", "gate", "output")):
            raise ValueError(f"Parameter counts do not sum for {key}")
    return result


def ablation_provenance():
    data = read_json(ABLATION / "manifest.json")
    fusion = fusion_provenance()
    if data.get("head_seed") != 0 or data.get("pretrain_seed") != 0:
        raise ValueError("Unexpected ablation seeds")
    expected = {(m, a) for m in ("p0", "p2") for a in ("current", "bottleneck_gelu", "bottleneck_linear", "bottleneck_sigmoid", "single_linear", "single_gelu", "single_sigmoid")}
    entries = index_unique(data["sources"], ("method", "algorithm"))
    require_keys(entries, expected, "ablation provenance")
    for (method, _), row in entries.items():
        expected_source = fusion[(method, "current", 0)]
        if row.get("checkpoint_sha256") != expected_source.get("checkpoint_sha256") or row.get("split_manifest_sha256") != expected_source.get("split_manifest_sha256"):
            raise ValueError(f"Ablation checkpoint/split differs for {method}")
    return entries


def make_ablation():
    ablation_provenance()
    params = load_parameters()
    raw = read_csv(ABLATION / "validation_selected_results.csv", ("method", "algorithm", "label", "task", "metric", "selected_exit_best_validation", "validation_value", "selected_test_value"))
    entries = index_unique(raw, ("method", "algorithm", "task"))
    paired = read_csv(FUSION / "per_seed_fixed_exit_results.csv", ("method", "algorithm", "head_seed", "task", "exit", "test_value", "selected_by_validation"))
    seed_zero = index_unique([r for r in paired if r["head_seed"] == "0" and r["selected_by_validation"].lower() == "true"], ("method", "algorithm", "task"))
    algorithms = ("current", "bottleneck_gelu", "bottleneck_linear", "bottleneck_sigmoid", "single_linear", "single_gelu", "single_sigmoid")
    require_keys(entries, {(m, a, t) for m in ("p0", "p2") for a in algorithms for t in TASK_ORDER}, "ablation")
    rows = []
    for method in ("p0", "p2"):
        for task in TASK_ORDER:
            for alg in algorithms:
                row = entries[(method, alg, task)]
                validate_metric(row, task)
                exit_layer = int(row["selected_exit_best_validation"])
                if alg in ALGORITHMS:
                    matching = seed_zero[(method, alg, task)]
                    if int(matching["exit"]) != exit_layer:
                        raise ValueError(f"Ablation and multiseed selection differ: {method}/{alg}/{task}")
                    check_close(matching["test_value"], row["selected_test_value"], f"ablation test {method}/{alg}/{task}")
                if exit_layer not in EXITS:
                    raise ValueError("Invalid ablation exit")
                param = params[(method, alg, task, str(exit_layer))]
                _, _, unit, factor = task_metric(task)
                rows.append({"method": method, "algorithm": alg, "task": task, "metric": row["metric"], "unit": unit,
                             "selected_exit": exit_layer, "validation_value": number(row["validation_value"]) * factor,
                             "test_value": number(row["selected_test_value"]) * factor,
                             "head_parameters": int(param["total"]), "head_seed": 0})
    write_csv(OUT / "tables/adapter_ablation.csv", rows)
    labels = {"current":"Base", "bottleneck_gelu":"B-GELU", "bottleneck_linear":"B-Lin", "bottleneck_sigmoid":"B-Sig", "single_linear":"S-Lin", "single_gelu":"S-GELU", "single_sigmoid":"S-Sig"}
    lines = []
    for method in ("p0", "p2"):
        lines += [r"\begin{table*}[t]", r"\centering", r"\caption{" + method.upper() + r" adapter pilot: selected test performance (exit in parentheses), head seed 0. Macro accuracy is percent; positioning error is metres. B and S denote bottleneck and single-layer adapters; Lin and Sig denote linear and sigmoid. Head parameter counts are in the accompanying CSV.}",
                  r"\begingroup\small\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{lrrrrrrr}", r"\hline", "Task & " + " & ".join(labels[a] for a in algorithms) + r" \\", r"\hline"]
        for task in TASK_ORDER:
            cells = []
            for alg in algorithms:
                r = next(r for r in rows if (r["method"], r["task"], r["algorithm"]) == (method, task, alg))
                digits = 2 if r["unit"] == "%" else 3
                cells.append(f"{r['test_value']:.{digits}f} (L{r['selected_exit']})")
            lines.append(" & ".join([latex_escape(TASKS[task]), *cells]) + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\endgroup", r"\end{table*}", ""]
    write_text(OUT / "tables/adapter_ablation.tex", "\n".join(lines))
    return "complete", "Single-head-seed validation-selected pilot; parameter counts joined at the selected exit."


def load_matched_benchmark(path=None):
    """Validate all measurements before allowing any performance/cost join."""
    path = Path(path or BENCHMARK)
    data = read_json(path)
    protocol = data.get("protocol", {})
    if data.get("schema_version") != 1 or data.get("study") != "multi_block_n6_matched_exit":
        raise ValueError("Incompatible matched benchmark schema or study")
    required = ("batch_size", "precision", "matmul_precision", "allow_tf32_matmul", "allow_tf32_cudnn",
                "warmup_iterations", "measure_iterations", "synchronize_each_inference", "flop_counter",
                "flop_convention", "flop_exclusions", "device", "gpu_name", "gpu_memory_bytes",
                "torch_version", "cuda_version", "python_version", "revision", "working_tree_dirty",
                "input_source", "head_execution")
    if any(k not in protocol or protocol[k] is None or protocol[k] == "" for k in required):
        raise ValueError("Benchmark protocol has missing metadata")
    if (protocol["batch_size"] != 1 or protocol["precision"] != "float32" or
        not str(protocol["device"]).startswith("cuda") or protocol["synchronize_each_inference"] is not True or
        protocol["warmup_iterations"] < 1 or protocol["measure_iterations"] < 2 or
        protocol["flop_counter"] != "torch.utils.flop_counter.FlopCounterMode" or
        protocol["input_source"] != "authenticated original seed-0 evaluation packets"):
        raise ValueError("Benchmark protocol incompatible with matched comparison")
    if protocol['working_tree_dirty'] is not False:
        raise ValueError("Benchmark code revision must have a clean working tree")
    provenance = fusion_provenance()
    params = load_parameters()
    entries = index_unique(data.get("entries", []), ("method", "algorithm", "task", "exit"))
    expected = {(m, a, t, e) for m in ("p0", "p2") for a in ALGORITHMS for t in TASK_ORDER for e in EXITS}
    require_keys(entries, expected, "matched benchmark")
    shapes = {}
    for key, row in entries.items():
        m, alg, task, layer = key
        expected_source = provenance[(m, alg, 0)]
        if row.get("checkpoint_sha256") != expected_source["checkpoint_sha256"] or row.get("packet_sha256") != expected_source["metrics_sha256"]:
            raise ValueError(f"Benchmark checkpoint or source packet mismatch: {key}")
        shape = row.get("input_shape")
        outputs = row.get("num_outputs")
        if not isinstance(shape, list) or len(shape) < 2 or shape[0] != 1 or any(not isinstance(x, int) or x <= 0 for x in shape) or not isinstance(outputs, int) or outputs < 1:
            raise ValueError(f"Invalid input/output shape: {key}")
        spec = (tuple(shape), outputs)
        if task in shapes and shapes[task] != spec:
            raise ValueError(f"Unmatched task shape or outputs: {task}")
        shapes[task] = spec
        if row.get("head_parameters") != int(params[(m, alg, task, str(layer))]["total"]):
            raise ValueError(f"Head parameter mismatch: {key}")
        if row.get("head_weights") not in ("initialized", "trained_seed_0") or (row["head_weights"] == "trained_seed_0" and not row.get("head_sha256")):
            raise ValueError(f"Head weight provenance invalid: {key}")
        check = row.get("execution", {})
        expected_fusion = [e for e in EXITS if e <= layer] if alg == "single_gelu" else []
        if check.get("executed_blocks") != list(range(1, layer + 1)) or check.get("fusion_layers") != expected_fusion:
            raise ValueError(f"Encoder prefix or fusion source mismatch: {key}")
        if number(row.get("flops", 0)) <= 0:
            raise ValueError(f"Invalid FLOP count: {key}")
        operators = row.get("counted_operators")
        if not isinstance(operators, dict) or not operators or any(number(x) < 0 for x in operators.values()) or sum(number(x) for x in operators.values()) != number(row["flops"]):
            raise ValueError(f"Missing or inconsistent FLOP operator coverage: {key}")
        latency = row.get("latency", {})
        for field in ("mean_ms", "median_ms", "standard_deviation_ms", "p95_ms"):
            if field not in latency or number(latency[field]) < 0 or (field != "standard_deviation_ms" and number(latency[field]) == 0):
                raise ValueError(f"Invalid latency {field}: {key}")
    return entries, protocol


def make_selected_cost():
    if not BENCHMARK.is_file():
        return "unavailable", f"Matched benchmark absent: {BENCHMARK}; no selected-exit costs generated."
    bench, protocol = load_matched_benchmark()
    selected = read_csv(FUSION / "per_seed_validation_selected_results.csv", ("method", "algorithm", "head_seed", "task", "metric", "higher_is_better", "selected_exit", "selected_test_value"))
    fixed = read_csv(FUSION / "per_seed_fixed_exit_results.csv", ("method", "algorithm", "head_seed", "task", "exit", "metric", "test_value", "selected_by_validation"))
    selected = index_unique(selected, ("method", "algorithm", "task", "head_seed"))
    fixed = index_unique(fixed, ("method", "algorithm", "task", "exit", "head_seed"))
    expected = {(m, a, t, seed) for m in ("p0", "p2") for a in ALGORITHMS for t in TASK_ORDER for seed in SEEDS}
    require_keys(selected, expected, "selected performance")
    require_keys(fixed, {(*k[:3], str(e), k[3]) for k in expected for e in EXITS}, "fixed performance")
    benchmark_rows = []
    for key, r in sorted(bench.items()):
        benchmark_rows.append({k: r[k] for k in ("method", "algorithm", "task", "exit", "num_outputs", "head_parameters", "checkpoint_sha256", "packet_sha256", "head_weights", "head_sha256", "flops")} | {
            "input_shape": json.dumps(r["input_shape"]), **r["latency"], "benchmark_sha256": sha(BENCHMARK), "counted_operators": json.dumps(r.get("counted_operators", {}), sort_keys=True)})
    per_seed, summary = [], []
    for method in ("p0", "p2"):
        for alg in ALGORITHMS:
            for task in TASK_ORDER:
                metric, higher, unit, factor = task_metric(task)
                costs, values, frequencies = [], [], {str(e): 0 for e in EXITS}
                ref_values = []
                for seed in sorted(SEEDS):
                    row = selected[(method, alg, task, seed)]
                    validate_metric(row, task)
                    layer = int(row["selected_exit"])
                    if layer not in EXITS:
                        raise ValueError(f"Invalid selected exit: {row}")
                    match = fixed[(method, alg, task, str(layer), seed)]
                    if match["metric"] != metric or match["selected_by_validation"].lower() != "true":
                        raise ValueError("Selected exit does not match fixed-exit source")
                    check_close(row["selected_test_value"], match["test_value"], "selected held-out value")
                    cost = bench[(method, alg, task, layer)]["flops"]
                    l8 = bench[(method, alg, task, 8)]["flops"]
                    reference = fixed[("p0", "current", task, "8", seed)]
                    validate_metric(reference, task)
                    value = number(row["selected_test_value"]) * factor
                    costs.append(cost); values.append(value); frequencies[str(layer)] += 1
                    ref_values.append(number(reference["test_value"]) * factor)
                    per_seed.append({"method": method, "algorithm": alg, "task": task, "head_seed": seed,
                        "selected_exit": layer, "metric": metric, "unit": unit, "selected_test_value": value,
                        "selected_flops": cost, "same_head_l8_flops": l8})
                avg = statistics.mean(costs)
                l8 = bench[(method, alg, task, 8)]["flops"]
                summary.append({"method": method, "algorithm": alg, "task": task, "metric": metric,
                    "direction": "higher" if higher else "lower", "unit": unit, "head_seed_count": 7,
                    "exit_frequencies": json.dumps(frequencies, sort_keys=True), "test_mean": statistics.mean(values),
                    "test_sample_sd": statistics.stdev(values), "selected_mean_gflops": avg / 1e9,
                    "same_head_l8_gflops": l8 / 1e9, "reduction_vs_same_head_l8_percent": 100 * (1 - avg / l8),
                    "latentwave_baseline_l8_test_mean": statistics.mean(ref_values),
                    "latentwave_baseline_l8_test_sample_sd": statistics.stdev(ref_values),
                    "latentwave_baseline_l8_gflops": bench[("p0", "current", task, 8)]["flops"] / 1e9})
    write_csv(OUT / "data/matched_exit_benchmarks.csv", benchmark_rows)
    write_csv(OUT / "data/selected_exit_cost_by_seed.csv", per_seed)
    write_csv(OUT / "tables/selected_exit_performance_cost.csv", summary)
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{Held-out test performance at validation-selected fixed exits, mean $\pm$ sample SD over seven downstream heads. GFLOPs are averaged over selected exits; reduction uses the same encoder and head at L8. L8 reference is LatentWave with the baseline head. No per-sample dynamic stopping is measured.}", r"\begin{tabular}{lllrrrrr}", r"\hline", r"Task & Encoder/head & L2/L4/L6/L8 & Test metric & GFLOPs & Reduction (\%) & Reference L8 metric & Reference GFLOPs \\", r"\hline"]
    for r in summary:
        freq = json.loads(r['exit_frequencies'])
        label = ('LatentWave' if r['method'] == 'p0' else 'Fine-to-coarse') + (' / baseline' if r['algorithm'] == 'current' else ' / fusion')
        digits = 3 if r['unit'] == 'm' else 2
        metric = f"{r['test_mean']:.{digits}f} $\\pm$ {r['test_sample_sd']:.{digits}f} " + (r"\%" if r['unit'] == '%' else 'm')
        reference = f"{r['latentwave_baseline_l8_test_mean']:.{digits}f} $\\pm$ {r['latentwave_baseline_l8_test_sample_sd']:.{digits}f}"
        lines.append(' & '.join([latex_escape(TASKS[r['task']]), label, '/'.join(str(freq[str(e)]) for e in EXITS), metric,
            f"{r['selected_mean_gflops']:.3f}", f"{r['reduction_vs_same_head_l8_percent']:.1f}", reference,
            f"{r['latentwave_baseline_l8_gflops']:.3f}"]) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write_text(OUT / "tables/selected_exit_performance_cost.tex", '\n'.join(lines) + '\n')
    return "complete", "Matched fixed-exit costs averaged over seven validation-selected heads; no dynamic stopping."


def make_historical_selected_cost():
    """Expose reported baseline costs without inventing fusion measurements."""
    provenance = fusion_provenance()
    mask = mask_provenance("multi_block_n6")
    manifest = read_json(MASK / "multi_block_n6/manifest.json")
    cost = read_csv(MASK / "multi_block_n6/multiseed_inference_cost.csv", ("method", "task", "exit", "flops", "latency_mean_ms", "latency_sd_ms", "checkpoint_sha256", "benchmark_sha256"))
    costs = index_unique([r for r in cost if r["method"] in ("p0", "p2")], ("method", "task", "exit"))
    require_keys(costs, {(m, t, str(e)) for m in ("p0", "p2") for t in TASK_ORDER for e in EXITS}, "historical baseline costs")
    selected = index_unique(read_csv(FUSION / "per_seed_validation_selected_results.csv", ("method", "algorithm", "task", "head_seed", "metric", "higher_is_better", "selected_exit", "selected_test_value")), ("method", "algorithm", "task", "head_seed"))
    fixed = index_unique(read_csv(FUSION / "per_seed_fixed_exit_results.csv", ("method", "algorithm", "task", "exit", "head_seed", "metric", "higher_is_better", "test_value", "selected_by_validation")), ("method", "algorithm", "task", "exit", "head_seed"))
    expected = {(m, a, t, seed) for m in ("p0", "p2") for a in ALGORITHMS for t in TASK_ORDER for seed in SEEDS}
    require_keys(selected, expected, "historical selected performance")
    require_keys(fixed, {(*k[:3], str(e), k[3]) for k in expected for e in EXITS}, "historical fixed performance")
    for (method, task, layer), row in costs.items():
        source = provenance[(method, "current", 0)]
        if (row["checkpoint_sha256"] != source["checkpoint_sha256"] or row["checkpoint_sha256"] != mask[method]["checkpoint_sha256"] or
            source["split_manifest_sha256"] != mask[method]["split_manifest_sha256"] or
            row["benchmark_sha256"] != manifest["benchmark_artifacts"][method]["sha256"]):
            raise ValueError(f"Historical cost provenance mismatch: {method}/{task}/L{layer}")
        if number(row["flops"]) <= 0 or number(row["latency_mean_ms"]) <= 0 or number(row["latency_sd_ms"]) < 0:
            raise ValueError(f"Invalid historical measurement: {method}/{task}/L{layer}")
    seed_rows, summary = [], []
    for method in ("p0", "p2"):
        for alg in ALGORITHMS:
            for task in TASK_ORDER:
                metric, higher, unit, factor = task_metric(task)
                values, flops, latencies = [], [], []
                frequencies = {str(e): 0 for e in EXITS}
                for seed in sorted(SEEDS):
                    row = selected[method, alg, task, seed]
                    validate_metric(row, task)
                    layer = row["selected_exit"]
                    if layer not in {str(e) for e in EXITS}:
                        raise ValueError(f"Invalid selected exit: {layer}")
                    selected_fixed = fixed[method, alg, task, layer, seed]
                    validate_metric(selected_fixed, task)
                    if selected_fixed["selected_by_validation"].lower() != "true":
                        raise ValueError("Selected exit does not match fixed-exit report")
                    check_close(row["selected_test_value"], selected_fixed["test_value"], "selected test performance")
                    value = number(row["selected_test_value"]) * factor
                    values.append(value)
                    frequencies[layer] += 1
                    entry = costs[method, task, layer] if alg == "current" else None
                    flop = int(entry["flops"]) if entry else ""
                    latency = number(entry["latency_mean_ms"]) if entry else ""
                    if entry:
                        flops.append(flop)
                        latencies.append(latency)
                    seed_rows.append({"method": method, "algorithm": alg, "task": task, "head_seed": seed,
                        "selected_exit": layer, "metric": metric, "unit": unit, "selected_test_value": value,
                        "reported_flops": flop, "reported_latency_mean_ms": latency,
                        "cost_status": "provisional_historical_baseline" if entry else "unavailable_fusion"})
                l8 = costs[method, task, "8"] if alg == "current" else None
                avg = statistics.mean(flops) if flops else None
                summary.append({"method": method, "algorithm": alg, "task": task, "metric": metric,
                    "direction": "higher" if higher else "lower", "unit": unit, "head_seed_count": 7,
                    "exit_frequencies": json.dumps(frequencies, sort_keys=True),
                    "test_mean": statistics.mean(values), "test_sample_sd": statistics.stdev(values),
                    "reported_selected_mean_gflops": avg / 1e9 if avg is not None else "",
                    "reported_selected_mean_latency_ms": statistics.mean(latencies) if latencies else "",
                    "reported_same_head_l8_gflops": int(l8["flops"]) / 1e9 if l8 else "",
                    "reported_reduction_vs_same_head_l8_percent": 100 * (1 - avg / int(l8["flops"])) if l8 else "",
                    "cost_status": "provisional_historical_baseline" if l8 else "unavailable_fusion",
                    "benchmark_sha256": l8["benchmark_sha256"] if l8 else "",
                    "checkpoint_sha256": provenance[(method, alg, 0)]["checkpoint_sha256"],
                    "limitation": "Original benchmark artifact, input shape and full timing/FLOP protocol unavailable" if l8 else "Single-GELU fusion execution path not measured"})
    write_csv(OUT / "data/historical_selected_cost_by_seed.csv", seed_rows)
    write_csv(OUT / "tables/historical_selected_performance_cost.csv", summary)
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{Historical baseline costs at exits selected on validation, with held-out test performance. GFLOPs are averaged over seven head seeds; reduction uses the same encoder and baseline head at L8. Original benchmark input shapes and measurement protocol cannot be verified; baseline costs are provisional and fusion costs are unavailable. This is not per-sample dynamic stopping.}", r"\begin{tabular}{lllrrrr}", r"\hline", r"Task & Encoder/head & L2/L4/L6/L8 & Test metric & GFLOPs & Latency (ms) & Reduction (\%) \\", r"\hline"]
    for row in summary:
        freq = json.loads(row["exit_frequencies"])
        label = ("LatentWave" if row["method"] == "p0" else "Fine-to-coarse") + (" / baseline" if row["algorithm"] == "current" else " / fusion")
        digits = 3 if row["unit"] == "m" else 2
        metric_text = f"{row['test_mean']:.{digits}f} $\\pm$ {row['test_sample_sd']:.{digits}f} " + (r"\%" if row["unit"] == "%" else "m")
        def fmt(value):
            return f"{value:.3f}" if value != "" else "--"
        lines.append(" & ".join([latex_escape(TASKS[row["task"]]), label, "/".join(str(freq[str(e)]) for e in EXITS),
            metric_text, fmt(row["reported_selected_mean_gflops"]), fmt(row["reported_selected_mean_latency_ms"]),
            fmt(row["reported_reduction_vs_same_head_l8_percent"])]) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    write_text(OUT / "tables/historical_selected_performance_cost.tex", "\n".join(lines) + "\n")
    return "partial", "Historical baseline costs reproduced from summaries; original protocol unavailable and fusion costs unmeasured."


def make_cost():
    matched = load_matched_benchmark() if BENCHMARK.is_file() else None
    fusion = load_fusion()
    fusion_sources = fusion_provenance()
    params = load_parameters()
    ablation_provenance()
    provenance = mask_provenance("multi_block_n6")
    mask_manifest = read_json(MASK / "multi_block_n6/manifest.json")
    raw = read_csv(MASK / "multi_block_n6/multiseed_inference_cost.csv", ("method", "task", "exit", "flops", "latency_mean_ms", "latency_sd_ms", "checkpoint_sha256", "benchmark_sha256"))
    entries = index_unique([r for r in raw if r["method"] in ("p0", "p2")], ("method", "task", "exit"))
    require_keys(entries, {(m, t, str(e)) for m in ("p0", "p2") for t in TASK_ORDER for e in EXITS}, "baseline cost")
    results = []
    for r in fusion:
        method, task, exit_layer, alg = r["method"], r["task"], r["exit"], r["algorithm"]
        expected_checkpoint = fusion_sources[(method, alg, 0)]["checkpoint_sha256"]
        if expected_checkpoint != provenance[method]["checkpoint_sha256"] or fusion_sources[(method, alg, 0)]["split_manifest_sha256"] != provenance[method]["split_manifest_sha256"]:
            raise ValueError(f"Cost/performance checkpoint or split mismatch for {method}")
        param = params[(method, alg, task, str(exit_layer))]
        record = {"method": method, "algorithm": alg, "task": task, "exit": exit_layer, "metric": r["metric"],
                  "unit": r["unit"], "test_mean": r["mean"], "test_sample_sd": r["sample_sd"], "head_seed_count": 7,
                  "head_parameters": int(param["total"]), "flops": "", "latency_mean_ms": "", "latency_sd_ms": "",
                  "cost_status": "unavailable_fusion_path", "benchmark_sha256": "", "checkpoint_sha256": expected_checkpoint,
                  "device": "", "batch_size": "", "precision": "", "input_shape": "", "measurement_protocol": ""}
        if alg == "current":
            cost = entries[(method, task, str(exit_layer))]
            if cost["checkpoint_sha256"] != expected_checkpoint:
                raise ValueError(f"Benchmark checkpoint mismatch for {method}/{task}/L{exit_layer}")
            benchmark = mask_manifest["benchmark_artifacts"][method]
            if cost["benchmark_sha256"] != benchmark["sha256"]:
                raise ValueError(f"Benchmark hash mismatch for {method}")
            record.update(flops=int(cost["flops"]), latency_mean_ms=number(cost["latency_mean_ms"]),
                          latency_sd_ms=number(cost["latency_sd_ms"]), benchmark_sha256=cost["benchmark_sha256"],
                          device=(benchmark.get("runtime") or {}).get("gpu", "unknown"),
                          cost_status="reported_original_benchmark_unavailable_for_protocol_audit", batch_size="",
                          precision="",
                          measurement_protocol="synchronized when CUDA; warm-up/iterations not verified: original benchmark unavailable")
        if matched is not None:
            measured, protocol = matched
            item = measured[(method, alg, task, int(exit_layer))]
            record.update(flops=item["flops"], latency_mean_ms=item["latency"]["mean_ms"],
                          latency_sd_ms=item["latency"]["standard_deviation_ms"], benchmark_sha256=sha(BENCHMARK),
                          device=protocol["gpu_name"], batch_size=protocol["batch_size"], precision=protocol["precision"],
                          input_shape=json.dumps(item["input_shape"]), cost_status="matched_benchmark",
                          measurement_protocol="Matched synchronized GPU protocol; see benchmark JSON")
        results.append(record)
    write_csv(OUT / "tables/ee_performance_cost.csv", results)
    lines = []
    for method in ("p0", "p2"):
        lines += [r"\begin{table*}[t]", r"\centering", r"\caption{" + method.upper() + r" performance and reported cost at L4. Mean $\pm$ SD over seven head seeds; macro accuracy (\%) or position error (m). " + (r"Matched benchmark for both heads." if matched is not None else r"Fusion cost is unmeasured. Baseline benchmark protocol details require the original benchmark artifact.") + r"}",
                  r"\begin{tabular}{llrrrrr}", r"\hline", r"Task & Head & Exit & Test metric & Parameters & GFLOPs & Latency (ms) \\", r"\hline"]
        for task in TASK_ORDER:
            choices = [(method, "current", 4), (method, "single_gelu", 4)]
            if method == "p0":
                choices.append(("p0", "current", 8))
            for m, alg, e in choices:
                r = next(r for r in results if (r["method"], r["algorithm"], r["task"], r["exit"]) == (m, alg, task, e))
                digits = 2 if r["unit"] == "%" else 3
                metric = f"{r['test_mean']:.{digits}f} $\\pm$ {r['test_sample_sd']:.{digits}f}"
                flops = f"{r['flops']/1e9:.3f}" if r["flops"] != "" else "--"
                latency = f"{r['latency_mean_ms']:.3f}" if r["latency_mean_ms"] != "" else "--"
                lines.append(" & ".join([latex_escape(TASKS[task]), "Baseline" if alg == "current" else "Single GELU", f"L{e}", metric, str(r["head_parameters"]), flops, latency]) + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]
    write_text(OUT / "tables/ee_performance_cost.tex", "\n".join(lines))
    if matched is not None:
        return "complete", "Matched GPU benchmark for baseline and fusion; head-seed performance on fixed checkpoints."
    return "partial", "Fusion FLOPs/latency are unavailable; original baseline benchmark artifacts are absent, so full measurement protocol/input shape cannot be audited."


def make_supplement():
    rows = load_fusion()
    write_csv(OUT / "figures/supplementary/fixed_exit_results.csv", rows)
    gains = read_csv(FUSION / "per_seed_direction_aware_gains.csv", ("method", "task", "head_seed", "comparison", "fusion_exit", "reference_exit", "metric", "positive_is_improvement", "direction_aware_gain"))
    keyed = index_unique(gains, ("method", "task", "head_seed", "comparison"))
    comparisons = ("selected_vs_current", *(f"fixed_L{e}_vs_current" for e in EXITS))
    expected = {(m, t, seed, c) for m in ("p0", "p2") for t in TASK_ORDER for seed in SEEDS for c in comparisons}
    require_keys(keyed, expected, "paired gains")
    raw = read_csv(FUSION / "per_seed_fixed_exit_results.csv", ("method", "algorithm", "head_seed", "task", "exit", "test_value", "selected_by_validation"))
    fixed = index_unique(raw, ("method", "algorithm", "head_seed", "task", "exit"))
    for (method, task, seed, comparison), row in keyed.items():
        validate_metric(row, task)
        if row["positive_is_improvement"].lower() != "true":
            raise ValueError("Paired gain direction must be positive for improvement")
        fusion_exit, reference_exit = row["fusion_exit"], row["reference_exit"]
        if comparison != "selected_vs_current" and (fusion_exit != reference_exit or comparison != f"fixed_L{fusion_exit}_vs_current"):
            raise ValueError(f"Invalid paired fixed-exit labels: {row}")
        a = fixed[(method, "single_gelu", seed, task, fusion_exit)]
        b = fixed[(method, "current", seed, task, reference_exit)]
        if comparison == "selected_vs_current" and (a["selected_by_validation"].lower() != "true" or b["selected_by_validation"].lower() != "true"):
            raise ValueError(f"Paired selected exits do not match validation selection: {row}")
        direction = 1 if task_metric(task)[1] else -1
        check_close(row["direction_aware_gain"], direction * (number(a["test_value"]) - number(b["test_value"])), "paired gain")
    write_csv(OUT / "figures/supplementary/paired_head_seed_gains_source_units.csv", gains)
    return "complete", "Fixed-exit means/SD and source-unit paired head-seed gains."


def optional_pca():
    """Generate the paper PCA view only from traceable saved test coordinates."""
    base = ROOT / "runs/representation_trajectories/multi_block_n6/interference/seed_000"
    coords_path = base / "test/coordinates.npz"
    test_path = base / "test/provenance.json"
    meta_path = base / "validation/pca_metadata.json"
    validation_path = base / "validation/provenance.json"
    if not all(p.is_file() for p in (coords_path, test_path, meta_path, validation_path)):
        return "unavailable", "Need test/coordinates.npz, test/provenance.json, validation/pca_metadata.json, and validation/provenance.json in runs/."
    prov, meta, val = read_json(test_path), read_json(meta_path), read_json(validation_path)
    if (prov.get("task") != "interference" or prov.get("split") != "test"
            or meta.get("fit_split") != "validation" or meta.get("task") != "interference"
            or val.get("task") != "interference" or val.get("split") != "validation"):
        raise ValueError("PCA provenance must identify interference test coordinates and validation fit")
    hashes = {m: prov["checkpoints"][m]["sha256"] for m in METHOD_ORDER}
    if hashes != meta.get("checkpoint_sha256") or hashes != {m: val["checkpoints"][m]["sha256"] for m in METHOD_ORDER}:
        raise ValueError("PCA validation and test checkpoints differ")
    if (prov.get("split_manifest_sha256") != val.get("split_manifest_sha256")
            or not prov.get("split_manifest_sha256")
            or meta.get("split_indices_sha256") != val.get("split_indices_sha256")):
        raise ValueError("PCA fit and test split provenance does not match")
    if prov.get("diagnostic_limit_per_class") is not None or val.get("diagnostic_limit_per_class") is not None:
        raise ValueError("PCA coordinates are from a limited diagnostic")
    if prov.get("feature_definition") != meta.get("feature_definition") or prov.get("feature_definition") != val.get("feature_definition"):
        raise ValueError("PCA feature definitions differ")
    source(coords_path)
    with np.load(coords_path, allow_pickle=False) as archive:
        labels, indices = np.array(archive["labels"]), np.array(archive["indices"])
        layers = prov["feature_definition"]["layers"]
        columns = prov["feature_definition"]["temporal_columns"]
        if len(labels) != len(indices) or len(np.unique(indices)) != len(indices) or set(np.unique(labels)) != {0, 1}:
            raise ValueError("Invalid PCA sample IDs or interference classes")
        if any(e not in layers for e in (2, 6)) or any(t not in columns for t in (2, 4, 8, 14)):
            raise ValueError("Required layer or temporal-prefix coordinate is missing")
        index_hash = hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
        if index_hash != prov.get("sample_indices_sha256") or index_hash != prov.get("split_indices_sha256"):
            raise ValueError("PCA sample IDs do not match test split provenance")
        values = {m: np.array(archive[m]) for m in ("p0", "p2")}
    for m, value in values.items():
        if value.shape != (len(labels), len(columns), len(layers), 2) or not np.isfinite(value).all():
            raise ValueError(f"Invalid PCA coordinates for {m}")
    for m in values:
        ratios = meta.get("explained_variance_ratio", {}).get(m, [])
        if len(ratios) < 2 or not all(0 <= number(v) <= 1 for v in ratios[:2]):
            raise ValueError(f"Missing PCA explained variance for {m}")
    cloud, centroids = [], []
    for method in ("p0", "p2"):
        for layer in (2, 6):
            for cls in (0, 1):
                mask = labels == cls
                for sample_id, xy in zip(indices[mask], values[method][mask, columns.index(4), layers.index(layer)]):
                    cloud.append({"method": method, "exit": layer, "sample_id": int(sample_id), "class_id": cls,
                                  "temporal_patch_columns": 4, "pc1": float(xy[0]), "pc2": float(xy[1])})
                for t in (2, 4, 8, 14):
                    xy = values[method][mask, columns.index(t), layers.index(layer)].mean(axis=0)
                    centroids.append({"method": method, "exit": layer, "class_id": cls,
                                      "temporal_patch_columns": t, "pc1": float(xy[0]), "pc2": float(xy[1])})
    write_csv(OUT / "data/interference_pca_points.csv", cloud)
    write_csv(OUT / "data/interference_pca_centroids.csv", centroids)
    configure()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.9))
    class_dark = {0: "#0072B2", 1: "#D55E00"}
    class_light = {0: "#c1dce9", 1: "#f1d0b8"}
    markers = {0: "o", 1: "^"}
    for i, layer in enumerate((2, 6)):
        for j, method in enumerate(("p0", "p2")):
            ax = axes[i, j]
            for cls in (0, 1):
                points = values[method][labels == cls, columns.index(4), layers.index(layer)]
                ax.scatter(points[:, 0], points[:, 1], s=3, marker=".", c=class_light[cls], linewidths=0, zorder=1)
            for cls in (0, 1):
                trajectory = np.asarray([[r["pc1"], r["pc2"]] for r in centroids if r["method"] == method and r["exit"] == layer and r["class_id"] == cls])
                ax.plot(trajectory[:, 0], trajectory[:, 1], color=class_dark[cls], marker=markers[cls],
                        linestyle="-" if cls == 0 else "--", linewidth=1.2, markersize=3.5, zorder=3)
                for first, second in zip(trajectory[:-1], trajectory[1:]):
                    if np.linalg.norm(second-first) > 1e-10:
                        ax.annotate("", xy=second, xytext=first, arrowprops={"arrowstyle": "->", "color": class_dark[cls], "lw": .8}, zorder=4)
                for index, offset in ((0, (3, 3)), (-1, (3, -9))):
                    ax.annotate("2" if index == 0 else "14", trajectory[index], xytext=offset,
                                textcoords="offset points", fontsize=8, color=class_dark[cls], zorder=5)
            ratios = meta["explained_variance_ratio"][method]
            ax.set_title(f"{METHODS[method]} ? L{layer}", fontsize=9)
            ax.set_xlabel(f"PC 1 ({100 * ratios[0]:.1f}%)")
            ax.set_ylabel(f"PC 2 ({100 * ratios[1]:.1f}%)")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(color="0.9", linewidth=.5)
    legend = [Line2D([0], [0], color=class_dark[c], marker=markers[c], linestyle="-" if c == 0 else "--", label=f"Class {c}") for c in (0, 1)]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .99), ncol=2, frameon=False)
    fig.subplots_adjust(left=.10, right=.98, bottom=.09, top=.89, wspace=.32, hspace=.48)
    save_fig(fig, OUT / "figures/main/interference_pca_trajectories")
    return "complete", "Saved test points at four columns and centroid paths across 2, 4, 8, 14 columns; separate validation-fit PCA per method."


def revision():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def write_readme(statuses, rev):
    status_lines = "\n".join(f"| {name} | {record['status']} | {record['detail']} |" for name, record in statuses.items())
    cost_description = ("A validated matched GPU benchmark supplies baseline and fusion costs; see its recorded protocol and checkpoint provenance." if statuses.get("cost", {}).get("status") == "complete" else "Historical baseline costs are provisional: the original benchmark files, shapes and timing protocol are unavailable. Fusion cost is unmeasured and must not be replaced by baseline cost.")
    text = f'''# Paper results

Generated with `python scripts/paper_results/build_all.py` at repository revision `{rev}`. The input audit target is `{EXPECTED_REVISION}`. Run `python scripts/paper_results/build_all.py --help` for individual generators. The pipeline reads existing reports and does not run training or benchmarks. Python, NumPy, and Matplotlib are required; no checkpoint or dataset is required for the checked-in report inputs.

## Artifact status

| Artifact | Status | Detail |
|---|---|---|
{status_lines}

`complete` means required data were validated and the output was generated. `partial` means the named output exists with a material limitation. `unavailable` means an optional artifact could not be made. `failed` means validation or generation failed; inspect `manifest.json` and restore the input before using that output. `not_requested` applies to an individual build. The build removes that generator's existing output before attempting it, preventing stale output from appearing current. A build returns a nonzero status if any requested generator fails.

## Suggested captions and interpretation

**Fixed-exit comparison.** Held-out test macro accuracy (%, higher is better) or mean position error (m, lower is better) at L2/L4/L6/L8 for LatentWave and fine-to-coarse pretraining, with linear and single-GELU fusion heads, using the historical `multi_block_n6` condition. Lines show mean across seven independently initialized downstream heads on one fixed pretrained checkpoint and one shared split. Supplementary error bars show sample SD across those head seeds; they do not measure variation across pretrained encoders or sampling uncertainty. Test values are for fixed exits; these curves do not evaluate per-sample dynamic stopping. At L2 the residual fusion head has no earlier source representation.

**Validation-selected downstream results.** Each head and exit is selected on validation separately for each method and head seed, then evaluated on held-out test data. Table entries are mean ± sample SD across seven downstream head seeds on a fixed pretrained encoder. Accuracy is macro accuracy in percent; NR positioning is mean distance error in metres. Bold denotes the best observed mean, not statistical significance. See the supplementary frequency CSV for selected exits. `frequency_n6` and `multi_block_n6` are separate historical reports with the same nominal span-time geometry, not independent mask-family evidence. M-block LatentWave is a recovered epoch-64 checkpoint from a 240-epoch schedule; its same-scale method uses the frequency same-scale checkpoint. See `data/checkpoint_provenance.json` and the source reports for checkpoint mappings and unrecorded schedule details.

**RF and jamming figures.** RF (`rfs`) uses 20 classes and 362 test examples, with as few as three examples in some classes; jamming uses six classes and 750 balanced test examples. The width-axis prefix is called input patch columns because temporal semantics are not established for these caches. Full-input class-separation panels use mean pairwise centroid distance divided by mean within-class RMS spread; bars are 95% class-stratified paired bootstrap intervals, conditional on one checkpoint per method. In RF, some percentile intervals do not contain their point estimates; this is possible for a biased or skewed bootstrap distribution and needs statistical review, particularly with small classes. Prefix-accuracy plots show mean and sample SD across five full-input-trained heads evaluated unchanged at each width. Individual head-seed identifiers are not recorded in the checked-in summary. Checkpoint schedules differ, and the coarse-to-fine source split is unverified; results are descriptive.

**NR positioning figures.** `nrpos_full_input_error_vs_exit` uses held-out test results at fixed L2/L4/L6/L8 under the historical multi-block condition, with mean and sample SD across seven heads on each fixed checkpoint. `nrpos_validation_error_vs_time_patches` is a separate time-mask study showing fixed-exit validation errors from one saved head per method and exit at each input length. It contains LatentWave and fine-to-coarse only; the fine-to-coarse checkpoint is epoch 29 and the LatentWave training horizon and checkpoint hashes are not recorded in these source packets. The two methods share a split hash, but the time-sweep plot is validation-only and has no between-head uncertainty estimate. It must not be presented as a held-out test result or combined as if it used the same checkpoints as the full-input test plot. Lower error is better.

**Human-activity figures.** The human-activity task was selected after inspecting reported performance because it has higher full-input accuracy than interference; retain the interference figure to show the broader comparison. The test split contains 120 examples, 20 per class. The human-activity separation figure uses all 14 temporal columns: its score is the mean pairwise class-centroid distance divided by the mean within-class RMS radius in the original feature space. Its bars are 95% class-stratified paired bootstrap intervals over test examples, conditional on one pretrained checkpoint per method. The temporal-accuracy figure uses the same full-input-trained frozen heads at each input length, with mean and sample SD across five head seeds; the checked-in human-activity summary does not record the individual seed IDs. The dotted line marks 1/6 (16.7%) macro accuracy for a uniform or constant-class baseline. Mean plus one SD can exceed 100% because SD describes head-to-head variation rather than an accuracy bound. The coarse-to-fine run has a shorter training horizon and its original split provenance cannot be verified. Full-input comparisons and the task selection are descriptive; do not treat this selected high-accuracy task as evidence that all tasks improve.

**Temporal interference performance.** The `interference_accuracy_vs_time_patches` figure shows binary interference macro accuracy at L2/L4/L6/L8 as the number of visible temporal patch columns changes. Each of five heads per method and exit was trained on the full 14-column input, selected on validation, and then evaluated unchanged at all four input lengths. Points show held-out mean accuracy; error bars are sample SD across head seeds 2025 to 2029, conditional on the fixed pretrained checkpoint, and do not estimate pretraining-seed variability. The 50% line is the chance reference for this balanced binary task. Checkpoint schedules and training horizons differ; compare methods descriptively. These heads and checkpoint choices belong to the temporal robustness study and should not be conflated with the seven-seed fusion study.

**Paired representation comparison.** The `interference_paired_separation` figure shows class separation at each encoder exit and the paired change from LatentWave. Class separation is the class-centroid distance divided by the mean within-class RMS radius; higher values mean greater separation relative to spread, not necessarily higher downstream accuracy. Markers show held-out estimates and bars show 95% class-stratified paired bootstrap intervals based on 2,000 resamples of the same test examples. A difference crossing zero is compatible with either direction under this resampling procedure. Comparisons are conditional on a single pretrained checkpoint per method, use all 14 temporal patch columns, and are descriptive because the checkpoint schedules differ; coarse-to-fine source split equality is unverified.

**Representation analysis.** The main figure follows class separation across encoder layers for a fixed input length; the companion `interference_separation_vs_input` figure shows point estimates of the same score as the number of visible temporal patch columns increases, separately for each layer. Its CSV includes the 95% bootstrap intervals. A higher ratio means class centroids are farther apart relative to within-class spread; it is a descriptive geometric statistic and does not by itself establish downstream accuracy. The original class-centroid PCA trajectories follow changing input length within each layer, rather than connecting layers. The main interference figure uses held-out test features from all 14 temporal patch columns and frozen context encoders; the source report predeclared its separate four-column comparison, so the full-input analysis is descriptive. Class-separation ratio is mean pairwise distance between class centroids divided by mean within-class RMS radius in the original 256-dimensional feature space. Error bars are 95% class-stratified paired bootstrap intervals (2,000 resamples) conditional on the evaluated pretrained checkpoints, not head-seed variability or pretraining variability. LatentWave, same-scale, and fine-to-coarse are at epoch 59 with differing configured schedules; coarse-to-fine is epoch 49. The coarse-to-fine source split equality could not be verified. This is a descriptive analysis. In the optional PCA figure, pale dots are test examples at four visible temporal columns. Lines show each class centroid as the input grows through 2, 4, 8 and 14 columns; arrows indicate increasing input, and endpoint labels show 2 and 14. The panels compare L2 and L6, and axis percentages indicate the variance explained on the validation fit. PCA uses a separate validation-fitted basis per method; panel positions and axes are not directly comparable across methods. Panel limits can differ, and visual separation is descriptive rather than a quantitative performance measure. Additional task plots use the same four leading input patch columns, which are not necessarily temporal for every task; see each source summary.

**Performance and cost.** The full CSV reports fixed exits; LaTeX shows an L4 comparison and a LatentWave L8 reference. {cost_description} Latency SD is timing variation, not head-seed SD. FLOPs exclude operations without registered counter formulas; review the recorded operator coverage.

**Historical selected-exit costs.** `tables/historical_selected_performance_cost.csv` and `.tex` reproduce P0/P2 baseline FLOPs and latency from the prior multi-block summary, joined to validation-selected held-out results across seven heads. The `data/historical_selected_cost_by_seed.csv` file contains the per-head joins. Cost and its L8 reduction are provisional: the original benchmark artifacts, exact input shapes, FLOP convention and full timing protocol are unavailable. Fusion entries have no cost because the proposed path was not measured. Selected-exit averages describe head runs, not dynamic per-sample stopping. The reported latency is an average of prior fixed-exit means, not variation across head seeds.

**Matched benchmark.** See `scripts/paper_results/MATCHED_BENCHMARK.md` for exact inputs and commands. Run `python scripts/matched_exit_benchmark.py --help` on a CUDA machine with the two original checkpoints and four authenticated seed-0 evaluation packets. The benchmark checks exact packet hashes, verified task shapes, model execution and parameter counts; it does not train. Supply the resulting JSON with `python scripts/paper_results/build_all.py --matched-benchmark <path>`. Selected-exit costs average fixed exit costs over seven validation-selected head runs, not an input-dependent stopping policy. Matched-capacity fusion controls and matched independent pretraining seeds are future experiments; optional PCA requires original features or coordinates.

**Adapter ablation.** Head seed 0 pilot. Validation chooses the exit; reported test value and head parameter count correspond to that selected exit. Baseline and each adapter retain their own selected exit. Single-layer width does not imply fewer parameters. See the CSV for exact parameters. L2 has no earlier source and is a reproducibility control.

Source summaries and per-input SHA-256 values are in `manifest.json`; the manifest records the repository revision and plotting package versions. Original external paths in the source manifests are historical provenance and may not be available on this machine. No unrecorded benchmark configuration is inferred from them. Paired gain CSV values retain the source units: accuracy fractions or metres; positive gain favors fusion. Select figure variants without tuning against the held-out test set. Comparisons are descriptive and do not establish statistical significance or causal separation of head capacity and fusion.

## Submission checks

Inspect the PDF at final print size and in grayscale; check clipping, line distinctions, labels, panel order, font embedding and submission-template fit. IEEE recommends vector graphics and embedded fonts. A venue-specific template check and Overleaf insertion are still required. The locally generated files have not been automatically checked by an IEEE submission service.
'''
    write_text(OUT / "README.md", text)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("fusion", "selected", "representations", "cost", "ablation", "supplement", "pca", "temporal_performance", "human_activity", "rfs", "jamming", "nrpos", "selected_cost", "historical_cost")
    parser.add_argument("--only", choices=names, help="Build one artifact group")
    parser.add_argument("--matched-benchmark", type=Path, help="Validated matched benchmark JSON; defaults to reports/matched_exit_benchmark/benchmark.json")
    args = parser.parse_args(argv)
    global BENCHMARK
    if args.matched_benchmark is not None:
        BENCHMARK = args.matched_benchmark.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    rev = revision()
    print(f"Repository revision: {rev}; audited target: {EXPECTED_REVISION}")
    groups = {
        "fusion": (make_fusion, ["data/ee_pretraining_fusion_six_tasks.csv", "figures/main/ee_pretraining_fusion_six_tasks.pdf", "figures/main/ee_pretraining_fusion_six_tasks.png", "figures/supplementary/ee_pretraining_fusion_uncertainty.pdf", "figures/supplementary/ee_pretraining_fusion_uncertainty.png"]),
        "selected": (make_selected, ["tables/pretraining_comparison.csv", "tables/pretraining_comparison.tex", "figures/supplementary/selected_exit_frequencies.csv", "data/checkpoint_provenance.json"]),
        "representations": (lambda: make_representations("interference", True), ["data/layerwise_representation_analysis.csv", "figures/main/layerwise_representation_analysis.pdf", "figures/main/layerwise_representation_analysis.png"]),
        "temporal_performance": (make_temporal_performance, ["data/interference_accuracy_vs_time_patches.csv", "figures/main/interference_accuracy_vs_time_patches.pdf", "figures/main/interference_accuracy_vs_time_patches.png"]),
        "nrpos_depth": (make_nrpos_depth, ["data/nrpos_full_input_error_vs_exit.csv", "figures/main/nrpos_full_input_error_vs_exit.pdf", "figures/main/nrpos_full_input_error_vs_exit.png"]),
        "nrpos_time": (make_nrpos_time, ["data/nrpos_validation_error_vs_time_patches.csv", "figures/main/nrpos_validation_error_vs_time_patches.pdf", "figures/main/nrpos_validation_error_vs_time_patches.png"]),
        "rfs_representation": (lambda: make_paired_representation("rfs"), ["data/rfs_paired_separation.csv", "figures/main/rfs_paired_separation.pdf", "figures/main/rfs_paired_separation.png"]),
        "rfs_accuracy": (lambda: make_temporal_performance("rfs"), ["data/rfs_accuracy_vs_input_patches.csv", "figures/main/rfs_accuracy_vs_input_patches.pdf", "figures/main/rfs_accuracy_vs_input_patches.png"]),
        "jamming_representation": (lambda: make_paired_representation("jamming"), ["data/jamming_paired_separation.csv", "figures/main/jamming_paired_separation.pdf", "figures/main/jamming_paired_separation.png"]),
        "jamming_accuracy": (lambda: make_temporal_performance("jamming"), ["data/jamming_accuracy_vs_input_patches.csv", "figures/main/jamming_accuracy_vs_input_patches.pdf", "figures/main/jamming_accuracy_vs_input_patches.png"]),
        "human_activity_representation": (lambda: make_paired_representation("human-activity"), ["data/human_activity_paired_separation.csv", "figures/main/human_activity_paired_separation.pdf", "figures/main/human_activity_paired_separation.png"]),
        "human_activity_temporal": (lambda: make_temporal_performance("human-activity"), ["data/human_activity_accuracy_vs_time_patches.csv", "figures/main/human_activity_accuracy_vs_time_patches.pdf", "figures/main/human_activity_accuracy_vs_time_patches.png"]),
        "representations_paired": (make_paired_representation, ["data/interference_paired_separation.csv", "figures/main/interference_paired_separation.pdf", "figures/main/interference_paired_separation.png"]),
        "representations_trajectory": (make_interference_trajectory, ["data/interference_separation_vs_input.csv", "figures/supplementary/interference_separation_vs_input.pdf", "figures/supplementary/interference_separation_vs_input.png"]),
        "historical_cost": (make_historical_selected_cost, ["data/historical_selected_cost_by_seed.csv", "tables/historical_selected_performance_cost.csv", "tables/historical_selected_performance_cost.tex"]),
        "selected_cost": (make_selected_cost, ["data/matched_exit_benchmarks.csv", "data/selected_exit_cost_by_seed.csv", "tables/selected_exit_performance_cost.csv", "tables/selected_exit_performance_cost.tex"]),
        "cost": (make_cost, ["tables/ee_performance_cost.csv", "tables/ee_performance_cost.tex"]),
        "ablation": (make_ablation, ["tables/adapter_ablation.csv", "tables/adapter_ablation.tex"]),
        "supplement": (make_supplement, ["figures/supplementary/fixed_exit_results.csv", "figures/supplementary/paired_head_seed_gains_source_units.csv"]),
        "pca": (optional_pca, ["data/interference_pca_points.csv", "data/interference_pca_centroids.csv", "figures/main/interference_pca_trajectories.pdf", "figures/main/interference_pca_trajectories.png"]),
    }
    for task in ("rfs", "jamming", "human-activity"):
        groups[f"representations_{task}"] = (lambda task=task: make_representations(task), [f"data/representations_{task}.csv", f"figures/supplementary/representations_{task}.pdf", f"figures/supplementary/representations_{task}.png"])
    statuses = {}
    for name, (fn, paths) in groups.items():
        selected = args.only is None or args.only == name or (args.only == "representations" and name.startswith("representations_")) or (args.only == "human_activity" and name.startswith("human_activity_")) or (args.only in ("rfs", "jamming", "nrpos") and name.startswith(args.only + "_"))
        if not selected:
            statuses[name] = {"status": "not_requested", "detail": "Not part of this invocation.", "outputs": []}
            continue
        for rel in paths:
            p = OUT / rel
            if p.is_file(): p.unlink()
        try:
            state, detail = fn()
            outputs = [p for p in paths if (OUT / p).is_file()]
            if state in ("complete", "partial") and len(outputs) != len(paths):
                raise RuntimeError(f"Expected outputs missing: {set(paths) - set(outputs)}")
            statuses[name] = {"status": state, "detail": detail, "outputs": outputs}
        except Exception as exc:
            for rel in paths:
                p = OUT / rel
                if p.is_file(): p.unlink()
            statuses[name] = {"status": "failed", "detail": f"{type(exc).__name__}: {exc}", "outputs": []}
        print(f"{name}: {statuses[name]['status']} - {statuses[name]['detail']}")
    write_readme(statuses, rev)
    try:
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    manifest = {"schema_version": 1, "revision": rev, "working_tree_dirty": dirty, "audited_input_revision": EXPECTED_REVISION,
                "revision_matches_audit": rev == EXPECTED_REVISION,
                "command": "python scripts/paper_results/build_all.py" + (f" --only {args.only}" if args.only else "") + (f" --matched-benchmark {args.matched_benchmark}" if args.matched_benchmark else ""),
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
                "settings": {"width_inches": WIDTH, "font": "DejaVu Sans", "font_size_pt": 9, "png_dpi": 400, "pdf_fonttype": 42},
                "input_files": [{"path": (str(p.relative_to(ROOT)).replace("\\", "/") if p.is_relative_to(ROOT) else str(p)), "sha256": sha(p) if p.is_file() else None} for p in sorted(INPUTS)],
                "generator_files": [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p)} for p in (ROOT / "scripts/paper_results/pipeline.py", ROOT / "scripts/paper_results/style.py", ROOT / "scripts/paper_results/build_all.py", ROOT / "scripts/matched_exit_benchmark.py")],
                "output_files": [{"path": str((OUT / rel).relative_to(ROOT)).replace("\\", "/"), "sha256": sha(OUT / rel)} for entry in statuses.values() for rel in entry["outputs"]],
                "artifacts": statuses,
                "limitations": ["One fixed pretrained checkpoint per method/condition; head-seed SD is not pretraining variability.", "Frequency and multi-block share nominal span-time geometry.", "Representation P2-Reverse source split unverified.", ("Fusion cost not measured; historical baseline benchmark protocol is not fully auditable in this checkout." if not BENCHMARK.is_file() else "Matched measurements require review of benchmark operator coverage and its recorded initialized or trained head weights.")]}
    write_text(OUT / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 1 if any(v["status"] == "failed" for v in statuses.values()) else 0
