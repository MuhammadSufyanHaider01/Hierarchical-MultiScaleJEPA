#!/usr/bin/env python3
"""Update only multi-block downstream tables and figures with hierarchy ablations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/masking_study/multi_block_n6"
TASK_LABELS = {
    "rfs": "RF sensing",
    "jamming": "Jamming classification",
    "human-activity": "Human activity sensing",
    "interference": "Interference classification",
    "nrpos-outdoor": "NR positioning",
    "deepmimo-beam": "DeepMIMO beam",
}
METHOD_LABELS = {
    "b0": "Original LatentWave",
    "b1": "LatentWave with Post-hoc Frozen Early Exits",
    "p1": "P1 Same-scale Intermediate Supervision",
    "p2_reverse": "P2-Reverse Coarse-to-Fine JEPA",
    "p2": "Proposed Hierarchical Multi-Scale JEPA",
}
METHOD_ORDER = ("b1", "p1", "p2_reverse", "p2")
EXIT_LAYERS = (2, 4, 6, 8)
OUTPUTS = {
    "table_primary_fixed_exits.csv", "table_primary_fixed_exits.md",
    "table_selected_exits.csv", "table_selected_exits.md",
    "table_all_methods_downstream.csv", "table_all_methods_downstream.md",
    "early_exit_performance.png", "early_exit_performance.pdf",
    "selected_exit_performance.png", "selected_exit_performance.pdf",
    "all_methods_downstream_performance.png", "all_methods_downstream_performance.pdf",
    "all_methods_selected_performance.png", "all_methods_selected_performance.pdf",
    "hierarchy_ablation_provenance.json",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows)
    return "\n".join(lines)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(path: Path, method: str) -> dict:
    data = json.loads(path.read_text())
    if data.get("method") != method or data.get("dataset_suite") != "primary":
        raise ValueError(f"{path} is not a {method} primary downstream result")
    if not data.get("protocol", {}).get("test_evaluation"):
        raise ValueError(f"{path} does not include post-selection test evaluation")
    tasks = data.get("tasks", {})
    if set(tasks) != set(TASK_LABELS):
        raise ValueError(f"{path} must contain all six primary tasks, found {sorted(tasks)}")
    for task_name, task in tasks.items():
        if set(task.get("exits", {})) != {str(layer) for layer in EXIT_LAYERS}:
            raise ValueError(f"{path} is missing fixed exits for {task_name}")
        for field in ("selected_test_best_validation", "selected_test_earliest_within_tolerance"):
            if field not in task:
                raise ValueError(f"{path} is missing {field} for {task_name}")
    return data


def make_rows(data: dict) -> tuple[list[dict], list[dict], list[dict]]:
    method = data["method"]
    fixed, selected, headline = [], [], []
    for task_name, task in data["tasks"].items():
        spec = task["spec"]
        metric = spec["metric_name"]
        higher = bool(spec["higher_is_better"])
        for layer in EXIT_LAYERS:
            record = task["exits"][str(layer)]
            fixed.append({
                "method": method, "task": task_name, "exit_layer": layer, "metric": metric,
                "value": record["test"][metric], "validation_value": record["validation"][metric],
                "best_epoch": record.get("best_epoch"),
                "head_epochs_completed": record.get("head_epochs_completed"),
                "head_stopped_early": record.get("head_stopped_early"),
            })
        for policy, exit_key, test_key in (
            ("best_validation", "selected_exit_best_validation", "selected_test_best_validation"),
            ("earliest_within_tolerance", "selected_exit_earliest_within_tolerance", "selected_test_earliest_within_tolerance"),
        ):
            selected.append({
                "method": method, "task": task_name, "evaluation": policy,
                "exit_layer": int(task[exit_key]), "metric": metric,
                "value": task[test_key][metric], "higher_is_better": higher,
            })
        layer = int(task["selected_exit_earliest_within_tolerance"])
        headline.append({
            "method": method, "method_label": METHOD_LABELS[method], "task": task_name,
            "exit_layer": layer, "selection": "earliest within 5% of validation best",
            "metric": metric, "value": task["selected_test_earliest_within_tolerance"][metric],
            "protocol": "frozen context; seed-2025 split regenerated for hierarchy ablation",
        })
    return fixed, selected, headline


def plot_downstream(report_dir: Path, fixed: list[dict], selected: list[dict], headline: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = {"b0": "#009E73", "b1": "#0072B2", "p1": "#CC79A7", "p2_reverse": "#E69F00", "p2": "#D55E00"}
    for filename, kind in (
        ("early_exit_performance", "fixed"),
        ("selected_exit_performance", "selected"),
        ("all_methods_downstream_performance", "all_fixed"),
        ("all_methods_selected_performance", "headline"),
    ):
        tasks = tuple(TASK_LABELS)
        ncols, nrows = 3, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(15.3, 7.6), squeeze=False)
        for axis, task_name in zip(axes.flat, tasks):
            if kind in {"fixed", "all_fixed"}:
                task_rows = [row for row in fixed if row["task"] == task_name]
                if kind == "all_fixed":
                    base = next(row for row in headline if row["method"] == "b0" and row["task"] == task_name)
                    axis.plot([8], [float(base["value"])], marker="D", linestyle="None", color=colors["b0"], label=METHOD_LABELS["b0"])
                for method in METHOD_ORDER:
                    method_rows = sorted((row for row in task_rows if row["method"] == method), key=lambda row: int(row["exit_layer"]))
                    axis.plot([int(r["exit_layer"]) for r in method_rows], [float(r["value"]) for r in method_rows], marker="o", color=colors[method], label=METHOD_LABELS[method])
                higher = task_rows[0]["metric"] != "mean_distance_error"
                axis.set_xticks(EXIT_LAYERS)
                axis.set_xlabel("Encoder exit layer")
            elif kind == "selected":
                task_rows = [row for row in selected if row["task"] == task_name]
                positions, values, bar_colors, labels = [], [], [], []
                for method in METHOD_ORDER:
                    for policy in ("best_validation", "earliest_within_tolerance"):
                        record = next(r for r in task_rows if r["method"] == method and r["evaluation"] == policy)
                        positions.append(len(positions)); values.append(float(record["value"])); bar_colors.append(colors[method])
                        labels.append(f"{method}\nL{record['exit_layer']}")
                axis.bar(positions, values, color=bar_colors, edgecolor="#222")
                axis.set_xticks(positions, labels, fontsize=7)
                higher = bool(task_rows[0]["higher_is_better"])
            else:
                task_rows = [row for row in headline if row["task"] == task_name]
                methods = ("b0", *METHOD_ORDER)
                axis.bar(range(len(methods)), [float(next(r["value"] for r in task_rows if r["method"] == m)) for m in methods], color=[colors[m] for m in methods], edgecolor="#222")
                axis.set_xticks(range(len(methods)), [m.replace("_", "\n") for m in methods], fontsize=8)
                higher = task_rows[0]["metric"] != "mean_distance_error"
            axis.set_title(TASK_LABELS[task_name], fontsize=10, weight="bold")
            axis.set_ylabel("Accuracy" if higher else "Mean distance error (m)")
            axis.grid(axis="y", alpha=.25)
            if higher:
                axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        if kind in {"fixed", "all_fixed"}:
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
        title = {
            "fixed": "Frozen-head downstream performance by encoder depth",
            "selected": "Validation-selected exits and held-out test performance",
            "all_fixed": "All methods: downstream performance by encoder depth",
            "headline": "Validation-selected downstream comparison",
        }[kind]
        fig.suptitle(title, y=1.02)
        fig.text(.5, .005, "Single-seed point estimates; new head/exit selection uses validation only. Historical P0/P2 values are reused.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .025, 1, .93))
        for suffix, options in (("png", {"dpi": 300}), ("pdf", {})):
            fig.savefig(report_dir / f"{filename}.{suffix}", bbox_inches="tight", **options)
        plt.close(fig)


def update(report_dir: Path, p1_path: Path, reverse_path: Path) -> None:
    p1, reverse = load_run(p1_path, "p1"), load_run(reverse_path, "p2_reverse")
    new_fixed, new_selected, new_headline = [], [], []
    for run in (p1, reverse):
        fixed_rows, selected_rows, headline_rows = make_rows(run)
        new_fixed.extend(fixed_rows); new_selected.extend(selected_rows); new_headline.extend(headline_rows)
    paths = {name: report_dir / name for name in (
        "table_primary_fixed_exits.csv", "table_selected_exits.csv", "table_all_methods_downstream.csv")}
    fixed = [r for r in read_csv(paths["table_primary_fixed_exits.csv"]) if r["method"] not in {"p1", "p2_reverse"}] + new_fixed
    selected = [r for r in read_csv(paths["table_selected_exits.csv"]) if r["method"] not in {"p1", "p2_reverse"}] + new_selected
    headline = [r for r in read_csv(paths["table_all_methods_downstream.csv"]) if r["method"] not in {"p1", "p2_reverse"}] + new_headline
    if not any(r["method"] == "b0" for r in headline):
        raise ValueError("historical downstream table is missing the Original LatentWave L8 reference")
    before = {p.name: file_hash(p) for p in report_dir.iterdir() if p.is_file()}

    write_csv(paths["table_primary_fixed_exits.csv"], fixed, ["method", "task", "exit_layer", "metric", "value", "validation_value", "best_epoch", "head_epochs_completed", "head_stopped_early"])
    write_csv(paths["table_selected_exits.csv"], selected, ["method", "task", "evaluation", "exit_layer", "metric", "value", "higher_is_better"])
    write_csv(paths["table_all_methods_downstream.csv"], headline, ["method", "method_label", "task", "exit_layer", "selection", "metric", "value", "protocol"])
    fixed_md = [[METHOD_LABELS.get(r["method"], r["method"]), TASK_LABELS[r["task"]], f"L{r['exit_layer']}", r["metric"], r["value"], r["validation_value"], r["best_epoch"]] for r in fixed]
    selected_md = [[METHOD_LABELS.get(r["method"], r["method"]), TASK_LABELS[r["task"]], r["evaluation"], f"L{r['exit_layer']}", r["value"]] for r in selected]
    headline_md = [[r["method_label"], TASK_LABELS[r["task"]], f"L{r['exit_layer']}", r["selection"], r["value"]] for r in headline]
    (report_dir / "table_primary_fixed_exits.md").write_text("# Detailed frozen-exit downstream metrics\n\nValidation is used for head and exit selection; test is evaluated after selection.\n\n" + markdown_table(["Method", "Task", "Exit", "Metric", "Test", "Validation", "Best epoch"], fixed_md) + "\n")
    (report_dir / "table_selected_exits.md").write_text("# Detailed selected exits\n\n" + markdown_table(["Method", "Task", "Policy", "Exit", "Test value"], selected_md) + "\n")
    (report_dir / "table_all_methods_downstream.md").write_text("# All-method headline downstream comparison\n\n" + markdown_table(["Method", "Task", "Exit", "Selection", "Test value"], headline_md) + "\n")
    plot_downstream(report_dir, fixed, selected, headline)
    provenance = {
        "methods_added": ["p1", "p2_reverse"],
        "new_run_inputs": {"p1": str(p1_path), "p2_reverse": str(reverse_path)},
        "pretrain_seed": 0, "head_split_seed": 2025,
        "split_protocol": "80/10/10; regenerated from current dataset checkout",
        "limitation": "Historical P0/P2 split manifest and checkpoints are absent; exact split-hash equivalence with historical tables cannot be verified.",
        "historical_p0_p2_retrained": False, "inference_artifacts_modified": False,
        "updated_outputs": sorted(OUTPUTS),
    }
    (report_dir / "hierarchy_ablation_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    after = {p.name: file_hash(p) for p in report_dir.iterdir() if p.is_file()}
    changed = {n for n in before.keys() | after.keys() if before.get(n) != after.get(n)}
    if changed - OUTPUTS:
        raise RuntimeError(f"changed files outside downstream allowlist: {sorted(changed - OUTPUTS)}")
    if any("inference" in n.lower() for n in changed):
        raise RuntimeError("an inference report file changed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-metrics", type=Path, required=True)
    parser.add_argument("--p2-reverse-metrics", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    args = parser.parse_args()
    update(args.report_dir, args.p1_metrics, args.p2_reverse_metrics)
    print(f"Updated downstream-only report in {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
