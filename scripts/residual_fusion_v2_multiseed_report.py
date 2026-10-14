#!/usr/bin/env python3
"""Fail-closed seven-seed report for Current EE versus Residual Dynamic Fusion V2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("p0", "p1", "p2", "p2_reverse")
ALGORITHMS = ("current", "residual_projected_dynamic_fusion")
SEEDS = tuple(range(7))
TASKS = ("rfs", "jamming", "human-activity", "interference", "nrpos-outdoor", "deepmimo-beam")
EXITS = (2, 4, 6, 8)
CHECKPOINT_HASHES = {"p0": "8faa7245b6692e2d74d753fde2268fb712264c02c1da19632c045d022d192f1f",
                     "p1": "2469d6c797461409f128e2732696cecfd3540f9e609d7f8bcaebdf9eff7f0025",
                     "p2": "7e1e27d96183b8f9d1c0d6b7493b93945633ccfb1514555135b4d04f22bf18ac",
                     "p2_reverse": "a1d38e7e951e52765ec0ba5de0b6aed27de6d7b8e604569cd9df5b40eecd3973"}
CURRENT_ROOT = ROOT / "runs/masking_head_seeds/multi_block_n6"
V2_ROOT = ROOT / "runs/ee_optimization/multi_block_n6/residual_dynamic_fusion"
OUT = ROOT / "reports/early_exit_optimization/multi_block_n6/residual_dynamic_fusion_multiseed"
CHECKPOINT_PATHS = {
    "p0": ROOT / "runs/masking_head_seeds/checkpoint_bundle/checkpoints/multi_block_n6/P0/last.ckpt",
    "p1": ROOT / "runs/masking_head_seeds/checkpoint_bundle/checkpoints/frequency_n6/P1/last.ckpt",
    "p2": ROOT / "runs/masking_head_seeds/checkpoint_bundle/checkpoints/multi_block_n6/P2/last.ckpt",
    "p2_reverse": ROOT / "runs/masking_study/multi_block_n6/hierarchy_ablation/p2_reverse/seed_000/checkpoints/last.ckpt",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packet_path(method: str, algorithm: str, seed: int, current_root=CURRENT_ROOT, v2_root=V2_ROOT) -> Path:
    if algorithm == "current":
        return current_root / f"head_seed_{seed:03d}" / method / "seed_000/downstream/exit_metrics.json"
    if algorithm == "residual_projected_dynamic_fusion":
        return v2_root / f"head_seed_{seed:03d}" / method / "seed_000/downstream/exit_metrics.json"
    raise ValueError(f"unsupported algorithm: {algorithm}")


def load_packet(method: str, algorithm: str, seed: int, current_root=CURRENT_ROOT, v2_root=V2_ROOT) -> tuple[Path, dict]:
    path = packet_path(method, algorithm, seed, current_root, v2_root)
    if not path.is_file():
        raise FileNotFoundError(f"required {algorithm} packet missing: {path}")
    packet = json.loads(path.read_text())
    if (packet.get("method") != method or int(packet.get("head_seed", -1)) != seed
            or int(packet.get("pretrain_seed", -1)) != 0 or packet.get("split_seed") not in (None, 2025)):
        raise ValueError(f"packet provenance mismatch: {path}")
    if algorithm != "current" and packet.get("head_algorithm") != algorithm:
        raise ValueError(f"algorithm identity mismatch: {path}")
    if packet.get("protocol", {}).get("test_evaluation") is not True or set(packet.get("tasks", {})) != set(TASKS):
        raise ValueError(f"test/task coverage incomplete: {path}")
    checkpoint = Path(packet.get("checkpoint", ""))
    if (not checkpoint.is_file() or checkpoint.resolve() != CHECKPOINT_PATHS[method].resolve()
            or sha(checkpoint) != CHECKPOINT_HASHES[method]):
        raise ValueError(f"fixed checkpoint identity invalid: {path}")
    if packet.get("checkpoint_sha256") not in (None, CHECKPOINT_HASHES[method]):
        raise ValueError(f"checkpoint hash mismatch: {path}")
    split = Path(packet.get("split_manifest", ""))
    if not split.is_file() or int(json.loads(split.read_text()).get("seed", -1)) != 2025:
        raise ValueError(f"locked split invalid: {path}")
    split_sha = sha(split)
    if packet.get("split_manifest_sha256") not in (None, split_sha):
        raise ValueError(f"split hash mismatch: {path}")
    for task in TASKS:
        row = packet["tasks"][task]
        if set(row.get("exits", {})) != {str(layer) for layer in EXITS}:
            raise ValueError(f"exit coverage incomplete: {path}:{task}")
        if int(row.get("selected_exit_best_validation", -1)) not in EXITS:
            raise ValueError(f"validation-selected exit missing: {path}:{task}")
        metric = row.get("spec", {}).get("metric_name")
        for layer in EXITS:
            entry = row["exits"][str(layer)]
            if metric not in entry.get("validation", {}) or metric not in entry.get("test", {}):
                raise ValueError(f"metric coverage incomplete: {path}:{task}:L{layer}")
            head = Path(entry.get("head_path", ""))
            if not head.is_file():
                raise FileNotFoundError(f"head missing: {path}:{task}:L{layer}")
            if algorithm == "residual_projected_dynamic_fusion":
                if entry.get("checkpoint_sha256") != sha(head):
                    raise ValueError(f"V2 head hash mismatch: {path}:{task}:L{layer}")
                if set(entry.get("fusion_diagnostics", {})) != {str(x) for x in EXITS if x < layer}:
                    raise ValueError(f"V2 diagnostics incomplete: {path}:{task}:L{layer}")
    packet["_checkpoint_sha256"] = sha(checkpoint)
    packet["_split_sha256"] = split_sha
    return path, packet


def stats(values: list[float]) -> dict:
    if len(values) != len(SEEDS):
        raise ValueError(f"seven values required, got {len(values)}")
    return {"n": len(values), "mean": statistics.fmean(values), "sample_sd": statistics.stdev(values),
            "min": min(values), "max": max(values)}


def direction_aware_gain(v2_value: float, current_value: float, higher_is_better: bool) -> float:
    return v2_value - current_value if higher_is_better else current_value - v2_value


def summarize_gains(gain_groups: dict) -> list[dict]:
    summaries = []
    for key, rows in sorted(gain_groups.items()):
        method, task, comparison = key
        summaries.append({
            "method": method,
            "task": task,
            "comparison": comparison,
            "metric": rows[0]["metric"],
            **stats([float(row["direction_aware_gain"]) for row in rows]),
            "per_seed": json.dumps({row["head_seed"]: row["direction_aware_gain"] for row in rows}, sort_keys=True),
            "v2_exit_by_seed": json.dumps({row["head_seed"]: row["v2_exit"] for row in rows}, sort_keys=True),
            "reference_exit_by_seed": json.dumps({row["head_seed"]: row["reference_exit"] for row in rows}, sort_keys=True),
        })
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing empty output: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_figures(summary: list[dict], output: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    indexed = {(r["method"], r["algorithm"], r["task"], int(r["exit"])): r for r in summary}
    expected = {(m, a, t, e) for m in METHODS for a in ALGORITHMS for t in TASKS for e in EXITS}
    if set(indexed) != expected:
        raise ValueError("figure coverage mismatch")
    labels = {"current": "Current EE", "residual_projected_dynamic_fusion": "Residual Dynamic Fusion V2"}
    method_labels = {"p0": "P0", "p1": "P1", "p2": "P2", "p2_reverse": "P2-Reverse"}
    colors = {"current": "#0072B2", "residual_projected_dynamic_fusion": "#009E73"}
    markers = {"current": "o", "residual_projected_dynamic_fusion": "^"}
    task_labels = {"rfs": "RF sensing", "jamming": "Jamming classification", "human-activity": "Human activity sensing",
                   "interference": "Interference classification", "nrpos-outdoor": "NR positioning", "deepmimo-beam": "DeepMIMO beam prediction"}
    paths = []

    def draw(ax, method, task, show_sd=True):
        for algorithm in ALGORITHMS:
            rows = [indexed[method, algorithm, task, layer] for layer in EXITS]
            if show_sd:
                ax.errorbar(EXITS, [r["mean"] for r in rows], yerr=[r["sample_sd"] for r in rows],
                            color=colors[algorithm], marker=markers[algorithm], capsize=3, linewidth=1.8, label=labels[algorithm])
            else:
                ax.plot(EXITS, [r["mean"] for r in rows], color=colors[algorithm], marker=markers[algorithm],
                        linewidth=1.8, linestyle="-", label=labels[algorithm])
        ax.set_xticks(EXITS, [f"L{x}" for x in EXITS]); ax.set_xlabel("Encoder exit layer")
        ax.set_ylabel("Mean distance error (m; lower is better)" if task == "nrpos-outdoor" else "Macro accuracy")
        ax.grid(axis="y", alpha=0.25)

    for method in METHODS:
        for task in TASKS:
            fig, ax = plt.subplots(figsize=(7, 4)); draw(ax, method, task); ax.set_title(f"{method_labels[method]} — {task_labels[task]}"); ax.legend(); fig.tight_layout()
            for suffix, kwargs in (("png", {"dpi": 180}), ("pdf", {})):
                name = f"{method}_{task}_exit_performance.{suffix}"; fig.savefig(output / name, **kwargs); paths.append(name)
            plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9)); fig.suptitle(f"{method_labels[method]} — Layer-wise performance of early-exit algorithms", fontsize=17, y=0.99)
        for ax, task in zip(axes.flat, TASKS): draw(ax, method, task, show_sd=False); ax.set_title(task_labels[task])
        handles = [Line2D([0], [0], color=colors[a], marker=markers[a], label=labels[a]) for a in ALGORITHMS]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2, frameon=False)
        fig.text(0.5, 0.015, "Solid lines show mean performance across seven head seeds.", ha="center")
        fig.tight_layout(rect=(0.015, 0.045, 0.985, 0.86))
        stem = {"p0": "p0_algorithm_layerwise_performance", "p1": "p1_algorithm_layerwise_performance",
                "p2": "algorithm_layerwise_performance", "p2_reverse": "p2_reverse_algorithm_layerwise_performance"}[method]
        for suffix, kwargs in (("png", {"dpi": 300, "bbox_inches": "tight"}), ("pdf", {"bbox_inches": "tight"})):
            name = f"{stem}.{suffix}"; fig.savefig(output / name, **kwargs); paths.append(name)
        plt.close(fig)
    paths.extend(plot_p0_p2_comparison(summary, output))
    return paths


def plot_p0_p2_comparison(summary: list[dict], output: Path) -> list[str]:
    """Plot the four requested P0/P2 approaches over all six tasks, without SD bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    methods = ("p0", "p2")
    approaches = (
        ("p0", "current", "No multiscale and no fusion", "#0072B2", ":", "o"),
        ("p0", "residual_projected_dynamic_fusion", "No multiscale but fusion", "#0072B2", "-", "o"),
        ("p2", "current", "Multiscale with no fusion", "#D55E00", ":", "s"),
        ("p2", "residual_projected_dynamic_fusion", "Multiscale with fusion", "#D55E00", "-", "s"),
    )
    indexed = {(row["method"], row["algorithm"], row["task"], int(row["exit"])): row
               for row in summary if row["method"] in methods}
    expected = {(method, algorithm, task, layer)
                for method, algorithm, *_ in approaches for task in TASKS for layer in EXITS}
    if set(indexed) != expected:
        raise ValueError("P0/P2 comparison figure requires complete fixed-exit coverage")
    if any(int(row.get("n", 0)) != len(SEEDS) for row in indexed.values()):
        raise ValueError("P0/P2 comparison figure requires seven-seed aggregates")

    task_labels = {
        "rfs": "RF sensing",
        "jamming": "Jamming classification",
        "human-activity": "Human activity sensing",
        "interference": "Interference classification",
        "nrpos-outdoor": "NR positioning",
        "deepmimo-beam": "DeepMIMO beam prediction",
    }
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("P0/P2 Fusion V2 comparison across six downstream tasks", fontsize=17, y=0.99)
    for ax, task in zip(axes.flat, TASKS):
        for method, algorithm, label, color, linestyle, marker in approaches:
            rows = [indexed[method, algorithm, task, layer] for layer in EXITS]
            ax.plot(EXITS, [float(row["mean"]) for row in rows], color=color, marker=marker,
                    linewidth=1.9, markersize=5.5, linestyle=linestyle)
        ax.set_title(task_labels[task])
        ax.set_xticks(EXITS, [f"L{layer}" for layer in EXITS])
        ax.set_xlabel("Encoder exit layer")
        ax.set_ylabel("Mean distance error (m; lower is better)" if task == "nrpos-outdoor"
                      else "Mean macro accuracy")
        ax.grid(axis="y", alpha=0.25)

    handles = [Line2D([0], [0], color=color, marker=marker, linewidth=1.9,
                      linestyle=linestyle, label=label)
               for _, _, label, color, linestyle, marker in approaches]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94),
               ncol=2, frameon=False)
    fig.text(0.5, 0.015,
             "Lines show seven-seed mean test performance at each fixed exit (solid Fusion V2, "
             "dotted Current EE); positioning is lower-is-better.", ha="center")
    fig.tight_layout(rect=(0.015, 0.05, 0.985, 0.85))
    output.mkdir(parents=True, exist_ok=True)
    names = ["p0_p2_six_task_fusion_comparison.png", "p0_p2_six_task_fusion_comparison.pdf"]
    for name, kwargs in zip(names, ({"dpi": 300, "bbox_inches": "tight"}, {"bbox_inches": "tight"})):
        fig.savefig(output / name, **kwargs)
    plt.close(fig)
    return names


def build_p0_p2_supplement(output: Path = OUT) -> dict:
    """Add the P0/P2 four-approach figure to an already generated multiseed report."""
    output = Path(output)
    summary_path = output / "fixed_exit_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"existing fixed-exit summary missing: {summary_path}")
    with summary_path.open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    for row in summary:
        row["exit"] = int(row["exit"])
        row["n"] = int(row["n"])
        row["mean"] = float(row["mean"])
    names = plot_p0_p2_comparison(summary, output)

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["figures"] = list(dict.fromkeys([*manifest.get("figures", []), *names]))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    readme_path = output / "README.md"
    readme = readme_path.read_text() if readme_path.is_file() else ""
    note = ("\n## P0/P2 four-approach comparison\n\n"
            "`p0_p2_six_task_fusion_comparison.png` and `.pdf` compare Current EE and "
            "Residual Dynamic Fusion V2 at fixed exits for P0 (no multiscale) and P2 "
            "(multiscale). Each panel covers one downstream task; lines show the seven-seed "
            "mean without deviation bars (Current EE dotted, Fusion V2 solid).\n")
    if "p0_p2_six_task_fusion_comparison.png" not in readme:
        readme_path.write_text(readme.rstrip() + "\n" + note)
    return {"output": str(output), "figures": names}


def _build_report(current_root, v2_root, output) -> dict:
    sources, split_hashes = {}, set()
    for method in METHODS:
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                path, packet = load_packet(method, algorithm, seed, current_root, v2_root)
                sources[method, algorithm, seed] = (path, packet); split_hashes.add(packet["_split_sha256"])
    if len(split_hashes) != 1:
        raise ValueError("comparison split identities differ")
    fixed, selected, gains, gates, provenance = [], [], [], [], []
    fixed_groups, selected_groups, gain_groups, gate_groups = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for method in METHODS:
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                path, packet = sources[method, algorithm, seed]
                source_record = {"method": method, "algorithm": algorithm, "head_seed": seed, "metrics_path": str(path), "metrics_sha256": sha(path),
                                 "checkpoint_path": str(packet["checkpoint"]),
                                 "checkpoint_sha256": packet["_checkpoint_sha256"], "split_manifest_sha256": packet["_split_sha256"]}
                if algorithm == "residual_projected_dynamic_fusion":
                    runtime_path = v2_root / "runtime" / method / f"head_seed_{seed:03d}.json"
                    source_record["runtime_metadata_path"] = str(runtime_path) if runtime_path.is_file() else None
                    source_record["runtime_metadata"] = json.loads(runtime_path.read_text()) if runtime_path.is_file() else None
                provenance.append(source_record)
                for task in TASKS:
                    row, metric, higher = packet["tasks"][task], packet["tasks"][task]["spec"]["metric_name"], bool(packet["tasks"][task]["spec"]["higher_is_better"])
                    chosen_exit = int(row["selected_exit_best_validation"]); chosen = row["exits"][str(chosen_exit)]
                    selected.append({"method": method, "algorithm": algorithm, "head_seed": seed, "task": task, "metric": metric, "higher_is_better": higher,
                                     "selected_exit": chosen_exit, "validation_value": chosen["validation"][metric], "selected_test_value": chosen["test"][metric]})
                    selected_groups[method, algorithm, task].append(selected[-1])
                    for layer in EXITS:
                        entry = row["exits"][str(layer)]
                        item = {"method": method, "algorithm": algorithm, "head_seed": seed, "task": task, "exit": layer, "metric": metric,
                                "higher_is_better": higher, "validation_value": entry["validation"][metric], "test_value": entry["test"][metric], "selected_by_validation": layer == chosen_exit}
                        fixed.append(item); fixed_groups[method, algorithm, task, layer].append(item)
                        if algorithm == "residual_projected_dynamic_fusion":
                            for source, diagnostic in entry["fusion_diagnostics"].items():
                                gate = {"method": method, "task": task, "head_seed": seed, "target_exit": layer, "source_exit": int(source), **diagnostic}
                                gates.append(gate); gate_groups[method, task, layer, int(source)].append(gate)
        for seed in SEEDS:
            current = sources[method, "current", seed][1]
            v2 = sources[method, "residual_projected_dynamic_fusion", seed][1]
            for task in TASKS:
                metric = v2["tasks"][task]["spec"]["metric_name"]; higher = bool(v2["tasks"][task]["spec"]["higher_is_better"])
                for label, layer, reference in [("selected_vs_current", int(v2["tasks"][task]["selected_exit_best_validation"]), int(current["tasks"][task]["selected_exit_best_validation"]))] + [(f"fixed_L{x}_vs_current", x, x) for x in EXITS]:
                    v = float(v2["tasks"][task]["exits"][str(layer)]["test"][metric]); b = float(current["tasks"][task]["exits"][str(reference)]["test"][metric])
                    item = {"method": method, "task": task, "head_seed": seed, "comparison": label, "v2_exit": layer, "reference_exit": reference,
                            "metric": metric, "positive_is_improvement": True, "direction_aware_gain": direction_aware_gain(v, b, higher)}
                    gains.append(item); gain_groups[method, task, label].append(item)
    summary = []
    for key, rows in sorted(fixed_groups.items()):
        method, algorithm, task, layer = key; values = [float(r["test_value"]) for r in rows]
        summary.append({"method": method, "algorithm": algorithm, "task": task, "exit": layer, "metric": rows[0]["metric"], "higher_is_better": rows[0]["higher_is_better"], **stats(values), "per_seed": json.dumps({r["head_seed"]: r["test_value"] for r in rows}, sort_keys=True)})
    selected_summary = []
    for key, rows in sorted(selected_groups.items()):
        method, algorithm, task = key; values = [float(r["selected_test_value"]) for r in rows]; counts = Counter(int(r["selected_exit"]) for r in rows)
        selected_summary.append({"method": method, "algorithm": algorithm, "task": task, "metric": rows[0]["metric"], "higher_is_better": rows[0]["higher_is_better"], **stats(values), "per_seed": json.dumps({r["head_seed"]: r["selected_test_value"] for r in rows}, sort_keys=True), "selected_exit_by_seed": json.dumps({r["head_seed"]: r["selected_exit"] for r in rows}, sort_keys=True), "selection_frequency": json.dumps(dict(sorted(counts.items())), sort_keys=True)})
    gain_summary = summarize_gains(gain_groups)
    gate_summary = []
    for key, rows in sorted(gate_groups.items()):
        result = {"method": key[0], "task": key[1], "target_exit": key[2], "source_exit": key[3]}
        for field in ("gate_mean", "gate_median", "gate_p10", "gate_p90", "gate_sd", "relative_residual_norm_mean", "residual_norm_mean"):
            result.update({f"{field}_{name}": value for name, value in stats([float(r[field]) for r in rows]).items()})
        gate_summary.append(result)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_seed_fixed_exit_results.csv", fixed); write_csv(output / "per_seed_validation_selected_results.csv", selected)
    write_csv(output / "per_seed_direction_aware_gains.csv", gains); write_csv(output / "per_seed_v2_gate_diagnostics.csv", gates)
    write_csv(output / "fixed_exit_summary.csv", summary); write_csv(output / "validation_selected_summary.csv", selected_summary)
    write_csv(output / "direction_aware_gain_summary.csv", gain_summary); write_csv(output / "v2_gate_diagnostic_summary.csv", gate_summary)
    figures = plot_figures(summary, output)
    manifest = {"study": "residual_dynamic_fusion_multiseed", "methods": METHODS, "algorithms": ALGORITHMS, "head_seeds": SEEDS,
                "pretrain_seed": 0, "split_seed": 2025, "tasks": TASKS, "exits": EXITS, "selection": "best_validation", "test_used_for_selection": False,
                "figures": figures, "sources": provenance}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "README.md").write_text("# Residual Dynamic Fusion V2 — seven-head-seed study\n\n"
        "Current EE and Residual Dynamic Fusion V2 are compared for P0, P1, P2, and P2-Reverse using head seeds 0–6. Validation selects each exit; test evaluates it afterward. "
        "Aggregate tables report mean, sample SD, min/max, per-seed values, and selected-exit frequencies. Positive direction-aware gains favor V2; for positioning they are error reductions.\n")
    return {"output": str(output), "fixed_rows": len(fixed), "selected_rows": len(selected), "gain_rows": len(gains), "sources": len(provenance), "figures": len(figures)}


def build_report(current_root=CURRENT_ROOT, v2_root=V2_ROOT, output=OUT) -> dict:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fusion-v2-multiseed-", dir=output.parent) as temporary:
        staged = Path(temporary)
        result = _build_report(current_root, v2_root, staged)
        products = sorted(staged.iterdir(), key=lambda path: (path.name == "manifest.json", path.name))
        output.mkdir(parents=True, exist_ok=True)
        for product in products:
            os.replace(product, output / product.name)
    return {**result, "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT); parser.add_argument("--current-root", type=Path, default=CURRENT_ROOT); parser.add_argument("--v2-root", type=Path, default=V2_ROOT)
    parser.add_argument("--p0-p2-supplement", action="store_true",
                        help="add the P0/P2 six-task figure to an existing multiseed report")
    args = parser.parse_args()
    if args.p0_p2_supplement:
        print(json.dumps(build_p0_p2_supplement(args.output), indent=2))
        return 0
    print(json.dumps(build_report(args.current_root, args.v2_root, args.output), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
