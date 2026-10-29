#!/usr/bin/env python3
"""Fail-closed seven-seed report for Current EE versus Single-GELU Fusion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import tempfile
import uuid
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("p0", "p2")
ALGORITHMS = ("current", "single_gelu")
SEEDS = tuple(range(7))
TASKS = ("rfs", "jamming", "human-activity", "interference", "nrpos-outdoor", "deepmimo-beam")
EXITS = (2, 4, 6, 8)
CURRENT_ROOT = ROOT / "runs/masking_head_seeds/multi_block_n6"
SINGLE_GELU_ROOT = ROOT / "runs/ee_optimization/multi_block_n6/residual_fusion_adapter_ablation"
OUT = ROOT / "reports/early_exit_optimization/multi_block_n6/residual_dynamic_fusion_multiseed"
CHECKPOINT_HASHES = {
    "p0": "8faa7245b6692e2d74d753fde2268fb712264c02c1da19632c045d022d192f1f",
    "p2": "7e1e27d96183b8f9d1c0d6b7493b93945633ccfb1514555135b4d04f22bf18ac",
}
CHECKPOINT_PATHS = {
    "p0": ROOT / "runs/masking_head_seeds/checkpoint_bundle/checkpoints/multi_block_n6/P0/last.ckpt",
    "p2": ROOT / "runs/masking_head_seeds/checkpoint_bundle/checkpoints/multi_block_n6/P2/last.ckpt",
}
LABELS = {"current": "Current EE", "single_gelu": "Single-GELU Fusion"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packet_path(method: str, algorithm: str, seed: int, current_root=CURRENT_ROOT,
                single_gelu_root=SINGLE_GELU_ROOT) -> Path:
    if method not in METHODS or seed not in SEEDS:
        raise ValueError("unsupported method or head seed")
    if algorithm == "current":
        return Path(current_root) / f"head_seed_{seed:03d}" / method / "seed_000/downstream/exit_metrics.json"
    if algorithm == "single_gelu":
        return (Path(single_gelu_root) / f"head_seed_{seed:03d}" / "single_gelu" / method
                / "seed_000/downstream/exit_metrics.json")
    raise ValueError(f"unsupported algorithm: {algorithm}")


def _validate_protocol(packet: dict, path: Path) -> None:
    protocol = packet.get("protocol", {})
    if (protocol.get("head_max_epochs") != 200 or protocol.get("batch_size") != 32
            or protocol.get("learning_rate") != 1e-3 or protocol.get("weight_decay") != 0.05
            or tuple(protocol.get("tasks", ())) != TASKS or tuple(protocol.get("exit_layers", ())) != EXITS
            or protocol.get("head_early_stopping", {}).get("enabled") is not True
            or protocol.get("head_early_stopping", {}).get("patience") != 20
            or protocol.get("test_evaluation") is not True):
        raise ValueError(f"locked protocol mismatch: {path}")


def load_packet(method: str, algorithm: str, seed: int, current_root=CURRENT_ROOT,
                single_gelu_root=SINGLE_GELU_ROOT) -> tuple[Path, dict]:
    path = packet_path(method, algorithm, seed, current_root, single_gelu_root)
    if not path.is_file():
        raise FileNotFoundError(f"required {algorithm} packet missing: {path}")
    packet = json.loads(path.read_text())
    if (packet.get("method") != method or int(packet.get("head_seed", -1)) != seed
            or int(packet.get("pretrain_seed", -1)) != 0 or packet.get("split_seed") not in (None, 2025)):
        raise ValueError(f"packet provenance mismatch: {path}")
    if algorithm == "single_gelu":
        if (packet.get("head_algorithm") != "residual_fusion_adapter_ablation"
                or packet.get("adapter_variant") != "single_gelu"):
            raise ValueError(f"Single-GELU algorithm identity mismatch: {path}")
        _validate_protocol(packet, path)
    elif packet.get("protocol", {}).get("test_evaluation") is not True:
        raise ValueError(f"Current EE test evaluation incomplete: {path}")
    if set(packet.get("tasks", {})) != set(TASKS):
        raise ValueError(f"task coverage incomplete: {path}")
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
            if algorithm == "single_gelu":
                if entry.get("checkpoint_sha256") != sha(head):
                    raise ValueError(f"Single-GELU head hash mismatch: {path}:{task}:L{layer}")
                expected_sources = {str(source) for source in EXITS if source < layer}
                if set(entry.get("fusion_diagnostics", {})) != expected_sources:
                    raise ValueError(f"Single-GELU diagnostics incomplete: {path}:{task}:L{layer}")
    packet["_checkpoint_sha256"] = sha(checkpoint)
    packet["_split_sha256"] = split_sha
    return path, packet


def stats(values: list[float]) -> dict:
    if len(values) != len(SEEDS):
        raise ValueError(f"seven values required, got {len(values)}")
    return {"n": len(values), "mean": statistics.fmean(values), "sample_sd": statistics.stdev(values),
            "min": min(values), "max": max(values)}


def split_stats(values: list[float]) -> tuple[dict, dict]:
    """Keep central summaries and variability in separate report tables."""
    aggregate = stats(values)
    mean_summary = {key: value for key, value in aggregate.items() if key != "sample_sd"}
    variability = {"n": aggregate["n"], "sample_variance": statistics.variance(values),
                   "sample_sd": aggregate["sample_sd"]}
    return mean_summary, variability


def direction_aware_gain(fusion_value: float, current_value: float, higher_is_better: bool) -> float:
    return fusion_value - current_value if higher_is_better else current_value - fusion_value


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing empty output: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _task_label(task: str) -> str:
    return {"rfs": "RF sensing", "jamming": "Jamming classification",
            "human-activity": "Human activity sensing", "interference": "Interference classification",
            "nrpos-outdoor": "NR positioning", "deepmimo-beam": "DeepMIMO beam prediction"}[task]


def plot_figures(summary: list[dict], output: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    indexed = {(row["method"], row["algorithm"], row["task"], int(row["exit"])): row for row in summary}
    expected = {(method, algorithm, task, layer) for method in METHODS for algorithm in ALGORITHMS
                for task in TASKS for layer in EXITS}
    if set(indexed) != expected:
        raise ValueError("figure coverage mismatch")
    colors, markers = {"current": "#666666", "single_gelu": "#009E73"}, {"current": "o", "single_gelu": "^"}
    names = []
    for method in METHODS:
        for task in TASKS:
            fig, axis = plt.subplots(figsize=(7, 4))
            for algorithm in ALGORITHMS:
                rows = [indexed[method, algorithm, task, layer] for layer in EXITS]
                axis.plot(EXITS, [row["mean"] for row in rows], color=colors[algorithm],
                          marker=markers[algorithm], linestyle="-", linewidth=1.8,
                          label=LABELS[algorithm])
            axis.set_title(f"{method.upper()} — {_task_label(task)}")
            axis.set_xticks(EXITS, [f"L{layer}" for layer in EXITS])
            axis.set_xlabel("Encoder exit layer")
            axis.set_ylabel("Mean distance error (m; lower is better)" if task == "nrpos-outdoor" else "Macro accuracy")
            axis.grid(axis="y", alpha=.25)
            axis.legend()
            fig.tight_layout()
            for suffix, kwargs in (("png", {"dpi": 180}), ("pdf", {})):
                name = f"{method}_{task}_exit_performance.{suffix}"
                fig.savefig(output / name, **kwargs)
                names.append(name)
            plt.close(fig)

    approaches = (
        ("p0", "current", "No multiscale and no fusion", "#0072B2", "-", "o"),
        ("p0", "single_gelu", "No multiscale but fusion", "#0072B2", "-", "^"),
        ("p2", "current", "Multiscale with no fusion", "#D55E00", "-", "s"),
        ("p2", "single_gelu", "Multiscale with fusion", "#D55E00", "-", "D"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("P0/P2 Single-GELU Fusion comparison across six downstream tasks", fontsize=17, y=.99)
    for axis, task in zip(axes.flat, TASKS):
        for method, algorithm, _label, color, linestyle, marker in approaches:
            rows = [indexed[method, algorithm, task, layer] for layer in EXITS]
            axis.plot(EXITS, [float(row["mean"]) for row in rows], color=color, marker=marker,
                      linewidth=1.9, markersize=5.5, linestyle=linestyle)
        axis.set_title(_task_label(task))
        axis.set_xticks(EXITS, [f"L{layer}" for layer in EXITS])
        axis.set_xlabel("Encoder exit layer")
        axis.set_ylabel("Mean distance error (m; lower is better)" if task == "nrpos-outdoor" else "Mean macro accuracy")
        axis.grid(axis="y", alpha=.25)
    handles = [Line2D([0], [0], color=color, marker=marker, linewidth=1.9, linestyle=linestyle, label=label)
               for _, _, label, color, linestyle, marker in approaches]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .94), ncol=2, frameon=False)
    fig.text(.5, .015, "Solid lines show seven-seed mean test performance at each fixed exit; "
             "markers distinguish Current EE from Fusion; positioning is lower-is-better.", ha="center")
    fig.tight_layout(rect=(.015, .05, .985, .85))
    for suffix, kwargs in (("png", {"dpi": 300, "bbox_inches": "tight"}), ("pdf", {"bbox_inches": "tight"})):
        name = f"p0_p2_six_task_single_gelu_comparison.{suffix}"
        fig.savefig(output / name, **kwargs)
        names.append(name)
    plt.close(fig)
    return names


def _summaries(groups: dict, value_key: str) -> list[dict]:
    result = []
    for key, rows in sorted(groups.items()):
        method, algorithm, task, *tail = key
        values = [float(row[value_key]) for row in rows]
        record = {"method": method, "algorithm": algorithm, "task": task, **stats(values),
                  "per_seed": json.dumps({row["head_seed"]: row[value_key] for row in rows}, sort_keys=True)}
        if tail:
            record["exit"] = tail[0]
        result.append(record)
    return result


def _build_report(current_root: Path, single_gelu_root: Path, output: Path) -> dict:
    sources, split_hashes = {}, set()
    for method in METHODS:
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                path, packet = load_packet(method, algorithm, seed, current_root, single_gelu_root)
                sources[method, algorithm, seed] = (path, packet)
                split_hashes.add(packet["_split_sha256"])
    if len(split_hashes) != 1:
        raise ValueError("comparison split identities differ")

    fixed, selected, gains, gates, provenance = [], [], [], [], []
    fixed_groups, selected_groups, gain_groups, gate_groups = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for method in METHODS:
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                path, packet = sources[method, algorithm, seed]
                record = {"method": method, "algorithm": algorithm, "head_seed": seed,
                          "metrics_path": str(path), "metrics_sha256": sha(path),
                          "checkpoint_path": str(packet["checkpoint"]),
                          "checkpoint_sha256": packet["_checkpoint_sha256"],
                          "split_manifest_sha256": packet["_split_sha256"]}
                if algorithm == "single_gelu":
                    runtime = single_gelu_root / "runtime_single_gelu_multiseed" / method / f"head_seed_{seed:03d}.json"
                    record["runtime_metadata_path"] = str(runtime) if runtime.is_file() else None
                    record["runtime_metadata"] = json.loads(runtime.read_text()) if runtime.is_file() else None
                provenance.append(record)
                for task in TASKS:
                    row, spec = packet["tasks"][task], packet["tasks"][task]["spec"]
                    metric, higher = spec["metric_name"], bool(spec["higher_is_better"])
                    chosen_exit = int(row["selected_exit_best_validation"])
                    chosen = row["exits"][str(chosen_exit)]
                    chosen_row = {"method": method, "algorithm": algorithm, "head_seed": seed, "task": task,
                                  "metric": metric, "higher_is_better": higher, "selected_exit": chosen_exit,
                                  "validation_value": chosen["validation"][metric],
                                  "selected_test_value": chosen["test"][metric]}
                    selected.append(chosen_row)
                    selected_groups[method, algorithm, task].append(chosen_row)
                    for layer in EXITS:
                        entry = row["exits"][str(layer)]
                        fixed_row = {"method": method, "algorithm": algorithm, "head_seed": seed, "task": task,
                                     "exit": layer, "metric": metric, "higher_is_better": higher,
                                     "validation_value": entry["validation"][metric], "test_value": entry["test"][metric],
                                     "selected_by_validation": layer == chosen_exit}
                        fixed.append(fixed_row)
                        fixed_groups[method, algorithm, task, layer].append(fixed_row)
                        if algorithm == "single_gelu":
                            for source, diagnostic in entry["fusion_diagnostics"].items():
                                gate = {"method": method, "task": task, "head_seed": seed, "target_exit": layer,
                                        "source_exit": int(source), **diagnostic}
                                gates.append(gate)
                                gate_groups[method, task, layer, int(source)].append(gate)
        for seed in SEEDS:
            current = sources[method, "current", seed][1]
            fusion = sources[method, "single_gelu", seed][1]
            for task in TASKS:
                metric = fusion["tasks"][task]["spec"]["metric_name"]
                higher = bool(fusion["tasks"][task]["spec"]["higher_is_better"])
                comparisons = [("selected_vs_current", int(fusion["tasks"][task]["selected_exit_best_validation"]),
                                int(current["tasks"][task]["selected_exit_best_validation"]))]
                comparisons += [(f"fixed_L{layer}_vs_current", layer, layer) for layer in EXITS]
                for comparison, fusion_exit, current_exit in comparisons:
                    gain = {"method": method, "task": task, "head_seed": seed, "comparison": comparison,
                            "fusion_exit": fusion_exit, "reference_exit": current_exit, "metric": metric,
                            "positive_is_improvement": True,
                            "direction_aware_gain": direction_aware_gain(
                                float(fusion["tasks"][task]["exits"][str(fusion_exit)]["test"][metric]),
                                float(current["tasks"][task]["exits"][str(current_exit)]["test"][metric]), higher)}
                    gains.append(gain)
                    gain_groups[method, task, comparison].append(gain)

    fixed_summary, fixed_variability = [], []
    for key, rows in sorted(fixed_groups.items()):
        method, algorithm, task, layer = key
        values = [float(row["test_value"]) for row in rows]
        central, variability = split_stats(values)
        identity = {"method": method, "algorithm": algorithm, "task": task, "exit": layer,
                    "metric": rows[0]["metric"], "higher_is_better": rows[0]["higher_is_better"]}
        fixed_summary.append({**identity, **central,
                              "per_seed": json.dumps({row["head_seed"]: row["test_value"] for row in rows}, sort_keys=True)})
        fixed_variability.append({**identity, **variability})
    selected_summary, selected_variability = [], []
    for key, rows in sorted(selected_groups.items()):
        method, algorithm, task = key
        counts = Counter(int(row["selected_exit"]) for row in rows)
        values = [float(row["selected_test_value"]) for row in rows]
        central, variability = split_stats(values)
        identity = {"method": method, "algorithm": algorithm, "task": task,
                    "metric": rows[0]["metric"], "higher_is_better": rows[0]["higher_is_better"]}
        selected_summary.append({**identity, **central,
                                 "per_seed": json.dumps({row["head_seed"]: row["selected_test_value"] for row in rows}, sort_keys=True),
                                 "selected_exit_by_seed": json.dumps({row["head_seed"]: row["selected_exit"] for row in rows}, sort_keys=True),
                                 "selection_frequency": json.dumps(dict(sorted(counts.items())), sort_keys=True)})
        selected_variability.append({**identity, **variability})
    gain_summary, gain_variability = [], []
    for (method, task, comparison), rows in sorted(gain_groups.items()):
        values = [float(row["direction_aware_gain"]) for row in rows]
        central, variability = split_stats(values)
        identity = {"method": method, "task": task, "comparison": comparison, "metric": rows[0]["metric"]}
        gain_summary.append({**identity, **central,
                             "per_seed": json.dumps({row["head_seed"]: row["direction_aware_gain"] for row in rows}, sort_keys=True),
                             "fusion_exit_by_seed": json.dumps({row["head_seed"]: row["fusion_exit"] for row in rows}, sort_keys=True),
                             "reference_exit_by_seed": json.dumps({row["head_seed"]: row["reference_exit"] for row in rows}, sort_keys=True)})
        gain_variability.append({**identity, **variability})
    gate_summary, gate_variability = [], []
    for (method, task, target_exit, source_exit), rows in sorted(gate_groups.items()):
        aggregate = {"method": method, "task": task, "target_exit": target_exit, "source_exit": source_exit}
        spread = {"method": method, "task": task, "target_exit": target_exit, "source_exit": source_exit,
                  "n": len(rows)}
        for field in ("gate_mean", "gate_median", "gate_p10", "gate_p90", "gate_sd",
                      "relative_residual_norm_mean", "residual_norm_mean"):
            central, variability = split_stats([float(row[field]) for row in rows])
            aggregate.update({f"{field}_{key}": value for key, value in central.items()})
            spread.update({f"{field}_{key}": value for key, value in variability.items() if key != "n"})
        gate_summary.append(aggregate)
        gate_variability.append(spread)

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_seed_fixed_exit_results.csv", fixed)
    write_csv(output / "per_seed_validation_selected_results.csv", selected)
    write_csv(output / "per_seed_direction_aware_gains.csv", gains)
    write_csv(output / "per_seed_single_gelu_gate_diagnostics.csv", gates)
    write_csv(output / "fixed_exit_summary.csv", fixed_summary)
    write_csv(output / "validation_selected_summary.csv", selected_summary)
    write_csv(output / "direction_aware_gain_summary.csv", gain_summary)
    write_csv(output / "single_gelu_gate_diagnostic_summary.csv", gate_summary)
    write_csv(output / "fixed_exit_variance_sd.csv", fixed_variability)
    write_csv(output / "validation_selected_variance_sd.csv", selected_variability)
    write_csv(output / "direction_aware_gain_variance_sd.csv", gain_variability)
    write_csv(output / "single_gelu_gate_variance_sd.csv", gate_variability)
    figures = plot_figures(fixed_summary, output)
    manifest = {"study": "single_gelu_fusion_multiseed", "methods": METHODS, "algorithms": ALGORITHMS,
                "head_seeds": SEEDS, "pretrain_seed": 0, "split_seed": 2025, "tasks": TASKS, "exits": EXITS,
                "selection": "best_validation", "test_used_for_selection": False, "figures": figures,
                "sources": provenance}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "README.md").write_text(
        "# Single-GELU Fusion — seven-head-seed study\n\n"
        "Current EE and Single-GELU Fusion are compared for P0 and P2 using head seeds 0–6. "
        "Validation selects each exit and held-out test evaluates it afterward. Aggregate tables report "
        "mean, min/max, per-seed values, and selected-exit frequencies. Figures draw mean lines without "
        "variance/error bars; all plotted lines are solid and marker shapes distinguish methods in the combined figure. "
        "Sample variance and sample SD are stored separately in the `*_variance_sd.csv` files. "
        "Positive direction-aware "
        "gains favor Single-GELU Fusion; for positioning they are error reductions.\n")
    return {"output": str(output), "fixed_rows": len(fixed), "selected_rows": len(selected),
            "gain_rows": len(gains), "sources": len(provenance), "figures": len(figures)}


def build_report(current_root=CURRENT_ROOT, single_gelu_root=SINGLE_GELU_ROOT, output=OUT) -> dict:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.staged-", dir=output.parent) as temporary:
        staged = Path(temporary)
        result = _build_report(Path(current_root), Path(single_gelu_root), staged)
        backup = output.parent / f".{output.name}.previous-{uuid.uuid4().hex}"
        moved_previous = False
        try:
            if output.exists():
                os.replace(output, backup)
                moved_previous = True
            os.replace(staged, output)
        except Exception:
            if moved_previous and backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
        if moved_previous:
            shutil.rmtree(backup)
    return {**result, "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--current-root", type=Path, default=CURRENT_ROOT)
    parser.add_argument("--single-gelu-root", type=Path, default=SINGLE_GELU_ROOT)
    args = parser.parse_args()
    print(json.dumps(build_report(args.current_root, args.single_gelu_root, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
