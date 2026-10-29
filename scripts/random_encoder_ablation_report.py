#!/usr/bin/env python3
"""Fail-closed single-seed random versus pretrained encoder comparison."""
from __future__ import annotations

import csv
import json
import math
import tempfile
from pathlib import Path

import residual_fusion_adapter_ablation_report as prior
import random_encoder_ablation as random_control
import single_gelu_fusion_multiseed_report as multiseed

ROOT = prior.ROOT
OUT = ROOT / "reports/early_exit_optimization/multi_block_n6/random_encoder_ablation_single_seed"
TASKS, EXITS = prior.TASKS, prior.EXITS
ALGORITHMS = ("current", "single_gelu")
CONDITIONS = ("random", "p0", "p2")


def random_path(algorithm, run_root=random_control.RUN_ROOT, seed=0):
    root = Path(run_root)
    if algorithm == "current":
        return root / "current/p0/seed_000/downstream/exit_metrics.json"
    if algorithm == "single_gelu":
        return root / f"fusion/residual_fusion_adapter_ablation/head_seed_{seed:03d}/single_gelu/p0/seed_000/downstream/exit_metrics.json"
    raise ValueError("unknown algorithm")


def load_random(algorithm, run_root=random_control.RUN_ROOT, seed=0):
    root, path = Path(run_root), random_path(algorithm, run_root, seed)
    if not path.is_file():
        raise FileNotFoundError(f"random {algorithm} packet missing: {path}")
    record = random_control.validate(root / f"random_encoder_seed_{seed:03d}.pt", root / f"random_encoder_seed_{seed:03d}.json", seed)
    packet = json.loads(path.read_text())
    protocol = packet.get("protocol", {})
    if (packet.get("encoder_kind") != "random" or packet.get("method") != "p0"
            or packet.get("head_seed") != seed or packet.get("pretrain_seed") is not None or packet.get("split_seed") != 2025
            or Path(packet.get("checkpoint", "")).resolve() != Path(record["checkpoint"]).resolve()
            or packet.get("checkpoint_sha256") != record["checkpoint_sha256"]
            or packet.get("random_encoder_metadata") != str((root / f"random_encoder_seed_{seed:03d}.json").resolve())
            or Path(packet.get("split_manifest", "")).resolve() != random_control.SPLIT.resolve()
            or packet.get("split_manifest_sha256") != random_control.sha(random_control.SPLIT)):
        raise ValueError(f"random provenance mismatch: {path}")
    if (protocol.get("head_max_epochs") != 200 or protocol.get("batch_size") != 32
            or protocol.get("learning_rate") != 0.001 or protocol.get("weight_decay") != 0.05
            or tuple(protocol.get("tasks", ())) != TASKS or tuple(protocol.get("exit_layers", ())) != EXITS
            or protocol.get("head_early_stopping", {}).get("enabled") is not True
            or protocol.get("head_early_stopping", {}).get("patience") != 20
            or protocol.get("test_evaluation") is not True or protocol.get("selection_uses") != "validation_only"):
        raise ValueError(f"random protocol mismatch: {path}")
    if algorithm == "single_gelu" and (packet.get("head_algorithm") != "residual_fusion_adapter_ablation" or packet.get("adapter_variant") != "single_gelu"):
        raise ValueError("random fusion algorithm mismatch")
    if algorithm == "current" and packet.get("head_algorithm") != "current_ee":
        raise ValueError("random Current EE algorithm mismatch")
    if set(packet.get("tasks", {})) != set(TASKS):
        raise ValueError("random task coverage incomplete")
    for task, row in packet["tasks"].items():
        if set(row.get("exits", {})) != {str(x) for x in EXITS} or row.get("selected_exit_best_validation") not in EXITS:
            raise ValueError(f"random exit coverage incomplete: {task}")
        metric = row["spec"]["metric_name"]
        for layer in EXITS:
            entry = row["exits"][str(layer)]
            if not all(metric in entry.get(part, {}) and math.isfinite(float(entry[part][metric])) for part in ("validation", "test")):
                raise ValueError(f"random metric incomplete: {task} L{layer}")
            head = Path(entry.get("head_path", ""))
            if not head.is_file():
                raise FileNotFoundError(f"random head missing: {task} L{layer}")
            if algorithm == "single_gelu" and (entry.get("checkpoint_sha256") != random_control.sha(head)
                    or set(entry.get("fusion_diagnostics", {})) != {str(source) for source in EXITS if source < layer}):
                raise ValueError(f"random fusion diagnostics or head mismatch: {task} L{layer}")
    return path, packet


def read_sources(run_root=random_control.RUN_ROOT):
    sources = {(method, algorithm): prior.load_packet(method, algorithm) for method in ("p0", "p2") for algorithm in ALGORITHMS}
    for method in ("p0", "p2"):
        for algorithm in ALGORITHMS:
            multiseed.load_packet(method, algorithm, 0)
    sources.update({("random", algorithm): load_random(algorithm, run_root) for algorithm in ALGORITHMS})
    hashes = {random_control.sha(Path(packet["split_manifest"])) for _, packet in sources.values()}
    if hashes != {random_control.sha(random_control.SPLIT)}:
        raise ValueError("source split identities differ")
    for task in TASKS:
        specs = [packet["tasks"][task]["spec"] for _, packet in sources.values()]
        if any((x["metric_name"], x["higher_is_better"]) != (specs[0]["metric_name"], specs[0]["higher_is_better"]) for x in specs):
            raise ValueError(f"task metric mismatch: {task}")
    return sources


def figures(fixed, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    index = {(r["encoder"], r["algorithm"], r["task"], r["exit"]): r for r in fixed}
    files = []
    for method in ("p0", "p2"):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for task, axis in zip(TASKS, axes.flat):
            for condition, algorithm, label, color, marker in (("random", "current", "Random — Current EE", "#777777", "o"),
                    ("random", "single_gelu", "Random — Single-GELU Fusion", "#777777", "^"),
                    (method, "current", f"{method.upper()} pretrained — Current EE", "#0072B2", "o"),
                    (method, "single_gelu", f"{method.upper()} pretrained — Single-GELU Fusion", "#0072B2", "^")):
                axis.plot(EXITS, [index[condition, algorithm, task, layer]["test_value"] for layer in EXITS],
                          color=color, marker=marker, linestyle="-", label=label)
            axis.set_title(task); axis.set_xticks(EXITS, [f"L{x}" for x in EXITS]); axis.grid(axis="y", alpha=.25)
            axis.set_ylabel("Mean distance error (lower is better)" if task == "nrpos-outdoor" else "Macro accuracy")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, .9))
        for suffix in ("png", "pdf"):
            name = f"{method}_random_encoder_ablation_layerwise_performance.{suffix}"
            fig.savefig(output / name, dpi=220); files.append(name)
        plt.close(fig)
    return files


def build_report(run_root=random_control.RUN_ROOT, output=OUT):
    sources = read_sources(run_root)
    fixed, selected, gains, diagnostics, provenance = [], [], [], [], []
    for (condition, algorithm), (path, packet) in sources.items():
        provenance.append({"encoder": condition, "algorithm": algorithm, "metrics_path": str(path),
                           "metrics_sha256": random_control.sha(path), "checkpoint_sha256": random_control.sha(Path(packet["checkpoint"])),
                           "split_manifest_sha256": random_control.sha(Path(packet["split_manifest"]))})
        for task in TASKS:
            row = packet["tasks"][task]; spec = row["spec"]; metric = spec["metric_name"]
            chosen = row["selected_exit_best_validation"]
            for layer in EXITS:
                entry = row["exits"][str(layer)]
                fixed.append({"encoder": condition, "algorithm": algorithm, "task": task, "exit": layer,
                              "metric": metric, "higher_is_better": spec["higher_is_better"],
                              "validation_value": entry["validation"][metric], "test_value": entry["test"][metric]})
                if algorithm == "single_gelu":
                    diagnostics.extend({"encoder": condition, "task": task, "target_exit": layer, "source_exit": int(source), **values}
                                       for source, values in entry["fusion_diagnostics"].items())
            entry = row["exits"][str(chosen)]
            selected.append({"encoder": condition, "algorithm": algorithm, "task": task, "metric": metric,
                             "selected_exit": chosen, "validation_value": entry["validation"][metric], "test_value": entry["test"][metric]})
    comparisons = [(method, algorithm, "random", algorithm, "pretraining") for method in ("p0", "p2") for algorithm in ALGORITHMS]
    comparisons += [(condition, "single_gelu", condition, "current", "fusion") for condition in CONDITIONS]
    for condition, algorithm, reference, reference_algo, kind in comparisons:
        left, right = sources[condition, algorithm][1], sources[reference, reference_algo][1]
        for task in TASKS:
            spec = left["tasks"][task]["spec"]; metric, higher = spec["metric_name"], spec["higher_is_better"]
            pairs = [(f"fixed_L{layer}", layer, layer) for layer in EXITS]
            pairs += [("validation_selected", left["tasks"][task]["selected_exit_best_validation"], right["tasks"][task]["selected_exit_best_validation"])]
            for comparison, layer, baseline_layer in pairs:
                value = float(left["tasks"][task]["exits"][str(layer)]["test"][metric])
                base = float(right["tasks"][task]["exits"][str(baseline_layer)]["test"][metric])
                gains.append({"comparison_type": kind, "encoder": condition, "algorithm": algorithm, "reference_encoder": reference,
                              "reference_algorithm": reference_algo, "task": task, "metric": metric, "comparison": comparison,
                              "exit": layer, "reference_exit": baseline_layer, "value": value, "reference_value": base,
                              "direction_aware_gain": prior.gain(value, base, higher)})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="random_encoder_report_", dir=output.parent) as tmp:
        staged = Path(tmp)
        for name, rows in (("fixed_exit_results.csv", fixed), ("validation_selected_results.csv", selected),
                           ("direction_aware_gains.csv", gains), ("gate_diagnostics.csv", diagnostics)):
            prior.write_csv(staged / name, rows)
        plots = figures(fixed, staged)
        runtime = {}
        for algorithm in ALGORITHMS:
            runtime_path = Path(run_root) / "runtime" / f"{algorithm}.json"
            if not runtime_path.is_file():
                raise FileNotFoundError(f"runtime allocation metadata missing: {runtime_path}")
            runtime[algorithm] = json.loads(runtime_path.read_text())
        (staged / "manifest.json").write_text(json.dumps({"study": "random_encoder_ablation_single_seed", "head_seed": 0,
            "encoder_initialization_seed": 0, "split_seed": 2025, "tasks": TASKS, "exits": EXITS,
            "random_encoder": random_control.validate(Path(run_root) / "random_encoder_seed_000.pt", Path(run_root) / "random_encoder_seed_000.json"),
            "runtime": runtime, "sources": provenance, "figures": plots}, indent=2, sort_keys=True) + "\n")
        (staged / "README.md").write_text("# Random encoder ablation — single-seed study\n\n"
            "A shared frozen random encoder is compared with pretrained P0 and P2 using Current EE and Single-GELU Fusion. "
            "Validation selects exits before held-out testing. Positive gains indicate better performance; for positioning this means lower error. "
            "Each condition uses one head seed and the random condition uses one encoder initialization; variability and statistical significance are not estimated.\n")
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing report: {output}")
        staged.rename(output)
        output.chmod(0o755)
    return {"sources": len(provenance), "fixed_rows": len(fixed), "selected_rows": len(selected), "gains": len(gains), "figures": plots, "output": str(output)}


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
