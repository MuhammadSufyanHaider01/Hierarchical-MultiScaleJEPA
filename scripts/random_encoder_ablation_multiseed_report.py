#!/usr/bin/env python3
"""Seven matched random encoder/head seeds versus fixed pretrained P0/P2 encoders."""
from __future__ import annotations

import json
import math
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import random_encoder_ablation as control
import random_encoder_ablation_report as random_report
import single_gelu_fusion_multiseed_report as prior

ROOT = control.ROOT
RUN_ROOT = ROOT / "runs/ee_optimization/multi_block_n6/random_encoder_ablation_multiseed"
OUT = ROOT / "reports/early_exit_optimization/multi_block_n6/random_encoder_ablation_multiseed"
SEEDS, TASKS, EXITS = prior.SEEDS, prior.TASKS, prior.EXITS
ALGORITHMS, CONDITIONS = prior.ALGORITHMS, ("random", "p0", "p2")


def seed_root(seed, run_root=RUN_ROOT):
    return control.RUN_ROOT if seed == 0 else Path(run_root) / f"seed_{seed:03d}"


def read_sources(run_root=RUN_ROOT):
    sources, split_hashes = {}, set()
    for seed in SEEDS:
        for algorithm in ALGORITHMS:
            path, packet = random_report.load_random(algorithm, seed_root(seed, run_root), seed)
            sources["random", algorithm, seed] = (path, packet)
            split_hashes.add(control.sha(Path(packet["split_manifest"])))
            for method in ("p0", "p2"):
                path, packet = prior.load_packet(method, algorithm, seed)
                sources[method, algorithm, seed] = (path, packet)
                split_hashes.add(packet["_split_sha256"])
        left = sources["random", "current", seed][1]
        right = sources["random", "single_gelu", seed][1]
        if left["checkpoint_sha256"] != right["checkpoint_sha256"]:
            raise ValueError(f"random algorithms do not share encoder seed {seed}")
    if split_hashes != {control.sha(control.SPLIT)}:
        raise ValueError("split manifest mismatch")
    hashes = [sources["random", "current", seed][1]["checkpoint_sha256"] for seed in SEEDS]
    if len(set(hashes)) != len(SEEDS):
        raise ValueError("random encoder checkpoints are not distinct")
    for task in TASKS:
        specs = [packet["tasks"][task]["spec"] for _, packet in sources.values()]
        if any((x["metric_name"], x["higher_is_better"]) != (specs[0]["metric_name"], specs[0]["higher_is_better"]) for x in specs):
            raise ValueError(f"metric mismatch for {task}")
        for path, packet in sources.values():
            metric = packet["tasks"][task]["spec"]["metric_name"]
            for layer in EXITS:
                entry = packet["tasks"][task]["exits"][str(layer)]
                if not all(math.isfinite(float(entry[part][metric])) for part in ("validation", "test")):
                    raise ValueError(f"non-finite metric: {path}:{task}:L{layer}")
    if len(sources) != 42:
        raise ValueError("expected 42 complete source packets")
    registry_path = Path(run_root) / "registry.json"
    if not registry_path.is_file():
        raise FileNotFoundError(f"study registry missing: {registry_path}")
    registry = json.loads(registry_path.read_text())
    for item in registry["inputs"]:
        path, _ = sources[item["encoder"], item["algorithm"], item["seed"]]
        if str(path) != item["metrics"] or control.sha(path) != item["metrics_sha256"]:
            raise ValueError(f"audited input changed: {path}")
    return sources


def summarize(rows, keys, value):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    means, variability = [], []
    for group, values in sorted(groups.items()):
        if {r["seed"] for r in values} != set(SEEDS) or len(values) != len(SEEDS):
            raise ValueError(f"incomplete seed coverage: {group}")
        nums = [float(r[value]) for r in values]
        prefix = dict(zip(keys, group))
        summary = {**prefix, "n": len(nums), "mean": statistics.fmean(nums), "min": min(nums), "max": max(nums),
                   "per_seed": json.dumps({r["seed"]: r[value] for r in values}, sort_keys=True)}
        if "selected_exit" in values[0]:
            summary["selected_exit_by_seed"] = json.dumps({r["seed"]: r["selected_exit"] for r in values}, sort_keys=True)
            summary["selection_frequency"] = json.dumps(dict(Counter(r["selected_exit"] for r in values)), sort_keys=True)
        means.append(summary)
        variability.append({**prefix, "n": len(nums), "sample_sd": statistics.stdev(nums), "sample_variance": statistics.variance(nums)})
    return means, variability


def figures(fixed, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    index = {(r["encoder"], r["algorithm"], r["task"], r["exit"]): r["mean"] for r in fixed}
    names = []
    for method in ("p0", "p2"):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for task, ax in zip(TASKS, axes.flat):
            for condition, algorithm, label, color, marker in (
                ("random", "current", "Random — Current EE", "#777777", "o"),
                ("random", "single_gelu", "Random — Single-GELU Fusion", "#777777", "^"),
                (method, "current", f"{method.upper()} pretrained — Current EE", "#0072B2", "o"),
                (method, "single_gelu", f"{method.upper()} pretrained — Single-GELU Fusion", "#0072B2", "^")):
                ax.plot(EXITS, [index[condition, algorithm, task, layer] for layer in EXITS], color=color, marker=marker, linestyle="-", label=label)
            ax.set_title(prior._task_label(task)); ax.set_xticks(EXITS, [f"L{x}" for x in EXITS]); ax.grid(axis="y", alpha=.25)
            ax.set_ylabel("Mean distance error (m; lower is better)" if task == "nrpos-outdoor" else "Mean macro accuracy")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, .9))
        for suffix in ("png", "pdf"):
            name = f"{method}_random_encoder_ablation_layerwise_performance.{suffix}"
            fig.savefig(output / name, dpi=220); names.append(name)
        plt.close(fig)
    return names


def build_report(run_root=RUN_ROOT, output=OUT):
    sources = read_sources(run_root)
    fixed, selected, gains, provenance, gates = [], [], [], [], []
    for (condition, algorithm, seed), (path, packet) in sources.items():
        provenance.append({"encoder": condition, "algorithm": algorithm, "head_seed": seed,
            "encoder_initialization_seed": seed if condition == "random" else None,
            "metrics_path": str(path), "metrics_sha256": control.sha(path),
            "checkpoint_path": packet["checkpoint"], "checkpoint_sha256": control.sha(Path(packet["checkpoint"])),
            "split_manifest_sha256": control.sha(Path(packet["split_manifest"]))})
        for task in TASKS:
            row = packet["tasks"][task]; spec = row["spec"]; metric = spec["metric_name"]
            base = {"encoder": condition, "algorithm": algorithm, "seed": seed, "task": task, "metric": metric, "higher_is_better": spec["higher_is_better"]}
            for layer in EXITS:
                entry = row["exits"][str(layer)]
                fixed.append({**base, "exit": layer, "validation_value": entry["validation"][metric], "test_value": entry["test"][metric]})
                if algorithm == "single_gelu":
                    gates.extend({"encoder": condition, "seed": seed, "task": task, "target_exit": layer, "source_exit": int(source), **values} for source, values in entry["fusion_diagnostics"].items())
            layer = row["selected_exit_best_validation"]; entry = row["exits"][str(layer)]
            selected.append({**base, "selected_exit": layer, "validation_value": entry["validation"][metric], "test_value": entry["test"][metric]})
    comparisons = [(method, algorithm, "random", algorithm, "pretraining") for method in ("p0", "p2") for algorithm in ALGORITHMS]
    comparisons += [(condition, "single_gelu", condition, "current", "fusion") for condition in CONDITIONS]
    for condition, algorithm, reference, ref_algorithm, kind in comparisons:
        for seed in SEEDS:
            left, right = sources[condition, algorithm, seed][1], sources[reference, ref_algorithm, seed][1]
            for task in TASKS:
                spec = left["tasks"][task]["spec"]; metric = spec["metric_name"]
                pairs = [(f"fixed_L{x}", x, x) for x in EXITS]
                pairs += [("validation_selected", left["tasks"][task]["selected_exit_best_validation"], right["tasks"][task]["selected_exit_best_validation"])]
                for comparison, layer, ref_layer in pairs:
                    value = float(left["tasks"][task]["exits"][str(layer)]["test"][metric])
                    ref = float(right["tasks"][task]["exits"][str(ref_layer)]["test"][metric])
                    gains.append({"comparison_type": kind, "encoder": condition, "algorithm": algorithm, "reference_encoder": reference,
                        "reference_algorithm": ref_algorithm, "seed": seed, "task": task, "metric": metric, "comparison": comparison,
                        "exit": layer, "reference_exit": ref_layer, "value": value, "reference_value": ref,
                        "direction_aware_gain": prior.direction_aware_gain(value, ref, spec["higher_is_better"])})
    fixed_summary, fixed_sd = summarize(fixed, ("encoder", "algorithm", "task", "exit", "metric", "higher_is_better"), "test_value")
    selected_summary, selected_sd = summarize(selected, ("encoder", "algorithm", "task", "metric", "higher_is_better"), "test_value")
    gain_summary, gain_sd = summarize(gains, ("comparison_type", "encoder", "algorithm", "reference_encoder", "reference_algorithm", "task", "metric", "comparison"), "direction_aware_gain")
    registry_path = Path(run_root) / "registry.json"
    submission_path = Path(run_root) / "submissions.json"
    if not registry_path.is_file() or not submission_path.is_file():
        raise FileNotFoundError("study registry or submission manifest missing")
    runtime = []
    for seed in SEEDS:
        for algorithm in ALGORITHMS:
            path = seed_root(seed, run_root) / "runtime" / f"{algorithm}.json"
            if not path.is_file():
                raise FileNotFoundError(f"runtime allocation metadata missing: {path}")
            runtime.append({"seed": seed, "algorithm": algorithm, **json.loads(path.read_text())})
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="random_encoder_multiseed_", dir=output.parent) as tmp:
        staged = Path(tmp)
        for name, rows in (("per_seed_fixed_exit_results.csv", fixed), ("fixed_exit_summary.csv", fixed_summary), ("fixed_exit_variance_sd.csv", fixed_sd),
            ("per_seed_validation_selected_results.csv", selected), ("validation_selected_summary.csv", selected_summary), ("validation_selected_variance_sd.csv", selected_sd),
            ("per_seed_direction_aware_gains.csv", gains), ("direction_aware_gain_summary.csv", gain_summary), ("direction_aware_gain_variance_sd.csv", gain_sd), ("per_seed_gate_diagnostics.csv", gates)):
            prior.write_csv(staged / name, rows)
        plots = figures(fixed_summary, staged)
        (staged / "manifest.json").write_text(json.dumps({"study": "random_encoder_ablation_multiseed", "seeds": SEEDS, "split_seed": 2025,
            "tasks": TASKS, "exits": EXITS, "sources": provenance, "figures": plots, "runtime": runtime,
            "registry_path": str(registry_path), "registry_sha256": control.sha(registry_path),
            "submission_manifest_path": str(submission_path), "submission_manifest_sha256": control.sha(submission_path),
            "pairing": "random encoder initialization seed equals head seed; pretrained encoder checkpoint fixed across head seeds"}, indent=2, sort_keys=True) + "\n")
        (staged / "README.md").write_text("# Random encoder ablation — seven-seed study\n\n"
            "Random encoders initialized with seeds 0–6 are compared with fixed pretrained P0 and P2 encoders, using matching head seeds, a shared split, Current EE and Single-GELU Fusion. "
            "Both random algorithms share the same encoder per seed. Validation selects exits before held-out testing; selected exits may differ between conditions. "
            "Positive direction-aware gains are improvements; positioning gains are reductions in distance error. "
            "Solid figure lines show fixed-exit seven-seed means; sample SD and variance are in separate CSVs. "
            "Paired gains vary random initialization and head seed together; pretrained checkpoint and split are fixed, so variability across pretrained initializations or dataset splits is not measured.\n")
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing report: {output}")
        staged.rename(output); output.chmod(0o755)
    return {"sources": len(provenance), "output": str(output), "figures": plots}


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
