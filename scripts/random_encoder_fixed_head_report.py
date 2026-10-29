#!/usr/bin/env python3
"""Report fixed pretrained seed-0 EE heads evaluated over seven random encoders."""
from __future__ import annotations

import csv
import json
import math
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import random_encoder_ablation as control
import single_gelu_fusion_multiseed_report as prior

ROOT = control.ROOT
RUN_ROOT = ROOT / "runs/ee_optimization/multi_block_n6/random_encoder_fixed_head_eval_multiseed"
OUT = ROOT / "reports/early_exit_optimization/multi_block_n6/random_encoder_ablation_multiseed"
SEEDS, METHODS, ALGORITHMS, TASKS, EXITS = prior.SEEDS, prior.METHODS, prior.ALGORITHMS, prior.TASKS, prior.EXITS


def packet_path(seed: int, method: str, algorithm: str, run_root=RUN_ROOT) -> Path:
    root = Path(run_root) / f"random_seed_{seed:03d}" / method / algorithm
    if algorithm == "current":
        return root / "current" / method / "seed_000/downstream/exit_metrics.json"
    return root / "fusion/residual_fusion_adapter_ablation/head_seed_000/single_gelu" / method / "seed_000/downstream/exit_metrics.json"


def split_hash_matches(packet, expected_hash: str) -> bool:
    # Current-EE packets predate the explicit split hash field; their exact shared
    # manifest path is still checked below. Fusion packets carry the hash.
    return packet.get("split_manifest_sha256") in (None, expected_hash)


def current_head_identity_matches(packet) -> bool:
    # Legacy Current-EE evaluation packets encode identity in the validated
    # source packet and fixed head paths rather than a head_algorithm field.
    return packet.get("head_algorithm") in (None, "current_ee")


def validate_eval(seed, method, algorithm, run_root=RUN_ROOT):
    path = packet_path(seed, method, algorithm, run_root)
    if not path.is_file():
        raise FileNotFoundError(f"random encoder evaluation packet missing: {path}")
    packet = json.loads(path.read_text())
    encoder_root = control.RUN_ROOT if seed == 0 else ROOT / "runs/ee_optimization/multi_block_n6/random_encoder_ablation_multiseed" / f"seed_{seed:03d}"
    checkpoint = encoder_root / f"random_encoder_seed_{seed:03d}.pt"
    metadata = encoder_root / f"random_encoder_seed_{seed:03d}.json"
    record = control.validate(checkpoint, metadata, seed)
    if (packet.get("encoder_kind") != "random" or packet.get("method") != method
            or packet.get("head_seed") != 0 or packet.get("pretrain_seed") is not None
            or packet.get("random_encoder_seed") != seed or packet.get("checkpoint_sha256") != record["checkpoint_sha256"]
            or Path(packet.get("checkpoint", "")).resolve() != checkpoint.resolve()
            or Path(packet.get("random_encoder_metadata", "")).resolve() != metadata.resolve()
            or Path(packet.get("split_manifest", "")).resolve() != control.SPLIT.resolve()
            or not split_hash_matches(packet, control.sha(control.SPLIT))):
        raise ValueError(f"random encoder or fixed-head provenance mismatch: {path}")
    if algorithm == "current" and not current_head_identity_matches(packet):
        raise ValueError(f"wrong Current EE head identity: {path}")
    if algorithm == "single_gelu" and (packet.get("head_algorithm") != "residual_fusion_adapter_ablation"
            or packet.get("adapter_variant") != "single_gelu"):
        raise ValueError(f"wrong Fusion head identity: {path}")
    if packet.get("protocol", {}).get("test_evaluation") is not True or packet.get("protocol", {}).get("selection_uses") != "validation_only":
        raise ValueError(f"test evaluation is incomplete: {path}")
    source_path, source = prior.load_packet(method, algorithm, 0)
    if source["head_seed"] != 0 or set(packet.get("tasks", {})) != set(TASKS):
        raise ValueError(f"fixed source heads or task coverage invalid: {path}")
    for task in TASKS:
        task_row = packet["tasks"][task]
        source_row = source["tasks"][task]
        if int(task_row["selected_exit_best_validation"]) != int(source_row["selected_exit_best_validation"]):
            raise ValueError(f"exit selection changed from fixed source heads: {path}:{task}")
        if set(task_row.get("exits", {})) != {str(layer) for layer in EXITS}:
            raise ValueError(f"exit coverage incomplete: {path}:{task}")
        metric = task_row["spec"]["metric_name"]
        for layer in EXITS:
            entry = task_row["exits"][str(layer)]
            source_entry = source_row["exits"][str(layer)]
            if Path(entry.get("head_path", "")).resolve() != Path(source_entry["head_path"]).resolve():
                raise ValueError(f"head weights were not kept fixed: {path}:{task}:L{layer}")
            if algorithm == "single_gelu" and entry.get("checkpoint_sha256") != control.sha(Path(source_entry["head_path"])):
                raise ValueError(f"Fusion head hash mismatch: {path}:{task}:L{layer}")
            if metric not in entry.get("test", {}) or not math.isfinite(float(entry["test"][metric])):
                raise ValueError(f"test metric missing: {path}:{task}:L{layer}")
    return path, packet, source_path, source


def load_sources(run_root=RUN_ROOT):
    sources = {}
    for seed in SEEDS:
        for method in METHODS:
            for algorithm in ALGORITHMS:
                sources[seed, method, algorithm] = validate_eval(seed, method, algorithm, run_root)
    return sources


def summarize(rows, keys, value_key, include_selection=False):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    summary, variability = [], []
    for group, values in sorted(groups.items()):
        if {row["encoder_seed"] for row in values} != set(SEEDS) or len(values) != len(SEEDS):
            raise ValueError(f"incomplete random encoder seed coverage: {group}")
        nums = [float(row[value_key]) for row in values]
        base = dict(zip(keys, group))
        record = {**base, "n": len(nums), "mean": statistics.fmean(nums), "min": min(nums), "max": max(nums),
                  "per_encoder_seed": json.dumps({row["encoder_seed"]: row[value_key] for row in values}, sort_keys=True)}
        if include_selection:
            record["selected_exit_by_encoder_seed"] = json.dumps({row["encoder_seed"]: row["selected_exit"] for row in values}, sort_keys=True)
            record["selection_frequency"] = json.dumps(dict(Counter(row["selected_exit"] for row in values)), sort_keys=True)
        summary.append(record)
        variability.append({**base, "n": len(nums), "sample_variance": statistics.variance(nums), "sample_sd": statistics.stdev(nums)})
    return summary, variability


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing empty report table: {path}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def plot_figures(random_summary, reference_rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    index = {(row["method"], row["algorithm"], row["task"], int(row["exit"])): row["mean"] for row in random_summary}
    refs = {(row["method"], row["algorithm"], row["task"], int(row["exit"])): float(row["test_value"]) for row in reference_rows}
    names = []
    colors = {"current": "#0072B2", "single_gelu": "#009E73"}
    random_colors = {"current": "#56B4E9", "single_gelu": "#66C2A5"}
    for method in METHODS:
        for task in TASKS:
            fig, ax = plt.subplots(figsize=(7, 4))
            for algorithm in ALGORITHMS:
                ax.plot(EXITS, [refs[method, algorithm, task, layer] for layer in EXITS], color=colors[algorithm],
                    marker="o", linestyle="-", linewidth=1.8, label=f"{method.upper()} pretrained encoder · {algorithm}")
                ax.plot(EXITS, [index[method, algorithm, task, layer] for layer in EXITS], color=random_colors[algorithm],
                    marker="^", linestyle="-", linewidth=1.8, label=f"Random encoder mean · {algorithm}")
            ax.set_title(f"{method.upper()} fixed seed-0 heads — {prior._task_label(task)}")
            ax.set_xticks(EXITS, [f"L{x}" for x in EXITS]); ax.set_xlabel("Encoder exit layer")
            ax.set_ylabel("Mean distance error (m; lower is better)" if task == "nrpos-outdoor" else "Macro accuracy")
            ax.grid(axis="y", alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
            for suffix in ("png", "pdf"):
                name = f"{method}_{task}_fixed_head_random_encoder.{suffix}"
                fig.savefig(output / name, dpi=220 if suffix == "png" else None); names.append(name)
            plt.close(fig)
    return names


def build_report(run_root=RUN_ROOT, output=OUT):
    sources = load_sources(run_root)
    per_seed, selected, gains, provenance, refs = [], [], [], [], []
    for method in METHODS:
        for algorithm in ALGORITHMS:
            source_path, source = prior.load_packet(method, algorithm, 0)
            for task in TASKS:
                source_row = source["tasks"][task]; metric = source_row["spec"]["metric_name"]
                for layer in EXITS:
                    ref_entry = source_row["exits"][str(layer)]
                    refs.append({"method": method, "algorithm": algorithm, "task": task, "exit": layer,
                        "metric": metric, "test_value": ref_entry["test"][metric], "source_head_seed": 0,
                        "source_checkpoint_sha256": source["_checkpoint_sha256"]})
            for seed in SEEDS:
                path, packet, _, _ = sources[seed, method, algorithm]
                source_head_hashes = {f"{task}:L{layer}": control.sha(Path(source["tasks"][task]["exits"][str(layer)]["head_path"]))
                                      for task in TASKS for layer in EXITS}
                provenance.append({"encoder_seed": seed, "method": method, "algorithm": algorithm,
                    "fixed_head_seed": 0, "metrics_path": str(path), "metrics_sha256": control.sha(path),
                    "random_encoder_checkpoint": packet["checkpoint"], "random_encoder_checkpoint_sha256": packet["checkpoint_sha256"],
                    "head_source_metrics": str(source_path), "head_source_metrics_sha256": control.sha(source_path),
                    "fixed_head_sha256": source_head_hashes,
                    "split_manifest_sha256": control.sha(control.SPLIT)})
                for task in TASKS:
                    row = packet["tasks"][task]; metric = row["spec"]["metric_name"]
                    selected_exit = int(row["selected_exit_best_validation"])
                    for layer in EXITS:
                        value = float(row["exits"][str(layer)]["test"][metric])
                        per_seed.append({"method": method, "algorithm": algorithm, "task": task, "exit": layer,
                            "encoder_seed": seed, "fixed_head_seed": 0, "metric": metric,
                            "test_value": value, "selected_exit": selected_exit})
                        baseline = float(source["tasks"][task]["exits"][str(layer)]["test"][metric])
                        higher = bool(row["spec"]["higher_is_better"])
                        gains.append({"method": method, "algorithm": algorithm, "task": task, "exit": layer,
                            "encoder_seed": seed, "metric": metric, "pretrained_reference": baseline,
                            "random_encoder_value": value,
                            "pretraining_gain": baseline - value if higher else value - baseline})
                    value = float(row["exits"][str(selected_exit)]["test"][metric])
                    selected.append({"method": method, "algorithm": algorithm, "task": task,
                        "encoder_seed": seed, "fixed_head_seed": 0, "metric": metric,
                        "selected_exit": selected_exit, "test_value": value})
    fixed_summary, fixed_var = summarize(per_seed, ("method", "algorithm", "task", "exit", "metric"), "test_value")
    selected_summary, selected_var = summarize(selected, ("method", "algorithm", "task", "metric"), "test_value", include_selection=True)
    gain_summary, gain_var = summarize(gains, ("method", "algorithm", "task", "exit", "metric"), "pretraining_gain")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="random_fixed_heads_", dir=output.parent) as temp:
        stage = Path(temp)
        tables = (("per_seed_fixed_exit_results.csv", per_seed), ("fixed_exit_summary.csv", fixed_summary),
            ("fixed_exit_variance_sd.csv", fixed_var), ("per_seed_validation_selected_results.csv", selected),
            ("validation_selected_summary.csv", selected_summary), ("validation_selected_variance_sd.csv", selected_var),
            ("per_seed_pretraining_gains.csv", gains), ("pretraining_gain_summary.csv", gain_summary),
            ("pretraining_gain_variance_sd.csv", gain_var), ("pretrained_seed0_reference.csv", refs))
        for name, rows in tables:
            write_csv(stage / name, rows)
        figures = plot_figures(fixed_summary, refs, stage)
        runtime, submissions = [], {}
        for seed in SEEDS:
            allocation = (control.RUN_ROOT / "runtime_fixed_head/allocation.json" if seed == 0 else
                ROOT / "runs/ee_optimization/multi_block_n6/random_encoder_ablation_multiseed" / f"seed_{seed:03d}/runtime_fixed_head/allocation.json")
            if not allocation.is_file():
                raise FileNotFoundError(f"fixed-head evaluation allocation metadata missing: {allocation}")
            runtime.append(json.loads(allocation.read_text()))
        submission_path = Path(run_root) / "submissions.json"
        if not submission_path.is_file():
            raise FileNotFoundError(f"evaluation submission manifest missing: {submission_path}")
        submissions = json.loads(submission_path.read_text())
        (stage / "manifest.json").write_text(json.dumps({"study": "random_encoder_ablation_multiseed_fixed_pretrained_heads",
            "encoder_seeds": SEEDS, "fixed_head_seed": 0, "methods": METHODS, "algorithms": ALGORITHMS,
            "tasks": TASKS, "exits": EXITS, "sources": provenance, "runtime": runtime,
            "submission_manifest": submissions, "figures": figures,
            "selection_policy": "reuse the seed-0 pretrained-head validation-selected exit for all random encoders"}, indent=2, sort_keys=True) + "\n")
        (stage / "README.md").write_text("# Random encoder ablation — fixed-head seven-seed study\n\n"
            "Each P0/P2 Current EE or Single-GELU head set trained with head seed 0 is held fixed while its encoder is replaced with random frozen encoder seeds 0–6. "
            "The pretrained encoder result for the same seed-0 heads is the reference. No downstream head is retrained on a random encoder. "
            "Validation-selected exits are inherited from the pretrained seed-0 head packet and held constant across random encoder seeds. "
            "Mean lines are solid; variance and sample SD are in separate CSV files. Positive pretraining gains favor the pretrained encoder; positioning uses error reduction.\n")
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing report: {output}")
        stage.rename(output); output.chmod(0o755)
    return {"sources": len(provenance), "rows": len(per_seed), "output": str(output), "figures": figures}


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
