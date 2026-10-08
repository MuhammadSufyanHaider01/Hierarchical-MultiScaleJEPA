#!/usr/bin/env python3
"""Validate complete EE artifacts and generate the multi-seed study report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

METHODS = ("p0", "p2")
ALGORITHMS = ("current", "causal_fusion", "fusion_teacher_kd")
TASKS = ("rfs", "jamming", "human-activity", "interference", "nrpos-outdoor", "deepmimo-beam")
SEEDS = tuple(range(7))
EXITS = (2, 4, 6, 8)
RUNS = ROOT / "runs/ee_optimization/multi_block_n6"
BASELINE = ROOT / "runs/masking_head_seeds/multi_block_n6"
OUT = ROOT / "reports/early_exit_optimization/multi_block_n6"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_packet(method: str, algorithm: str, seed: int) -> tuple[Path, dict]:
    if algorithm == "current":
        path = BASELINE / f"head_seed_{seed:03d}" / method / "seed_000/downstream/exit_metrics.json"
    else:
        name = "fusion" if algorithm == "causal_fusion" else "fusion_kd"
        path = RUNS / name / f"head_seed_{seed:03d}" / method / "seed_000/downstream/exit_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"required {algorithm} packet missing: {path}")
    packet = json.loads(path.read_text())
    if packet.get("method") != method or int(packet.get("head_seed", -1)) != seed or int(packet.get("pretrain_seed", -1)) != 0:
        raise ValueError(f"method/seed provenance mismatch: {path}")
    if algorithm != "current" and packet.get("head_algorithm") != algorithm:
        raise ValueError(f"head algorithm mismatch: {path}")
    if packet.get("protocol", {}).get("test_evaluation") is not True:
        raise ValueError(f"held-out test results missing: {path}")
    if set(packet.get("tasks", {})) != set(TASKS):
        raise ValueError(f"task coverage mismatch: {path}")
    split = Path(packet["split_manifest"])
    if not split.is_file() or int(json.loads(split.read_text()).get("seed", -1)) != 2025:
        raise ValueError(f"split provenance missing or invalid: {path}")
    expected_split_sha = packet.get("split_manifest_sha256")
    if expected_split_sha and sha(split) != expected_split_sha:
        raise ValueError(f"split hash mismatch: {path}")
    checkpoint_path = Path(packet.get("checkpoint", ""))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"pretrained checkpoint missing: {checkpoint_path}")
    actual_checkpoint_sha = sha(checkpoint_path)
    if actual_checkpoint_sha != CHECKPOINT_HASHES[method]:
        raise ValueError(f"fixed study checkpoint identity mismatch: {path}")
    if algorithm != "current" and packet.get("checkpoint_sha256") != actual_checkpoint_sha:
        raise ValueError(f"checkpoint hash mismatch: {path}")
    for task in TASKS:
        row = packet["tasks"][task]
        if set(row.get("exits", {})) != {str(layer) for layer in EXITS}:
            raise ValueError(f"exit coverage mismatch: {path}:{task}")
        if int(row.get("selected_exit_best_validation", -1)) not in EXITS:
            raise ValueError(f"validation-selected exit absent: {path}:{task}")
        for layer in EXITS:
            exit_row = row["exits"][str(layer)]
            metric = row.get("spec", {}).get("metric_name")
            if metric not in exit_row.get("test", {}) or metric not in exit_row.get("validation", {}):
                raise ValueError(f"validation/test metric absent: {path}:{task}:L{layer}")
            if algorithm != "current":
                head = Path(exit_row.get("head_path", ""))
                if not head.is_file():
                    raise FileNotFoundError(f"trained head missing: {head}")
                if algorithm == "causal_fusion" and not exit_row.get("fusion_weights"):
                    raise ValueError(f"fusion weights missing: {path}:{task}:L{layer}")
                if algorithm == "fusion_teacher_kd":
                    teacher = Path(exit_row.get("teacher_path", ""))
                    teacher_packet_path = Path(exit_row.get("teacher_metrics_path", ""))
                    if not teacher.is_file() or not teacher_packet_path.is_file():
                        raise FileNotFoundError(f"KD teacher provenance missing: {path}:{task}:L{layer}")
                    if sha(teacher) != exit_row.get("teacher_sha256") or sha(teacher_packet_path) != exit_row.get("teacher_metrics_sha256"):
                        raise ValueError(f"KD teacher provenance hash mismatch: {path}:{task}:L{layer}")
                    teacher_packet = json.loads(teacher_packet_path.read_text())
                    teacher_record = teacher_packet.get("tasks", {}).get(task, {}).get("exits", {}).get(str(layer), {})
                    if (teacher_packet.get("head_algorithm") != "causal_fusion"
                            or teacher_packet.get("method") != method
                            or int(teacher_packet.get("head_seed", -1)) != seed
                            or teacher_packet.get("split_manifest_sha256") != packet.get("split_manifest_sha256")
                            or Path(teacher_record.get("head_path", "")).resolve() != teacher.resolve()):
                        raise ValueError(f"KD teacher identity mismatch: {path}:{task}:L{layer}")
    return path, packet


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty report table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    sources = {}
    split_identities = set()
    for method in METHODS:
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                path, packet = read_packet(method, algorithm, seed)
                sources[(method, algorithm, seed)] = {"path": path, "packet": packet}
                split_identities.add(sha(Path(packet["split_manifest"])))
    if len(split_identities) != 1:
        raise ValueError(f"split identity mismatch across study packets: {sorted(split_identities)}")

    per_exit, selected, weights, teacher_gain, provenance = [], [], [], [], []
    groups = {}
    for (method, algorithm, seed), source in sources.items():
        packet = source["packet"]
        provenance.append({"method": method, "algorithm": algorithm, "head_seed": seed,
                           "metrics_path": str(source["path"]), "metrics_sha256": sha(source["path"]),
                           "checkpoint_sha256": sha(Path(packet["checkpoint"])),
                           "split_manifest": packet.get("split_manifest"),
                           "split_manifest_sha256": packet.get("split_manifest_sha256"),
                           "heads": [{"task": task, "exit": layer,
                               "path": row["tasks"][task]["exits"][str(layer)]["head_path"],
                               "sha256": sha(Path(row["tasks"][task]["exits"][str(layer)]["head_path"]))}
                               for task in TASKS for layer in EXITS]})
        for task in TASKS:
            row = packet["tasks"][task]
            metric = row["spec"]["metric_name"]
            direction = bool(row["spec"]["higher_is_better"])
            chosen = int(row["selected_exit_best_validation"])
            selected.append({"method": method, "algorithm": algorithm, "task": task, "head_seed": seed,
                             "metric": metric, "higher_is_better": direction, "selected_exit": chosen,
                             "validation_value": row["exits"][str(chosen)]["validation"][metric],
                             "selected_test_value": row["exits"][str(chosen)]["test"][metric]})
            for layer in EXITS:
                exit_row = row["exits"][str(layer)]
                value = exit_row["test"][metric]
                per_exit.append({"method": method, "algorithm": algorithm, "task": task, "head_seed": seed,
                                 "exit": layer, "metric": metric, "higher_is_better": direction,
                                 "validation_value": exit_row["validation"][metric], "test_value": value,
                                 "selected_exit": chosen})
                if algorithm == "causal_fusion":
                    for source_layer, alpha in exit_row["fusion_weights"].items():
                        weights.append({"method": method, "task": task, "head_seed": seed,
                                        "target_exit": layer, "source_exit": int(source_layer), "alpha": alpha})
                groups.setdefault((method, algorithm, task, layer), []).append(float(value))
                if algorithm in {"causal_fusion", "fusion_teacher_kd"}:
                    current = sources[(method, "current", seed)]["packet"]["tasks"][task]
                    new_value = exit_row["validation"][metric]
                    current_value = current["exits"][str(layer)]["validation"][metric]
                    gain = new_value - current_value if direction else current_value - new_value
                    teacher_gain.append({"method": method, "algorithm": algorithm, "task": task,
                        "head_seed": seed, "exit": layer, "metric": metric,
                        "positive_is_improvement": True, "validation_gain_vs_current": gain})

    summary = []
    for (method, algorithm, task, layer), values in sorted(groups.items()):
        summary.append({"method": method, "algorithm": algorithm, "task": task, "exit": layer,
                        "n": len(values), "mean": statistics.fmean(values),
                        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "min": min(values), "max": max(values), "per_seed": json.dumps(values)})
    exit_frequency = []
    for method in METHODS:
        for algorithm in ALGORITHMS:
            for task in TASKS:
                selected_rows = [r for r in selected if r["method"] == method and r["algorithm"] == algorithm and r["task"] == task]
                counts = Counter(r["selected_exit"] for r in selected_rows)
                for layer in EXITS:
                    exit_frequency.append({"method": method, "algorithm": algorithm, "task": task,
                                           "exit": layer, "count": counts.get(layer, 0), "seeds": len(selected_rows)})
    selected_summary = []
    direction_gains = []
    early_depth = []
    for method in METHODS:
        for algorithm in ("causal_fusion", "fusion_teacher_kd"):
            for seed in SEEDS:
                new_packet = sources[(method, algorithm, seed)]["packet"]
                current_packet = sources[(method, "current", seed)]["packet"]
                for task in TASKS:
                    new_task = new_packet["tasks"][task]
                    base_task = current_packet["tasks"][task]
                    metric = new_task["spec"]["metric_name"]
                    higher = bool(new_task["spec"]["higher_is_better"])
                    for layer in EXITS:
                        new_value = float(new_task["exits"][str(layer)]["test"][metric])
                        base_value = float(base_task["exits"][str(layer)]["test"][metric])
                        direction_gains.append({"method": method, "algorithm": algorithm, "task": task,
                            "head_seed": seed, "comparison": "fixed_exit", "exit": layer,
                            "metric": metric, "positive_is_improvement": True,
                            "gain_vs_current": new_value - base_value if higher else base_value - new_value})
                    for new_layer, current_layer in ((2, 4), (4, 6), (6, 8)):
                        new_value = float(new_task["exits"][str(new_layer)]["test"][metric])
                        current_value = float(base_task["exits"][str(current_layer)]["test"][metric])
                        gain = new_value - current_value if higher else current_value - new_value
                        early_depth.append({"method": method, "algorithm": algorithm, "task": task,
                            "head_seed": seed, "new_exit": new_layer, "current_reference_exit": current_layer,
                            "metric": metric, "positive_is_improvement": True,
                            "direction_aware_gain": gain, "matches_or_exceeds": gain >= 0})
                    new_selected = int(new_task["selected_exit_best_validation"])
                    base_selected = int(base_task["selected_exit_best_validation"])
                    new_value = float(new_task["exits"][str(new_selected)]["test"][metric])
                    base_value = float(base_task["exits"][str(base_selected)]["test"][metric])
                    direction_gains.append({"method": method, "algorithm": algorithm, "task": task,
                        "head_seed": seed, "comparison": "validation_selected", "exit": new_selected,
                        "current_selected_exit": base_selected, "metric": metric,
                        "positive_is_improvement": True,
                        "gain_vs_current": new_value - base_value if higher else base_value - new_value})
    for method in METHODS:
        for algorithm in ALGORITHMS:
            for task in TASKS:
                rows = [r for r in selected if r["method"] == method and r["algorithm"] == algorithm and r["task"] == task]
                vals = [float(r["selected_test_value"]) for r in rows]
                selected_summary.append({"method": method, "algorithm": algorithm, "task": task,
                    "metric": rows[0]["metric"], "higher_is_better": rows[0]["higher_is_better"],
                    "n": len(vals), "mean": statistics.fmean(vals),
                    "sample_sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "min": min(vals), "max": max(vals),
                    "per_seed": json.dumps({r["head_seed"]: r["selected_test_value"] for r in rows}),
                    "selected_exit_by_seed": json.dumps({r["head_seed"]: r["selected_exit"] for r in rows})})
    weight_groups = {}
    for row in weights:
        weight_groups.setdefault((row["method"], row["task"], row["target_exit"], row["source_exit"]), []).append(float(row["alpha"]))
    weight_summary = [{"method": method, "task": task, "target_exit": target, "source_exit": source,
        "n": len(vals), "mean": statistics.fmean(vals),
        "sample_sd": statistics.stdev(vals) if len(vals) > 1 else 0.0}
        for (method, task, target, source), vals in sorted(weight_groups.items())]
    early_groups = {}
    for row in early_depth:
        early_groups.setdefault((row["method"], row["algorithm"], row["task"],
                                 row["new_exit"], row["current_reference_exit"]), []).append(row)
    early_summary = [{"method": method, "algorithm": algorithm, "task": task,
        "new_exit": new_exit, "current_reference_exit": current_exit,
        "seeds": len(rows), "matches_or_exceeds_seeds": sum(bool(r["matches_or_exceeds"]) for r in rows),
        "matches_or_exceeds_fraction": sum(bool(r["matches_or_exceeds"]) for r in rows) / len(rows),
        "mean_direction_aware_gain": statistics.fmean(float(r["direction_aware_gain"]) for r in rows)}
        for (method, algorithm, task, new_exit, current_exit), rows in sorted(early_groups.items())]
    # Fusion-KD deploys the Current EE student; cost is the validated baseline cost.
    costs = []
    for method in METHODS:
        path = RUNS / "benchmarks/fusion" / method / "seed_000/benchmarks.json"
        if not path.is_file():
            raise FileNotFoundError(f"causal-fusion benchmark missing: {path}")
        fusion_bench = json.loads(path.read_text())
        if fusion_bench.get("head_algorithm") != "causal_fusion" or fusion_bench.get("batch_size") != 1:
            raise ValueError(f"incompatible Fusion benchmark: {path}")
        expected = {(task, layer) for task in TASKS for layer in EXITS}
        actual = {(e.get("task"), int(e.get("exit_layer", -1))) for e in fusion_bench.get("entries", [])}
        if actual != expected:
            raise ValueError(f"Fusion benchmark coverage mismatch: {path}")
        base_path = ROOT / "runs/masking_head_seeds/multi_block_n6/benchmarks" / method / "seed_000/benchmarks.json"
        if not base_path.is_file():
            raise FileNotFoundError(f"Current EE benchmark missing: {base_path}")
        base_bench = json.loads(base_path.read_text())
        if (base_bench.get("batch_size") != 1 or base_bench.get("method") != method
                or sha(Path(base_bench["checkpoint"])) != CHECKPOINT_HASHES[method]):
            raise ValueError(f"incompatible Current EE benchmark: {base_path}")
        base_actual = {(e.get("task"), int(e.get("exit_layer", -1))) for e in base_bench.get("entries", [])}
        if base_actual != expected:
            raise ValueError(f"Current EE benchmark coverage mismatch: {base_path}")
        if fusion_bench.get("checkpoint_sha256") != CHECKPOINT_HASHES[method]:
            raise ValueError(f"Fusion benchmark checkpoint identity mismatch: {path}")
        current_entries = {(e["task"], int(e["exit_layer"])): e for e in base_bench["entries"]}
        for entry in fusion_bench["entries"]:
            current = current_entries[(entry["task"], int(entry["exit_layer"]))]
            costs.append({"method": method, "task": entry["task"], "exit": entry["exit_layer"],
                          "algorithm": "current", "flops": current["flops"],
                          "latency_mean_ms": current["latency"]["mean_ms"], "teacher_training_only": False})
            costs.append({"method": method, "task": entry["task"], "exit": entry["exit_layer"],
                          "algorithm": "causal_fusion", "flops": entry["flops"],
                          "latency_mean_ms": entry["latency"]["mean_ms"], "teacher_training_only": False})
            costs.append({"method": method, "task": entry["task"], "exit": entry["exit_layer"],
                          "algorithm": "fusion_teacher_kd", "flops": current["flops"],
                          "latency_mean_ms": current["latency"]["mean_ms"], "teacher_training_only": True})
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_exit_per_seed.csv", per_exit)
    write_csv(out / "per_exit_multiseed_summary.csv", summary)
    write_csv(out / "selected_exit_per_seed.csv", selected)
    write_csv(out / "selected_exit_summary.csv", selected_summary)
    write_csv(out / "selected_exit_frequency.csv", exit_frequency)
    write_csv(out / "fusion_weights_per_seed.csv", weights)
    write_csv(out / "fusion_weights_summary.csv", weight_summary)
    write_csv(out / "fusion_teacher_gain.csv", teacher_gain)
    write_csv(out / "direction_aware_gains.csv", direction_gains)
    write_csv(out / "early_depth_comparisons.csv", early_depth)
    write_csv(out / "early_depth_comparison_summary.csv", early_summary)
    write_csv(out / "inference_cost.csv", costs)
    # Compact exported comparison plots for each method and task.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for method in METHODS:
            for task in TASKS:
                fig, ax = plt.subplots(figsize=(7, 4))
                for algorithm in ALGORITHMS:
                    rows = [r for r in summary if r["method"] == method and r["algorithm"] == algorithm and r["task"] == task]
                    ax.errorbar([r["exit"] for r in rows], [r["mean"] for r in rows],
                                yerr=[r["sample_sd"] for r in rows], marker="o", capsize=3, label=algorithm)
                ax.set(title=f"{method.upper()} — {task}", xlabel="Exit layer", ylabel="Test metric")
                ax.set_xticks(EXITS); ax.legend(); fig.tight_layout()
                fig.savefig(out / f"{method}_{task}_exit_performance.png", dpi=180)
                fig.savefig(out / f"{method}_{task}_exit_performance.pdf")
                plt.close(fig)
    except ImportError:
        raise RuntimeError("matplotlib is required for the EE study report")
    manifest = {"study": "two_stage_early_exit_optimization", "mask": "multi_block_n6",
                "methods": METHODS, "algorithms": ALGORITHMS, "tasks": TASKS,
                "head_seeds": SEEDS, "exit_layers": EXITS, "split_seed": 2025,
                "sources": provenance, "selection": "best_validation", "test_used_for_selection": False,
                "fusion_benchmark": "separately measured", "fusion_kd_cost": "Current EE batch-1 cost; teacher training only"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out / "README.md").write_text("# Two-stage early-exit optimization\n\n"
        "P0/P2 on `multi_block_n6`; seven head seeds; validation selects the reported exit and test evaluates it afterward. "
        "Classification gains are new minus current; positioning gains are current error minus new error. "
        "Fusion-Teacher KD deploys the single-representation student; its teacher is training-only.\n")
    print(json.dumps({"report": str(out), "seed_packets": len(provenance), "per_exit_rows": len(per_exit)}, indent=2))
    return 0


CHECKPOINTS = {"p0": ROOT / "runs/masking_head_seeds/checkpoint_bundle/checkpoints/multi_block_n6/P0/last.ckpt",
               "p2": ROOT / "runs/masking_head_seeds/checkpoint_bundle/checkpoints/multi_block_n6/P2/last.ckpt"}
CHECKPOINT_HASHES = {"p0": "8faa7245b6692e2d74d753fde2268fb712264c02c1da19632c045d022d192f1f",
                     "p2": "7e1e27d96183b8f9d1c0d6b7493b93945633ccfb1514555135b4d04f22bf18ac"}

if __name__ == "__main__":
    raise SystemExit(main())
