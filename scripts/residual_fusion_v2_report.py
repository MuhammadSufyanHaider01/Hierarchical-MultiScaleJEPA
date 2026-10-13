#!/usr/bin/env python3
"""Build the fail-closed single-seed comparison of Current, Fusion V1, and V2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("rfs", "jamming", "human-activity", "interference", "nrpos-outdoor", "deepmimo-beam")
EXITS = (2, 4, 6, 8)
METHODS = ("p0", "p2")
CHECKPOINT_HASHES = {"p0": "8faa7245b6692e2d74d753fde2268fb712264c02c1da19632c045d022d192f1f",
                     "p2": "7e1e27d96183b8f9d1c0d6b7493b93945633ccfb1514555135b4d04f22bf18ac"}
BASE = ROOT / "runs/masking_head_seeds/multi_block_n6/head_seed_000"
RUNS = ROOT / "runs/ee_optimization/multi_block_n6"
OUT = ROOT / "reports/early_exit_optimization/multi_block_n6/residual_dynamic_fusion_single_seed"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_packet(method: str, algorithm: str) -> tuple[Path, dict]:
    if algorithm == "current":
        path = BASE / method / "seed_000/downstream/exit_metrics.json"
    elif algorithm == "causal_fusion":
        path = RUNS / "fusion/head_seed_000" / method / "seed_000/downstream/exit_metrics.json"
    else:
        path = RUNS / "residual_dynamic_fusion/head_seed_000" / method / "seed_000/downstream/exit_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"required {algorithm} packet missing: {path}")
    packet = json.loads(path.read_text())
    if packet.get("method") != method or int(packet.get("head_seed", -1)) != 0 or int(packet.get("pretrain_seed", -1)) != 0:
        raise ValueError(f"method/seed provenance mismatch: {path}")
    if algorithm != "current" and packet.get("head_algorithm") != algorithm:
        raise ValueError(f"algorithm identity mismatch: {path}")
    if packet.get("protocol", {}).get("test_evaluation") is not True or set(packet.get("tasks", {})) != set(TASKS):
        raise ValueError(f"test/task coverage incomplete: {path}")
    checkpoint = Path(packet.get("checkpoint", ""))
    if not checkpoint.is_file() or sha(checkpoint) != CHECKPOINT_HASHES[method]:
        raise ValueError(f"fixed checkpoint identity invalid: {path}")
    if packet.get("checkpoint_sha256") not in (None, CHECKPOINT_HASHES[method]):
        raise ValueError(f"checkpoint hash mismatch: {path}")
    split = Path(packet.get("split_manifest", ""))
    if not split.is_file() or int(json.loads(split.read_text()).get("seed", -1)) != 2025:
        raise ValueError(f"locked split missing or invalid: {path}")
    if packet.get("split_manifest_sha256") not in (None, sha(split)):
        raise ValueError(f"split identity mismatch: {path}")
    packet["_validated_split_sha256"] = sha(split)
    for task in TASKS:
        row = packet["tasks"][task]
        if set(row.get("exits", {})) != {str(x) for x in EXITS}:
            raise ValueError(f"exit coverage incomplete: {path}:{task}")
        selected = int(row.get("selected_exit_best_validation", -1))
        if selected not in EXITS:
            raise ValueError(f"validation-selected exit absent: {path}:{task}")
        metric = row.get("spec", {}).get("metric_name")
        for layer in EXITS:
            e = row["exits"][str(layer)]
            if metric not in e.get("validation", {}) or metric not in e.get("test", {}):
                raise ValueError(f"validation/test metric absent: {path}:{task}:L{layer}")
            head = Path(e.get("head_path", ""))
            if not head.is_file():
                raise FileNotFoundError(f"trained head missing: {head}")
            if algorithm != "current" and e.get("checkpoint_sha256") not in (None, sha(head)):
                raise ValueError(f"head hash mismatch: {path}:{task}:L{layer}")
            if algorithm == "residual_projected_dynamic_fusion" and set(e.get("fusion_diagnostics", {})) != {str(x) for x in EXITS if x < layer}:
                raise ValueError(f"V2 diagnostics incomplete: {path}:{task}:L{layer}")
    return path, packet


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    sources = {}
    split_ids = set()
    for method in METHODS:
        for algorithm in ("current", "causal_fusion", "residual_projected_dynamic_fusion"):
            path, packet = load_packet(method, algorithm)
            sources[method, algorithm] = (path, packet)
            split_ids.add(packet["_validated_split_sha256"])
    if len(split_ids) != 1:
        raise ValueError(f"comparison split identities differ: {sorted(split_ids)}")

    fixed, selected, gains, gates = [], [], [], []
    provenance = []
    for method in METHODS:
        for algorithm in ("current", "causal_fusion", "residual_projected_dynamic_fusion"):
            path, packet = sources[method, algorithm]
            provenance.append({"method": method, "algorithm": algorithm, "metrics_path": str(path),
                "metrics_sha256": sha(path), "checkpoint_sha256": packet["checkpoint_sha256"],
                "split_manifest": packet["split_manifest"], "split_manifest_sha256": packet["_validated_split_sha256"]})
            for task in TASKS:
                row = packet["tasks"][task]
                metric = row["spec"]["metric_name"]
                higher = bool(row["spec"]["higher_is_better"])
                exit_id = int(row["selected_exit_best_validation"])
                chosen = row["exits"][str(exit_id)]
                selected.append({"method": method, "algorithm": algorithm, "task": task, "metric": metric,
                    "selected_exit": exit_id, "validation_value": chosen["validation"][metric],
                    "selected_test_value": chosen["test"][metric]})
                for layer in EXITS:
                    e = row["exits"][str(layer)]
                    fixed.append({"method": method, "algorithm": algorithm, "task": task, "exit": layer,
                        "metric": metric, "higher_is_better": higher, "validation_value": e["validation"][metric],
                        "test_value": e["test"][metric], "selected_by_validation": layer == exit_id})
                    if algorithm == "residual_projected_dynamic_fusion":
                        for source, stats in e["fusion_diagnostics"].items():
                            gates.append({"method": method, "task": task, "target_exit": layer,
                                "source_exit": int(source), **stats})
                if algorithm == "residual_projected_dynamic_fusion":
                    for baseline in ("current", "causal_fusion"):
                        b = sources[method, baseline][1]["tasks"][task]
                        bexit = int(b["selected_exit_best_validation"])
                        bmetric = b["spec"]["metric_name"]
                        bvalue = float(b["exits"][str(bexit)]["test"][bmetric])
                        vvalue = float(chosen["test"][metric])
                        gains.append({"method": method, "task": task, "comparison": f"selected_vs_{baseline}",
                            "v2_exit": exit_id, "reference_exit": bexit, "metric": metric,
                            "positive_is_improvement": True,
                            "direction_aware_gain": vvalue - bvalue if higher else bvalue - vvalue})
                    for layer in EXITS:
                        for baseline in ("current", "causal_fusion"):
                            b = sources[method, baseline][1]["tasks"][task]
                            vvalue = float(row["exits"][str(layer)]["test"][metric])
                            bvalue = float(b["exits"][str(layer)]["test"][metric])
                            gains.append({"method": method, "task": task, "comparison": f"fixed_L{layer}_vs_{baseline}",
                                "v2_exit": layer, "reference_exit": layer, "metric": metric,
                                "positive_is_improvement": True,
                                "direction_aware_gain": vvalue - bvalue if higher else bvalue - vvalue})
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "fixed_exit_results.csv", fixed)
    write_csv(out / "validation_selected_results.csv", selected)
    write_csv(out / "direction_aware_gains.csv", gains)
    write_csv(out / "v2_gate_diagnostics.csv", gates)
    manifest = {"study": "residual_dynamic_fusion_single_seed", "methods": METHODS,
        "algorithms": ("current", "causal_fusion", "residual_projected_dynamic_fusion"),
        "head_seed": 0, "pretrain_seed": 0, "split_seed": 2025, "tasks": TASKS,
        "exits": EXITS, "selection": "best_validation", "test_used_for_selection": False,
        "sources": provenance}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out / "README.md").write_text("# Residual Dynamic Fusion V2 — single-seed pilot\n\n"
        "This report compares Current EE, causal Fusion V1, and Residual Projected Dynamic Fusion V2 for P0/P2, head seed 0. "
        "All headline selected-exit rows use validation-only selection; the held-out test split is used only for evaluation. "
        "Direction-aware gains are positive when V2 improves the task metric. The single seed is a pilot, not an estimate of across-seed variability.\n")
    print(json.dumps({"output": str(out), "fixed_exit_rows": len(fixed), "selected_rows": len(selected), "gain_rows": len(gains)}, indent=2))


if __name__ == "__main__":
    main()
