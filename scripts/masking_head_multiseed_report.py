#!/usr/bin/env python3
"""Build the seven-head-seed masking reports from raw metrics and benchmark packets.

This deliberately fails closed: no Markdown/CSV report is written unless every
method, task, exit, head seed, checkpoint, split, head weight, and benchmark is
present and traceable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from masking_head_study import (  # noqa: E402
    EXITS, MASKS, METHODS, SEEDS, TASKS, aggregate_method, derive_b0_from_p0,
)

from masking_head_figures import write_figures  # noqa: E402

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required raw artifact is missing: {path}")
    return json.loads(path.read_text())


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def load_packet(registry: dict[str, Any], entry: dict[str, Any], seed: int, run_root: Path):
    mask, method = entry["mask"], entry["method"]
    if seed == 0 and entry.get("seed0_reusable"):
        metrics_path = resolve(entry["seed0_metrics_path"])
        reused = True
    else:
        metrics_path = run_root / mask / f"head_seed_{seed:03d}" / method / "seed_000/downstream/exit_metrics.json"
        reused = False
    packet = read_json(metrics_path)
    checkpoint = resolve(entry["checkpoint_path"])
    if digest(checkpoint) != entry["checkpoint_sha256"]:
        raise ValueError(f"checkpoint checksum changed: {checkpoint}")
    packet_checkpoint = resolve(packet["checkpoint"])
    if digest(packet_checkpoint) != entry["checkpoint_sha256"]:
        raise ValueError(f"metrics/checkpoint identity mismatch: {metrics_path}")
    if packet.get("split_manifest") is None:
        raise ValueError(f"missing split manifest reference: {metrics_path}")
    split_path = resolve(packet["split_manifest"])
    split = read_json(split_path)
    if int(split.get("seed", -1)) != 2025:
        raise ValueError(f"wrong split seed in {split_path}")
    split_signature = digest(split_path)
    head_paths = []
    for task in TASKS:
        for layer in EXITS:
            head = resolve(packet["tasks"][task]["exits"][str(layer)]["head_path"])
            if not head.is_file():
                raise FileNotFoundError(f"missing trained head: {head}")
            head_paths.append({"path": str(head), "sha256": digest(head)})
    runtime_variant = "head_seed_000" if reused else f"head_seed_{seed:03d}"
    runtime_path = run_root / "runtime" / mask / method / runtime_variant / "runtime.json"
    if runtime_path.is_file():
        runtime = read_json(runtime_path)
        runtime_source = {"path": str(runtime_path), "sha256": digest(runtime_path)}
    elif reused:
        runtime = {"runtime_profile": "reused_seed0_profile_not_recorded"}
        runtime_source = None
    else:
        raise FileNotFoundError(f"missing runtime profile record: {runtime_path}")
    expected_profile = "source_faithful_p0" if method == "p0" else "performance_safe_hierarchical"
    if not reused and runtime.get("runtime_profile") != expected_profile:
        raise ValueError(f"runtime profile mismatch in {runtime_path}: expected {expected_profile}")
    checkpoint_source_note = entry.get("checkpoint_source_note")
    if mask == "multi_block_n6" and method == "p1":
        checkpoint_source_note = "M-block P1 reuses the frequency_n6 P1 checkpoint, as recorded in the checkpoint mapping manifest."
    validated = {
        "metrics": str(metrics_path), "metrics_sha256": digest(metrics_path),
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
        "checkpoint_source_note": checkpoint_source_note,
        "split_manifest": str(split_path), "split_sha256": split_signature,
        "head_weights": head_paths, "runtime": runtime,
        "runtime_artifact": runtime_source,
        "provenance": "reused_seed0" if reused else "newly_produced",
    }
    return packet, validated


def benchmark_packet(entry: dict[str, Any], run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    mask, method = entry["mask"], entry["method"]
    path = (
        resolve(entry["benchmark_source_path"])
        if entry.get("benchmark_reusable")
        else run_root / mask / "benchmarks" / method / "seed_000/benchmarks.json"
    )
    packet = read_json(path)
    if packet.get("method") != method or int(packet.get("pretrain_seed", -1)) != 0:
        raise ValueError(f"benchmark method/checkpoint seed mismatch: {path}")
    if int(packet.get("batch_size", -1)) != 1:
        raise ValueError(f"benchmark is not batch-1: {path}")
    if len(packet.get("entries", [])) != len(TASKS) * len(EXITS):
        raise ValueError(f"benchmark must contain six tasks x four exits: {path}")
    expected = {(task, layer) for task in TASKS for layer in EXITS}
    actual = {(row.get("task"), int(row.get("exit_layer", -1))) for row in packet["entries"]}
    if actual != expected:
        raise ValueError(f"benchmark task/exit coverage mismatch: {path}")
    cp = resolve(entry["checkpoint_path"])
    if digest(resolve(packet["checkpoint"])) != digest(cp):
        raise ValueError(f"benchmark checkpoint mismatch: {path}")
    runtime_path = run_root / "runtime" / mask / method / "benchmark/runtime.json"
    runtime = read_json(runtime_path) if runtime_path.is_file() else None
    if runtime is None and not entry.get("benchmark_reusable"):
        raise FileNotFoundError(f"missing benchmark runtime profile: {runtime_path}")
    expected_profile = "source_faithful_p0" if method == "p0" else "performance_safe_hierarchical"
    if runtime is not None and runtime.get("runtime_profile") != expected_profile:
        raise ValueError(f"benchmark runtime profile mismatch in {runtime_path}")
    return packet, {
        "path": str(path), "sha256": digest(path), "runtime": runtime,
        "runtime_artifact": (
            {"path": str(runtime_path), "sha256": digest(runtime_path)}
            if runtime_path.is_file() else None
        ),
        "provenance": "imported_reused" if entry.get("benchmark_reusable") else "newly_produced",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty report table: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = list(rows[0])
    table = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    table.extend("| " + " | ".join(str(row.get(key, "")) for key in columns) + " |" for row in rows)
    path.with_suffix(".md").write_text("\n".join(table) + "\n")


def _json_ready(value: Any) -> Any:
    """Normalize mapping keys before sorted JSON serialization.

    JSON object keys are strings; aggregations include integer exit/seed keys,
    and ``sort_keys=True`` cannot order those alongside string keys.
    """
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def build_condition(mask: str, registry: dict[str, Any], run_root: Path, output: Path) -> dict[str, Any]:
    entries = {(e["mask"], e["method"]): e for e in registry["entries"]}
    aggregates, sources, benchmarks = {}, {}, {}
    for method in METHODS:
        entry = entries[(mask, method)]
        packets, source_rows = {}, {}
        for seed in SEEDS:
            packets[seed], source_rows[seed] = load_packet(registry, entry, seed, run_root)
        split_identities = {source_rows[seed]["split_sha256"] for seed in SEEDS}
        if len(split_identities) != 1:
            raise ValueError(f"split identity mismatch across head seeds for {mask}/{method}")
        aggregates[method] = aggregate_method(packets, method=method)
        sources[method] = source_rows
        benchmarks[method], benchmark_source = benchmark_packet(entry, run_root)
        sources[method]["benchmark"] = benchmark_source
    aggregates["b0"] = derive_b0_from_p0(aggregates["p0"])

    fixed_rows, selected_rows, all_rows = [], [], []
    for method, aggregate in aggregates.items():
        for task in TASKS:
            row = aggregate["tasks"][task]
            for layer, summary in row.get("fixed", {}).items():
                values = summary["per_seed"]
                fixed_rows.append({
                    "method": method, "task": task, "metric": row["metric"],
                    "higher_is_better": row["higher_is_better"], "exit": layer,
                    "mean": summary["mean"], "sample_sd": summary["sd"],
                    "min": summary["min"], "max": summary["max"],
                    "per_seed": json.dumps(values, sort_keys=True),
                })
            selected = row["selected"]
            selected_rows.append({
                "method": method, "task": task, "metric": row["metric"],
                "higher_is_better": row["higher_is_better"],
                "mean": selected["mean"], "sample_sd": selected["sd"],
                "min": selected["min"], "max": selected["max"],
                "per_seed": json.dumps(selected["per_seed"], sort_keys=True),
                "selected_exit_by_seed": json.dumps(selected["selected_exit_by_seed"], sort_keys=True),
                "selection_frequency": json.dumps(selected["selection_frequency"], sort_keys=True),
            })
    for method in METHODS:
        benchmark = benchmarks[method]
        for row in benchmark["entries"]:
            all_rows.append({
                "method": method, "task": row["task"], "exit": row["exit_layer"],
                "flops": row["flops"], "latency_mean_ms": row["latency"]["mean_ms"],
                "latency_sd_ms": row["latency"]["standard_deviation_ms"],
                "checkpoint_sha256": sources[method][0]["checkpoint_sha256"],
                "benchmark_sha256": sources[method]["benchmark"]["sha256"],
                "derived_from": "",
            })
    for row in [row for row in all_rows if row["method"] == "p0" and int(row["exit"]) == 8]:
        all_rows.append({**row, "method": "b0", "derived_from": "p0:L8"})
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "multiseed_fixed_exits.csv", fixed_rows)
    write_csv(output / "multiseed_validation_selected.csv", selected_rows)
    write_csv(output / "multiseed_inference_cost.csv", all_rows)
    # Replace the established single-seed table filenames in place as well.
    write_csv(output / "table_primary_fixed_exits.csv", fixed_rows)
    write_csv(output / "table_selected_exits.csv", selected_rows)
    write_csv(output / "table_b0_context_l8_reference.csv", [r for r in fixed_rows if r["method"] == "b0"])
    write_csv(output / "table_inference_cost.csv", all_rows)
    write_csv(output / "table_all_methods_downstream.csv", selected_rows)
    write_csv(output / "table_all_methods_inference.csv", all_rows)
    figures = write_figures(output, mask, aggregates, benchmarks)
    condition_notes = []
    if mask in {"frequency_n6", "multi_block_n6"}:
        condition_notes.append(
            "frequency_n6 and multi_block_n6 are separate repeated historical conditions "
            "with the same nominal span-time geometry; they are not independent mask-family evidence."
        )
    condition_notes.extend(
        f"{method}: {sources[method][0]['checkpoint_source_note']}"
        for method in METHODS if sources[method][0].get("checkpoint_source_note")
    )
    lines = [
        f"# Seven-seed downstream masking report: {mask}", "",
        "Frozen pretrained checkpoints; downstream head seeds 0–6; fixed split seed 2025.",
        "Each aggregate is mean ± sample SD and includes per-seed values in the CSV tables.",
        "NR positioning error is lower-is-better. B0 is P0 at L8; P0 is reported at each exit.",
        "No backbone pretraining was run for this study.",
        "The imported P2-Reverse head_seed=2025 metrics are excluded from the head seeds 0–6 aggregate.", "",
        *condition_notes, "",
        "| Method | Task | Validation-selected test mean ± SD | Selected exits by seed |",
        "|---|---|---:|---|",
    ]
    for row in selected_rows:
        lines.append(
            f"| {row['method']} | {row['task']} | {row['mean']:.6g} ± {row['sample_sd']:.4g} "
            f"| {row['selected_exit_by_seed']} |"
        )
    lines += ["", "CSV/Markdown tables and the existing figure panels regenerated as PNG and PDF are adjacent.",
              "Artifact hashes and source provenance are recorded in manifest.json.", ""]
    (output / "README.md").write_text("\n".join(lines))
    manifest = {
        "report": "seven_seed_downstream_head_masking_study",
        "mask": mask, "head_seeds": list(SEEDS), "split_seed": 2025,
        "pretrain_seed": 0, "tasks": list(TASKS), "exits": list(EXITS),
        "methods": list(METHODS), "aggregates": aggregates, "sources": sources,
        "benchmark_artifacts": {m: sources[m]["benchmark"] for m in METHODS},
        "generated_figures": figures,
        "generated_tables": [
            "multiseed_fixed_exits.csv", "multiseed_fixed_exits.md",
            "multiseed_validation_selected.csv", "multiseed_validation_selected.md",
            "multiseed_inference_cost.csv", "multiseed_inference_cost.md",
            "table_primary_fixed_exits.csv", "table_primary_fixed_exits.md",
            "table_selected_exits.csv", "table_selected_exits.md",
            "table_b0_context_l8_reference.csv", "table_b0_context_l8_reference.md",
            "table_inference_cost.csv", "table_inference_cost.md",
            "table_all_methods_downstream.csv", "table_all_methods_downstream.md",
            "table_all_methods_inference.csv", "table_all_methods_inference.md",
        ],
        "provenance_note": "Seed-0 P2-Reverse packets with head_seed=2025 are excluded.",
    }
    (output / "manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/masking_head_seeds")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports/masking_study")
    args = parser.parse_args()
    registry_path = args.registry or args.run_root / "registry.json"
    registry = read_json(registry_path)
    if not registry.get("checkpoint_gate_passed"):
        raise RuntimeError("artifact registry checkpoint gate has not passed")
    validate_all_conditions(registry, args.run_root)
    for mask in MASKS:
        build_condition(mask, registry, args.run_root, args.output_root / mask)
    print(json.dumps({"updated_reports": [str(args.output_root / mask) for mask in MASKS]}, indent=2))
    return 0


def validate_all_conditions(registry: dict[str, Any], run_root: Path) -> None:
    """Validate all required sources before overwriting any condition report."""
    import matplotlib  # noqa: F401
    entries = {(entry["mask"], entry["method"]): entry for entry in registry["entries"]}
    for mask in MASKS:
        for method in METHODS:
            entry = entries[(mask, method)]
            packets, split_identities = {}, set()
            for seed in SEEDS:
                packet, source = load_packet(registry, entry, seed, run_root)
                packets[seed] = packet
                split_identities.add(source["split_sha256"])
            if len(split_identities) != 1:
                raise ValueError(f"split identity mismatch across head seeds for {mask}/{method}")
            aggregate_method(packets, method=method)
            benchmark_packet(entry, run_root)


if __name__ == "__main__":
    raise SystemExit(main())
