#!/usr/bin/env python3
"""Read-only audit of paired P0/P2 sources for synthetic availability studies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("nrpos-outdoor", "interference", "jamming")
LAYERS = (2, 4, 6, 8)
DEFAULTS = {
    "p0": "runs/masking_study/time_n6/p0/seed_000/downstream/exit_metrics.json",
    "p2": "runs/masking_study/time_n6/checkpoint_epoch029_six_tasks/p2/seed_000/downstream/exit_metrics.json",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def source_root(metrics_path: Path) -> Path:
    for parent in metrics_path.resolve().parents:
        if parent.name == "runs":
            return parent.parent
    raise ValueError(f"cannot infer repository root from metrics path: {metrics_path}")


def resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def inspect_source(path: Path, expected_method: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path if path.is_absolute() else ROOT / path
    data = json.loads(path.read_text())
    if data.get("method") != expected_method:
        raise ValueError(f"{path} method is {data.get('method')!r}, expected {expected_method!r}")
    if float(data.get("protocol", {}).get("tolerance", -1)) != 0.05:
        raise ValueError(f"{path} does not use the required 5% exit-selection protocol")
    repository = source_root(path)
    checkpoint = resolve(str(data.get("checkpoint", "")), repository)
    manifest = resolve(str(data.get("split_manifest", "")), repository)
    if not checkpoint.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"{expected_method} checkpoint or split manifest is missing")
    split_data = json.loads(manifest.read_text())
    if split_data.get("schema_version") != 3 or not isinstance(split_data.get("tasks"), dict):
        raise ValueError(f"{expected_method} source manifest must be a task-indexed v3 split")
    source = {
        "method": expected_method,
        "metrics": {"path": str(path.resolve()), "sha256": digest(path)},
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": digest(checkpoint)},
        "split_manifest": {"path": str(manifest.resolve()), "sha256": digest(manifest)},
    }
    return {"metrics": data, "split": split_data}, source


def audit(p0_path: Path, p2_path: Path) -> dict[str, Any]:
    (p0, p0_record) = inspect_source(p0_path, "p0")
    (p2, p2_record) = inspect_source(p2_path, "p2")
    results: dict[str, Any] = {}
    for task in TASKS:
        records = [p0["metrics"].get("tasks", {}).get(task), p2["metrics"].get("tasks", {}).get(task)]
        if any(not isinstance(record, dict) for record in records):
            raise ValueError(f"both source packets must contain task {task}")
        if records[0].get("spec") != records[1].get("spec"):
            raise ValueError(f"P0/P2 task specifications differ for {task}")
        for source_record, task_record in zip((p0, p2), records):
            for layer in LAYERS:
                exit_record = task_record.get("exits", {}).get(str(layer))
                if not isinstance(exit_record, dict):
                    raise ValueError(f"source is missing {task} L{layer} exit")
                head = resolve(str(exit_record.get("head_path", "")), source_root(Path(
                    p0_record["metrics"]["path"] if source_record is p0 else p2_record["metrics"]["path"]
                )))
                if not head.is_file():
                    raise FileNotFoundError(f"missing frozen head: {head}")
        split_records = [source["split"].get("tasks", {}).get(task, {}) for source in (p0, p2)]
        indices = [record.get("indices") for record in split_records]
        if any(not isinstance(value, dict) for value in indices):
            raise ValueError(f"P0/P2 split manifests lack {task} source indices")
        if indices[0] != indices[1]:
            raise ValueError(f"P0/P2 original-data split indices differ for {task}")
        results[task] = {
            "task_spec": records[0]["spec"],
            "frozen_layers": list(LAYERS),
            "source_split_counts": {name: len(rows) for name, rows in indices[0].items()},
            "source_train_validation_test_ids_match": True,
        }
    return {
        "schema_version": 1,
        "study": "paired_synthetic_availability_sources",
        "status": "ready_for_synthetic_adapter_and_simulator_gates",
        "source_packets": {"p0": p0_record, "p2": p2_record},
        "tasks": results,
        "invariants": {
            "method_weights_are_read_only": True,
            "training_required": False,
            "exit_selection_tolerance": 0.05,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-metrics", type=Path, default=Path(DEFAULTS["p0"]))
    parser.add_argument("--p2-metrics", type=Path, default=Path(DEFAULTS["p2"]))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.p0_metrics, args.p2_metrics)
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        target = args.output if args.output.is_absolute() else ROOT / args.output
        if target.exists():
            raise FileExistsError(f"refusing to overwrite audit packet: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output)
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
