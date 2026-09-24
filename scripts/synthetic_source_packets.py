"""Read-only provenance checks for the frozen P0/P2 synthetic-study sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKETS = {
    "p0": ROOT / "runs/masking_study/time_n6/p0/seed_000/downstream/exit_metrics.json",
    "p2": ROOT / "runs/masking_study/time_n6/checkpoint_epoch029_six_tasks/p2/seed_000/downstream/exit_metrics.json",
}
EXIT_LAYERS = (2, 4, 6, 8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_source_packets(task: str, profile) -> dict[str, dict[str, object]]:
    """Require the matching P0/P2 checkpoint, split manifest, and four frozen heads."""
    records: dict[str, dict[str, object]] = {}
    for method, metrics_path in SOURCE_PACKETS.items():
        if not metrics_path.is_file():
            raise FileNotFoundError(f"required {method.upper()} source packet is missing: {metrics_path}")
        data = json.loads(metrics_path.read_text())
        if data.get("method") != method:
            raise ValueError(f"{metrics_path} identifies method={data.get('method')!r}, expected {method!r}")
        task_record = data.get("tasks", {}).get(task)
        if not isinstance(task_record, dict):
            raise ValueError(f"{method.upper()} source packet lacks task {task}")
        spec = task_record.get("spec", {})
        if spec.get("target_type") != profile.target_type or int(spec.get("num_outputs", -1)) != profile.num_outputs:
            raise ValueError(f"factory {task} output contract does not match the {method.upper()} frozen head")

        def resolve(value: str) -> Path:
            path = Path(value)
            return path if path.is_absolute() else ROOT / path

        checkpoint = resolve(str(data.get("checkpoint", "")))
        split = resolve(str(data.get("split_manifest", "")))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{method.upper()} checkpoint is missing: {checkpoint}")
        if not split.is_file():
            raise FileNotFoundError(f"{method.upper()} source split manifest is missing: {split}")
        heads: dict[str, dict[str, str]] = {}
        for layer in EXIT_LAYERS:
            head = resolve(str(task_record.get("exits", {}).get(str(layer), {}).get("head_path", "")))
            if not head.is_file():
                raise FileNotFoundError(f"{method.upper()} {task} L{layer} head is missing: {head}")
            heads[str(layer)] = {"path": str(head.resolve()), "sha256": sha256(head)}
        records[method] = {
            "metrics": {"path": str(metrics_path.resolve()), "sha256": sha256(metrics_path)},
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)},
            "split_manifest": {"path": str(split.resolve()), "sha256": sha256(split)},
            "heads": heads,
        }
    return records
