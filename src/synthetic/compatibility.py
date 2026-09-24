"""Read-only audit for frozen P2 heads and synthetic-task semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .schema import load_task_profile


TASK_CONFIGS = {
    "nrpos-outdoor": "factory_ofdma_nrpos_paper_v3.yml",
    "interference": "factory_ofdma_interference_paper_v3.yml",
    "jamming": "factory_ofdma_jamming_paper_v3.yml",
}
EXIT_LAYERS = (2, 4, 6, 8)
DEFAULT_SOURCE_METRICS = (
    "runs/masking_study/time_n6/checkpoint_epoch029_six_tasks/"
    "p2/seed_000/downstream/exit_metrics.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_repository_root(source_path: Path) -> Path:
    for parent in source_path.resolve().parents:
        if parent.name == "runs":
            return parent.parent
    raise ValueError(f"cannot infer repository root from source metrics path {source_path}")


def source_file(path_value: str, source_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else source_root / path


def _file_record(path: Path, *, digest: bool = False) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }
    if digest:
        result["sha256"] = sha256_file(path)
    return result


def _label_counts(path: Path, label_key: str) -> dict[str, int]:
    with h5py.File(path, "r") as h5:
        values = np.asarray(h5[label_key][:]).reshape(-1)
    unique, counts = np.unique(values, return_counts=True)
    return {str(value.item() if hasattr(value, "item") else value): int(count) for value, count in zip(unique, counts)}


def _json_attribute(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _source_task(source: dict[str, Any], task: str) -> dict[str, Any]:
    record = source.get("tasks", {}).get(task)
    if not isinstance(record, dict):
        raise ValueError(f"P2 source metrics do not contain task {task}")
    return record


def audit_synthetic_compatibility(
    source_metrics_path: str | Path,
    *,
    repository_root: str | Path,
    config_root: str | Path | None = None,
    tasks: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    metrics_path = Path(source_metrics_path)
    if not metrics_path.is_absolute():
        metrics_path = root / metrics_path
    source = json.loads(metrics_path.read_text())
    if source.get("method") != "p2":
        raise ValueError(f"synthetic availability requires P2 metrics, got {source.get('method')!r}")
    tolerance = float(source.get("protocol", {}).get("tolerance", -1))
    if not np.isclose(tolerance, 0.05, rtol=0.0, atol=1e-12):
        raise ValueError(f"source P2 artifact must use tolerance 0.05, got {tolerance}")
    source_root = source_repository_root(metrics_path)
    checkpoint_path = source_file(str(source.get("checkpoint", "")), source_root)
    manifest_path = source_file(str(source.get("split_manifest", "")), source_root)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"P2 checkpoint is missing: {checkpoint_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"P2 source split manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 3:
        raise ValueError(f"expected six-task v3 P2 split manifest, got {manifest.get('schema_version')!r}")

    configs = Path(config_root) if config_root else root / "configs" / "sionna"
    task_reports: dict[str, Any] = {}
    requested_tasks = tuple(TASK_CONFIGS) if tasks is None else tuple(tasks)
    unknown = set(requested_tasks) - set(TASK_CONFIGS)
    if unknown:
        raise ValueError(f"unknown synthetic audit task(s): {sorted(unknown)}")
    for task in requested_tasks:
        config_name = TASK_CONFIGS[task]
        profile, config = load_task_profile(configs / config_name, task_name=task)
        task_record = _source_task(source, task)
        source_spec = task_record.get("spec", {})
        mismatches = []
        for field in ("target_type", "num_outputs", "metric_name", "higher_is_better"):
            if source_spec.get(field) != getattr(profile, field):
                mismatches.append(
                    f"source head spec {field}={source_spec.get(field)!r} "
                    f"does not match profile {getattr(profile, field)!r}"
                )
        if mismatches:
            raise ValueError(f"incompatible P2 task specification for {task}: {'; '.join(mismatches)}")
        exits = task_record.get("exits", {})
        head_records: dict[str, object] = {}
        for layer in EXIT_LAYERS:
            exit_record = exits.get(str(layer))
            if not isinstance(exit_record, dict):
                raise ValueError(f"P2 source is missing {task} L{layer} exit record")
            head_path = source_file(str(exit_record.get("head_path", "")), source_root)
            if not head_path.is_file():
                raise FileNotFoundError(f"P2 source head is missing: {head_path}")
            head_records[str(layer)] = _file_record(head_path)

        manifest_task = manifest.get("tasks", {}).get(task, {})
        if not isinstance(manifest_task.get("indices"), dict):
            raise ValueError(f"P2 v3 source manifest lacks task indices for {task}")

        task_config = config.get("task") or config["tasks"][task]
        source_data_path = Path(task_config["source_path"])
        if not source_data_path.is_absolute():
            source_data_path = root / source_data_path
        blockers = profile.semantic_blockers()
        if not source_data_path.is_file():
            raise FileNotFoundError(f"local source HDF5 is missing: {source_data_path}")
        with h5py.File(source_data_path, "r") as h5:
            if profile.sample_key not in h5 or profile.label_key not in h5:
                raise KeyError(
                    f"{task} cache must contain {profile.sample_key!r} and {profile.label_key!r}"
                )
            sample = h5[profile.sample_key]
            labels = np.asarray(h5[profile.label_key][:])
            attrs = {str(k): _json_attribute(v) for k, v in h5.attrs.items()}
            sample_shape = list(sample.shape[1:])
            label_shape = list(h5[profile.label_key].shape)
            unique, counts = np.unique(labels.reshape(-1), return_counts=True)
            counts_record = {
                str(v.item() if hasattr(v, "item") else v): int(n)
                for v, n in zip(unique, counts)
            }
            source_map = None
            if "label_to_index" in h5.attrs:
                label_to_index = _json_attribute(h5.attrs["label_to_index"])
                if isinstance(label_to_index, dict):
                    source_map = {int(index): str(name) for name, index in label_to_index.items()}
        if task == "jamming" and source_map is not None and source_map != profile.label_map:
            blockers.append(f"configured class map differs from HDF5 label_to_index: {source_map}")
        task_splits = manifest_task["indices"]
        task_reports[task] = {
            "status": (
                "blocked" if blockers else
                "ready_assumption_based" if profile.specification_status == "assumption_based_synthetic_domain" else
                "ready"
            ),
            "semantic_basis": profile.specification_status,
            "assumptions_approved": profile.assumptions_approved,
            "assumptions": list(profile.assumptions),
            "blockers": blockers,
            "profile": config_name,
            "source_hdf5": _file_record(source_data_path),
            "sample_shape": sample_shape,
            "label_shape": label_shape,
            "label_counts": counts_record,
            "hdf5_attributes": attrs,
            "source_class_map": source_map,
            "p2_task_spec": source_spec,
            "p2_heads": head_records,
            "source_split_counts": {
                name: len(indices) for name, indices in task_splits.items() if isinstance(indices, list)
            },
        }

    return {
        "schema_version": 1,
        "study": "sionna_synthetic_availability",
        "source_method": "p2",
        "source_metrics": _file_record(metrics_path, digest=True),
        "source_checkpoint": _file_record(checkpoint_path),
        "source_split_manifest": _file_record(manifest_path, digest=True),
        "source_split_schema_version": manifest.get("schema_version"),
        "source_repository_root": str(source_root),
        "source_tolerance": tolerance,
        "tasks": task_reports,
    }
