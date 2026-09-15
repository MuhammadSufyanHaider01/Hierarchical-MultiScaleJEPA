"""Phase 6 held-out OOD dataset registry, audit, and split helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from datasets import ImageDataset
from phase5 import TaskSpec, split_indices


OOD_SPLIT_SCHEMA_VERSION = 1
OOD_SPLIT_FRACTIONS = {"train": 0.80, "validation": 0.10, "test": 0.10}
OOD_TASKS = ("jamming", "interference")

_OOD_FILES = {
    "jamming": {
        "filename": "jamming.h5",
        "sample_key": "normalized_spectrograms",
        "label_key": "labels",
    },
    "interference": {
        "filename": "interference.h5",
        "sample_key": "cls_spectrograms",
        "label_key": "cls_labels",
    },
}


def default_ood_root(repository_root: Path) -> Path:
    """Return the existing local parent directory holding non-primary HDF5 files."""

    return Path(repository_root).resolve().parent


def _entry(task: str) -> dict[str, str]:
    try:
        return _OOD_FILES[task]
    except KeyError as error:
        raise ValueError(f"unsupported Phase 6 OOD task: {task}") from error


class _RemappedLabelDataset(Dataset):
    """Map non-zero-based source labels in memory without changing an HDF5 file."""

    def __init__(self, dataset: Dataset, mapping: dict[int, int]):
        self.dataset = dataset
        self.mapping = mapping
        self.sample_shape = dataset.sample_shape

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample, label = self.dataset[index]
        return sample, torch.tensor(self.mapping[int(label.item())], dtype=torch.long)


def _labels_and_mapping(path: Path, label_key: str) -> tuple[list[int], int, dict[int, int]]:
    with h5py.File(path, "r") as h5:
        labels = np.asarray(h5[label_key][:]).reshape(-1)
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"{path}:{label_key} must be integer labels, got {labels.dtype}")
    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError(f"{path}:{label_key} must contain at least two classes")
    mapping = {int(value): index for index, value in enumerate(classes.tolist())}
    return [mapping[int(value)] for value in labels.tolist()], int(classes.size), mapping


def build_ood_task_dataset(task: str, ood_root: Path) -> tuple[Dataset, TaskSpec]:
    entry = _entry(task)
    path = Path(ood_root) / entry["filename"]
    if not path.is_file():
        raise FileNotFoundError(path)
    labels, classes, mapping = _labels_and_mapping(path, entry["label_key"])
    dataset = ImageDataset(path, sample_key=entry["sample_key"], label_key=entry["label_key"])
    if any(source != target for source, target in mapping.items()):
        dataset = _RemappedLabelDataset(dataset, mapping)
    if len(labels) != len(dataset):
        raise ValueError(f"{path} has mismatched sample and label counts")
    return dataset, TaskSpec(
        name=task,
        path=str(path),
        target_type="classification",
        num_outputs=classes,
        higher_is_better=True,
        metric_name="macro_accuracy",
        sample_shape=dataset.sample_shape,
    )


def _finite_status(dataset: h5py.Dataset, *, scan_finite: bool) -> dict[str, object]:
    if not np.issubdtype(dataset.dtype, np.inexact):
        return {"checked": False, "finite": None, "mode": "not_applicable"}
    if scan_finite:
        for start in range(0, dataset.shape[0], 128):
            if not np.isfinite(dataset[start:start + 128]).all():
                return {"checked": True, "finite": False, "mode": "full"}
        return {"checked": True, "finite": True, "mode": "full"}
    indices = sorted({0, max(0, dataset.shape[0] - 1)})
    return {
        "checked": True,
        "finite": bool(np.isfinite(dataset[indices]).all()),
        "mode": "first_last_sample",
    }


def audit_ood_datasets(ood_root: Path, *, scan_finite: bool = False) -> dict[str, object]:
    """Read-only schema audit for all approved held-out OOD files."""

    root = Path(ood_root).resolve()
    tasks: dict[str, object] = {}
    for task in OOD_TASKS:
        entry = _entry(task)
        path = root / entry["filename"]
        record: dict[str, object] = {
            "path": str(path),
            "exists": path.is_file(),
            "sample_key": entry["sample_key"],
            "label_key": entry["label_key"],
            "errors": [],
        }
        if not path.is_file():
            record["errors"].append("missing file")
            record["valid"] = False
            tasks[task] = record
            continue
        with h5py.File(path, "r") as h5:
            if entry["sample_key"] not in h5 or entry["label_key"] not in h5:
                record["errors"].append("required sample or label key is absent")
            else:
                samples = h5[entry["sample_key"]]
                labels = h5[entry["label_key"]]
                record["sample"] = {
                    "shape": list(samples.shape),
                    "dtype": str(samples.dtype),
                    "finite": _finite_status(samples, scan_finite=scan_finite),
                }
                record["labels"] = {"shape": list(labels.shape), "dtype": str(labels.dtype)}
                if samples.ndim != 4 or samples.shape[0] != labels.shape[0]:
                    record["errors"].append("samples must be [N,C,H,W] with matching label count")
        if not record["errors"]:
            try:
                labels, class_count, mapping = _labels_and_mapping(path, entry["label_key"])
                record["count"] = len(labels)
                record["class_count"] = class_count
                record["label_mapping"] = {str(source): target for source, target in mapping.items()}
            except (TypeError, ValueError) as error:
                record["errors"].append(str(error))
        record["valid"] = not record["errors"]
        tasks[task] = record
    return {
        "schema_version": OOD_SPLIT_SCHEMA_VERSION,
        "ood_root": str(root),
        "finite_scan": scan_finite,
        "tasks": tasks,
        "valid": all(record["valid"] for record in tasks.values()),
    }


def create_ood_split_manifest(ood_root: Path, *, seed: int) -> dict[str, object]:
    root = Path(ood_root).resolve()
    tasks: dict[str, object] = {}
    for task in OOD_TASKS:
        dataset, spec = build_ood_task_dataset(task, root)
        labels, _, mapping = _labels_and_mapping(Path(spec.path), _entry(task)["label_key"])
        split = split_indices(len(dataset), seed=seed, labels=labels)
        tasks[task] = {
            "spec": asdict(spec),
            "label_mapping": {str(source): target for source, target in mapping.items()},
            "counts": {name: len(indices) for name, indices in split.items()},
            "indices": split,
        }
    return {
        "schema_version": OOD_SPLIT_SCHEMA_VERSION,
        "seed": int(seed),
        "fractions": OOD_SPLIT_FRACTIONS,
        "ood_root": str(root),
        "tasks": tasks,
    }


def load_or_create_ood_split_manifest(path: Path, ood_root: Path, *, seed: int) -> dict[str, object]:
    path = Path(path)
    root = Path(ood_root).resolve()
    if path.is_file():
        manifest = json.loads(path.read_text())
        if manifest.get("schema_version") != OOD_SPLIT_SCHEMA_VERSION:
            raise ValueError(f"unsupported OOD split manifest schema: {manifest.get('schema_version')}")
        if int(manifest.get("seed", -1)) != int(seed):
            raise ValueError(f"OOD split manifest seed {manifest.get('seed')} does not match requested seed {seed}")
        if manifest.get("fractions") != OOD_SPLIT_FRACTIONS:
            raise ValueError("OOD split manifest fractions do not match the 80/10/10 protocol")
        if Path(manifest.get("ood_root", "")).resolve() != root:
            raise ValueError("OOD split manifest root does not match the requested OOD root")
        return manifest
    manifest = create_ood_split_manifest(root, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def ood_task_subsets(dataset: Dataset, split_record: dict[str, object]) -> dict[str, Subset]:
    indices = split_record["indices"]
    return {name: Subset(dataset, [int(index) for index in values]) for name, values in indices.items()}
