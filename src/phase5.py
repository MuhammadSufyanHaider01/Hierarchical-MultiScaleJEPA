"""Shared Phase 5 downstream-task and fixed-split utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from datasets import DeepMIMO, ImageDataset


SPLIT_SCHEMA_VERSION = 3
SPLIT_FRACTIONS = {"train": 0.80, "validation": 0.10, "test": 0.10}
PRIMARY_TASKS = (
    "rfs",
    "jamming",
    "human-activity",
    "interference",
    "nrpos-outdoor",
    "deepmimo-beam",
)


def _path_record(path: Path, root: Path) -> str:
    """Use a repository-relative path when possible, otherwise preserve it."""

    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    path: str
    target_type: str
    num_outputs: int
    higher_is_better: bool
    metric_name: str
    sample_shape: tuple[int, ...]
    coord_min: list[float] | None = None
    coord_max: list[float] | None = None


class _RemappedLabelDataset(Dataset):
    """Map source classification labels to contiguous IDs without editing HDF5."""

    def __init__(self, dataset: ImageDataset, mapping: dict[int, int]):
        self.dataset = dataset
        self.mapping = mapping
        self.sample_shape = dataset.sample_shape

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample, label = self.dataset[index]
        return sample, torch.tensor(self.mapping[int(label.item())], dtype=torch.long)


def _classification_dataset(
    path: Path,
    *,
    sample_key: str,
    label_key: str,
) -> tuple[Dataset, int, dict[int, int]]:
    """Load a classification cache and make labels contiguous in memory."""

    with h5py.File(path, "r") as h5:
        labels = np.asarray(h5[label_key][:]).reshape(-1)
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"{path}:{label_key} must contain integer labels, got {labels.dtype}")
    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError(f"{path}:{label_key} must contain at least two classes")
    mapping = {int(value): index for index, value in enumerate(classes.tolist())}
    dataset: Dataset = ImageDataset(path, sample_key=sample_key, label_key=label_key)
    if any(source != target for source, target in mapping.items()):
        dataset = _RemappedLabelDataset(dataset, mapping)
    if len(dataset) != len(labels):
        raise ValueError(f"{path} has mismatched sample and label counts")
    return dataset, int(classes.size), mapping


def build_task_dataset(task: str, repository_root: Path) -> tuple[Dataset, TaskSpec]:
    root = Path(repository_root)
    if task == "rfs":
        path = root / "data/downstream/rfs.h5"
        dataset = ImageDataset(path, sample_key="image", label_key="label")
        return dataset, TaskSpec(
            name=task,
            path=str(path.relative_to(root)),
            target_type="classification",
            num_outputs=20,
            higher_is_better=True,
            metric_name="macro_accuracy",
            sample_shape=dataset.sample_shape,
        )
    if task == "nrpos-outdoor":
        path = root / "data/downstream/nrpos-outdoor.h5"
        dataset = ImageDataset(path, sample_key="features", label_key="label", label_dtype=torch.float32)
        with h5py.File(path, "r") as h5:
            coord_min = json.loads(h5.attrs["coord_nominal_min"])
            coord_max = json.loads(h5.attrs["coord_nominal_max"])
        return dataset, TaskSpec(
            name=task,
            path=str(path.relative_to(root)),
            target_type="position",
            num_outputs=3,
            higher_is_better=False,
            metric_name="mean_distance_error",
            sample_shape=dataset.sample_shape,
            coord_min=[float(value) for value in coord_min],
            coord_max=[float(value) for value in coord_max],
        )
    if task == "deepmimo-beam":
        path = root / "data/downstream/deepmimo.h5"
        dataset = DeepMIMO(path, n_beams=16)
        outputs = int(dataset.effective_n_beams or dataset.selected_n_beams or 16)
        return dataset, TaskSpec(
            name=task,
            path=str(path.relative_to(root)),
            target_type="classification",
            num_outputs=outputs,
            higher_is_better=True,
            metric_name="macro_accuracy",
            sample_shape=dataset.sample_shape,
        )
    if task == "human-activity":
        # HAS is kept alongside the repository, like the other local datasets
        # that were not copied into Git. It is now a primary task rather than
        # a Phase 6 OOD task.
        path = root.parent / "has.h5"
        dataset = ImageDataset(path, sample_key="csi", label_key="label")
        return dataset, TaskSpec(
            name=task,
            path=_path_record(path, root),
            target_type="classification",
            num_outputs=6,
            higher_is_better=True,
            metric_name="macro_accuracy",
            sample_shape=dataset.sample_shape,
        )
    if task == "jamming":
        path = root.parent / "jamming.h5"
        dataset, outputs, _ = _classification_dataset(
            path,
            sample_key="normalized_spectrograms",
            label_key="labels",
        )
        return dataset, TaskSpec(
            name=task,
            path=_path_record(path, root),
            target_type="classification",
            num_outputs=outputs,
            higher_is_better=True,
            metric_name="macro_accuracy",
            sample_shape=dataset.sample_shape,
        )
    if task == "interference":
        path = root.parent / "interference.h5"
        dataset, outputs, _ = _classification_dataset(
            path,
            sample_key="cls_spectrograms",
            label_key="cls_labels",
        )
        return dataset, TaskSpec(
            name=task,
            path=_path_record(path, root),
            target_type="classification",
            num_outputs=outputs,
            higher_is_better=True,
            metric_name="macro_accuracy",
            sample_shape=dataset.sample_shape,
        )
    raise ValueError(f"unsupported Phase 5 task: {task}")


def _shuffle(indices: list[int], generator: torch.Generator) -> list[int]:
    if len(indices) < 2:
        return indices
    order = torch.randperm(len(indices), generator=generator).tolist()
    return [indices[index] for index in order]


def _three_way_counts(total: int) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError("each split group must contain at least three examples")
    validation = max(1, round(total * SPLIT_FRACTIONS["validation"]))
    test = max(1, round(total * SPLIT_FRACTIONS["test"]))
    train = total - validation - test
    if train < 1:
        raise ValueError(f"cannot create an 80/10/10 split from {total} examples")
    return train, validation, test


def split_indices(
    count: int,
    *,
    seed: int,
    labels: Sequence[int] | None = None,
) -> dict[str, list[int]]:
    """Create a fixed 80/10/10 split, stratified when scalar labels exist."""

    generator = torch.Generator().manual_seed(int(seed))
    if labels is None:
        train_count, validation_count, _ = _three_way_counts(count)
        order = torch.randperm(count, generator=generator).tolist()
        return {
            "train": order[:train_count],
            "validation": order[train_count:train_count + validation_count],
            "test": order[train_count + validation_count:],
        }
    if len(labels) != count:
        raise ValueError(f"received {len(labels)} labels for {count} examples")

    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    split = {"train": [], "validation": [], "test": []}
    for label in sorted(groups):
        group = _shuffle(groups[label], generator)
        train_count, validation_count, _ = _three_way_counts(len(group))
        split["train"].extend(group[:train_count])
        split["validation"].extend(group[train_count:train_count + validation_count])
        split["test"].extend(group[train_count + validation_count:])
    return {name: _shuffle(indices, generator) for name, indices in split.items()}


def _labels(dataset: Dataset) -> list[int] | None:
    mapping: dict[int, int] | None = None
    if isinstance(dataset, _RemappedLabelDataset):
        mapping = dataset.mapping
        dataset = dataset.dataset
    if not isinstance(dataset, ImageDataset):
        return None
    if dataset.label_dtype is not torch.long:
        return None
    with h5py.File(dataset.h5_path, "r") as h5:
        labels = torch.as_tensor(h5[dataset.label_key][:]).reshape(-1)
    values = [int(value) for value in labels.tolist()]
    return [mapping[value] for value in values] if mapping is not None else values


def create_split_manifest(repository_root: Path, *, seed: int) -> dict[str, object]:
    tasks: dict[str, object] = {}
    for task in PRIMARY_TASKS:
        dataset, spec = build_task_dataset(task, repository_root)
        split = split_indices(len(dataset), seed=seed, labels=_labels(dataset))
        tasks[task] = {
            "spec": asdict(spec),
            "counts": {name: len(indices) for name, indices in split.items()},
            "indices": split,
        }
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "seed": int(seed),
        "fractions": SPLIT_FRACTIONS,
        "tasks": tasks,
    }


def load_or_create_split_manifest(path: Path, repository_root: Path, *, seed: int) -> dict[str, object]:
    path = Path(path)
    if path.is_file():
        manifest = json.loads(path.read_text())
        if manifest.get("schema_version") != SPLIT_SCHEMA_VERSION:
            raise ValueError(f"unsupported split manifest schema: {manifest.get('schema_version')}")
        if int(manifest.get("seed", -1)) != int(seed):
            raise ValueError(f"split manifest seed {manifest.get('seed')} does not match requested seed {seed}")
        if manifest.get("fractions") != SPLIT_FRACTIONS:
            raise ValueError("split manifest fractions do not match the Phase 5 80/10/10 protocol")
        return manifest
    manifest = create_split_manifest(repository_root, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def task_subsets(dataset: Dataset, split_record: dict[str, object]) -> dict[str, Subset]:
    indices = split_record["indices"]
    return {name: Subset(dataset, [int(index) for index in values]) for name, values in indices.items()}
