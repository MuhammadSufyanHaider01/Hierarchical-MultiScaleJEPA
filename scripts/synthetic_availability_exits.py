#!/usr/bin/env python3
"""Validation-only frozen-P2 early-exit selection on synthetic master data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.flop_counter import FlopCounterMode

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from configs import init_jepa_config  # noqa: E402
from datasets import ImageDataset  # noqa: E402
from evaluation.exits import select_best_exit, select_earliest_within_tolerance  # noqa: E402
from models import Encoder, FrozenEncoderExit, load_context_encoder_checkpoint  # noqa: E402
from models.components import set_attention_backend  # noqa: E402
from phase5 import TaskSpec  # noqa: E402
from synthetic.compatibility import DEFAULT_SOURCE_METRICS, source_file, source_repository_root  # noqa: E402
from synthetic.dataset import SyntheticAvailabilityDataset  # noqa: E402
from synthetic.manifests import validate_realization_manifest  # noqa: E402
from synthetic.schema import load_task_profile, require_verified_semantics  # noqa: E402
from synthetic.statistics import bootstrap_interval  # noqa: E402

EXIT_LAYERS = (2, 4, 6, 8)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "test"))
    parser.add_argument("--task", choices=("nrpos-outdoor", "interference", "jamming"), required=True)
    parser.add_argument("--sweep", choices=("antenna", "time"), required=True)
    parser.add_argument("--source-metrics", type=Path, default=Path(DEFAULT_SOURCE_METRICS))
    parser.add_argument("--synthetic-root", type=Path, default=Path("data/synthetic/sionna_v3"))
    parser.add_argument("--anchor", type=Path, help="passing calibration anchor artifact required for select")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--measure-iterations", type=int, default=100)
    parser.add_argument("--matmul-precision", choices=("high", "highest"), default="high")
    parser.add_argument("--attention-backend", choices=("math", "fused"), default="math")
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source(path: Path, required_task: str | None = None) -> tuple[dict[str, Any], Path, Path]:
    path = path if path.is_absolute() else REPOSITORY_ROOT / path
    source = json.loads(path.read_text())
    if source.get("method") != "p2":
        raise ValueError("synthetic availability requires the authoritative P2 source artifact")
    if float(source.get("protocol", {}).get("tolerance", -1)) != 0.05:
        raise ValueError("P2 source artifact must use the 5% earliest-within-best protocol")
    source_root = source_repository_root(path)
    checkpoint = source_file(str(source.get("checkpoint", "")), source_root)
    manifest = source_file(str(source.get("split_manifest", "")), source_root)
    if not checkpoint.is_file() or not manifest.is_file():
        raise FileNotFoundError("P2 source checkpoint or source split manifest is missing")
    split_record = json.loads(manifest.read_text())
    if split_record.get("schema_version") != 3:
        raise ValueError("synthetic study requires the authoritative six-task v3 P2 split manifest")
    if not isinstance(split_record.get("tasks"), dict):
        raise ValueError("P2 source split manifest has no per-task records")
    for task in ((required_task,) if required_task else ()):
        if not isinstance(split_record["tasks"].get(task, {}).get("indices"), dict):
            raise ValueError(f"P2 source split manifest is missing task indices for {task}")
    return source, checkpoint, manifest


def _task_spec(source: dict[str, Any], task: str) -> TaskSpec:
    record = source.get("tasks", {}).get(task)
    if not isinstance(record, dict):
        raise ValueError(f"P2 source metrics do not contain task {task}")
    spec = dict(record["spec"])
    spec["sample_shape"] = tuple(spec["sample_shape"])
    return TaskSpec(**spec)


def _head_paths(source: dict[str, Any], task: str, source_root: Path) -> dict[int, Path]:
    records = source["tasks"][task]["exits"]
    paths = {
        layer: source_file(str(records[str(layer)]["head_path"]), source_root)
        for layer in EXIT_LAYERS
    }
    for layer, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"P2 {task} L{layer} frozen head is missing: {path}")
    return paths


def _source_config_path(source: dict[str, Any], source_root: Path) -> Path:
    config_value = source.get("config_path") or source.get("config")
    if isinstance(config_value, dict):
        config_value = config_value.get("path")
    if config_value:
        path = source_file(str(config_value), source_root)
    else:
        # The P2 source packet omits config_path; the P2 run family and
        # checkpoint schema identify the checked-in P2 config.
        path = REPOSITORY_ROOT / "configs" / "p2.yml"
    if not path.is_file():
        raise FileNotFoundError(f"authoritative P2 encoder config is missing: {path}")
    return path


def _new_encoder(
    checkpoint: Path, device: torch.device, precision: str, backend: str,
    config_path: Path | None = None,
) -> Encoder:
    config_path = config_path or REPOSITORY_ROOT / "configs" / "p2.yml"
    config = init_jepa_config(str(config_path))
    encoder = Encoder(
        img_size=tuple(config.encoder.img_size),
        patch_size=config.encoder.patch_size,
        embed_dim=config.encoder.embed_dim,
        depth=config.encoder.depth,
        num_heads=config.encoder.num_heads,
    ).to(device)
    load_context_encoder_checkpoint(encoder, checkpoint)
    set_attention_backend(encoder, backend)
    encoder.eval()
    return encoder


def _load_exit_models(
    encoder: Encoder, paths: dict[int, Path], spec: TaskSpec, device: torch.device
) -> dict[int, FrozenEncoderExit]:
    result = {}
    for layer, path in paths.items():
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("head"), dict):
            raise ValueError(f"frozen head file has no 'head' state dictionary: {path}")
        model = FrozenEncoderExit(encoder, exit_layer=layer, num_outputs=spec.num_outputs).to(device)
        model.head.load_state_dict(checkpoint["head"], strict=True)
        for parameter in model.head.parameters():
            parameter.requires_grad_(False)
        model.eval()
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("encoder and frozen head parameters must remain frozen")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("loaded inference model unexpectedly has parameter gradients")
        result[layer] = model
    return result


def _loader(dataset, indices: list[int], args: argparse.Namespace, device: torch.device) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def _compute_metrics(
    model: FrozenEncoderExit, loader: DataLoader, spec: TaskSpec, device: torch.device
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    coord_min = torch.tensor(spec.coord_min, device=device) if spec.coord_min else None
    coord_max = torch.tensor(spec.coord_max, device=device) if spec.coord_max else None
    model.eval()
    with torch.inference_mode():
        for samples, labels in loader:
            samples = samples.to(device=device, dtype=torch.float32, non_blocking=True)
            logits = model(samples)
            if spec.target_type == "classification":
                labels_all.append(labels.detach().cpu().numpy().reshape(-1))
                predictions_all.append(logits.argmax(dim=1).detach().cpu().numpy().reshape(-1))
            else:
                labels_all.append(labels.detach().cpu().numpy())
                labels = labels.to(device=device, dtype=torch.float32)
                prediction = (logits + 1.0) * 0.5 * (coord_max - coord_min) + coord_min
                target = (labels + 1.0) * 0.5 * (coord_max - coord_min) + coord_min
                predictions_all.append(torch.linalg.vector_norm(prediction - target, dim=-1).cpu().numpy())
    truth = np.concatenate(labels_all) if labels_all else np.empty((0,), dtype=np.int64)
    values = np.concatenate(predictions_all)
    if spec.target_type == "classification":
        recalls = [
            float(np.mean(values[truth == class_id] == class_id))
            for class_id in range(spec.num_outputs)
            if np.any(truth == class_id)
        ]
        metrics = {
            "accuracy": float(np.mean(values == truth)),
            "macro_accuracy": float(np.mean(recalls)),
        }
    else:
        metrics = {
            "mean_distance_error": float(np.mean(values)),
            "median_distance_error": float(np.median(values)),
        }
    return metrics, truth, values


def _bootstrap_metrics(
    spec: TaskSpec, labels: np.ndarray, values: np.ndarray, *, seed: int
) -> dict[str, dict[str, float | int]]:
    if spec.target_type == "position":
        return {
            "mean_distance_error": bootstrap_interval(
                values, statistic=np.mean, seed=seed, resamples=1000
            ),
            "median_distance_error": bootstrap_interval(
                values, statistic=np.median, seed=seed + 1, resamples=1000
            ),
        }

    pair = np.stack([labels.astype(np.int64), values.astype(np.int64)], axis=1)

    def accuracy(rows: np.ndarray) -> float:
        return float(np.mean(rows[:, 0] == rows[:, 1]))

    def macro_accuracy(rows: np.ndarray) -> float:
        recalls = [
            np.mean(rows[rows[:, 0] == class_id, 1] == class_id)
            for class_id in range(spec.num_outputs)
            if np.any(rows[:, 0] == class_id)
        ]
        return float(np.mean(recalls))

    return {
        "accuracy": bootstrap_interval(
            pair, statistic=accuracy, seed=seed, resamples=1000, strata=labels
        ),
        "macro_accuracy": bootstrap_interval(
            pair, statistic=macro_accuracy, seed=seed + 1, resamples=1000, strata=labels
        ),
    }


def _paired_metric_difference_ci(
    spec: TaskSpec,
    labels: np.ndarray,
    current: np.ndarray,
    reference: np.ndarray,
    *,
    seed: int,
) -> dict[str, float | int]:
    if len(current) != len(reference) or len(labels) != len(current):
        raise ValueError("paired bootstrap arrays must have identical realization order")
    if spec.target_type == "position":
        return bootstrap_interval(
            np.asarray(current) - np.asarray(reference),
            statistic=np.mean,
            seed=seed,
            resamples=1000,
        )
    rows = np.stack(
        [labels.astype(np.int64), current.astype(np.int64), reference.astype(np.int64)],
        axis=1,
    )

    def delta_macro_accuracy(values: np.ndarray) -> float:
        truth = values[:, 0]
        recalls = []
        for class_id in range(spec.num_outputs):
            selected = truth == class_id
            if np.any(selected):
                recalls.append(
                    np.mean(values[selected, 1] == class_id)
                    - np.mean(values[selected, 2] == class_id)
                )
        return float(np.mean(recalls))

    return bootstrap_interval(
        rows,
        statistic=delta_macro_accuracy,
        seed=seed,
        resamples=1000,
        strata=labels,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    model: FrozenEncoderExit,
    sample_shape: tuple[int, ...],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    sample = torch.zeros((1, *sample_shape), device=device, dtype=torch.float32)
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        model(sample)
    _synchronize(device)
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        _synchronize(device)
        for _ in range(iterations):
            _synchronize(device)
            started = time.perf_counter_ns()
            model(sample)
            _synchronize(device)
            times.append((time.perf_counter_ns() - started) / 1e6)
    ordered = sorted(times)
    return {
        "flops": int(counter.get_total_flops()),
        "batch_size": 1,
        "latency_ms": {
            "mean": float(np.mean(times)),
            "median": float(np.median(times)),
            "p95": float(np.quantile(times, 0.95)),
            "standard_deviation": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
        },
        "warmup_iterations": warmup,
        "measure_iterations": iterations,
        "synchronized": device.type == "cuda",
    }


def _load_master(
    task: str, sweep: str, synthetic_root: Path
) -> tuple[ImageDataset, dict[str, Any], Path]:
    task_root = synthetic_root / task
    manifest_path = task_root / "split_manifest.json"
    master_path = task_root / "master.h5"
    if not manifest_path.is_file() or not master_path.is_file():
        raise FileNotFoundError(f"synthetic master or manifest missing under {task_root}")
    manifest = json.loads(manifest_path.read_text())
    validate_realization_manifest(manifest)
    if manifest.get("task") != task:
        raise ValueError(f"synthetic manifest task {manifest.get('task')!r} != {task}")
    sample_plan = manifest.get("sample_plan")
    if not isinstance(sample_plan, dict) or sample_plan.get("stage") != "sample_count_locked" or sample_plan.get("task") != task:
        raise ValueError("formal synthetic manifest lacks its pre-selection locked sample-count plan")
    dataset = ImageDataset(
        master_path,
        sample_key="features",
        label_key="label",
        label_dtype=torch.float32 if task == "nrpos-outdoor" else torch.long,
    )
    with h5py.File(master_path, "r") as h5:
        semantic_basis = h5.attrs.get("semantic_basis", "")
        if isinstance(semantic_basis, bytes):
            semantic_basis = semantic_basis.decode()
        if semantic_basis == "assumption_based_synthetic_domain":
            if int(h5.attrs.get("assumptions_approved", 0)) != 1 or int(h5.attrs.get("semantics_verified", 0)) != 0:
                raise RuntimeError(f"synthetic master has inconsistent assumption metadata: {master_path}")
        elif h5.attrs.get("semantics_verified", 0) != 1:
            raise RuntimeError(f"synthetic master has no verified-semantics marker: {master_path}")
        if h5.attrs.get("task", "") != task:
            raise ValueError(f"synthetic master task attribute does not match {task}")
        if h5.attrs.get("split_kind", "") != "formal":
            raise ValueError("selection and test require a formal master, not a pilot")
        ids = [value.decode() if isinstance(value, bytes) else str(value) for value in h5["realization_id"][:]]
    if ids != manifest["realization_ids"] or len(dataset) != len(ids):
        raise ValueError("master row order does not match the immutable realization manifest")
    if int(sample_plan.get("formal_total", -1)) != len(ids):
        raise ValueError("formal master count does not match the locked sample plan")
    config_name = {"nrpos-outdoor": "nrpos_outdoor_sionna_v5_assumed.yml", "interference": "interference.yml", "jamming": "jamming.yml"}[task]
    profile_path = REPOSITORY_ROOT / "configs" / "sionna" / config_name
    profile, _ = load_task_profile(profile_path)
    if manifest.get("task_profile_sha256") != _sha256(profile_path):
        raise ValueError("synthetic master was generated from a different task profile")
    if manifest.get("common_config_sha256") != _sha256(REPOSITORY_ROOT / "configs" / "sionna" / "common.yml"):
        raise ValueError("synthetic master was generated from a different array/sweep configuration")
    if not str(manifest.get("sionna_version", "")).startswith("2.1."):
        raise ValueError("synthetic master was not generated with pinned Sionna 2.1.x")
    master_stat = manifest.get("master_stat", {})
    stat = master_path.stat()
    if (
        int(master_stat.get("size_bytes", -1)) != stat.st_size
        or int(master_stat.get("modified_ns", -1)) != stat.st_mtime_ns
    ):
        raise ValueError("synthetic master does not match the immutable manifest metadata")
    expected_master_hash = manifest.get("master_sha256")
    if not expected_master_hash or _sha256(master_path) != expected_master_hash:
        raise ValueError("synthetic master content does not match the immutable manifest digest")
    with h5py.File(master_path, "r") as h5:
        feature_mode = h5.attrs.get("feature_mode", "")
        if isinstance(feature_mode, bytes):
            feature_mode = feature_mode.decode()
        if feature_mode != profile.feature_mode:
            raise ValueError("synthetic master feature_mode does not match the audited profile")
    if dataset.sample_shape[0] != 16:
        raise ValueError("synthetic master must contain the full 16-element aperture")
    expected = [224, 224]
    expected[profile.temporal_axis - 1] = max(profile.temporal_levels) * 16
    if list(dataset.sample_shape[1:]) != expected:
        raise ValueError(f"master shape {dataset.sample_shape} does not match {expected}")
    return dataset, manifest, master_path


def _assert_master_source_provenance(
    manifest: dict[str, Any], source_metrics: Path, checkpoint: Path, source_split: Path
) -> None:
    metrics_record = manifest.get("source_p2_metrics", {})
    checkpoint_record = manifest.get("source_p2_checkpoint", {})
    split_record = manifest.get("source_p2_split_manifest", {})
    if (
        Path(metrics_record.get("path", "")).resolve() != source_metrics.resolve()
        or metrics_record.get("sha256") != _sha256(source_metrics)
        or Path(checkpoint_record.get("path", "")).resolve() != checkpoint.resolve()
        or Path(split_record.get("path", "")).resolve() != source_split.resolve()
        or split_record.get("sha256") != _sha256(source_split)
    ):
        raise ValueError("synthetic master was created for a different authoritative P2 source packet")


def _indices(manifest: dict[str, Any], split: str) -> list[int]:
    index = {realization_id: row for row, realization_id in enumerate(manifest["realization_ids"])}
    ids = manifest["splits"].get(split)
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"manifest has no {split} realization IDs")
    return [index[realization_id] for realization_id in ids]


def _condition_values(profile, sweep: str) -> tuple[tuple[int, ...], str]:
    return (profile.antenna_levels, "available_channels") if sweep == "antenna" else (
        profile.temporal_levels,
        "temporal_patches",
    )


def _make_views(dataset, profile, sweep: str):
    levels, _ = _condition_values(profile, sweep)
    mode = "antenna" if sweep == "antenna" else "time"
    return [
        SyntheticAvailabilityDataset(
            dataset,
            mode=mode,
            level=level,
            temporal_axis=profile.temporal_axis,
            reference_channels=profile.reference_channels,
            reference_time_patches=profile.reference_time_patches,
            source_sample_shape=dataset.sample_shape,
            max_channels=16,
            array_subset_policy=profile.array_subset_policy,
        )
        for level in levels
    ]


def _selection(
    args: argparse.Namespace,
    *,
    source: dict[str, Any],
    source_path: Path,
    source_root: Path,
    checkpoint: Path,
    manifest_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    profile_path = REPOSITORY_ROOT / "configs" / "sionna" / {
        "nrpos-outdoor": "nrpos_outdoor_sionna_v5_assumed.yml",
        "interference": "interference.yml",
        "jamming": "jamming.yml",
    }[args.task]
    profile, task_config = load_task_profile(profile_path)
    require_verified_semantics(profile)
    spec = _task_spec(source, args.task)
    if spec.target_type != profile.target_type or spec.num_outputs != profile.num_outputs:
        raise ValueError(f"synthetic profile does not match the frozen P2 {args.task} head")
    synthetic_root = args.synthetic_root if args.synthetic_root.is_absolute() else REPOSITORY_ROOT / args.synthetic_root
    dataset, synthetic_manifest, master_path = _load_master(args.task, args.sweep, synthetic_root)
    _assert_master_source_provenance(synthetic_manifest, source_path, checkpoint, manifest_path)
    if args.anchor is None:
        raise ValueError("selection requires a reviewed passing --anchor calibration artifact")
    anchor_path = args.anchor if args.anchor.is_absolute() else REPOSITORY_ROOT / args.anchor
    anchor = json.loads(anchor_path.read_text())
    if anchor.get("stage") != "anchor_passed":
        raise ValueError("selection requires an anchor artifact with stage=anchor_passed")
    if anchor.get("task") != args.task or anchor.get("sweep") != args.sweep:
        raise ValueError("calibration anchor task/sweep does not match this selection")
    source_config = _source_config_path(source, source_root)
    heads = _head_paths(source, args.task, source_root)
    anchor_provenance = anchor["provenance"]
    if (
        anchor_provenance.get("attention_backend") != args.attention_backend
        or anchor_provenance.get("matmul_precision") != args.matmul_precision
    ):
        raise ValueError("selection backend and precision must match the reviewed calibration anchor")
    if (
        anchor_provenance.get("source_metrics_sha256") != _sha256(source_path)
        or anchor_provenance.get("source_checkpoint_sha256") != _sha256(checkpoint)
        or anchor_provenance.get("source_config", {}).get("sha256") != _sha256(source_config)
        or anchor_provenance.get("task_profile_sha256") != _sha256(profile_path)
    ):
        raise ValueError("calibration anchor was evaluated with a different P2 source packet")
    for layer, path in heads.items():
        if anchor_provenance.get("task_head_sha256", {}).get(str(layer)) != _sha256(path):
            raise ValueError(f"calibration anchor used a different P2 L{layer} head")
    anchor_master = anchor["provenance"]["synthetic_master"]
    if (
        Path(anchor_master["path"]).resolve() != master_path.resolve()
        or int(anchor_master["size_bytes"]) != master_path.stat().st_size
        or int(anchor_master["modified_ns"]) != master_path.stat().st_mtime_ns
    ):
        raise ValueError("calibration anchor does not match the current synthetic master")
    manifest_digest = _sha256(synthetic_root / args.task / "split_manifest.json")
    if anchor["provenance"].get("synthetic_manifest_sha256") != manifest_digest:
        raise ValueError("calibration anchor does not match the immutable synthetic split")
    if task_config["task"]["name"] != args.task:
        raise ValueError("task profile name mismatch")
    levels, level_field = _condition_values(profile, args.sweep)
    encoder = _new_encoder(checkpoint, device, args.matmul_precision, args.attention_backend, source_config)
    models = _load_exit_models(encoder, heads, spec, device)
    selection_indices = _indices(synthetic_manifest, "selection")
    conditions = []
    for view in _make_views(dataset, profile, args.sweep):
        level = view.level
        loader = _loader(view, selection_indices, args, device)
        per_exit = {}
        for layer, model in models.items():
            metrics, _, _ = _compute_metrics(model, loader, spec, device)
            per_exit[str(layer)] = metrics
        values = {layer: per_exit[str(layer)][spec.metric_name] for layer in EXIT_LAYERS}
        selected_best = select_best_exit(values, higher_is_better=spec.higher_is_better)
        selected = select_earliest_within_tolerance(
            values, higher_is_better=spec.higher_is_better, tolerance=0.05
        )
        grid = [size // 16 for size in view.sample_shape[1:]]
        conditions.append(
            {
                "resource_level": level,
                level_field: level,
                "input_shape": list(view.sample_shape),
                "patch_grid": grid,
                "token_count": int(view.sample_shape[0] * grid[0] * grid[1]),
                "antenna_indices": list(view.antenna_indices),
                "validation": per_exit,
                "validation_l8_gap": {
                    str(layer): (
                        per_exit[str(layer)][spec.metric_name] - per_exit["8"][spec.metric_name]
                        if spec.higher_is_better
                        else per_exit["8"][spec.metric_name] - per_exit[str(layer)][spec.metric_name]
                    )
                    for layer in (2, 4, 6)
                },
                "selected_exit_best_validation": selected_best,
                "selected_exit_earliest_within_tolerance": selected,
                "compute": {
                    str(layer): _measure(
                        model,
                        tuple(view.sample_shape),
                        device,
                        args.warmup_iterations,
                        args.measure_iterations,
                    )
                    for layer, model in models.items()
                },
            }
        )
    provenance = {
        "source_metrics": {
            "path": str(source_path.resolve()),
            "sha256": _sha256(source_path),
        },
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "source_config": {
            "path": str(source_config.resolve()),
            "sha256": _sha256(source_config),
        },
        "source_split_manifest": str(manifest_path.resolve()),
        "source_split_manifest_sha256": _sha256(manifest_path),
        "synthetic_master": {
            "path": str(master_path.resolve()),
            "size_bytes": master_path.stat().st_size,
            "modified_ns": master_path.stat().st_mtime_ns,
        },
        "synthetic_manifest": str((synthetic_root / args.task / "split_manifest.json").resolve()),
        "synthetic_manifest_sha256": _sha256(synthetic_root / args.task / "split_manifest.json"),
        "calibration_anchor": {
            "path": str(anchor_path.resolve()),
            "sha256": _sha256(anchor_path),
        },
        "source_task_heads": {str(layer): str(path.resolve()) for layer, path in heads.items()},
        "source_task_head_sha256": {str(layer): _sha256(path) for layer, path in heads.items()},
        "task_profile": {
            "path": str(profile_path.resolve()),
            "sha256": _sha256(profile_path),
        },
    }
    return {
        "schema_version": 1,
        "study": "sionna_synthetic_availability",
        "stage": "selection_complete",
        "source_method": "p2",
        "task": args.task,
        "sweep": args.sweep,
        "test_evaluation": False,
        "task_spec": asdict(spec),
        "levels": list(levels),
        "provenance": provenance,
        "protocol": {
            "selection_uses": "synthetic_selection_split_only",
            "tolerance": 0.05,
            "candidate_exit_layers": list(EXIT_LAYERS),
            "foundation_weights_updated": False,
            "head_weights_updated": False,
            "synthetic_split": "calibration/selection/test = 20/20/60",
            "temporal_axis": profile.temporal_axis,
            "attention_backend": args.attention_backend,
            "array_subset_policy": profile.array_subset_policy,
            "specification_status": profile.specification_status,
            "matmul_precision": args.matmul_precision,
            "device": str(device),
        },
        "real_p2_validation": {
            str(layer): source["tasks"][args.task]["exits"][str(layer)]["validation"]
            for layer in EXIT_LAYERS
        },
        "conditions": conditions,
    }


def _test(
    selection_path: Path, output_path: Path, args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text())
    if selection.get("stage") != "selection_complete" or selection.get("test_evaluation") is not False:
        raise ValueError("test requires a completed validation-only selection artifact")
    if selection["task"] != args.task or selection["sweep"] != args.sweep:
        raise ValueError("requested test task/sweep does not match selection artifact")
    provenance = selection["provenance"]
    if selection.get("protocol", {}).get("tolerance") != 0.05:
        raise ValueError("selection artifact does not record the required 5% selector")
    if selection.get("protocol", {}).get("attention_backend") != args.attention_backend:
        raise ValueError("test attention backend must match validation selection")
    if selection.get("protocol", {}).get("matmul_precision") != args.matmul_precision:
        raise ValueError("test matmul precision must match validation selection")
    source_path = Path(provenance["source_metrics"]["path"])
    source, checkpoint, source_split_path = _load_source(source_path, args.task)
    source_root = source_repository_root(source_path)
    profile_path = REPOSITORY_ROOT / "configs" / "sionna" / {
        "nrpos-outdoor": "nrpos_outdoor_sionna_v5_assumed.yml",
        "interference": "interference.yml",
        "jamming": "jamming.yml",
    }[args.task]
    profile, _ = load_task_profile(profile_path)
    require_verified_semantics(profile)
    if (
        provenance.get("task_profile", {}).get("path") != str(profile_path.resolve())
        or provenance.get("task_profile", {}).get("sha256") != _sha256(profile_path)
    ):
        raise ValueError("task profile changed since validation selection")
    dataset, synthetic_manifest, master_path = _load_master(
        args.task,
        args.sweep,
        (args.synthetic_root if args.synthetic_root.is_absolute() else REPOSITORY_ROOT / args.synthetic_root),
    )
    _assert_master_source_provenance(synthetic_manifest, source_path, checkpoint, source_split_path)
    master_record = provenance["synthetic_master"]
    if (
        master_path.resolve() != Path(master_record["path"]).resolve()
        or master_path.stat().st_size != int(master_record["size_bytes"])
        or master_path.stat().st_mtime_ns != int(master_record["modified_ns"])
    ):
        raise ValueError("synthetic master changed since selection")
    synthetic_manifest_path = Path(provenance["synthetic_manifest"])
    expected_manifest_path = (
        args.synthetic_root if args.synthetic_root.is_absolute()
        else REPOSITORY_ROOT / args.synthetic_root
    ) / args.task / "split_manifest.json"
    if synthetic_manifest_path.resolve() != expected_manifest_path.resolve():
        raise ValueError("synthetic split manifest path does not match the requested data root")
    if _sha256(synthetic_manifest_path) != provenance["synthetic_manifest_sha256"]:
        raise ValueError("synthetic split manifest changed since selection")
    if _sha256(source_path) != provenance["source_metrics"]["sha256"]:
        raise ValueError("authoritative P2 source metrics changed since selection")
    if checkpoint.resolve() != Path(provenance["source_checkpoint"]).resolve():
        raise ValueError("P2 checkpoint path changed since selection")
    source_config = _source_config_path(source, source_root)
    if (
        _sha256(checkpoint) != provenance["source_checkpoint_sha256"]
        or _sha256(source_config) != provenance["source_config"]["sha256"]
        or _sha256(source_split_path) != provenance["source_split_manifest_sha256"]
    ):
        raise ValueError("P2 checkpoint/config/source split changed since selection")
    spec = _task_spec(source, args.task)
    heads = _head_paths(source, args.task, source_root)
    for layer, path in heads.items():
        if _sha256(path) != provenance["source_task_head_sha256"][str(layer)]:
            raise ValueError(f"frozen P2 L{layer} head changed since selection")
    encoder = _new_encoder(
        checkpoint, device, args.matmul_precision, args.attention_backend, source_config
    )
    models = _load_exit_models(encoder, heads, spec, device)
    test_indices = _indices(synthetic_manifest, "test")
    results = []
    raw_by_level: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for condition in selection["conditions"]:
        level = int(condition["resource_level"])
        view = SyntheticAvailabilityDataset(
            dataset,
            mode=args.sweep,
            level=level,
            temporal_axis=profile.temporal_axis,
            reference_channels=profile.reference_channels,
            reference_time_patches=profile.reference_time_patches,
            source_sample_shape=dataset.sample_shape,
            max_channels=16,
            array_subset_policy=profile.array_subset_policy,
        )
        loader = _loader(view, test_indices, args, device)
        fixed_exit_results = {}
        raw_values = {}
        for layer, model in models.items():
            metrics, labels, values = _compute_metrics(model, loader, spec, device)
            fixed_exit_results[str(layer)] = {
                "metrics": metrics,
                "confidence_intervals": _bootstrap_metrics(
                    spec, labels, values, seed=2026 + level * 10 + layer
                ),
            }
            raw_values[layer] = (labels, values)
        raw_by_level[level] = raw_values
        selected_layer = int(condition["selected_exit_earliest_within_tolerance"])
        selected = fixed_exit_results[str(selected_layer)]
        results.append(
            {
                "resource_level": level,
                "fixed_exits": fixed_exit_results,
                "selected_exit": selected_layer,
                "selected_metrics": selected,
                "sample_count": len(test_indices),
            }
        )
    full_level = max(int(value) for value in selection["levels"])
    full_result = next(result for result in results if int(result["resource_level"]) == full_level)
    full_selected_layer = int(full_result["selected_exit"])
    for result in results:
        level = int(result["resource_level"])
        selected_layer = int(result["selected_exit"])
        current_labels, current_values = raw_by_level[level][selected_layer]
        reference_labels, reference_values = raw_by_level[full_level][full_selected_layer]
        if not np.array_equal(current_labels, reference_labels):
            raise ValueError("paired test conditions changed labels or realization order")
        result["paired_selected_metric_difference_vs_full_information"] = (
            _paired_metric_difference_ci(
                spec,
                current_labels,
                current_values,
                reference_values,
                seed=4040 + level,
            )
        )
    return {
        "schema_version": 1,
        "study": "sionna_synthetic_availability",
        "stage": "test_complete",
        "task": args.task,
        "sweep": args.sweep,
        "test_evaluation": True,
        "selection_artifact": str(selection_path.resolve()),
        "selection_artifact_sha256": _sha256(selection_path),
        "test_split": "synthetic_test_only",
        "conditions": results,
    }


def main() -> int:
    args = _args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    device = _device(args.device)
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = args.matmul_precision != "highest"
    torch.backends.cudnn.allow_tf32 = args.matmul_precision != "highest"
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)

    if args.stage == "select":
        if args.output_root is None or args.artifact is not None or args.output is not None:
            raise ValueError("select requires --output-root only")
        if args.anchor is None:
            raise ValueError("select requires --anchor with a passing calibration artifact")
        source_path = args.source_metrics if args.source_metrics.is_absolute() else REPOSITORY_ROOT / args.source_metrics
        source, checkpoint, manifest_path = _load_source(source_path, args.task)
        output_path = args.output_root if args.output_root.is_absolute() else REPOSITORY_ROOT / args.output_root
        output_path = output_path / "input_information_exit_metrics.json"
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite selection artifact: {output_path}")
        artifact = _selection(
            args,
            source=source,
            source_path=source_path,
            source_root=source_repository_root(source_path),
            checkpoint=checkpoint,
            manifest_path=manifest_path,
            device=device,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"artifact": str(output_path), "stage": artifact["stage"]}, indent=2))
        return 0

    if args.artifact is None or args.output is None or args.output_root is not None:
        raise ValueError("test requires --artifact and --output only")
    artifact_path = args.artifact if args.artifact.is_absolute() else REPOSITORY_ROOT / args.artifact
    output_path = args.output if args.output.is_absolute() else REPOSITORY_ROOT / args.output
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite test artifact: {output_path}")
    result = _test(artifact_path, output_path, args, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(output_path), "stage": result["stage"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
