#!/usr/bin/env python3
"""Inference-only input-information characterization for frozen early exits.

The ``select`` stage reads validation data only.  The ``test`` stage can run
only from its completed selection artifact, so availability-conditioned exit
selection cannot observe held-out examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
os.environ.setdefault("WANDB_KEY", "offline")

from configs import init_jepa_config  # noqa: E402
from datasets import InputAvailabilityDataset  # noqa: E402
from evaluation import select_best_exit, select_earliest_within_tolerance  # noqa: E402
from models import Encoder, FrozenEncoderExit, load_context_encoder_checkpoint  # noqa: E402
from phase5 import TaskSpec, build_task_dataset, task_subsets  # noqa: E402


EXIT_LAYERS = (2, 4, 6, 8)
STUDIES = {
    "antenna": {
        "task": "nrpos-outdoor",
        "mode": "antenna_prefix",
        "levels": (1, 2, 3, 4),
        "level_name": "available_channels",
    },
    "temporal": {
        "task": "human-activity",
        "mode": "leading_temporal_patches",
        "levels": (2, 4, 8, 14),
        "level_name": "temporal_patch_columns",
    },
}
ARTIFACT_NAME = "input_information_exit_metrics.json"
SCHEMA_VERSION = 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "test"))
    parser.add_argument("--source-metrics", type=Path, help="Existing full-information exit_metrics.json (select only).")
    parser.add_argument("--artifact", type=Path, help="Completed selection artifact (test only).")
    parser.add_argument("--output-root", type=Path, help="Directory for the selection artifact (select only).")
    parser.add_argument("--config", type=Path, default=None, help="Defaults to configs/<source method>.yml.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--measure-iterations", type=int, default=200)
    return parser.parse_args()


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _source_repository_root(source_path: Path) -> Path:
    """Infer the repository root for paths recorded relative to ``runs/``."""

    for parent in source_path.resolve().parents:
        if parent.name == "runs":
            return parent.parent
    return REPOSITORY_ROOT


def _source_path(path: str | Path, *, source_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else source_root / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, include_hash: bool = False) -> dict[str, object]:
    stat = path.stat()
    record: dict[str, object] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }
    if include_hash:
        record["sha256"] = _sha256(path)
    return record


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else value)


def _loader(dataset, *, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _new_encoder(config, checkpoint: Path, device: torch.device) -> Encoder:
    encoder = Encoder(
        img_size=tuple(config.encoder.img_size),
        patch_size=config.encoder.patch_size,
        embed_dim=config.encoder.embed_dim,
        depth=config.encoder.depth,
        num_heads=config.encoder.num_heads,
    ).to(device)
    load_context_encoder_checkpoint(encoder, checkpoint)
    encoder.eval()
    return encoder


def _load_head(encoder: Encoder, *, exit_layer: int, spec: TaskSpec, path: Path, device: torch.device) -> FrozenEncoderExit:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "head" not in checkpoint:
        raise KeyError(f"head checkpoint has no 'head' state dict: {path}")
    model = FrozenEncoderExit(encoder, exit_layer=exit_layer, num_outputs=spec.num_outputs).to(device)
    model.head.load_state_dict(checkpoint["head"], strict=True)
    model.eval()
    return model


def _evaluate(model: FrozenEncoderExit, loader: DataLoader, spec: TaskSpec, device: torch.device) -> dict[str, float]:
    model.eval()
    total_examples = 0
    if spec.target_type == "classification":
        class_correct = torch.zeros(spec.num_outputs, device=device)
        class_total = torch.zeros(spec.num_outputs, device=device)
        correct = 0
    else:
        errors = []
        coord_min = torch.tensor(spec.coord_min, device=device)
        coord_max = torch.tensor(spec.coord_max, device=device)

    with torch.inference_mode():
        for batch in loader:
            samples, labels = batch[:2]
            samples = samples.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            logits = model(samples)
            batch_size = samples.size(0)
            total_examples += batch_size
            if spec.target_type == "classification":
                labels = labels.long()
                prediction = logits.argmax(dim=1)
                correct += int((prediction == labels).sum().item())
                class_total.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
                class_correct.scatter_add_(0, labels, (prediction == labels).to(torch.float32))
            else:
                labels = labels.float()
                prediction = (logits + 1.0) * 0.5 * (coord_max - coord_min) + coord_min
                target = (labels + 1.0) * 0.5 * (coord_max - coord_min) + coord_min
                errors.append(torch.linalg.vector_norm(prediction - target, dim=-1))

    if total_examples == 0:
        raise RuntimeError("evaluation loader produced no examples")
    if spec.target_type == "classification":
        present = class_total > 0
        return {
            "accuracy": correct / total_examples,
            "macro_accuracy": float((class_correct[present] / class_total[present]).mean().item()),
        }
    distances = torch.cat(errors)
    return {
        "mean_distance_error": float(distances.mean().item()),
        "median_distance_error": float(distances.median().item()),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_selected_compute(
    model: FrozenEncoderExit,
    *,
    sample_shape: tuple[int, ...],
    device: torch.device,
    warmup_iterations: int,
    measure_iterations: int,
) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(0)
    sample = torch.randn((1, *sample_shape), generator=generator, device=device, dtype=torch.float32)
    _synchronize(device)
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        model(sample)
    _synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(sample)
        _synchronize(device)
        samples_ms = []
        for _ in range(measure_iterations):
            _synchronize(device)
            started = time.perf_counter_ns()
            model(sample)
            _synchronize(device)
            samples_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    ordered = sorted(samples_ms)
    p95_index = max(0, min(len(ordered) - 1, round(0.95 * (len(ordered) - 1))))
    return {
        "flops": int(counter.get_total_flops()),
        "batch_size": 1,
        "latency_ms": {
            "mean": statistics.fmean(samples_ms),
            "median": statistics.median(samples_ms),
            "p95": ordered[p95_index],
            "standard_deviation": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        },
        "warmup_iterations": warmup_iterations,
        "measure_iterations": measure_iterations,
        "synchronized": device.type == "cuda",
    }


def _metric_gap(values: dict[int, float], *, higher_is_better: bool) -> dict[str, float]:
    l8 = values[8]
    return {
        str(layer): (l8 - value if higher_is_better else value - l8)
        for layer, value in sorted(values.items())
    }


def _validate_source_metrics(source_path: Path) -> tuple[dict[str, Any], Path, Path, Path, float]:
    source = _read_json(source_path)
    source_root = _source_repository_root(source_path)
    method = source.get("method")
    if method not in {"p0", "p2"}:
        raise ValueError(f"source metrics must describe method 'p0' or 'p2', got {method!r}")
    checkpoint = _source_path(source.get("checkpoint", ""), source_root=source_root)
    manifest = _source_path(source.get("split_manifest", ""), source_root=source_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint is missing: {checkpoint}")
    if not manifest.is_file():
        raise FileNotFoundError(f"source split manifest is missing: {manifest}")
    tolerance = source.get("protocol", {}).get("tolerance")
    if tolerance is None or not math.isclose(float(tolerance), 0.05, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"source metrics must use the unchanged 5% tolerance, got {tolerance!r}")

    tasks = source.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("source metrics has no task mapping")
    missing = []
    for study in STUDIES.values():
        task = study["task"]
        records = tasks.get(task, {}).get("exits", {}) if isinstance(tasks.get(task), dict) else {}
        absent_layers = [f"L{layer}" for layer in EXIT_LAYERS if not isinstance(records.get(str(layer)), dict)]
        unavailable_heads = [
            f"L{layer}"
            for layer in EXIT_LAYERS
            if isinstance(records.get(str(layer)), dict)
            and not _source_path(records[str(layer)].get("head_path", ""), source_root=source_root).is_file()
        ]
        if absent_layers or unavailable_heads:
            details = ", ".join([*(f"missing record {layer}" for layer in absent_layers), *(f"missing head {layer}" for layer in unavailable_heads)])
            missing.append(f"{task}: {details}")
    if missing:
        raise RuntimeError(
            "input-information inference requires completed full-information frozen heads for both tasks; "
            + "; ".join(missing)
        )
    return source, checkpoint, manifest, source_root, float(tolerance)


def _read_manifest(path: Path, *, required_tasks: tuple[str, ...]) -> dict[str, Any]:
    manifest = _read_json(path)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError(f"split manifest has no task mapping: {path}")
    for task in required_tasks:
        indices = tasks.get(task, {}).get("indices") if isinstance(tasks.get(task), dict) else None
        if not isinstance(indices, dict) or not isinstance(indices.get("validation"), list) or not isinstance(indices.get("test"), list):
            raise ValueError(f"split manifest lacks validation/test indices for task {task}: {path}")
    return manifest


def _source_head_paths(source: dict[str, Any], task: str, *, source_root: Path) -> dict[int, Path]:
    records = source["tasks"][task]["exits"]
    return {
        layer: _source_path(records[str(layer)]["head_path"], source_root=source_root)
        for layer in EXIT_LAYERS
    }


def _source_metric_values(source: dict[str, Any], task: str, spec: TaskSpec) -> dict[int, float]:
    records = source["tasks"][task]["exits"]
    return {layer: float(records[str(layer)]["validation"][spec.metric_name]) for layer in EXIT_LAYERS}


def _validate_task_compatibility(source: dict[str, Any], task: str, spec: TaskSpec) -> None:
    source_spec = source["tasks"][task].get("spec", {})
    for field in ("target_type", "num_outputs", "higher_is_better", "metric_name"):
        if source_spec.get(field) != getattr(spec, field):
            raise ValueError(
                f"source task specification for {task} is incompatible at {field}: "
                f"{source_spec.get(field)!r} != {getattr(spec, field)!r}"
            )


def _condition_name(study_name: str, level: int) -> str:
    return f"C{level}" if study_name == "antenna" else f"Tp{level}"


def _condition_record(
    *,
    study_name: str,
    level: int,
    base_dataset,
    split_record: dict[str, Any],
    spec: TaskSpec,
    head_paths: dict[int, Path],
    encoder: Encoder,
    device: torch.device,
    tolerance: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    study = STUDIES[study_name]
    dataset = InputAvailabilityDataset(base_dataset, mode=study["mode"], level=level)
    subsets = task_subsets(dataset, split_record)
    validation_loader = _loader(subsets["validation"], batch_size=args.batch_size, num_workers=args.num_workers, device=device)
    models = {layer: _load_head(encoder, exit_layer=layer, spec=spec, path=head_paths[layer], device=device) for layer in EXIT_LAYERS}
    validation = {layer: _evaluate(models[layer], validation_loader, spec, device) for layer in EXIT_LAYERS}
    values = {layer: float(validation[layer][spec.metric_name]) for layer in EXIT_LAYERS}
    selected_best = select_best_exit(values, higher_is_better=spec.higher_is_better)
    selected_tolerance = select_earliest_within_tolerance(
        values,
        higher_is_better=spec.higher_is_better,
        tolerance=tolerance,
    )
    channels, height, width = dataset.sample_shape
    patch_size = encoder.patch_embed.patch_size
    if height % patch_size or width % patch_size:
        raise RuntimeError(f"availability view is not patch aligned: {dataset.sample_shape}")
    grid = (height // patch_size, width // patch_size)
    record: dict[str, object] = {
        "condition": _condition_name(study_name, level),
        study["level_name"]: level,
        "input_shape": list(dataset.sample_shape),
        "patch_grid": list(grid),
        "token_count": channels * grid[0] * grid[1],
        "validation": {str(layer): validation[layer] for layer in EXIT_LAYERS},
        "validation_l8_relative_gap": _metric_gap(values, higher_is_better=spec.higher_is_better),
        "selected_exit_best_validation": selected_best,
        "selected_exit_earliest_within_tolerance": selected_tolerance,
        "selected_exit_compute": _measure_selected_compute(
            models[selected_tolerance],
            sample_shape=dataset.sample_shape,
            device=device,
            warmup_iterations=args.warmup_iterations,
            measure_iterations=args.measure_iterations,
        ),
    }
    return record


def _assert_full_information_control(
    *,
    study_name: str,
    record: dict[str, object],
    source_values: dict[int, float],
    spec: TaskSpec,
) -> None:
    full_level = max(STUDIES[study_name]["levels"])
    if int(record[STUDIES[study_name]["level_name"]]) != full_level:
        return
    observed = record["validation"]
    assert isinstance(observed, dict)
    mismatches = []
    for layer in EXIT_LAYERS:
        value = float(observed[str(layer)][spec.metric_name])
        if not math.isclose(value, source_values[layer], rel_tol=0.0, abs_tol=1.0e-6):
            mismatches.append(f"L{layer}: observed={value:.8f}, source={source_values[layer]:.8f}")
    if mismatches:
        raise RuntimeError(
            f"full-information control for {study_name} does not reproduce source validation metrics: "
            + "; ".join(mismatches)
        )


def _selection_artifact(
    *,
    source_path: Path,
    source: dict[str, Any],
    checkpoint: Path,
    manifest_path: Path,
    source_root: Path,
    config_path: Path,
    tolerance: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    manifest = _read_manifest(manifest_path, required_tasks=tuple(entry["task"] for entry in STUDIES.values()))
    config = init_jepa_config(str(config_path))
    provenance: dict[str, object] = {
        "source_metrics": _file_record(source_path, include_hash=True),
        "checkpoint": _file_record(checkpoint),
        "split_manifest": _file_record(manifest_path, include_hash=True),
        "config": _file_record(config_path, include_hash=True),
        "source_method": source["method"],
        "source_pretrain_seed": source.get("pretrain_seed"),
        "source_repository_root": str(source_root.resolve()),
        "source_head_files": {},
    }
    output: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "study": "input_information_early_exit_inference",
        "stage": "selection_complete",
        "test_evaluation": False,
        "provenance": provenance,
        "protocol": {
            "foundation_weights_updated": False,
            "downstream_head_weights_updated": False,
            "split_manifest_created": False,
            "selection_uses": "validation_only",
            "tolerance": tolerance,
            "candidate_exit_layers": list(EXIT_LAYERS),
            "antenna_policy": "fixed ordered prefix channels [0:C)",
            "temporal_policy": "leading patch-aligned processed-cache window [0:16*T_p)",
            "cost_measurement": "selected earliest-within-tolerance exit only",
            "device": str(device),
        },
        "studies": {},
    }
    for study_name, study in STUDIES.items():
        task = study["task"]
        dataset, spec = build_task_dataset(task, REPOSITORY_ROOT)
        _validate_task_compatibility(source, task, spec)
        split_record = manifest["tasks"][task]
        for split_name in ("validation", "test"):
            indices = [int(index) for index in split_record["indices"][split_name]]
            if not indices or min(indices) < 0 or max(indices) >= len(dataset):
                raise ValueError(f"invalid {split_name} indices for {task} in {manifest_path}")
        head_paths = _source_head_paths(source, task, source_root=source_root)
        provenance["source_head_files"][task] = {str(layer): _file_record(path) for layer, path in head_paths.items()}
        encoder = _new_encoder(config, checkpoint, device)
        conditions = []
        source_values = _source_metric_values(source, task, spec)
        for level in study["levels"]:
            record = _condition_record(
                study_name=study_name,
                level=level,
                base_dataset=dataset,
                split_record=split_record,
                spec=spec,
                head_paths=head_paths,
                encoder=encoder,
                device=device,
                tolerance=tolerance,
                args=args,
            )
            _assert_full_information_control(
                study_name=study_name,
                record=record,
                source_values=source_values,
                spec=spec,
            )
            conditions.append(record)
        output["studies"][study_name] = {
            "task": task,
            "task_spec": asdict(spec),
            "availability_mode": study["mode"],
            "levels": list(study["levels"]),
            "conditions": conditions,
        }
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return output


def _run_test(artifact_path: Path, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    artifact = _read_json(artifact_path)
    if artifact.get("schema_version") != SCHEMA_VERSION or artifact.get("stage") != "selection_complete":
        raise ValueError(f"test requires a completed selection artifact: {artifact_path}")
    if artifact.get("test_evaluation"):
        raise ValueError(f"test data was already evaluated for artifact: {artifact_path}")
    provenance = artifact.get("provenance", {})
    source_path = Path(provenance.get("source_metrics", {}).get("path", ""))
    source, checkpoint, manifest_path, source_root, tolerance = _validate_source_metrics(source_path)
    if not math.isclose(float(artifact.get("protocol", {}).get("tolerance", -1)), tolerance, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("selection artifact tolerance does not match the authoritative source metrics")
    config_path = Path(provenance.get("config", {}).get("path", ""))
    if not config_path.is_file():
        raise FileNotFoundError(f"selection artifact config is missing: {config_path}")
    manifest = _read_manifest(manifest_path, required_tasks=tuple(entry["task"] for entry in STUDIES.values()))
    config = init_jepa_config(str(config_path))

    for study_name, study in STUDIES.items():
        study_output = artifact.get("studies", {}).get(study_name)
        if not isinstance(study_output, dict):
            raise ValueError(f"selection artifact lacks study {study_name}")
        task = study["task"]
        dataset, spec = build_task_dataset(task, REPOSITORY_ROOT)
        _validate_task_compatibility(source, task, spec)
        split_record = manifest["tasks"][task]
        head_paths = _source_head_paths(source, task, source_root=source_root)
        encoder = _new_encoder(config, checkpoint, device)
        for condition in study_output.get("conditions", []):
            level = int(condition[study["level_name"]])
            available = InputAvailabilityDataset(dataset, mode=study["mode"], level=level)
            test_loader = _loader(
                task_subsets(available, split_record)["test"],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
            models = {layer: _load_head(encoder, exit_layer=layer, spec=spec, path=head_paths[layer], device=device) for layer in EXIT_LAYERS}
            test = {str(layer): _evaluate(models[layer], test_loader, spec, device) for layer in EXIT_LAYERS}
            selected = str(int(condition["selected_exit_earliest_within_tolerance"]))
            condition["test"] = test
            condition["selected_test_earliest_within_tolerance"] = test[selected]
        if device.type == "cuda":
            torch.cuda.empty_cache()
    artifact["stage"] = "test_complete"
    artifact["test_evaluation"] = True
    artifact["protocol"]["test_evaluation_mode"] = "post_selection_fixed_heads"
    return artifact


def main() -> int:
    args = _arguments()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers must be non-negative")
    if args.warmup_iterations < 1 or args.measure_iterations < 1:
        raise ValueError("warmup and measurement iterations must be positive")
    if args.stage == "select" and (args.source_metrics is None or args.output_root is None or args.artifact is not None):
        raise ValueError("select requires --source-metrics and --output-root, and does not accept --artifact")
    if args.stage == "test" and (args.artifact is None or args.source_metrics is not None or args.output_root is not None):
        raise ValueError("test requires --artifact only; source and output provenance come from that artifact")
    device = _device(args.device)
    torch.set_float32_matmul_precision("high")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    if args.stage == "select":
        source_path = _repository_path(args.source_metrics)
        source, checkpoint, manifest_path, source_root, tolerance = _validate_source_metrics(source_path)
        config_path = _repository_path(args.config) if args.config else REPOSITORY_ROOT / "configs" / f"{source['method']}.yml"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        output_path = _repository_path(args.output_root) / ARTIFACT_NAME
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite an existing study artifact: {output_path}")
        artifact = _selection_artifact(
            source_path=source_path,
            source=source,
            checkpoint=checkpoint,
            manifest_path=manifest_path,
            source_root=source_root,
            config_path=config_path,
            tolerance=tolerance,
            args=args,
            device=device,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))
        print(json.dumps({"artifact": str(output_path), "stage": "selection_complete"}, indent=2))
    else:
        artifact_path = _repository_path(args.artifact)
        artifact = _run_test(artifact_path, args, device)
        artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))
        print(json.dumps({"artifact": str(artifact_path), "stage": "test_complete"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
