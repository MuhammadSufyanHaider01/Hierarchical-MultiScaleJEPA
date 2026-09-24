#!/usr/bin/env python3
"""Paired P0/P2 anchor, validation-selection, and post-review test for factory-OFDMA data.

This is an explicitly out-of-domain stress-test runner. It never trains or updates
the encoders or frozen heads. Selection reads only the immutable synthetic
selection split; test is a separate invocation after review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation.exits import select_best_exit, select_earliest_within_tolerance  # noqa: E402
from phase5 import TaskSpec  # noqa: E402
from synthetic.compatibility import source_file, source_repository_root  # noqa: E402
from synthetic.dataset import SyntheticAvailabilityDataset  # noqa: E402
from synthetic.manifests import validate_realization_manifest  # noqa: E402
from synthetic.schema import load_task_profile, require_verified_semantics  # noqa: E402
from synthetic.statistics import bootstrap_interval  # noqa: E402
from synthetic_availability_exits import (  # noqa: E402
    EXIT_LAYERS,
    _bootstrap_metrics,
    _compute_metrics,
    _device,
    _load_exit_models,
    _measure,
    _new_encoder,
)
from datasets import ImageDataset  # noqa: E402


PROFILE_NAMES = {
    "nrpos-outdoor": "factory_ofdma_nrpos_paper_v3.yml",
    "interference": "factory_ofdma_interference_paper_v3.yml",
    "jamming": "factory_ofdma_jamming_paper_v3.yml",
}
SOURCE_METRICS = {
    "p0": ROOT / "runs/masking_study/time_n6/p0/seed_000/downstream/exit_metrics.json",
    "p2": ROOT / "runs/masking_study/time_n6/checkpoint_epoch029_six_tasks/p2/seed_000/downstream/exit_metrics.json",
}
SWEEP_LEVELS = {
    "antenna": (1, 2, 4, 6, 8, 10, 12, 16),
    "time": (2, 4, 6, 8, 10, 12, 14, 18, 22, 28, 36, 44, 56),
}
METHOD_NAMES = {"p0": "latentwave_ee", "p2": "hierarchical"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def source_packet(method: str, task: str) -> tuple[dict[str, Any], Path, Path, Path, dict[int, Path]]:
    metrics_path = SOURCE_METRICS[method]
    record = json.loads(metrics_path.read_text())
    if record.get("method") != method or float(record.get("protocol", {}).get("tolerance", -1)) != 0.05:
        raise ValueError(f"invalid {method.upper()} source packet or non-5% exit protocol")
    source_root = source_repository_root(metrics_path)
    checkpoint = source_file(str(record.get("checkpoint", "")), source_root)
    manifest = source_file(str(record.get("split_manifest", "")), source_root)
    if not checkpoint.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"{method.upper()} checkpoint or real-data split manifest is missing")
    split = json.loads(manifest.read_text())
    if split.get("schema_version") != 3 or task not in split.get("tasks", {}):
        raise ValueError(f"{method.upper()} source must have a six-task v3 split with {task}")
    if not isinstance(split["tasks"][task].get("indices"), dict):
        raise ValueError(f"{method.upper()} source split lacks indices for {task}")
    source_task = record.get("tasks", {}).get(task)
    if not isinstance(source_task, dict):
        raise ValueError(f"{method.upper()} metrics do not contain {task}")
    heads: dict[int, Path] = {}
    for layer in EXIT_LAYERS:
        exit_record = source_task.get("exits", {}).get(str(layer))
        if not isinstance(exit_record, dict):
            raise ValueError(f"{method.upper()} source is missing {task} L{layer}")
        head = source_file(str(exit_record.get("head_path", "")), source_root)
        if not head.is_file():
            raise FileNotFoundError(f"{method.upper()} {task} L{layer} head missing: {head}")
        heads[layer] = head
    config_value = record.get("config_path") or record.get("config")
    if isinstance(config_value, dict):
        config_value = config_value.get("path")
    config = source_file(str(config_value), source_root) if config_value else ROOT / "configs" / f"{method}.yml"
    if not config.is_file():
        raise FileNotFoundError(f"{method.upper()} encoder config is missing: {config}")
    return record, metrics_path, checkpoint, manifest, heads | {-1: config}


def profile_for(task: str):
    profile, config = load_task_profile(ROOT / "configs/sionna" / PROFILE_NAMES[task])
    require_verified_semantics(profile)  # approved assumption-based synthetic profiles are explicit
    return profile, config


def formal_data(task: str, data_root: Path):
    task_root = data_root / task
    master = task_root / "master.h5"
    manifest_path = task_root / "split_manifest.json"
    provenance_path = task_root / "formal_provenance.json"
    if not all(path.is_file() for path in (master, manifest_path, provenance_path)):
        raise FileNotFoundError(f"formal paper-v3 master, manifest, or provenance missing under {task_root}")
    manifest = json.loads(manifest_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    validate_realization_manifest(manifest)
    if manifest.get("task") != task or manifest.get("stage") != "formal_count_locked_before_selection":
        raise ValueError("formal synthetic manifest task/stage is invalid")
    stat = master.stat()
    record = provenance.get("master", {})
    if (record.get("sha256") != manifest.get("master_sha256")
            or int(record.get("size_bytes", -1)) != stat.st_size
            or int(record.get("modified_ns", -1)) != stat.st_mtime_ns):
        raise ValueError("formal master does not match immutable provenance")
    if int(provenance.get("count", -1)) != len(manifest["realization_ids"]):
        raise ValueError("formal master count does not match manifest")
    dataset = ImageDataset(
        master,
        sample_key="features",
        label_key="label",
        label_dtype=torch.float32 if task == "nrpos-outdoor" else torch.long,
    )
    with h5py.File(master, "r") as handle:
        ids = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["realization_id"][:]]
        if handle.attrs.get("task", "") != task or handle.attrs.get("split_kind", "") != "formal":
            raise ValueError("formal HDF5 task or split kind is invalid")
        if handle.attrs.get("semantic_basis", "") != "assumption_based_synthetic_domain":
            raise ValueError("expected assumption-based synthetic semantics metadata")
        if int(handle.attrs.get("assumptions_approved", 0)) != 1:
            raise ValueError("synthetic task assumptions are not marked approved")
        feature_mode = handle.attrs.get("feature_mode", "")
        if isinstance(feature_mode, bytes):
            feature_mode = feature_mode.decode()
    if ids != manifest["realization_ids"] or len(dataset) != len(ids):
        raise ValueError("HDF5 row order differs from the immutable realization manifest")
    if dataset.sample_shape != (16, 224, 896):
        raise ValueError(f"unexpected full-information shape: {dataset.sample_shape}")
    profile, _ = profile_for(task)
    if feature_mode != profile.feature_mode:
        raise ValueError("HDF5 feature transform does not match its task profile")
    return dataset, manifest, manifest_path, provenance, master


def task_spec(task: str, source: dict[str, Any], profile) -> TaskSpec:
    raw = source["tasks"][task]["spec"]
    if (raw.get("target_type") != profile.target_type
            or int(raw.get("num_outputs", -1)) != profile.num_outputs
            or raw.get("metric_name") != profile.metric_name
            or bool(raw.get("higher_is_better")) != profile.higher_is_better):
        raise ValueError(f"synthetic profile is incompatible with {task} frozen head")
    # Position targets are factory scene coordinates, not source NR coordinates.
    return TaskSpec(
        name=task, path="synthetic-factory-ofdma", target_type=profile.target_type,
        num_outputs=profile.num_outputs, higher_is_better=profile.higher_is_better,
        metric_name=profile.metric_name, sample_shape=(16, 224, 896),
        coord_min=list(profile.coord_min) if profile.coord_min else None,
        coord_max=list(profile.coord_max) if profile.coord_max else None,
    )


def load_models(method: str, task: str, source: dict[str, Any], checkpoint: Path,
                config: Path, heads: dict[int, Path], spec: TaskSpec, device: torch.device,
                backend: str, precision: str):
    from configs import init_jepa_config
    from models import Encoder, FrozenEncoderExit, load_context_encoder_checkpoint
    from models.components import set_attention_backend

    cfg = init_jepa_config(str(config))
    encoder = Encoder(
        img_size=tuple(cfg.encoder.img_size), patch_size=cfg.encoder.patch_size,
        embed_dim=cfg.encoder.embed_dim, depth=cfg.encoder.depth,
        num_heads=cfg.encoder.num_heads,
    ).to(device)
    load_context_encoder_checkpoint(encoder, checkpoint)
    set_attention_backend(encoder, backend)
    encoder.eval()
    models = _load_exit_models(encoder, heads, spec, device)
    for model in models.values():
        if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("paired evaluation requires frozen, gradient-free model parameters")
    return models


def view_for(dataset, profile, sweep: str, level: int):
    mode = "antenna" if sweep == "antenna" else "time"
    return SyntheticAvailabilityDataset(
        dataset, mode=mode, level=level, temporal_axis=profile.temporal_axis,
        reference_channels=profile.reference_channels,
        reference_time_patches=profile.reference_time_patches,
        source_sample_shape=dataset.sample_shape, max_channels=16,
        array_subset_policy=profile.array_subset_policy,
    )


def rows_for_split(manifest: dict[str, Any], name: str) -> list[int]:
    index = {realization_id: row for row, realization_id in enumerate(manifest["realization_ids"])}
    values = manifest.get("splits", {}).get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"synthetic manifest has no {name} IDs")
    return [index[value] for value in values]


def loader(dataset, indices: list[int], args, device):
    return DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                      shuffle=False, num_workers=args.num_workers,
                      pin_memory=device.type == "cuda")


def position_anchor(spec: TaskSpec, labels: np.ndarray, errors: np.ndarray):
    lo = np.asarray(spec.coord_min, dtype=np.float64)
    hi = np.asarray(spec.coord_max, dtype=np.float64)
    targets = (labels.astype(np.float64) + 1.0) * 0.5 * (hi - lo) + lo
    centroid = targets.mean(axis=0)
    centroid_error = np.linalg.norm(targets - centroid, axis=1)
    delta = centroid_error - errors
    return {
        "centroid_mean_distance_error_m": float(centroid_error.mean()),
        "model_mean_distance_error_m": float(errors.mean()),
        "centroid_minus_model_ci_m": bootstrap_interval(delta, statistic=np.mean, seed=2601, resamples=2000),
        "passes_centroid_gate": bool(bootstrap_interval(delta, statistic=np.mean, seed=2601, resamples=2000)["lower"] > 0.0),
    }

def classification_anchor_gate(num_outputs: int, confidence_intervals: dict[str, Any]) -> dict[str, Any]:
    """Apply the planned macro-accuracy-over-chance calibration gate."""
    chance = 1.0 / int(num_outputs)
    macro_ci = confidence_intervals["macro_accuracy"]
    return {
        "chance_macro_accuracy": chance,
        "macro_accuracy_ci": macro_ci,
        "passes_chance_gate": bool(float(macro_ci["lower"]) > chance),
    }


def anchor_task(args, task: str, sweep: str, dataset, manifest, manifest_path,
                data_provenance, master: Path, profile, device):
    level = max(SWEEP_LEVELS[sweep]) if sweep == "antenna" else max(SWEEP_LEVELS["time"])
    view = view_for(dataset, profile, sweep, level)
    indices = rows_for_split(manifest, "calibration")
    per_method = {}
    source_split_digests = set()
    for method in ("p0", "p2"):
        source, source_path, checkpoint, source_manifest, mixed_heads = source_packet(method, task)
        source_split_digests.add(sha256(source_manifest))
        config = mixed_heads.pop(-1)
        heads = mixed_heads
        spec = task_spec(task, source, profile)
        models = load_models(method, task, source, checkpoint, config, heads, spec,
                             device, args.attention_backend, args.matmul_precision)
        metric, labels, values = _compute_metrics(models[8], loader(view, indices, args, device), spec, device)
        ci = _bootstrap_metrics(spec, labels, values, seed=3100 + (0 if method == "p0" else 1))
        if profile.target_type == "position":
            gate = position_anchor(spec, labels, values)
        else:
            gate = classification_anchor_gate(profile.num_outputs, ci)
        real_l8 = source["tasks"][task]["exits"]["8"]["validation"]
        per_method[method] = {
            "metrics": metric, "confidence_intervals": ci, "anchor_gate": gate,
            "real_source_l8_validation": real_l8,
            "real_vs_synthetic_gap": {
                profile.metric_name: metric[profile.metric_name] - real_l8[profile.metric_name]
                if profile.higher_is_better else real_l8[profile.metric_name] - metric[profile.metric_name]
            },
            "provenance": source_provenance(method, source_path, checkpoint, config,
                                            source_manifest, heads),
        }
    if len(source_split_digests) != 1:
        raise ValueError("P0 and P2 do not share the same authoritative real-data v3 split manifest")
    return {
        "schema_version": 1, "stage": "formal_calibration_anchor_complete",
        "scenario_id": data_provenance["scenario_id"], "task": task, "sweep": sweep,
        "split": "calibration_only", "resource_level": level,
        "input_shape": list(view.sample_shape), "realization_count": len(indices),
        "formal_master": file_provenance(master, data_provenance["master"]["sha256"]),
        "formal_manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
        "formal_generation_anchor_gates_pass": data_provenance.get("anchor_gates_pass"),
        "scope": "zero-shot synthetic factory-domain stress test; not a recreation of measured task data",
        "methods": per_method,
    }


def file_provenance(path: Path, digest: str | None = None):
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size,
            "modified_ns": stat.st_mtime_ns, "sha256": digest or sha256(path)}


def source_provenance(method: str, source_path: Path, checkpoint: Path, config: Path,
                      manifest: Path, heads: dict[int, Path]):
    return {
        "method": method,
        "metrics": file_provenance(source_path),
        "checkpoint": file_provenance(checkpoint),
        "config": file_provenance(config),
        "real_split_manifest": file_provenance(manifest),
        "heads": {str(layer): file_provenance(path) for layer, path in heads.items()},
    }


def select_task(args, task: str, sweep: str, method: str, dataset, manifest, manifest_path,
                data_provenance, master: Path, profile, device):
    if not args.anchor:
        raise ValueError("selection requires the formal calibration anchor artifact")
    anchor_path = resolve(args.anchor)
    anchor = json.loads(anchor_path.read_text())
    if anchor.get("stage") != "formal_calibration_anchor_complete" or anchor.get("task") != task or anchor.get("sweep") != sweep:
        raise ValueError("formal calibration anchor task/sweep/stage mismatch")
    method_anchor = anchor["methods"][method]
    gate = method_anchor["anchor_gate"]
    passed = bool(gate.get("passes_chance_gate" if profile.target_type == "classification" else "passes_centroid_gate"))
    if not passed and not args.allow_uncalibrated_stress_test:
        raise RuntimeError("formal calibration gate failed; rerun with --allow-uncalibrated-stress-test only for OOD stress-test results")
    source, source_path, checkpoint, source_manifest, mixed_heads = source_packet(method, task)
    config = mixed_heads.pop(-1)
    heads = mixed_heads
    recorded_source = method_anchor["provenance"]
    if (recorded_source["metrics"]["sha256"] != sha256(source_path)
            or recorded_source["checkpoint"]["sha256"] != sha256(checkpoint)
            or recorded_source["config"]["sha256"] != sha256(config)
            or recorded_source["real_split_manifest"]["sha256"] != sha256(source_manifest)):
        raise ValueError("formal anchor used a different source model packet")
    for layer, path in heads.items():
        if recorded_source["heads"][str(layer)]["sha256"] != sha256(path):
            raise ValueError(f"formal anchor used a different {method.upper()} L{layer} head")
    if (anchor["formal_manifest"]["sha256"] != sha256(manifest_path)
            or anchor["formal_master"]["sha256"] != data_provenance["master"]["sha256"]):
        raise ValueError("formal anchor does not match this immutable synthetic master/manifest")
    spec = task_spec(task, source, profile)
    models = load_models(method, task, source, checkpoint, config, heads, spec,
                         device, args.attention_backend, args.matmul_precision)
    selection_indices = rows_for_split(manifest, "selection")
    conditions = []
    reference_labels = None
    for level in SWEEP_LEVELS[sweep]:
        view = view_for(dataset, profile, sweep, level)
        per_exit = {}
        raw = {}
        for layer, model in models.items():
            metrics, labels, values = _compute_metrics(model, loader(view, selection_indices, args, device), spec, device)
            if reference_labels is None:
                reference_labels = labels
            elif not np.array_equal(reference_labels, labels):
                raise ValueError("availability condition changed labels or realization ordering")
            per_exit[str(layer)] = metrics
            raw[layer] = (labels, values)
        values_by_layer = {layer: per_exit[str(layer)][profile.metric_name] for layer in EXIT_LAYERS}
        best = select_best_exit(values_by_layer, higher_is_better=profile.higher_is_better)
        selected = select_earliest_within_tolerance(values_by_layer,
                                                    higher_is_better=profile.higher_is_better,
                                                    tolerance=0.05)
        grid = [size // 16 for size in view.sample_shape[1:]]
        conditions.append({
            "resource_level": level,
            "input_shape": list(view.sample_shape),
            "patch_grid": grid,
            "token_count": int(view.sample_shape[0] * grid[0] * grid[1]),
            "antenna_indices": list(view.antenna_indices),
            "validation_by_exit": per_exit,
            "validation_l8_relative_gap": {
                str(layer): (per_exit[str(layer)][profile.metric_name] - per_exit["8"][profile.metric_name]
                             if profile.higher_is_better else
                             per_exit["8"][profile.metric_name] - per_exit[str(layer)][profile.metric_name])
                for layer in (2, 4, 6)
            },
            "best_exit_validation": best,
            "selected_exit_earliest_within_tolerance": selected,
            "selected_exit_compute": _measure(models[selected], tuple(view.sample_shape), device,
                                                args.warmup_iterations, args.measure_iterations),
        })
    return {
        "schema_version": 1, "study": "factory_ofdma_paper_v3_synthetic_availability",
        "stage": "selection_complete", "task": task, "sweep": sweep, "method": method,
        "display_method": "LatentWave + EE" if method == "p0" else "Proposed hierarchical approach",
        "test_evaluation": False, "task_spec": asdict(spec),
        "levels": list(SWEEP_LEVELS[sweep]), "conditions": conditions,
        "provenance": {
            "source": source_provenance(method, source_path, checkpoint, config, source_manifest, heads),
            "formal_master": file_provenance(master, data_provenance["master"]["sha256"]),
            "formal_manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
            "formal_provenance": {"path": str((master.parent / "formal_provenance.json").resolve()),
                                  "sha256": sha256(master.parent / "formal_provenance.json")},
            "sample_plan": data_provenance["sample_plan"],
            "anchor_artifact": file_provenance(anchor_path),
            "profile": file_provenance(ROOT / "configs/sionna" / PROFILE_NAMES[task]),
        },
        "protocol": {
            "selection_uses": "synthetic_selection_split_only", "split": "realization-level 20/20/60",
            "tolerance": 0.05, "candidate_exit_layers": list(EXIT_LAYERS),
            "attention_backend": args.attention_backend, "matmul_precision": args.matmul_precision,
            "device": str(device), "foundation_weights_updated": False,
            "head_weights_updated": False, "formal_calibration_gate_passed": passed,
            "uncalibrated_stress_test_override": bool(not passed and args.allow_uncalibrated_stress_test),
            "scientific_scope": "synthetic factory-domain OOD stress test; not source-task accuracy or measured-data reproduction",
            "semantics": "assumption-based labels and feature adapter; see task profile and per-realization metadata",
        },
        "anchor_metrics": method_anchor,
        "real_source_l8_validation": method_anchor["real_source_l8_validation"],
    }


def test_task(args, selection_path: Path, dataset, manifest, profile, device):
    selection = json.loads(selection_path.read_text())
    if selection.get("stage") != "selection_complete" or selection.get("test_evaluation") is not False:
        raise ValueError("test requires completed validation-only selection")
    if selection.get("task") != args.task or selection.get("sweep") != args.sweep:
        raise ValueError("test request does not match selection artifact")
    if args.method != selection.get("method"):
        raise ValueError("test method does not match the method encoded in selection artifact")
    sidecar = selection_path.with_suffix(selection_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().strip() != sha256(selection_path):
        raise ValueError("selection artifact integrity digest mismatch")
    method = selection["method"]
    source, source_path, checkpoint, source_manifest, mixed_heads = source_packet(method, args.task)
    source_provenance_record = selection["provenance"]["source"]
    if (sha256(source_path) != source_provenance_record["metrics"]["sha256"]
            or sha256(checkpoint) != source_provenance_record["checkpoint"]["sha256"]
            or sha256(source_manifest) != source_provenance_record["real_split_manifest"]["sha256"]):
        raise ValueError("source packet changed after validation selection")
    config = mixed_heads.pop(-1)
    heads = mixed_heads
    spec = task_spec(args.task, source, profile)
    models = load_models(method, args.task, source, checkpoint, config, heads, spec,
                         device, args.attention_backend, args.matmul_precision)
    test_indices = rows_for_split(manifest, "test")
    results = []
    for condition in selection["conditions"]:
        level = int(condition["resource_level"])
        view = view_for(dataset, profile, args.sweep, level)
        per_exit = {}
        raw = {}
        for layer, model in models.items():
            metrics, labels, values = _compute_metrics(model, loader(view, test_indices, args, device), spec, device)
            per_exit[str(layer)] = {
                "metrics": metrics,
                "confidence_intervals": _bootstrap_metrics(spec, labels, values,
                                                              seed=7100 + level * 10 + layer),
            }
            raw[layer] = (labels, values)
        selected = int(condition["selected_exit_earliest_within_tolerance"])
        results.append({"resource_level": level, "fixed_exits": per_exit,
                        "selected_exit": selected, "selected_metrics": per_exit[str(selected)],
                        "sample_count": len(test_indices)})
    return {"schema_version": 1, "stage": "test_complete", "task": args.task,
            "sweep": args.sweep, "method": method, "test_evaluation": True,
            "selection_artifact": str(selection_path.resolve()),
            "selection_sha256": sha256(selection_path), "split": "synthetic_test_only",
            "formal_manifest_sha256": sha256(args.data_root / args.task / "split_manifest.json"),
            "conditions": results,
            "scope": selection["protocol"]["scientific_scope"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("anchor", "select", "test"))
    parser.add_argument("--task", choices=tuple(PROFILE_NAMES), required=True)
    parser.add_argument("--sweep", choices=("antenna", "time"), required=True)
    parser.add_argument("--method", choices=("p0", "p2"), help="required for select/test; anchor always evaluates both")
    parser.add_argument("--data-root", type=Path, default=Path("data/synthetic/factory_ofdma_paper_v3"))
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-uncalibrated-stress-test", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--measure-iterations", type=int, default=100)
    parser.add_argument("--matmul-precision", choices=("high", "highest"), default="high")
    parser.add_argument("--attention-backend", choices=("math", "fused"), default="math")
    args = parser.parse_args()
    if args.stage != "anchor" and args.method not in ("p0", "p2"):
        parser.error("select/test require --method p0 or p2")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch size must be positive and workers non-negative")
    args.data_root = resolve(args.data_root)
    device = _device(args.device)
    torch.set_float32_matmul_precision(args.matmul_precision)
    dataset, manifest, manifest_path, data_provenance, master = formal_data(args.task, args.data_root)
    profile, _ = profile_for(args.task)
    if args.stage == "anchor":
        artifact = anchor_task(args, args.task, args.sweep, dataset, manifest,
                               manifest_path, data_provenance, master, profile, device)
        output = args.output or Path("runs/synthetic_availability/factory_ofdma_paper_v3_formal") / args.task / args.sweep / "calibration_anchor.json"
    elif args.stage == "select":
        artifact = select_task(args, args.task, args.sweep, args.method, dataset,
                               manifest, manifest_path, data_provenance, master, profile, device)
        method_name = METHOD_NAMES[args.method]
        folder = "vary_antenna" if args.sweep == "antenna" else "vary_time_samples"
        output = args.output or Path("experiments/synthetic_data") / folder / args.task / method_name / "input_information_exit_metrics.json"
    else:
        if args.selection is None:
            parser.error("test requires --selection")
        selection = resolve(args.selection)
        artifact = test_task(args, selection, dataset, manifest, profile, device)
        method_name = METHOD_NAMES[args.method]
        folder = "vary_antenna" if args.sweep == "antenna" else "vary_time_samples"
        output = args.output or Path("experiments/synthetic_data") / folder / args.task / method_name / "synthetic_test_metrics.json"
    output = resolve(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    output.write_text(payload)
    if args.stage == "select":
        output.with_suffix(output.suffix + ".sha256").write_text(sha256(output) + "\n")
    print(json.dumps({"stage": args.stage, "task": args.task, "sweep": args.sweep,
                      "method": args.method or "p0+p2", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
