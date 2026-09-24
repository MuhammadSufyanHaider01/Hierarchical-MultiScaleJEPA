#!/usr/bin/env python3
"""Run a paired P0/P2 L8 full-information anchor on a factory pilot split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("WANDB_KEY", "offline")

from configs import init_jepa_config  # noqa: E402
from datasets import ImageDataset  # noqa: E402
from models import Encoder, FrozenEncoderExit, load_context_encoder_checkpoint  # noqa: E402
from phase5 import TaskSpec  # noqa: E402
from synthetic_source_packets import SOURCE_PACKETS, audit_source_packets  # noqa: E402
from synthetic.dataset import SyntheticAvailabilityDataset  # noqa: E402
from synthetic.manifests import validate_realization_manifest  # noqa: E402
from synthetic.schema import load_task_profile, require_verified_semantics  # noqa: E402
from synthetic.statistics import bootstrap_interval, classification_anchor_passes, position_anchor_passes  # noqa: E402


EXIT_LAYER = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile_path(task: str, scenario_version: str) -> Path:
    name = {"nrpos-outdoor": "nrpos", "interference": "interference", "jamming": "jamming"}[task]
    suffix = f"_{scenario_version}" if scenario_version in ("paper_v2", "paper_v3") else ("_v2" if scenario_version == "v2" and task == "nrpos-outdoor" else "")
    return ROOT / "configs/sionna" / f"factory_ofdma_{name}{suffix}.yml"


def _task_spec(source: dict, task: str, profile, sample_shape: tuple[int, ...]) -> TaskSpec:
    raw = source["tasks"][task]["spec"]
    return TaskSpec(
        name=task,
        path=str(raw.get("path", "synthetic")),
        target_type=profile.target_type,
        num_outputs=profile.num_outputs,
        higher_is_better=profile.higher_is_better,
        metric_name=profile.metric_name,
        sample_shape=sample_shape,
        coord_min=list(profile.coord_min) if profile.coord_min else None,
        coord_max=list(profile.coord_max) if profile.coord_max else None,
    )


def _evaluate(model, loader, spec: TaskSpec, device: torch.device):
    model.eval()
    labels_out, values_out = [], []
    coord_min = torch.tensor(spec.coord_min, device=device) if spec.coord_min else None
    coord_max = torch.tensor(spec.coord_max, device=device) if spec.coord_max else None
    with torch.inference_mode():
        for samples, labels in loader:
            samples = samples.to(device=device, dtype=torch.float32, non_blocking=True)
            logits = model(samples)
            if spec.target_type == "classification":
                prediction = logits.argmax(1).cpu().numpy()
                labels_out.append(labels.numpy().astype(np.int64, copy=False))
                values_out.append(prediction.astype(np.int64, copy=False))
            else:
                assert coord_min is not None and coord_max is not None
                prediction = (logits + 1.0) * 0.5 * (coord_max - coord_min) + coord_min
                target = (labels.to(device=device, dtype=torch.float32) + 1.0) * 0.5 * (coord_max - coord_min) + coord_min
                errors = torch.linalg.vector_norm(prediction - target, dim=-1)
                labels_out.append(labels.numpy().astype(np.float32, copy=False))
                values_out.append(errors.cpu().numpy())
    return np.concatenate(labels_out), np.concatenate(values_out)


def _make_model(method: str, task: str, profile, source: dict, source_record: dict, device):
    metrics_path = SOURCE_PACKETS[method]
    source_root = ROOT
    data = json.loads(metrics_path.read_text())
    checkpoint = Path(data["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = source_root / checkpoint
    config_value = data.get("config_path") or data.get("config")
    if isinstance(config_value, dict):
        config_value = config_value.get("path")
    if config_value:
        config_path = Path(config_value)
        if not config_path.is_absolute():
            config_path = source_root / config_path
    else:
        config_path = ROOT / "configs" / ("p0.yml" if method == "p0" else "p2.yml")
    config = init_jepa_config(str(config_path))
    encoder = Encoder(
        img_size=tuple(config.encoder.img_size),
        patch_size=config.encoder.patch_size,
        embed_dim=config.encoder.embed_dim,
        depth=config.encoder.depth,
        num_heads=config.encoder.num_heads,
    ).to(device)
    load_context_encoder_checkpoint(encoder, checkpoint)
    head_path = Path(source_record["heads"]["8"]["path"])
    head_checkpoint = torch.load(head_path, map_location=device, weights_only=False)
    if not isinstance(head_checkpoint, dict) or "head" not in head_checkpoint:
        raise KeyError(f"head checkpoint lacks head state: {head_path}")
    model = FrozenEncoderExit(encoder, exit_layer=EXIT_LAYER, num_outputs=profile.num_outputs).to(device)
    model.head.load_state_dict(head_checkpoint["head"], strict=True)
    model.eval()
    return model, checkpoint, config_path, head_path


def _metric_and_gate(spec: TaskSpec, labels: np.ndarray, values: np.ndarray):
    if spec.target_type == "classification":
        correct = (labels == values).astype(np.float64)
        macro = {}
        for class_id in np.unique(labels):
            mask = labels == class_id
            macro[int(class_id)] = float(correct[mask].mean())
        accuracy = float(correct.mean())
        macro_accuracy = float(np.mean(list(macro.values())))
        interval = bootstrap_interval(
            correct,
            statistic=np.mean,
            strata=labels,
            seed=2026,
            resamples=1000,
        )
        passes = classification_anchor_passes(float(interval["lower"]), spec.num_outputs)
        return {"accuracy": accuracy, "macro_accuracy": macro_accuracy}, {
            "criterion": "95% stratified bootstrap lower bound of accuracy > 1/K",
            "class_accuracy": {str(k): v for k, v in macro.items()},
            "chance_accuracy": 1.0 / spec.num_outputs,
            "accuracy_ci": interval,
            "passes": passes,
        }
    lo = np.asarray(spec.coord_min, dtype=np.float64)
    hi = np.asarray(spec.coord_max, dtype=np.float64)
    targets = (labels.astype(np.float64) + 1.0) * 0.5 * (hi - lo) + lo
    centroid = targets.mean(0)
    centroid_error = np.linalg.norm(targets - centroid, axis=1)
    improvement = bootstrap_interval(
        centroid_error - values,
        statistic=np.mean,
        seed=2026,
        resamples=1000,
    )
    model_ci = bootstrap_interval(values, statistic=np.mean, seed=2027, resamples=1000)
    passes = position_anchor_passes(float(improvement["lower"]))
    return {"mean_distance_error": float(values.mean()), "median_distance_error": float(np.median(values))}, {
        "criterion": "95% paired bootstrap lower bound of centroid error minus model error > 0 m",
        "centroid_mean_distance_error_m": float(centroid_error.mean()),
        "model_error_ci_m": model_ci,
        "centroid_minus_model_error_ci_m": improvement,
        "passes": passes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("nrpos-outdoor", "interference", "jamming"), required=True)
    parser.add_argument("--sweep", choices=("antenna", "time"), required=True)
    parser.add_argument("--scenario-version", choices=("v1", "v2", "paper_v2", "paper_v3"), default="v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("runs/synthetic_availability/factory_ofdma_v1"))
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    if args.scenario_version == "v2" and args.task != "nrpos-outdoor":
        raise ValueError("factory-OFDMA v2 currently revises NR positioning only")
    profile_path = _profile_path(args.task, args.scenario_version)
    profile, _ = load_task_profile(profile_path)
    require_verified_semantics(profile)
    source_records = audit_source_packets(args.task, profile)
    data_root = ROOT / f"data/synthetic/factory_ofdma_{args.scenario_version}" / args.task
    master_path = data_root / "pilot.h5"
    manifest_path = data_root / "pilot_split_manifest.json"
    provenance_path = data_root / "pilot_provenance.json"
    if not all(path.is_file() for path in (master_path, manifest_path, provenance_path)):
        raise FileNotFoundError(f"pilot master, split manifest, or provenance missing under {data_root}")
    manifest = json.loads(manifest_path.read_text())
    validate_realization_manifest(manifest)
    ids = manifest["realization_ids"]
    calibration_ids = manifest["splits"]["calibration"]
    row_for_id = {value: i for i, value in enumerate(ids)}
    calibration_indices = [row_for_id[value] for value in calibration_ids]
    dataset = ImageDataset(
        master_path,
        sample_key="features",
        label_key="label",
        label_dtype=torch.float32 if profile.target_type == "position" else torch.long,
    )
    with h5py.File(master_path, "r") as h5:
        if h5.attrs.get("split_kind", "") != "pilot":
            raise ValueError("pilot anchor only accepts split_kind=pilot masters")
        if h5["realization_id"].shape[0] != len(ids):
            raise ValueError("pilot HDF5 row order differs from the split manifest")

    if args.sweep == "antenna":
        level = max(profile.antenna_levels)
        mode = "antenna"
    else:
        level = max(profile.temporal_levels)
        mode = "time"
    view = SyntheticAvailabilityDataset(
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
    loader = DataLoader(
        Subset(view, calibration_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output_path = output_root / args.task / args.sweep / "pilot_anchors.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite pilot anchor: {output_path}")
    records = {}
    for method in ("p0", "p2"):
        source_path = SOURCE_PACKETS[method]
        source = json.loads(source_path.read_text())
        record = source_records[method]
        model, checkpoint, config_path, head_path = _make_model(method, args.task, profile, source, record, device)
        spec = _task_spec(source, args.task, profile, dataset.sample_shape)
        labels, values = _evaluate(model, loader, spec, device)
        metrics, gate = _metric_and_gate(spec, labels, values)
        if spec.target_type == "classification":
            paired = np.stack([labels.astype(np.int64), values.astype(np.int64)], axis=1)
            def macro_accuracy(rows):
                recalls = [np.mean(rows[rows[:, 0] == class_id, 1] == class_id)
                           for class_id in range(spec.num_outputs) if np.any(rows[:, 0] == class_id)]
                return float(np.mean(recalls))
            precision = bootstrap_interval(paired, statistic=macro_accuracy, strata=labels, seed=2028, resamples=1000)
        else:
            precision = gate["model_error_ci_m"]
        source_metric = float(source["tasks"][args.task]["exits"]["8"]["validation"][spec.metric_name])
        records[method] = {
            "source_metrics": str(source_path.resolve()),
            "source_metrics_sha256": _sha256(source_path),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "source_config": str(config_path.resolve()),
            "source_config_sha256": _sha256(config_path),
            "l8_head": str(head_path.resolve()),
            "l8_head_sha256": _sha256(head_path),
            "synthetic_metrics": metrics,
            "anchor_gate": gate,
            "precision_metric_ci": precision,
            "real_data_l8_validation_metric": source_metric,
            "synthetic_minus_real_metric": float(metrics[spec.metric_name] - source_metric),
        }
    conservative_method = max(records, key=lambda method: records[method]["precision_metric_ci"]["half_width"])
    conservative_precision = {**records[conservative_method]["precision_metric_ci"], "conservative_method": conservative_method}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1,
        "study": f"factory_ofdma_mobile_array_{args.scenario_version}",
        "stage": "pilot_precision_estimate",
        "formal_generation_allowed": False,
        "task": args.task,
        "sweep": args.sweep,
        "full_information_condition": {
            "antenna": {"channels": level, "temporal_patches": profile.reference_time_patches},
            "time": {"channels": profile.reference_channels, "temporal_patches": level},
        }[args.sweep],
        "input_shape": list(view.sample_shape),
        "sample_count": len(calibration_indices),
        "pilot_master_count": len(dataset),
        "realization_ids_sha256": hashlib.sha256(
            json.dumps(calibration_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "pilot_master": {"path": str(master_path.resolve()), "sha256": _sha256(master_path)},
        "pilot_manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256(manifest_path)},
        "task_profile": {"path": str(profile_path.resolve()), "sha256": _sha256(profile_path)},
        "device": str(device),
        "methods": records,
        "precision_metric_ci": conservative_precision,
        "provenance": {
            "synthetic_master": {"path": str(master_path.resolve()), "size_bytes": master_path.stat().st_size, "modified_ns": master_path.stat().st_mtime_ns},
            "synthetic_manifest_sha256": _sha256(manifest_path),
        },
        "all_anchor_gates_pass": all(item["anchor_gate"]["passes"] for item in records.values()),
        "note": "Pilot anchor only; it does not perform validation exit selection or held-out test evaluation.",
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(output_path), "all_anchor_gates_pass": artifact["all_anchor_gates_pass"], "methods": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
