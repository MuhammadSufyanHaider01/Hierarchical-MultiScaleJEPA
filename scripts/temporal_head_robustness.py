#!/usr/bin/env python3
"""Evaluate matched frozen heads across fixed-grid input-prefix conditions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("WANDB_KEY", "offline")

from configs import configure_method_variant, init_jepa_config  # noqa: E402
from models import build_encoder, load_context_encoder_checkpoint  # noqa: E402
from phase5 import build_task_dataset, _labels  # noqa: E402
from representation_trajectories import (  # noqa: E402
    LAYERS, METHODS, TEMPORAL_LEVELS, IndexedSamples, load_task_split,
)
from temporal_head_robustness import (  # noqa: E402
    COMPARISONS, HEAD_SEEDS, METHOD_LABELS, assert_full_input_logit_parity,
    build_metric_tables, hierarchical_paired_bootstrap, join_geometry_metrics,
    load_exit_heads, read_json, save_json, sha256_file, sha256_indices,
    validate_head_packet, write_csv, write_figures, write_summary,
    evaluate_temporal_prefixes,
)

TASKS = ("interference", "human-activity", "rfs", "jamming")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="interference")
    parser.add_argument("--trajectory-provenance", type=Path)
    parser.add_argument("--trajectory-test-features", type=Path)
    parser.add_argument("--head-root", type=Path)
    parser.add_argument("--head-seeds", type=int, nargs="+", default=list(HEAD_SEEDS))
    parser.add_argument("--inference-repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2025)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-root", type=Path)
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _load_geometry_rows(provenance: dict, provenance_path: Path) -> tuple[Path, list[dict[str, str]]]:
    artifacts = provenance.get("artifacts", {})
    candidate = Path(artifacts.get("metrics", provenance_path.parent / "metrics.csv"))
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if not candidate.is_file():
        candidate = provenance_path.parent / "metrics.csv"
    if not candidate.is_file():
        raise FileNotFoundError(f"trajectory geometry metrics are missing: {candidate}")
    return candidate.resolve(), _read_csv(candidate)


def _prepare_inputs(args: argparse.Namespace):
    base = ROOT / "runs/representation_trajectories/multi_block_n6" / args.task / "seed_000"
    provenance_path = (args.trajectory_provenance or base / "test/provenance.json").resolve()
    features_path = (args.trajectory_test_features or base / "test/features.npz").resolve()
    provenance = read_json(provenance_path)
    if provenance.get("task") != args.task or provenance.get("split") != "test":
        raise ValueError(f"trajectory provenance must describe the {args.task} test run")
    if set(provenance.get("checkpoints", {})) != set(METHODS):
        raise ValueError(f"trajectory provenance must contain methods {METHODS}")
    split_path = _resolve(provenance["split_manifest"])
    if sha256_file(split_path) != provenance.get("split_manifest_sha256"):
        raise ValueError("trajectory split manifest hash changed")
    dataset, spec = build_task_dataset(args.task, ROOT)
    if spec.target_type != "classification":
        raise ValueError(f"{args.task} is not a classification")
    num_classes = int(spec.num_outputs)
    labels_by_index = _labels(dataset)
    if labels_by_index is None:
        raise ValueError(f"Phase-5 dataset did not expose canonical labels for {args.task}")
    split_indices, mapped_labels = load_task_split(split_path, args.task, np.asarray(labels_by_index), num_classes)
    indices = np.asarray(split_indices["test"], dtype=np.int64)
    labels = mapped_labels[indices]
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(f"test split lacks one or more {args.task} classes: {counts.tolist()}")
    if list(spec.sample_shape) != provenance["data"]["sample_shape"]:
        raise ValueError(f"{args.task} cache geometry differs from trajectory provenance")
    if sha256_indices(indices) != provenance.get("split_indices_sha256"):
        raise ValueError("persisted test indices differ from trajectory provenance")
    if not features_path.is_file():
        raise FileNotFoundError(f"trajectory test feature cache is missing: {features_path}")
    with np.load(features_path, allow_pickle=False) as values:
        cached = {method: np.asarray(values[method]) for method in METHODS}
        cached_labels = np.asarray(values["labels"], dtype=np.int64)
        cached_indices = np.asarray(values["indices"], dtype=np.int64)
    if not np.array_equal(cached_labels, labels) or not np.array_equal(cached_indices, indices):
        raise ValueError("trajectory features do not align to canonical test samples and labels")
    expected = (len(indices), len(TEMPORAL_LEVELS), len(LAYERS), 256)
    for method in METHODS:
        if cached[method].shape != expected or not np.isfinite(cached[method]).all():
            raise ValueError(f"{method} trajectory cache must have finite shape {expected}")
        expected_hash = provenance["checkpoints"][method]["sha256"]
        checkpoint_path = Path(provenance["checkpoints"][method]["path"])
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != expected_hash:
            raise ValueError(f"{method} trajectory checkpoint missing or changed")
    data_path = Path(spec.path)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    return provenance_path, features_path, provenance, split_path, dataset, indices, labels, cached, num_classes, data_path


def _load_config_and_encoder(method: str, provenance: dict, device: torch.device):
    config_path = Path(provenance["configs"][method]["path"])
    if not config_path.is_file():
        raise FileNotFoundError(f"{method} config missing: {config_path}")
    config = init_jepa_config(str(config_path))
    configure_method_variant(config, method)
    encoder = build_encoder(config.encoder).to(device)
    checkpoint = Path(provenance["checkpoints"][method]["path"])
    load_context_encoder_checkpoint(encoder, checkpoint)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder, config_path


def _stage_directory(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


def run(args: argparse.Namespace) -> None:
    if len(set(args.head_seeds)) != len(args.head_seeds) or not args.head_seeds:
        raise ValueError("head seeds must be a non-empty unique sequence")
    if args.inference_repeats < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("inference repeats/batch size must be positive and workers nonnegative")
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    device = _device(args.device)
    torch.set_float32_matmul_precision("high")

    (provenance_path, features_path, trajectory, split_path, base_dataset,
     indices, labels, cached_features, num_classes, data_path) = _prepare_inputs(args)
    task_base = ROOT / "runs/representation_trajectories/multi_block_n6" / args.task / "seed_000"
    head_root = (args.head_root or task_base / "temporal_head_robustness/heads").resolve()
    output_dir = (args.output_dir or task_base / "temporal_head_robustness/evaluation").resolve()
    report_dir = (args.report_root or ROOT / "reports/representation_trajectories/multi_block_n6" / args.task / "seed_000").resolve()
    geometry_path, geometry_rows = _load_geometry_rows(trajectory, provenance_path)
    split_hash = trajectory["split_manifest_sha256"]

    # Validate every artifact before doing GPU inference.
    packet_records: dict[tuple[str, int], dict[str, Any]] = {}
    head_paths: dict[tuple[str, int], dict[int, Path]] = {}
    for method in METHODS:
        checkpoint = Path(trajectory["checkpoints"][method]["path"])
        checkpoint_hash = trajectory["checkpoints"][method]["sha256"]
        for seed in args.head_seeds:
            packet = head_root / f"head_seed_{seed}" / method / "seed_000" / "downstream" / "exit_metrics.json"
            paths = validate_head_packet(
                packet, method=method, head_seed=seed, checkpoint_path=checkpoint,
                checkpoint_sha256=checkpoint_hash, split_manifest=split_path, split_sha256=split_hash,
                task=args.task, num_classes=num_classes)
            head_paths[(method, seed)] = paths
            packet_records[(method, seed)] = read_json(packet)

    loader = DataLoader(
        IndexedSamples(base_dataset, indices.tolist()),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    all_probabilities = np.empty((len(METHODS), len(args.head_seeds), args.inference_repeats,
                                  len(LAYERS), len(TEMPORAL_LEVELS), len(indices), num_classes), dtype=np.float32)
    all_predictions = np.empty(all_probabilities.shape[:-1], dtype=np.int64)
    head_file_records: dict[str, Any] = {}
    for mi, method in enumerate(METHODS):
        print(f"evaluating method {method}", flush=True)
        encoder, config_path = _load_config_and_encoder(method, trajectory, device)
        heads_by_layer_seed = {layer: {} for layer in LAYERS}
        for seed_index, seed in enumerate(args.head_seeds):
            loaded = load_exit_heads(head_paths[(method, seed)], device=device, num_classes=num_classes)
            for layer in LAYERS:
                heads_by_layer_seed[layer][seed] = loaded[layer]
                head_file_records.setdefault(f"{method}/head_seed_{seed}", {})[str(layer)] = {
                    "path": str(head_paths[(method, seed)][layer].resolve()),
                    "sha256": sha256_file(head_paths[(method, seed)][layer]),
                    "effective_training_seed": int(seed + layer),
                }
        method_prob, method_pred = evaluate_temporal_prefixes(
            encoder, heads_by_layer_seed, loader, indices=indices,
            cached_features=cached_features[method], device=device,
            inference_repeats=args.inference_repeats, parity_check=True)
        all_probabilities[mi] = method_prob
        all_predictions[mi] = method_pred
        del encoder, heads_by_layer_seed
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del loader, base_dataset

    if np.any(all_predictions != all_predictions[:, :, :1]):
        raise RuntimeError("repeat-level prediction instability detected; reports were not published")
    raw_rows, per_seed_rows, aggregate_rows = build_metric_tables(
        all_probabilities, all_predictions, labels, args.head_seeds, num_classes)
    if len(raw_rows) != len(METHODS) * len(args.head_seeds) * args.inference_repeats * len(LAYERS) * len(TEMPORAL_LEVELS):
        raise RuntimeError("unexpected raw inference row count")
    if len(per_seed_rows) != len(METHODS) * len(args.head_seeds) * len(LAYERS) * len(TEMPORAL_LEVELS):
        raise RuntimeError("unexpected per-head-seed row count")
    if len(aggregate_rows) != len(METHODS) * len(LAYERS) * len(TEMPORAL_LEVELS):
        raise RuntimeError("unexpected aggregate row count")
    point_estimates = {}
    for row in aggregate_rows:
        for metric, key in (("macro_accuracy", "mean_macro_accuracy"),
                            ("absolute_degradation", "mean_absolute_degradation")):
            point_estimates[(row["method"], row["layer"], row["temporal_columns"], metric)] = row[key]
    bootstrap_rows = hierarchical_paired_bootstrap(
        all_predictions, labels, args.head_seeds, replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed, point_estimates=point_estimates, num_classes=num_classes)
    alignment_rows = join_geometry_metrics(geometry_rows, aggregate_rows)
    report_aggregate = [dict(row) for row in aggregate_rows]
    provenance_out = {
        "schema_version": 1,
        "task": args.task,
        "num_classes": num_classes,
        "pretraining_seed": 0,
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                       text=True, capture_output=True).stdout.strip(),
        "trajectory_provenance": {"path": str(provenance_path), "sha256": sha256_file(provenance_path)},
        "trajectory_test_features": {"path": str(features_path), "sha256": sha256_file(features_path)},
        "geometry_metrics": {"path": str(geometry_path), "sha256": sha256_file(geometry_path)},
        "data": {"path": str(data_path.resolve()), "sample_shape": list(trajectory["data"]["sample_shape"])},
        "split": {"path": str(split_path.resolve()), "sha256": sha256_file(split_path),
                  "indices_sha256": sha256_indices(indices), "sample_indices_sha256": sha256_indices(indices),
                  "class_counts": {str(cls): int(np.sum(labels == cls)) for cls in range(num_classes)}},
        "checkpoints": trajectory["checkpoints"],
        "configs": trajectory["configs"],
        "head_seeds": list(args.head_seeds),
        "effective_exit_head_seeds": {
            str(seed): {str(layer): int(seed + layer) for layer in LAYERS} for seed in args.head_seeds},
        "head_packets": {
            f"{method}/head_seed_{seed}": {
                "path": str((head_root / f"head_seed_{seed}" / method / "seed_000" /
                             "downstream/exit_metrics.json").resolve()),
                "sha256": sha256_file(head_root / f"head_seed_{seed}" / method / "seed_000" /
                                      "downstream/exit_metrics.json"),
            } for method in METHODS for seed in args.head_seeds},
        "head_checkpoints": head_file_records,
        "head_protocol": {"selection": "validation_only", "training_input": "full information Tp14 only",
                          "backbone_frozen": True, "epochs_max": 200, "batch_size": 32,
                          "learning_rate": 1e-3, "weight_decay": 0.05,
                          "early_stopping_patience": 20, "minimum_delta": 0.0},
        "prefix_axis": trajectory.get("prefix_axis", "width"),
        "prefix_semantics": trajectory.get("prefix_semantics", "temporal"),
        "evaluation": {"temporal_columns": list(TEMPORAL_LEVELS),
                       "token_counts": [28, 56, 112, 196],
                       "token_selection": "row-major leading width columns on unchanged 14x14 positional grid",
                       "layers": list(LAYERS), "feature_definition": trajectory["feature_definition"],
                       "inference_repeats": args.inference_repeats, "shuffle": False,
                       "batch_size": args.batch_size, "num_workers": args.num_workers,
                       "device": str(device), "repeat_prediction_disagreement": 0,
                       "max_probability_delta": float(np.max(np.abs(all_probabilities - all_probabilities[:, :, :1])))},
        "bootstrap": {"method": "paired hierarchical; resample head seeds and test examples within class",
                      "replicates": args.bootstrap_replicates, "seed": args.bootstrap_seed,
                      "comparisons": [list(pair) for pair in COMPARISONS]},
        "uncertainty_scope": "head-seed and test-example variation conditional on pretraining seed 0; no pretraining-seed uncertainty",
        "p2_reverse_caveat": "epoch 49 / step 11300 from a 50-epoch run; exploratory against 60-epoch methods",
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device)},
        "outputs": {"raw_rows": len(raw_rows), "per_head_seed_rows": len(per_seed_rows),
                    "aggregate_rows": len(aggregate_rows), "bootstrap_rows": len(bootstrap_rows)},
    }

    output_stage = _stage_directory(output_dir.parent, ".temporal-head-evaluation-")
    report_stage = _stage_directory(report_dir.parent, ".temporal-head-report-")
    try:
        write_csv(output_stage / "raw_inference_runs.csv", raw_rows)
        write_csv(output_stage / "per_head_seed_metrics.csv", per_seed_rows)
        write_csv(output_stage / "aggregate_metrics.csv", aggregate_rows)
        write_csv(output_stage / "bootstrap_intervals.csv", bootstrap_rows)
        np.savez_compressed(output_stage / "predictions.npz",
                            probabilities=all_probabilities, predictions=all_predictions,
                            labels=labels, indices=indices, methods=np.asarray(METHODS),
                            head_seeds=np.asarray(args.head_seeds), layers=np.asarray(LAYERS),
                            temporal_columns=np.asarray(TEMPORAL_LEVELS))
        save_json(output_stage / "provenance.json", provenance_out)

        write_csv(report_stage / "temporal_head_robustness.csv", report_aggregate)
        write_csv(report_stage / "geometry_functional_alignment.csv", alignment_rows)
        write_figures(aggregate_rows, report_stage, task=args.task,
                      prefix_semantics=trajectory.get("prefix_semantics", "temporal"))
        write_summary(report_stage / "temporal_head_robustness_summary.md",
                      aggregate_rows, bootstrap_rows, provenance_out, task=args.task)

        output_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        for name in ("raw_inference_runs.csv", "per_head_seed_metrics.csv", "aggregate_metrics.csv",
                     "bootstrap_intervals.csv", "predictions.npz", "provenance.json"):
            os.replace(output_stage / name, output_dir / name)
        for name in ("temporal_head_robustness.csv", "geometry_functional_alignment.csv",
                     "temporal_head_robustness.png", "temporal_head_robustness.pdf",
                     "temporal_head_degradation.png", "temporal_head_degradation.pdf",
                     "temporal_head_robustness_summary.md"):
            os.replace(report_stage / name, report_dir / name)
    finally:
        shutil.rmtree(output_stage, ignore_errors=True)
        shutil.rmtree(report_stage, ignore_errors=True)
    print(json.dumps({"output_dir": str(output_dir), "report_dir": str(report_dir),
                      "rows": provenance_out["outputs"], "device": str(device)}, indent=2))


if __name__ == "__main__":
    run(_args())
