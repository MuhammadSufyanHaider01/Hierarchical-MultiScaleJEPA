#!/usr/bin/env python3
"""Extract and compare P0/P2 Interference representation trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
os.environ.setdefault("WANDB_KEY", "offline")

from configs import init_jepa_config  # noqa: E402
from datasets import ImageDataset  # noqa: E402
from models import Encoder, load_context_encoder_checkpoint  # noqa: E402
from representation_trajectories import (  # noqa: E402
    EMBED_DIM, LAYERS, TEMPORAL_LEVELS, IndexedSamples, extract_features, fit_pca,
    load_split_and_labels, paired_bootstrap, sha256_file, sha256_indices,
    transform_pca, trajectory_metrics, validate_checkpoint_metadata,
    validate_source_metrics, write_figures, write_summary,
)

DEFAULTS = {
    "p0_checkpoint": Path("/home/waves-lab/LatentWave_Data/preprocessed_datasets/MultiScaleJEPA/runs/phase5_formal/p0/seed_000/checkpoints/epoch=epoch=059-step=step=13560.ckpt"),
    "p2_checkpoint": Path("/home/waves-lab/LatentWave_Data/preprocessed_datasets/MultiScaleJEPA/runs/phase5_formal/p2/seed_000/checkpoints/last.ckpt"),
    "p0_source_metrics": Path("runs/masking_study/multi_block_n6/p0/seed_000/downstream/exit_metrics.json"),
    "p2_source_metrics": Path("runs/masking_study/multi_block_n6/p2/seed_000/downstream/exit_metrics.json"),
    "split_manifest": Path("runs/masking_study/multi_block_n6/splits_seed_2025_primary_v3.json"),
    "interference_h5": Path("/home/waves-lab/LatentWave_Data/preprocessed_datasets/interference.h5"),
    "output_root": Path("runs/representation_trajectories/multi_block_n6/interference/seed_000"),
    "report_root": Path("reports/representation_trajectories/multi_block_n6/interference/seed_000"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default)
    parser.add_argument("--p0-config", type=Path, default=Path("configs/p0.yml"))
    parser.add_argument("--p2-config", type=Path, default=Path("configs/p2.yml"))
    parser.add_argument("--pca-fit-dir", type=Path, help="Validation output root, required for --split test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit-per-class", type=int, help="Diagnostic smoke-test cap; omit for production")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else value)


def _load_model(method: str, config_path: Path, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("metadata"), dict):
        raise ValueError(f"checkpoint lacks Phase 5 metadata: {checkpoint_path}")
    meta = validate_checkpoint_metadata(checkpoint["metadata"], method=method)
    if int(checkpoint.get("epoch", -1)) != 59 or int(checkpoint.get("global_step", -1)) != 13560:
        raise ValueError(f"{method} checkpoint must be epoch=59/global_step=13560")
    config = init_jepa_config(str(config_path))
    metadata_config = checkpoint["metadata"].get("config", {})
    for key in ("img_size", "patch_size", "embed_dim", "depth", "num_heads"):
        saved_value = metadata_config.get("encoder", {}).get(key)
        local_value = getattr(config.encoder, key)
        if key == "img_size":
            saved_value, local_value = tuple(saved_value or ()), tuple(local_value)
        if saved_value != local_value:
            raise ValueError(f"{method} config/checkpoint encoder mismatch at {key}")
    if config.masks.type != "multi_block" or config.masks.multi_block.aspect_ratio_type != "span_time":
        raise ValueError(f"{method} local config must use multi_block/span_time")
    if config.masks.multi_block.num_pred_masks != 6:
        raise ValueError(f"{method} local config must specify six prediction masks")
    expected_hierarchy = method == "p2"
    if config.hierarchy.enabled != expected_hierarchy:
        raise ValueError(f"{method} local config hierarchy enabled must be {expected_hierarchy}")
    if expected_hierarchy and (config.hierarchy.mode != "mask_relative_fine_to_coarse" or config.hierarchy.layers != [2, 4, 6]):
        raise ValueError("P2 local config must define fine-to-coarse hierarchy at L2/L4/L6")
    encoder = Encoder(
        img_size=tuple(config.encoder.img_size), patch_size=config.encoder.patch_size,
        embed_dim=config.encoder.embed_dim, depth=config.encoder.depth,
        num_heads=config.encoder.num_heads,
    ).to(device)
    load_context_encoder_checkpoint(encoder, checkpoint_path)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    meta["actual_epoch"] = int(checkpoint["epoch"])
    meta["actual_global_step"] = int(checkpoint["global_step"])
    return encoder, meta


def _save_pca(path: Path, pca) -> None:
    np.savez_compressed(path, mean=pca.mean_, components=pca.components_,
                        explained_variance=pca.explained_variance_,
                        explained_variance_ratio=pca.explained_variance_ratio_,
                        singular_values=pca.singular_values_)


def _load_pca(path: Path):
    from sklearn.decomposition import PCA
    with np.load(path) as values:
        pca = PCA(n_components=2)
        pca.mean_ = values["mean"]
        pca.components_ = values["components"]
        pca.explained_variance_ = values["explained_variance"]
        pca.explained_variance_ratio_ = values["explained_variance_ratio"]
        pca.singular_values_ = values["singular_values"]
        pca.n_features_in_ = pca.mean_.shape[0]
        pca.n_samples_ = 0
        pca.noise_variance_ = 0.0
        return pca


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and num-workers nonnegative")
    if args.limit_per_class is not None and args.limit_per_class < 2:
        raise ValueError("limit-per-class must be at least 2")
    paths = {name: _resolve(getattr(args, name)) for name in DEFAULTS}
    for name, path in paths.items():
        if name not in ("output_root", "report_root") and not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    split_manifest = paths["split_manifest"]
    packets = {}
    checkpoint_paths = {"p0": paths["p0_checkpoint"], "p2": paths["p2_checkpoint"]}
    for method in ("p0", "p2"):
        source = paths[f"{method}_source_metrics"]
        packets[method] = validate_source_metrics(source, method=method, manifest_path=split_manifest)
    if packets["p0"].get("split_manifest") != packets["p2"].get("split_manifest"):
        raise ValueError("P0 and P2 source metrics do not share the same split manifest")

    with h5py.File(paths["interference_h5"], "r") as h5:
        raw_labels = np.asarray(h5["cls_labels"][:]).reshape(-1)
        sample_shape = tuple(h5["cls_spectrograms"].shape[1:])
    split_indices, mapped_labels = load_split_and_labels(split_manifest, raw_labels)
    if sample_shape[-2:] != (224, 224):
        raise ValueError(f"expected 224x224 input spectrograms, got {sample_shape}")
    indices = split_indices[args.split]
    if args.limit_per_class is not None:
        chosen = []
        for cls in (0, 1):
            chosen.extend([i for i in indices if mapped_labels[i] == cls][:args.limit_per_class])
        indices = chosen
    expected_labels = mapped_labels[np.asarray(indices, dtype=np.int64)]
    base_dataset = ImageDataset(paths["interference_h5"], sample_key="cls_spectrograms", label_key="cls_labels", label_dtype=None)
    loader = DataLoader(IndexedSamples(base_dataset, indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=args.device != "cpu", persistent_workers=args.num_workers > 0)

    device = _device(args.device)
    features: dict[str, np.ndarray] = {}
    model_metadata: dict[str, Any] = {}
    for method in ("p0", "p2"):
        encoder, model_metadata[method] = _load_model(
            method, _resolve(getattr(args, f"{method}_config")), checkpoint_paths[method], device
        )
        features[method], labels, seen_indices = extract_features(
            encoder, loader, mapped_labels, device=device,
        )
        if not np.array_equal(seen_indices, np.asarray(indices)) or not np.array_equal(labels, expected_labels):
            raise RuntimeError(f"{method} extracted samples are not aligned to persisted {args.split} order")
        if args.limit_per_class is None and not np.array_equal(labels, mapped_labels[np.asarray(split_indices[args.split])]):
            raise RuntimeError(f"{method} labels differ from persisted split labels")
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not np.array_equal(labels, expected_labels):
        raise RuntimeError("P0/P2 sample labels are not aligned")

    output_dir = paths["output_root"] / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    class_counts = {str(cls): int(np.sum(labels == cls)) for cls in (0, 1)}
    np.savez_compressed(output_dir / "features.npz", labels=labels, indices=seen_indices,
                        p0=features["p0"], p2=features["p2"],
                        temporal_levels=np.asarray(TEMPORAL_LEVELS), layers=np.asarray(LAYERS))
    provenance = {
        "schema_version": 1,
        "task": "interference",
        "split": args.split,
        "split_manifest": str(split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(split_manifest),
        "split_indices_sha256": sha256_indices(split_indices[args.split]),
        "sample_indices_sha256": sha256_indices(seen_indices.tolist()),
        "class_counts": class_counts,
        "checkpoints": {},
        "configs": {},
        "source_metrics": {},
        "p0_packet_checkpoint_mismatch": None,
        "data": {"path": str(paths["interference_h5"].resolve()), "sample_shape": sample_shape},
        "feature_definition": {
            "encoder": "frozen context_encoder", "layers": list(LAYERS),
            "temporal_columns": list(TEMPORAL_LEVELS), "grid": [14, 14],
            "selection": "row-major leading-column patch indices after full-grid positional embedding",
            "token_counts": [28, 56, 112, 196], "pooling": "mean over normalized tokens",
            "dimension": EMBED_DIM,
        },
        "pca": {"fit_split": "validation", "n_components": 2, "svd_solver": "full",
                "basis": "separate per model; jointly fit over all layers and temporal levels"},
        "bootstrap": {"replicates": 2000, "seed": 2025, "sampling": "paired, stratified within class"},
        "diagnostic_limit_per_class": args.limit_per_class,
        "environment": {"torch": torch.__version__, "device": str(device)},
    }
    for method in ("p0", "p2"):
        checkpoint = torch.load(checkpoint_paths[method], map_location="cpu", weights_only=False)
        provenance["checkpoints"][method] = {
            "path": str(checkpoint_paths[method].resolve()), "sha256": sha256_file(checkpoint_paths[method]),
            **model_metadata[method],
        }
        provenance["configs"][method] = {"path": str(_resolve(getattr(args, f"{method}_config")).resolve())}
        metric_path = paths[f"{method}_source_metrics"]
        provenance["source_metrics"][method] = {"path": str(metric_path.resolve()), "sha256": sha256_file(metric_path)}
        if method == "p0":
            provenance["p0_packet_checkpoint_mismatch"] = {
                "packet_checkpoint": packets[method].get("checkpoint"),
                "analysis_checkpoint": str(checkpoint_paths[method].resolve()),
                "reason": "Use the epoch-59 checkpoint matched to P2; packet points to epoch-64 last.ckpt.",
            }
        del checkpoint
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))

    coords: dict[str, np.ndarray] = {}
    if args.split == "validation":
        for method in ("p0", "p2"):
            pca = fit_pca(features[method])
            _save_pca(output_dir / f"pca_{method}.npz", pca)
            coords[method] = transform_pca(pca, features[method]).astype(np.float32)
        (output_dir / "pca_metadata.json").write_text(json.dumps({
            "schema_version": 1, "fit_split": "validation",
            "checkpoint_sha256": {m: provenance["checkpoints"][m]["sha256"] for m in ("p0", "p2")},
            "feature_definition": provenance["feature_definition"],
        }, indent=2, sort_keys=True))
        np.savez_compressed(output_dir / "coordinates.npz", labels=labels, indices=seen_indices,
                            p0=coords["p0"], p2=coords["p2"])
        return

    if args.pca_fit_dir is None:
        raise ValueError("--split test requires --pca-fit-dir pointing to the validation output directory")
    pca_fit_dir = _resolve(args.pca_fit_dir)
    pca_meta = json.loads((pca_fit_dir / "pca_metadata.json").read_text())
    if pca_meta.get("fit_split") != "validation":
        raise ValueError("PCA artifact was not fit on validation")
    current_hashes = {m: provenance["checkpoints"][m]["sha256"] for m in ("p0", "p2")}
    if pca_meta.get("checkpoint_sha256") != current_hashes:
        raise ValueError("PCA artifacts were fit with different checkpoints")
    for method in ("p0", "p2"):
        coords[method] = transform_pca(_load_pca(pca_fit_dir / f"pca_{method}.npz"), features[method]).astype(np.float32)
    np.savez_compressed(output_dir / "coordinates.npz", labels=labels, indices=seen_indices,
                        p0=coords["p0"], p2=coords["p2"])
    metric_rows = trajectory_metrics(features, labels)
    interval_rows = paired_bootstrap(features, labels)
    _write_csv(output_dir / "metrics.csv", metric_rows)
    _write_csv(output_dir / "bootstrap_intervals.csv", interval_rows)
    report_dir = paths["report_root"]
    write_figures(features, coords, labels, metric_rows, interval_rows, report_dir)
    write_summary(report_dir / "summary.md", metric_rows, interval_rows, provenance)
    provenance["artifacts"] = {"features": str((output_dir / "features.npz").resolve()),
                               "coordinates": str((output_dir / "coordinates.npz").resolve()),
                               "metrics": str((output_dir / "metrics.csv").resolve()),
                               "figures": str(report_dir.resolve())}
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    run(arguments())
