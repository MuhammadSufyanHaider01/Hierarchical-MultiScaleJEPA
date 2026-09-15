#!/usr/bin/env python3
"""Run the pinned LatentWave source-style P0 L8 validation probe.

This is deliberately separate from ``phase5_exits.py``.  The pinned
``src/pretrain.py`` evaluated the EMA ``target_encoder`` at L8, mean pooled
its full-input tokens, and used scikit-learn linear probes over 80/20 splits.
Phase 5 early-exit experiments instead use the optimized context encoder and
their own persisted 80/10/10 split.  Their results must never be combined.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from configs import init_jepa_config  # noqa: E402
from datasets import DeepMIMO, ImageDataset  # noqa: E402
from evaluation import evaluate_class, evaluate_pos  # noqa: E402
from models import Encoder, load_target_encoder_checkpoint  # noqa: E402


SOURCE_TASKS = ("rfs", "nrpos-outdoor", "deepmimo-beam", "human-activity")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/p0.yml")
    parser.add_argument("--task", choices=SOURCE_TASKS, action="append")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON result path, normally below the gitignored runs/ directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing explicit --output file.")
    return parser.parse_args()


def _repository_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _device_name(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _task_dataset(task: str, config):
    if task == "rfs":
        return ImageDataset(_repository_path(config.data.rfs_path)), "rfs_pca"
    if task == "nrpos-outdoor":
        return (
            ImageDataset(
                _repository_path(config.data.pos_path),
                sample_key="features",
                label_dtype=torch.float32,
            ),
            "pos_error",
        )
    if task == "deepmimo-beam":
        return (
            DeepMIMO(
                _repository_path(config.data.beam_path),
                label_key="label_beam",
                n_beams=16,
            ),
            "beam_pca",
        )
    if task == "human-activity":
        return ImageDataset(REPOSITORY_ROOT.parent / "has.h5", sample_key="csi", label_key="label"), "has_pca"
    raise ValueError(f"unsupported source-style task: {task}")


def _extract_representations(encoder: Encoder, loader: DataLoader, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    encoder.eval()
    with torch.inference_mode():
        for batch in loader:
            samples, targets = batch[:2]
            samples = samples.to(device=device, dtype=torch.float32, non_blocking=True)
            # This exactly mirrors the pinned ValidationModule: target L8,
            # followed by mean pooling over all full-input tokens.
            embeddings.append(encoder(samples).mean(dim=1).cpu())
            labels.append(targets.cpu())
    if not embeddings:
        raise RuntimeError("source-style probe dataset produced no batches")
    return embeddings, labels


def main() -> int:
    args = _arguments()
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("batch-size must be positive when set")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")

    checkpoint = _repository_path(args.checkpoint)
    config_path = _repository_path(args.config)
    output = _repository_path(args.output)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"probe output already exists: {output}; pass --overwrite to replace it")

    device = torch.device(_device_name(args.device))
    torch.set_float32_matmul_precision("high")
    config = init_jepa_config(str(config_path))
    batch_size = config.data.batch_size if args.batch_size is None else args.batch_size
    encoder = Encoder(
        img_size=tuple(config.encoder.img_size),
        patch_size=config.encoder.patch_size,
        embed_dim=config.encoder.embed_dim,
        depth=config.encoder.depth,
        num_heads=config.encoder.num_heads,
    ).to(device)
    load_target_encoder_checkpoint(encoder, checkpoint)

    tasks = tuple(args.task or SOURCE_TASKS)
    started = time.perf_counter()
    results: dict[str, object] = {
        "protocol": {
            "name": "pinned_latentwave_source_style_l8_probe",
            "reference": "src/pretrain.py and src/models/jepa.py at pinned commit 330e9ef5f6eb69e728ba43b3755036cffdd086cd",
            "backbone": "EMA target_encoder",
            "representation": "full-input L8 tokens, mean pooled",
            "classification_probe": {
                "split": "stratified 80/20 train_test_split",
                "split_seeds": [0, 1, 2],
                "model": "StandardScaler + LogisticRegression(C=5, solver=newton-cg, max_iter=1000)",
                "metric": "mean balanced accuracy across the three source split seeds",
            },
            "position_probe": {
                "split": "80/20 train_test_split with random_state=0",
                "model": "StandardScaler + LinearRegression",
                "metric": "mean physical Euclidean position error using the pinned source coordinate constants",
            },
            "not_comparable_to": "Phase 5 context-encoder early exits with persisted 80/10/10 splits",
        },
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "device": str(device),
        "batch_size": batch_size,
        "tasks": {},
    }

    for task in tasks:
        dataset, metric_name = _task_dataset(task, config)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        embeddings, labels = _extract_representations(encoder, loader, device)
        score = evaluate_pos(embeddings, labels) if task == "nrpos-outdoor" else evaluate_class(embeddings, labels)
        results["tasks"][task] = {
            "metric_name": metric_name,
            "value": float(score),
            "examples": int(sum(values.shape[0] for values in labels)),
        }
        print(f"[{task}] {metric_name}={float(score):.6f}", flush=True)

    results["wall_time_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output), "tasks": list(tasks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
