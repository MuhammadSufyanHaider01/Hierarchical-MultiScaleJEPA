#!/usr/bin/env python3
"""Train one tiny head per frozen P0 exit on one real local dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from configs import init_jepa_config  # noqa: E402
from datasets import DeepMIMO, ImageDataset  # noqa: E402
from models import Encoder, FrozenEncoderExit, load_target_encoder_checkpoint  # noqa: E402


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _dataset(config, task: str):
    if task == "rfs":
        return ImageDataset(_path(config.data.rfs_path), sample_key="image", label_key="label"), 20, True
    if task == "nrpos-outdoor":
        return (
            ImageDataset(
                _path(config.data.pos_path),
                sample_key="features",
                label_key="label",
                label_dtype=torch.float32,
            ),
            3,
            False,
        )
    if task == "deepmimo-beam":
        return DeepMIMO(_path(config.data.beam_path), n_beams=16), 16, True
    if task == "deepmimo-los":
        return DeepMIMO(_path(config.data.beam_path), label_key="label_los"), 2, True
    raise ValueError(f"unknown task: {task}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p0.yml")
    parser.add_argument("--task", choices=("rfs", "nrpos-outdoor", "deepmimo-beam", "deepmimo-los"), default="rfs")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("--steps and --batch-size must be positive")
    os.environ.setdefault("WANDB_KEY", "offline")
    config = init_jepa_config(str(_path(args.config)))
    dataset, num_outputs, classification = _dataset(config, args.task)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    iterator = iter(loader)

    encoder = Encoder(
        img_size=tuple(config.encoder.img_size),
        patch_size=config.encoder.patch_size,
        embed_dim=config.encoder.embed_dim,
        depth=config.encoder.depth,
        num_heads=config.encoder.num_heads,
    ).to(device)
    if args.checkpoint:
        load_target_encoder_checkpoint(encoder, _path(args.checkpoint))
    encoder.eval()

    losses = {}
    for exit_layer in (2, 4, 6, 8):
        model = FrozenEncoderExit(encoder, exit_layer=exit_layer, num_outputs=num_outputs).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=1.0e-3)
        layer_losses = []
        for _ in range(args.steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            samples, labels = batch[:2]
            samples = samples.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            logits = model(samples)
            if classification:
                loss = torch.nn.functional.cross_entropy(logits, labels.long())
            else:
                loss = torch.nn.functional.mse_loss(logits, labels.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None for parameter in encoder.parameters()):
                raise RuntimeError("frozen target encoder received gradients")
            optimizer.step()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at exit L{exit_layer}: {loss.item()}")
            layer_losses.append(float(loss.detach().cpu()))
        losses[str(exit_layer)] = layer_losses

    print(json.dumps({
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "task": args.task,
        "checkpoint": str(_path(args.checkpoint)) if args.checkpoint else None,
        "backbone_frozen": all(not parameter.requires_grad for parameter in encoder.parameters()),
        "losses": losses,
    }, indent=2))


if __name__ == "__main__":
    main()
