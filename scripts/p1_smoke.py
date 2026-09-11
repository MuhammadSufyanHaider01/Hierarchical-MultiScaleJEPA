#!/usr/bin/env python3
"""Run a small, real-data P1 same-scale JEPA training step on CPU or CUDA."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
os.environ.setdefault("WANDB_KEY", "offline")

from configs import init_jepa_config  # noqa: E402
from datasets import PretrainCache  # noqa: E402
from models import JEPA  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/p1.yml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=12, help="epoch index used for hierarchy warm-up validation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.epoch < 0:
        raise ValueError("steps and batch-size must be positive; epoch must be non-negative")
    config = init_jepa_config(str(args.config))
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    first_dataset = config.data.pretrain_datasets[0]
    dataset = PretrainCache(first_dataset.path, first_dataset.key)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader)).to(device=device, dtype=torch.float32)

    model = JEPA(
        config,
        iterations_per_epoch=max(1, len(loader)),
        eval_fns=[lambda *_: 0.0],
        metrics=["smoke"],
    ).to(device)
    model.train()
    optimizer = model.configure_optimizers()[0][0]
    num_patches = (config.encoder.img_size[0] // config.encoder.patch_size) ** 2
    visible = num_patches // 2
    masks_enc = [torch.arange(visible, device=device).repeat(batch.size(0), 1)]
    masks_pred = [torch.arange(visible, num_patches, device=device).repeat(batch.size(0), 1)]
    hierarchy_weight = model.hierarchy_weight(epoch=args.epoch)

    losses = []
    auxiliary_losses = {layer: [] for layer in model.hierarchy_layers}
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        prediction, target, _, auxiliary_predictions, auxiliary_targets = model(
            batch, masks_enc, masks_pred, return_hierarchy=True
        )
        anchor_loss = F.mse_loss(prediction, target)
        loss = anchor_loss
        for layer in model.hierarchy_layers:
            auxiliary_loss = F.mse_loss(auxiliary_predictions[layer], auxiliary_targets[layer])
            auxiliary_losses[layer].append(float(auxiliary_loss.detach().cpu()))
            loss = loss + hierarchy_weight * auxiliary_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite P1 loss: {loss.item()}")
        loss.backward()
        if any(parameter.grad is not None for parameter in model.target_encoder.parameters()):
            raise RuntimeError("target encoder received gradients")
        if not all(
            any(parameter.grad is not None for parameter in predictor.parameters())
            for predictor in model.aux_predictors.values()
        ):
            raise RuntimeError("an auxiliary predictor received no gradients")
        optimizer.step()
        model.update_target_encoder(model.ema_start)
        losses.append(float(loss.detach().cpu()))

    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"dataset={first_dataset.name}")
    print(f"batch_shape={tuple(batch.shape)}")
    print(f"steps={args.steps}")
    print(f"hierarchy_weight={hierarchy_weight}")
    print(f"anchor_losses={losses}")
    print(f"auxiliary_losses={auxiliary_losses}")
    print(f"target_frozen={all(not p.requires_grad for p in model.target_encoder.parameters())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
