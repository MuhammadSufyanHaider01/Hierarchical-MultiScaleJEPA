#!/usr/bin/env python3
"""Run a small, real-data P2 mask-relative JEPA training step on CPU or CUDA."""

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
from collators import get_collator  # noqa: E402
from datasets import PretrainCache  # noqa: E402
from models import JEPA  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/p2.yml")
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
    if config.hierarchy.mode != "mask_relative_fine_to_coarse":
        raise ValueError("P2 smoke requires hierarchy.mode=mask_relative_fine_to_coarse")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    first_dataset = config.data.pretrain_datasets[0]
    dataset = PretrainCache(first_dataset.path, first_dataset.key)
    collator = get_collator(config.masks.type)(
        input_size=config.encoder.img_size,
        patch_size=config.encoder.patch_size,
        **getattr(config.masks, config.masks.type).model_dump(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    batch, masks_enc, masks_pred = next(iter(loader))
    batch = batch.to(device=device, dtype=torch.float32)
    masks_enc = [mask.to(device=device, dtype=torch.long) for mask in masks_enc]
    masks_pred = [mask.to(device=device, dtype=torch.long) for mask in masks_pred]

    model = JEPA(
        config,
        iterations_per_epoch=max(1, len(loader)),
        eval_fns=[lambda *_: 0.0],
        metrics=["smoke"],
    ).to(device)
    model.train()
    optimizer = model.configure_optimizers()[0][0]
    hierarchy_weight = model.hierarchy_weight(epoch=args.epoch)

    anchor_losses = []
    total_losses = []
    auxiliary_losses = {layer: [] for layer in model.hierarchy_layers}
    region_counts = {}
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        prediction, target, _, auxiliary_predictions, auxiliary_targets = model(
            batch, masks_enc, masks_pred, return_hierarchy=True
        )
        anchor_loss = F.mse_loss(prediction, target)
        auxiliary_loss_values = {}
        for layer in model.hierarchy_layers:
            auxiliary_loss = F.mse_loss(auxiliary_predictions[layer], auxiliary_targets[layer])
            auxiliary_losses[layer].append(float(auxiliary_loss.detach().cpu()))
            auxiliary_loss_values[layer] = auxiliary_loss
            region_counts[layer] = int(auxiliary_targets[layer].shape[-2])
        loss = model._hierarchy_total_loss(anchor_loss, auxiliary_loss_values, hierarchy_weight)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite P2 loss: {loss.item()}")
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
        anchor_losses.append(float(anchor_loss.detach().cpu()))
        total_losses.append(float(loss.detach().cpu()))

    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"dataset={first_dataset.name}")
    print(f"batch_shape={tuple(batch.shape)}")
    print(f"steps={args.steps}")
    print(f"hierarchy_weight={hierarchy_weight}")
    print(f"region_counts={region_counts}")
    print(f"anchor_losses={anchor_losses}")
    print(f"auxiliary_losses={auxiliary_losses}")
    print(f"total_losses={total_losses}")
    print(f"target_frozen={all(not p.requires_grad for p in model.target_encoder.parameters())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
