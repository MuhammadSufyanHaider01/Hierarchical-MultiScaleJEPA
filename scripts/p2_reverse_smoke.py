#!/usr/bin/env python3
"""Run a synthetic finite-loss/gradient smoke for reverse-P2 on CPU or CUDA."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
os.environ.setdefault("WANDB_KEY", "offline")

from configs import init_jepa_config  # noqa: E402
from models import JEPA  # noqa: E402
from models.regions import build_mask_relative_region_layout  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/p2_reverse.yml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def _rectangular_masks(batch_size: int, device: torch.device):
    grid = 14
    all_indices = torch.arange(grid * grid, device=device)
    predictions = []
    for start_row in range(6):
        rows = torch.arange(start_row * 2, start_row * 2 + 2, device=device)
        mask = (rows[:, None] * grid + torch.arange(grid, device=device)[None, :]).reshape(1, -1)
        predictions.append(mask.expand(batch_size, -1))
    union = torch.cat(predictions, dim=1).unique()
    visible = all_indices[~torch.isin(all_indices, union)].unsqueeze(0).expand(batch_size, -1)
    return [visible], predictions


def main() -> int:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    config = init_jepa_config(str(args.config))
    if config.hierarchy.mode != "mask_relative_coarse_to_fine":
        raise ValueError("reverse-P2 smoke requires mask_relative_coarse_to_fine")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    masks_enc, masks_pred = _rectangular_masks(args.batch_size, device)
    layout = build_mask_relative_region_layout(masks_pred, grid_size=14)
    expected = {2: 1, 4: 7, 6: 28}
    actual = {2: layout.for_layer(6).shape[-2], 4: layout.for_layer(4).shape[-2], 6: layout.for_layer(2).shape[-2]}
    if actual != expected:
        raise AssertionError(f"reverse region counts differ: {actual}")
    for regions in (layout.for_layer(2), layout.for_layer(4), layout.for_layer(6)):
        if not torch.isin(regions, torch.cat(masks_pred, dim=0)).all():
            raise AssertionError("a reverse-P2 region contains a patch outside prediction masks")

    model = JEPA(config, iterations_per_epoch=1, eval_fns=[lambda *_: 0.0], metrics=["smoke"]).to(device)
    model.train()
    optimizer = model.configure_optimizers()[0][0]
    batch = torch.randn(args.batch_size, 1, 224, 224, device=device)
    anchor_losses, total_losses = [], []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        anchor_pred, anchor_target, _, predictions, targets = model(
            batch, masks_enc, masks_pred, return_hierarchy=True
        )
        if set(predictions) != {2, 4, 6}:
            raise AssertionError(f"unexpected auxiliary layers: {set(predictions)}")
        aux_losses = {}
        for layer, count in expected.items():
            if predictions[layer].shape != targets[layer].shape:
                raise AssertionError(f"L{layer} prediction/target shape mismatch")
            if predictions[layer].shape[-2] != count:
                raise AssertionError(f"L{layer} expected {count} regions, got {predictions[layer].shape[-2]}")
            aux_losses[layer] = F.mse_loss(predictions[layer], targets[layer])
        anchor_loss = F.mse_loss(anchor_pred, anchor_target)
        loss = model._hierarchy_total_loss(anchor_loss, aux_losses, model.hierarchy_weight())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite reverse-P2 loss: {loss.item()}")
        loss.backward()
        if any(parameter.grad is not None for parameter in model.target_encoder.parameters()):
            raise AssertionError("target encoder received gradients")
        if not all(any(parameter.grad is not None for parameter in predictor.parameters())
                   for predictor in model.aux_predictors.values()):
            raise AssertionError("an auxiliary predictor received no gradients")
        if not any(parameter.grad is not None for parameter in model.context_encoder.parameters()):
            raise AssertionError("context encoder received no gradients")
        optimizer.step()
        model.update_target_encoder(model.ema_start)
        anchor_losses.append(float(anchor_loss.detach().cpu()))
        total_losses.append(float(loss.detach().cpu()))

    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"region_counts={expected}")
    print(f"anchor_losses={anchor_losses}")
    print(f"total_losses={total_losses}")
    print("target_encoder_gradients=none; context_and_auxiliary_gradients=present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
