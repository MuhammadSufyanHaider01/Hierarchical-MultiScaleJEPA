"""Frozen target-encoder exits for Phase 2 downstream evaluation.

The encoder itself remains the pinned LatentWave implementation. These
wrappers only add full-input intermediate reads and independently trainable
task heads; they do not alter the P0 pretraining path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import nn


class ExitHead(nn.Module):
    """Mean-pool encoder tokens, normalize, and apply a task projection."""

    def __init__(self, embed_dim: int, num_outputs: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, num_outputs)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"expected encoder tokens with shape [B, N, D], got {tuple(tokens.shape)}")
        pooled = tokens.mean(dim=1)
        return self.projection(self.norm(pooled))


class FrozenEncoderExit(nn.Module):
    """A task head attached to one frozen, full-input encoder exit."""

    def __init__(self, encoder: nn.Module, exit_layer: int, num_outputs: int):
        super().__init__()
        depth = len(encoder.blocks)
        exit_layer = int(exit_layer)
        if exit_layer < 1 or exit_layer > depth:
            raise ValueError(f"exit_layer must be in [1, {depth}], got {exit_layer}")
        self.encoder = encoder
        self.exit_layer = exit_layer
        self.head = ExitHead(encoder.pos_embed.shape[-1], int(num_outputs))
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

    def train(self, mode: bool = True):
        # Keep the complete target backbone in inference mode even when the
        # task head is switched to training mode by a surrounding trainer.
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            tokens = self.encoder.forward_to_layer(x, self.exit_layer)
        return self.head(tokens)


def _checkpoint_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"expected a checkpoint mapping, got {type(checkpoint).__name__}")
    for key in ("state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def _select_encoder_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Lightning P0 checkpoints use ``target_encoder.*``. The other prefixes
    # make the loader useful for ordinary wrapped module checkpoints without
    # changing the pinned model's state-dict layout.
    prefixes = (
        "target_encoder.",
        "module.target_encoder.",
        "encoder.",
        "module.encoder.",
    )
    for prefix in prefixes:
        selected = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
        if selected:
            return selected

    encoder_keys = ("patch_embed.", "pos_embed", "blocks.", "norm.")
    selected = {
        key: value
        for key, value in state_dict.items()
        if isinstance(key, str) and key.startswith(encoder_keys)
    }
    if selected:
        return selected
    raise KeyError("checkpoint does not contain target-encoder or encoder weights")


def load_target_encoder_checkpoint(encoder: nn.Module, checkpoint_path: str | Path) -> None:
    """Load target-encoder weights and fail if the backbone is incomplete."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _select_encoder_state_dict(_checkpoint_state_dict(checkpoint))
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "target encoder checkpoint mismatch: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
