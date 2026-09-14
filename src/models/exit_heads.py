"""Frozen context-encoder exits for Phase 2 downstream evaluation.

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
        # Keep the complete context backbone in inference mode even when the
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


def _select_prefixed_encoder_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    for prefix in prefixes:
        selected = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
        if selected:
            return selected

    return {}


def _select_unprefixed_encoder_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a direct encoder state dict from a legacy standalone checkpoint."""

    encoder_keys = ("patch_embed.", "pos_embed", "blocks.", "norm.")
    selected = {
        key: value
        for key, value in state_dict.items()
        if isinstance(key, str) and key.startswith(encoder_keys)
    }
    return selected


def _select_encoder_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    branch: str,
) -> dict[str, torch.Tensor]:
    """Select an encoder branch without silently substituting the wrong one.

    The Version 1 early-exit protocol uses the optimized context branch and
    permits a target-only fallback for old P0 checkpoints.  The source-style
    P0 reproduction probe instead requires the EMA target branch; it must not
    accidentally evaluate context weights when both namespaces are present.
    """

    if branch == "context":
        primary = ("context_encoder.", "module.context_encoder.")
        fallback = ("target_encoder.", "module.target_encoder.", "encoder.", "module.encoder.")
    elif branch == "target":
        primary = ("target_encoder.", "module.target_encoder.")
        # A bare encoder checkpoint can only be interpreted as a standalone
        # encoder.  Do not fall back to context_encoder.* here: that would
        # invalidate a source-style target-branch probe.
        fallback = ("encoder.", "module.encoder.")
    else:
        raise ValueError(f"unsupported encoder branch: {branch!r}")

    selected = _select_prefixed_encoder_state_dict(state_dict, primary)
    if selected:
        return selected
    selected = _select_prefixed_encoder_state_dict(state_dict, fallback)
    if selected:
        return selected
    selected = _select_unprefixed_encoder_state_dict(state_dict)
    if selected:
        return selected
    raise KeyError(f"checkpoint does not contain {branch}-encoder weights")


def _load_encoder_checkpoint(encoder: nn.Module, checkpoint_path: str | Path, *, branch: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _select_encoder_state_dict(_checkpoint_state_dict(checkpoint), branch=branch)
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{branch} encoder checkpoint mismatch: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )


def load_context_encoder_checkpoint(encoder: nn.Module, checkpoint_path: str | Path) -> None:
    """Load the downstream context backbone, with legacy-prefix fallback."""

    _load_encoder_checkpoint(encoder, checkpoint_path, branch="context")


def load_target_encoder_checkpoint(encoder: nn.Module, checkpoint_path: str | Path) -> None:
    """Load the EMA target branch for the original LatentWave L8 probe."""

    _load_encoder_checkpoint(encoder, checkpoint_path, branch="target")
