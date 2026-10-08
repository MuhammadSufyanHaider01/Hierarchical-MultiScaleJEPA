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
        depth = int(getattr(encoder, "logical_depth", len(encoder.blocks)))
        exit_layer = int(exit_layer)
        if exit_layer < 1 or exit_layer > depth:
            raise ValueError(f"exit_layer must be in [1, {depth}], got {exit_layer}")
        self.encoder = encoder
        self.exit_layer = exit_layer
        embed_dim = getattr(encoder, "exit_embed_dim", None)
        if embed_dim is None:
            embed_dim = encoder.pos_embed.shape[-1]
        self.head = ExitHead(int(embed_dim), int(num_outputs))
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


class CausalFusionExitHead(nn.Module):
    """Mean-pool and learn a convex fusion of causal encoder exits."""

    def __init__(self, embed_dim: int, num_outputs: int, source_layers):
        super().__init__()
        layers = tuple(int(layer) for layer in source_layers)
        if not layers or tuple(sorted(set(layers))) != layers:
            raise ValueError("source_layers must be a non-empty, strictly increasing sequence")
        self.source_layers = layers
        self.fusion_logits = nn.Parameter(torch.zeros(len(layers))) if len(layers) > 1 else None
        self.norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, num_outputs)

    @property
    def fusion_weights(self) -> dict[int, float]:
        if self.fusion_logits is None:
            return {self.source_layers[0]: 1.0}
        weights = self.fusion_logits.detach().softmax(dim=0).cpu().tolist()
        return dict(zip(self.source_layers, map(float, weights)))

    def forward(self, representations: Mapping[int, torch.Tensor]) -> torch.Tensor:
        missing = set(self.source_layers).difference(representations)
        if missing:
            raise ValueError(f"missing source-layer representations: {sorted(missing)}")
        pooled = []
        shape = None
        for layer in self.source_layers:
            tokens = representations[layer]
            if tokens.ndim != 3:
                raise ValueError(f"L{layer} tokens must have shape [B,N,D], got {tuple(tokens.shape)}")
            if shape is None:
                shape = (tokens.shape[0], tokens.shape[2])
            elif (tokens.shape[0], tokens.shape[2]) != shape:
                raise ValueError("source layers must have matching batch and embedding dimensions")
            pooled.append(tokens.mean(dim=1))
        if self.fusion_logits is None:
            fused = pooled[0]
        else:
            weights = self.fusion_logits.softmax(dim=0)
            fused = torch.stack(pooled, dim=1).mul(weights.view(1, -1, 1)).sum(dim=1)
        return self.projection(self.norm(fused))


class FrozenEncoderFusionExit(nn.Module):
    """A causal fusion head attached to a frozen encoder prefix."""

    def __init__(self, encoder: nn.Module, exit_layer: int, num_outputs: int):
        super().__init__()
        depth = int(getattr(encoder, "logical_depth", len(encoder.blocks)))
        self.exit_layer = int(exit_layer)
        if self.exit_layer not in (2, 4, 6, 8) or self.exit_layer > depth:
            raise ValueError(f"unsupported fusion exit L{self.exit_layer} for encoder depth {depth}")
        self.source_layers = tuple(layer for layer in (2, 4, 6, 8) if layer <= self.exit_layer)
        self.encoder = encoder
        embed_dim = getattr(encoder, "exit_embed_dim", None)
        if embed_dim is None:
            embed_dim = encoder.pos_embed.shape[-1]
        self.head = CausalFusionExitHead(int(embed_dim), int(num_outputs), self.source_layers)
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def representations(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder.forward_to_layers(x, self.source_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.representations(x))


class FrozenEncoderFusionTeacherStudent(nn.Module):
    """Frozen causal teacher and trainable single-exit student over one prefix."""

    def __init__(self, encoder: nn.Module, exit_layer: int, num_outputs: int,
                 teacher_state: Mapping[str, torch.Tensor]):
        super().__init__()
        self.exit_layer = int(exit_layer)
        self.source_layers = tuple(layer for layer in (2, 4, 6, 8) if layer <= self.exit_layer)
        self.encoder = encoder
        embed_dim = getattr(encoder, "exit_embed_dim", None) or encoder.pos_embed.shape[-1]
        self.teacher = CausalFusionExitHead(int(embed_dim), int(num_outputs), self.source_layers)
        self.teacher.load_state_dict(teacher_state, strict=True)
        self.student = ExitHead(int(embed_dim), int(num_outputs))
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.encoder.eval()
        self.teacher.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        self.teacher.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.encoder.eval()
        with torch.no_grad():
            representations = self.encoder.forward_to_layers(x, self.source_layers)
            teacher_output = self.teacher(representations)
        student_output = self.student(representations[self.exit_layer])
        return student_output, teacher_output


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
