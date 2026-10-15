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
from torch.nn import functional as F


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


class ResidualProjectedFusionExitHead(nn.Module):
    """Anchor-preserving, sample-gated residual fusion of earlier exits."""

    ALLOWED_EXITS = (2, 4, 6, 8)
    BOTTLENECK_DIM = 64
    GATE_DIM = 64

    def __init__(self, embed_dim: int, num_outputs: int, exit_layer: int,
                 source_layers, bottleneck_dim: int = 64, gate_dim: int = 64):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_outputs = int(num_outputs)
        self.exit_layer = int(exit_layer)
        self.source_layers = tuple(int(layer) for layer in source_layers)
        self.bottleneck_dim = int(bottleneck_dim)
        self.gate_dim = int(gate_dim)
        if self.exit_layer not in self.ALLOWED_EXITS:
            raise ValueError(f"exit_layer must be one of {self.ALLOWED_EXITS}, got {self.exit_layer}")
        expected = tuple(layer for layer in self.ALLOWED_EXITS if layer < self.exit_layer)
        if self.source_layers != expected:
            raise ValueError(f"L{self.exit_layer} requires source_layers={expected}, got {self.source_layers}")
        if self.embed_dim < 1 or self.num_outputs < 1 or self.bottleneck_dim < 1 or self.gate_dim < 1:
            raise ValueError("dimensions and num_outputs must be positive")

        self.source_norms = nn.ModuleDict({str(j): nn.LayerNorm(self.embed_dim) for j in expected})
        self.down_projections = nn.ModuleDict({str(j): nn.Linear(self.embed_dim, self.bottleneck_dim) for j in expected})
        self.up_projections = nn.ModuleDict({str(j): nn.Linear(self.bottleneck_dim, self.embed_dim) for j in expected})
        self.anchor_gate_norm = nn.LayerNorm(self.embed_dim) if expected else None
        self.anchor_gate_projection = nn.Linear(self.embed_dim, self.gate_dim) if expected else None
        self.source_gate_projections = nn.ModuleDict({str(j): nn.Linear(self.bottleneck_dim, self.gate_dim) for j in expected})
        self.gate_scores = nn.ModuleDict({str(j): nn.Linear(self.gate_dim, 1) for j in expected})
        for up in self.up_projections.values():
            nn.init.zeros_(up.weight)
            nn.init.zeros_(up.bias)
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self.output_projection = nn.Linear(self.embed_dim, self.num_outputs)

    def _fuse_features(self, representations: Mapping[int, torch.Tensor], *, return_diagnostics=False):
        requested = set(self.source_layers) | {self.exit_layer}
        missing = requested.difference(representations)
        if missing:
            raise ValueError(f"missing required representations: {sorted(missing)}")
        pooled = {}
        batch_dim = None
        for layer in sorted(requested):
            tokens = representations[layer]
            if tokens.ndim != 3:
                raise ValueError(f"L{layer} tokens must have shape [B,N,D], got {tuple(tokens.shape)}")
            if tokens.shape[1] < 1 or tokens.shape[2] != self.embed_dim:
                raise ValueError(f"L{layer} tokens must have non-empty N and D={self.embed_dim}, got {tuple(tokens.shape)}")
            if batch_dim is None:
                batch_dim = tokens.shape[0]
            elif tokens.shape[0] != batch_dim:
                raise ValueError("all exit representations must have matching batch dimensions")
            pooled[layer] = tokens.mean(dim=1)
        anchor = pooled[self.exit_layer]
        diagnostics = {}
        if self.source_layers:
            query = self.anchor_gate_projection(self.anchor_gate_norm(anchor))
            residual_sum = torch.zeros_like(anchor)
            anchor_norm = torch.linalg.vector_norm(anchor, dim=-1)
            for layer in self.source_layers:
                key = str(layer)
                hidden = F.gelu(self.down_projections[key](self.source_norms[key](pooled[layer])))
                correction = self.up_projections[key](hidden)
                gate_hidden = torch.tanh(query + self.source_gate_projections[key](hidden))
                gate = torch.sigmoid(self.gate_scores[key](gate_hidden))
                residual = gate * correction
                residual_sum = residual_sum + residual
                if return_diagnostics:
                    residual_norm = torch.linalg.vector_norm(residual, dim=-1)
                    diagnostics[layer] = {"gate": gate.squeeze(-1), "residual_norm": residual_norm,
                                          "relative_residual_norm": residual_norm / (anchor_norm + 1e-8)}
            fused = anchor + residual_sum
        else:
            fused = anchor
        return (fused, diagnostics) if return_diagnostics else fused

    def forward(self, representations: Mapping[int, torch.Tensor], return_diagnostics: bool = False):
        result = self._fuse_features(representations, return_diagnostics=return_diagnostics)
        if return_diagnostics:
            fused, diagnostics = result
            return self.output_projection(self.output_norm(fused)), diagnostics
        return self.output_projection(self.output_norm(result))


class FrozenEncoderResidualFusionExit(nn.Module):
    """V2 residual fusion head over one frozen causal encoder prefix."""

    def __init__(self, encoder: nn.Module, exit_layer: int, num_outputs: int):
        super().__init__()
        depth = int(getattr(encoder, "logical_depth", len(encoder.blocks)))
        self.exit_layer = int(exit_layer)
        if self.exit_layer not in ResidualProjectedFusionExitHead.ALLOWED_EXITS or self.exit_layer > depth:
            raise ValueError(f"unsupported residual fusion exit L{self.exit_layer} for encoder depth {depth}")
        self.source_layers = tuple(layer for layer in ResidualProjectedFusionExitHead.ALLOWED_EXITS if layer < self.exit_layer)
        self.requested_layers = self.source_layers + (self.exit_layer,)
        self.encoder = encoder
        embed_dim = getattr(encoder, "exit_embed_dim", None) or encoder.pos_embed.shape[-1]
        self.head = ResidualProjectedFusionExitHead(int(embed_dim), int(num_outputs), self.exit_layer, self.source_layers)
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
            return self.encoder.forward_to_layers(x, self.requested_layers)

    def forward(self, x: torch.Tensor, return_diagnostics: bool = False):
        return self.head(self.representations(x), return_diagnostics=return_diagnostics)


ADAPTER_ABLATION_VARIANTS = (
    "bottleneck_linear",
    "bottleneck_sigmoid",
    "single_linear",
    "single_gelu",
    "single_sigmoid",
)


class ResidualFusionAblationExitHead(nn.Module):
    """Configurable residual-fusion adapter ablation head.

    The gate, anchor, pooling, and output head intentionally match Fusion V2.
    Only the source-to-correction adapter differs by ``adapter_variant``.
    """

    ALLOWED_EXITS = ResidualProjectedFusionExitHead.ALLOWED_EXITS

    def __init__(self, embed_dim: int, num_outputs: int, exit_layer: int,
                 source_layers, adapter_variant: str, bottleneck_dim: int = 64,
                 gate_dim: int = 64):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_outputs = int(num_outputs)
        self.exit_layer = int(exit_layer)
        self.source_layers = tuple(int(layer) for layer in source_layers)
        self.adapter_variant = str(adapter_variant)
        self.bottleneck_dim = int(bottleneck_dim)
        self.gate_dim = int(gate_dim)
        if self.adapter_variant not in ADAPTER_ABLATION_VARIANTS:
            raise ValueError(f"unsupported adapter_variant={adapter_variant!r}; expected one of {ADAPTER_ABLATION_VARIANTS}")
        if self.exit_layer not in self.ALLOWED_EXITS:
            raise ValueError(f"exit_layer must be one of {self.ALLOWED_EXITS}, got {self.exit_layer}")
        expected = tuple(layer for layer in self.ALLOWED_EXITS if layer < self.exit_layer)
        if self.source_layers != expected:
            raise ValueError(f"L{self.exit_layer} requires source_layers={expected}, got {self.source_layers}")
        if min(self.embed_dim, self.num_outputs, self.bottleneck_dim, self.gate_dim) < 1:
            raise ValueError("dimensions and num_outputs must be positive")

        self.source_norms = nn.ModuleDict({str(layer): nn.LayerNorm(self.embed_dim) for layer in expected})
        self.is_bottleneck = self.adapter_variant.startswith("bottleneck_")
        if self.is_bottleneck:
            self.down_projections = nn.ModuleDict({str(layer): nn.Linear(self.embed_dim, self.bottleneck_dim) for layer in expected})
            self.up_projections = nn.ModuleDict({str(layer): nn.Linear(self.bottleneck_dim, self.embed_dim) for layer in expected})
            self.single_projections = nn.ModuleDict()
            gate_input_dim = self.bottleneck_dim
            for projection in self.up_projections.values():
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
        else:
            self.down_projections = nn.ModuleDict()
            self.up_projections = nn.ModuleDict()
            self.single_projections = nn.ModuleDict({str(layer): nn.Linear(self.embed_dim, self.embed_dim) for layer in expected})
            gate_input_dim = self.embed_dim
            for projection in self.single_projections.values():
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)

        self.anchor_gate_norm = nn.LayerNorm(self.embed_dim) if expected else None
        self.anchor_gate_projection = nn.Linear(self.embed_dim, self.gate_dim) if expected else None
        self.source_gate_projections = nn.ModuleDict({
            str(layer): nn.Linear(gate_input_dim, self.gate_dim) for layer in expected
        })
        self.gate_scores = nn.ModuleDict({str(layer): nn.Linear(self.gate_dim, 1) for layer in expected})
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self.output_projection = nn.Linear(self.embed_dim, self.num_outputs)

    def _source_correction_and_gate_feature(self, source_layer: int, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = str(source_layer)
        normalized = self.source_norms[key](source)
        if self.is_bottleneck:
            hidden = self.down_projections[key](normalized)
            if self.adapter_variant == "bottleneck_sigmoid":
                hidden = torch.sigmoid(hidden)
            return self.up_projections[key](hidden), hidden
        preactivation = self.single_projections[key](normalized)
        if self.adapter_variant == "single_linear":
            correction = preactivation
        elif self.adapter_variant == "single_gelu":
            correction = F.gelu(preactivation)
        else:  # single_sigmoid: centered so a zero projection remains zero-residual.
            correction = 2.0 * torch.sigmoid(preactivation) - 1.0
        return correction, correction

    def _fuse_features(self, representations: Mapping[int, torch.Tensor], *, return_diagnostics: bool = False):
        requested = set(self.source_layers) | {self.exit_layer}
        missing = requested.difference(representations)
        if missing:
            raise ValueError(f"missing required representations: {sorted(missing)}")
        pooled, batch_dim = {}, None
        for layer in sorted(requested):
            tokens = representations[layer]
            if tokens.ndim != 3:
                raise ValueError(f"L{layer} tokens must have shape [B,N,D], got {tuple(tokens.shape)}")
            if tokens.shape[1] < 1 or tokens.shape[2] != self.embed_dim:
                raise ValueError(f"L{layer} tokens must have non-empty N and D={self.embed_dim}, got {tuple(tokens.shape)}")
            if batch_dim is None:
                batch_dim = tokens.shape[0]
            elif tokens.shape[0] != batch_dim:
                raise ValueError("all exit representations must have matching batch dimensions")
            pooled[layer] = tokens.mean(dim=1)
        anchor = pooled[self.exit_layer]
        diagnostics = {}
        if not self.source_layers:
            return (anchor, diagnostics) if return_diagnostics else anchor
        query = self.anchor_gate_projection(self.anchor_gate_norm(anchor))
        residual_sum = torch.zeros_like(anchor)
        anchor_norm = torch.linalg.vector_norm(anchor, dim=-1)
        for layer in self.source_layers:
            correction, gate_feature = self._source_correction_and_gate_feature(layer, pooled[layer])
            key = str(layer)
            gate = torch.sigmoid(self.gate_scores[key](torch.tanh(query + self.source_gate_projections[key](gate_feature))))
            residual = gate * correction
            residual_sum = residual_sum + residual
            if return_diagnostics:
                residual_norm = torch.linalg.vector_norm(residual, dim=-1)
                diagnostics[layer] = {
                    "gate": gate.squeeze(-1),
                    "residual_norm": residual_norm,
                    "relative_residual_norm": residual_norm / (anchor_norm + 1e-8),
                }
        fused = anchor + residual_sum
        return (fused, diagnostics) if return_diagnostics else fused

    def forward(self, representations: Mapping[int, torch.Tensor], return_diagnostics: bool = False):
        result = self._fuse_features(representations, return_diagnostics=return_diagnostics)
        if return_diagnostics:
            fused, diagnostics = result
            return self.output_projection(self.output_norm(fused)), diagnostics
        return self.output_projection(self.output_norm(result))


class FrozenEncoderResidualFusionAblationExit(nn.Module):
    """Frozen causal-prefix wrapper for a configurable adapter ablation head."""

    def __init__(self, encoder: nn.Module, exit_layer: int, num_outputs: int,
                 adapter_variant: str, bottleneck_dim: int = 64, gate_dim: int = 64):
        super().__init__()
        depth = int(getattr(encoder, "logical_depth", len(encoder.blocks)))
        self.exit_layer = int(exit_layer)
        if self.exit_layer not in ResidualFusionAblationExitHead.ALLOWED_EXITS or self.exit_layer > depth:
            raise ValueError(f"unsupported residual-fusion exit L{self.exit_layer} for encoder depth {depth}")
        self.source_layers = tuple(layer for layer in ResidualFusionAblationExitHead.ALLOWED_EXITS if layer < self.exit_layer)
        self.requested_layers = self.source_layers + (self.exit_layer,)
        self.encoder = encoder
        embed_dim = getattr(encoder, "exit_embed_dim", None) or encoder.pos_embed.shape[-1]
        self.head = ResidualFusionAblationExitHead(
            int(embed_dim), int(num_outputs), self.exit_layer, self.source_layers,
            adapter_variant, bottleneck_dim, gate_dim,
        )
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
            return self.encoder.forward_to_layers(x, self.requested_layers)

    def forward(self, x: torch.Tensor, return_diagnostics: bool = False):
        return self.head(self.representations(x), return_diagnostics=return_diagnostics)


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
