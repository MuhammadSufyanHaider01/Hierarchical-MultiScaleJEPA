"""Classification-only entropy routing for already-trained frozen exits."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from models.exit_heads import ExitHead


def normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Return predictive entropy normalized to [0, 1] for each example."""

    if logits.ndim != 2 or logits.size(1) < 2:
        raise ValueError(f"classification logits must be [B, K] with K >= 2, got {tuple(logits.shape)}")
    probabilities = logits.softmax(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum(dim=-1)
    return entropy / torch.log(torch.tensor(float(logits.size(1)), device=logits.device, dtype=logits.dtype))


def route_logits(logits_by_layer: Mapping[int, torch.Tensor], threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the earliest confident prediction, falling back to the deepest exit."""

    layers = tuple(sorted(int(layer) for layer in logits_by_layer))
    if len(layers) < 2:
        raise ValueError("entropy routing requires at least two exits")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("normalized entropy threshold must be in [0, 1]")
    reference = logits_by_layer[layers[-1]]
    if any(tuple(logits_by_layer[layer].shape) != tuple(reference.shape) for layer in layers):
        raise ValueError("all exits must provide logits with the same [B, K] shape")
    selected_logits = reference.clone()
    selected_layers = torch.full((reference.size(0),), layers[-1], dtype=torch.long, device=reference.device)
    unresolved = torch.ones(reference.size(0), dtype=torch.bool, device=reference.device)
    for layer in layers[:-1]:
        confident = normalized_entropy(logits_by_layer[layer]) <= threshold
        take = unresolved & confident
        selected_logits[take] = logits_by_layer[layer][take]
        selected_layers[take] = layer
        unresolved[take] = False
    return selected_logits, selected_layers


def macro_accuracy(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    predictions = logits.argmax(dim=-1)
    correct = torch.zeros(num_classes, device=logits.device)
    total = torch.zeros(num_classes, device=logits.device)
    total.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    correct.scatter_add_(0, labels, (predictions == labels).to(torch.float32))
    present = total > 0
    return float((correct[present] / total[present]).mean().item())


def choose_entropy_threshold(
    validation_logits: Mapping[int, torch.Tensor],
    labels: torch.Tensor,
    *,
    num_classes: int,
    tolerance: float,
    thresholds: tuple[float, ...] = tuple(index / 100 for index in range(101)),
) -> dict[str, float | int]:
    """Select a validation-only threshold with near-L8 accuracy and minimal depth."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    deepest = max(validation_logits)
    full_accuracy = macro_accuracy(validation_logits[deepest], labels, num_classes)
    required_accuracy = full_accuracy * (1.0 - tolerance)
    candidates = []
    for threshold in thresholds:
        routed_logits, routed_layers = route_logits(validation_logits, threshold)
        accuracy = macro_accuracy(routed_logits, labels, num_classes)
        if accuracy >= required_accuracy:
            candidates.append((float(routed_layers.float().mean().item()), -accuracy, threshold))
    if not candidates:
        raise RuntimeError("no entropy threshold satisfies the validation tolerance")
    mean_layer, negative_accuracy, threshold = min(candidates)
    return {
        "threshold": threshold,
        "validation_macro_accuracy": -negative_accuracy,
        "validation_l8_macro_accuracy": full_accuracy,
        "required_validation_macro_accuracy": required_accuracy,
        "validation_mean_exit_layer": mean_layer,
    }


class DynamicFrozenExits(nn.Module):
    """Run a frozen encoder once and expose classification logits at each exit."""

    def __init__(self, encoder: nn.Module, heads: Mapping[int, ExitHead]):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict({str(int(layer)): head for layer, head in heads.items()})
        self.layers = tuple(sorted(int(layer) for layer in heads))
        if not self.layers:
            raise ValueError("at least one exit head is required")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward_all(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        self.encoder.eval()
        with torch.no_grad():
            tokens = self.encoder._prepare_tokens(x)
            logits = {}
            for layer, block in enumerate(self.encoder.blocks, start=1):
                tokens = block(tokens)
                if layer in self.layers:
                    logits[layer] = self.heads[str(layer)](self.encoder.norm(tokens))
        return logits

    def forward(self, x: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward_all(x)
        return route_logits(logits, threshold)
