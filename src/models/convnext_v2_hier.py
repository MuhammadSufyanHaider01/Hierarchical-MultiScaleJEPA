"""ConvNeXt V2-style hierarchical encoder for JEPA.

The model exposes four logical exits (L2/L4/L6/L8) while retaining the
14-by-14 token grid consumed by the repository's existing JEPA predictors.
Context masks are applied to image patches before convolution so masked input
values cannot leak into neighboring context features.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .utils import apply_masks


class GlobalResponseNorm(nn.Module):
    """ConvNeXt V2 global response normalization over spatial positions."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        response = torch.linalg.vector_norm(x, ord=2, dim=(1, 2), keepdim=True)
        normalized = response / (response.mean(dim=-1, keepdim=True) + self.eps)
        return x + self.gamma * (x * normalized) + self.beta


class ConvNeXtV2Block(nn.Module):
    """A native PyTorch ConvNeXt V2 block using NHWC pointwise operations."""

    def __init__(self, channels: int, expansion: int = 4, eps: float = 1e-6):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = nn.LayerNorm(channels, eps=eps)
        self.expand = nn.Linear(channels, channels * expansion)
        self.activation = nn.GELU()
        self.grn = GlobalResponseNorm(channels * expansion, eps=eps)
        self.project = nn.Linear(channels * expansion, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x).permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.grn(x)
        x = self.project(x).permute(0, 3, 1, 2)
        return residual + x


class CrossChannelFusion(nn.Module):
    """Residual per-channel fusion with the mean representation of all inputs."""

    def __init__(self, channels: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.alpha = nn.Parameter(torch.zeros(1, 1, channels, 1, 1))

    def forward(self, x: torch.Tensor, batch: int, physical_channels: int) -> torch.Tensor:
        _, channels, height, width = x.shape
        grouped = x.reshape(batch, physical_channels, channels, height, width)
        mean = grouped.mean(dim=1, keepdim=True).expand_as(grouped)
        joined = torch.cat((grouped, mean), dim=2).reshape(
            batch * physical_channels, channels * 2, height, width
        )
        update = self.fusion(joined).reshape_as(grouped)
        return grouped + self.alpha * update


class ConvNeXtV2HierEncoder(nn.Module):
    """Shared per-channel CNN with canonical latent tokens at four exits.

    The defaults implement the Femto-sized stage specification from the
    CNN-HierJEPA design: depths (2, 2, 6, 2), widths (48, 96, 192, 384), and
    256-dimensional tokens on the patch-size-defined 14x14 grid.
    """

    exit_layers = (2, 4, 6, 8)
    logical_depth = 8
    has_stage_local_routing = True

    def __init__(
        self,
        img_size: tuple[int, int] = (224, 224),
        patch_size: int = 16,
        embed_dim: int = 256,
        stage_depths: Sequence[int] = (2, 2, 6, 2),
        stage_channels: Sequence[int] = (48, 96, 192, 384),
        mask_value: float = 0.0,
        gradient_checkpointing: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if len(stage_depths) != 4 or len(stage_channels) != 4:
            raise ValueError("ConvNeXt V2 requires exactly four stage depths and widths")
        if any(int(depth) < 1 for depth in stage_depths):
            raise ValueError(f"stage depths must be positive, got {tuple(stage_depths)}")
        if patch_size < 4 or patch_size % 4:
            raise ValueError("patch_size must be a positive multiple of the 4-pixel CNN stem")
        self.img_size = tuple(int(value) for value in img_size)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.exit_embed_dim = int(embed_dim)
        self.base_grid_size = (
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
        )
        self.mask_value = float(mask_value)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        widths = tuple(int(value) for value in stage_channels)
        self.stage_channels = widths

        self.stem = nn.Conv2d(1, widths[0], kernel_size=4, stride=4)
        self.downsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(widths[index], eps=norm_eps),
                    nn.Conv2d(widths[index], widths[index + 1], kernel_size=2, stride=2),
                )
                for index in range(3)
            ]
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *(ConvNeXtV2Block(width, eps=norm_eps) for _ in range(int(depth)))
                )
                for width, depth in zip(widths, stage_depths, strict=True)
            ]
        )
        self.channel_fusions = nn.ModuleList(CrossChannelFusion(width) for width in widths)
        self.adapters = nn.ModuleList(nn.Conv2d(width, embed_dim, kernel_size=1) for width in widths)
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.apply(self._init_weights)

    @property
    def blocks(self) -> tuple[int, ...]:
        """Compatibility view for callers that use the encoder's logical depth."""
        return tuple(range(1, self.logical_depth + 1))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        for layer in module.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def normalize_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(tokens)

    def _validate_input(self, x: torch.Tensor) -> tuple[int, int, int, int]:
        if x.ndim != 4:
            raise ValueError(f"expected input [B,C,H,W], got {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        if channels < 1:
            raise ValueError("the physical-channel dimension must be non-empty")
        if height % 32 or width % 32:
            raise ValueError(f"input dimensions must be divisible by 32, got {(height, width)}")
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"input dimensions must be divisible by patch_size={self.patch_size}, got {(height, width)}"
            )
        if (height // self.patch_size, width // self.patch_size) != self.base_grid_size:
            raise ValueError(
                "CNN input grid must match configured img_size; "
                f"expected {self.base_grid_size}, got {(height // self.patch_size, width // self.patch_size)}"
            )
        return batch, channels, height, width

    def _mask_context_input(self, x: torch.Tensor, masks: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        batch, channels, height, width = x.shape
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        outputs = []
        for mask_index, mask in enumerate(masks):
            if mask.ndim != 2 or mask.shape[0] != batch:
                raise ValueError(
                    f"encoder mask {mask_index} must have shape [B,K] with B={batch}, got {tuple(mask.shape)}"
                )
            if mask.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise TypeError(f"encoder mask indices must be integer-valued, got {mask.dtype}")
            if torch.any(mask < 0) or torch.any(mask >= grid_h * grid_w):
                raise ValueError(f"encoder mask {mask_index} contains indices outside the {grid_h}x{grid_w} grid")
            visible = torch.zeros(batch, grid_h * grid_w, dtype=torch.bool, device=x.device)
            visible.scatter_(1, mask.to(device=x.device, dtype=torch.long), True)
            pixels = x.reshape(batch, channels, grid_h, self.patch_size, grid_w, self.patch_size)
            pixels = pixels.masked_fill(
                ~visible.reshape(batch, 1, grid_h, 1, grid_w, 1), self.mask_value
            )
            outputs.append(pixels.reshape_as(x))
        if not outputs:
            raise ValueError("at least one context mask is required")
        return outputs

    def _run_stage(self, x: torch.Tensor, stage_index: int, batch: int, channels: int) -> torch.Tensor:
        for block in self.stages[stage_index]:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.channel_fusions[stage_index](x, batch, channels)
        return x.reshape(batch * channels, x.size(2), x.size(3), x.size(4))

    def _adapt_stage(self, feature: torch.Tensor, stage_index: int, batch: int, channels: int,
                     *, normalize: bool) -> torch.Tensor:
        height, width = self.base_grid_size
        if stage_index in (0, 1):  # fixed-factor pooling keeps deterministic CUDA backward supported
            if feature.shape[-2] % height or feature.shape[-1] % width:
                raise ValueError(f"stage {stage_index + 1} features cannot be evenly pooled to {(height, width)}")
            factor = (feature.shape[-2] // height, feature.shape[-1] // width)
            feature = F.avg_pool2d(feature, kernel_size=factor, stride=factor)
        elif stage_index == 2:
            if tuple(feature.shape[-2:]) != (height, width):
                raise ValueError(f"stage 3 should have {height}x{width} features, got {tuple(feature.shape[-2:])}")
        else:
            feature = F.interpolate(feature, size=(height, width), mode="nearest")
        feature = self.adapters[stage_index](feature)
        tokens = feature.reshape(batch, channels, self.embed_dim, height * width)
        tokens = tokens.permute(0, 1, 3, 2)
        tokens = tokens.reshape(batch, channels, height * width, self.embed_dim)
        if normalize:
            tokens = self.norm(tokens)
        return tokens

    def _encode_one(self, image: torch.Tensor, requested: set[int], *, normalize: bool) -> dict[int, torch.Tensor]:
        batch, channels, _, _ = image.shape
        feature = self.stem(image.reshape(batch * channels, 1, image.shape[-2], image.shape[-1]))
        outputs = {}
        for stage_index, layer in enumerate(self.exit_layers):
            if stage_index:
                # LayerNorm operates on NHWC, then the downsample convolution
                # returns NCHW for the depthwise stage blocks.
                downsample = self.downsamples[stage_index - 1]
                feature = downsample[0](feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                feature = downsample[1](feature)
            feature = self._run_stage(feature, stage_index, batch, channels)
            if layer in requested:
                outputs[layer] = self._adapt_stage(
                    feature, stage_index, batch, channels, normalize=normalize
                )
        return outputs

    def _pack_mask_outputs(
        self,
        outputs_by_mask: list[dict[int, torch.Tensor]],
        masks: Sequence[torch.Tensor] | None,
        requested: Sequence[int],
    ) -> dict[int, torch.Tensor]:
        packed = {}
        for layer in requested:
            mask_outputs = []
            for mask_index, outputs in enumerate(outputs_by_mask):
                tokens = outputs[layer]
                if masks is not None:
                    tokens = apply_masks(tokens, [masks[mask_index]])
                mask_outputs.append(tokens.reshape(tokens.shape[0], -1, self.embed_dim))
            packed[layer] = torch.cat(mask_outputs, dim=0)
        return packed

    def forward(
        self,
        x: torch.Tensor,
        masks=None,
        return_layers=None,
        *,
        normalize_exits: bool = True,
    ):
        batch, _, _, _ = self._validate_input(x)
        requested = self.exit_layers if return_layers is None else tuple(dict.fromkeys(int(v) for v in return_layers))
        invalid = sorted(set(requested).difference(self.exit_layers))
        if invalid:
            raise ValueError(f"CNN exits are {self.exit_layers}; unsupported logical layers: {invalid}")
        if masks is None:
            outputs = self._encode_one(x, set(requested), normalize=normalize_exits)
            if return_layers is not None:
                return {layer: outputs[layer].reshape(batch, -1, self.embed_dim) for layer in requested}
            return outputs[self.exit_layers[-1]].reshape(batch, -1, self.embed_dim)

        mask_list = masks if isinstance(masks, list) else [masks]
        masked_inputs = self._mask_context_input(x, mask_list)
        outputs_by_mask = [
            self._encode_one(masked, set(requested), normalize=normalize_exits)
            for masked in masked_inputs
        ]
        outputs = self._pack_mask_outputs(outputs_by_mask, mask_list, requested)
        return outputs if return_layers is not None else outputs[self.exit_layers[-1]]

    def forward_to_layer(self, x: torch.Tensor, exit_layer: int) -> torch.Tensor:
        exit_layer = int(exit_layer)
        if exit_layer not in self.exit_layers:
            raise ValueError(f"CNN exits are {self.exit_layers}, got L{exit_layer}")
        return self.forward(x, return_layers=[exit_layer])[exit_layer]

    def forward_local_layers(self, x: torch.Tensor, masks, return_layers: Sequence[int]) -> dict[int, torch.Tensor]:
        """Compute auxiliary exits with gradients isolated to their own stage."""
        batch, _, _, _ = self._validate_input(x)
        requested = tuple(dict.fromkeys(int(layer) for layer in return_layers))
        invalid = sorted(set(requested).difference(self.exit_layers[:-1]))
        if invalid:
            raise ValueError(f"local-gradient exits must be intermediate CNN layers, got {invalid}")
        mask_list = masks if isinstance(masks, list) else [masks]
        masked_inputs = self._mask_context_input(x, mask_list)
        outputs_by_mask = []
        for masked in masked_inputs:
            stage_outputs = {}
            for layer in requested:
                stage_index = self.exit_layers.index(layer)
                features = masked.reshape(batch * masked.size(1), 1, masked.size(-2), masked.size(-1))
                if stage_index == 0:  # fixed-factor pooling keeps deterministic CUDA backward supported
                    features = self.stem(features)
                else:
                    with torch.no_grad():
                        features = self.stem(features)
                        for prefix_index in range(stage_index):
                            features = self._run_stage(features, prefix_index, batch, masked.size(1))
                            if prefix_index < stage_index - 1:
                                features = self.downsamples[prefix_index][0](
                                    features.permute(0, 2, 3, 1)
                                ).permute(0, 3, 1, 2)
                                features = self.downsamples[prefix_index][1](features)
                    # The transition into the supervised stage is trainable,
                    # but consumes a detached representation from earlier stages.
                    features = self.downsamples[stage_index - 1][0](
                        features.detach().permute(0, 2, 3, 1)
                    ).permute(0, 3, 1, 2)
                    features = self.downsamples[stage_index - 1][1](features)
                features = self._run_stage(features, stage_index, batch, masked.size(1))
                tokens = self._adapt_stage(features, stage_index, batch, masked.size(1), normalize=True)
                tokens = apply_masks(tokens, [mask_list[len(outputs_by_mask)]])
                stage_outputs[layer] = tokens.reshape(tokens.shape[0], -1, self.embed_dim)
            outputs_by_mask.append(stage_outputs)
        return {
            layer: torch.cat([outputs[layer] for outputs in outputs_by_mask], dim=0)
            for layer in requested
        }
