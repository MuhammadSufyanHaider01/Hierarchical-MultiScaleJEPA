"""Mask-relative target regions for Version 1 P2 hierarchical JEPA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


P2_LAYERS = (2, 4, 6)


@dataclass(frozen=True)
class MaskRelativeRegionLayout:
    """Fine-token membership for each P2 target region.

    Every tensor in ``region_indices`` has explicit
    ``[prediction_mask, batch, region, fine_patch]`` axes. L2 regions contain
    one fine patch, L4 regions contain four local patches, and L6 contains the
    complete rectangular prediction block.
    """

    grid_size: int
    batch_size: int
    mask_count: int
    region_indices: Mapping[int, torch.Tensor]

    def for_layer(self, layer: int) -> torch.Tensor:
        try:
            return self.region_indices[int(layer)]
        except KeyError as error:
            raise ValueError(f"no P2 region layout is available for L{layer}") from error


def _as_mask_list(masks_pred: torch.Tensor | Sequence[torch.Tensor]) -> list[torch.Tensor]:
    if torch.is_tensor(masks_pred):
        if masks_pred.ndim == 2:
            return [masks_pred]
        if masks_pred.ndim == 3:
            return [masks_pred[index] for index in range(masks_pred.size(0))]
        raise ValueError(
            "prediction masks must have shape [B, N] or [M, B, N], "
            f"got {tuple(masks_pred.shape)}"
        )
    masks = list(masks_pred)
    if not masks:
        raise ValueError("at least one prediction mask is required for P2")
    return masks


def _rectangular_mask_indices(mask: torch.Tensor, grid_size: int, *, description: str) -> torch.Tensor:
    if mask.ndim != 1 or mask.numel() == 0:
        raise ValueError(f"{description} must contain at least one flattened patch index")
    if mask.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"{description} must use an integer dtype, got {mask.dtype}")

    mask = mask.to(dtype=torch.long)
    num_patches = grid_size * grid_size
    if torch.any(mask < 0) or torch.any(mask >= num_patches):
        raise ValueError(f"{description} contains an index outside [0, {num_patches})")

    ordered = torch.sort(mask).values
    if torch.unique_consecutive(ordered).numel() != ordered.numel():
        raise ValueError(f"{description} contains duplicate patch indices")

    rows = torch.div(ordered, grid_size, rounding_mode="floor")
    cols = ordered.remainder(grid_size)
    unique_rows = torch.unique(rows, sorted=True)
    unique_cols = torch.unique(cols, sorted=True)
    height, width = unique_rows.numel(), unique_cols.numel()
    if height * width != ordered.numel():
        raise ValueError(f"{description} is not a complete rectangular block")

    expected_rows = torch.arange(
        unique_rows[0], unique_rows[0] + height, device=mask.device, dtype=torch.long
    )
    expected_cols = torch.arange(
        unique_cols[0], unique_cols[0] + width, device=mask.device, dtype=torch.long
    )
    if not torch.equal(unique_rows, expected_rows) or not torch.equal(unique_cols, expected_cols):
        raise ValueError(f"{description} is not a contiguous rectangular block")

    expected = (expected_rows[:, None] * grid_size + expected_cols[None, :]).reshape(-1)
    if not torch.equal(ordered, expected):
        raise ValueError(f"{description} is not a complete rectangular block")
    return expected.reshape(height, width)


def build_mask_relative_region_layout(
    masks_pred: torch.Tensor | Sequence[torch.Tensor],
    *,
    grid_size: int,
) -> MaskRelativeRegionLayout:
    """Derive P2 regions from the actual, fully masked prediction blocks.

    P2 supports contiguous rectangular blocks whose height and width are both
    divisible by two. This permits exact local L4 ``2x2`` regions and an L6
    whole-block region without global or adaptive pooling assumptions.
    """

    if grid_size < 2:
        raise ValueError(f"P2 requires a grid of at least 2x2, got {grid_size}")
    masks = _as_mask_list(masks_pred)
    first = masks[0]
    if first.ndim != 2:
        raise ValueError(f"prediction mask 0 must have shape [B, N], got {tuple(first.shape)}")
    batch_size, token_count = first.shape
    if token_count == 0:
        raise ValueError("prediction masks must contain at least one patch")

    fine_by_mask: list[torch.Tensor] = []
    local_by_mask: list[torch.Tensor] = []
    coarse_by_mask: list[torch.Tensor] = []
    for mask_index, mask in enumerate(masks):
        if mask.ndim != 2 or tuple(mask.shape) != (batch_size, token_count):
            raise ValueError(
                "all prediction masks must share shape "
                f"[{batch_size}, {token_count}], got {tuple(mask.shape)} at mask {mask_index}"
            )
        fine_by_batch: list[torch.Tensor] = []
        local_by_batch: list[torch.Tensor] = []
        coarse_by_batch: list[torch.Tensor] = []
        for batch_index, sample_mask in enumerate(mask):
            block = _rectangular_mask_indices(
                sample_mask,
                grid_size,
                description=f"prediction mask {mask_index}, sample {batch_index}",
            )
            height, width = block.shape
            if height % 2 or width % 2:
                raise ValueError(
                    f"prediction mask {mask_index}, sample {batch_index} has geometry "
                    f"{height}x{width}; P2 L4 requires both dimensions divisible by two"
                )

            fine_regions = block.reshape(-1, 1)
            local_regions = (
                block.reshape(height // 2, 2, width // 2, 2)
                .permute(0, 2, 1, 3)
                .reshape(-1, 4)
            )
            coarse_region = block.reshape(1, -1)

            # The construction is intentionally defensive: every target token
            # must be one of this sample's prediction-mask tokens.
            masked_tokens = torch.sort(sample_mask.to(dtype=torch.long)).values
            for regions in (fine_regions, local_regions, coarse_region):
                if not torch.isin(regions.reshape(-1), masked_tokens).all():
                    raise AssertionError("P2 target region contains a visible patch")

            fine_by_batch.append(fine_regions)
            local_by_batch.append(local_regions)
            coarse_by_batch.append(coarse_region)

        fine_by_mask.append(torch.stack(fine_by_batch))
        local_by_mask.append(torch.stack(local_by_batch))
        coarse_by_mask.append(torch.stack(coarse_by_batch))

    return MaskRelativeRegionLayout(
        grid_size=grid_size,
        batch_size=batch_size,
        mask_count=len(masks),
        region_indices={
            2: torch.stack(fine_by_mask),
            4: torch.stack(local_by_mask),
            6: torch.stack(coarse_by_mask),
        },
    )


def gather_target_regions(
    target_tokens: torch.Tensor,
    *,
    channels: int,
    layout: MaskRelativeRegionLayout,
    layer: int,
) -> torch.Tensor:
    """Gather unnormalized full-grid target tokens by explicit P2 membership.

    The returned tensor has axes
    ``[prediction_mask, batch, channel, region, fine_patch, embedding]``.
    """

    if target_tokens.ndim != 3:
        raise ValueError(f"target tokens must have shape [B, C*P, D], got {tuple(target_tokens.shape)}")
    batch_size, flattened_tokens, embedding_dim = target_tokens.shape
    if batch_size != layout.batch_size:
        raise ValueError(
            f"target batch size {batch_size} does not match region layout batch size {layout.batch_size}"
        )
    if channels < 1 or flattened_tokens % channels:
        raise ValueError(
            f"target token count {flattened_tokens} is incompatible with channels={channels}"
        )
    patches_per_channel = flattened_tokens // channels
    if patches_per_channel != layout.grid_size * layout.grid_size:
        raise ValueError(
            f"target grid has {patches_per_channel} patches per channel, expected "
            f"{layout.grid_size * layout.grid_size}"
        )

    region_indices = layout.for_layer(layer).to(device=target_tokens.device, dtype=torch.long)
    mask_count, layout_batch_size, regions, fine_patches = region_indices.shape
    if layout_batch_size != batch_size:
        raise ValueError("region layout batch axis does not match target tokens")

    target_grid = target_tokens.reshape(batch_size, channels, patches_per_channel, embedding_dim)
    expanded_target = target_grid.unsqueeze(0).unsqueeze(3).expand(
        mask_count,
        batch_size,
        channels,
        regions,
        patches_per_channel,
        embedding_dim,
    )
    gather_indices = region_indices[:, :, None, :, :, None].expand(
        mask_count,
        batch_size,
        channels,
        regions,
        fine_patches,
        embedding_dim,
    )
    return torch.gather(expanded_target, dim=4, index=gather_indices)


def pool_target_regions(
    target_tokens: torch.Tensor,
    *,
    channels: int,
    layout: MaskRelativeRegionLayout,
    layer: int,
) -> torch.Tensor:
    """Mean-pool target regions, then apply the Version 1 target normalization."""

    gathered = gather_target_regions(
        target_tokens,
        channels=channels,
        layout=layout,
        layer=layer,
    )
    pooled = gathered.mean(dim=-2)
    return F.layer_norm(pooled, (pooled.size(-1),))
