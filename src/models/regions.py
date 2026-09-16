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
    one fine patch, L4 regions contain local contiguous regions, and L6
    contains the complete rectangular prediction block. Regions with unequal
    sizes are padded and described by ``region_sizes``.
    """

    grid_size: int
    batch_size: int
    mask_count: int
    region_indices: Mapping[int, torch.Tensor]
    region_sizes: Mapping[int, torch.Tensor]

    def for_layer(self, layer: int) -> torch.Tensor:
        try:
            return self.region_indices[int(layer)]
        except KeyError as error:
            raise ValueError(f"no P2 region layout is available for L{layer}") from error

    def sizes_for_layer(self, layer: int) -> torch.Tensor:
        try:
            return self.region_sizes[int(layer)]
        except KeyError as error:
            raise ValueError(f"no P2 region sizes are available for L{layer}") from error


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


def _local_regions(block: torch.Tensor) -> list[torch.Tensor]:
    """Partition one masked rectangle into contiguous L4 regions.

    The canonical even-by-even case retains exact non-overlapping ``2x2``
    regions. For odd dimensions, split each axis into at most two contiguous
    balanced bands. The resulting quadrant regions may have different sizes,
    which are padded by ``_pack_regions`` and weighted correctly during target
    pooling and positional averaging.
    """

    height, width = block.shape
    if height % 2 == 0 and width % 2 == 0:
        return [
            region.reshape(-1)
            for region in block.reshape(height // 2, 2, width // 2, 2)
            .permute(0, 2, 1, 3)
            .reshape(-1, 4)
        ]

    row_bands = [(0, height)] if height < 2 else [(0, (height + 1) // 2), ((height + 1) // 2, height)]
    col_bands = [(0, width)] if width < 2 else [(0, (width + 1) // 2), ((width + 1) // 2, width)]
    return [block[row_start:row_end, col_start:col_end].reshape(-1)
            for row_start, row_end in row_bands
            for col_start, col_end in col_bands]


def _pack_regions(regions_by_mask: list[list[list[torch.Tensor]]], *, description: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-size regions while retaining explicit membership counts."""

    if not regions_by_mask or not regions_by_mask[0]:
        raise ValueError(f"{description} cannot be empty")
    region_count = len(regions_by_mask[0][0])
    if region_count < 1:
        raise ValueError(f"{description} must contain at least one region")
    max_members = max(
        int(region.numel())
        for regions_by_batch in regions_by_mask
        for regions in regions_by_batch
        for region in regions
    )
    mask_count = len(regions_by_mask)
    batch_size = len(regions_by_mask[0])
    packed = torch.zeros(
        (mask_count, batch_size, region_count, max_members),
        dtype=torch.long,
        device=regions_by_mask[0][0][0].device,
    )
    sizes = torch.zeros((mask_count, batch_size, region_count), dtype=torch.long, device=packed.device)
    for mask_index, regions_by_batch in enumerate(regions_by_mask):
        if len(regions_by_batch) != batch_size:
            raise ValueError(f"{description} has inconsistent batch dimensions")
        for batch_index, regions in enumerate(regions_by_batch):
            if len(regions) != region_count:
                raise ValueError(
                    f"{description} has inconsistent region counts: expected {region_count}, got {len(regions)}"
                )
            for region_index, region in enumerate(regions):
                region = region.reshape(-1).to(dtype=torch.long)
                if region.numel() == 0:
                    raise ValueError(f"{description} contains an empty region")
                packed[mask_index, batch_index, region_index, : region.numel()] = region
                sizes[mask_index, batch_index, region_index] = region.numel()
    return packed, sizes


def build_mask_relative_region_layout(
    masks_pred: torch.Tensor | Sequence[torch.Tensor],
    *,
    grid_size: int,
) -> MaskRelativeRegionLayout:
    """Derive P2 regions from the actual, fully masked prediction blocks.

    P2 supports contiguous rectangular blocks of any positive geometry. Even
    blocks use exact local L4 ``2x2`` regions. Odd blocks use contiguous,
    balanced quadrant regions with explicit per-region sizes, while L6 always
    uses the complete masked block. No global or adaptive pooling assumption
    is used.
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
            fine_regions = [patch.reshape(-1) for patch in block.reshape(-1, 1)]
            local_regions = _local_regions(block)
            coarse_region = [block.reshape(-1)]

            # The construction is intentionally defensive: every target token
            # must be one of this sample's prediction-mask tokens.
            masked_tokens = torch.sort(sample_mask.to(dtype=torch.long)).values
            for regions in (fine_regions, local_regions, coarse_region):
                for region in regions:
                    if not torch.isin(region, masked_tokens).all():
                        raise AssertionError("P2 target region contains a visible patch")

            fine_by_batch.append(fine_regions)
            local_by_batch.append(local_regions)
            coarse_by_batch.append(coarse_region)

        fine_by_mask.append(fine_by_batch)
        local_by_mask.append(local_by_batch)
        coarse_by_mask.append(coarse_by_batch)

    fine_indices, fine_sizes = _pack_regions(fine_by_mask, description="L2 regions")
    local_indices, local_sizes = _pack_regions(local_by_mask, description="L4 regions")
    coarse_indices, coarse_sizes = _pack_regions(coarse_by_mask, description="L6 regions")

    return MaskRelativeRegionLayout(
        grid_size=grid_size,
        batch_size=batch_size,
        mask_count=len(masks),
        region_indices={
            2: fine_indices,
            4: local_indices,
            6: coarse_indices,
        },
        region_sizes={
            2: fine_sizes,
            4: local_sizes,
            6: coarse_sizes,
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
    region_sizes = layout.sizes_for_layer(layer).to(device=target_tokens.device, dtype=torch.long)
    mask_count, layout_batch_size, regions, fine_patches = region_indices.shape
    if layout_batch_size != batch_size:
        raise ValueError("region layout batch axis does not match target tokens")
    if tuple(region_sizes.shape) != (mask_count, batch_size, regions):
        raise ValueError("region size axes do not match region indices")
    if torch.any(region_sizes < 1) or torch.any(region_sizes > fine_patches):
        raise ValueError("region sizes must be within the padded fine-patch axis")

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
    fine_patches = gathered.size(-2)
    region_sizes = layout.sizes_for_layer(layer).to(device=gathered.device, dtype=torch.long)
    valid_members = torch.arange(fine_patches, device=gathered.device).reshape(1, 1, 1, 1, -1, 1)
    valid_members = valid_members < region_sizes.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
    pooled = (gathered * valid_members.to(dtype=gathered.dtype)).sum(dim=-2)
    pooled = pooled / region_sizes.unsqueeze(2).unsqueeze(-1).to(dtype=gathered.dtype)
    return F.layer_norm(pooled, (pooled.size(-1),))
