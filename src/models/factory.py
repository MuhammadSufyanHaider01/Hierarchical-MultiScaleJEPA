"""Backbone construction helpers with backward-compatible ViT defaults."""

from __future__ import annotations

from .convnext_v2_hier import ConvNeXtV2HierEncoder
from .encoder import Encoder


def build_encoder(encoder_config):
    """Build the configured encoder while retaining legacy ViT checkpoints.

    Existing YAML files omit ``backbone`` and therefore continue to select the
    original LatentWave ViT. CNN checkpoints are reconstructed from the CNN
    fields stored in their serialized config.
    """
    backbone = getattr(encoder_config, "backbone", "vit")
    common = {
        "img_size": tuple(encoder_config.img_size),
        "patch_size": int(encoder_config.patch_size),
        "embed_dim": int(encoder_config.embed_dim),
    }
    if backbone == "vit":
        return Encoder(
            **common,
            depth=int(encoder_config.depth),
            num_heads=int(encoder_config.num_heads),
        )
    if backbone == "convnext_v2_hier":
        return ConvNeXtV2HierEncoder(
            **common,
            stage_depths=encoder_config.stage_depths,
            stage_channels=encoder_config.stage_channels,
            mask_value=encoder_config.mask_value,
            gradient_checkpointing=getattr(encoder_config, "gradient_checkpointing", False),
        )
    raise ValueError(f"unsupported encoder backbone: {backbone!r}")
