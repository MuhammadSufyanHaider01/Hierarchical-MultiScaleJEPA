"""Configuration exports and helpers for reproducible JEPA variants."""

from .jepa import Config as JEPAConfig
from .jepa import get_config as get_jepa_config
from .jepa import init_config as init_jepa_config


CNN_VARIANT_BASES = {
    "cnn_p0": "p0",
    "cnn_p1": "p1",
    "cnn_p2": "p2",
    "cnn_p2_reverse": "p2",
}


def configure_method_variant(config: JEPAConfig, method: str) -> JEPAConfig:
    """Apply architecture/objective variants while preserving shared YAML recipes."""
    if method.startswith("cnn_"):
        if method not in CNN_VARIANT_BASES:
            raise ValueError(f"unsupported CNN method: {method!r}")
        config.encoder.backbone = "convnext_v2_hier"
        config.encoder.stage_depths = (2, 2, 6, 2)
        config.encoder.stage_channels = (48, 96, 192, 384)
        config.encoder.mask_value = 0.0
        config.encoder.gradient_checkpointing = True
    if method in {"p2_reverse", "cnn_p2_reverse"}:
        config.hierarchy.mode = "mask_relative_coarse_to_fine"
    if method == "p3":
        config.hierarchy.gradient_routing = "pair_local"
    return config


__all__ = [
    "JEPAConfig",
    "get_jepa_config",
    "init_jepa_config",
    "configure_method_variant",
]
