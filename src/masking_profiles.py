"""Registered masking profiles for the controlled masking study.

Profiles intentionally contain only mask overrides.  The selected P0/P2 YAML
file remains the source of all model, optimizer, EMA, scheduler, dataset, and
seed settings.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIRECTORY = REPOSITORY_ROOT / "configs" / "masking_study"
PROFILE_NAMES = ("frequency_n6", "region_n6", "time_n6", "random_75", "adaptive_frequency_n6")


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _validate_profile(name: str, profile: dict[str, Any]) -> None:
    if profile.get("name") != name:
        raise ValueError(f"mask profile name mismatch: requested {name!r}, file declares {profile.get('name')!r}")
    masks = profile.get("masks")
    if not isinstance(masks, dict):
        raise ValueError(f"mask profile {name!r} must contain a mapping under 'masks'")

    expected = {
        "frequency_n6": ("multi_block", "span_time"),
        "region_n6": ("multi_block", "random"),
        "time_n6": ("multi_block", "span_freq"),
    }
    if name in expected:
        mask_type, aspect_ratio_type = expected[name]
        if masks.get("type") != mask_type:
            raise ValueError(f"{name} must use masks.type={mask_type!r}")
        multi_block = masks.get("multi_block", {})
        if multi_block.get("aspect_ratio_type") != aspect_ratio_type:
            raise ValueError(f"{name} must use aspect_ratio_type={aspect_ratio_type!r}")
        if multi_block.get("num_pred_masks") != 6:
            raise ValueError(f"{name} must use num_pred_masks=6")
    elif name == "random_75":
        if masks.get("type") != "random":
            raise ValueError("random_75 must use masks.type='random'")
        ratios = masks.get("random", {}).get("ratios")
        if ratios != [0.75, 0.75]:
            raise ValueError("random_75 must use the fixed ratios [0.75, 0.75]")
    elif name == "adaptive_frequency_n6":
        if masks.get("type") != "variation_aware_multi_block":
            raise ValueError(
                "adaptive_frequency_n6 must use masks.type='variation_aware_multi_block'"
            )
        adaptive = masks.get("variation_aware_multi_block", {})
        if adaptive.get("aspect_ratio_type") != "span_time":
            raise ValueError("adaptive_frequency_n6 must use aspect_ratio_type='span_time'")
        if adaptive.get("num_pred_masks") != 6:
            raise ValueError("adaptive_frequency_n6 must use num_pred_masks=6")
        if adaptive.get("alpha") != 0.5:
            raise ValueError("adaptive_frequency_n6 must use alpha=0.5")
        if adaptive.get("exploration_eta") != 0.25:
            raise ValueError("adaptive_frequency_n6 must use exploration_eta=0.25")
        if adaptive.get("epsilon") != 1.0e-6:
            raise ValueError("adaptive_frequency_n6 must use epsilon=1e-6")
    else:
        raise ValueError(f"unregistered mask profile: {name!r}")


def load_mask_profile(name: str) -> tuple[dict[str, Any], Path, str]:
    """Load and validate one registered profile, returning its file checksum."""

    if name not in PROFILE_NAMES:
        raise ValueError(f"unknown mask profile {name!r}; choose from {', '.join(PROFILE_NAMES)}")
    path = PROFILE_DIRECTORY / f"{name}.yml"
    if not path.is_file():
        raise FileNotFoundError(f"registered mask profile is missing: {path}")
    profile = yaml.safe_load(path.read_text())
    if not isinstance(profile, dict):
        raise ValueError(f"mask profile {path} must contain a YAML mapping")
    _validate_profile(name, profile)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return profile, path, checksum


def apply_mask_profile(config: Any, name: str) -> dict[str, Any]:
    """Apply only the selected profile's mask overrides to a loaded config."""

    profile, path, checksum = load_mask_profile(name)
    base_masks = config.masks.model_dump(mode="json")
    resolved_masks = _deep_merge(base_masks, profile["masks"])
    config.masks = config.masks.__class__(**resolved_masks)
    return {
        "name": name,
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "sha256": checksum,
        "overrides": profile["masks"],
        "resolved": config.masks.model_dump(mode="json"),
    }
