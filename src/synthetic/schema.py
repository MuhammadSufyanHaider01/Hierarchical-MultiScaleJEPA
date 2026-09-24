"""Configuration and compatibility rules for synthetic availability tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SyntheticTaskProfile:
    name: str
    target_type: str
    num_outputs: int
    metric_name: str
    higher_is_better: bool
    temporal_axis: int
    reference_channels: int
    reference_time_patches: int
    antenna_levels: tuple[int, ...]
    temporal_levels: tuple[int, ...]
    sample_key: str
    label_key: str
    feature_mode: str | None
    label_map: dict[int, str]
    assumptions_approved: bool
    assumptions: tuple[str, ...]
    coord_min: tuple[float, ...] | None
    coord_max: tuple[float, ...] | None
    semantics_verified: bool
    specification_status: str
    array_subset_policy: str
    requires_source_verified_semantics: bool
    unresolved_requirements: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "SyntheticTaskProfile":
        labels = value.get("label_map") or {}
        coordinates = value.get("coordinates") or {}
        return cls(
            name=str(value["name"]),
            target_type=str(value["target_type"]),
            num_outputs=int(value["num_outputs"]),
            metric_name=str(value["metric_name"]),
            higher_is_better=bool(value["higher_is_better"]),
            temporal_axis=int(value["temporal_axis"]),
            reference_channels=int(value["reference_channels"]),
            reference_time_patches=int(value["reference_time_patches"]),
            antenna_levels=tuple(int(v) for v in value["antenna_levels"]),
            temporal_levels=tuple(int(v) for v in value["temporal_levels"]),
            sample_key=str(value["sample_key"]),
            label_key=str(value["label_key"]),
            feature_mode=value.get("feature_mode"),
            label_map={int(k): str(v) for k, v in labels.items()},
            assumptions_approved=bool(value.get("assumptions_approved", False)),
            assumptions=tuple(str(item) for item in value.get("assumptions", ())),
            coord_min=tuple(float(v) for v in coordinates["min"]) if coordinates.get("min") else None,
            coord_max=tuple(float(v) for v in coordinates["max"]) if coordinates.get("max") else None,
            semantics_verified=bool(value.get("semantics_verified", False)),
            requires_source_verified_semantics=bool(value.get("requires_source_verified_semantics", False)),
            unresolved_requirements=tuple(str(item) for item in value.get("unresolved_requirements", ())),
            specification_status=str(value.get("specification_status", "unresolved")),
            array_subset_policy=str(value.get("array_subset_policy", "fixed_aperture_uniform")),
        )

    def semantic_blockers(self) -> list[str]:
        blockers: list[str] = []
        if not self.semantics_verified and self.specification_status != "assumption_based_synthetic_domain":
            blockers.append("task signal/label semantics are not marked verified")
        if self.specification_status not in {
            "source_verified",
            "distribution_calibrated_design",
            "assumption_based_synthetic_domain",
        }:
            blockers.append(
                "specification_status must be source_verified or distribution_calibrated_design"
            )
        if self.requires_source_verified_semantics and self.specification_status != "source_verified":
            blockers.append("this task requires source-verified semantics; a distribution-calibrated design is insufficient")
        if self.specification_status == "assumption_based_synthetic_domain":
            if not self.assumptions_approved:
                blockers.append("assumption-based generation requires explicit user approval")
            if not self.assumptions:
                blockers.append("assumption-based generation must enumerate its assumptions")
        blockers.extend(self.unresolved_requirements)
        if not self.feature_mode:
            blockers.append("feature_mode is missing")
        if self.target_type == "classification":
            if len(self.label_map) != self.num_outputs:
                blockers.append(
                    f"label_map has {len(self.label_map)} entries, expected {self.num_outputs}"
                )
            if sorted(self.label_map) != list(range(self.num_outputs)):
                blockers.append("label_map keys must match the frozen head's contiguous output IDs")
        if self.target_type == "position" and (self.coord_min is None or self.coord_max is None):
            blockers.append("position coordinate bounds are missing")
        if self.temporal_axis not in (1, 2):
            blockers.append("temporal_axis must be 1 or 2 for [C,H,W] samples")
        if self.reference_channels < 1 or self.reference_time_patches < 1:
            blockers.append("reference input dimensions must be positive")
        if self.array_subset_policy not in {
            "fixed_aperture_uniform",
            "site_balanced_nested",
            "fixed_aperture_center_out_nested",
        }:
            blockers.append(f"unsupported array_subset_policy {self.array_subset_policy!r}")
        return blockers

def load_task_profile(
    path: str | Path, task_name: str | None = None
) -> tuple[SyntheticTaskProfile, dict[str, Any]]:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise TypeError(f"task profile must be a YAML mapping: {path}")
    if isinstance(config.get("task"), dict):
        task_record = config["task"]
    elif task_name is not None and isinstance(config.get("tasks"), dict):
        if task_name not in config["tasks"]:
            raise KeyError(f"study profile {path} has no task {task_name}")
        task_record = config["tasks"][task_name]
    else:
        raise KeyError(f"profile {path} must contain a task record or a named tasks mapping")
    if task_name is not None and task_record.get("name", task_name) != task_name:
        raise ValueError(f"requested task {task_name} differs from profile name {task_record.get('name')}")
    return SyntheticTaskProfile.from_mapping(task_record), config

def require_verified_semantics(profile: SyntheticTaskProfile) -> None:
    blockers = profile.semantic_blockers()
    if blockers:
        details = "; ".join(blockers)
        raise RuntimeError(
            f"synthetic generation is blocked for {profile.name}: {details}. "
            "Verify the source specification or explicitly record and approve an "
            "assumption-based synthetic domain. Never describe assumptions as recovered source semantics."
        )
