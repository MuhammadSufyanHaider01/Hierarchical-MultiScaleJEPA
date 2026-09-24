#!/usr/bin/env python3
"""Lock the formal realization count from paired pilot precision estimates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

TARGETS = {
    "nrpos-outdoor": ("mean_distance_error", 0.25, "m"),
    "interference": ("macro_accuracy", 0.03, "absolute"),
    "jamming": ("macro_accuracy", 0.03, "absolute"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TARGETS), required=True)
    parser.add_argument("--antenna-pilot", type=Path, required=True)
    parser.add_argument("--time-pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pilot_paths = []
    pilot_records = []
    for sweep, value in (("antenna", args.antenna_pilot), ("time", args.time_pilot)):
        path = value if value.is_absolute() else REPOSITORY_ROOT / value
        artifact = json.loads(path.read_text())
        if artifact.get("task") != args.task or artifact.get("sweep") != sweep:
            raise ValueError(f"{sweep} pilot anchor does not match task/sweep")
        if artifact.get("stage") != "pilot_precision_estimate":
            raise ValueError("sample count must use pilot anchor outputs, not formal test results")
        master = artifact["provenance"]["synthetic_master"]
        pilot_paths.append((master["path"], master["size_bytes"], master["modified_ns"]))
        pilot_records.append((path, artifact))
    if pilot_paths[0] != pilot_paths[1]:
        raise ValueError("antenna and time pilot anchors must use the same paired master")
    pilot_n = int(pilot_records[0][1].get("pilot_master_count", pilot_records[0][1]["sample_count"]))
    anchor_n = int(pilot_records[0][1]["sample_count"])
    if not 256 <= pilot_n <= 512 or any(int(row[1].get("pilot_master_count", row[1]["sample_count"])) != pilot_n for row in pilot_records) or any(int(row[1]["sample_count"]) != anchor_n for row in pilot_records) or not 2 <= anchor_n <= pilot_n:
        raise ValueError("paired pilot anchors must each contain the same 256-512 realizations")

    metric, target, units = TARGETS[args.task]
    half_widths = []
    for _, artifact in pilot_records:
        interval = artifact["precision_metric_ci"]
        if "half_width" not in interval:
            raise ValueError(f"pilot anchor does not provide a {metric} interval")
        half_widths.append(float(interval["half_width"]))
    conservative_half_width = max(half_widths)
    if not math.isfinite(conservative_half_width) or conservative_half_width < 0:
        raise ValueError("pilot precision estimate must be finite and non-negative")

    common = yaml.safe_load((REPOSITORY_ROOT / "configs" / "sionna" / "common.yml").read_text())
    cap = int(common["source_count_caps"][args.task])
    test_fraction = float(common["splits"]["test"])
    if conservative_half_width == 0:
        estimate = 512
    else:
        required_test_count = anchor_n * (conservative_half_width / target) ** 2
        estimate = max(512, math.ceil(required_test_count / test_fraction))
    formal_total = min(estimate, cap)
    projected_cap_half_width = conservative_half_width * math.sqrt(anchor_n / (formal_total * test_fraction))
    result = {
        "schema_version": 1,
        "stage": "sample_count_locked",
        "task": args.task,
        "pilot_realizations": pilot_n,
        "pilot_anchor_realizations": anchor_n,
        "pilot_master": {
            "path": pilot_paths[0][0],
            "size_bytes": pilot_paths[0][1],
            "modified_ns": pilot_paths[0][2],
        },
        "pilot_anchors": {
            sweep: {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "metric_half_width": float(artifact["precision_metric_ci"]["half_width"]),
            }
            for sweep, (path, artifact) in zip(("antenna", "time"), pilot_records)
        },
        "precision_metric": metric,
        "precision_target_half_width": target,
        "units": units,
        "conservative_pilot_half_width": conservative_half_width,
        "test_fraction": test_fraction,
        "uncapped_total_estimate": estimate,
        "source_count_cap": cap,
        "formal_total": formal_total,
        "expected_split_counts": {
            "calibration": math.floor(formal_total * common["splits"]["calibration"]),
            "selection": math.floor(formal_total * common["splits"]["selection"]),
            "test": formal_total
            - math.floor(formal_total * common["splits"]["calibration"])
            - math.floor(formal_total * common["splits"]["selection"]),
        },
        "projected_test_half_width_at_cap_or_estimate": projected_cap_half_width,
        "target_missed_at_source_cap": bool(estimate > cap),
        "count_locked_before_selection": True,
        "sample_size_assumption": "pilot bootstrap half-width scales as 1/sqrt(n); no test-driven extension",
    }
    output = args.output if args.output.is_absolute() else REPOSITORY_ROOT / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite locked sample plan: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "formal_total": formal_total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
