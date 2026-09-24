#!/usr/bin/env python3
"""Generate a locked paper-v3 formal master and immutable 20/20/60 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from synthetic_source_packets import audit_source_packets, sha256  # noqa: E402
from synthetic.factory_ofdma_paper_v3_plugin import factory_ofdma_paper_v3_realization  # noqa: E402
from synthetic.generator import generate_master_file  # noqa: E402
from synthetic.manifests import create_realization_manifest, write_manifest  # noqa: E402
from synthetic.schema import load_task_profile, require_verified_semantics  # noqa: E402


TASK_PROFILE = {
    "nrpos-outdoor": "factory_ofdma_nrpos_paper_v3.yml",
    "interference": "factory_ofdma_interference_paper_v3.yml",
    "jamming": "factory_ofdma_jamming_paper_v3.yml",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASK_PROFILE), required=True)
    parser.add_argument("--sample-plan", type=Path, required=True)
    parser.add_argument("--pilot-anchors-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/synthetic/factory_ofdma_paper_v3"))
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--ray-count", type=int, default=100_000)
    args = parser.parse_args()
    if args.ray_count < 1:
        raise ValueError("ray-count must be positive")

    plan_path = args.sample_plan if args.sample_plan.is_absolute() else ROOT / args.sample_plan
    plan = json.loads(plan_path.read_text())
    if plan.get("stage") != "sample_count_locked" or plan.get("task") != args.task:
        raise ValueError("formal generation requires the matching locked sample-count plan")
    if plan.get("count_locked_before_selection") is not True:
        raise ValueError("sample plan does not guarantee count locking before selection")
    count = int(plan["formal_total"])
    if count != int(plan.get("formal_total", -1)) or count < 512:
        raise ValueError("invalid locked formal count")
    if count > int(plan["source_count_cap"]):
        raise ValueError("locked formal count exceeds the declared source-count cap")

    anchor_root = args.pilot_anchors_root
    if not anchor_root.is_absolute():
        anchor_root = ROOT / anchor_root
    anchor_paths = {
        sweep: anchor_root / args.task / sweep / "pilot_anchors.json"
        for sweep in ("antenna", "time")
    }
    anchors = {sweep: json.loads(path.read_text()) for sweep, path in anchor_paths.items()}
    for sweep, record in anchors.items():
        if record.get("task") != args.task or record.get("sweep") != sweep:
            raise ValueError(f"{sweep} anchor does not match requested task")
        if record.get("stage") != "pilot_precision_estimate":
            raise ValueError(f"{sweep} anchor lacks pilot-only precision estimate")
        if int(record.get("pilot_master_count", -1)) != int(plan["pilot_realizations"]):
            raise ValueError("sample plan and precision anchors refer to different pilot counts")

    profile_path = ROOT / "configs/sionna" / TASK_PROFILE[args.task]
    registry_path = ROOT / "configs/sionna/factory_ofdma_paper_v3.yml"
    profile, task_config = load_task_profile(profile_path)
    require_verified_semantics(profile)
    registry = yaml.safe_load(registry_path.read_text())
    source_packets = audit_source_packets(args.task, profile)
    generator_code_hashes = {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in {"runner": Path(__file__), "plugin": ROOT / "src/synthetic/factory_ofdma_paper_v3_plugin.py", "scheduler": ROOT / "src/synthetic/factory_ofdma_paper_v3_scheduler.py", "signals": ROOT / "src/synthetic/factory_ofdma_paper_v3_signals.py", "ray_tracing": ROOT / "src/synthetic/factory_ofdma_paper_v3_rt.py"}.items()}
    task_config["paper_radio_parameters"] = registry["paper_radio_parameters"]
    task_config["sionna"].update(ray_count_per_source=args.ray_count)
    sionna_version = version("sionna")
    if not sionna_version.startswith("2.1."):
        raise RuntimeError(f"paper-v3 adapter requires Sionna 2.1.x, found {sionna_version}")

    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    task_root = data_root / args.task
    master_path = task_root / "master.h5"
    manifest_path = task_root / "split_manifest.json"
    provenance_path = task_root / "formal_provenance.json"
    if any(path.exists() for path in (master_path, manifest_path, provenance_path)):
        raise FileExistsError(f"refusing to overwrite formal paper-v3 outputs under {task_root}")
    ids, labels, master_digest = generate_master_file(
        task=args.task,
        profile=profile,
        task_config=task_config,
        generator=factory_ofdma_paper_v3_realization,
        destination=master_path,
        count=count,
        seed=args.seed,
        realization_prefix=f"factory-ofdma-paper-v3-{args.task}-formal",
        sionna_version=sionna_version,
        split_kind="formal",
    )
    manifest = create_realization_manifest(
        ids,
        labels,
        task=args.task,
        target_type=profile.target_type,
        seed=args.seed,
        coordinate_min=profile.coord_min,
        coordinate_max=profile.coord_max,
    )
    manifest.update({
        "stage": "formal_count_locked_before_selection",
        "scenario_id": registry["scenario_id"],
        "master_sha256": master_digest,
        "sample_plan": plan,
        "sample_plan_sha256": sha256(plan_path),
        "task_profile_sha256": sha256(profile_path),
        "scenario_registry_sha256": sha256(registry_path),
        "sionna_version": sionna_version,
        "generator_plugin": "synthetic.factory_ofdma_paper_v3_plugin:factory_ofdma_paper_v3_realization",
        "generator_code_hashes": generator_code_hashes,
        "split_protocol": "stratified calibration/selection/test = 20/20/60; realization IDs shared across both sweeps",
        "source_packets": source_packets,
    })
    manifest_digest = write_manifest(manifest_path, manifest)
    stat = master_path.stat()
    record = {
        "scenario_id": registry["scenario_id"],
        "task": args.task,
        "count": count,
        "seed": args.seed,
        "sionna_version": sionna_version,
        "ray_count_per_source": args.ray_count,
        "stage": "formal_master_only_no_selection_no_test",
        "generator_code_hashes": generator_code_hashes,
        "master": {"path": str(master_path.resolve()), "sha256": master_digest,
                   "size_bytes": stat.st_size, "modified_ns": stat.st_mtime_ns},
        "manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_digest},
        "sample_plan": {"path": str(plan_path.resolve()), "sha256": sha256(plan_path)},
        "pilot_anchors": {sweep: {"path": str(path.resolve()), "sha256": sha256(path)}
                           for sweep, path in anchor_paths.items()},
        "task_profile": {"path": str(profile_path.resolve()), "sha256": sha256(profile_path)},
        "scenario_registry": {"path": str(registry_path.resolve()), "sha256": sha256(registry_path)},
        "source_packets": source_packets,
        "anchor_gates_pass": {sweep: record["all_anchor_gates_pass"] for sweep, record in anchors.items()},
        "evaluation_scope": "zero-shot synthetic factory-domain stress test; failed anchors prohibit claims of matched-task accuracy",
        "models_and_heads_unchanged": True,
    }
    provenance_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
