#!/usr/bin/env python3
"""Generate one write-once pilot for the paper-config factory-OFDMA study."""

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

from synthetic_source_packets import SOURCE_PACKETS, audit_source_packets, sha256  # noqa: E402
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
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--output-root", type=Path, default=Path("data/synthetic/factory_ofdma_paper_v3"))
    parser.add_argument("--ray-count", type=int, default=100_000)
    args = parser.parse_args()
    if not 256 <= args.count <= 512:
        raise ValueError("registered pilot size must be between 256 and 512")
    if args.ray_count < 1:
        raise ValueError("ray-count must be positive")

    registry_path = ROOT / "configs/sionna/factory_ofdma_paper_v3.yml"
    profile_path = ROOT / "configs/sionna" / TASK_PROFILE[args.task]
    profile, task_config = load_task_profile(profile_path)
    require_verified_semantics(profile)
    source_packets = audit_source_packets(args.task, profile)
    generator_code_hashes = {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in {"runner": Path(__file__), "plugin": ROOT / "src/synthetic/factory_ofdma_paper_v3_plugin.py", "scheduler": ROOT / "src/synthetic/factory_ofdma_paper_v3_scheduler.py", "signals": ROOT / "src/synthetic/factory_ofdma_paper_v3_signals.py", "ray_tracing": ROOT / "src/synthetic/factory_ofdma_paper_v3_rt.py"}.items()}
    registry = yaml.safe_load(registry_path.read_text())
    task_config["paper_radio_parameters"] = registry["paper_radio_parameters"]
    task_config["sionna"].update(ray_count_per_source=args.ray_count)
    sionna_version = version("sionna")
    if not sionna_version.startswith("2.1."):
        raise RuntimeError(f"paper-v3 adapter requires Sionna 2.1.x, found {sionna_version}")

    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    task_root = output_root / args.task
    master = task_root / "pilot.h5"
    manifest_path = task_root / "pilot_split_manifest.json"
    provenance_path = task_root / "pilot_provenance.json"
    if any(path.exists() for path in (master, manifest_path, provenance_path)):
        raise FileExistsError(f"refusing to overwrite existing paper-v3 pilot under {task_root}")

    ids, labels, master_digest = generate_master_file(
        task=args.task,
        profile=profile,
        task_config=task_config,
        generator=factory_ofdma_paper_v3_realization,
        destination=master,
        count=args.count,
        seed=args.seed,
        realization_prefix=f"factory-ofdma-paper-v3-{args.task}-pilot",
        sionna_version=sionna_version,
        split_kind="pilot",
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
    manifest["stage"] = "pilot_only"
    manifest["scenario_id"] = registry["scenario_id"]
    manifest["master_sha256"] = master_digest
    manifest_digest = write_manifest(manifest_path, manifest)
    record = {
        "scenario_id": registry["scenario_id"],
        "task": args.task,
        "count": args.count,
        "seed": args.seed,
        "sionna_version": sionna_version,
        "ray_count_per_source": args.ray_count,
        "stage": "pilot_only_no_exit_selection_no_test",
        "generator_code_hashes": generator_code_hashes,
        "master": {"path": str(master.resolve()), "sha256": master_digest,
                   "size_bytes": master.stat().st_size},
        "split_manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_digest},
        "task_profile": {"path": str(profile_path.resolve()), "sha256": sha256(profile_path)},
        "scenario_registry": {"path": str(registry_path.resolve()), "sha256": sha256(registry_path)},
        "source_packets": source_packets,
        "paper_source_config": registry["paper_source"]["radio_config"],
        "assumptions_and_deviations": registry["study_extensions"],
        "models_and_heads_unchanged": True,
    }
    provenance_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
