#!/usr/bin/env python3
"""Generate immutable paired factory/OFDMA pilot or formal master data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synthetic_study.ofdma import balanced_jammer_labels, generate_realization, make_scene  # noqa: E402
from synthetic_study.protocol import make_split_manifest  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "formal"), required=True)
    parser.add_argument("--count", type=int, help="pilot count; formal count must come from a locked plan")
    parser.add_argument("--seed", type=int, default=341901)
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/synthetic_trained_heads/factory_ofdma_v2.yml")
    parser.add_argument("--count-plan", type=Path, help="required for formal mode")
    parser.add_argument("--output", type=Path, help="immutable output directory; defaults below profile outputs.data_root")
    parser.add_argument("--ray-count-override", type=int, help="smoke tests only; recorded and disallowed for formal mode")
    args = parser.parse_args()
    profile_path = args.profile.resolve()

    if args.count is not None and args.count < 12:
        parser.error("at least 12 realizations are required for the stratified six-class splits")
    if args.ray_count_override is not None and args.ray_count_override < 1:
        parser.error("--ray-count-override must be positive")
    if args.mode == "formal":
        if args.count_plan is None or not args.count_plan.is_file():
            parser.error("formal generation requires --count-plan from the reviewed pilot count planner")
        if args.ray_count_override is not None:
            parser.error("formal data must use the immutable profile ray count; smoke overrides are pilot-only")
        plan = json.loads(args.count_plan.read_text())
        if plan.get("status") != "locked" or int(plan.get("shared_total_count", plan.get("total_count", -1))) < 12:
            parser.error("count plan must have status=locked and a valid total_count")
        if args.count is not None and args.count != int(plan.get("shared_total_count", plan.get("total_count"))):
            parser.error("--count disagrees with the locked count plan")
        if plan.get("profile_sha256") != sha256(profile_path):
            parser.error("count plan profile hash does not match the supplied profile")
        if plan.get("formal_seed") is None or args.seed != int(plan["formal_seed"]):
            parser.error("formal seed must exactly match the seed locked in count_plan.json")
        count = int(plan.get("shared_total_count", plan.get("total_count")))
    else:
        count = args.count or 512
        plan = None

    profile = yaml.safe_load(profile_path.read_text())
    if not str(profile.get("study_id", "")).startswith("factory_ofdma_synthetic_trained_heads_v"):
        parser.error("unsupported or altered study profile")
    if args.mode == "pilot":
        outdir = args.output or (ROOT / profile["outputs"]["data_root"] / f"pilot_n{count}_seed{args.seed}")
    else:
        outdir = args.output or (ROOT / profile["outputs"]["data_root"] / f"formal_n{count}_seed{args.seed}")
    outdir = outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        parser.error(f"refusing to overwrite non-empty output directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    data_path = outdir / "master.h5"
    manifest_path = outdir / "split_manifest.json"
    metadata_path = outdir / "realization_metadata.jsonl"
    if any(path.exists() for path in (data_path, manifest_path, metadata_path)):
        parser.error("output files already exist; choose a new versioned output directory")

    rng = np.random.default_rng(args.seed)
    jammer_labels = balanced_jammer_labels(count, rng)
    ids = [f"{profile['study_id']}-{args.seed:08d}-{index:06d}" for index in range(count)]
    scene_path = ROOT / "src/synthetic_study/assets/factory_scene/scene0/scene.xml"
    scene = make_scene(scene_path, profile)
    metadata_records = []
    started = time.monotonic()
    compression = "lzf"
    with h5py.File(data_path, "x") as h5:
        shape = tuple(int(value) for value in profile["feature"]["master_shape"])
        features = h5.create_dataset(
            "features", shape=(count, *shape), dtype=np.float16,
            chunks=(1, *shape), compression=compression, shuffle=True,
        )
        h5.create_dataset("receiver_xyz_m", shape=(count, 3), dtype=np.float32)
        h5.create_dataset("jammer_mode", data=jammer_labels, dtype=np.uint8)
        h5.create_dataset("interference_label", data=(jammer_labels > 0).astype(np.uint8), dtype=np.uint8)
        h5.create_dataset("realization_id", data=np.asarray(ids, dtype=h5py.string_dtype("utf-8")))
        h5.attrs["schema_version"] = 1
        h5.attrs["study_id"] = profile["study_id"]
        h5.attrs["pilot"] = args.mode == "pilot"
        h5.attrs["seed"] = args.seed
        h5.attrs["profile_sha256"] = sha256(profile_path)
        h5.attrs["scene_sha256"] = sha256(scene_path)
        h5.attrs["git_revision"] = git_revision()
        h5.attrs["compression"] = compression
        h5.attrs["ray_count_per_source"] = int(args.ray_count_override or profile["scenario"]["propagation"]["ray_count_per_source"])
        h5.attrs["ray_count_override"] = int(args.ray_count_override or 0)
        h5.attrs["sionna_version"] = __import__("sionna").__version__
        h5.attrs["python_version"] = platform.python_version()
        for index, jammer_mode in enumerate(jammer_labels.tolist()):
            sample_seed = int(args.seed + index * 1009)
            x, coords, metadata = generate_realization(
                scene=scene,
                jammer_mode=int(jammer_mode),
                seed=sample_seed,
                config=profile,
                ray_count_per_source=args.ray_count_override,
            )
            if x.shape != shape or not np.isfinite(x).all():
                raise RuntimeError(f"invalid feature output for realization {ids[index]}")
            features[index] = x.astype(np.float16)
            h5["receiver_xyz_m"][index] = coords
            metadata_records.append({
                "realization_id": ids[index], "seed": sample_seed,
                "jammer_mode": int(jammer_mode), "receiver_xyz_m": coords.tolist(),
                **metadata,
            })
            if (index + 1) % 16 == 0 or index + 1 == count:
                h5.flush()
                print(f"generated {index + 1}/{count}; elapsed={time.monotonic()-started:.1f}s", flush=True)

    with metadata_path.open("x") as stream:
        for record in metadata_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    split = make_split_manifest(ids, jammer_labels, seed=args.seed + 17, pilot=args.mode == "pilot")
    split.update({
        "master_sha256": sha256(data_path),
        "profile_sha256": sha256(profile_path),
        "scene_sha256": sha256(scene_path),
        "created_from_git_revision": git_revision(),
        "generation_mode": args.mode,
        "count_plan": str(args.count_plan.resolve()) if args.count_plan else None,
        "count_plan_sha256": sha256(args.count_plan.resolve()) if args.count_plan else None,
        "metadata_sha256": sha256(metadata_path),
    })
    write_json(manifest_path, split)
    write_json(outdir / "generation_record.json", {
        "study_id": profile["study_id"], "mode": args.mode, "count": count,
        "seed": args.seed, "profile": str(profile_path), "profile_sha256": sha256(profile_path),
        "scene": str(scene_path), "scene_sha256": sha256(scene_path),
        "data": str(data_path), "data_sha256": sha256(data_path),
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "ray_count_per_source": int(args.ray_count_override or profile["scenario"]["propagation"]["ray_count_per_source"]),
        "ray_count_override": args.ray_count_override,
        "count_plan": plan,
        "python": platform.python_version(), "sionna": __import__("sionna").__version__,
    })
    print(json.dumps({"output": str(outdir), "count": count, "seconds": round(time.monotonic()-started, 2), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
