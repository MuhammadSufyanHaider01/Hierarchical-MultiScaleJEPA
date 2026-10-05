#!/usr/bin/env python3
"""Prepare the matched, full-information frozen heads for temporal robustness."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from representation_trajectories import METHODS  # noqa: E402
from temporal_head_robustness import HEAD_SEEDS, sha256_file, validate_head_packet  # noqa: E402

DEFAULT_PROVENANCE = ROOT / "runs/representation_trajectories/multi_block_n6/interference/seed_000/test/provenance.json"
DEFAULT_HEAD_ROOT = ROOT / "runs/representation_trajectories/multi_block_n6/interference/seed_000/temporal_head_robustness/heads"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--head-root", type=Path, default=DEFAULT_HEAD_ROOT)
    parser.add_argument("--head-seeds", type=int, nargs="+", default=list(HEAD_SEEDS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = _args()
    if len(set(args.head_seeds)) != len(args.head_seeds) or not args.head_seeds:
        raise ValueError("head seeds must be a non-empty unique list")
    provenance_path = args.trajectory_provenance.resolve()
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("task") != "interference" or provenance.get("split") != "test":
        raise ValueError("trajectory provenance must describe the Interference test analysis")
    canonical_split = _resolve(provenance["split_manifest"])
    expected_split_hash = provenance["split_manifest_sha256"]
    if sha256_file(canonical_split) != expected_split_hash:
        raise ValueError("canonical split manifest hash differs from trajectory provenance")
    head_root = args.head_root.resolve()
    phase5_script = ROOT / "scripts/phase5_exits.py"

    commands = []
    for seed in args.head_seeds:
        seed_root = head_root / f"head_seed_{seed}"
        split_copy = seed_root / canonical_split.name
        if split_copy.exists():
            if sha256_file(split_copy) != expected_split_hash:
                raise ValueError(f"existing head-seed split differs from canonical: {split_copy}")
        elif not args.dry_run:
            seed_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(canonical_split, split_copy)
            if sha256_file(split_copy) != expected_split_hash:
                raise RuntimeError(f"split copy verification failed: {split_copy}")

        for method in METHODS:
            checkpoint = Path(provenance["checkpoints"][method]["path"])
            checkpoint_hash = provenance["checkpoints"][method]["sha256"]
            if not checkpoint.is_file() or sha256_file(checkpoint) != checkpoint_hash:
                raise ValueError(f"{method} checkpoint missing or changed: {checkpoint}")
            config = Path(provenance["configs"][method]["path"])
            if not config.is_file():
                raise FileNotFoundError(f"{method} config is missing: {config}")
            packet = seed_root / method / "seed_000" / "downstream" / "exit_metrics.json"
            if packet.is_file():
                validate_head_packet(packet, method=method, head_seed=seed,
                                     checkpoint_path=checkpoint, checkpoint_sha256=checkpoint_hash,
                                     split_manifest=split_copy, split_sha256=expected_split_hash)
                print(f"already prepared: {method} head_seed={seed}")
                continue
            cmd = [
                sys.executable, str(phase5_script),
                "--checkpoint", str(checkpoint),
                "--method", method, "--pretrain-seed", "0", "--head-seed", str(seed),
                "--split-seed", "2025", "--config", str(config),
                "--run-root", str(seed_root), "--dataset-suite", "primary",
                "--task", "interference", "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
                "--learning-rate", str(args.learning_rate), "--weight-decay", str(args.weight_decay),
                "--head-early-stopping-patience", str(args.patience),
                "--head-early-stopping-min-delta", "0", "--selection-only", "--device", args.device,
            ]
            commands.append(cmd)
            if args.dry_run:
                print(json.dumps(cmd))
                continue
            print(f"training selection-only heads: {method}, head_seed={seed}", flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
            validate_head_packet(packet, method=method, head_seed=seed,
                                 checkpoint_path=checkpoint, checkpoint_sha256=checkpoint_hash,
                                 split_manifest=split_copy, split_sha256=expected_split_hash)
    print(f"head preparation complete: {len(commands)} jobs requested")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
