#!/usr/bin/env python3
"""Evaluate fixed seed-0 pretrained EE heads over random frozen encoders."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import random_encoder_ablation as control
import single_gelu_fusion_multiseed_report as prior

ROOT = control.ROOT
ALGORITHMS = ("current", "single_gelu")
METHODS = ("p0", "p2")


def source_packet(method: str, algorithm: str) -> Path:
    return prior.packet_path(method, algorithm, 0)


def stage_source(method: str, algorithm: str, seed: int, checkpoint: Path, metadata: Path, run_root: Path) -> Path:
    source = source_packet(method, algorithm)
    packet = json.loads(source.read_text())
    packet.update({"encoder_kind": "random", "pretrain_seed": None,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": control.sha(checkpoint),
        "random_encoder_metadata": str(metadata.resolve()), "random_encoder_seed": seed})
    if algorithm == "single_gelu":
        output = (run_root / "fusion" / "residual_fusion_adapter_ablation" / "head_seed_000" / "single_gelu"
                  / method / "seed_000/downstream/exit_metrics.json")
    else:
        output = run_root / "current" / method / "seed_000/downstream/exit_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    return output


def evaluate(method: str, algorithm: str, seed: int, checkpoint: Path, metadata: Path, split: Path, run_root: Path) -> Path:
    stage_source(method, algorithm, seed, checkpoint, metadata, run_root)
    if algorithm == "current":
        command = [sys.executable, "scripts/phase5_exits.py", "--method", method, "--pretrain-seed", "0", "--head-seed", "0",
            "--split-seed", "2025", "--split-manifest", str(split), "--run-root", str(run_root / "current"),
            "--checkpoint", str(checkpoint), "--encoder-kind", "random", "--random-encoder-metadata", str(metadata),
            "--random-encoder-seed", str(seed), "--test-only", "--device", "cuda"]
    else:
        command = [sys.executable, "scripts/phase5_residual_fusion_ablation_exits.py", "--method", method,
            "--adapter-variant", "single_gelu", "--pretrain-seed", "0", "--head-seed", "0",
            "--split-seed", "2025", "--run-root", str(run_root / "fusion"), "--checkpoint", str(checkpoint),
            "--encoder-kind", "random", "--random-encoder-metadata", str(metadata), "--random-encoder-seed", str(seed),
            "--test-only", "--device", "cuda"]
    subprocess.run(command, cwd=ROOT, check=True)
    if algorithm == "current":
        return run_root / "current" / method / "seed_000/downstream/exit_metrics.json"
    return run_root / "fusion" / "residual_fusion_adapter_ablation" / "head_seed_000" / "single_gelu" / method / "seed_000/downstream/exit_metrics.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=range(7))
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    seed_root = (control.RUN_ROOT if args.seed == 0 else
        control.ROOT / "runs/ee_optimization/multi_block_n6/random_encoder_ablation_multiseed" / f"seed_{args.seed:03d}")
    checkpoint = seed_root / f"random_encoder_seed_{args.seed:03d}.pt"
    metadata = seed_root / f"random_encoder_seed_{args.seed:03d}.json"
    record = control.validate(checkpoint, metadata, args.seed)
    seed_output = args.run_root / f"random_seed_{args.seed:03d}"
    if seed_output.exists() and any(seed_output.iterdir()):
        raise FileExistsError(f"evaluation output is not empty: {seed_output}")
    split = control.SPLIT
    for method in METHODS:
        for algorithm in ALGORITHMS:
            prior.load_packet(method, algorithm, 0)
            evaluate(method, algorithm, args.seed, checkpoint, metadata, split,
                     seed_output / method / algorithm)
    runtime = seed_root / "runtime_fixed_head" / "allocation.json"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    code = ('import json,os,socket,subprocess,sys,torch;from pathlib import Path;p=Path(sys.argv[1]);'
        'json.dump({"job_id":os.getenv("SLURM_JOB_ID"),"partition":os.getenv("SLURM_JOB_PARTITION"),'
        '"nodes":os.getenv("SLURM_JOB_NODELIST"),"cpus":os.getenv("SLURM_CPUS_PER_TASK"),'
        '"hostname":socket.gethostname(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'
        '"git_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),'
        '"encoder_seed":' + str(args.seed) + '},p.open("w"),sort_keys=True)')
    subprocess.run([sys.executable, "-c", code, str(runtime)], cwd=ROOT, check=True)
    print(json.dumps({"encoder_seed": args.seed, "checkpoint_sha256": record["checkpoint_sha256"], "evaluated": 4}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
