#!/usr/bin/env python3
"""Audit, prepare, or submit the seven-seed Residual Dynamic Fusion V2 study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import shlex
import subprocess
from pathlib import Path

import residual_fusion_v2_multiseed_report as report

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/ee_optimization/multi_block_n6/residual_dynamic_fusion"
MISSING_SEEDS = tuple(seed for seed in report.SEEDS if seed != 0)
PILOT_METHODS = ("p0", "p2")
CHECKPOINTS = {method: report.CHECKPOINT_PATHS[method] for method in PILOT_METHODS}


def audit(current_root=report.CURRENT_ROOT, v2_root=RUN_ROOT) -> dict:
    entries, failures = [], []
    split_hashes = set()
    for method in PILOT_METHODS:
        for algorithm, seeds in (("current", report.SEEDS), ("residual_projected_dynamic_fusion", (0,))):
            for seed in seeds:
                try:
                    path, packet = report.load_packet(method, algorithm, seed, current_root, v2_root)
                    split_hashes.add(packet["_split_sha256"])
                    entries.append({"method": method, "algorithm": algorithm, "head_seed": seed, "metrics_path": str(path),
                                    "metrics_sha256": report.sha(path), "checkpoint_sha256": packet["_checkpoint_sha256"], "split_sha256": packet["_split_sha256"]})
                except (FileNotFoundError, ValueError) as error:
                    failures.append(str(error))
    if len(split_hashes) > 1:
        failures.append("split manifest identity differs across reusable inputs")
    return {"study": "residual_dynamic_fusion_multiseed", "methods": PILOT_METHODS, "head_seeds": report.SEEDS,
            "reuse_v2_seed_zero": True, "missing_v2_seeds": MISSING_SEEDS, "inputs": entries, "failures": failures, "passed": not failures}


def _runtime_code(profile: str) -> str:
    return ('import json,os,socket,subprocess,sys,torch;from pathlib import Path;'
            'p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);'
            'rev=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip();'
            f'json.dump({{"job_id":os.getenv("SLURM_JOB_ID"),"partition":os.getenv("SLURM_JOB_PARTITION"),"nodes":os.getenv("SLURM_JOB_NODELIST"),"hostname":socket.gethostname(),"cpus":os.getenv("SLURM_CPUS_PER_TASK"),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"runtime_profile":"{profile}","git_revision":rev}},p.open("w"),sort_keys=True)')


def job_script(method: str, seed: int, partition: str, run_root=RUN_ROOT) -> str:
    if method not in PILOT_METHODS or seed not in MISSING_SEEDS:
        raise ValueError("jobs are only prepared for P0/P2 V2 seeds 1–6")
    name = f"eev2ms-{method}-hs{seed:03d}"
    logs = run_root / "slurm_multiseed" / f"{name}-%j"
    runtime = run_root / "runtime" / method / f"head_seed_{seed:03d}.json"
    cmd = [".venv/bin/python", "scripts/phase5_residual_fusion_exits.py", "--method", method, "--pretrain-seed", "0", "--head-seed", str(seed),
           "--split-seed", "2025", "--checkpoint", str(CHECKPOINTS[method]), "--run-root", str(run_root.parent), "--epochs", "200", "--batch-size", "32",
           "--learning-rate", "0.001", "--weight-decay", "0.05", "--head-early-stopping-patience", "20", "--device", "cuda"]
    return "\n".join(["#!/usr/bin/env bash", f"#SBATCH --job-name={name}", f"#SBATCH --partition={partition}", "#SBATCH --gres=gpu:l40:1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=32G", "#SBATCH --time=1-00:00:00", f"#SBATCH --output={logs}.out", f"#SBATCH --error={logs}.err", "set -euo pipefail", f"cd {shlex.quote(str(ROOT))}", "export PYTHONPATH=src:scripts", "export WANDB_KEY=offline", "export PYTORCH_ALLOC_CONF=expandable_segments:True", f".venv/bin/python -c {shlex.quote(_runtime_code('residual_dynamic_fusion_v2'))} {shlex.quote(str(runtime))}", shlex.join(cmd)]) + "\n"


def report_script(partition: str, run_root=RUN_ROOT) -> str:
    name, logs = "eev2ms-report", run_root / "slurm_multiseed" / "eev2ms-report-%j"
    return "\n".join(["#!/usr/bin/env bash", f"#SBATCH --job-name={name}", f"#SBATCH --partition={partition}", "#SBATCH --cpus-per-task=2", "#SBATCH --mem=8G", "#SBATCH --time=2:00:00", f"#SBATCH --output={logs}.out", f"#SBATCH --error={logs}.err", "set -euo pipefail", f"cd {shlex.quote(str(ROOT))}", "export PYTHONPATH=src:scripts", ".venv/bin/python scripts/residual_fusion_v2_multiseed_report.py"]) + "\n"


def prepare(registry: dict, gpu_partition: str, cpu_partition: str, run_root=RUN_ROOT) -> dict:
    if not registry["passed"]:
        raise RuntimeError("input audit failed; refusing to prepare jobs")
    scripts = run_root / "slurm_multiseed"; scripts.mkdir(parents=True, exist_ok=True)
    records = []
    for method in PILOT_METHODS:
        for seed in MISSING_SEEDS:
            path = scripts / f"v2_{method}_seed{seed:03d}.sbatch"; path.write_text(job_script(method, seed, gpu_partition, run_root))
            records.append({"kind": "v2_train", "method": method, "head_seed": seed, "script": str(path)})
    report_path = scripts / "report.sbatch"; report_path.write_text(report_script(cpu_partition, run_root))
    registry = {**registry, "prepared_jobs": records, "report_script": str(report_path), "gpu_partition": gpu_partition, "cpu_partition": cpu_partition}
    (run_root / "multiseed_registry.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    return registry


def submit(prepared: dict, run_root=RUN_ROOT) -> dict:
    submitted = []
    submission_path = run_root / "multiseed_submissions.json"
    for item in prepared["prepared_jobs"]:
        output = subprocess.run(["sbatch", "--parsable", item["script"]], check=True, capture_output=True, text=True).stdout.strip()
        job_id = output.split(";", 1)[0]; submitted.append({**item, "job_id": job_id, "submitted_at_utc": datetime.now(timezone.utc).isoformat()})
        submission_path.write_text(json.dumps({"training_jobs": submitted, "report_job": None}, indent=2, sort_keys=True) + "\n")
    dependency = ":".join(item["job_id"] for item in submitted)
    output = subprocess.run(["sbatch", "--parsable", f"--dependency=afterok:{dependency}", prepared["report_script"]], check=True, capture_output=True, text=True).stdout.strip()
    result = {"training_jobs": submitted, "report_job": {"script": prepared["report_script"], "job_id": output.split(";", 1)[0], "dependency": f"afterok:{dependency}", "submitted_at_utc": datetime.now(timezone.utc).isoformat()}}
    submission_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("audit", "prepare", "submit"), default="audit")
    parser.add_argument("--gpu-partition", default="gpu-l40"); parser.add_argument("--cpu-partition", default="cpu2023")
    args = parser.parse_args(); registry = audit()
    if args.action == "audit": print(json.dumps(registry, indent=2)); return 0 if registry["passed"] else 2
    prepared = prepare(registry, args.gpu_partition, args.cpu_partition)
    if args.action == "prepare": print(json.dumps({"registry": str(RUN_ROOT / "multiseed_registry.json"), "jobs": len(prepared["prepared_jobs"])}, indent=2)); return 0
    result = submit(prepared); print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
