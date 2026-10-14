#!/usr/bin/env python3
"""Audit, prepare, or submit P1/P2-Reverse Fusion V2 seven-seed jobs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shlex
import subprocess
from pathlib import Path

import residual_fusion_v2_multiseed_report as report

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("p1", "p2_reverse")
SEEDS = report.SEEDS
RUN_ROOT = report.V2_ROOT
JOB_ROOT = RUN_ROOT / "slurm_multiseed_p1_p2_reverse"
REGISTRY_PATH = RUN_ROOT / "p1_p2_reverse_multiseed_registry.json"
SUBMISSION_PATH = RUN_ROOT / "p1_p2_reverse_multiseed_submissions.json"


def audit(current_root=report.CURRENT_ROOT) -> dict:
    entries, failures = [], []
    split_hashes = set()
    for method in METHODS:
        checkpoint = report.CHECKPOINT_PATHS[method]
        if not checkpoint.is_file() or report.sha(checkpoint) != report.CHECKPOINT_HASHES[method]:
            failures.append(f"fixed {method} checkpoint missing or hash mismatch: {checkpoint}")
            continue
        for seed in SEEDS:
            try:
                path, packet = report.load_packet(method, "current", seed, current_root, report.V2_ROOT)
                split_hashes.add(packet["_split_sha256"])
                entries.append({"method": method, "algorithm": "current", "head_seed": seed,
                                "metrics_path": str(path), "metrics_sha256": report.sha(path),
                                "checkpoint_path": str(checkpoint), "checkpoint_sha256": packet["_checkpoint_sha256"],
                                "split_manifest_sha256": packet["_split_sha256"]})
            except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
                failures.append(str(error))
    if len(split_hashes) > 1:
        failures.append("split manifest identity differs across P1/P2-Reverse baseline packets")
    if len(entries) != len(METHODS) * len(SEEDS):
        failures.append(f"expected 14 validated Current EE packets, found {len(entries)}")
    return {"study": "residual_dynamic_fusion_multiseed_p1_p2_reverse", "methods": METHODS,
            "head_seeds": SEEDS, "reuse_current_ee_seeds": True, "reuse_v2_seed_zero": False,
            "inputs": entries, "failures": failures, "passed": not failures}


def _runtime_code(method: str) -> str:
    return ('import json,os,socket,subprocess,sys,torch;from pathlib import Path;'
            'p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);'
            'rev=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip();'
            f'json.dump({{"job_id":os.getenv("SLURM_JOB_ID"),"partition":os.getenv("SLURM_JOB_PARTITION"),"nodes":os.getenv("SLURM_JOB_NODELIST"),"hostname":socket.gethostname(),"cpus":os.getenv("SLURM_CPUS_PER_TASK"),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"method":"{method}","runtime_profile":"residual_dynamic_fusion_v2","git_revision":rev}},p.open("w"),sort_keys=True)')


def job_script(method: str, seed: int, partition: str, run_root=RUN_ROOT) -> str:
    if method not in METHODS or seed not in SEEDS:
        raise ValueError("additional-method jobs require P1/P2-Reverse and head seeds 0–6")
    checkpoint = report.CHECKPOINT_PATHS[method]
    name = f"eev2ms-{method}-hs{seed:03d}"
    logs = JOB_ROOT / f"{name}-%j"
    runtime = run_root / "runtime" / method / f"head_seed_{seed:03d}.json"
    cmd = [".venv/bin/python", "scripts/phase5_residual_fusion_exits.py", "--method", method,
           "--pretrain-seed", "0", "--head-seed", str(seed), "--split-seed", "2025",
           "--checkpoint", str(checkpoint), "--run-root", str(run_root.parent), "--epochs", "200",
           "--batch-size", "32", "--learning-rate", "0.001", "--weight-decay", "0.05",
           "--head-early-stopping-patience", "20", "--device", "cuda"]
    return "\n".join(["#!/usr/bin/env bash", f"#SBATCH --job-name={name}",
        f"#SBATCH --partition={partition}", "#SBATCH --gres=gpu:l40:1", "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=32G", "#SBATCH --time=1-00:00:00", f"#SBATCH --output={logs}.out",
        f"#SBATCH --error={logs}.err", "set -euo pipefail", f"cd {shlex.quote(str(ROOT))}",
        "export PYTHONPATH=src:scripts", "export WANDB_KEY=offline",
        "export PYTORCH_ALLOC_CONF=expandable_segments:True",
        f".venv/bin/python -c {shlex.quote(_runtime_code(method))} {shlex.quote(str(runtime))}",
        shlex.join(cmd)]) + "\n"


def report_script(partition: str) -> str:
    name = "eev2ms-p1-p2-reverse-report"
    logs = JOB_ROOT / f"{name}-%j"
    return "\n".join(["#!/usr/bin/env bash", f"#SBATCH --job-name={name}",
        f"#SBATCH --partition={partition}", "#SBATCH --cpus-per-task=2", "#SBATCH --mem=8G",
        "#SBATCH --time=2:00:00", f"#SBATCH --output={logs}.out", f"#SBATCH --error={logs}.err",
        "set -euo pipefail", f"cd {shlex.quote(str(ROOT))}", "export PYTHONPATH=src:scripts",
        ".venv/bin/python scripts/residual_fusion_v2_multiseed_report.py"]) + "\n"


def prepare(registry: dict, gpu_partition: str, cpu_partition: str, run_root=RUN_ROOT) -> dict:
    if not registry["passed"]:
        raise RuntimeError("input audit failed; refusing to prepare P1/P2-Reverse jobs")
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for method in METHODS:
        for seed in SEEDS:
            path = JOB_ROOT / f"v2_{method}_seed{seed:03d}.sbatch"
            script = job_script(method, seed, gpu_partition, run_root)
            path.write_text(script)
            records.append({"kind": "v2_train", "method": method, "head_seed": seed,
                            "script": str(path), "script_sha256": report.sha(path),
                            "checkpoint_sha256": report.CHECKPOINT_HASHES[method]})
    report_path = JOB_ROOT / "report.sbatch"
    report_path.write_text(report_script(cpu_partition))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    prepared = {**registry, "prepared_jobs": records, "report_script": str(report_path),
                "report_script_sha256": report.sha(report_path), "gpu_partition": gpu_partition,
                "cpu_partition": cpu_partition, "resources": {"gpu": "l40:1", "cpus": 8,
                "memory_gb": 32, "time_limit": "24:00:00", "report_cpus": 2,
                "report_memory_gb": 8}, "git_revision": revision,
                "prepared_at_utc": datetime.now(timezone.utc).isoformat()}
    REGISTRY_PATH.write_text(json.dumps(prepared, indent=2, sort_keys=True) + "\n")
    return prepared


def submit(prepared: dict) -> dict:
    if len(prepared.get("prepared_jobs", [])) != 14:
        raise RuntimeError("refusing submission unless exactly 14 training jobs are prepared")
    submitted = []
    for item in prepared["prepared_jobs"]:
        output = subprocess.run(["sbatch", "--parsable", item["script"]], check=True,
                                capture_output=True, text=True).stdout.strip()
        submitted.append({**item, "job_id": output.split(";", 1)[0],
                          "submitted_at_utc": datetime.now(timezone.utc).isoformat()})
        SUBMISSION_PATH.write_text(json.dumps({"training_jobs": submitted, "report_job": None},
                                              indent=2, sort_keys=True) + "\n")
    dependency = ":".join(item["job_id"] for item in submitted)
    output = subprocess.run(["sbatch", "--parsable", f"--dependency=afterok:{dependency}",
                             prepared["report_script"]], check=True, capture_output=True, text=True).stdout.strip()
    result = {"training_jobs": submitted, "report_job": {"script": prepared["report_script"],
              "job_id": output.split(";", 1)[0], "dependency": f"afterok:{dependency}",
              "submitted_at_utc": datetime.now(timezone.utc).isoformat()}}
    SUBMISSION_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("audit", "prepare", "submit"), default="audit")
    parser.add_argument("--gpu-partition", default="gpu-l40")
    parser.add_argument("--cpu-partition", default="cpu2023")
    args = parser.parse_args()
    registry = audit()
    if args.action == "audit":
        print(json.dumps(registry, indent=2))
        return 0 if registry["passed"] else 2
    prepared = prepare(registry, args.gpu_partition, args.cpu_partition)
    if args.action == "prepare":
        print(json.dumps({"registry": str(REGISTRY_PATH), "jobs": len(prepared["prepared_jobs"])}, indent=2))
        return 0
    result = submit(prepared)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
