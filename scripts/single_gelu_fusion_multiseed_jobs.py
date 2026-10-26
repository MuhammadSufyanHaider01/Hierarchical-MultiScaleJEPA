#!/usr/bin/env python3
"""Audit, prepare, or submit the P0/P2 seven-seed Single-GELU Fusion study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shlex
import subprocess
from pathlib import Path

import single_gelu_fusion_multiseed_report as report

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = report.SINGLE_GELU_ROOT
JOB_ROOT = RUN_ROOT / "slurm_single_gelu_multiseed"
REGISTRY_PATH = RUN_ROOT / "single_gelu_multiseed_registry.json"
SUBMISSION_PATH = RUN_ROOT / "single_gelu_multiseed_submissions.json"
MISSING_SEEDS = tuple(seed for seed in report.SEEDS if seed != 0)


def audit(current_root=report.CURRENT_ROOT, single_gelu_root=RUN_ROOT) -> dict:
    inputs, failures, split_hashes = [], [], set()
    for method in report.METHODS:
        for algorithm, seeds in (("current", report.SEEDS), ("single_gelu", (0,))):
            for seed in seeds:
                try:
                    path, packet = report.load_packet(method, algorithm, seed, current_root, single_gelu_root)
                    split_hashes.add(packet["_split_sha256"])
                    inputs.append({"method": method, "algorithm": algorithm, "head_seed": seed,
                                   "metrics_path": str(path), "metrics_sha256": report.sha(path),
                                   "checkpoint_path": str(packet["checkpoint"]),
                                   "checkpoint_sha256": packet["_checkpoint_sha256"],
                                   "split_manifest_sha256": packet["_split_sha256"]})
                except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
                    failures.append(str(error))
    expected_current = len(report.METHODS) * len(report.SEEDS)
    if sum(row["algorithm"] == "current" for row in inputs) != expected_current:
        failures.append(f"expected {expected_current} validated Current EE packets")
    if sum(row["algorithm"] == "single_gelu" for row in inputs) != len(report.METHODS):
        failures.append("expected two reusable Single-GELU seed-0 packets")
    if len(split_hashes) != 1:
        failures.append("inputs do not share exactly one split manifest identity")
    return {"study": "single_gelu_fusion_multiseed", "methods": report.METHODS,
            "head_seeds": report.SEEDS, "reused_single_gelu_seed_zero": True,
            "missing_single_gelu_seeds": MISSING_SEEDS, "inputs": inputs,
            "failures": failures, "passed": not failures}


def _runtime_code(method: str) -> str:
    return ('import json,os,socket,subprocess,sys,torch;from pathlib import Path;'
            'p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);'
            'rev=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip();'
            f'json.dump({{"job_id":os.getenv("SLURM_JOB_ID"),"partition":os.getenv("SLURM_JOB_PARTITION"),'
            f'"nodes":os.getenv("SLURM_JOB_NODELIST"),"hostname":socket.gethostname(),'
            f'"cpus":os.getenv("SLURM_CPUS_PER_TASK"),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'
            f'"method":"{method}","adapter_variant":"single_gelu","runtime_profile":"single_gelu_fusion_multiseed",'
            '"git_revision":rev},p.open("w"),sort_keys=True)')


def job_script(method: str, seed: int, partition: str, run_root=RUN_ROOT) -> str:
    if method not in report.METHODS or seed not in MISSING_SEEDS:
        raise ValueError("jobs are only prepared for P0/P2 Single-GELU seeds 1–6")
    run_root = Path(run_root)
    name = f"eesgelums-{method}-hs{seed:03d}"
    logs = run_root / "slurm_single_gelu_multiseed" / f"{name}-%j"
    runtime = run_root / "runtime_single_gelu_multiseed" / method / f"head_seed_{seed:03d}.json"
    command = [".venv/bin/python", "scripts/phase5_residual_fusion_ablation_exits.py", "--method", method,
               "--adapter-variant", "single_gelu", "--pretrain-seed", "0", "--head-seed", str(seed),
               "--split-seed", "2025", "--checkpoint", str(report.CHECKPOINT_PATHS[method]),
               "--run-root", str(run_root.parent), "--epochs", "200", "--batch-size", "32",
               "--learning-rate", "0.001", "--weight-decay", "0.05", "--head-early-stopping-patience", "20",
               "--device", "cuda"]
    return "\n".join(["#!/usr/bin/env bash", f"#SBATCH --job-name={name}", f"#SBATCH --partition={partition}",
        "#SBATCH --gres=gpu:l40:1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=32G", "#SBATCH --time=1-00:00:00",
        f"#SBATCH --output={logs}.out", f"#SBATCH --error={logs}.err", "set -euo pipefail",
        f"cd {shlex.quote(str(ROOT))}", "export PYTHONPATH=src:scripts", "export WANDB_KEY=offline",
        "export PYTORCH_ALLOC_CONF=expandable_segments:True",
        f".venv/bin/python -c {shlex.quote(_runtime_code(method))} {shlex.quote(str(runtime))}",
        shlex.join(command)]) + "\n"


def report_script(partition: str, run_root=RUN_ROOT) -> str:
    logs = Path(run_root) / "slurm_single_gelu_multiseed" / "eesgelums-report-%j"
    return "\n".join(["#!/usr/bin/env bash", "#SBATCH --job-name=eesgelums-report", f"#SBATCH --partition={partition}",
        "#SBATCH --cpus-per-task=2", "#SBATCH --mem=8G", "#SBATCH --time=2:00:00", f"#SBATCH --output={logs}.out",
        f"#SBATCH --error={logs}.err", "set -euo pipefail", f"cd {shlex.quote(str(ROOT))}",
        "export PYTHONPATH=src:scripts", ".venv/bin/python scripts/single_gelu_fusion_multiseed_report.py"]) + "\n"


def prepare(registry: dict, gpu_partition: str, cpu_partition: str, run_root=RUN_ROOT) -> dict:
    if not registry["passed"]:
        raise RuntimeError("input audit failed; refusing to prepare jobs")
    run_root, scripts = Path(run_root), Path(run_root) / "slurm_single_gelu_multiseed"
    scripts.mkdir(parents=True, exist_ok=True)
    records = []
    for method in report.METHODS:
        for seed in MISSING_SEEDS:
            path = scripts / f"single_gelu_{method}_seed{seed:03d}.sbatch"
            path.write_text(job_script(method, seed, gpu_partition, run_root))
            records.append({"kind": "single_gelu_fusion_train", "method": method, "head_seed": seed,
                            "adapter_variant": "single_gelu", "script": str(path), "script_sha256": report.sha(path),
                            "checkpoint_sha256": report.CHECKPOINT_HASHES[method]})
    report_path = scripts / "report.sbatch"
    report_path.write_text(report_script(cpu_partition, run_root))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    prepared = {**registry, "prepared_jobs": records, "report_script": str(report_path),
                "report_script_sha256": report.sha(report_path), "gpu_partition": gpu_partition,
                "cpu_partition": cpu_partition, "resources": {"gpu": "l40:1", "cpus": 8, "memory_gb": 32,
                "time_limit": "24:00:00", "report_cpus": 2, "report_memory_gb": 8}, "git_revision": revision,
                "prepared_at_utc": datetime.now(timezone.utc).isoformat()}
    (run_root / "single_gelu_multiseed_registry.json").write_text(json.dumps(prepared, indent=2, sort_keys=True) + "\n")
    return prepared


def submit(prepared: dict, run_root=RUN_ROOT) -> dict:
    if len(prepared.get("prepared_jobs", [])) != len(report.METHODS) * len(MISSING_SEEDS):
        raise RuntimeError("refusing submission unless exactly 12 training jobs are prepared")
    run_root, submitted = Path(run_root), []
    submission_path = run_root / "single_gelu_multiseed_submissions.json"
    for item in prepared["prepared_jobs"]:
        output = subprocess.run(["sbatch", "--parsable", item["script"]], check=True, capture_output=True, text=True).stdout.strip()
        submitted.append({**item, "job_id": output.split(";", 1)[0], "submitted_at_utc": datetime.now(timezone.utc).isoformat()})
        submission_path.write_text(json.dumps({"training_jobs": submitted, "report_job": None}, indent=2, sort_keys=True) + "\n")
    dependency = ":".join(item["job_id"] for item in submitted)
    output = subprocess.run(["sbatch", "--parsable", f"--dependency=afterok:{dependency}", prepared["report_script"]],
                            check=True, capture_output=True, text=True).stdout.strip()
    result = {"training_jobs": submitted, "report_job": {"script": prepared["report_script"],
              "job_id": output.split(";", 1)[0], "dependency": f"afterok:{dependency}",
              "submitted_at_utc": datetime.now(timezone.utc).isoformat()}}
    submission_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
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
    print(json.dumps(submit(prepared), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
