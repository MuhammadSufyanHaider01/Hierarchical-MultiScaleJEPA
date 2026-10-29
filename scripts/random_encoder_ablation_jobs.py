#!/usr/bin/env python3
"""Audit, prepare, and submit the shared random encoder ablation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shlex
import subprocess
from pathlib import Path

import random_encoder_ablation as random_control
import single_gelu_fusion_multiseed_report as prior

ROOT, RUN_ROOT = random_control.ROOT, random_control.RUN_ROOT
ALGORITHMS = ("current", "single_gelu")


def source_check(run_root):
    code = ('import hashlib,json,sys;from pathlib import Path;'
            'r=json.loads(Path(sys.argv[1]).read_text());'
            'bad=[p for p,h in r["source_sha256"].items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];'
            'sys.exit("source files changed since preparation: "+str(bad) if bad else 0)')
    return f".venv/bin/python -c {shlex.quote(code)} {shlex.quote(str(Path(run_root) / 'registry.json'))}"


def audit():
    inputs, errors, splits = [], [], set()
    for method in ("p0", "p2"):
        for algorithm in ALGORITHMS:
            try:
                path, packet = prior.load_packet(method, algorithm, 0)
                splits.add(packet["_split_sha256"])
                inputs.append({"method": method, "algorithm": algorithm, "metrics": str(path),
                               "metrics_sha256": prior.sha(path), "checkpoint_sha256": packet["_checkpoint_sha256"],
                               "split_sha256": packet["_split_sha256"]})
            except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
                errors.append(str(exc))
    if len(inputs) != 4 or splits != {random_control.sha(random_control.SPLIT)}:
        errors.append("four matching reference packets and one shared split are required")
    return {"study": "random_encoder_ablation_single_seed", "passed": not errors, "errors": errors, "inputs": inputs}


def job_script(algorithm, run_root=RUN_ROOT):
    if algorithm not in ALGORITHMS:
        raise ValueError("unknown algorithm")
    root = Path(run_root); log = root / "slurm" / f"random-{algorithm}-%j"
    checkpoint, metadata = root / "random_encoder_seed_000.pt", root / "random_encoder_seed_000.json"
    if algorithm == "current":
        command = [".venv/bin/python", "scripts/phase5_exits.py", "--method", "p0", "--pretrain-seed", "0",
                   "--head-seed", "0", "--split-seed", "2025", "--split-manifest", str(random_control.SPLIT),
                   "--run-root", str(root / "current")]
    else:
        command = [".venv/bin/python", "scripts/phase5_residual_fusion_ablation_exits.py", "--method", "p0",
                   "--adapter-variant", "single_gelu", "--pretrain-seed", "0", "--head-seed", "0",
                   "--split-seed", "2025", "--run-root", str(root / "fusion")]
    command += ["--checkpoint", str(checkpoint), "--encoder-kind", "random", "--random-encoder-metadata", str(metadata),
                "--epochs", "200", "--batch-size", "32", "--learning-rate", "0.001", "--weight-decay", "0.05",
                "--head-early-stopping-patience", "20", "--device", "cuda"]
    runtime = root / "runtime" / f"{algorithm}.json"
    runtime_code = ('import json,os,socket,subprocess,sys,torch;from pathlib import Path;'
        'p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);'
        'json.dump({"job_id":os.getenv("SLURM_JOB_ID"),"partition":os.getenv("SLURM_JOB_PARTITION"),'
        '"nodes":os.getenv("SLURM_JOB_NODELIST"),"cpus":os.getenv("SLURM_CPUS_PER_TASK"),'
        '"hostname":socket.gethostname(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'
        '"git_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()},p.open("w"))')
    return "\n".join(["#!/bin/bash", f"#SBATCH --job-name=ee-random-{algorithm}", "#SBATCH --partition=gpu-l40",
        "#SBATCH --gres=gpu:l40:1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=32G", "#SBATCH --time=1-00:00:00",
        f"#SBATCH --output={log}.out", f"#SBATCH --error={log}.err", "set -euo pipefail", f"cd {shlex.quote(str(ROOT))}",
        "export PYTHONPATH=src:scripts", "export WANDB_KEY=offline", "export PYTORCH_ALLOC_CONF=expandable_segments:True",
        source_check(root),
        f".venv/bin/python -c {shlex.quote(runtime_code)} {shlex.quote(str(runtime))}", shlex.join(command)]) + "\n"


def report_script(run_root=RUN_ROOT):
    log = Path(run_root) / "slurm" / "random-report-%j"
    return "\n".join(["#!/bin/bash", "#SBATCH --job-name=ee-random-report", "#SBATCH --partition=cpu2023",
        "#SBATCH --cpus-per-task=2", "#SBATCH --mem=8G", "#SBATCH --time=2:00:00", f"#SBATCH --output={log}.out",
        f"#SBATCH --error={log}.err", "set -euo pipefail", f"cd {shlex.quote(str(ROOT))}",
        "export PYTHONPATH=src:scripts", source_check(run_root), ".venv/bin/python scripts/random_encoder_ablation_report.py"]) + "\n"


def prepare(registry, run_root=RUN_ROOT):
    if not registry["passed"]:
        raise ValueError("reference audit failed")
    root = Path(run_root); scripts = root / "slurm"; scripts.mkdir(parents=True, exist_ok=True)
    random_record = random_control.generate(root / "random_encoder_seed_000.pt", root / "random_encoder_seed_000.json")
    records = []
    for algorithm in ALGORITHMS:
        path = scripts / f"{algorithm}.sbatch"; path.write_text(job_script(algorithm, root))
        records.append({"algorithm": algorithm, "script": str(path), "script_sha256": random_control.sha(path)})
    path = scripts / "report.sbatch"; path.write_text(report_script(root))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_paths = list((ROOT / "src").rglob("*.py")) + [ROOT / "configs/p0.yml", ROOT / "configs/p2.yml"]
    source_paths += [ROOT / "scripts" / name for name in ("phase5_exits.py", "phase5_residual_fusion_ablation_exits.py",
        "ee_progress.py", "random_encoder_ablation.py", "random_encoder_ablation_jobs.py", "random_encoder_ablation_report.py",
        "single_gelu_fusion_multiseed_report.py", "residual_fusion_adapter_ablation_report.py")]
    source_hashes = {str(path): random_control.sha(path) for path in source_paths}
    prepared = {**registry, "jobs": records, "report_script": str(path), "report_script_sha256": random_control.sha(path),
                "random_encoder": random_record, "git_revision": revision, "source_sha256": source_hashes,
                "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                "resources": {"partition": "gpu-l40", "gpu": "l40:1", "cpus": 8, "memory_gb": 32, "hours": 24}}
    (root / "registry.json").write_text(json.dumps(prepared, indent=2, sort_keys=True) + "\n")
    return prepared


def submit(prepared, run_root=RUN_ROOT):
    root = Path(run_root); path = root / "submissions.json"
    if path.exists():
        raise FileExistsError("submission manifest already exists; inspect it before submitting again")
    if {item["algorithm"] for item in prepared["jobs"]} != set(ALGORITHMS) or len(prepared["jobs"]) != 2:
        raise ValueError("exactly two random encoder training jobs are required")
    submitted = []
    for job in prepared["jobs"]:
        job_id = subprocess.check_output(["sbatch", "--parsable", job["script"]], text=True).strip().split(";", 1)[0]
        submitted.append({**job, "job_id": job_id, "submitted_at_utc": datetime.now(timezone.utc).isoformat()})
        path.write_text(json.dumps({"training_jobs": submitted, "report_job": None}, indent=2) + "\n")
    dependency = "afterok:" + ":".join(item["job_id"] for item in submitted)
    job_id = subprocess.check_output(["sbatch", "--parsable", f"--dependency={dependency}", prepared["report_script"]], text=True).strip().split(";", 1)[0]
    result = {"training_jobs": submitted, "report_job": {"job_id": job_id, "dependency": dependency,
              "submitted_at_utc": datetime.now(timezone.utc).isoformat(), "script": prepared["report_script"]}}
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("audit", "prepare", "submit"), default="audit")
    args = parser.parse_args(); registry = audit()
    if args.action == "audit":
        print(json.dumps(registry, indent=2)); return 0 if registry["passed"] else 2
    prepared = prepare(registry)
    print(json.dumps(prepared if args.action == "prepare" else submit(prepared), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
