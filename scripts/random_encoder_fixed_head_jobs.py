#!/usr/bin/env python3
"""Audit, prepare, and submit evaluations with unchanged pretrained seed-0 heads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shlex
import subprocess
from pathlib import Path

import random_encoder_ablation as control
import random_encoder_fixed_head_report as report
import single_gelu_fusion_multiseed_report as prior

ROOT, RUN_ROOT = control.ROOT, report.RUN_ROOT


def encoder_paths(seed):
    root = control.RUN_ROOT if seed == 0 else ROOT / "runs/ee_optimization/multi_block_n6/random_encoder_ablation_multiseed" / f"seed_{seed:03d}"
    return root / f"random_encoder_seed_{seed:03d}.pt", root / f"random_encoder_seed_{seed:03d}.json"


def audit():
    errors, inputs, hashes = [], [], set()
    for seed in report.SEEDS:
        checkpoint, metadata = encoder_paths(seed)
        try:
            record = control.validate(checkpoint, metadata, seed)
            hashes.add(record["checkpoint_sha256"])
            inputs.append({"kind": "random_encoder", "seed": seed, "checkpoint": str(checkpoint),
                "checkpoint_sha256": record["checkpoint_sha256"], "metadata": str(metadata), "metadata_sha256": control.sha(metadata)})
        except (ValueError, FileNotFoundError, KeyError, TypeError) as error:
            errors.append(str(error))
    for method in report.METHODS:
        for algorithm in report.ALGORITHMS:
            try:
                path, packet = prior.load_packet(method, algorithm, 0)
                inputs.append({"kind": "fixed_head_source", "method": method, "algorithm": algorithm, "head_seed": 0,
                    "metrics": str(path), "metrics_sha256": control.sha(path), "encoder_checkpoint_sha256": packet["_checkpoint_sha256"]})
            except (ValueError, FileNotFoundError, KeyError, TypeError) as error:
                errors.append(str(error))
    if len(hashes) != 7:
        errors.append("seven distinct random encoder checkpoints are required")
    if len(inputs) != 11:
        errors.append("expected seven random encoder checkpoints and four pretrained seed-0 head sources")
    return {"study": "random_encoder_fixed_pretrained_heads", "passed": not errors, "inputs": inputs, "errors": errors,
        "encoder_seeds": report.SEEDS, "fixed_head_seed": 0, "methods": report.METHODS, "algorithms": report.ALGORITHMS}


def job_script(seed, gpu_partition="gpu-l40"):
    if seed not in report.SEEDS:
        raise ValueError("encoder seed must be 0–6")
    checkpoint, metadata = encoder_paths(seed)
    name = f"ee-rand-fixedhead-s{seed:03d}"
    logs = RUN_ROOT / "slurm" / f"{name}-%j"
    runtime = (control.RUN_ROOT / "runtime_fixed_head" / "allocation.json" if seed == 0 else
        ROOT / "runs/ee_optimization/multi_block_n6/random_encoder_ablation_multiseed" / f"seed_{seed:03d}/runtime_fixed_head/allocation.json")
    runtime_code = ('import json,os,socket,subprocess,sys,torch;from pathlib import Path;'
        'p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);'
        'json.dump({"job_id":os.getenv("SLURM_JOB_ID"),"partition":os.getenv("SLURM_JOB_PARTITION"),'
        '"nodes":os.getenv("SLURM_JOB_NODELIST"),"cpus":os.getenv("SLURM_CPUS_PER_TASK"),'
        '"hostname":socket.gethostname(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'
        '"git_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),'
        f'"encoder_seed":{seed},"head_seed":0,"mode":"fixed_head_evaluation"}},p.open("w"),sort_keys=True)')
    command = [".venv/bin/python", "scripts/random_encoder_fixed_head_eval.py", "--seed", str(seed), "--run-root", str(RUN_ROOT)]
    source_guard = (".venv/bin/python -c " + shlex.quote('import hashlib,json,sys;from pathlib import Path;r=json.loads(Path(sys.argv[1]).read_text());bad=[p for p,h in r["source_sha256"].items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];sys.exit("source files changed since preparation: "+str(bad) if bad else 0)')
        + " " + shlex.quote(str(RUN_ROOT / "registry.json")))
    return "\n".join(["#!/usr/bin/env bash", f"#SBATCH --job-name={name}", f"#SBATCH --partition={gpu_partition}",
        "#SBATCH --gres=gpu:l40:1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=32G", "#SBATCH --time=1-00:00:00",
        f"#SBATCH --output={logs}.out", f"#SBATCH --error={logs}.err", "set -euo pipefail", f"cd {shlex.quote(str(ROOT))}",
        "export PYTHONPATH=src:scripts", "export WANDB_KEY=offline", "export PYTORCH_ALLOC_CONF=expandable_segments:True",
        source_guard, f".venv/bin/python -c {shlex.quote(runtime_code)} {shlex.quote(str(runtime))}", shlex.join(command)]) + "\n"


def report_script(cpu_partition="cpu2023"):
    log = RUN_ROOT / "slurm" / "report-%j"
    source_guard = (".venv/bin/python -c " + shlex.quote('import hashlib,json,sys;from pathlib import Path;r=json.loads(Path(sys.argv[1]).read_text());bad=[p for p,h in r["source_sha256"].items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];sys.exit("source files changed since preparation: "+str(bad) if bad else 0)')
        + " " + shlex.quote(str(RUN_ROOT / "registry.json")))
    return "\n".join(["#!/usr/bin/env bash", "#SBATCH --job-name=ee-rand-fixedhead-report", f"#SBATCH --partition={cpu_partition}",
        "#SBATCH --cpus-per-task=2", "#SBATCH --mem=8G", "#SBATCH --time=2:00:00", f"#SBATCH --output={log}.out",
        f"#SBATCH --error={log}.err", "set -euo pipefail", f"cd {shlex.quote(str(ROOT))}",
        "export PYTHONPATH=src:scripts", source_guard, ".venv/bin/python scripts/random_encoder_fixed_head_report.py"]) + "\n"


def prepare(registry, gpu_partition="gpu-l40", cpu_partition="cpu2023"):
    if not registry["passed"]:
        raise ValueError("fixed-head input audit failed")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if (RUN_ROOT / "submissions.json").exists():
        raise FileExistsError("submission manifest exists; refusing duplicate jobs")
    scripts = RUN_ROOT / "slurm"; scripts.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in report.SEEDS:
        path = scripts / f"fixed_head_eval_seed_{seed:03d}.sbatch"; path.write_text(job_script(seed, gpu_partition))
        checkpoint, metadata = encoder_paths(seed)
        jobs.append({"encoder_seed": seed, "fixed_head_seed": 0, "checkpoint_sha256": control.sha(checkpoint),
            "script": str(path), "script_sha256": control.sha(path), "mode": "evaluation_only"})
    report_path = scripts / "report.sbatch"; report_path.write_text(report_script(cpu_partition))
    names = ("phase5_exits.py", "phase5_residual_fusion_ablation_exits.py", "ee_progress.py", "random_encoder_ablation.py",
        "random_encoder_fixed_head_eval.py", "random_encoder_fixed_head_jobs.py", "random_encoder_fixed_head_report.py",
        "single_gelu_fusion_multiseed_report.py", "residual_fusion_adapter_ablation_report.py")
    paths = list((ROOT / "src").rglob("*.py")) + [ROOT / "configs/p0.yml", ROOT / "configs/p2.yml"] + [ROOT / "scripts" / name for name in names]
    prepared = {**registry, "jobs": jobs, "report_script": str(report_path), "report_script_sha256": control.sha(report_path),
        "source_sha256": {str(path): control.sha(path) for path in paths},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "gpu_partition": gpu_partition, "cpu_partition": cpu_partition,
        "resources": {"gpu": "l40:1", "cpus": 8, "memory_gb": 32, "time_limit": "24:00:00", "report_cpus": 2, "report_memory_gb": 8}}
    (RUN_ROOT / "registry.json").write_text(json.dumps(prepared, indent=2, sort_keys=True) + "\n")
    return prepared


def submit(prepared):
    jobs = prepared.get("jobs", [])
    if len(jobs) != 7 or {job["encoder_seed"] for job in jobs} != set(report.SEEDS):
        raise ValueError("exactly seven encoder evaluation jobs are required")
    submission_path = RUN_ROOT / "submissions.json"
    if submission_path.exists():
        raise FileExistsError("submission manifest exists")
    submitted = []
    for job in jobs:
        output = subprocess.run(["sbatch", "--parsable", job["script"]], check=True, capture_output=True, text=True).stdout.strip()
        submitted.append({**job, "job_id": output.split(";", 1)[0], "submitted_at_utc": datetime.now(timezone.utc).isoformat()})
        submission_path.write_text(json.dumps({"evaluation_jobs": submitted, "report_job": None}, indent=2) + "\n")
    dependency = "afterok:" + ":".join(job["job_id"] for job in submitted)
    output = subprocess.run(["sbatch", "--parsable", f"--dependency={dependency}", prepared["report_script"]], check=True, capture_output=True, text=True).stdout.strip()
    result = {"evaluation_jobs": submitted, "report_job": {"job_id": output.split(";", 1)[0], "dependency": dependency,
        "script": prepared["report_script"], "submitted_at_utc": datetime.now(timezone.utc).isoformat()}}
    submission_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def retry_report():
    """Resubmit only the dependent CPU report after a report-code correction."""
    registry_path = RUN_ROOT / "registry.json"
    registry = json.loads(registry_path.read_text())
    expected = {str(ROOT / "scripts/random_encoder_fixed_head_report.py"),
                str(ROOT / "scripts/random_encoder_fixed_head_jobs.py")}
    for path, digest in registry["source_sha256"].items():
        actual = control.sha(Path(path))
        if actual != digest and path not in expected:
            raise ValueError(f"unexpected prepared source changed; refusing report retry: {path}")
        registry["source_sha256"][path] = actual
    report_path = RUN_ROOT / "slurm/report.sbatch"
    report_path.write_text(report_script(registry.get("cpu_partition", "cpu2023")))
    registry["report_script"] = str(report_path)
    registry["report_script_sha256"] = control.sha(report_path)
    registry["report_retry_prepared_at_utc"] = datetime.now(timezone.utc).isoformat()
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")

    submission_path = RUN_ROOT / "submissions.json"
    submissions = json.loads(submission_path.read_text())
    evaluation_jobs = submissions.get("evaluation_jobs", [])
    if len(evaluation_jobs) != len(report.SEEDS) or {int(job["encoder_seed"]) for job in evaluation_jobs} != set(report.SEEDS):
        raise ValueError("report retry requires the complete seven-seed evaluation submission")
    prerequisite_ids = [str(job["job_id"]) for job in evaluation_jobs]
    accounting = subprocess.run(["sacct", "-nP", "-j", ",".join(prerequisite_ids),
                                 "--format=JobID,State,ExitCode"], check=True, capture_output=True, text=True).stdout
    completed = {}
    for line in accounting.splitlines():
        fields = line.split("|")
        if len(fields) == 3 and fields[0] in prerequisite_ids:
            completed[fields[0]] = (fields[1], fields[2])
    if set(completed) != set(prerequisite_ids) or any(state != "COMPLETED" or code != "0:0"
                                                       for state, code in completed.values()):
        raise ValueError(f"report retry requires all seven evaluations completed successfully: {completed}")
    # Slurm may purge completed IDs from its dependency table. Since all seven
    # afterok prerequisites were verified successful above, submit the retry now.
    output = subprocess.run(["sbatch", "--parsable", str(report_path)],
                            check=True, capture_output=True, text=True).stdout.strip()
    prior_report = submissions.get("report_job")
    attempts = submissions.setdefault("report_attempts", [])
    if prior_report and prior_report not in attempts:
        attempts.append(prior_report)
    new_report = {"job_id": output.split(";", 1)[0], "dependency": None,
                  "script": str(report_path), "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                  "retry_of": prior_report.get("job_id") if prior_report else None,
                  "verified_afterok_prerequisites": prerequisite_ids,
                  "prerequisite_states": completed}
    attempts.append(new_report)
    submissions["report_job"] = new_report
    submission_path.write_text(json.dumps(submissions, indent=2, sort_keys=True) + "\n")
    return new_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("audit", "prepare", "submit", "retry-report"), default="audit")
    parser.add_argument("--gpu-partition", default="gpu-l40")
    parser.add_argument("--cpu-partition", default="cpu2023")
    args = parser.parse_args(); registry = audit()
    if args.action == "audit":
        print(json.dumps(registry, indent=2)); return 0 if registry["passed"] else 2
    if args.action == "retry-report":
        if not registry["passed"]:
            raise ValueError("fixed-head input audit failed; report retry refused")
        print(json.dumps(retry_report(), indent=2)); return 0
    prepared = prepare(registry, args.gpu_partition, args.cpu_partition)
    print(json.dumps(prepared if args.action == "prepare" else submit(prepared), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
