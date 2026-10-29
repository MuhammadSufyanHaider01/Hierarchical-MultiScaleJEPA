#!/usr/bin/env python3
"""Audit, prepare, and submit the seven-seed random-encoder ablation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
from pathlib import Path

import random_encoder_ablation as control
import random_encoder_ablation_jobs as single
import random_encoder_ablation_multiseed_report as report
import random_encoder_ablation_report as random_report
import single_gelu_fusion_multiseed_report as prior

ROOT, RUN_ROOT = control.ROOT, report.RUN_ROOT
SEEDS, ALGORITHMS = report.SEEDS, report.ALGORITHMS


def audit():
    inputs, errors, split_hashes = [], [], set()
    for method in ("p0", "p2"):
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                try:
                    path, packet = prior.load_packet(method, algorithm, seed)
                    split_hashes.add(packet["_split_sha256"])
                    inputs.append({"encoder": method, "algorithm": algorithm, "seed": seed,
                        "metrics": str(path), "metrics_sha256": control.sha(path), "checkpoint_sha256": packet["_checkpoint_sha256"]})
                except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
                    errors.append(str(exc))
    for algorithm in ALGORITHMS:
        try:
            path, packet = random_report.load_random(algorithm)
            split_hashes.add(packet["split_manifest_sha256"])
            inputs.append({"encoder": "random", "algorithm": algorithm, "seed": 0,
                "metrics": str(path), "metrics_sha256": control.sha(path), "checkpoint_sha256": packet["checkpoint_sha256"]})
        except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
            errors.append(str(exc))
    if len(inputs) != 30 or split_hashes != {control.sha(control.SPLIT)}:
        errors.append("expected 28 pretrained and two random seed-0 packets with the shared split")
    return {"study": "random_encoder_ablation_multiseed", "passed": not errors, "errors": errors, "inputs": inputs}


def job_script(algorithm, seed, run_root=RUN_ROOT):
    if algorithm not in ALGORITHMS or seed not in SEEDS[1:]:
        raise ValueError("expected random algorithm and seed 1–6")
    root = Path(run_root) / f"seed_{seed:03d}"
    script = single.job_script(algorithm, root)
    script = script.replace("--head-seed 0", f"--head-seed {seed}")
    script = script.replace("random_encoder_seed_000", f"random_encoder_seed_{seed:03d}")
    script = script.replace(f"#SBATCH --job-name=ee-random-{algorithm}", f"#SBATCH --job-name=ee-rand-{algorithm}-s{seed}")
    script = script.replace(single.source_check(root), single.source_check(run_root))
    return script


def report_script(run_root=RUN_ROOT):
    script = single.report_script(run_root)
    return script.replace("scripts/random_encoder_ablation_report.py", "scripts/random_encoder_ablation_multiseed_report.py").replace("ee-random-report", "ee-random-ms-report")


def prepare(registry, run_root=RUN_ROOT):
    root = Path(run_root)
    if not registry["passed"]:
        raise ValueError("reference audit failed")
    if (root / "submissions.json").exists():
        raise FileExistsError("submission manifest already exists; refusing duplicate preparation")
    scripts = root / "slurm"; scripts.mkdir(parents=True, exist_ok=True)
    records, checkpoints = [], []
    for seed in SEEDS[1:]:
        seed_dir = root / f"seed_{seed:03d}"
        for subdir in ("slurm", "runtime"):
            (seed_dir / subdir).mkdir(parents=True, exist_ok=True)
        checkpoint = seed_dir / f"random_encoder_seed_{seed:03d}.pt"
        metadata = seed_dir / f"random_encoder_seed_{seed:03d}.json"
        checkpoints.append(control.generate(checkpoint, metadata, seed))
        for algorithm in ALGORITHMS:
            path = scripts / f"{algorithm}_seed_{seed:03d}.sbatch"; path.write_text(job_script(algorithm, seed, root))
            records.append({"algorithm": algorithm, "seed": seed, "checkpoint_sha256": control.sha(checkpoint),
                "script": str(path), "script_sha256": control.sha(path)})
    report_path = scripts / "report.sbatch"; report_path.write_text(report_script(root))
    names = ("phase5_exits.py", "phase5_residual_fusion_ablation_exits.py", "ee_progress.py", "random_encoder_ablation.py",
        "random_encoder_ablation_jobs.py", "random_encoder_ablation_report.py", "random_encoder_ablation_multiseed_jobs.py",
        "random_encoder_ablation_multiseed_report.py", "single_gelu_fusion_multiseed_report.py", "residual_fusion_adapter_ablation_report.py")
    paths = list((ROOT / "src").rglob("*.py")) + [ROOT / "configs/p0.yml", ROOT / "configs/p2.yml"] + [ROOT / "scripts" / name for name in names]
    result = {**registry, "jobs": records, "random_encoders": checkpoints, "report_script": str(report_path),
        "report_script_sha256": control.sha(report_path), "source_sha256": {str(path): control.sha(path) for path in paths},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "resources": {"partition": "gpu-l40", "gpu": "l40:1", "cpus": 8, "memory_gb": 32, "hours": 24}}
    (root / "registry.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def submit(prepared, run_root=RUN_ROOT):
    root = Path(run_root); path = root / "submissions.json"
    if path.exists():
        raise FileExistsError("submission manifest already exists")
    expected = {(algorithm, seed) for algorithm in ALGORITHMS for seed in SEEDS[1:]}
    jobs = prepared["jobs"]
    if len(jobs) != 12 or {(j["algorithm"], j["seed"]) for j in jobs} != expected:
        raise ValueError("exactly 12 distinct random-control jobs are required")
    submitted = []
    for job in jobs:
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
