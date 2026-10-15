#!/usr/bin/env python3
"""Audit, prepare, or submit the ten-job Fusion V2 adapter-ablation pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shlex
import subprocess
from pathlib import Path

import residual_fusion_adapter_ablation_report as report

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/ee_optimization/multi_block_n6/residual_fusion_adapter_ablation"
JOB_ROOT = RUN_ROOT / "slurm_adapter_ablation"
METHODS, VARIANTS = report.METHODS, report.VARIANTS


def audit() -> dict:
    inputs, failures, split_hashes = [], [], set()
    for method in METHODS:
        for algorithm in ("current", "bottleneck_gelu"):
            try:
                path, packet = report.load_packet(method, algorithm)
                split_hash = report.sha(Path(packet["split_manifest"])); split_hashes.add(split_hash)
                inputs.append({"method": method, "algorithm": algorithm, "metrics_path": str(path),
                               "metrics_sha256": report.sha(path), "checkpoint_sha256": report.sha(Path(packet["checkpoint"])),
                               "split_manifest_sha256": split_hash})
            except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
                failures.append(str(error))
    if len(inputs) != 4:
        failures.append(f"expected four validated contextual/reference packets, found {len(inputs)}")
    if len(split_hashes) != 1:
        failures.append("reference packets do not share exactly one split identity")
    return {"study": "residual_fusion_adapter_ablation_single_seed", "methods": METHODS, "variants": VARIANTS,
            "head_seed": 0, "inputs": inputs, "failures": failures, "passed": not failures}


def _runtime_code(method: str, variant: str) -> str:
    return ('import json,os,socket,subprocess,torch;from pathlib import Path;'
            'p=Path(os.environ["ABLATION_RUNTIME"]);p.parent.mkdir(parents=True,exist_ok=True);'
            'rev=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip();'
            f'json.dump({{"job_id":os.getenv("SLURM_JOB_ID"),"partition":os.getenv("SLURM_JOB_PARTITION"),"hostname":socket.gethostname(),"cpus":os.getenv("SLURM_CPUS_PER_TASK"),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"method":"{method}","adapter_variant":"{variant}","runtime_profile":"residual_fusion_adapter_ablation","git_revision":rev}},p.open("w"),sort_keys=True)')


def job_script(method: str, variant: str, partition: str, run_root: Path = RUN_ROOT) -> str:
    if method not in METHODS or variant not in VARIANTS:
        raise ValueError("unsupported ablation method or adapter variant")
    name = f"eev2ab-{method}-{variant.replace('_', '-') }"
    log = run_root / "slurm_adapter_ablation" / f"{name}-%j"
    runtime = run_root / "runtime" / variant / method / "head_seed_000.json"
    checkpoint = report.CHECKPOINT_HASHES[method]
    checkpoint_path = {
        "p0": ROOT / "runs/masking_head_seeds/checkpoint_bundle/checkpoints/multi_block_n6/P0/last.ckpt",
        "p2": ROOT / "runs/masking_head_seeds/checkpoint_bundle/checkpoints/multi_block_n6/P2/last.ckpt",
    }[method]
    command = [".venv/bin/python", "scripts/phase5_residual_fusion_ablation_exits.py", "--method", method,
               "--adapter-variant", variant, "--pretrain-seed", "0", "--head-seed", "0", "--split-seed", "2025",
               "--checkpoint", str(checkpoint_path), "--run-root", str(run_root.parent), "--epochs", "200", "--batch-size", "32",
               "--learning-rate", "0.001", "--weight-decay", "0.05", "--head-early-stopping-patience", "20", "--device", "cuda"]
    return "\n".join(["#!/usr/bin/env bash", f"#SBATCH --job-name={name}", f"#SBATCH --partition={partition}",
        "#SBATCH --gres=gpu:l40:1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=32G", "#SBATCH --time=1-00:00:00",
        f"#SBATCH --output={log}.out", f"#SBATCH --error={log}.err", "set -euo pipefail", f"cd {shlex.quote(str(ROOT))}",
        "export PYTHONPATH=src:scripts", "export WANDB_KEY=offline", "export PYTORCH_ALLOC_CONF=expandable_segments:True",
        f"export ABLATION_RUNTIME={shlex.quote(str(runtime))}", f".venv/bin/python -c {shlex.quote(_runtime_code(method, variant))}",
        f"# Locked {method} checkpoint SHA256: {checkpoint}", shlex.join(command)]) + "\n"


def report_script(partition: str, run_root: Path = RUN_ROOT) -> str:
    log = run_root / "slurm_adapter_ablation" / "eev2ab-report-%j"
    return "\n".join(["#!/usr/bin/env bash", "#SBATCH --job-name=eev2ab-report", f"#SBATCH --partition={partition}",
        "#SBATCH --cpus-per-task=2", "#SBATCH --mem=8G", "#SBATCH --time=2:00:00", f"#SBATCH --output={log}.out", f"#SBATCH --error={log}.err",
        "set -euo pipefail", f"cd {shlex.quote(str(ROOT))}", "export PYTHONPATH=src:scripts", ".venv/bin/python scripts/residual_fusion_adapter_ablation_report.py"]) + "\n"


def prepare(registry: dict, gpu_partition: str, cpu_partition: str, run_root: Path = RUN_ROOT) -> dict:
    if not registry["passed"]:
        raise RuntimeError("input audit failed; refusing to prepare jobs")
    scripts = run_root / "slurm_adapter_ablation"; scripts.mkdir(parents=True, exist_ok=True)
    records = []
    for variant in VARIANTS:
        for method in METHODS:
            path = scripts / f"{method}_{variant}.sbatch"; path.write_text(job_script(method, variant, gpu_partition, run_root))
            records.append({"kind": "adapter_ablation_train", "method": method, "adapter_variant": variant, "head_seed": 0, "script": str(path)})
    report_path = scripts / "report.sbatch"; report_path.write_text(report_script(cpu_partition, run_root))
    result = {**registry, "prepared_jobs": records, "report_script": str(report_path), "gpu_partition": gpu_partition, "cpu_partition": cpu_partition}
    (run_root / "registry.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def submit(prepared: dict, run_root: Path = RUN_ROOT) -> dict:
    if len(prepared.get("prepared_jobs", [])) != 10:
        raise RuntimeError("refusing submission unless exactly ten training jobs are prepared")
    submitted, path = [], run_root / "submissions.json"
    for item in prepared["prepared_jobs"]:
        output = subprocess.run(["sbatch", "--parsable", item["script"]], check=True, capture_output=True, text=True).stdout.strip()
        submitted.append({**item, "job_id": output.split(";", 1)[0], "submitted_at_utc": datetime.now(timezone.utc).isoformat()})
        path.write_text(json.dumps({"training_jobs": submitted, "report_job": None}, indent=2, sort_keys=True) + "\n")
    dependency = ":".join(item["job_id"] for item in submitted)
    output = subprocess.run(["sbatch", "--parsable", f"--dependency=afterok:{dependency}", prepared["report_script"]], check=True, capture_output=True, text=True).stdout.strip()
    result = {"training_jobs": submitted, "report_job": {"script": prepared["report_script"], "job_id": output.split(";", 1)[0], "dependency": f"afterok:{dependency}", "submitted_at_utc": datetime.now(timezone.utc).isoformat()}}
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("audit", "prepare", "submit"), default="audit")
    parser.add_argument("--gpu-partition", default="gpu-l40"); parser.add_argument("--cpu-partition", default="cpu2023")
    args = parser.parse_args(); registry = audit()
    if args.action == "audit": print(json.dumps(registry, indent=2)); return 0 if registry["passed"] else 2
    prepared = prepare(registry, args.gpu_partition, args.cpu_partition)
    if args.action == "prepare": print(json.dumps({"registry": str(RUN_ROOT / "registry.json"), "jobs": len(prepared["prepared_jobs"])}, indent=2)); return 0
    print(json.dumps(submit(prepared), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
