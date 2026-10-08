#!/usr/bin/env python3
"""Audit, prepare, or submit the locked two-stage EE optimization study."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/ee_optimization/multi_block_n6"
METHODS = ("p0", "p2")
SEEDS = (0,)
TASKS = ("rfs", "jamming", "human-activity", "interference", "nrpos-outdoor", "deepmimo-beam")
EXITS = (2, 4, 6, 8)
CHECKPOINTS = {"p0": ROOT / "runs/masking_head_seeds/checkpoint_bundle/checkpoints/multi_block_n6/P0/last.ckpt",
               "p2": ROOT / "runs/masking_head_seeds/checkpoint_bundle/checkpoints/multi_block_n6/P2/last.ckpt"}
CHECKPOINT_HASHES = {"p0": "8faa7245b6692e2d74d753fde2268fb712264c02c1da19632c045d022d192f1f",
                     "p2": "7e1e27d96183b8f9d1c0d6b7493b93945633ccfb1514555135b4d04f22bf18ac"}
BASELINE = ROOT / "runs/masking_head_seeds/multi_block_n6"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit() -> dict:
    entries, failures = [], []
    split_hashes = set()
    for method in METHODS:
        checkpoint = CHECKPOINTS[method]
        if not checkpoint.is_file():
            failures.append(f"missing checkpoint: {checkpoint}")
            continue
        checkpoint_sha = sha(checkpoint)
        if checkpoint_sha != CHECKPOINT_HASHES[method]:
            failures.append(f"checkpoint SHA256 mismatch for {method}: {checkpoint}")
        for seed in SEEDS:
            metrics_path = BASELINE / f"head_seed_{seed:03d}" / method / "seed_000/downstream/exit_metrics.json"
            if not metrics_path.is_file():
                failures.append(f"missing baseline packet: {metrics_path}")
                continue
            packet = json.loads(metrics_path.read_text())
            checks = (packet.get("method") == method, int(packet.get("head_seed", -1)) == seed,
                      int(packet.get("pretrain_seed", -1)) == 0,
                      Path(packet.get("checkpoint", "")).resolve() == checkpoint.resolve(),
                      (packet.get("checkpoint_sha256") in (None, checkpoint_sha)),
                      packet.get("protocol", {}).get("test_evaluation") is True,
                      set(packet.get("tasks", {})) == set(TASKS))
            if not all(checks):
                failures.append(f"incompatible baseline metadata: {metrics_path}")
            split = Path(packet.get("split_manifest", ""))
            if not split.is_file() or int(json.loads(split.read_text()).get("seed", -1)) != 2025:
                failures.append(f"missing/wrong split manifest: {split}")
                split_sha = None
            else:
                split_sha = sha(split)
                split_hashes.add(split_sha)
            for task in TASKS:
                row = packet.get("tasks", {}).get(task, {})
                if set(row.get("exits", {})) != {str(x) for x in EXITS}:
                    failures.append(f"incomplete baseline exits: {metrics_path}:{task}")
                for layer in EXITS:
                    head = Path(row.get("exits", {}).get(str(layer), {}).get("head_path", ""))
                    if not head.is_file() or "test" not in row.get("exits", {}).get(str(layer), {}):
                        failures.append(f"missing baseline head/test: {metrics_path}:{task}:L{layer}")
            entries.append({"method": method, "head_seed": seed, "metrics_path": str(metrics_path),
                            "metrics_sha256": sha(metrics_path), "checkpoint_path": str(checkpoint),
                            "checkpoint_sha256": checkpoint_sha, "split_sha256": split_sha})
    if len(split_hashes) > 1:
        failures.append("baseline split manifest identities differ across seeds/methods")
    return {"study": "ee_optimization/multi_block_n6", "methods": METHODS,
            "head_seeds": SEEDS, "tasks": TASKS, "exits": EXITS,
            "baseline_entries": entries, "failures": failures, "passed": not failures}


def job_script(stage: str, method: str, seed: int, partition: str) -> str:
    root = RUN_ROOT
    name = f"ee-{stage}-{method}-hs{seed:03d}"
    log = root / "slurm_single_seed" / f"{name}-%j"
    runtime = root / "runtime" / stage / method / f"head_seed_{seed:03d}.json"
    ckpt = CHECKPOINTS[method]
    task_script = "phase5_fusion_exits.py" if stage == "fusion" else "phase5_fusion_distill.py"
    cmd = [".venv/bin/python", f"scripts/{task_script}", "--method", method,
           "--pretrain-seed", "0", "--head-seed", str(seed), "--split-seed", "2025",
           "--checkpoint", str(ckpt), "--run-root", str(root), "--epochs", "200",
           "--batch-size", "32", "--learning-rate", "0.001", "--weight-decay", "0.05",
           "--head-early-stopping-patience", "20", "--device", "cuda"]
    if stage == "fusion_kd":
        cmd.extend(["--fusion-teacher-root", str(root / "fusion"), "--kd-temperature", "2.0", "--kd-weight", "0.5"])
    runtime_code = ('import json,os,socket,subprocess,sys,torch;from pathlib import Path;'
        'p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);'
        'rev=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip();'
        'json.dump({"job_id":os.getenv("SLURM_JOB_ID"),"partition":os.getenv("SLURM_JOB_PARTITION"),'
        '"nodes":os.getenv("SLURM_JOB_NODELIST"),"hostname":socket.gethostname(),'
        '"cpus":os.getenv("SLURM_CPUS_PER_TASK"),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'
        '"runtime_profile":"frozen_encoder_ee","git_revision":rev},p.open("w"),sort_keys=True)')
    lines = ["#!/usr/bin/env bash", f"#SBATCH --job-name={name}", f"#SBATCH --partition={partition}",
             "#SBATCH --gres=gpu:1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=32G",
             "#SBATCH --time=1-00:00:00", f"#SBATCH --output={log}.out", f"#SBATCH --error={log}.err",
             "set -euo pipefail", f"cd {shlex.quote(str(ROOT))}", "export PYTHONPATH=src:scripts",
             "export WANDB_KEY=offline", "export PYTORCH_ALLOC_CONF=expandable_segments:True",
             f".venv/bin/python -c {shlex.quote(runtime_code)} {shlex.quote(str(runtime))}",
             shlex.join(cmd)]
    return "\n".join(lines) + "\n"


def prepare(registry: dict, partition: str) -> dict:
    if not registry["passed"]:
        raise RuntimeError("baseline audit failed; refusing to prepare partial study")
    # Keep this first single-seed batch isolated from the earlier seven-seed
    # prepared scripts, which remain available for later expansion.
    slurm = RUN_ROOT / "slurm_single_seed"
    slurm.mkdir(parents=True, exist_ok=True)
    records = []
    for method in METHODS:
        for seed in SEEDS:
            for stage in ("fusion", "fusion_kd"):
                path = slurm / f"{stage}_{method}_seed{seed:03d}.sbatch"
                path.write_text(job_script(stage, method, seed, partition))
                records.append({"stage": stage, "method": method, "head_seed": seed,
                                "script": str(path), "dependency_stage": "fusion" if stage == "fusion_kd" else None,
                                "dependency_method": method if stage == "fusion_kd" else None,
                                "dependency_head_seed": seed if stage == "fusion_kd" else None})
    result = {**registry, "prepared_jobs": records}
    (RUN_ROOT / "registry.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("audit", "prepare", "submit"), default="audit")
    parser.add_argument("--partition", default="gpu-a100,gpu-h100,gpu-l40")
    args = parser.parse_args()
    result = audit()
    if args.action == "audit":
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 2
    if args.action == "prepare":
        result = prepare(result, args.partition)
        print(json.dumps({"registry": str(RUN_ROOT / "registry.json"), "jobs": len(result["prepared_jobs"])}, indent=2))
        return 0
    if not result["passed"]:
        raise RuntimeError("baseline audit failed; refusing submission")
    prepared = prepare(result, args.partition)
    submitted = {}
    for item in prepared["prepared_jobs"]:
        dependency = item.get("dependency_method")
        dependency_id = submitted.get(("fusion", dependency, item["dependency_head_seed"])) if dependency else None
        command = ["sbatch", "--parsable"]
        if dependency_id:
            command.append(f"--dependency=afterok:{dependency_id}")
        response = subprocess.run(command + [item["script"]], check=True, capture_output=True, text=True)
        submitted[(item["stage"], item["method"], item["head_seed"])] = response.stdout.strip().split(";", 1)[0]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
