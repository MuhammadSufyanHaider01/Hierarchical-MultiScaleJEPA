#!/usr/bin/env python3
"""Audit, prepare, or submit seven-seed frozen-head masking-study jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from masking_head_study import EXITS, MASKS, METHODS, SEEDS, SPLIT_SEED, TASKS, packet_metrics  # noqa: E402

INPUT_KEYS = {
    "p0": "b1_metrics",
    "p1": "p1_metrics",
    "p2": "p2_metrics",
    "p2_reverse": "p2_reverse_metrics",
}
BENCHMARK_KEYS = {
    "p0": "b1_benchmark",
    "p1": "p1_benchmark",
    "p2": "p2_benchmark",
    "p2_reverse": "p2_reverse_benchmark",
}
DEFAULT_RUN_ROOT = REPOSITORY_ROOT / "runs/masking_head_seeds"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_recorded_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    raw = Path(value).expanduser()
    if raw.is_file():
        return raw.resolve()
    text = str(value)
    for marker in ("runs/", "imports/"):
        if marker in text:
            candidate = REPOSITORY_ROOT / marker.rstrip("/") / text.split(marker, 1)[1]
            if candidate.is_file():
                return candidate.resolve()
    return raw


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def report_input(manifest: dict[str, Any], method: str) -> Path | None:
    raw = manifest.get("included_inputs", {}).get(INPUT_KEYS[method])
    return resolve_recorded_path(raw)


def validated_benchmark(manifest: dict[str, Any], method: str, checkpoint: Path | None) -> tuple[Path | None, str]:
    raw = manifest.get("included_inputs", {}).get(BENCHMARK_KEYS[method])
    path = resolve_recorded_path(raw)
    packet = load_json(path)
    if not path or packet is None:
        return None, "no accessible raw benchmark packet"
    try:
        if packet.get("method") != method or int(packet.get("pretrain_seed", -1)) != 0:
            raise ValueError("method or pretrained seed mismatch")
        if int(packet.get("batch_size", -1)) != 1:
            raise ValueError("benchmark is not batch-1")
        expected = {(task, layer) for task in TASKS for layer in EXITS}
        actual = {(row.get("task"), int(row.get("exit_layer", -1))) for row in packet.get("entries", [])}
        if actual != expected:
            raise ValueError("benchmark task/exit coverage mismatch")
        recorded_checkpoint = resolve_recorded_path(packet.get("checkpoint"))
        if not checkpoint or not recorded_checkpoint or sha256(recorded_checkpoint) != sha256(checkpoint):
            raise ValueError("benchmark checkpoint identity mismatch")
        return path, "validated imported batch-1 benchmark and checkpoint identity"
    except (ValueError, OSError, TypeError) as error:
        return None, str(error)

def expected_checkpoint(manifest: dict[str, Any], method: str, packet: dict[str, Any] | None) -> Path | None:
    if packet and packet.get("checkpoint"):
        resolved = resolve_recorded_path(packet["checkpoint"])
        if resolved and resolved.is_file():
            return resolved
    if method == "p0":
        root = manifest.get("run_root")
        relative = "p0/seed_000/checkpoints/last.ckpt"
    elif method == "p1":
        root = manifest.get("p1_checkpoint")
        relative = None
    elif method == "p2":
        root = manifest.get("p2_run_root")
        relative = "p2/seed_000/checkpoints/last.ckpt"
    else:
        return None
    if root:
        candidate = resolve_recorded_path(root)
        if relative:
            raw = Path(root) / relative
            candidate = resolve_recorded_path(raw)
        if candidate and candidate.is_file():
            return candidate
    return None


def _existing_head_files(packet_path: Path, packet: dict[str, Any]) -> tuple[bool, list[str]]:
    missing = []
    for task_name, task in packet.get("tasks", {}).items():
        for exit_layer in EXITS:
            row = task.get("exits", {}).get(str(exit_layer), {})
            path = resolve_recorded_path(row.get("head_path"))
            if path is None or not path.is_file():
                missing.append(f"{task_name}/L{exit_layer}")
    split_path = resolve_recorded_path(packet.get("split_manifest"))
    if split_path is None or not split_path.is_file():
        missing.append("split_manifest")
    elif split_path:
        try:
            split = json.loads(split_path.read_text())
            if int(split.get("seed", -1)) != SPLIT_SEED:
                missing.append("split_seed_mismatch")
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            missing.append("invalid_split_manifest")
    return not missing, missing


def build_registry(checkpoint_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    entries = []
    missing_checkpoints = []
    for mask in MASKS:
        manifest_path = REPOSITORY_ROOT / "reports/masking_study" / mask / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing condition report manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        for method in METHODS:
            metrics_path = report_input(manifest, method)
            packet = load_json(metrics_path)
            override = (checkpoint_overrides or {}).get(mask, {}).get(method)
            checkpoint = resolve_recorded_path(override) if override else expected_checkpoint(manifest, method, packet)
            if checkpoint is None or not checkpoint.is_file():
                missing_checkpoints.append(f"{mask}/{method}")
            checkpoint_hash = sha256(checkpoint) if checkpoint and checkpoint.is_file() else None
            benchmark_path, benchmark_reason = validated_benchmark(manifest, method, checkpoint)
            reusable = False
            reuse_reason = "no raw seed-0 metrics packet"
            if packet is not None and metrics_path is not None:
                try:
                    packet_metrics(packet, method=method, head_seed=0)
                    packet_checkpoint = resolve_recorded_path(packet.get("checkpoint"))
                    if not checkpoint or not packet_checkpoint or sha256(packet_checkpoint) != checkpoint_hash:
                        raise ValueError("checkpoint identity mismatch")
                    complete, missing = _existing_head_files(metrics_path, packet)
                    if not complete:
                        raise ValueError("missing " + ",".join(missing))
                    reusable = True
                    reuse_reason = "validated seed-0 metrics, all head weights, checkpoint, and split"
                except (ValueError, OSError) as error:
                    reuse_reason = str(error)
            entries.append({
                "mask": mask,
                "method": method,
                "source_report_manifest": str(manifest_path),
                "seed0_metrics_path": str(metrics_path) if metrics_path else manifest.get("included_inputs", {}).get(INPUT_KEYS[method]),
                "seed0_head_seed": packet.get("head_seed") if packet else None,
                "checkpoint_path": str(checkpoint) if checkpoint else None,
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_available": bool(checkpoint and checkpoint.is_file()),
                "checkpoint_source_note": (
                    "Recovered phase5_formal checkpoint, epoch 64 of configured 240; differs from the report-referenced M-block P0 checkpoint."
                    if mask == "multi_block_n6" and method == "p0"
                    else "M-block P1 reuses the frequency_n6 P1 checkpoint, as recorded in the checkpoint mapping manifest."
                    if mask == "multi_block_n6" and method == "p1"
                    else "Checkpoint source/mapping is documented in checkpoint_bundle/README.md."
                ),
                "benchmark_reusable": benchmark_path is not None,
                "benchmark_source_path": str(benchmark_path) if benchmark_path else None,
                "benchmark_sha256": sha256(benchmark_path) if benchmark_path else None,
                "benchmark_reuse_reason": benchmark_reason,
                "seed0_reusable": reusable,
                "seed0_reuse_reason": reuse_reason,
                "head_seeds": [
                    {"seed": seed, "action": "reuse" if seed == 0 and reusable else "run"}
                    for seed in SEEDS
                ],
            })
    return {
        "protocol": "seven_seed_frozen_downstream_heads",
        "masks": list(MASKS),
        "methods": list(METHODS),
        "tasks": list(TASKS),
        "exits": list(EXITS),
        "head_seeds": list(SEEDS),
        "pretrain_seed": 0,
        "split_seed": SPLIT_SEED,
        "head_max_epochs": 200,
        "head_early_stopping_patience": 20,
        "missing_checkpoints": missing_checkpoints,
        "checkpoint_gate_passed": not missing_checkpoints,
        "entries": entries,
    }


def _job_script(entry: dict[str, Any], seed: int, run_root: Path, *, benchmark: bool, partition: str) -> str:
    mask, method = entry["mask"], entry["method"]
    suffix = "bench" if benchmark else f"hs{seed:03d}"
    job_name = f"mh-{mask}-{method}-{suffix}"
    log_dir = run_root / "slurm"
    output = log_dir / "%x-%j.out"
    error = log_dir / "%x-%j.err"
    common = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=32G",
        "#SBATCH --time=1-00:00:00",
        f"#SBATCH --output={output}",
        f"#SBATCH --error={error}",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPOSITORY_ROOT))}",
        "export PYTHONPATH=src",
        "export WANDB_KEY=offline",
        "export PYTORCH_ALLOC_CONF=expandable_segments:True",
    ]
    runtime_variant = "benchmark" if benchmark else f"head_seed_{seed:03d}"
    runtime_path = run_root / "runtime" / mask / method / runtime_variant / "runtime.json"
    profile = "source_faithful_p0" if method == "p0" else "performance_safe_hierarchical"
    runtime_code = "import json,os,socket,sys,torch; from pathlib import Path; p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); json.dump({\"job_id\":os.getenv(\"SLURM_JOB_ID\"),\"partition\":os.getenv(\"SLURM_JOB_PARTITION\"),\"nodes\":os.getenv(\"SLURM_JOB_NODELIST\"),\"hostname\":socket.gethostname(),\"cpus\":os.getenv(\"SLURM_CPUS_PER_TASK\"),\"gpu\":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,\"runtime_profile\":sys.argv[2]},p.open(\"w\"),sort_keys=True)"
    common.append(f".venv/bin/python -c {shlex.quote(runtime_code)} {shlex.quote(str(runtime_path))} {profile}")
    checkpoint = shlex.quote(entry["checkpoint_path"])
    if benchmark:
        command = (
            f".venv/bin/python scripts/phase5_benchmark.py --method {method} "
            f"--pretrain-seed 0 --checkpoint {checkpoint} --dataset-suite primary "
            f"--run-root {shlex.quote(str(run_root / mask / 'benchmarks'))} --device cuda"
        )
    else:
        isolated_root = run_root / mask / f"head_seed_{seed:03d}"
        command = (
            f".venv/bin/python scripts/phase5_exits.py --method {method} "
            f"--pretrain-seed 0 --head-seed {seed} --split-seed {SPLIT_SEED} "
            f"--checkpoint {checkpoint} --dataset-suite primary --epochs 200 "
            f"--head-early-stopping-patience 20 --run-root {shlex.quote(str(isolated_root))} "
            "--device cuda --resume"
        )
    return "\n".join(common + [command, ""])


def prepare_jobs(
    registry: dict[str, Any],
    run_root: Path,
    *,
    partition: str,
    include_benchmarks: bool = True,
) -> list[Path]:
    if not registry["checkpoint_gate_passed"]:
        missing = ", ".join(registry["missing_checkpoints"])
        raise RuntimeError(f"refusing to create/submit partial study; missing fixed checkpoints: {missing}")
    run_root.mkdir(parents=True, exist_ok=True)
    slurm_dir = run_root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_root / "registry.json"
    json_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    scripts = []
    for entry in registry["entries"]:
        for seed in SEEDS:
            if seed == 0 and entry["seed0_reusable"]:
                continue
            path = slurm_dir / f"heads_{entry['mask']}_{entry['method']}_seed{seed:03d}.sbatch"
            path.write_text(_job_script(entry, seed, run_root, benchmark=False, partition=partition))
            scripts.append(path)
        if include_benchmarks and not entry.get("benchmark_reusable", False):
            path = slurm_dir / f"benchmark_{entry['mask']}_{entry['method']}.sbatch"
            path.write_text(_job_script(entry, 0, run_root, benchmark=True, partition=partition))
            scripts.append(path)
    return scripts


def _job_log_path(script: Path, run_root: Path, job_id: str) -> Path:
    for line in script.read_text().splitlines():
        if line.startswith("#SBATCH --job-name="):
            job_name = line.split("=", 1)[1]
            return run_root / "slurm" / f"{job_name}-{job_id}.out"
    raise ValueError(f"Slurm script has no job name: {script}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("audit", "prepare", "submit"), default="audit")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--checkpoint-map", type=Path)
    parser.add_argument("--partition", default="gpu-a100,gpu-h100,gpu-l40")
    parser.add_argument("--no-benchmarks", action="store_true")
    args = parser.parse_args()
    overrides = json.loads(args.checkpoint_map.read_text()) if args.checkpoint_map else None
    registry = build_registry(overrides)
    if args.action == "audit":
        print(json.dumps(registry, indent=2))
        return 0 if registry["checkpoint_gate_passed"] else 2
    try:
        scripts = prepare_jobs(
            registry, args.run_root, partition=args.partition, include_benchmarks=not args.no_benchmarks
        )
    except RuntimeError as error:
        print(json.dumps(registry, indent=2))
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2
    if args.action == "prepare":
        for path in scripts:
            print(path)
            print(f"  submission tail command: tail -F {shlex.quote(str(_job_log_path(path, args.run_root, '<JOBID>')))}")
        return 0
    queued = []
    submissions_path = args.run_root / "submissions.json"
    for path in scripts:
        completed = subprocess.run(["sbatch", "--parsable", str(path)], text=True, capture_output=True, check=True)
        job_id = completed.stdout.strip().split(";", 1)[0]
        log_path = _job_log_path(path, args.run_root, job_id)
        queued.append({"script": str(path), "job_id": job_id, "log_path": str(log_path), "tail_command": f"tail -F {shlex.quote(str(log_path))}"})
        submissions_path.write_text(json.dumps({"submitted": queued}, indent=2, sort_keys=True) + "\n")
        print(f"submitted {job_id}: {path}\n  log: {log_path}\n  {queued[-1]['tail_command']}", flush=True)
    submissions_path.write_text(json.dumps({"submitted": queued}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"submitted": queued, "registry": str(submissions_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
