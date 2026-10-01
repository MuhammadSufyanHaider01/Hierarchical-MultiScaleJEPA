#!/usr/bin/env python3
"""Measure per-task, batch-1 frozen-exit FLOPs and latency for Phase 5.

The benchmark is intentionally separate from pretraining.  It measures the
frozen context encoder plus its mean-pool/LayerNorm/linear exit head under the
same device, precision, warm-up, and synchronization protocol for L2/L4/L6/L8.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
os.environ.setdefault("WANDB_KEY", "offline")

from checkpointing import environment_metadata  # noqa: E402
from configs import configure_method_variant, init_jepa_config  # noqa: E402
from models import Encoder, FrozenEncoderExit, build_encoder, load_context_encoder_checkpoint  # noqa: E402
from phase5 import PRIMARY_TASKS, build_task_dataset  # noqa: E402
from phase6 import OOD_TASKS, build_ood_task_dataset, default_ood_root  # noqa: E402


TASKS = tuple(dict.fromkeys(PRIMARY_TASKS + OOD_TASKS))
EXIT_LAYERS = (2, 4, 6, 8)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=("p0", "p1", "p2", "p3", "p2_reverse", "cnn_p1", "cnn_p2", "cnn_p2_reverse"), required=True)
    parser.add_argument("--pretrain-seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs/phase5")
    parser.add_argument("--dataset-suite", choices=("primary", "ood"), default="primary")
    parser.add_argument("--ood-root", type=Path, default=default_ood_root(REPOSITORY_ROOT))
    parser.add_argument("--task", choices=TASKS, action="append")
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--measure-iterations", type=int, default=200)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def _new_encoder(config, checkpoint: Path, device: torch.device) -> Encoder:
    encoder = build_encoder(config.encoder).to(device)
    load_context_encoder_checkpoint(encoder, checkpoint)
    encoder.eval()
    return encoder


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_flops(model: FrozenEncoderExit, sample: torch.Tensor, device: torch.device) -> int:
    _synchronize(device)
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        model(sample)
    _synchronize(device)
    return int(counter.get_total_flops())


def _measure_latency(
    model: FrozenEncoderExit,
    sample: torch.Tensor,
    device: torch.device,
    *,
    warmup_iterations: int,
    measure_iterations: int,
) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(sample)
        _synchronize(device)
        samples_ms = []
        for _ in range(measure_iterations):
            _synchronize(device)
            started = time.perf_counter_ns()
            model(sample)
            _synchronize(device)
            samples_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    ordered = sorted(samples_ms)
    percentile_index = max(0, min(len(ordered) - 1, round(0.95 * (len(ordered) - 1))))
    return {
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p95_ms": ordered[percentile_index],
        "standard_deviation_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
    }


def main() -> int:
    args = _arguments()
    if min(args.warmup_iterations, args.measure_iterations) < 1:
        raise ValueError("warmup and measurement iterations must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPOSITORY_ROOT / args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config_method = {"cnn_p1": "p1", "cnn_p2": "p2", "cnn_p2_reverse": "p2", "p2_reverse": "p2", "p3": "p2"}.get(args.method, args.method)
    config_path = args.config or (REPOSITORY_ROOT / "configs" / f"{config_method}.yml")
    config = init_jepa_config(str(config_path))
    configure_method_variant(config, args.method)
    encoder = _new_encoder(config, checkpoint, device)
    generator = torch.Generator(device=device).manual_seed(0)
    if args.dataset_suite == "primary":
        allowed_tasks = PRIMARY_TASKS
        build_dataset = lambda task: build_task_dataset(task, REPOSITORY_ROOT)
    else:
        allowed_tasks = OOD_TASKS
        build_dataset = lambda task: build_ood_task_dataset(task, args.ood_root)
    tasks = tuple(args.task or allowed_tasks)
    unsupported = sorted(set(tasks).difference(allowed_tasks))
    if unsupported:
        raise ValueError(f"tasks {unsupported} are not in the {args.dataset_suite} dataset suite")
    entries = []
    for task_name in tasks:
        _, spec = build_dataset(task_name)
        sample = torch.randn((1, *spec.sample_shape), generator=generator, device=device, dtype=torch.float32)
        for exit_layer in EXIT_LAYERS:
            model = FrozenEncoderExit(encoder, exit_layer=exit_layer, num_outputs=spec.num_outputs).to(device).eval()
            entries.append({
                "task": task_name,
                "sample_shape": list(spec.sample_shape),
                "exit_layer": exit_layer,
                "flops": _measure_flops(model, sample, device),
                "latency": _measure_latency(
                    model,
                    sample,
                    device,
                    warmup_iterations=args.warmup_iterations,
                    measure_iterations=args.measure_iterations,
                ),
            })

    output = args.run_root / args.method / f"seed_{args.pretrain_seed:03d}" / "benchmarks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "method": args.method,
        "pretrain_seed": args.pretrain_seed,
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "dataset_suite": args.dataset_suite,
        "ood_root": str(args.ood_root.resolve()) if args.dataset_suite == "ood" else None,
        "device": str(device),
        "precision": "float32 with torch.set_float32_matmul_precision('high')",
        "batch_size": 1,
        "warmup_iterations": args.warmup_iterations,
        "measure_iterations": args.measure_iterations,
        "synchronized": device.type == "cuda",
        "environment": environment_metadata(),
        "entries": entries,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({"benchmarks": str(output), "entries": len(entries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
