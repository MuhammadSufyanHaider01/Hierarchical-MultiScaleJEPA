#!/usr/bin/env python3
"""Measure batch-1 causal fusion inference cost for each task and exit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from configs import configure_method_variant, init_jepa_config  # noqa: E402
from models import FrozenEncoderFusionExit  # noqa: E402
from phase5 import PRIMARY_TASKS, build_task_dataset  # noqa: E402
from phase5_exits import _new_encoder  # noqa: E402
from phase5_benchmark import _measure_latency, _synchronize  # noqa: E402
from phase5_fusion_exits import digest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=("p0", "p2"), required=True)
    parser.add_argument("--pretrain-seed", type=int, default=0)
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--measure-iterations", type=int, default=200)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/ee_optimization/multi_block_n6")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    config = init_jepa_config(str(ROOT / "configs" / f"{args.method}.yml"))
    configure_method_variant(config, args.method)
    encoder = _new_encoder(config, checkpoint, device)
    results = []
    for task in PRIMARY_TASKS:
        _, spec = build_task_dataset(task, ROOT)
        sample = torch.randn((1, *spec.sample_shape), device=device)
        for layer in (2, 4, 6, 8):
            model = FrozenEncoderFusionExit(encoder, layer, spec.num_outputs).to(device).eval()
            _synchronize(device)
            with torch.inference_mode(), FlopCounterMode(display=False) as counter:
                model(sample)
            results.append({"task": task, "exit_layer": layer, "flops": int(counter.get_total_flops()),
                "latency": _measure_latency(model, sample, device,
                    warmup_iterations=args.warmup_iterations, measure_iterations=args.measure_iterations)})
    output = args.run_root / "benchmarks" / "fusion" / args.method / f"seed_{args.pretrain_seed:03d}" / "benchmarks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"head_algorithm": "causal_fusion", "method": args.method,
        "pretrain_seed": args.pretrain_seed, "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest(checkpoint), "batch_size": 1, "device": str(device),
        "warmup_iterations": args.warmup_iterations, "measure_iterations": args.measure_iterations,
        "entries": results}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"benchmarks": str(output), "entries": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
