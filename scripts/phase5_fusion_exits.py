#!/usr/bin/env python3
"""Train and evaluate causal fusion heads over frozen Phase 5 encoders."""

from __future__ import annotations

import hashlib
import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation import select_best_exit, select_earliest_within_tolerance  # noqa: E402
from models import FrozenEncoderFusionExit  # noqa: E402
from phase5 import PRIMARY_TASKS, build_task_dataset, task_subsets  # noqa: E402
from phase5_exits import (  # noqa: E402
    EXIT_LAYERS, _class_weights, _criterion, _evaluate, _loader,
    _new_encoder, _seed_everything,
)
from configs import configure_method_variant, init_jepa_config  # noqa: E402
from ee_progress import finish_epoch_progress, render_batch_progress  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=("p0", "p2"), required=True)
    parser.add_argument("--pretrain-seed", type=int, default=0)
    parser.add_argument("--head-seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=2025)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/ee_optimization/multi_block_n6")
    parser.add_argument("--task", choices=PRIMARY_TASKS, action="append")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-validation-batches", type=int)
    parser.add_argument("--limit-test-batches", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--head-early-stopping-patience", type=int, default=20)
    parser.add_argument("--head-early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--disable-head-early-stopping", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    return parser.parse_args()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def train_exit(encoder, *, layer, spec, train_loader, validation_loader, args, device, seed, head_path):
    _seed_everything(seed)
    model = FrozenEncoderFusionExit(encoder, layer, spec.num_outputs).to(device)
    criterion = _criterion(spec, device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best = -math.inf if spec.higher_is_better else math.inf
    best_epoch = None
    stale = 0
    total_batches = len(train_loader)
    if args.limit_train_batches is not None:
        total_batches = min(total_batches, args.limit_train_batches)
    progress_label = f"Fusion {spec.name} L{layer}"
    for epoch in range(args.epochs):
        model.train()
        for index, batch in enumerate(train_loader):
            if args.limit_train_batches is not None and index >= args.limit_train_batches:
                break
            x, y = batch[:2]
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, non_blocking=True)
            prediction = model(x)
            loss = criterion(prediction, y.long() if spec.target_type == "classification" else y.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None for p in encoder.parameters()):
                raise RuntimeError(f"encoder received gradients at L{layer}")
            optimizer.step()
            render_batch_progress(progress_label, epoch + 1, args.epochs, index, total_batches)
        validation = _evaluate(model, validation_loader, spec, criterion, device,
                               limit_batches=args.limit_validation_batches)
        value = float(validation[spec.metric_name])
        finish_epoch_progress(progress_label, epoch + 1, args.epochs, spec.metric_name, value)
        improved = value > best + args.head_early_stopping_min_delta if spec.higher_is_better else value < best - args.head_early_stopping_min_delta
        if improved:
            best, best_epoch, stale = value, epoch + 1, 0
            head_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"head_algorithm": "causal_fusion", "exit_layer": layer,
                        "source_layers": list(model.source_layers), "head": model.head.state_dict(),
                        "fusion_weights": model.head.fusion_weights, "epoch": best_epoch,
                        "validation": validation}, head_path)
        else:
            stale += 1
        if not args.disable_head_early_stopping and stale >= args.head_early_stopping_patience:
            break
    if best_epoch is None:
        raise RuntimeError(f"no finite validation metric for {spec.name} L{layer}")
    checkpoint = torch.load(head_path, map_location=device, weights_only=False)
    return {"best_epoch": best_epoch, "validation": checkpoint["validation"],
            "head_path": str(head_path), "source_layers": checkpoint["source_layers"],
            "fusion_weights": checkpoint["fusion_weights"]}


def load_model(encoder, layer, spec, head_path, device):
    model = FrozenEncoderFusionExit(encoder, layer, spec.num_outputs).to(device)
    payload = torch.load(head_path, map_location=device, weights_only=False)
    if payload.get("head_algorithm") != "causal_fusion" or int(payload.get("exit_layer", -1)) != layer:
        raise ValueError(f"incompatible fusion checkpoint: {head_path}")
    model.head.load_state_dict(payload["head"], strict=True)
    return model


def main() -> int:
    args = arguments()
    if args.split_seed != 2025:
        raise ValueError("EE optimization is locked to split seed 2025")
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs/batch size must be positive and worker count non-negative")
    if args.method not in {"p0", "p2"}:
        raise ValueError("EE optimization supports only p0 and p2")
    if args.selection_only and args.test_only:
        raise ValueError("--selection-only and --test-only are mutually exclusive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    seed = args.pretrain_seed if args.head_seed is None else args.head_seed
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    run_dir = args.run_root / "fusion" / f"head_seed_{seed:03d}" / args.method / f"seed_{args.pretrain_seed:03d}" / "downstream"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "runs/masking_head_seeds/multi_block_n6/head_seed_000/splits_seed_2025_primary_v3.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"locked Current EE split manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("seed", -1)) != 2025:
        raise ValueError(f"locked split has the wrong seed: {manifest_path}")
    config = init_jepa_config(str(args.config or ROOT / "configs" / f"{args.method}.yml"))
    configure_method_variant(config, args.method)
    tasks = tuple(args.task or PRIMARY_TASKS)
    invalid = sorted(set(tasks).difference(PRIMARY_TASKS))
    if invalid:
        raise ValueError(f"unsupported primary tasks: {invalid}")
    output = run_dir / "exit_metrics.json"
    packet = {"head_algorithm": "causal_fusion", "method": args.method,
              "pretrain_seed": args.pretrain_seed, "head_seed": seed,
              "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest(checkpoint),
              "split_manifest": str(manifest_path.resolve()), "split_manifest_sha256": digest(manifest_path),
              "split_seed": 2025, "protocol": {"head_max_epochs": args.epochs,
              "batch_size": args.batch_size, "learning_rate": args.learning_rate,
              "weight_decay": args.weight_decay,
              "head_early_stopping": {"enabled": not args.disable_head_early_stopping,
                  "patience": args.head_early_stopping_patience,
                  "min_delta": args.head_early_stopping_min_delta},
              "tolerance": args.tolerance,
              "selection_uses": "validation_only", "test_evaluation": not args.selection_only}, "tasks": {}}
    if args.test_only:
        if not output.is_file():
            raise FileNotFoundError(f"--test-only requires saved selection packet: {output}")
        packet = json.loads(output.read_text())
        if (packet.get("head_algorithm") != "causal_fusion" or packet.get("checkpoint_sha256") != digest(checkpoint)
                or int(packet.get("head_seed", -1)) != seed or int(packet.get("pretrain_seed", -1)) != args.pretrain_seed
                or packet.get("split_manifest_sha256") != digest(manifest_path)):
            raise ValueError("saved selection packet is incompatible with this checkpoint/algorithm")
    elif args.resume and output.is_file():
        old = json.loads(output.read_text())
        if old.get("head_algorithm") != "causal_fusion" or old.get("checkpoint_sha256") != packet["checkpoint_sha256"] or old.get("protocol", {}) != packet["protocol"]:
            raise ValueError("cannot resume an incompatible fusion packet")
        packet["tasks"] = old.get("tasks", {})

    for task_index, task_name in enumerate(tasks):
        if task_name in packet["tasks"] and not args.test_only:
            continue
        dataset, spec = build_task_dataset(task_name, ROOT)
        subsets = task_subsets(dataset, manifest["tasks"][task_name])
        train_loader = _loader(subsets["train"], batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, seed=seed + task_index, pin_memory=device.type == "cuda")
        val_loader = _loader(subsets["validation"], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, seed=seed, pin_memory=device.type == "cuda")
        test_loader = None if args.selection_only else _loader(subsets["test"], batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers, seed=seed, pin_memory=device.type == "cuda")
        encoder = _new_encoder(config, checkpoint, device)
        result = packet["tasks"].setdefault(task_name, {"exits": {}})
        if args.test_only:
            values = {int(k): v["validation"][spec.metric_name] for k, v in result["exits"].items()}
        else:
            result["spec"] = spec.__dict__
            result["exits"] = {}
            values = {}
            for layer in EXIT_LAYERS:
                trained = train_exit(encoder, layer=layer, spec=spec, train_loader=train_loader,
                                     validation_loader=val_loader, args=args, device=device,
                                     seed=seed + task_index * 100 + layer,
                                     head_path=run_dir / "heads" / task_name / f"L{layer}.pt")
                values[layer] = trained["validation"][spec.metric_name]
                result["exits"][str(layer)] = trained
            result["selected_exit_best_validation"] = select_best_exit(values, higher_is_better=spec.higher_is_better)
            result["selected_exit_earliest_within_tolerance"] = select_earliest_within_tolerance(
                values, higher_is_better=spec.higher_is_better, tolerance=args.tolerance)
        if test_loader is not None:
            criterion = _criterion(spec, device)
            for layer in EXIT_LAYERS:
                model = load_model(encoder, layer, spec, Path(result["exits"][str(layer)]["head_path"]), device)
                result["exits"][str(layer)]["test"] = _evaluate(model, test_loader, spec, criterion, device,
                                                                  limit_batches=args.limit_test_batches)
            result["selected_test_best_validation"] = result["exits"][str(result["selected_exit_best_validation"])]["test"]
        packet["protocol"]["test_evaluation"] = test_loader is not None
        output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps({"metrics": str(output), "tasks": list(tasks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
