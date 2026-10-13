#!/usr/bin/env python3
"""Train and evaluate V2 residual-dynamic fusion heads on frozen encoders."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation import select_best_exit, select_earliest_within_tolerance  # noqa: E402
from models import FrozenEncoderResidualFusionExit  # noqa: E402
from phase5 import PRIMARY_TASKS, build_task_dataset, task_subsets  # noqa: E402
from phase5_exits import EXIT_LAYERS, _criterion, _evaluate, _loader, _new_encoder, _seed_everything  # noqa: E402
from configs import configure_method_variant, init_jepa_config  # noqa: E402
from ee_progress import finish_epoch_progress, render_batch_progress  # noqa: E402

ALGORITHM = "residual_projected_dynamic_fusion"
MANIFEST = ROOT / "runs/masking_head_seeds/multi_block_n6/head_seed_000/splits_seed_2025_primary_v3.json"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=("p0", "p2"), required=True)
    parser.add_argument("--pretrain-seed", type=int, default=0)
    parser.add_argument("--head-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=2025)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/ee_optimization/multi_block_n6")
    parser.add_argument("--task", choices=PRIMARY_TASKS, action="append")
    parser.add_argument("--exit-layer", type=int, choices=EXIT_LAYERS, action="append")
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


def optimizer_for(model, args):
    decay, no_decay = [], []
    decayed_weights = {id(module.weight) for group in (model.head.down_projections, model.head.up_projections)
                       for module in group.values()}
    decayed_weights.add(id(model.head.output_projection.weight))
    for parameter in model.head.parameters():
        (decay if id(parameter) in decayed_weights else no_decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=args.learning_rate)


def diagnostic_pass(model, loader, device, limit_batches=None):
    samples = {layer: {"gate": [], "residual_norm": [], "relative_residual_norm": []} for layer in model.source_layers}
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if limit_batches is not None and index >= limit_batches:
                break
            x = batch[0].to(device=device, dtype=torch.float32, non_blocking=True)
            _, diagnostics = model(x, return_diagnostics=True)
            for layer, values in diagnostics.items():
                for name in samples[layer]:
                    samples[layer][name].append(values[name].detach().cpu().reshape(-1))
    output = {}
    for layer, fields in samples.items():
        vals = {name: torch.cat(chunks).numpy() for name, chunks in fields.items()}
        output[str(layer)] = {
            "gate_mean": float(vals["gate"].mean()), "gate_sd": float(vals["gate"].std()),
            "gate_median": float(np.median(vals["gate"])), "gate_p10": float(np.percentile(vals["gate"], 10)),
            "gate_p90": float(np.percentile(vals["gate"], 90)),
            "residual_norm_mean": float(vals["residual_norm"].mean()),
            "relative_residual_norm_mean": float(vals["relative_residual_norm"].mean()),
        }
    return output


def train_exit(encoder, *, layer, spec, train_loader, validation_loader, args, device, seed, head_path):
    _seed_everything(seed)
    model = FrozenEncoderResidualFusionExit(encoder, layer, spec.num_outputs).to(device)
    criterion = _criterion(spec, device)
    optimizer = optimizer_for(model, args)
    best = -math.inf if spec.higher_is_better else math.inf
    best_epoch = None
    stale = 0
    total = min(len(train_loader), args.limit_train_batches or len(train_loader))
    label = f"FusionV2 {spec.name} L{layer}"
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
            render_batch_progress(label, epoch + 1, args.epochs, index, total)
        validation = _evaluate(model, validation_loader, spec, criterion, device,
                               limit_batches=args.limit_validation_batches)
        value = float(validation[spec.metric_name])
        finish_epoch_progress(label, epoch + 1, args.epochs, spec.metric_name, value)
        improved = value > best + args.head_early_stopping_min_delta if spec.higher_is_better else value < best - args.head_early_stopping_min_delta
        if improved:
            best, best_epoch, stale = value, epoch + 1, 0
            head_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"head_algorithm": ALGORITHM, "exit_layer": layer,
                        "source_layers": list(model.source_layers), "embed_dim": model.head.embed_dim,
                        "bottleneck_dim": model.head.bottleneck_dim, "gate_dim": model.head.gate_dim,
                        "head": model.head.state_dict(), "epoch": best_epoch, "validation": validation}, head_path)
        else:
            stale += 1
        if not args.disable_head_early_stopping and stale >= args.head_early_stopping_patience:
            break
    if best_epoch is None:
        raise RuntimeError(f"no finite validation metric for {spec.name} L{layer}")
    payload = torch.load(head_path, map_location=device, weights_only=False)
    model.head.load_state_dict(payload["head"], strict=True)
    diagnostics = diagnostic_pass(model, validation_loader, device, args.limit_validation_batches)
    return {"best_epoch": int(payload["epoch"]), "validation": payload["validation"],
            "head_path": str(head_path), "source_layers": payload["source_layers"],
            "checkpoint_sha256": digest(head_path), "fusion_diagnostics": diagnostics}


def load_model(encoder, layer, spec, head_path, device):
    model = FrozenEncoderResidualFusionExit(encoder, layer, spec.num_outputs).to(device)
    payload = torch.load(head_path, map_location=device, weights_only=False)
    expected_sources = [j for j in EXIT_LAYERS if j < layer]
    if (payload.get("head_algorithm") != ALGORITHM or int(payload.get("exit_layer", -1)) != layer
            or payload.get("source_layers") != expected_sources):
        raise ValueError(f"incompatible residual-fusion checkpoint: {head_path}")
    model.head.load_state_dict(payload["head"], strict=True)
    return model


def main() -> int:
    args = arguments()
    if args.split_seed != 2025:
        raise ValueError("EE optimization is locked to split seed 2025")
    if args.selection_only and args.test_only:
        raise ValueError("--selection-only and --test-only are mutually exclusive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    seed = args.head_seed
    run_dir = args.run_root / "residual_dynamic_fusion" / f"head_seed_{seed:03d}" / args.method / f"seed_{args.pretrain_seed:03d}" / "downstream"
    run_dir.mkdir(parents=True, exist_ok=True)
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"locked split manifest missing: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text())
    if int(manifest.get("seed", -1)) != 2025:
        raise ValueError("locked split manifest has incorrect seed")
    config = init_jepa_config(str(args.config or ROOT / "configs" / f"{args.method}.yml"))
    configure_method_variant(config, args.method)
    tasks = tuple(args.task or PRIMARY_TASKS)
    layers = tuple(dict.fromkeys(args.exit_layer or EXIT_LAYERS))
    output = run_dir / "exit_metrics.json"
    protocol = {"head_max_epochs": args.epochs, "batch_size": args.batch_size,
                "num_workers": args.num_workers, "tasks": list(tasks), "exit_layers": list(layers),
                "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                "head_early_stopping": {"enabled": not args.disable_head_early_stopping,
                    "patience": args.head_early_stopping_patience, "min_delta": args.head_early_stopping_min_delta},
                "tolerance": args.tolerance, "selection_uses": "validation_only",
                "test_evaluation": not args.selection_only}
    packet = {"head_algorithm": ALGORITHM, "method": args.method, "pretrain_seed": args.pretrain_seed,
              "head_seed": seed, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest(checkpoint),
              "split_manifest": str(MANIFEST.resolve()), "split_manifest_sha256": digest(MANIFEST),
              "split_seed": 2025, "protocol": protocol, "tasks": {}}
    if args.test_only or args.resume:
        if not output.is_file():
            raise FileNotFoundError(f"saved selection packet missing: {output}")
        old = json.loads(output.read_text())
        old_protocol = dict(old.get("protocol", {}))
        new_protocol = dict(protocol)
        old_protocol.pop("test_evaluation", None)
        new_protocol.pop("test_evaluation", None)
        if (old.get("head_algorithm") != ALGORITHM or old.get("method") != args.method
                or old.get("checkpoint_sha256") != packet["checkpoint_sha256"]
                or old.get("split_manifest_sha256") != packet["split_manifest_sha256"]
                or int(old.get("head_seed", -1)) != seed or old_protocol != new_protocol):
            raise ValueError("saved residual-fusion packet is incompatible with this run")
        if args.test_only:
            packet = old
            packet.setdefault("protocol", {})["test_evaluation"] = True
        else:
            packet["tasks"] = old.get("tasks", {})

    for ti, task_name in enumerate(tasks):
        if task_name in packet["tasks"] and not args.test_only:
            continue
        dataset, spec = build_task_dataset(task_name, ROOT)
        subsets = task_subsets(dataset, manifest["tasks"][task_name])
        train_loader = _loader(subsets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                               seed=seed + ti, pin_memory=device.type == "cuda")
        val_loader = _loader(subsets["validation"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             seed=seed, pin_memory=device.type == "cuda")
        test_loader = None if args.selection_only else _loader(subsets["test"], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, seed=seed, pin_memory=device.type == "cuda")
        encoder = _new_encoder(config, checkpoint, device)
        result = packet["tasks"].setdefault(task_name, {"exits": {}})
        if args.test_only:
            values = {int(k): v["validation"][spec.metric_name] for k, v in result["exits"].items()}
        else:
            result["spec"] = spec.__dict__
            result["exits"] = {}
            values = {}
            for layer in layers:
                trained = train_exit(encoder, layer=layer, spec=spec, train_loader=train_loader,
                    validation_loader=val_loader, args=args, device=device,
                    seed=seed + ti * 100 + layer,
                    head_path=run_dir / "heads" / task_name / f"L{layer}.pt")
                values[layer] = trained["validation"][spec.metric_name]
                result["exits"][str(layer)] = trained
            result["selected_exit_best_validation"] = select_best_exit(values, higher_is_better=spec.higher_is_better)
            result["selected_exit_earliest_within_tolerance"] = select_earliest_within_tolerance(
                values, higher_is_better=spec.higher_is_better, tolerance=args.tolerance)
        if test_loader is not None:
            criterion = _criterion(spec, device)
            for layer in layers if not args.test_only else tuple(int(x) for x in result["exits"]):
                entry = result["exits"][str(layer)]
                model = load_model(encoder, layer, spec, Path(entry["head_path"]), device)
                entry["test"] = _evaluate(model, test_loader, spec, criterion, device, limit_batches=args.limit_test_batches)
            result["selected_test_best_validation"] = result["exits"][str(result["selected_exit_best_validation"])]["test"]
        output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps({"metrics": str(output), "tasks": list(tasks), "exits": list(layers)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
