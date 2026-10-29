#!/usr/bin/env python3
"""Train one zero-residual Fusion V2 adapter ablation on frozen P0/P2 encoders."""

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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from configs import configure_method_variant, init_jepa_config  # noqa: E402
from evaluation import select_best_exit, select_earliest_within_tolerance  # noqa: E402
from models import ADAPTER_ABLATION_VARIANTS, FrozenEncoderResidualFusionAblationExit  # noqa: E402
from phase5 import PRIMARY_TASKS, build_task_dataset, task_subsets  # noqa: E402
from phase5_exits import EXIT_LAYERS, _criterion, _evaluate, _loader, _new_encoder, _seed_everything  # noqa: E402
from ee_progress import finish_epoch_progress, render_batch_progress  # noqa: E402

ALGORITHM = "residual_fusion_adapter_ablation"
MANIFEST = ROOT / "runs/masking_head_seeds/multi_block_n6/head_seed_000/splits_seed_2025_primary_v3.json"
CHECKPOINT_HASHES = {
    "p0": "8faa7245b6692e2d74d753fde2268fb712264c02c1da19632c045d022d192f1f",
    "p2": "7e1e27d96183b8f9d1c0d6b7493b93945633ccfb1514555135b4d04f22bf18ac",
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-kind", choices=("pretrained", "random"), default="pretrained")
    parser.add_argument("--random-encoder-metadata", type=Path)
    parser.add_argument("--random-encoder-seed", type=int, default=None,
                        help="random encoder initialization seed when it differs from the trained head seed")
    parser.add_argument("--method", choices=("p0", "p2"), required=True)
    parser.add_argument("--adapter-variant", choices=ADAPTER_ABLATION_VARIANTS, required=True)
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
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def parameter_counts(head) -> dict[str, int]:
    adapter_modules = [head.source_norms, head.down_projections, head.up_projections, head.single_projections]
    gate_modules = [head.anchor_gate_norm, head.anchor_gate_projection, head.source_gate_projections, head.gate_scores]
    output_modules = [head.output_norm, head.output_projection]
    count = lambda modules: sum(parameter.numel() for module in modules if module is not None for parameter in module.parameters())
    result = {"adapter": count(adapter_modules), "gate": count(gate_modules), "output": count(output_modules)}
    result["total"] = sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
    if result["total"] != result["adapter"] + result["gate"] + result["output"]:
        raise RuntimeError("trainable parameter categories are incomplete or overlapping")
    return result


def optimizer_for(model, args):
    head = model.head
    decayed = {id(head.output_projection.weight)}
    for modules in (head.down_projections, head.up_projections, head.single_projections):
        decayed.update(id(module.weight) for module in modules.values())
    decay, no_decay = [], []
    for parameter in head.parameters():
        (decay if id(parameter) in decayed else no_decay).append(parameter)
    decay_ids, no_decay_ids = {id(p) for p in decay}, {id(p) for p in no_decay}
    required = {id(p) for p in head.parameters() if p.requires_grad}
    if decay_ids & no_decay_ids or decay_ids | no_decay_ids != required:
        raise RuntimeError("optimizer parameter groups must cover every trainable parameter exactly once")
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.learning_rate)


def diagnostic_pass(model, loader, device, limit_batches=None):
    values = {layer: {"gate": [], "residual_norm": [], "relative_residual_norm": []} for layer in model.source_layers}
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if limit_batches is not None and index >= limit_batches:
                break
            output, diagnostics = model(batch[0].to(device=device, dtype=torch.float32, non_blocking=True), return_diagnostics=True)
            del output
            for layer, fields in diagnostics.items():
                for name in values[layer]:
                    values[layer][name].append(fields[name].detach().cpu().reshape(-1))
    result = {}
    for layer, fields in values.items():
        combined = {name: torch.cat(chunks).numpy() for name, chunks in fields.items()}
        result[str(layer)] = {
            "gate_mean": float(combined["gate"].mean()), "gate_sd": float(combined["gate"].std()),
            "gate_median": float(np.median(combined["gate"])), "gate_p10": float(np.percentile(combined["gate"], 10)),
            "gate_p90": float(np.percentile(combined["gate"], 90)),
            "residual_norm_mean": float(combined["residual_norm"].mean()),
            "relative_residual_norm_mean": float(combined["relative_residual_norm"].mean()),
        }
    return result


def train_exit(encoder, *, layer, spec, train_loader, validation_loader, args, device, seed, head_path):
    _seed_everything(seed)
    model = FrozenEncoderResidualFusionAblationExit(encoder, layer, spec.num_outputs, args.adapter_variant).to(device)
    criterion, optimizer = _criterion(spec, device), optimizer_for(model, args)
    best, best_epoch, stale = (-math.inf if spec.higher_is_better else math.inf), None, 0
    total_batches = len(train_loader)
    if args.limit_train_batches is not None:
        total_batches = min(total_batches, args.limit_train_batches)
    progress_label = f"{args.adapter_variant} {spec.name} L{layer}"
    for epoch in range(args.epochs):
        model.train()
        for index, batch in enumerate(train_loader):
            if args.limit_train_batches is not None and index >= args.limit_train_batches:
                break
            x, y = batch[:2]
            output = model(x.to(device=device, dtype=torch.float32, non_blocking=True))
            loss = criterion(output, y.to(device=device, non_blocking=True).long() if spec.target_type == "classification" else y.to(device=device, non_blocking=True).float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None for parameter in encoder.parameters()):
                raise RuntimeError(f"encoder received gradients at L{layer}")
            optimizer.step()
            render_batch_progress(progress_label, epoch + 1, args.epochs, index, total_batches)
        validation = _evaluate(model, validation_loader, spec, criterion, device, limit_batches=args.limit_validation_batches)
        value = float(validation[spec.metric_name])
        finish_epoch_progress(progress_label, epoch + 1, args.epochs, spec.metric_name, value)
        improved = value > best + args.head_early_stopping_min_delta if spec.higher_is_better else value < best - args.head_early_stopping_min_delta
        if improved:
            best, best_epoch, stale = value, epoch + 1, 0
            head_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"head_algorithm": ALGORITHM, "adapter_variant": args.adapter_variant, "exit_layer": layer,
                        "source_layers": list(model.source_layers), "embed_dim": model.head.embed_dim,
                        "bottleneck_dim": model.head.bottleneck_dim if model.head.is_bottleneck else None,
                        "gate_dim": model.head.gate_dim, "initialization": "zero_residual",
                        "head": model.head.state_dict(), "epoch": best_epoch, "validation": validation}, head_path)
        else:
            stale += 1
        if not args.disable_head_early_stopping and stale >= args.head_early_stopping_patience:
            break
    if best_epoch is None:
        raise RuntimeError(f"no finite validation metric for {spec.name} L{layer}")
    payload = torch.load(head_path, map_location=device, weights_only=False)
    model.head.load_state_dict(payload["head"], strict=True)
    return {"best_epoch": int(payload["epoch"]), "validation": payload["validation"], "head_path": str(head_path),
            "source_layers": payload["source_layers"], "checkpoint_sha256": digest(head_path),
            "fusion_diagnostics": diagnostic_pass(model, validation_loader, device, args.limit_validation_batches),
            "trainable_parameters": parameter_counts(model.head)}


def load_model(encoder, layer, spec, head_path, device, adapter_variant):
    model = FrozenEncoderResidualFusionAblationExit(encoder, layer, spec.num_outputs, adapter_variant).to(device)
    payload = torch.load(head_path, map_location=device, weights_only=False)
    expected_sources = [source for source in EXIT_LAYERS if source < layer]
    if (payload.get("head_algorithm") != ALGORITHM or payload.get("adapter_variant") != adapter_variant
            or int(payload.get("exit_layer", -1)) != layer or payload.get("source_layers") != expected_sources):
        raise ValueError(f"incompatible adapter-ablation checkpoint: {head_path}")
    model.head.load_state_dict(payload["head"], strict=True)
    return model


def main() -> int:
    args = arguments()
    if args.split_seed != 2025 or args.pretrain_seed != 0:
        raise ValueError("adapter ablation is locked to pretrain seed 0 and split seed 2025")
    if args.selection_only and args.test_only:
        raise ValueError("--selection-only and --test-only are mutually exclusive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    if args.encoder_kind == "random":
        from random_encoder_ablation import validate
        random_seed = args.head_seed if args.random_encoder_seed is None else args.random_encoder_seed
        if args.method not in ("p0", "p2") or args.head_seed < 0 or random_seed < 0 or args.random_encoder_metadata is None:
            raise ValueError("random encoder mode requires P0/P2 architecture, nonnegative seeds, and metadata")
        random_metadata = validate(checkpoint, args.random_encoder_metadata, random_seed)
    elif args.random_encoder_metadata is not None or args.random_encoder_seed is not None:
        raise ValueError("random encoder metadata requires random mode")
    elif not checkpoint.is_file() or digest(checkpoint) != CHECKPOINT_HASHES[args.method]:
        raise ValueError(f"checkpoint does not match the locked {args.method} reference hash: {checkpoint}")
    if not MANIFEST.is_file() or int(json.loads(MANIFEST.read_text()).get("seed", -1)) != 2025:
        raise ValueError(f"locked split manifest missing or incompatible: {MANIFEST}")
    manifest, seed = json.loads(MANIFEST.read_text()), args.head_seed
    run_dir = args.run_root / "residual_fusion_adapter_ablation" / f"head_seed_{seed:03d}" / args.adapter_variant / args.method / "seed_000/downstream"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = init_jepa_config(str(args.config or ROOT / "configs" / f"{args.method}.yml"))
    configure_method_variant(config, args.method)
    tasks, layers, output = tuple(args.task or PRIMARY_TASKS), tuple(dict.fromkeys(args.exit_layer or EXIT_LAYERS)), run_dir / "exit_metrics.json"
    protocol = {"head_max_epochs": args.epochs, "batch_size": args.batch_size, "num_workers": args.num_workers,
                "tasks": list(tasks), "exit_layers": list(layers), "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay, "head_early_stopping": {"enabled": not args.disable_head_early_stopping,
                "patience": args.head_early_stopping_patience, "min_delta": args.head_early_stopping_min_delta},
                "selection_uses": "validation_only", "test_evaluation": not args.selection_only}
    packet = {"head_algorithm": ALGORITHM, "adapter_variant": args.adapter_variant, "method": args.method,
              "pretrain_seed": 0, "head_seed": seed, "split_seed": 2025, "checkpoint": str(checkpoint.resolve()),
              "checkpoint_sha256": digest(checkpoint), "split_manifest": str(MANIFEST.resolve()),
              "split_manifest_sha256": digest(MANIFEST), "protocol": protocol, "tasks": {}}
    if args.encoder_kind == "random":
        packet.update(encoder_kind="random", pretrain_seed=None, random_encoder_metadata=str(args.random_encoder_metadata.resolve()),
                      random_encoder_seed=random_metadata["initialization_seed"])
    if args.resume or args.test_only:
        if not output.is_file():
            raise FileNotFoundError(f"saved selection packet missing: {output}")
        old = json.loads(output.read_text())
        for field in ("head_algorithm", "adapter_variant", "method", "checkpoint_sha256", "split_manifest_sha256", "head_seed", "encoder_kind"):
            if old.get(field) != packet.get(field):
                raise ValueError("saved adapter-ablation packet is incompatible with this run")
        packet = old if args.test_only else {**packet, "tasks": old.get("tasks", {})}
        packet.setdefault("protocol", {})["test_evaluation"] = True if args.test_only else packet["protocol"].get("test_evaluation")
    for task_index, task_name in enumerate(tasks):
        if task_name in packet["tasks"] and not args.test_only:
            continue
        dataset, spec = build_task_dataset(task_name, ROOT)
        subsets = task_subsets(dataset, manifest["tasks"][task_name])
        train = _loader(subsets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, seed=seed + task_index, pin_memory=device.type == "cuda")
        validation = _loader(subsets["validation"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, seed=seed, pin_memory=device.type == "cuda")
        test = None if args.selection_only else _loader(subsets["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, seed=seed, pin_memory=device.type == "cuda")
        encoder = _new_encoder(config, checkpoint, device)
        row = packet["tasks"].setdefault(task_name, {"exits": {}})
        if not args.test_only:
            row["spec"], row["exits"], values = spec.__dict__, {}, {}
            for layer in layers:
                entry = train_exit(encoder, layer=layer, spec=spec, train_loader=train, validation_loader=validation, args=args,
                                   device=device, seed=seed + task_index * 100 + layer,
                                   head_path=run_dir / "heads" / task_name / f"L{layer}.pt")
                row["exits"][str(layer)], values[layer] = entry, entry["validation"][spec.metric_name]
            row["selected_exit_best_validation"] = select_best_exit(values, higher_is_better=spec.higher_is_better)
            row["selected_exit_earliest_within_tolerance"] = select_earliest_within_tolerance(values, higher_is_better=spec.higher_is_better, tolerance=args.tolerance)
        if test is not None:
            criterion = _criterion(spec, device)
            for layer in layers if not args.test_only else tuple(int(value) for value in row["exits"]):
                entry = row["exits"][str(layer)]
                entry["test"] = _evaluate(load_model(encoder, layer, spec, Path(entry["head_path"]), device, args.adapter_variant), test, spec, criterion, device, limit_batches=args.limit_test_batches)
            row["selected_test_best_validation"] = row["exits"][str(row["selected_exit_best_validation"])]["test"]
        output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps({"metrics": str(output), "variant": args.adapter_variant, "tasks": list(tasks), "exits": list(layers)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
