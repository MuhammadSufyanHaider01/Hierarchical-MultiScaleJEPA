#!/usr/bin/env python3
"""Train and evaluate fixed frozen-context exits for one Phase 5 checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
os.environ.setdefault("WANDB_KEY", "offline")

from configs import init_jepa_config  # noqa: E402
from evaluation import select_best_exit, select_earliest_within_tolerance  # noqa: E402
from models import Encoder, FrozenEncoderExit, load_context_encoder_checkpoint  # noqa: E402
from phase5 import TaskSpec, build_task_dataset, load_or_create_split_manifest, task_subsets  # noqa: E402
from phase6 import (  # noqa: E402
    OOD_TASKS,
    build_ood_task_dataset,
    default_ood_root,
    load_or_create_ood_split_manifest,
    ood_task_subsets,
)


PRIMARY_TASKS = ("rfs", "nrpos-outdoor", "deepmimo-beam", "human-activity")
TASKS = tuple(dict.fromkeys(PRIMARY_TASKS + OOD_TASKS))
EXIT_LAYERS = (2, 4, 6, 8)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=("p0", "p1", "p2"), required=True)
    parser.add_argument("--pretrain-seed", type=int, required=True)
    parser.add_argument("--head-seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=2025)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Encoder config; defaults to the config corresponding to --method.",
    )
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs/phase5")
    parser.add_argument("--dataset-suite", choices=("primary", "ood"), default="primary")
    parser.add_argument("--ood-root", type=Path, default=default_ood_root(REPOSITORY_ROOT))
    parser.add_argument("--task", choices=TASKS, action="append")
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum head-training epochs; validation-based head early stopping may finish sooner.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-train-batches", type=int, default=None, help="Smoke-test limit only; omitted uses all batches.")
    parser.add_argument("--limit-validation-batches", type=int, default=None, help="Smoke-test limit only; omitted uses all batches.")
    parser.add_argument("--limit-test-batches", type=int, default=None, help="Smoke-test limit only; omitted uses all batches.")
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--head-early-stopping-patience",
        type=int,
        default=20,
        help="Stop an independent frozen head after this many non-improving validation epochs.",
    )
    parser.add_argument(
        "--head-early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Required validation-metric improvement before resetting head early-stopping patience.",
    )
    parser.add_argument(
        "--disable-head-early-stopping",
        action="store_true",
        help="Use the full --epochs budget for a diagnostic head run.",
    )
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true", help="Keep completed task results in exit_metrics.json.")
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help=(
            "Train frozen heads and select exits using validation only. Do not construct, read, "
            "or report the test split; use this for checkpoint-selection probes."
        ),
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help=(
            "Evaluate the held-out split using heads saved by an earlier --selection-only run. "
            "This never trains heads or revisits validation for exit selection."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int, seed: int, pin_memory: bool):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator if shuffle else None,
    )


def _class_weights(spec: TaskSpec, device: torch.device) -> torch.Tensor | None:
    attr = {
        "deepmimo-beam": "class_weights_beam_16",
    }.get(spec.name)
    if attr is None:
        return None
    with h5py.File(REPOSITORY_ROOT / spec.path, "r") as h5:
        raw_weights = h5.attrs[attr]
    if isinstance(raw_weights, bytes):
        raw_weights = raw_weights.decode("utf-8")
    if isinstance(raw_weights, str):
        weights = torch.from_numpy(np.fromstring(raw_weights.strip().strip("[]"), sep=" ", dtype=np.float32))
    else:
        weights = torch.as_tensor(raw_weights, dtype=torch.float32)
    if weights.numel() != spec.num_outputs:
        raise ValueError(f"{attr} has {weights.numel()} entries, expected {spec.num_outputs}")
    return weights.to(device)


def _criterion(spec: TaskSpec, device: torch.device) -> nn.Module:
    if spec.target_type == "classification":
        return nn.CrossEntropyLoss(weight=_class_weights(spec, device))
    return nn.MSELoss()


def _evaluate(
    model: FrozenEncoderExit,
    loader,
    spec: TaskSpec,
    criterion: nn.Module,
    device: torch.device,
    *,
    limit_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    if spec.target_type == "classification":
        class_correct = torch.zeros(spec.num_outputs, device=device)
        class_total = torch.zeros(spec.num_outputs, device=device)
        correct = 0
    else:
        distance_errors = []
        coord_min = torch.tensor(spec.coord_min, device=device)
        coord_max = torch.tensor(spec.coord_max, device=device)

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if limit_batches is not None and batch_index >= limit_batches:
                break
            samples, labels = batch[:2]
            samples = samples.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            logits = model(samples)
            if spec.target_type == "classification":
                labels = labels.long()
                loss = criterion(logits, labels)
                prediction = logits.argmax(dim=1)
                correct += int((prediction == labels).sum().item())
                class_total.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
                class_correct.scatter_add_(0, labels, (prediction == labels).to(torch.float32))
            else:
                labels = labels.float()
                loss = criterion(logits, labels)
                prediction = (logits + 1.0) * 0.5 * (coord_max - coord_min) + coord_min
                target = (labels + 1.0) * 0.5 * (coord_max - coord_min) + coord_min
                distance_errors.append(torch.linalg.vector_norm(prediction - target, dim=-1))
            batch_size = samples.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

    metrics = {"loss": total_loss / max(1, total_examples)}
    if spec.target_type == "classification":
        present = class_total > 0
        per_class = torch.where(present, class_correct / class_total, torch.zeros_like(class_total))
        metrics["accuracy"] = correct / max(1, total_examples)
        metrics["macro_accuracy"] = float(per_class[present].mean().item())
    else:
        errors = torch.cat(distance_errors)
        metrics["mean_distance_error"] = float(errors.mean().item())
        metrics["median_distance_error"] = float(errors.median().item())
    return metrics


def _new_encoder(config, checkpoint: Path, device: torch.device) -> Encoder:
    encoder = Encoder(
        img_size=tuple(config.encoder.img_size),
        patch_size=config.encoder.patch_size,
        embed_dim=config.encoder.embed_dim,
        depth=config.encoder.depth,
        num_heads=config.encoder.num_heads,
    ).to(device)
    load_context_encoder_checkpoint(encoder, checkpoint)
    encoder.eval()
    return encoder


def _train_exit(
    encoder: Encoder,
    *,
    exit_layer: int,
    spec: TaskSpec,
    train_loader,
    validation_loader,
    args: argparse.Namespace,
    device: torch.device,
    head_seed: int,
    head_path: Path,
) -> dict[str, object]:
    _seed_everything(head_seed)
    model = FrozenEncoderExit(encoder, exit_layer=exit_layer, num_outputs=spec.num_outputs).to(device)
    criterion = _criterion(spec, device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_value = -math.inf if spec.higher_is_better else math.inf
    best_epoch = None
    non_improving_epochs = 0
    stopped_early = False
    epochs_completed = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_examples = 0
        for batch_index, batch in enumerate(train_loader):
            if args.limit_train_batches is not None and batch_index >= args.limit_train_batches:
                break
            samples, labels = batch[:2]
            samples = samples.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            output = model(samples)
            loss = criterion(output, labels.long() if spec.target_type == "classification" else labels.float())
            batch_size = samples.size(0)
            train_loss_sum += float(loss.detach().item()) * batch_size
            train_examples += batch_size
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None for parameter in encoder.parameters()):
                raise RuntimeError(f"context backbone received gradients at L{exit_layer}")
            optimizer.step()

        validation = _evaluate(
            model,
            validation_loader,
            spec,
            criterion,
            device,
            limit_batches=args.limit_validation_batches,
        )
        value = validation[spec.metric_name]
        train_loss = train_loss_sum / max(1, train_examples)
        print(
            f"[{spec.name} L{exit_layer} epoch {epoch + 1:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={validation['loss']:.6f} "
            f"val_{spec.metric_name}={value:.6f}",
            flush=True,
        )
        if not math.isfinite(float(value)):
            raise RuntimeError(f"non-finite validation metric for task {spec.name} at L{exit_layer}: {value}")
        if spec.higher_is_better:
            improved = value > best_value + args.head_early_stopping_min_delta
        else:
            improved = value < best_value - args.head_early_stopping_min_delta
        if improved:
            best_value = value
            best_epoch = epoch + 1
            non_improving_epochs = 0
            head_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"head": model.head.state_dict(), "epoch": best_epoch, "validation": validation}, head_path)
        else:
            non_improving_epochs += 1
        epochs_completed = epoch + 1
        if not args.disable_head_early_stopping and non_improving_epochs >= args.head_early_stopping_patience:
            stopped_early = True
            print(
                f"[{spec.name} L{exit_layer}] early_stop after epoch {epochs_completed:03d}; "
                f"best_epoch={best_epoch:03d} best_val_{spec.metric_name}={best_value:.6f}",
                flush=True,
            )
            break

    if best_epoch is None:
        raise RuntimeError(f"no finite validation metric for task {spec.name} at L{exit_layer}")
    checkpoint = torch.load(head_path, map_location=device, weights_only=False)
    return {
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "stopped_early": stopped_early,
        "validation": checkpoint["validation"],
        "head_path": str(head_path),
    }


def _evaluate_saved_head(
    encoder: Encoder,
    *,
    exit_layer: int,
    spec: TaskSpec,
    head_path: Path,
    test_loader,
    device: torch.device,
    limit_batches: int | None = None,
) -> dict[str, float]:
    model = FrozenEncoderExit(encoder, exit_layer=exit_layer, num_outputs=spec.num_outputs).to(device)
    checkpoint = torch.load(head_path, map_location=device, weights_only=False)
    model.head.load_state_dict(checkpoint["head"], strict=True)
    return _evaluate(
        model,
        test_loader,
        spec,
        _criterion(spec, device),
        device,
        limit_batches=limit_batches,
    )


def main() -> int:
    args = _arguments()
    if min(args.epochs, args.batch_size, args.num_workers + 1) < 1:
        raise ValueError("epochs, batch size, and workers must be positive")
    if any(limit is not None and limit < 1 for limit in (args.limit_train_batches, args.limit_validation_batches, args.limit_test_batches)):
        raise ValueError("batch limits must be positive when set")
    if args.tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if args.selection_only and args.test_only:
        raise ValueError("--selection-only and --test-only are mutually exclusive")
    if args.head_early_stopping_patience < 1:
        raise ValueError("head-early-stopping-patience must be positive")
    if args.head_early_stopping_min_delta < 0:
        raise ValueError("head-early-stopping-min-delta must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    torch.set_float32_matmul_precision("high")
    head_seed = args.pretrain_seed if args.head_seed is None else args.head_seed
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPOSITORY_ROOT / args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    run_dir = args.run_root / args.method / f"seed_{args.pretrain_seed:03d}" / "downstream"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset_suite == "primary":
        allowed_tasks = PRIMARY_TASKS
        manifest_path = args.run_root / f"splits_seed_{args.split_seed}_primary_v2.json"
        manifest = load_or_create_split_manifest(manifest_path, REPOSITORY_ROOT, seed=args.split_seed)
        build_dataset = lambda task: build_task_dataset(task, REPOSITORY_ROOT)
        make_subsets = task_subsets
    else:
        allowed_tasks = OOD_TASKS
        manifest_path = args.run_root / f"ood_splits_seed_{args.split_seed}.json"
        manifest = load_or_create_ood_split_manifest(manifest_path, args.ood_root, seed=args.split_seed)
        build_dataset = lambda task: build_ood_task_dataset(task, args.ood_root)
        make_subsets = ood_task_subsets
    config_path = args.config or (REPOSITORY_ROOT / "configs" / f"{args.method}.yml")
    config = init_jepa_config(str(config_path))
    tasks = tuple(args.task or allowed_tasks)
    unsupported = sorted(set(tasks).difference(allowed_tasks))
    if unsupported:
        raise ValueError(f"tasks {unsupported} are not in the {args.dataset_suite} dataset suite")
    output = run_dir / "exit_metrics.json"
    if args.test_only:
        if not output.is_file():
            raise FileNotFoundError(
                f"--test-only requires saved validation-selected heads and metrics: {output}"
            )
        all_results = json.loads(output.read_text())
        if all_results.get("method") != args.method or int(all_results.get("pretrain_seed", -1)) != args.pretrain_seed:
            raise ValueError(f"cannot test incompatible metrics file: {output}")
        for task_name in tasks:
            if task_name not in all_results.get("tasks", {}):
                raise ValueError(f"--test-only is missing saved results for task: {task_name}")
            dataset, spec = build_dataset(task_name)
            split_record = manifest["tasks"][task_name]
            subsets = make_subsets(dataset, split_record)
            test_loader = _loader(
                subsets["test"],
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                seed=head_seed,
                pin_memory=device.type == "cuda",
            )
            encoder = _new_encoder(config, checkpoint, device)
            task_result = all_results["tasks"][task_name]
            exit_results = task_result.get("exits", {})
            for exit_layer in EXIT_LAYERS:
                record = exit_results.get(str(exit_layer))
                if record is None:
                    raise ValueError(f"--test-only is missing L{exit_layer} head metadata for {task_name}")
                head_path = Path(record["head_path"])
                if not head_path.is_file():
                    raise FileNotFoundError(f"--test-only head is missing: {head_path}")
                record["test"] = _evaluate_saved_head(
                    encoder,
                    exit_layer=exit_layer,
                    spec=spec,
                    head_path=head_path,
                    test_loader=test_loader,
                    device=device,
                    limit_batches=args.limit_test_batches,
                )
            selected_best = int(task_result["selected_exit_best_validation"])
            selected_tolerance = int(task_result["selected_exit_earliest_within_tolerance"])
            task_result["selected_test_best_validation"] = exit_results[str(selected_best)]["test"]
            task_result["selected_test_earliest_within_tolerance"] = exit_results[str(selected_tolerance)]["test"]
            if device.type == "cuda":
                torch.cuda.empty_cache()
        all_results.setdefault("protocol", {})["test_evaluation"] = True
        all_results["protocol"]["test_evaluation_mode"] = "post_selection_saved_heads"
        all_results["protocol"]["selection_uses"] = "validation_only"
        output.write_text(json.dumps(all_results, indent=2, sort_keys=True))
        print(json.dumps({"metrics": str(output), "tasks": list(tasks), "mode": "test_only"}, indent=2))
        return 0
    all_results = {
        "method": args.method,
        "pretrain_seed": args.pretrain_seed,
        "head_seed": head_seed,
        "checkpoint": str(checkpoint),
        "dataset_suite": args.dataset_suite,
        "ood_root": str(args.ood_root.resolve()) if args.dataset_suite == "ood" else None,
        "split_manifest": str(manifest_path),
        "protocol": {
            "head_max_epochs": args.epochs,
            "head_early_stopping": {
                "enabled": not args.disable_head_early_stopping,
                "monitor": "task-specific validation metric",
                "patience": args.head_early_stopping_patience,
                "min_delta": args.head_early_stopping_min_delta,
            },
            "tolerance": args.tolerance,
            "selection_uses": "validation_only",
            "test_evaluation": not args.selection_only,
            "batch_limits": {
                "train": args.limit_train_batches,
                "validation": args.limit_validation_batches,
                "test": args.limit_test_batches,
            },
        },
        "tasks": {},
    }
    if args.resume and output.is_file():
        previous = json.loads(output.read_text())
        if previous.get("method") != args.method or int(previous.get("pretrain_seed", -1)) != args.pretrain_seed:
            raise ValueError(f"cannot resume incompatible metrics file: {output}")
        all_results["tasks"] = previous.get("tasks", {})

    for task_index, task_name in enumerate(tasks):
        if task_name in all_results["tasks"]:
            continue
        dataset, spec = build_dataset(task_name)
        split_record = manifest["tasks"][task_name]
        if int(split_record["counts"]["train"]) + int(split_record["counts"]["validation"]) + int(split_record["counts"]["test"]) != len(dataset):
            raise ValueError(f"split manifest size mismatch for {task_name}")
        subsets = make_subsets(dataset, split_record)
        train_loader = _loader(subsets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                               seed=head_seed + task_index, pin_memory=device.type == "cuda")
        validation_loader = _loader(subsets["validation"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                    seed=head_seed, pin_memory=device.type == "cuda")
        test_loader = None
        if not args.selection_only:
            test_loader = _loader(
                subsets["test"],
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                seed=head_seed,
                pin_memory=device.type == "cuda",
            )
        encoder = _new_encoder(config, checkpoint, device)
        heads_root = run_dir / "heads" / task_name
        exit_results = {}
        validation_values = {}
        for exit_layer in EXIT_LAYERS:
            print(f"training {task_name} L{exit_layer}", flush=True)
            result = _train_exit(
                encoder,
                exit_layer=exit_layer,
                spec=spec,
                train_loader=train_loader,
                validation_loader=validation_loader,
                args=args,
                device=device,
                head_seed=head_seed + task_index * 100 + exit_layer,
                head_path=heads_root / f"L{exit_layer}.pt",
            )
            validation_values[exit_layer] = result["validation"][spec.metric_name]
            exit_results[str(exit_layer)] = result

        selected_best = select_best_exit(validation_values, higher_is_better=spec.higher_is_better)
        selected_tolerance = select_earliest_within_tolerance(
            validation_values,
            higher_is_better=spec.higher_is_better,
            tolerance=args.tolerance,
        )
        task_result = {
            "spec": asdict(spec),
            "split_counts": split_record["counts"],
            "exits": exit_results,
            "selected_exit_best_validation": selected_best,
            "selected_exit_earliest_within_tolerance": selected_tolerance,
        }
        if not args.selection_only:
            # Exit selection is complete before this loop touches the test split.
            assert test_loader is not None
            for exit_layer in EXIT_LAYERS:
                exit_results[str(exit_layer)]["test"] = _evaluate_saved_head(
                    encoder,
                    exit_layer=exit_layer,
                    spec=spec,
                    head_path=Path(exit_results[str(exit_layer)]["head_path"]),
                    test_loader=test_loader,
                    device=device,
                    limit_batches=args.limit_test_batches,
                )
            task_result["selected_test_best_validation"] = exit_results[str(selected_best)]["test"]
            task_result["selected_test_earliest_within_tolerance"] = exit_results[str(selected_tolerance)]["test"]
        all_results["tasks"][task_name] = task_result
        output.write_text(json.dumps(all_results, indent=2, sort_keys=True))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output.write_text(json.dumps(all_results, indent=2, sort_keys=True))
    print(json.dumps({"metrics": str(output), "tasks": list(tasks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
