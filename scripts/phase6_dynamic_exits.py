#!/usr/bin/env python3
"""Calibrate classification-only dynamic entropy exits after fixed exits finish."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from configs import init_jepa_config  # noqa: E402
from dynamic_exits import DynamicFrozenExits, choose_entropy_threshold, macro_accuracy, route_logits  # noqa: E402
from models import Encoder, ExitHead, load_context_encoder_checkpoint  # noqa: E402
from phase5 import build_task_dataset, load_or_create_split_manifest, task_subsets  # noqa: E402
from phase6 import (  # noqa: E402
    OOD_TASKS,
    build_ood_task_dataset,
    default_ood_root,
    load_or_create_ood_split_manifest,
    ood_task_subsets,
)


PRIMARY_CLASSIFICATION_TASKS = ("rfs", "deepmimo-beam", "deepmimo-los")
ALL_TASKS = PRIMARY_CLASSIFICATION_TASKS + OOD_TASKS
EXIT_LAYERS = (2, 4, 6, 8)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=("p0", "p1", "p2"), required=True)
    parser.add_argument("--pretrain-seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs/phase5")
    parser.add_argument("--dataset-suite", choices=("primary", "ood"), default="primary")
    parser.add_argument("--ood-root", type=Path, default=default_ood_root(REPOSITORY_ROOT))
    parser.add_argument("--task", choices=ALL_TASKS, action="append")
    parser.add_argument("--split-seed", type=int, default=2025)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def _new_encoder(config, checkpoint: Path, device: torch.device) -> Encoder:
    encoder = Encoder(
        img_size=tuple(config.encoder.img_size),
        patch_size=config.encoder.patch_size,
        embed_dim=config.encoder.embed_dim,
        depth=config.encoder.depth,
        num_heads=config.encoder.num_heads,
    ).to(device)
    load_context_encoder_checkpoint(encoder, checkpoint)
    return encoder.eval()


def _load_router(encoder: Encoder, heads_root: Path, num_outputs: int, device: torch.device) -> DynamicFrozenExits:
    heads = {}
    for layer in EXIT_LAYERS:
        path = heads_root / f"L{layer}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"fixed exit is incomplete; missing {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        head = ExitHead(encoder.pos_embed.shape[-1], num_outputs).to(device)
        head.load_state_dict(checkpoint["head"], strict=True)
        heads[layer] = head.eval()
    return DynamicFrozenExits(encoder, heads).to(device).eval()


def _collect_logits(router: DynamicFrozenExits, loader, device: torch.device) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    logits_by_layer = {layer: [] for layer in EXIT_LAYERS}
    labels = []
    with torch.inference_mode():
        for batch in loader:
            samples, targets = batch[:2]
            outputs = router.forward_all(samples.to(device=device, dtype=torch.float32, non_blocking=True))
            for layer in EXIT_LAYERS:
                logits_by_layer[layer].append(outputs[layer].cpu())
            labels.append(targets.long().cpu())
    return {layer: torch.cat(values) for layer, values in logits_by_layer.items()}, torch.cat(labels)


def _evaluate(logits: dict[int, torch.Tensor], labels: torch.Tensor, *, num_classes: int, threshold: float) -> dict[str, object]:
    routed_logits, selected_layers = route_logits(logits, threshold)
    counts = {str(layer): int((selected_layers == layer).sum().item()) for layer in EXIT_LAYERS}
    return {
        "macro_accuracy": macro_accuracy(routed_logits, labels, num_classes),
        "l8_macro_accuracy": macro_accuracy(logits[8], labels, num_classes),
        "mean_exit_layer": float(selected_layers.float().mean().item()),
        "exit_counts": counts,
        "exit_fractions": {layer: count / len(labels) for layer, count in counts.items()},
    }


def main() -> int:
    args = _arguments()
    if min(args.batch_size, args.num_workers + 1) < 1 or args.tolerance < 0:
        raise ValueError("batch size/workers must be positive and tolerance must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPOSITORY_ROOT / args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    if args.dataset_suite == "primary":
        allowed_tasks = PRIMARY_CLASSIFICATION_TASKS
        manifest = load_or_create_split_manifest(
            args.run_root / f"splits_seed_{args.split_seed}.json", REPOSITORY_ROOT, seed=args.split_seed
        )
        build_dataset = lambda task: build_task_dataset(task, REPOSITORY_ROOT)
        make_subsets = task_subsets
    else:
        allowed_tasks = OOD_TASKS
        manifest = load_or_create_ood_split_manifest(
            args.run_root / f"ood_splits_seed_{args.split_seed}.json", args.ood_root, seed=args.split_seed
        )
        build_dataset = lambda task: build_ood_task_dataset(task, args.ood_root)
        make_subsets = ood_task_subsets
    tasks = tuple(args.task or allowed_tasks)
    if unsupported := sorted(set(tasks).difference(allowed_tasks)):
        raise ValueError(f"tasks {unsupported} are not classification tasks in the {args.dataset_suite} suite")

    config = init_jepa_config(str(args.config or REPOSITORY_ROOT / "configs" / f"{args.method}.yml"))
    run_dir = args.run_root / args.method / f"seed_{args.pretrain_seed:03d}" / "downstream"
    fixed_metrics = run_dir / "exit_metrics.json"
    if not fixed_metrics.is_file():
        raise FileNotFoundError(f"fixed exits must finish before dynamic routing: {fixed_metrics}")
    fixed_result = json.loads(fixed_metrics.read_text())
    results = {"method": args.method, "pretrain_seed": args.pretrain_seed, "dataset_suite": args.dataset_suite, "tasks": {}}
    for task_name in tasks:
        if task_name not in fixed_result.get("tasks", {}):
            raise ValueError(f"fixed exits are incomplete for {task_name}")
        dataset, spec = build_dataset(task_name)
        subsets = make_subsets(dataset, manifest["tasks"][task_name])
        validation_loader = DataLoader(subsets["validation"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(subsets["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        router = _load_router(_new_encoder(config, checkpoint, device), run_dir / "heads" / task_name, spec.num_outputs, device)

        validation_logits, validation_labels = _collect_logits(router, validation_loader, device)
        selection = choose_entropy_threshold(
            validation_logits,
            validation_labels,
            num_classes=spec.num_outputs,
            tolerance=args.tolerance,
        )
        # The threshold is fixed before the test loader is read.
        test_logits, test_labels = _collect_logits(router, test_loader, device)
        results["tasks"][task_name] = {
            "spec": {"num_outputs": spec.num_outputs, "metric": spec.metric_name},
            "selection": selection,
            "test": _evaluate(test_logits, test_labels, num_classes=spec.num_outputs, threshold=selection["threshold"]),
        }
    output = run_dir / "dynamic_entropy_metrics.json"
    output.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps({"metrics": str(output), "tasks": list(tasks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
