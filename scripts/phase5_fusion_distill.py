#!/usr/bin/env python3
"""Train single-exit students from same-depth causal fusion teachers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from configs import configure_method_variant, init_jepa_config  # noqa: E402
from evaluation import select_best_exit, select_earliest_within_tolerance  # noqa: E402
from models import FrozenEncoderFusionTeacherStudent  # noqa: E402
from phase5 import PRIMARY_TASKS, build_task_dataset, task_subsets  # noqa: E402
from phase5_exits import (  # noqa: E402
    EXIT_LAYERS, _arguments, _criterion, _evaluate, _loader, _new_encoder, _seed_everything,
)
from phase5_fusion_exits import digest  # noqa: E402
from ee_progress import finish_epoch_progress, render_batch_progress  # noqa: E402


def parse_args():
    # Keep the baseline runner's training/data options and add explicit teacher/KD controls.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=("p0", "p2"), required=True)
    parser.add_argument("--pretrain-seed", type=int, default=0)
    parser.add_argument("--head-seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=2025)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/ee_optimization/multi_block_n6")
    parser.add_argument("--fusion-teacher-root", type=Path, default=ROOT / "runs/ee_optimization/multi_block_n6/fusion")
    parser.add_argument("--task", choices=PRIMARY_TASKS, action="append")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--head-early-stopping-patience", type=int, default=20)
    parser.add_argument("--head-early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--disable-head-early-stopping", action="store_true")
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-validation-batches", type=int)
    parser.add_argument("--limit-test-batches", type=int)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    return parser.parse_args()


def validate_teacher(path: Path, metrics_path: Path, *, method: str, pretrain_seed: int,
                     head_seed: int, task: str, layer: int, checkpoint_sha: str,
                     split_sha: str) -> tuple[dict, dict]:
    if not path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"required fusion teacher or metrics packet is missing: {path}, {metrics_path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    packet = json.loads(metrics_path.read_text())
    record = packet.get("tasks", {}).get(task, {}).get("exits", {}).get(str(layer))
    expected_sources = [l for l in EXIT_LAYERS if l <= layer]
    checks = {
        "head_algorithm": (packet.get("head_algorithm"), "causal_fusion"),
        "method": (packet.get("method"), method),
        "pretrain_seed": (int(packet.get("pretrain_seed", -1)), pretrain_seed),
        "head_seed": (int(packet.get("head_seed", -1)), head_seed),
        "checkpoint_sha256": (packet.get("checkpoint_sha256"), checkpoint_sha),
        "split_manifest_sha256": (packet.get("split_manifest_sha256"), split_sha),
    }
    for key, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"fusion teacher {key} mismatch: expected {expected!r}, got {actual!r}")
    if record is None or payload.get("head_algorithm") != "causal_fusion":
        raise ValueError(f"fusion teacher record is missing or incompatible for {task} L{layer}")
    if int(payload.get("exit_layer", -1)) != layer or payload.get("source_layers") != expected_sources:
        raise ValueError(f"fusion teacher exit/source-layer mismatch for {task} L{layer}")
    if Path(record["head_path"]).resolve() != path.resolve():
        raise ValueError("teacher checkpoint path does not match metrics packet")
    if int(record.get("best_epoch", -1)) != int(payload.get("epoch", -2)):
        raise ValueError("teacher best epoch does not match metrics record")
    return payload, packet


def distillation_loss(student: torch.Tensor, teacher: torch.Tensor, labels: torch.Tensor,
                      spec, criterion, *, temperature: float, weight: float) -> torch.Tensor:
    """Combine task supervision with classification-logit or regression KD."""
    if spec.target_type == "classification":
        labels = labels.long()
        task_loss = criterion(student, labels)
        kd_loss = F.kl_div(F.log_softmax(student / temperature, dim=1),
                           F.softmax(teacher / temperature, dim=1), reduction="batchmean") * temperature ** 2
    else:
        labels = labels.float()
        task_loss = F.mse_loss(student, labels)
        kd_loss = F.mse_loss(student, teacher)
    return (1 - weight) * task_loss + weight * kd_loss


def main() -> int:
    args = parse_args()
    if args.split_seed != 2025 or args.kd_temperature <= 0 or not 0 <= args.kd_weight <= 1:
        raise ValueError("split seed must be 2025; KD temperature must be positive and weight in [0,1]")
    if args.selection_only and args.test_only:
        raise ValueError("--selection-only and --test-only are mutually exclusive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    method, seed = args.method, args.head_seed
    run_dir = args.run_root / "fusion_kd" / f"head_seed_{seed:03d}" / method / f"seed_{args.pretrain_seed:03d}" / "downstream"
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "exit_metrics.json"
    manifest_path = ROOT / "runs/masking_head_seeds/multi_block_n6/head_seed_000/splits_seed_2025_primary_v3.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"locked Current EE split manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("seed", -1)) != 2025:
        raise ValueError(f"locked split has the wrong seed: {manifest_path}")
    config = init_jepa_config(str(args.config or ROOT / "configs" / f"{method}.yml"))
    configure_method_variant(config, method)
    tasks = tuple(args.task or PRIMARY_TASKS)
    packet = {"head_algorithm": "fusion_teacher_kd", "method": method,
              "pretrain_seed": args.pretrain_seed, "head_seed": seed,
              "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest(checkpoint),
              "split_manifest": str(manifest_path.resolve()), "split_manifest_sha256": digest(manifest_path),
              "split_seed": 2025, "protocol": {"head_max_epochs": args.epochs,
              "batch_size": args.batch_size, "learning_rate": args.learning_rate,
              "weight_decay": args.weight_decay,
              "head_early_stopping": {"enabled": not args.disable_head_early_stopping,
                  "patience": args.head_early_stopping_patience,
                  "min_delta": args.head_early_stopping_min_delta},
              "kd_temperature": args.kd_temperature, "kd_weight": args.kd_weight,
              "tolerance": args.tolerance,
              "selection_uses": "validation_only", "test_evaluation": not args.selection_only}, "tasks": {}}
    if args.test_only:
        packet = json.loads(output.read_text())
        if (packet.get("head_algorithm") != "fusion_teacher_kd" or packet.get("checkpoint_sha256") != digest(checkpoint)
                or int(packet.get("head_seed", -1)) != seed or int(packet.get("pretrain_seed", -1)) != args.pretrain_seed
                or packet.get("split_manifest_sha256") != digest(manifest_path)):
            raise ValueError("saved KD selection packet is incompatible")
    elif args.resume and output.is_file():
        old = json.loads(output.read_text())
        if old.get("head_algorithm") != "fusion_teacher_kd" or old.get("checkpoint_sha256") != packet["checkpoint_sha256"] or old.get("protocol", {}) != packet["protocol"]:
            raise ValueError("cannot resume an incompatible KD packet")
        packet["tasks"] = old.get("tasks", {})

    for task_index, task_name in enumerate(tasks):
        if task_name in packet["tasks"] and not args.test_only:
            continue
        dataset, spec = build_task_dataset(task_name, ROOT)
        subsets = task_subsets(dataset, manifest["tasks"][task_name])
        train_loader = _loader(subsets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                               seed=seed + task_index, pin_memory=device.type == "cuda")
        val_loader = _loader(subsets["validation"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             seed=seed, pin_memory=device.type == "cuda")
        test_loader = None if args.selection_only else _loader(subsets["test"], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, seed=seed, pin_memory=device.type == "cuda")
        encoder = _new_encoder(config, checkpoint, device)
        result = packet["tasks"].setdefault(task_name, {"exits": {}})
        result["spec"] = spec.__dict__
        values = {}
        stage_models = {}
        criterion = _criterion(spec, device)
        for layer in EXIT_LAYERS:
            teacher_path = args.fusion_teacher_root / f"head_seed_{seed:03d}" / method / f"seed_{args.pretrain_seed:03d}" / "downstream" / "heads" / task_name / f"L{layer}.pt"
            teacher_metrics = teacher_path.parents[2] / "exit_metrics.json"
            teacher, teacher_packet = validate_teacher(teacher_path, teacher_metrics, method=method,
                pretrain_seed=args.pretrain_seed, head_seed=seed, task=task_name, layer=layer,
                checkpoint_sha=packet["checkpoint_sha256"], split_sha=packet["split_manifest_sha256"])
            _seed_everything(seed + task_index * 100 + layer)
            model = FrozenEncoderFusionTeacherStudent(encoder, layer, spec.num_outputs, teacher["head"]).to(device)
            optimizer = torch.optim.AdamW(model.student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            best = -math.inf if spec.higher_is_better else math.inf
            stale, best_epoch = 0, None
            head_path = run_dir / "heads" / task_name / f"L{layer}.pt"
            if args.test_only:
                saved = result["exits"][str(layer)]
                head_path = Path(saved["head_path"])
                student_state = torch.load(head_path, map_location=device, weights_only=False)["head"]
                model.student.load_state_dict(student_state, strict=True)
                validation = saved["validation"]
            else:
                total_batches = len(train_loader)
                if args.limit_train_batches is not None:
                    total_batches = min(total_batches, args.limit_train_batches)
                progress_label = f"Fusion-KD {task_name} L{layer}"
                for epoch in range(args.epochs):
                    model.train()
                    for batch_index, batch in enumerate(train_loader):
                        if args.limit_train_batches is not None and batch_index >= args.limit_train_batches:
                            break
                        x, y = batch[:2]
                        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
                        y = y.to(device=device, non_blocking=True)
                        student, teacher_logits = model(x)
                        loss = distillation_loss(student, teacher_logits, y, spec, criterion,
                            temperature=args.kd_temperature, weight=args.kd_weight)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if any(p.grad is not None for p in encoder.parameters()) or any(p.grad is not None for p in model.teacher.parameters()):
                            raise RuntimeError("frozen encoder or teacher received gradients")
                        optimizer.step()
                        render_batch_progress(progress_label, epoch + 1, args.epochs, batch_index, total_batches)
                    class StudentView:
                        def eval(self): model.eval(); return self
                        def __call__(self, x): return model(x)[0]
                    validation = _evaluate(StudentView(), val_loader, spec, criterion, device,
                                           limit_batches=args.limit_validation_batches)
                    value = float(validation[spec.metric_name])
                    finish_epoch_progress(progress_label, epoch + 1, args.epochs, spec.metric_name, value)
                    improved = value > best + args.head_early_stopping_min_delta if spec.higher_is_better else value < best - args.head_early_stopping_min_delta
                    if improved:
                        best, best_epoch, stale = value, epoch + 1, 0
                        head_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({"head_algorithm": "fusion_teacher_kd", "exit_layer": layer,
                            "head": model.student.state_dict(), "epoch": best_epoch, "validation": validation,
                            "teacher_path": str(teacher_path), "teacher_sha256": digest(teacher_path)}, head_path)
                    else:
                        stale += 1
                    if not args.disable_head_early_stopping and stale >= args.head_early_stopping_patience:
                        break
                if best_epoch is None:
                    raise RuntimeError(f"no finite validation metric for {task_name} L{layer}")
                validation = torch.load(head_path, map_location=device, weights_only=False)["validation"]
            values[layer] = validation[spec.metric_name]
            stage_models[layer] = model
            record = {"best_epoch": best_epoch or torch.load(head_path, map_location="cpu", weights_only=False)["epoch"],
                      "validation": validation, "head_path": str(head_path), "teacher_path": str(teacher_path),
                      "teacher_sha256": digest(teacher_path), "teacher_metrics_path": str(teacher_metrics),
                      "teacher_metrics_sha256": digest(teacher_metrics)}
            result["exits"][str(layer)] = record
        result["selected_exit_best_validation"] = select_best_exit(values, higher_is_better=spec.higher_is_better)
        result["selected_exit_earliest_within_tolerance"] = select_earliest_within_tolerance(
            values, higher_is_better=spec.higher_is_better, tolerance=args.tolerance)
        if test_loader is not None:
            # Freeze the validation-based choice before any held-out evaluation.
            for layer in EXIT_LAYERS:
                model = stage_models[layer]
                class StudentView:
                    def eval(self): model.eval(); return self
                    def __call__(self, x): return model(x)[0]
                result["exits"][str(layer)]["test"] = _evaluate(StudentView(), test_loader, spec, criterion, device,
                    limit_batches=args.limit_test_batches)
            result["selected_test_best_validation"] = result["exits"][str(result["selected_exit_best_validation"])]["test"]
        packet["protocol"]["test_evaluation"] = test_loader is not None
        output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metrics": str(output), "tasks": list(tasks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
