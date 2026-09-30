#!/usr/bin/env python3
"""Run one reproducible Phase 5 P0, P1, P2, or P2-Reverse pretraining job.

Formal jobs use a fixed 60-epoch Phase 5 pilot horizon and the complete
four-cache pretraining set. The pinned LatentWave 240-epoch source schedule is
retained as a separate optional reproduction protocol. Latent-loss early stopping is
retained only as an explicitly labelled diagnostic because an EMA target and
stochastic masks make that loss unsuitable for formal model selection.

The explicit ``--throughput-safe`` profile is a separate FP32 speed variant
for hierarchical jobs: microbatch 8, accumulation 8, effective batch 64, and
at least eight data-loader workers. It is intentionally distinct from the
conservative ``--memory-safe`` profile.

The explicit ``--performance-safe`` profile adds BF16 mixed precision, fused
attention, bounded prefetching, and epoch-only telemetry. It is recorded as a precision variant
and must be matched across methods for a final comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, RandomSampler, Subset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
os.environ.setdefault("WANDB_KEY", "offline")

from checkpointing import build_checkpoint_metadata  # noqa: E402
from collators import get_collator  # noqa: E402
from configs import init_jepa_config  # noqa: E402
from datasets import DeepMIMO, ImageDataset, PretrainCache  # noqa: E402
from evaluation import evaluate_class, evaluate_pos  # noqa: E402
from masking_profiles import PROFILE_NAMES, apply_mask_profile  # noqa: E402
from models import JEPA  # noqa: E402
from models.components import set_attention_backend  # noqa: E402


METHOD_CONFIGS = {
    "p0": REPOSITORY_ROOT / "configs/p0.yml",
    "p1": REPOSITORY_ROOT / "configs/p1.yml",
    "p2": REPOSITORY_ROOT / "configs/p2.yml",
    "p2_reverse": REPOSITORY_ROOT / "configs/p2_reverse.yml",
}
FORMAL_EPOCHS = 60


def _allocator_configuration() -> str | None:
    """Return the allocator setting passed before this Python process started."""

    return os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")


def _uses_expandable_segments(configuration: str | None) -> bool:
    if not configuration:
        return False
    return any(
        item.strip().lower() == "expandable_segments:true"
        for item in configuration.split(",")
    )


def _limit(value: str) -> int | float:
    if value.isdigit():
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("integer batch limits must be positive")
        return parsed
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("limit must be a positive integer or a fraction in (0, 1]") from error
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("fractional limits must be in (0, 1]")
    return parsed


def _limited_length(total: int, limit: int | float) -> int:
    if isinstance(limit, int):
        return min(total, limit)
    return max(1, math.ceil(total * limit))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHOD_CONFIGS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs/phase5")
    parser.add_argument("--run-name", type=str, default=None, help="Optional result-directory name, for ablations.")
    parser.add_argument(
        "--mask-profile",
        choices=PROFILE_NAMES,
        default=None,
        help="Apply one registered masking-study profile and record its checksum in run metadata.",
    )
    parser.add_argument(
        "--run-kind",
        choices=("formal", "diagnostic"),
        default="formal",
        help=(
            "formal uses the fixed 60-epoch Phase 5 pilot schedule; diagnostic permits "
            "short runs and latent-loss early stopping that cannot be reported as a main result"
        ),
    )
    parser.add_argument("--hierarchy-lambda", type=float, default=None, help="Override lambda_H for P1/P2 ablations.")
    parser.add_argument("--include-l8-coarse", action="store_true", help="Enable the opt-in P2 L8 whole-block auxiliary loss.")
    parser.add_argument("--max-epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulate-grad-batches", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--hdf5-worker-budget",
        type=int,
        default=None,
        help=(
            "Optional total worker budget shared across the four pretraining HDF5 caches. "
            "When set, it is divided deterministically across cache loaders instead of "
            "using --num-workers for every cache loader."
        ),
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches queued in each worker when num-workers is positive.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=1,
        help="Lightning logger interval; epoch summaries remain printed independently.",
    )
    parser.add_argument(
        "--epoch-metrics-only",
        action="store_true",
        help=(
            "Aggregate training telemetry by epoch without emitting per-step CSV/logger records. "
            "Terminal summaries remain one line per epoch."
        ),
    )
    parser.add_argument(
        "--progress-every-n-batches",
        type=int,
        default=25,
        help=(
            "Print the exact completed-batch count to the terminal at this interval. "
            "This does not create per-step CSV metrics."
        ),
    )
    parser.add_argument(
        "--attention-backend",
        choices=("math", "fused"),
        default="math",
        help=(
            "Attention implementation: math preserves the pinned path; fused uses PyTorch "
            "scaled-dot-product attention when attention maps are not requested."
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument(
        "--validate-every",
        type=int,
        default=0,
        help=(
            "Epoch interval for the legacy full downstream probe; 0 (the formal "
            "default) keeps held-out downstream records untouched before exit evaluation."
        ),
    )
    parser.add_argument("--limit-train-batches", type=_limit, default=1.0)
    parser.add_argument("--limit-val-batches", type=_limit, default=1.0)
    parser.add_argument("--precision", choices=("32-true", "16-mixed", "bf16-mixed"), default="32-true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--memory-safe",
        action="store_true",
        help=(
            "Require the P1/P2 RTX 5070 Ti profile: batch size 4, accumulation 16, "
            "FP32 CUDA, effective batch 64, and expandable allocator segments."
        ),
    )
    parser.add_argument(
        "--throughput-safe",
        action="store_true",
        help=(
            "Use the separately documented throughput profile: CUDA FP32, "
            "microbatch 8, accumulation 8, effective batch 64, and "
            "expandable allocator segments."
        ),
    )
    parser.add_argument(
        "--performance-safe",
        action="store_true",
        help=(
            "Use the bounded RTX 5070 Ti performance profile: BF16, microbatch 16, "
            "accumulation 4, fused attention, three workers per cache (12 total), "
            "prefetch 4, and epoch-only metrics."
        ),
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Diagnostic only: stop on held-out latent loss rather than completing the formal schedule.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-seed", type=int, default=2025)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1.0e-4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    L.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _allow_trusted_checkpoint_metadata_for_resume() -> None:
    """Allow PyTorch 2.6+ to restore our own Lightning checkpoint metadata.

    Lightning loads resume checkpoints with ``weights_only=True`` by default
    under current PyTorch releases.  The local checkpoints record
    ``TorchVersion`` in metadata, which is safe and required to restore the
    optimizer/scheduler/EMA state of a checkpoint produced by this project.
    """

    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])


def _loader_options(config, *, num_workers: int, prefetch_factor: int) -> dict:
    """Return worker settings without passing prefetching to a zero-worker loader."""

    options = {
        "num_workers": num_workers,
        "pin_memory": config.data.pin_mem,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        options["prefetch_factor"] = prefetch_factor
    return options


def _worker_allocation(*, cache_count: int, workers_per_cache: int, worker_budget: int | None) -> tuple[int, ...]:
    """Return deterministic per-cache worker counts.

    The legacy behavior gives every cache loader ``workers_per_cache`` workers.
    A total budget caps the aggregate process count while retaining one loader
    per cache and deterministic sample order.
    """

    if cache_count < 1:
        raise ValueError("cache_count must be positive")
    if workers_per_cache < 0:
        raise ValueError("workers_per_cache must be non-negative")
    if worker_budget is None:
        return (workers_per_cache,) * cache_count
    if worker_budget < 1:
        raise ValueError("hdf5 worker budget must be positive")
    quotient, remainder = divmod(worker_budget, cache_count)
    return tuple(quotient + (index < remainder) for index in range(cache_count))


def _build_pretrain_loader(config, *, prefetch_factor: int, worker_budget: int | None = None):
    collator = get_collator(config.masks.type)(
        input_size=config.encoder.img_size,
        patch_size=config.encoder.patch_size,
        **getattr(config.masks, config.masks.type).model_dump(),
    )
    dataset_configs = tuple(config.data.pretrain_datasets)
    worker_counts = _worker_allocation(
        cache_count=len(dataset_configs),
        workers_per_cache=config.data.num_workers,
        worker_budget=worker_budget,
    )
    dataloaders = {}
    for dataset_config, num_workers in zip(dataset_configs, worker_counts, strict=True):
        dataset = PretrainCache(dataset_config.path, dataset_config.key)
        dataloaders[dataset_config.name] = DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            sampler=RandomSampler(dataset),
            collate_fn=collator,
            **_loader_options(config, num_workers=num_workers, prefetch_factor=prefetch_factor),
        )
    loader = CombinedLoader(dataloaders, mode="max_size_cycle")
    iter(loader)
    return loader


def _split_pretrain_indices(size: int, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    validation_count = max(1, math.ceil(size * validation_fraction))
    if validation_count >= size:
        raise ValueError("validation split must leave at least one training sample per cache")
    indices = np.arange(size, dtype=np.int64)
    np.random.default_rng(seed).shuffle(indices)
    return np.sort(indices[validation_count:]), np.sort(indices[:validation_count])


def _build_pretrain_loaders_with_validation(
    config,
    *,
    validation_fraction: float,
    validation_seed: int,
    manifest_path: Path,
    prefetch_factor: int,
    worker_budget: int | None = None,
):
    """Build disjoint cache loaders and persist their deterministic split manifest."""
    train_collator = get_collator(config.masks.type)(
        input_size=config.encoder.img_size,
        patch_size=config.encoder.patch_size,
        **getattr(config.masks, config.masks.type).model_dump(),
    )
    train_dataloaders = {}
    validation_dataloaders = []
    manifest = {
        "schema_version": 1,
        "validation_fraction": validation_fraction,
        "validation_seed": validation_seed,
        "datasets": [],
    }
    dataset_configs = tuple(config.data.pretrain_datasets)
    worker_counts = _worker_allocation(
        cache_count=len(dataset_configs),
        workers_per_cache=config.data.num_workers,
        worker_budget=worker_budget,
    )
    for dataset_index, (dataset_config, num_workers) in enumerate(zip(dataset_configs, worker_counts, strict=True)):
        dataset = PretrainCache(dataset_config.path, dataset_config.key)
        train_indices, validation_indices = _split_pretrain_indices(
            len(dataset), validation_fraction, validation_seed + dataset_index
        )
        train_dataset = Subset(dataset, train_indices.tolist())
        validation_dataset = Subset(dataset, validation_indices.tolist())
        train_dataloaders[dataset_config.name] = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            sampler=RandomSampler(train_dataset),
            collate_fn=train_collator,
            **_loader_options(config, num_workers=num_workers, prefetch_factor=prefetch_factor),
        )
        validation_collator = get_collator(config.masks.type)(
            input_size=config.encoder.img_size,
            patch_size=config.encoder.patch_size,
            **getattr(config.masks, config.masks.type).model_dump(),
        )
        validation_dataloaders.append(
            DataLoader(
                validation_dataset,
                batch_size=config.data.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=config.data.pin_mem,
                collate_fn=validation_collator,
            )
        )
        manifest["datasets"].append(
            {
                "name": dataset_config.name,
                "path": dataset_config.path,
                "key": dataset_config.key,
                "count": len(dataset),
                "train_count": len(train_dataset),
                "validation_count": len(validation_dataset),
                "train_indices": train_indices.tolist(),
                "validation_indices": validation_indices.tolist(),
            }
        )

    manifest_path.write_text(json.dumps(manifest, indent=2))
    train_loader = CombinedLoader(train_dataloaders, mode="max_size_cycle")
    iter(train_loader)
    names = tuple(dataset_config.name for dataset_config in config.data.pretrain_datasets)
    return train_loader, validation_dataloaders, names, manifest


def _build_validation_loaders(config):
    rfs = ImageDataset(config.data.rfs_path)
    position = ImageDataset(config.data.pos_path, sample_key="features", label_dtype=torch.float32)
    beam = DeepMIMO(config.data.beam_path, label_key="label_beam", n_beams=16)
    return [
        DataLoader(dataset, batch_size=config.data.batch_size, num_workers=config.data.num_workers,
                   pin_memory=config.data.pin_mem, shuffle=False, persistent_workers=config.data.num_workers > 0)
        for dataset in (rfs, position, beam)
    ]


def _run_directory(args: argparse.Namespace) -> Path:
    name = args.run_name or args.method
    if Path(name).name != name:
        raise ValueError("run-name must be a single directory name")
    return args.run_root / name / f"seed_{args.seed:03d}"


def _validate_run_arguments(args: argparse.Namespace) -> None:
    """Reject combinations that would silently change the formal protocol."""

    if min(
        args.max_epochs,
        args.batch_size,
        args.accumulate_grad_batches,
        args.num_workers + 1,
        args.prefetch_factor,
        args.log_every_n_steps,
        args.progress_every_n_batches,
        args.checkpoint_every,
    ) < 1:
        raise ValueError(
            "epochs, batch size, accumulation, workers, prefetch factor, logging interval, progress interval, "
            "and checkpoint interval must be positive"
        )
    if args.hdf5_worker_budget is not None and args.hdf5_worker_budget < 1:
        raise ValueError("hdf5 worker budget must be positive")
    if args.validate_every < 0:
        raise ValueError("validate-every must be non-negative")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be in (0, 1)")
    if args.validation_seed < 0:
        raise ValueError("validation-seed must be non-negative")
    if args.early_stopping_patience < 1:
        raise ValueError("early-stopping-patience must be positive")
    if args.early_stopping_min_delta < 0:
        raise ValueError("early-stopping-min-delta must be non-negative")
    if args.early_stopping and args.validate_every:
        raise ValueError("--validate-every downstream probing cannot be combined with latent early stopping")
    if args.hierarchy_lambda is not None and (args.method == "p0" or args.hierarchy_lambda < 0):
        raise ValueError("hierarchy-lambda must be non-negative and is only valid for hierarchical methods")
    if args.include_l8_coarse and args.method != "p2":
        raise ValueError("include-l8-coarse is only valid for the P2 ablation")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    if args.run_kind == "formal":
        if args.early_stopping:
            raise ValueError("formal jobs cannot use latent-loss early stopping; use --run-kind diagnostic")
        if args.max_epochs != FORMAL_EPOCHS:
            raise ValueError(f"formal jobs require --max-epochs {FORMAL_EPOCHS}, got {args.max_epochs}")
        if args.limit_train_batches != 1.0 or args.limit_val_batches != 1.0:
            raise ValueError("formal jobs require full train and validation batch limits")
        if args.validate_every:
            raise ValueError("formal jobs do not run the legacy downstream probe during pretraining")
    elif args.early_stopping is False and args.validate_every:
        # The legacy downstream probe is neither a source-style reproducibility
        # check nor the fixed-exit protocol.  Keep it out of this runner.
        raise ValueError("legacy downstream probing is disabled; use the dedicated probe or exit runner")

    if args.early_stopping and args.run_kind != "diagnostic":
        raise ValueError("latent-loss early stopping is diagnostic-only")
    if args.memory_safe:
        if args.throughput_safe or args.performance_safe:
            raise ValueError("--memory-safe cannot be combined with another execution profile")
        if args.method not in {"p1", "p2", "p2_reverse"}:
            raise ValueError("--memory-safe is only defined for hierarchical jobs")
        if args.device != "cuda" or args.batch_size != 4 or args.accumulate_grad_batches != 16:
            raise ValueError("--memory-safe requires --device cuda --batch-size 4 --accumulate-grad-batches 16")
        if args.precision != "32-true":
            raise ValueError("--memory-safe requires --precision 32-true")
        if not _uses_expandable_segments(_allocator_configuration()):
            raise ValueError(
                "--memory-safe requires PYTORCH_ALLOC_CONF=expandable_segments:True before Python starts"
            )
    if args.throughput_safe:
        if args.performance_safe:
            raise ValueError("--throughput-safe and --performance-safe are mutually exclusive")
        if args.method not in {"p1", "p2", "p2_reverse"}:
            raise ValueError("--throughput-safe is only defined for hierarchical jobs")
        if args.device != "cuda" or args.batch_size != 8 or args.accumulate_grad_batches != 8:
            raise ValueError(
                "--throughput-safe requires --device cuda --batch-size 8 "
                "--accumulate-grad-batches 8"
            )
        if args.precision != "32-true":
            raise ValueError("--throughput-safe requires --precision 32-true")
        if not _uses_expandable_segments(_allocator_configuration()):
            raise ValueError(
                "--throughput-safe requires PYTORCH_ALLOC_CONF=expandable_segments:True "
                "before Python starts"
            )
        if args.num_workers < 8:
            raise ValueError(
                "--throughput-safe requires at least 8 data-loader workers; "
                "reduce workers only in a separately documented contention test"
            )
    if args.performance_safe:
        if args.method not in {"p1", "p2", "p2_reverse"}:
            raise ValueError("--performance-safe is only defined for hierarchical jobs")
        if args.device != "cuda" or args.batch_size != 16 or args.accumulate_grad_batches != 4:
            raise ValueError(
                "--performance-safe requires --device cuda --batch-size 16 "
                "--accumulate-grad-batches 4"
            )
        if args.precision != "bf16-mixed":
            raise ValueError("--performance-safe requires --precision bf16-mixed")
        if args.num_workers != 3 or args.prefetch_factor != 4:
            raise ValueError("--performance-safe requires --num-workers 3 --prefetch-factor 4")
        if args.hdf5_worker_budget != 12:
            raise ValueError("--performance-safe requires --hdf5-worker-budget 12")
        if not args.epoch_metrics_only:
            raise ValueError("--performance-safe requires --epoch-metrics-only")
        if args.attention_backend != "fused":
            raise ValueError("--performance-safe requires --attention-backend fused")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("--performance-safe requires CUDA BF16 support")
        if not _uses_expandable_segments(_allocator_configuration()):
            raise ValueError(
                "--performance-safe requires PYTORCH_ALLOC_CONF=expandable_segments:True "
                "before Python starts"
            )


def main() -> int:
    args = _arguments()
    _validate_run_arguments(args)

    run_dir = _run_directory(args)
    if run_dir.exists() and any(run_dir.iterdir()) and args.resume is None:
        raise FileExistsError(f"run directory already contains artifacts: {run_dir}; use --resume to continue")
    run_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)
    # This is the same matmul policy used by the pinned pretraining entrypoint.
    torch.set_float32_matmul_precision("high")

    config_path = METHOD_CONFIGS[args.method]
    config = init_jepa_config(str(config_path))
    mask_profile = apply_mask_profile(config, args.mask_profile) if args.mask_profile else None
    config.seed = args.seed
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    config.optimization.epochs = args.max_epochs
    if args.hierarchy_lambda is not None:
        config.hierarchy.lambda_h = args.hierarchy_lambda
    if args.include_l8_coarse:
        config.hierarchy.include_l8_coarse = True
    config.checkpoint.dir = str(run_dir / "checkpoints")
    config.checkpoint.every_n_epochs = args.checkpoint_every
    cache_count = len(config.data.pretrain_datasets)
    worker_allocation = _worker_allocation(
        cache_count=cache_count,
        workers_per_cache=args.num_workers,
        worker_budget=args.hdf5_worker_budget,
    )

    validation_dataset_names = None
    validation_manifest = None
    if args.early_stopping:
        train_loader, validation_loaders, validation_dataset_names, validation_manifest = (
            _build_pretrain_loaders_with_validation(
                config,
                validation_fraction=args.validation_fraction,
                validation_seed=args.validation_seed,
                manifest_path=run_dir / "pretrain_split.json",
                prefetch_factor=args.prefetch_factor,
                worker_budget=args.hdf5_worker_budget,
            )
        )
    else:
        train_loader = _build_pretrain_loader(
            config,
            prefetch_factor=args.prefetch_factor,
            worker_budget=args.hdf5_worker_budget,
        )
        validation_loaders = None
    microbatches_per_epoch = _limited_length(len(train_loader), args.limit_train_batches)
    optimizer_steps_per_epoch = math.ceil(microbatches_per_epoch / args.accumulate_grad_batches)
    runtime = {
        "phase": 5,
        "method": args.method,
        "run_name": args.run_name or args.method,
        "run_kind": args.run_kind,
        "pretraining_selection": (
            "fixed_source_horizon_last_checkpoint"
            if args.run_kind == "formal"
            else ("diagnostic_latent_validation_early_stopping" if args.early_stopping else "diagnostic_fixed_horizon")
        ),
        "seed": args.seed,
        "config_path": str(config_path.relative_to(REPOSITORY_ROOT)),
        "microbatch_size": args.batch_size,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "effective_batch_size": args.batch_size * args.accumulate_grad_batches,
        "microbatches_per_epoch": microbatches_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "max_epochs": args.max_epochs,
        "precision": args.precision,
        "attention_backend": args.attention_backend,
        "epoch_metrics_only": args.epoch_metrics_only,
        "cuda_allocator_conf": _allocator_configuration(),
        "memory_safe": args.memory_safe,
        "throughput_safe": args.throughput_safe,
        "performance_safe": args.performance_safe,
        "data_loader_workers": args.num_workers,
        "hdf5_worker_budget": args.hdf5_worker_budget,
        "data_loader_workers_per_cache": list(worker_allocation),
        "total_data_loader_workers": sum(worker_allocation),
        "data_loader_prefetch_factor": args.prefetch_factor,
        "log_every_n_steps": args.log_every_n_steps,
        "progress_every_n_batches": args.progress_every_n_batches,
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        "validate_every": args.validate_every,
        "hierarchy_lambda_override": args.hierarchy_lambda,
        "include_l8_coarse": args.include_l8_coarse,
        "early_stopping": args.early_stopping,
        "validation_fraction": args.validation_fraction if args.early_stopping else None,
        "validation_seed": args.validation_seed if args.early_stopping else None,
        "early_stopping_patience": args.early_stopping_patience if args.early_stopping else None,
        "early_stopping_min_delta": args.early_stopping_min_delta if args.early_stopping else None,
        "mask_profile": mask_profile,
    }
    metadata = build_checkpoint_metadata(config, seed=args.seed, config_path=config_path)
    metadata["phase5"] = runtime
    metadata["mask_profile"] = mask_profile
    (run_dir / "experiment.json").write_text(json.dumps({"metadata": metadata, "runtime": runtime}, indent=2))

    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "runtime": runtime, "dry_run": True}, indent=2))
        return 0

    if args.validate_every:
        validation_loaders = _build_validation_loaders(config)
    model = JEPA(
        config,
        iterations_per_epoch=optimizer_steps_per_epoch,
        eval_fns=[evaluate_class, evaluate_pos, evaluate_class],
        metrics=["rfs_pca", "pos_error", "beam_pca"],
    )
    model.configure_training_metrics(epoch_only=args.epoch_metrics_only)
    set_attention_backend(model, args.attention_backend)
    model.checkpoint_metadata = metadata
    if args.early_stopping:
        model.enable_pretrain_validation(validation_dataset_names)

    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=config.checkpoint.dir,
        filename="epoch={epoch:03d}-step={step}",
        every_n_epochs=args.checkpoint_every,
        save_top_k=-1,
        save_last=True,
    )
    class BatchProgressPrinter(callbacks.Callback):
        """Render completed microbatches as one updating terminal progress bar."""

        def __init__(self, total_batches: int, interval: int):
            super().__init__()
            self.total_batches = int(total_batches)
            self.interval = int(interval)
            self._rendered = False

        def _render(self, trainer, completed: int) -> None:
            width = 30
            fraction = min(1.0, completed / self.total_batches)
            filled = int(round(width * fraction))
            bar = "█" * filled + "░" * (width - filled)
            sys.stdout.write(
                f"\r[progress epoch {trainer.current_epoch + 1:03d}] "
                f"[{bar}] {completed:03d}/{self.total_batches:03d} batches "
                f"optimizer_step={trainer.global_step}"
            )
            sys.stdout.flush()
            self._rendered = True

        def on_train_epoch_start(self, trainer, pl_module):
            self._rendered = False
            self._render(trainer, 0)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            completed = int(batch_idx) + 1
            if completed % self.interval and completed != self.total_batches:
                return
            self._render(trainer, completed)

        def on_train_epoch_end(self, trainer, pl_module):
            if self._rendered:
                sys.stdout.write("\n")
                sys.stdout.flush()

    class EpochLossPrinter(callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._sums: dict[str, float] = {}
            self._counts: dict[str, int] = {}

        @staticmethod
        def _value(trainer, *keys: str) -> float | None:
            for key in keys:
                value = trainer.callback_metrics.get(key)
                if value is None:
                    continue
                return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            return None

        def _add(self, name: str, value: float | None) -> None:
            if value is None:
                return
            self._sums[name] = self._sums.get(name, 0.0) + value
            self._counts[name] = self._counts.get(name, 0) + 1

        def _mean(self, name: str) -> float | None:
            count = self._counts.get(name, 0)
            return self._sums[name] / count if count else None

        def _record_last_batch(self, trainer, pl_module) -> None:
            # Callback hooks run before Lightning publishes a batch's metrics.
            # At the next batch start (or at epoch end for the final batch),
            # callback_metrics therefore contains the previous microbatch's
            # on-step losses.  Accumulating them here gives a stable terminal
            # epoch summary without relying on CSV column positions.
            self._add("train_loss", self._value(trainer, "train_loss_step", "train_loss"))
            if not pl_module.hierarchy_enabled:
                return
            self._add("hierarchy_weight", self._value(trainer, "hierarchy_weight_step", "hierarchy_weight"))
            anchor_values = []
            auxiliary_values = {layer: [] for layer in pl_module.auxiliary_layers}
            for dataset_config in config.data.pretrain_datasets:
                anchor = self._value(
                    trainer,
                    f"train_loss_{dataset_config.name}_anchor_step",
                    f"train_loss_{dataset_config.name}_anchor",
                )
                if anchor is not None:
                    anchor_values.append(anchor)
                for layer in pl_module.auxiliary_layers:
                    auxiliary = self._value(
                        trainer,
                        f"train_loss_{dataset_config.name}_L{layer}_step",
                        f"train_loss_{dataset_config.name}_L{layer}",
                    )
                    if auxiliary is not None:
                        auxiliary_values[layer].append(auxiliary)
            self._add("anchor_loss", sum(anchor_values) / len(anchor_values) if anchor_values else None)
            for layer, layer_values in auxiliary_values.items():
                self._add(f"aux_L{layer}", sum(layer_values) / len(layer_values) if layer_values else None)

        def on_train_epoch_start(self, trainer, pl_module):
            self._sums = {}
            self._counts = {}

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if pl_module.epoch_metrics_only:
                return
            if batch_idx:
                self._record_last_batch(trainer, pl_module)

        def _training_fields(self, trainer, pl_module) -> list[str]:
            train_loss = self._mean("train_loss")
            if train_loss is None:
                train_loss = self._value(trainer, "train_loss_epoch", "train_loss")
            if train_loss is None:
                return []
            fields = [f"train_loss={train_loss:.6f}"]
            if pl_module.hierarchy_enabled:
                hierarchy_weight = self._mean("hierarchy_weight")
                if hierarchy_weight is None:
                    hierarchy_weight = self._value(trainer, "hierarchy_weight_epoch", "hierarchy_weight")
                if hierarchy_weight is not None:
                    fields.append(f"hierarchy_weight={hierarchy_weight:.6f}")
                anchor_loss = self._mean("anchor_loss")
                if anchor_loss is None:
                    anchor_values = [
                        self._value(
                            trainer,
                            f"train_loss_{dataset_config.name}_anchor_epoch",
                            f"train_loss_{dataset_config.name}_anchor",
                        )
                        for dataset_config in config.data.pretrain_datasets
                    ]
                    anchor_values = [value for value in anchor_values if value is not None]
                    anchor_loss = sum(anchor_values) / len(anchor_values) if anchor_values else None
                if anchor_loss is not None:
                    fields.append(f"anchor_loss={anchor_loss:.6f}")
                for layer in pl_module.auxiliary_layers:
                    auxiliary_loss = self._mean(f"aux_L{layer}")
                    if auxiliary_loss is None:
                        auxiliary_values = [
                            self._value(
                                trainer,
                                f"train_loss_{dataset_config.name}_L{layer}_epoch",
                                f"train_loss_{dataset_config.name}_L{layer}",
                            )
                            for dataset_config in config.data.pretrain_datasets
                        ]
                        auxiliary_values = [value for value in auxiliary_values if value is not None]
                        auxiliary_loss = (
                            sum(auxiliary_values) / len(auxiliary_values) if auxiliary_values else None
                        )
                    if auxiliary_loss is not None:
                        fields.append(f"aux_L{layer}={auxiliary_loss:.6f}")
            return fields

        def on_train_epoch_end(self, trainer, pl_module):
            if not pl_module.epoch_metrics_only:
                self._record_last_batch(trainer, pl_module)
            if args.early_stopping:
                return
            fields = self._training_fields(trainer, pl_module)
            if fields:
                print(f"[epoch {trainer.current_epoch + 1:03d}] {' '.join(fields)}", flush=True)

        def on_validation_end(self, trainer, pl_module):
            if not args.early_stopping or trainer.sanity_checking:
                return
            fields = self._training_fields(trainer, pl_module)
            validation_loss = self._value(trainer, "val_loss")
            if not fields or validation_loss is None:
                return
            fields.append(f"val_loss={validation_loss:.6f}")
            print(f"[epoch {trainer.current_epoch + 1:03d}] {' '.join(fields)}", flush=True)

    callbacks_list = [
        checkpoint_callback,
        callbacks.LearningRateMonitor(logging_interval="step"),
        BatchProgressPrinter(microbatches_per_epoch, args.progress_every_n_batches),
        EpochLossPrinter(),
    ]
    best_checkpoint_callback = None
    if args.early_stopping:
        best_checkpoint_callback = callbacks.ModelCheckpoint(
            dirpath=config.checkpoint.dir,
            filename="best-epoch={epoch:03d}-val_loss={val_loss:.6f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
        )
        callbacks_list.extend(
            [
                best_checkpoint_callback,
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=args.early_stopping_patience,
                    min_delta=args.early_stopping_min_delta,
                    check_on_train_epoch_end=False,
                    verbose=True,
                ),
            ]
        )
    trainer = L.Trainer(
        accelerator="gpu" if args.device == "cuda" else "cpu",
        devices=1,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks_list,
        logger=CSVLogger(save_dir=str(run_dir), name="logs"),
        default_root_dir=str(run_dir),
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.validate_every or 1,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        deterministic=True,
        precision=args.precision,
        gradient_clip_val=config.optimization.grad_clip,
        enable_progress_bar=True,
    )
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if args.resume:
        _allow_trusted_checkpoint_metadata_for_resume()
    trainer.fit(model, train_loader, validation_loaders, ckpt_path=str(args.resume) if args.resume else None)
    elapsed_seconds = time.perf_counter() - started
    global_step = int(trainer.global_step)
    summary = {
        "run_dir": str(run_dir),
        "last_checkpoint": str(Path(config.checkpoint.dir) / "last.ckpt"),
        "best_checkpoint": best_checkpoint_callback.best_model_path if best_checkpoint_callback is not None else None,
        "checkpoint_for_downstream": (
            best_checkpoint_callback.best_model_path
            if best_checkpoint_callback is not None
            else str(Path(config.checkpoint.dir) / "last.ckpt")
        ),
        "global_step": global_step,
        "current_epoch": int(trainer.current_epoch),
        "pretraining_overhead": {
            "wall_time_seconds": elapsed_seconds,
            "optimizer_steps": global_step,
            "optimizer_steps_per_second": global_step / elapsed_seconds if elapsed_seconds else None,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else None,
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if args.device == "cuda" else None,
        },
        "runtime": runtime,
    }
    if validation_manifest is not None:
        summary["pretrain_validation_manifest"] = str(run_dir / "pretrain_split.json")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
