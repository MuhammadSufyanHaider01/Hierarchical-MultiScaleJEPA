"""Compact live progress output for Slurm training logs."""

from __future__ import annotations

import sys


def render_batch_progress(label: str, epoch: int, epochs: int, batch_index: int, total_batches: int) -> None:
    total = max(1, total_batches)
    done = min(batch_index + 1, total)
    width = 24
    filled = int(width * done / total)
    bar = "=" * filled + (">" if filled < width else "") + "." * (width - filled - (1 if filled < width else 0))
    sys.stdout.write(f"\r{label} | epoch {epoch:03d}/{epochs:03d} | [{bar}] batch {done:03d}/{total:03d}")
    sys.stdout.flush()


def finish_epoch_progress(label: str, epoch: int, epochs: int, metric_name: str, metric_value: float) -> None:
    # Replace the in-place batch bar with one durable summary line in the Slurm log.
    sys.stdout.write(f"\r{label} | epoch {epoch:03d}/{epochs:03d} | {metric_name}={metric_value:.6g}\n")
    sys.stdout.flush()
