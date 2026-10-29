#!/usr/bin/env python3
"""Shared utilities for the frozen random encoder ablation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import torch
os.environ.setdefault("WANDB_KEY", "offline")
from configs import init_jepa_config
from models import build_encoder

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/ee_optimization/multi_block_n6/random_encoder_ablation_single_seed"
CHECKPOINT = RUN_ROOT / "random_encoder_seed_000.pt"
METADATA = RUN_ROOT / "random_encoder_seed_000.json"
CONFIG = ROOT / "configs/p0.yml"
SPLIT = ROOT / "runs/masking_head_seeds/multi_block_n6/head_seed_000/splits_seed_2025_primary_v3.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(checkpoint=CHECKPOINT, metadata=METADATA, seed=0):
    checkpoint, metadata = Path(checkpoint), Path(metadata)
    if checkpoint.exists() or metadata.exists():
        return validate(checkpoint, metadata, seed)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config = init_jepa_config(str(CONFIG))
    if config.encoder != init_jepa_config(str(ROOT / "configs/p2.yml")).encoder:
        raise ValueError("P0 and P2 encoder configurations differ; shared random control is invalid")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = build_encoder(config.encoder)
    torch.save({"state_dict": {"context_encoder." + name: value for name, value in encoder.state_dict().items()}}, checkpoint)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    record = {"encoder_kind": "random", "initialization_seed": seed, "architecture_config": str(CONFIG),
              "architecture_config_sha256": sha(CONFIG), "checkpoint": str(checkpoint.resolve()),
              "checkpoint_sha256": sha(checkpoint), "git_revision": revision,
              "torch_version": str(torch.__version__), "python_version": platform.python_version()}
    metadata.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return validate(checkpoint, metadata, seed)


def validate(checkpoint=CHECKPOINT, metadata=METADATA, seed=0):
    checkpoint, metadata = Path(checkpoint), Path(metadata)
    if not checkpoint.is_file() or not metadata.is_file():
        raise FileNotFoundError("random encoder checkpoint or metadata missing")
    record = json.loads(metadata.read_text())
    if (record.get("encoder_kind") != "random" or record.get("initialization_seed") != seed
            or record.get("architecture_config_sha256") != sha(CONFIG)
            or record.get("checkpoint") != str(checkpoint.resolve())
            or record.get("checkpoint_sha256") != sha(checkpoint)):
        raise ValueError("random encoder provenance mismatch")
    return record
