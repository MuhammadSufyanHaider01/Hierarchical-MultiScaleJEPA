"""Checkpoint metadata helpers for reproducible P0 experiments."""

from __future__ import annotations

import json
import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import torch


PINNED_LATENTWAVE_COMMIT = "330e9ef5f6eb69e728ba43b3755036cffdd086cd"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _model_dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_model_dump(item) for item in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {str(key): _model_dump(item) for key, item in vars(value).items() if not key.startswith("_")}
    return value


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _source_tree_fingerprint() -> dict[str, Any]:
    """Fingerprint all non-ignored source files, including uncommitted work."""

    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {"available": False}
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip() or "git ls-files failed"}

    digest = hashlib.sha256()
    count = 0
    for relative in sorted(line for line in result.stdout.splitlines() if line):
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        count += 1
    return {"available": True, "file_count": count, "sha256": digest.hexdigest()}


def _nvidia_smi_metadata() -> dict[str, Any]:
    query = "name,driver_version,memory.total"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {"available": False}
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip() or "nvidia-smi failed"}
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            rows.append({"name": fields[0], "driver_version": fields[1], "memory_total_mib": fields[2]})
    return {"available": bool(rows), "gpus": rows}


def environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "nvidia_smi": _nvidia_smi_metadata(),
    }
    if torch.cuda.is_available():
        metadata["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
    return metadata


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _hdf5_entry(path: Path, key: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": str(path.relative_to(REPOSITORY_ROOT)) if path.is_relative_to(REPOSITORY_ROOT) else str(path),
        "key": key,
        "exists": path.is_file(),
    }
    if not path.is_file():
        return entry
    entry["size_bytes"] = path.stat().st_size
    with h5py.File(path, "r") as h5:
        entry["keys"] = sorted(h5.keys())
        if key in h5:
            dataset = h5[key]
            entry["shape"] = list(dataset.shape)
            entry["dtype"] = str(dataset.dtype)
            entry["count"] = int(dataset.shape[0]) if dataset.shape else 1
    return entry


def dataset_manifest(config: Any) -> list[dict[str, Any]]:
    data_config = config.data
    manifest = []
    for dataset in data_config.pretrain_datasets:
        path = _resolve_path(dataset.path)
        manifest.append({"role": "pretraining", "name": dataset.name, **_hdf5_entry(path, dataset.key)})
    downstream = (
        ("rfs", data_config.rfs_path, "image"),
        ("position", data_config.pos_path, "features"),
        ("deepmimo", data_config.beam_path, "sample"),
    )
    for name, path_value, key in downstream:
        manifest.append({"role": "downstream", "name": name, **_hdf5_entry(_resolve_path(path_value), key)})
    return manifest


def build_checkpoint_metadata(config: Any, seed: int, config_path: str | Path | None = None) -> dict[str, Any]:
    metadata = {
        "schema_version": 1,
        "source": {
            "name": "latent-wave",
            "repository": "https://github.com/AhmedEssam19/latent-wave",
            "commit": PINNED_LATENTWAVE_COMMIT,
        },
        "git_revision": _git_revision(),
        "source_tree": _source_tree_fingerprint(),
        "seed": int(seed),
        "config_path": str(config_path) if config_path is not None else None,
        "config": _model_dump(config),
        "dataset_manifest": dataset_manifest(config),
        "environment": environment_metadata(),
    }
    json.dumps(metadata)
    return metadata
