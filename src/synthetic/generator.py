"""Streaming, immutable HDF5 master generation through an audited task plug-in.

A task-specific plug-in is required because the source caches do not fully specify
waveforms, scene geometry, or preprocessing. This module never invents them.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from .manifests import create_realization_manifest, write_manifest
from .schema import SyntheticTaskProfile, require_verified_semantics

MAX_CHANNELS = 16
PATCH_SIZE = 16


def expected_master_shape(profile: SyntheticTaskProfile) -> tuple[int, int, int]:
    time = max(profile.temporal_levels) * PATCH_SIZE
    height, width = (time, 224) if profile.temporal_axis == 1 else (224, time)
    return MAX_CHANNELS, height, width


def load_plugin(specification: str) -> Callable[..., Any]:
    if ":" not in specification:
        raise ValueError("generator plug-in must be specified as python.module:callable")
    module_name, function_name = specification.split(":", maxsplit=1)
    if not module_name or not function_name:
        raise ValueError("generator plug-in must name both module and callable")
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise TypeError(f"generator plug-in {specification!r} is not callable")
    return function


def _validate_label(profile: SyntheticTaskProfile, label: Any) -> np.ndarray:
    values = np.asarray(label)
    if profile.target_type == "classification":
        if values.size != 1 or not np.issubdtype(values.dtype, np.integer):
            raise TypeError("classification plug-ins must return one integer class ID")
        value = int(values.reshape(-1)[0])
        if value not in profile.label_map:
            raise ValueError(f"label ID {value} is absent from the verified label map")
        return np.asarray(value, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("position plug-ins must return a finite normalized [x,y,z] target")
    if np.any(values < -1.0001) or np.any(values > 1.0001):
        raise ValueError("position labels must use the frozen head's normalized [-1,1] units")
    return values


def _validate_feature(
    feature: Any, profile: SyntheticTaskProfile, expected_shape: tuple[int, int, int]
) -> np.ndarray:
    array = np.asarray(feature)
    if array.shape != expected_shape:
        raise ValueError(f"plug-in feature shape {array.shape} != required master shape {expected_shape}")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError("plug-in features must be finite floating-point processed-cache features")
    # The view code only crops and selects channels. This conversion is storage dtype,
    # not a per-condition resize or normalization.
    return np.asarray(array, dtype=np.float32)


def generate_master_file(
    *,
    task: str,
    profile: SyntheticTaskProfile,
    task_config: dict[str, Any],
    generator: Callable[..., Any],
    destination: str | Path,
    count: int,
    seed: int,
    realization_prefix: str,
    sionna_version: str,
    split_kind: str,
) -> tuple[list[str], np.ndarray, str]:
    """Generate one maximum-information observation per stable realization ID."""

    require_verified_semantics(profile)
    if count < 2:
        raise ValueError("master generation requires at least two realizations")
    if split_kind not in {"pilot", "formal"}:
        raise ValueError("split_kind must be pilot or formal")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite synthetic master: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shape = expected_master_shape(profile)
    ids = [f"{realization_prefix}-{index:08d}" for index in range(count)]
    labels = np.empty((count, 3), np.float32) if profile.target_type == "position" else np.empty(count, np.int64)
    encoded_ids = np.asarray(ids, dtype=h5py.string_dtype(encoding="utf-8"))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as h5:
            h5.attrs["schema_version"] = 1
            h5.attrs["task"] = task
            h5.attrs["split_kind"] = split_kind
            h5.attrs["semantics_verified"] = int(profile.semantics_verified)
            h5.attrs["assumptions_approved"] = int(profile.assumptions_approved)
            h5.attrs["semantic_basis"] = profile.specification_status
            h5.attrs["assumptions_json"] = json.dumps(list(profile.assumptions), sort_keys=True)
            h5.attrs["feature_mode"] = profile.feature_mode
            h5.attrs["label_map_json"] = json.dumps(profile.label_map, sort_keys=True)
            h5.attrs["temporal_axis"] = profile.temporal_axis
            h5.attrs["patch_size"] = PATCH_SIZE
            h5.attrs["generator"] = getattr(generator, "__module__", "unknown")
            h5.attrs["sionna_version"] = sionna_version
            h5.attrs["seed"] = int(seed)
            h5.attrs["max_channels"] = MAX_CHANNELS
            h5.create_dataset(
                "features",
                shape=(count, *shape),
                dtype=np.float32,
                chunks=(1, *shape),
                compression="lzf",
            )
            h5.create_dataset("label", shape=labels.shape, dtype=labels.dtype)
            h5.create_dataset("realization_id", data=encoded_ids)
            h5.create_dataset(
                "array_element_indices",
                data=np.asarray(task_config.get("sionna", {}).get("array", {}).get("selection_order", np.arange(MAX_CHANNELS)), dtype=np.int16),
            )
            metadata = h5.create_dataset(
                "realization_metadata",
                shape=(count,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            for index, realization_id in enumerate(ids):
                sample_seed = int(np.random.SeedSequence([seed, index]).generate_state(1)[0])
                generated = generator(
                    realization_id=realization_id,
                    seed=sample_seed,
                    task_profile=profile,
                    task_config=task_config,
                )
                if not isinstance(generated, tuple) or len(generated) != 3:
                    raise TypeError(
                        "plug-in must return (processed_feature, task_label, realization_metadata)"
                    )
                feature, label, record = generated
                if not isinstance(record, dict):
                    raise TypeError("realization metadata must be a JSON-compatible mapping")
                if not isinstance(record.get("simulator_parameters"), dict):
                    raise ValueError("metadata must include a simulator_parameters mapping")
                record = {**record, "task": task, "realization_id": realization_id,
                          "seed": sample_seed, "sionna_version": sionna_version}
                feature = _validate_feature(feature, profile, shape)
                label = _validate_label(profile, label)
                h5["features"][index] = feature
                h5["label"][index] = label
                labels[index] = label
                metadata[index] = json.dumps(record, sort_keys=True)
            h5.flush()
        if destination.exists():
            raise FileExistsError(f"refusing to replace concurrent synthetic master: {destination}")
        os.link(temporary, destination)
        temporary.unlink()
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    digest = hashlib.sha256()
    with destination.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return ids, labels, digest.hexdigest()





def sionna_operational_realization(
    *, realization_id: str, seed: int, task_profile: SyntheticTaskProfile,
    task_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate one explicitly distribution-calibrated Sionna PHY realization.

    This is a synthetic-domain operational design. It must not be described as
    a reconstruction of the real collection process. A Sionna flat-fading
    channel produces the common 16-receiver aperture; the deterministic master
    feature transform occurs before any availability view is constructed.
    """

    if task_profile.name == "nrpos-outdoor":
        raise RuntimeError(
            "NR synthetic generation is disabled until its LatentWave source preprocessing, "
            "four-channel meaning, temporal-axis meaning, and physical coordinate transform "
            "are source-verified; do not use the former analytic four-ray proxy."
        )

    try:
        import sionna
        import torch
        import torch.nn.functional as functional
        from sionna.phy.channel import FlatFadingChannel
    except ImportError as exc:
        raise RuntimeError(
            "The operational plug-in requires the isolated Sionna 2.1 environment"
        ) from exc
    if not str(sionna.__version__).startswith("2.1."):
        raise RuntimeError(f"expected Sionna 2.1.x, found {sionna.__version__}")

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    name = task_profile.name
    length = 57_600 if name == "interference" else 16_368
    qpsk = (
        (2 * rng.integers(0, 2, length) - 1)
        + 1j * (2 * rng.integers(0, 2, length) - 1)
    ).astype(np.complex64) / np.sqrt(2.0)
    label: np.ndarray
    if name == "interference":
        label = np.asarray(int(rng.integers(0, 2)), dtype=np.int64)
        tone = np.exp(1j * 2.0 * np.pi * rng.uniform(-0.45, 0.45) * np.arange(length))
        qpsk = qpsk + (0.25 if int(label) == 0 else 2.0) * tone.astype(np.complex64)
        snr_db = float(rng.uniform(-5.0, 25.0))
    elif name == "jamming":
        label = np.asarray(int(rng.integers(0, 6)), dtype=np.int64)
        time = np.arange(length, dtype=np.float32)
        if int(label) == 0:
            qpsk += 1.5 * (np.mod(time, 256.0) < 12.0)
        elif int(label) == 1:
            qpsk += 1.5 * np.exp(1j * 2.0 * np.pi * 0.05 * time)
        elif int(label) == 3:
            qpsk *= 1.0 + 0.75 * np.sin(2.0 * np.pi * 0.003 * time)
        elif int(label) == 4:
            qpsk += np.exp(1j * 2.0 * np.pi * (0.00001 * time**2))
        elif int(label) == 5:
            qpsk += np.exp(1j * 2.0 * np.pi * (0.03 * time + 0.7 * np.sin(0.002 * time)))
        snr_db = float(rng.uniform(-5.0, 25.0))
    else:
        raise ValueError(f"unsupported operational plug-in task: {name}")

    channel = FlatFadingChannel(
        num_tx_ant=1, num_rx_ant=16, return_channel=True, precision="single", device="cpu"
    )
    transmitted = torch.from_numpy(qpsk).reshape(-1, 1)
    noise_power = float(np.mean(np.abs(qpsk) ** 2) / (10.0 ** (snr_db / 10.0)))
    received, channel_matrix = channel(transmitted, noise_power)
    received = received.detach().cpu().numpy().T
    if name == "jamming":
        from scipy.signal import stft

        _, _, spectrum = stft(
            received,
            fs=float(task_config["sionna"]["sample_rate_hz"]),
            nperseg=256,
            noverlap=192,
            return_onesided=False,
            boundary="zeros",
            padded=True,
            axis=-1,
        )
        image = 20.0 * np.log10(np.maximum(np.abs(np.fft.fftshift(spectrum, axes=1)), 1e-8))
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(1)
    else:
        spectrum = torch.stft(
            torch.from_numpy(received), n_fft=256, hop_length=64, win_length=256,
            window=torch.hann_window(256), return_complex=True, center=True,
        )
        image = 20.0 * torch.log10(torch.clamp(spectrum.abs(), min=1e-8))
        image = torch.fft.fftshift(image, dim=1).unsqueeze(1)
    image = functional.interpolate(image, size=(224, 896), mode="bilinear", align_corners=False)
    feature = image[:, 0].detach().cpu().numpy().astype(np.float32)
    if name == "jamming":
        affine = task_config["sionna"]["source_preprocessing"]["source_global_affine"]
        feature = (feature - float(affine["mean"])) / float(affine["std"])
    else:
        feature = (feature - float(feature.mean())) / float(max(feature.std(), 1e-6))
    if task_profile.temporal_axis == 1:
        feature = np.swapaxes(feature, 1, 2)
    metadata = {
        "simulator_parameters": {
            "backend": "sionna.phy.FlatFadingChannel",
            "profile_status": task_config["sionna"]["profile_status"],
            "carrier_frequency_hz": task_config["sionna"]["carrier_frequency_hz"],
            "snr_db": snr_db,
            "normalization": (
                "source_global_affine" if name == "jamming"
                else "per-master placeholder; replace with a locked pilot affine before formal generation"
            ),
        },
        "feature_contract": task_profile.feature_mode,
        "realization_id": realization_id,
        "channel_shape": list(channel_matrix.shape),
    }
    return feature, label, metadata

    return ids, labels, digest.hexdigest()


def create_formal_manifest(
    *,
    ids: list[str],
    labels: np.ndarray,
    profile: SyntheticTaskProfile,
    task: str,
    seed: int,
    master_sha256: str,
    sample_plan: dict[str, Any],
    sample_plan_sha256: str,
) -> dict[str, Any]:
    manifest = create_realization_manifest(
        ids,
        labels,
        task=task,
        target_type=profile.target_type,
        seed=seed,
        coordinate_min=(-1.0, -1.0, -1.0) if profile.target_type == "position" else None,
        coordinate_max=(1.0, 1.0, 1.0) if profile.target_type == "position" else None,
    )
    if profile.target_type == "position":
        manifest["position_stratification_units"] = "frozen-head normalized [-1,1]"
    manifest["master_sha256"] = master_sha256
    manifest["sample_plan"] = sample_plan
    manifest["sample_plan_sha256"] = sample_plan_sha256
    manifest["split_protocol"] = "paired realization IDs, stratified 20/20/60"
    return manifest


def persist_formal_manifest(path: str | Path, manifest: dict[str, Any]) -> str:
    return write_manifest(path, manifest)
