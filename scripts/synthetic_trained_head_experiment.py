#!/usr/bin/env python3
"""Train synthetic-only exits, select them on validation, and test after review."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.exits import select_earliest_within_tolerance  # noqa: E402
from models.encoder import Encoder  # noqa: E402
from models.exit_heads import ExitHead, load_context_encoder_checkpoint  # noqa: E402
from synthetic_study.protocol import (  # noqa: E402
    ANTENNA_LEVELS, EXIT_LAYERS, TIME_PATCH_LEVELS, availability_view,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def index_lookup(h5: h5py.File) -> dict[str, int]:
    ids = h5["realization_id"].asstr()[:]
    return {value: index for index, value in enumerate(ids.tolist())}


def condition_spec(task: str, axis: str) -> tuple[list[int], int, int]:
    if axis == "antenna":
        return list(ANTENNA_LEVELS), 16, 14
    if axis == "time":
        fixed_channels = {"nrpos-outdoor": 4, "interference": 1, "jamming": 1}[task]
        return list(TIME_PATCH_LEVELS), fixed_channels, 56
    raise ValueError(f"axis must be antenna or time, got {axis!r}")


def full_condition(task: str, axis: str) -> tuple[str, int]:
    levels, _, _ = condition_spec(task, axis)
    return ("antenna", 16) if axis == "antenna" else ("time", max(levels))


def h5_read_indices(dataset: h5py.Dataset, indices: list[int]) -> np.ndarray:
    requested = np.asarray(indices, dtype=np.int64)
    order = np.argsort(requested)
    sorted_values = dataset[requested[order]]
    return np.asarray(sorted_values[np.argsort(order)])


def task_labels(task: str, h5: h5py.File, indices: list[int], profile: dict[str, Any]) -> np.ndarray:
    if task == "nrpos-outdoor":
        return np.asarray(h5_read_indices(h5["receiver_xyz_m"], indices), dtype=np.float32)
    key = "interference_label" if task == "interference" else "jammer_mode"
    return np.asarray(h5_read_indices(h5[key], indices), dtype=np.int64).reshape(-1)


def normalize_labels(task: str, labels: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    if task != "nrpos-outdoor":
        return labels.astype(np.int64, copy=False)
    spec = profile["tasks"][task]
    lo = np.asarray(spec["coordinate_min_m"], dtype=np.float32)
    hi = np.asarray(spec["coordinate_max_m"], dtype=np.float32)
    span = hi - lo
    safe = np.where(span > 0, span, 1.0)
    out = 2.0 * (labels - lo) / safe - 1.0
    out[:, span == 0] = 0.0
    return out.astype(np.float32)


def decode_positions(values: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    spec = profile["tasks"]["nrpos-outdoor"]
    lo = np.asarray(spec["coordinate_min_m"], dtype=np.float32)
    hi = np.asarray(spec["coordinate_max_m"], dtype=np.float32)
    out = (np.asarray(values, dtype=np.float32) + 1.0) * 0.5 * (hi - lo) + lo
    out[:, hi == lo] = lo[hi == lo]
    return out


def primary_metric(task: str, truth: np.ndarray, prediction: np.ndarray, profile: dict[str, Any]) -> float:
    if task == "nrpos-outdoor":
        error = np.linalg.norm(decode_positions(prediction, profile) - truth, axis=1)
        return float(error.mean())
    pred = np.argmax(prediction, axis=1)
    per_class = []
    for label in np.unique(truth):
        mask = truth == label
        if mask.any():
            per_class.append(float(np.mean(pred[mask] == label)))
    return float(np.mean(per_class))


def encoder_state_sha256(encoder: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(encoder.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def encoder_for(method: str, profile: dict[str, Any], device: torch.device) -> tuple[nn.Module, Path]:
    entry = profile["models"][method]
    cfg_path = ROOT / entry["method_config"]
    cfg = yaml.safe_load(cfg_path.read_text())["encoder"]
    encoder = Encoder(
        img_size=tuple(cfg["img_size"]), patch_size=int(cfg["patch_size"]),
        embed_dim=int(cfg["embed_dim"]), depth=int(cfg["depth"]),
        num_heads=int(cfg["num_heads"]),
    )
    checkpoint = ROOT / entry["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"authoritative {method} checkpoint not found: {checkpoint}")
    load_context_encoder_checkpoint(encoder, checkpoint)
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder, checkpoint


def fit_normalization(h5: h5py.File, indices: list[int], task: str, axis: str, level: int) -> tuple[float, float]:
    count = 0
    total = 0.0
    total_sq = 0.0
    for index in indices:
        x = availability_view(h5["features"][index], task=task, axis=axis, level=level).astype(np.float32)
        flat = x.reshape(-1).astype(np.float64)
        count += flat.size
        total += float(flat.sum())
        total_sq += float(np.square(flat).sum())
    mean = total / count
    variance = max(0.0, total_sq / count - mean * mean)
    return float(mean), float(max(np.sqrt(variance), 1.0e-6))


def encode_indices(
    h5: h5py.File, indices: list[int], *, task: str, axis: str, level: int,
    mean: float, std: float, encoder: nn.Module, device: torch.device,
    batch_size: int, verbose: bool = True,
) -> np.ndarray:
    result = np.empty((len(indices), len(EXIT_LAYERS), encoder.pos_embed.shape[-1]), dtype=np.float32)
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            array = np.stack([
                availability_view(h5["features"][i], task=task, axis=axis, level=level)
                for i in selected
            ]).astype(np.float32)
            array = (array - mean) / std
            x = torch.from_numpy(array).to(device=device, dtype=torch.float32)
            outputs = encoder(x, return_layers=EXIT_LAYERS)
            for layer_index, layer in enumerate(EXIT_LAYERS):
                result[start : start + len(selected), layer_index] = outputs[layer].mean(dim=1).float().cpu().numpy()
            del x, outputs
            if verbose and (start + len(selected) == len(indices) or (start + len(selected)) % max(batch_size * 100, 100) == 0):
                print(f"encoded {start + len(selected)}/{len(indices)} at {axis}={level}", flush=True)
    return result


def encode_cached(
    h5: h5py.File, indices: list[int], *, task: str, axis: str, level: int,
    mean: float, std: float, encoder: nn.Module, device: torch.device,
    batch_size: int, cache_root: Path, method: str, mode: str, checkpoint_hash: str,
) -> np.ndarray:
    channels = level if axis == "antenna" else {"nrpos-outdoor": 4, "interference": 1, "jamming": 1}[task]
    temporal_patches = 14 if axis == "antenna" else level
    all_ids = h5["realization_id"].asstr()[:]
    realization_ids = [str(all_ids[index]) for index in indices]
    identity = {
        "checkpoint": checkpoint_hash, "split_ids": hashlib.sha256("\n".join(realization_ids).encode()).hexdigest(),
        "channels": channels, "temporal_patches": temporal_patches,
        "mean": float(mean), "std": float(std),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = cache_root / method / mode / f"{key}.npy"
    if path.is_file():
        cached = np.load(path, allow_pickle=False)
        expected = (len(indices), len(EXIT_LAYERS), encoder.pos_embed.shape[-1])
        if cached.shape != expected or not np.isfinite(cached).all():
            raise RuntimeError(f"invalid cached encoder representations: {path}")
        return cached
    value = encode_indices(h5, indices, task=task, axis=axis, level=level, mean=mean, std=std, encoder=encoder, device=device, batch_size=batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    if temporary.exists():
        raise FileExistsError(f"temporary representation cache exists: {temporary}")
    np.save(temporary, value, allow_pickle=False)
    temporary.replace(path)
    return value

def metric_is_higher(task: str) -> bool:
    return task != "nrpos-outdoor"


def save_and_train_heads(
    *, task: str, axis: str, method: str, h5: h5py.File, split: dict[str, Any],
    id_map: dict[str, int], profile: dict[str, Any], device: torch.device,
    output_dir: Path, encoder_batch_size: int, head_batch_size: int,
) -> dict[str, Any]:
    _, full_level = full_condition(task, axis)
    full_axis = axis
    train_idx = [id_map[x] for x in split["realization_ids"]["head_train"]]
    val_idx = [id_map[x] for x in split["realization_ids"]["head_validation"]]
    mean, std = fit_normalization(h5, train_idx, task, full_axis, full_level)
    encoder, checkpoint = encoder_for(method, profile, device)
    encoder_hash_before = encoder_state_sha256(encoder)
    started = time.monotonic()
    cache_root = ROOT / profile["outputs"]["run_root"] / "representation_cache"
    checkpoint_hash = sha256(checkpoint)
    train_repr = encode_cached(h5, train_idx, task=task, axis=full_axis, level=full_level, mean=mean, std=std, encoder=encoder, device=device, batch_size=encoder_batch_size, cache_root=cache_root, method=method, mode="pilot" if split.get("pilot") else "formal", checkpoint_hash=checkpoint_hash)
    val_repr = encode_cached(h5, val_idx, task=task, axis=full_axis, level=full_level, mean=mean, std=std, encoder=encoder, device=device, batch_size=encoder_batch_size, cache_root=cache_root, method=method, mode="pilot" if split.get("pilot") else "formal", checkpoint_hash=checkpoint_hash)
    y_train_raw = task_labels(task, h5, train_idx, profile)
    y_val_raw = task_labels(task, h5, val_idx, profile)
    y_train = normalize_labels(task, y_train_raw, profile)
    y_val = normalize_labels(task, y_val_raw, profile)
    outputs = int(profile["tasks"][task]["num_outputs"])
    train_t = torch.from_numpy(train_repr).to(device)
    val_t = torch.from_numpy(val_repr).to(device)
    target_train = torch.from_numpy(y_train).to(device)
    target_val = torch.from_numpy(y_val).to(device)
    all_records = {}
    settings = profile["head_training"]
    for layer_idx, layer in enumerate(EXIT_LAYERS):
        torch.manual_seed(int(profile.get("seed", 0)) + layer)
        head = ExitHead(int(encoder.pos_embed.shape[-1]), outputs).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
        loss_fn = nn.MSELoss() if task == "nrpos-outdoor" else nn.CrossEntropyLoss()
        best_metric = float("inf") if not metric_is_higher(task) else -float("inf")
        best_state = None
        best_epoch = 0
        patience = int(settings["early_stopping_patience"])
        waiting = 0
        for epoch in range(int(settings["max_epochs"])):
            head.train()
            order = torch.randperm(len(train_t), device=device)
            for start in range(0, len(order), head_batch_size):
                batch_ids = order[start : start + head_batch_size]
                logits = head(train_t[batch_ids, layer_idx : layer_idx + 1])
                target = target_train[batch_ids]
                loss = loss_fn(logits, target if task == "nrpos-outdoor" else target.long())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            head.eval()
            with torch.inference_mode():
                pred = head(val_t[:, layer_idx : layer_idx + 1]).cpu().numpy()
            score = primary_metric(task, y_val_raw, pred, profile)
            improved = score < best_metric if not metric_is_higher(task) else score > best_metric
            if improved:
                best_metric, best_epoch, waiting = score, epoch + 1, 0
                best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
            else:
                waiting += 1
                if waiting >= patience:
                    break
        if best_state is None:
            raise RuntimeError(f"no best state selected for {task}/{axis}/{method}/L{layer}")
        head.load_state_dict(best_state)
        ckpt_dir = output_dir / "heads"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"L{layer}.pt"
        if ckpt_path.exists():
            raise FileExistsError(f"refusing to overwrite immutable head checkpoint {ckpt_path}")
        torch.save({"state_dict": best_state, "task": task, "axis": axis, "method": method, "exit_layer": layer, "full_condition_level": full_level, "normalization": {"mean": mean, "std": std}, "source_checkpoint_sha256": sha256(checkpoint), "best_epoch": best_epoch, "head_validation_metric": best_metric}, ckpt_path)
        all_records[str(layer)] = {"path": str(ckpt_path), "sha256": sha256(ckpt_path), "best_epoch": best_epoch, "head_validation_metric": best_metric}
        print(f"trained {method}/{task}/{axis} L{layer}: epoch={best_epoch}, head_validation_{profile['tasks'][task]['metric']}={best_metric:.6g}", flush=True)
    encoder_hash_after = encoder_state_sha256(encoder)
    if encoder_hash_after != encoder_hash_before:
        raise RuntimeError("frozen encoder weights changed during synthetic head training")
    state = {
        "method": method, "task": task, "axis": axis, "full_condition": {"axis": full_axis, "level": full_level},
        "normalization": {"mean": mean, "std": std}, "normalization_fit_ids_sha256": hashlib.sha256("\n".join(split["realization_ids"]["head_train"]).encode()).hexdigest(),
        "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": sha256(checkpoint),
        "master_sha256": split["master_sha256"],
        "profile_sha256": split.get("profile_sha256"), "scene_sha256": split.get("scene_sha256"),
        "split_manifest": split.get("_manifest_path"),
        "split_manifest_sha256": split.get("_manifest_sha256"),
        "encoder_state_sha256_before": encoder_hash_before,
        "encoder_state_sha256_after": encoder_hash_after,
        "head_checkpoints": all_records, "train_count": len(train_idx), "head_validation_count": len(val_idx),
        "training_seconds": round(time.monotonic() - started, 3),
    }
    json_write(output_dir / "training_record.json", state)
    return state


def anchor_gate(task: str, truth: np.ndarray, prediction: np.ndarray, train_truth: np.ndarray, profile: dict[str, Any], seed: int = 517) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(truth)
    values = np.empty(3000, dtype=np.float64)
    if task == "nrpos-outdoor":
        pred_m = decode_positions(prediction, profile)
        centroid = np.repeat(np.mean(train_truth, axis=0, keepdims=True), n, axis=0)
        model_error = np.linalg.norm(pred_m - truth, axis=1)
        baseline_error = np.linalg.norm(centroid - truth, axis=1)
        differences = baseline_error - model_error
        for repeat in range(len(values)):
            sample = rng.integers(0, n, n)
            values[repeat] = differences[sample].mean()
        lower, upper = np.quantile(values, [0.025, 0.975])
        return {
            "metric": float(model_error.mean()), "trivial_baseline_metric": float(baseline_error.mean()),
            "paired_improvement_baseline_minus_model": float(differences.mean()),
            "paired_improvement_ci95": [float(lower), float(upper)],
            "gate_passed": bool(lower > 0), "gate_rule": "95% paired bootstrap lower bound of centroid error minus model error is greater than zero",
            "bootstrap_replicates": len(values),
        }
    predicted_class = np.argmax(prediction, axis=1)
    chance = 1.0 / int(profile["tasks"][task]["num_outputs"])
    classes = np.unique(truth)
    groups = [np.flatnonzero(truth == label) for label in classes]
    for repeat in range(len(values)):
        sample = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        values[repeat] = np.mean([np.mean(predicted_class[sample][truth[sample] == label] == label) for label in classes]) - chance
    lower, upper = np.quantile(values, [0.025, 0.975])
    model_metric = primary_metric(task, truth, prediction, profile)
    return {
        "metric": model_metric, "chance_baseline_metric": chance,
        "improvement_over_chance": float(model_metric - chance),
        "improvement_ci95": [float(lower), float(upper)],
        "gate_passed": bool(lower > 0), "gate_rule": "95% bootstrap lower bound of macro accuracy minus uniform-chance accuracy is greater than zero",
        "bootstrap_replicates": len(values),
    }


def evaluate_anchor(*, task: str, axis: str, method: str, h5: h5py.File, split: dict[str, Any], id_map: dict[str, int], profile: dict[str, Any], device: torch.device, encoder_batch_size: int) -> dict[str, Any]:
    mode = "pilot" if bool(split.get("pilot")) else "formal"
    base = ROOT / profile["outputs"]["run_root"] / mode / axis / task / method
    path = base / "calibration_anchor.json"
    if path.exists():
        raise FileExistsError(f"calibration anchor is immutable and already exists: {path}")
    train_idx = [id_map[value] for value in split["realization_ids"]["head_train"]]
    anchor_split = "exit_selection" if mode == "pilot" else "calibration"
    anchor_idx = [id_map[value] for value in split["realization_ids"][anchor_split]]
    if anchor_split not in split["realization_ids"]:
        raise ValueError(f"manifest has no {anchor_split} split")
    training = json.loads((base / "training_record.json").read_text())
    encoder, heads, training = load_heads(method, task, axis, mode, profile, device)
    if training.get("master_sha256") not in (None, split["master_sha256"]):
        raise RuntimeError("head training record was produced from a different immutable master")
    expected_fit_ids = hashlib.sha256("\n".join(split["realization_ids"]["head_train"]).encode()).hexdigest()
    if training.get("normalization_fit_ids_sha256") != expected_fit_ids:
        raise RuntimeError("head training split IDs differ from the supplied immutable manifest")
    _, full_level = full_condition(task, axis)
    mean, std = float(training["normalization"]["mean"]), float(training["normalization"]["std"])
    features = encode_cached(h5, anchor_idx, task=task, axis=axis, level=full_level, mean=mean, std=std, encoder=encoder, device=device, batch_size=encoder_batch_size, cache_root=ROOT / profile["outputs"]["run_root"] / "representation_cache", method=method, mode=mode, checkpoint_hash=training["source_checkpoint_sha256"])
    with torch.inference_mode():
        logits = heads[8](torch.from_numpy(features[:, -1:, :]).to(device)).cpu().numpy()
    truth = task_labels(task, h5, anchor_idx, profile)
    train_truth = task_labels(task, h5, train_idx, profile)
    gate = anchor_gate(task, truth, logits, train_truth, profile)
    packet = {
        "schema_version": 1, "status": "passed" if gate["gate_passed"] else "failed",
        "mode": mode, "stage": "full_information_calibration_anchor", "method": method,
        "task": task, "axis": axis, "condition": {"axis": axis, "level": full_level},
        "metric": profile["tasks"][task]["metric"], "split_name": anchor_split,
        "split_ids_sha256": hashlib.sha256("\n".join(split["realization_ids"][anchor_split]).encode()).hexdigest(),
        "master_sha256": split["master_sha256"],
        "profile_sha256": split.get("profile_sha256"), "scene_sha256": split.get("scene_sha256"),
        "split_manifest": split.get("_manifest_path"), "split_manifest_sha256": split.get("_manifest_sha256"),
        "source_checkpoint_sha256": training["source_checkpoint_sha256"],
        "training_record_sha256": sha256(base / "training_record.json"),
        "normalization": training["normalization"], "gate": gate,
    }
    prediction_path = base / "calibration_anchor_predictions.npz"
    if prediction_path.exists():
        raise FileExistsError(f"anchor predictions already exist: {prediction_path}")
    np.savez_compressed(prediction_path, truth=truth, layer_8=logits, train_truth=train_truth)
    json_write(path, packet)
    print(f"{method}/{task}/{axis} full-info anchor: metric={gate['metric']:.5g}, passed={gate['gate_passed']}", flush=True)
    return packet

def approximate_flops(channels: int, temporal_patches: int, layer: int, outputs: int, embed_dim: int = 256, mlp_ratio: int = 4) -> int:
    n = channels * 14 * temporal_patches
    patch_macs = n * embed_dim * 16 * 16
    per_block_macs = 12 * n * embed_dim * embed_dim + 2 * n * n * embed_dim
    head_macs = embed_dim * outputs
    return int(2 * (patch_macs + layer * per_block_macs + head_macs))


def synchronized_latency(encoder: nn.Module, head: nn.Module, sample: torch.Tensor, layer: int, device: torch.device) -> float:
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(3):
            tokens = encoder.forward_to_layer(sample, layer)
            head(tokens)
        sync()
        start = time.perf_counter()
        repeats = 10
        for _ in range(repeats):
            tokens = encoder.forward_to_layer(sample, layer)
            head(tokens)
        sync()
    return (time.perf_counter() - start) / repeats * 1000.0


def load_heads(method: str, task: str, axis: str, mode: str, profile: dict[str, Any], device: torch.device) -> tuple[nn.Module, dict[int, ExitHead], dict[str, Any]]:
    base = ROOT / profile["outputs"]["run_root"] / mode / axis / task / method
    record = json.loads((base / "training_record.json").read_text())
    encoder, checkpoint = encoder_for(method, profile, device)
    if sha256(checkpoint) != record["source_checkpoint_sha256"]:
        raise RuntimeError("foundation checkpoint hash changed since head training")
    heads = {}
    for layer in EXIT_LAYERS:
        path = Path(record["head_checkpoints"][str(layer)]["path"])
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if sha256(path) != record["head_checkpoints"][str(layer)]["sha256"]:
            raise RuntimeError(f"head checkpoint hash mismatch: {path}")
        head = ExitHead(int(encoder.pos_embed.shape[-1]), int(profile["tasks"][task]["num_outputs"]))
        head.load_state_dict(ckpt["state_dict"], strict=True)
        head.to(device).eval()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        heads[layer] = head
    return encoder, heads, record


def evaluate_stage(
    *, task: str, axis: str, method: str, stage: str, h5: h5py.File,
    split: dict[str, Any], id_map: dict[str, int], profile: dict[str, Any],
    device: torch.device, encoder_batch_size: int,
) -> dict[str, Any]:
    mode = "pilot" if bool(split.get("pilot")) else "formal"
    base = ROOT / profile["outputs"]["run_root"] / mode / axis / task / method
    selection_path = base / "selection.json"
    if stage == "select" and mode == "pilot":
        raise PermissionError("pilot data are for calibration/count planning only; formal selection starts after the locked master")
    prediction_path = base / "selection_predictions.npz"
    if stage == "select" and (selection_path.exists() or prediction_path.exists()):
        raise FileExistsError(f"selection packet/predictions already exist (immutable): {base}")
    training = json.loads((base / "training_record.json").read_text())
    selected_artifact = None
    if stage == "select":
        for candidate_axis in ("antenna", "time"):
            for candidate_method in ("p0", "p2"):
                gate_path = ROOT / profile["outputs"]["run_root"] / mode / candidate_axis / task / candidate_method / "calibration_anchor.json"
                if not gate_path.is_file() or json.loads(gate_path.read_text()).get("status") != "passed":
                    raise PermissionError(f"formal selection requires passed P0/P2 antenna and time calibration anchors: {gate_path}")
    if stage == "test":
        selected_artifact = json.loads(selection_path.read_text())
        if selected_artifact.get("status") != "selection_complete_review_required":
            raise RuntimeError("selection artifact is incomplete or has an unexpected status")
    encoder, heads, training = load_heads(method, task, axis, mode, profile, device)
    if training.get("master_sha256") not in (None, split["master_sha256"]):
        raise RuntimeError("head training record was produced from a different immutable master")
    expected_fit_ids = hashlib.sha256("\n".join(split["realization_ids"]["head_train"]).encode()).hexdigest()
    if training.get("normalization_fit_ids_sha256") != expected_fit_ids:
        raise RuntimeError("head training split IDs differ from the supplied immutable manifest")
    mean, std = float(training["normalization"]["mean"]), float(training["normalization"]["std"])
    split_name = "exit_selection" if stage == "select" else "test"
    if stage == "test":
        approval = base / "review_approval.json"
        if not approval.is_file():
            raise PermissionError(f"test stage requires explicit review approval file: {approval}")
        approval_record = json.loads(approval.read_text())
        if approval_record.get("approved") is not True or approval_record.get("selection_sha256") != sha256(selection_path) or not approval_record.get("reviewer"):
            raise PermissionError("review approval must bind the reviewer and exact selection artifact SHA-256")
    evaluation_idx = [id_map[value] for value in split["realization_ids"][split_name]]
    truth = task_labels(task, h5, evaluation_idx, profile)
    levels, max_channels, max_time = condition_spec(task, axis)
    outputs = {}
    summaries = {}
    predictions_by_level = {}
    for level in levels:
        cond_axis, cond_level = axis, level
        if axis == "antenna":
            channels, patches = level, 14
        else:
            channels, patches = max_channels, level
        reps = encode_cached(h5, evaluation_idx, task=task, axis=cond_axis, level=cond_level, mean=mean, std=std, encoder=encoder, device=device, batch_size=encoder_batch_size, cache_root=ROOT / profile["outputs"]["run_root"] / "representation_cache", method=method, mode=mode, checkpoint_hash=training["source_checkpoint_sha256"])
        fixed = {}
        pred = {}
        for layer_idx, layer in enumerate(EXIT_LAYERS):
            with torch.inference_mode():
                tensor = torch.from_numpy(reps[:, layer_idx : layer_idx + 1]).to(device)
                value = heads[layer](tensor).cpu().numpy()
            fixed[str(layer)] = primary_metric(task, truth, value, profile)
            pred[str(layer)] = value
        if stage == "select":
            selected = select_earliest_within_tolerance(
                {int(key): value for key, value in fixed.items()},
                higher_is_better=metric_is_higher(task),
                tolerance=float(profile["head_training"]["selection_tolerance"]),
            )
        else:
            selected = int(selected_artifact["conditions"][str(level)]["selected_exit"])
        if selected is not None:
            raw = availability_view(h5["features"][evaluation_idx[0]], task=task, axis=axis, level=level).astype(np.float32)
            sample = torch.from_numpy(((raw - mean) / std)[None]).to(device)
            latency = synchronized_latency(encoder, heads[selected], sample, selected, device)
            flops = approximate_flops(channels, patches, selected, int(profile["tasks"][task]["num_outputs"]))
            selected_outputs = pred[str(selected)]
            gap = None
            l8 = fixed["8"]
            selected_metric = fixed[str(selected)]
            if metric_is_higher(task):
                gap = float(l8 - selected_metric)
            else:
                gap = float(selected_metric - l8)
            relative_gap = float(gap / max(abs(l8), 1.0e-12))
        else:
            latency, flops, gap, relative_gap = None, None, None, None
            selected_outputs = None
            selected_metric = None
        array_order = profile["availability"]["antenna_subset_order"]
        summaries[str(level)] = {
            "condition": {"axis": axis, "level": level, "channels": channels,
                          "array_element_indices": array_order[:channels],
                          "temporal_patches": patches, "time_columns": [0, patches * 16],
                          "token_count": channels * 14 * patches},
            "fixed_exit_metrics": fixed, "selected_exit": selected,
            "selected_metric": selected_metric, "l8_gap": gap,
            "l8_relative_gap": relative_gap,
            "selected_exit_flops_approx": flops, "selected_exit_latency_ms_batch1": latency,
        }
        predictions_by_level[str(level)] = {key: value.tolist() for key, value in pred.items()}
        if selected is not None:
            predictions_by_level[str(level)]["selected"] = selected_outputs.tolist()
        print(f"{method}/{task}/{axis}={level}: fixed={fixed}, selected={selected}", flush=True)
    packet = {
        "schema_version": 1, "status": "selection_complete_review_required" if stage == "select" else "test_complete",
        "stage": stage, "mode": mode, "method": method, "task": task, "axis": axis,
        "metric": profile["tasks"][task]["metric"], "higher_is_better": metric_is_higher(task),
        "tolerance": profile["head_training"]["selection_tolerance"], "split_name": split_name,
        "split_ids_sha256": hashlib.sha256("\n".join(split["realization_ids"][split_name]).encode()).hexdigest(),
        "master_sha256": split["master_sha256"],
        "profile_sha256": split.get("profile_sha256"), "scene_sha256": split.get("scene_sha256"),
        "split_manifest": split.get("_manifest_path"), "split_manifest_sha256": split.get("_manifest_sha256"),
        "source_checkpoint": training["source_checkpoint"],
        "source_checkpoint_sha256": training["source_checkpoint_sha256"], "training_record": str(base / "training_record.json"),
        "training_record_sha256": sha256(base / "training_record.json"), "normalization": training["normalization"],
        "conditions": summaries,
    }
    if stage == "select":
        temporary_predictions = base / "selection_predictions.tmp.npz"
        if temporary_predictions.exists():
            raise FileExistsError(f"temporary selection predictions exist: {temporary_predictions}")
        np.savez_compressed(temporary_predictions, **{
            f"level_{level}_truth": truth for level in levels
        }, **{
            f"level_{level}_layer_{layer}": np.asarray(predictions_by_level[str(level)][str(layer)], dtype=np.float32)
            for level in levels for layer in EXIT_LAYERS
        })
        temporary_predictions.replace(prediction_path)
        json_write(selection_path, packet)
    else:
        packet["selection_artifact_sha256"] = sha256(selection_path)
        packet["selection_artifact"] = str(selection_path)
        packet["review_approval_sha256"] = sha256(base / "review_approval.json")
        json_write(base / "test_metrics.json", packet)
    return packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "anchor", "select", "test"), required=True)
    parser.add_argument("--mode", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/synthetic_trained_heads/factory_ofdma_v2.yml")
    parser.add_argument("--task", choices=("nrpos-outdoor", "interference", "jamming"), required=True)
    parser.add_argument("--axis", choices=("antenna", "time"), required=True)
    parser.add_argument("--method", choices=("p0", "p2", "both"), default="both")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--encoder-batch-size", type=int, default=1)
    parser.add_argument("--head-batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(2)
    profile = yaml.safe_load(args.profile.read_text())
    manifest_path = args.manifest or args.master.parent / "split_manifest.json"
    split = json.loads(manifest_path.read_text())
    if args.mode == "formal":
        generation_record_path = args.master.parent / "generation_record.json"
        if not generation_record_path.is_file():
            parser.error("formal master is missing its locked generation_record.json")
        generation_record = json.loads(generation_record_path.read_text())
        locked_plan = generation_record.get("count_plan") or {}
        if args.task not in locked_plan.get("passed_tasks", []):
            parser.error(f"task {args.task} did not pass the pilot gates and is blocked from formal runs")
    split["_manifest_path"] = str(manifest_path.resolve())
    split["_manifest_sha256"] = sha256(manifest_path)
    if args.mode == "pilot" and (not split.get("pilot") or "test" in split["realization_ids"]):
        parser.error("pilot mode requires a pilot manifest with no test split")
    if args.mode == "formal" and split.get("pilot"):
        parser.error("formal mode requires the locked formal manifest")
    if args.stage == "test" and split.get("pilot"):
        parser.error("pilot dataset has no test split")
    methods = ("p0", "p2") if args.method == "both" else (args.method,)
    with h5py.File(args.master, "r") as h5:
        if h5.attrs.get("profile_sha256") != sha256(args.profile):
            parser.error("master profile hash does not match the supplied profile")
        id_map = index_lookup(h5)
        for split_ids in split["realization_ids"].values():
            missing = [value for value in split_ids if value not in id_map]
            if missing:
                parser.error(f"manifest references missing master IDs: {missing[:3]}")
        for method in methods:
            output_dir = ROOT / profile["outputs"]["run_root"] / args.mode / args.axis / args.task / method
            if args.stage == "train":
                if (output_dir / "training_record.json").exists():
                    parser.error(f"training record already exists; artifacts are write-once: {output_dir}")
                output_dir.mkdir(parents=True, exist_ok=True)
                save_and_train_heads(task=args.task, axis=args.axis, method=method, h5=h5, split=split, id_map=id_map, profile=profile, device=device, output_dir=output_dir, encoder_batch_size=args.encoder_batch_size, head_batch_size=args.head_batch_size)
            elif args.stage == "anchor":
                evaluate_anchor(task=args.task, axis=args.axis, method=method, h5=h5, split=split, id_map=id_map, profile=profile, device=device, encoder_batch_size=args.encoder_batch_size)
            else:
                evaluate_stage(task=args.task, axis=args.axis, method=method, stage=args.stage, h5=h5, split=split, id_map=id_map, profile=profile, device=device, encoder_batch_size=args.encoder_batch_size)


if __name__ == "__main__":
    main()
