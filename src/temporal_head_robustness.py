"""Frozen-head evaluation under controlled temporal-prefix availability."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from models import ExitHead
from representation_trajectories import LAYERS, METHODS, TEMPORAL_LEVELS, temporal_token_indices

HEAD_SEEDS = (2025, 2026, 2027, 2028, 2029)
COMPARISONS = (("p1", "p0"), ("p2", "p0"), ("p2_reverse", "p0"), ("p2", "p1"), ("p2", "p2_reverse"))
METHOD_LABELS = {"p0": "P0", "p1": "P1", "p2": "P2", "p2_reverse": "P2-Reverse"}
METHOD_COLORS = {"p0": "#277da1", "p1": "#f9c74f", "p2": "#f94144", "p2_reverse": "#7209b7"}
EMBED_DIM = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_indices(indices: Sequence[int] | np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def validate_head_packet(
    packet_path: Path,
    *,
    method: str,
    head_seed: int,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    split_manifest: Path,
    split_sha256: str,
    task: str = "interference",
    num_classes: int = 2,
) -> dict[int, Path]:
    """Validate selection-only packet metadata and return its four head files."""
    packet = read_json(packet_path)
    if packet.get("method") != method or int(packet.get("pretrain_seed", -1)) != 0:
        raise ValueError(f"{packet_path} must describe method={method}, pretrain_seed=0")
    if int(packet.get("head_seed", -1)) != int(head_seed):
        raise ValueError(f"{packet_path} has the wrong head seed")
    recorded_checkpoint = Path(str(packet.get("checkpoint", "")))
    if not recorded_checkpoint.is_file() or sha256_file(recorded_checkpoint) != checkpoint_sha256:
        raise ValueError(f"{packet_path} was not produced from the trajectory checkpoint")
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError(f"trajectory checkpoint hash changed: {checkpoint_path}")
    if packet.get("dataset_suite") != "primary":
        raise ValueError(f"{packet_path} is not a primary-suite head packet")
    protocol = packet.get("protocol", {})
    if protocol.get("selection_uses") != "validation_only" or protocol.get("test_evaluation") is not False:
        raise ValueError(f"{packet_path} must be produced by --selection-only")
    early = protocol.get("head_early_stopping", {})
    if int(protocol.get("head_max_epochs", -1)) != 200:
        raise ValueError(f"{packet_path} must use the 200-epoch head-training budget")
    if (early.get("enabled") is not True or int(early.get("patience", -1)) != 20
            or float(early.get("min_delta", -1.0)) != 0.0):
        raise ValueError(f"{packet_path} must use validation early stopping with patience 20 and min_delta 0")
    limits = protocol.get("batch_limits", {})
    if any(limits.get(key) is not None for key in ("train", "validation", "test")):
        raise ValueError(f"{packet_path} must be a full-data head run, not a batch-limited smoke run")
    recorded_split = Path(str(packet.get("split_manifest", "")))
    if not recorded_split.is_file() or sha256_file(recorded_split) != split_sha256:
        raise ValueError(f"{packet_path} did not use the canonical split manifest")
    if sha256_file(split_manifest) != split_sha256:
        raise ValueError(f"canonical split hash changed: {split_manifest}")
    task_record = packet.get("tasks", {}).get(task)
    if not isinstance(task_record, dict):
        raise ValueError(f"{packet_path} has no {task} task")
    spec = task_record.get("spec", {})
    if spec.get("target_type") != "classification" or int(spec.get("num_outputs", -1)) != int(num_classes):
        raise ValueError(f"{packet_path} must describe {task} classification with {num_classes} outputs")
    exits = task_record.get("exits", {})
    if set(exits) != {str(layer) for layer in LAYERS}:
        raise ValueError(f"{packet_path} must contain exactly exits {LAYERS}")
    head_paths: dict[int, Path] = {}
    for layer in LAYERS:
        item = exits[str(layer)]
        head_path = Path(item.get("head_path", ""))
        if not head_path.is_file():
            raise FileNotFoundError(f"missing L{layer} head checkpoint: {head_path}")
        saved = torch.load(head_path, map_location="cpu", weights_only=False)
        if not isinstance(saved, Mapping) or not isinstance(saved.get("head"), Mapping):
            raise ValueError(f"L{layer} head checkpoint is missing its head state")
        head = ExitHead(EMBED_DIM, num_classes)
        try:
            head.load_state_dict(saved["head"], strict=True)
        except RuntimeError as exc:
            raise ValueError(f"L{layer} head must have LayerNorm(256) and Linear(256, 2)") from exc
        head_paths[layer] = head_path
    return head_paths


def load_exit_heads(head_paths: Mapping[int, Path], *, device: torch.device, num_classes: int = 2) -> dict[int, nn.Module]:
    heads: dict[int, nn.Module] = {}
    for layer in LAYERS:
        payload = torch.load(head_paths[layer], map_location="cpu", weights_only=False)
        head = ExitHead(EMBED_DIM, num_classes)
        head.load_state_dict(payload["head"], strict=True)
        head.to(device).eval()
        heads[layer] = head
    return heads


def assert_cached_feature_parity(actual: np.ndarray | torch.Tensor, cached: np.ndarray | torch.Tensor,
                                  *, max_relative_l2: float = 2e-3, min_cosine: float = 0.99999) -> None:
    actual_tensor = torch.as_tensor(actual).detach().cpu()
    cached_tensor = torch.as_tensor(cached).detach().cpu()
    if actual_tensor.shape != cached_tensor.shape or actual_tensor.shape[-1] != EMBED_DIM:
        raise AssertionError(
            f"cached/runtime feature shape mismatch: {tuple(actual_tensor.shape)} vs {tuple(cached_tensor.shape)}"
        )
    actual_flat = actual_tensor.reshape(-1, EMBED_DIM).to(torch.float64)
    cached_flat = cached_tensor.reshape(-1, EMBED_DIM).to(torch.float64)
    relative_l2 = torch.linalg.vector_norm(actual_flat - cached_flat, dim=-1) / torch.linalg.vector_norm(
        cached_flat, dim=-1
    ).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(actual_flat, cached_flat, dim=-1)
    worst_relative_l2 = float(relative_l2.max())
    worst_cosine = float(cosine.min())
    if worst_relative_l2 > max_relative_l2 or worst_cosine < min_cosine:
        raise AssertionError(
            "cached/runtime feature parity failed: "
            f"max_relative_l2={worst_relative_l2:.6g} (limit {max_relative_l2}), "
            f"min_cosine={worst_cosine:.9f} (limit {min_cosine})"
        )


def assert_full_input_logit_parity(masked_logits: torch.Tensor, full_logits: torch.Tensor,
                                   *, rtol: float = 1e-5, atol: float = 1e-6) -> None:
    torch.testing.assert_close(masked_logits, full_logits, rtol=rtol, atol=atol)


def evaluate_temporal_prefixes(
    encoder: nn.Module,
    heads_by_layer_seed: Mapping[int, Mapping[int, nn.Module]],
    loader,
    *,
    indices: np.ndarray,
    cached_features: np.ndarray,
    device: torch.device,
    inference_repeats: int = 5,
    parity_check: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return probabilities/predictions shaped [H,R,L,T,N,(2)/()]."""
    head_seeds = tuple(next(iter(heads_by_layer_seed.values())).keys())
    n = len(indices)
    first_head = next(iter(next(iter(heads_by_layer_seed.values())).values()))
    num_classes = int(first_head.projection.out_features)
    if any(int(head.projection.out_features) != num_classes
           for layer_heads in heads_by_layer_seed.values() for head in layer_heads.values()):
        raise ValueError("all temporal-evaluation heads must have the same output class count")
    probabilities = np.empty((len(head_seeds), inference_repeats, len(LAYERS), len(TEMPORAL_LEVELS), n, num_classes),
                             dtype=np.float32)
    predictions = np.empty((len(head_seeds), inference_repeats, len(LAYERS), len(TEMPORAL_LEVELS), n),
                           dtype=np.int64)
    encoder.eval()
    for layer_heads in heads_by_layer_seed.values():
        for head in layer_heads.values():
            head.eval()
    for repeat in range(inference_repeats):
        seen: list[int] = []
        offset = 0
        with torch.inference_mode():
            for batch in loader:
                samples, source_indices = batch[:2]
                samples = samples.to(device=device, dtype=torch.float32, non_blocking=True)
                source_indices = np.asarray(source_indices, dtype=np.int64)
                batch_size = len(source_indices)
                if not np.array_equal(source_indices, indices[offset:offset + batch_size]):
                    raise ValueError("test loader order differs from the persisted trajectory test indices")
                seen.extend(source_indices.tolist())
                for ti, columns in enumerate(TEMPORAL_LEVELS):
                    keep = temporal_token_indices(columns).to(device).unsqueeze(0).expand(batch_size, -1)
                    outputs = encoder(samples, masks=[keep], return_layers=LAYERS, normalize_exits=True)
                    for li, layer in enumerate(LAYERS):
                        tokens = outputs[layer]
                        pooled = tokens.mean(dim=1)
                        if repeat == 0 and parity_check:
                            assert_cached_feature_parity(pooled, cached_features[offset:offset + batch_size, ti, li])
                        for hi, head_seed in enumerate(head_seeds):
                            head = heads_by_layer_seed[layer][head_seed]
                            logits = head(tokens)
                            if columns == 14 and parity_check and repeat == 0 and offset == 0:
                                reference_tokens = encoder.forward_to_layer(samples, layer)
                                assert_full_input_logit_parity(logits, head(reference_tokens))
                            probs = torch.softmax(logits, dim=-1)
                            probabilities[hi, repeat, li, ti, offset:offset + batch_size] = probs.cpu().numpy()
                            predictions[hi, repeat, li, ti, offset:offset + batch_size] = probs.argmax(dim=-1).cpu().numpy()
                offset += batch_size
        if offset != n or not np.array_equal(np.asarray(seen, dtype=np.int64), indices):
            raise ValueError("inference did not cover the complete ordered test split")
    disagreement = np.any(predictions != predictions[:, :1], axis=1)
    if np.any(disagreement):
        count = int(disagreement.sum())
        raise RuntimeError(f"prediction instability across repeated inference: {count} head/layer/prefix/sample cells")
    return probabilities, predictions


def classification_metrics(predictions: np.ndarray, labels: np.ndarray,
                            num_classes: int | None = None) -> dict[str, float]:
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have identical shape")
    num_classes = int(labels.max()) + 1 if num_classes is None and labels.size else int(num_classes or 0)
    if num_classes < 2 or not np.array_equal(np.unique(labels), np.arange(num_classes)):
        raise ValueError("labels must contain every contiguous class ID")
    class_accuracy = {}
    for cls in range(num_classes):
        select = labels == cls
        class_accuracy[cls] = float(np.mean(predictions[select] == cls))
    result = {"accuracy": float(np.mean(predictions == labels)),
              "macro_accuracy": float(np.mean(list(class_accuracy.values())))}
    result.update({f"class{cls}_accuracy": accuracy for cls, accuracy in class_accuracy.items()})
    return result


def build_metric_tables(probabilities: np.ndarray, predictions: np.ndarray, labels: np.ndarray,
                        head_seeds: Sequence[int], num_classes: int | None = None
                        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return _build_metric_tables_impl(probabilities, predictions, labels, head_seeds, num_classes)


def hierarchical_paired_bootstrap(
    predictions: np.ndarray,
    labels: np.ndarray,
    head_seeds: Sequence[int],
    *,
    replicates: int = 2000,
    seed: int = 2025,
    point_estimates: Mapping[tuple[str, int, int, str], float] | None = None,
    num_classes: int | None = None,
) -> list[dict[str, Any]]:
    return _hierarchical_paired_bootstrap_impl(predictions, labels, head_seeds,
                                               replicates=replicates, seed=seed,
                                               point_estimates=point_estimates, num_classes=num_classes)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

def _build_metric_tables_impl(probabilities: np.ndarray, predictions: np.ndarray, labels: np.ndarray,
                              head_seeds: Sequence[int], num_classes: int | None = None
                              ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create raw-repeat, per-head-seed and aggregate metrics."""
    probabilities = np.asarray(probabilities)
    predictions = np.asarray(predictions)
    labels = np.asarray(labels, dtype=np.int64)
    num_classes = int(labels.max()) + 1 if num_classes is None and labels.size else int(num_classes or 0)
    if num_classes < 2 or not np.array_equal(np.unique(labels), np.arange(num_classes)):
        raise ValueError("metric labels must include all contiguous classes")
    expected_prob_shape = (len(METHODS), len(head_seeds), predictions.shape[2],
                           len(LAYERS), len(TEMPORAL_LEVELS), len(labels), num_classes)
    if predictions.ndim != 6 or predictions.shape[:2] != (len(METHODS), len(head_seeds)):
        raise ValueError("predictions must have shape [M,H,R,L,T,N]")
    if probabilities.shape != expected_prob_shape or predictions.shape[3:6] != (
            len(LAYERS), len(TEMPORAL_LEVELS), len(labels)):
        raise ValueError("probability/prediction shapes do not match requested evaluation dimensions")
    if predictions.shape[2] < 1:
        raise ValueError("at least one inference repeat is required")

    raw_rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    repeat_scores: dict[tuple[str, int, int, int, int], dict[str, float]] = {}
    repeat_disagreement: dict[tuple[str, int, int, int, int], tuple[int, float]] = {}
    repeats = predictions.shape[2]
    for mi, method in enumerate(METHODS):
        for hi, head_seed in enumerate(head_seeds):
            for li, layer in enumerate(LAYERS):
                for ti, temporal in enumerate(TEMPORAL_LEVELS):
                    baseline = predictions[mi, hi, 0, li, ti]
                    for ri in range(repeats):
                        pred = predictions[mi, hi, ri, li, ti]
                        scores = classification_metrics(pred, labels, num_classes)
                        prob_delta = float(np.max(np.abs(
                            probabilities[mi, hi, ri, li, ti] - probabilities[mi, hi, 0, li, ti])))
                        disagree = int(np.count_nonzero(pred != baseline))
                        raw_rows.append({
                            "method": method, "head_seed": int(head_seed), "repeat": ri,
                            "layer": layer, "temporal_columns": temporal, **scores,
                            "prediction_disagreement_from_repeat0": disagree,
                            "max_probability_delta_from_repeat0": prob_delta,
                        })
                        repeat_scores[(method, int(head_seed), layer, temporal, ri)] = scores
                        repeat_disagreement[(method, int(head_seed), layer, temporal, ri)] = (disagree, prob_delta)

    seed_metrics: dict[tuple[str, int, int, int], dict[str, float]] = {}
    for method in METHODS:
        for head_seed in head_seeds:
            for layer in LAYERS:
                full_scores = [
                    repeat_scores[(method, int(head_seed), layer, 14, ri)]["macro_accuracy"]
                    for ri in range(repeats)
                ]
                for temporal in TEMPORAL_LEVELS:
                    values: dict[str, list[float]] = {
                        metric: [repeat_scores[(method, int(head_seed), layer, temporal, ri)][metric]
                                 for ri in range(repeats)]
                        for metric in ("macro_accuracy", "accuracy", *[f"class{cls}_accuracy" for cls in range(num_classes)])
                    }
                    means = {key: float(np.mean(value)) for key, value in values.items()}
                    stds = {key: float(np.std(value, ddof=1)) if repeats > 1 else 0.0
                            for key, value in values.items()}
                    full = float(np.mean(full_scores))
                    metrics = {
                        "macro_accuracy_mean_over_repeats": means["macro_accuracy"],
                        "macro_accuracy_std_over_repeats": stds["macro_accuracy"],
                        "accuracy_mean_over_repeats": means["accuracy"],
                        "accuracy_std_over_repeats": stds["accuracy"],
                        **{f"class{cls}_accuracy": means[f"class{cls}_accuracy"] for cls in range(num_classes)},
                        "full_information_macro_accuracy": full,
                        "absolute_degradation": full - means["macro_accuracy"],
                        "relative_retention": means["macro_accuracy"] / max(full, 1e-12),
                    }
                    seed_metrics[(method, int(head_seed), layer, temporal)] = metrics
                    per_seed_rows.append({
                        "method": method, "head_seed": int(head_seed), "layer": layer,
                        "temporal_columns": temporal, **metrics,
                    })

    for method in METHODS:
        for layer in LAYERS:
            for temporal in TEMPORAL_LEVELS:
                rows = [seed_metrics[(method, int(seed), layer, temporal)] for seed in head_seeds]
                accuracy = np.asarray([row["macro_accuracy_mean_over_repeats"] for row in rows])
                degradation = np.asarray([row["absolute_degradation"] for row in rows])
                disagreements = [repeat_disagreement[(method, int(seed), layer, temporal, ri)][0]
                                 for seed in head_seeds for ri in range(repeats)]
                aggregate_rows.append({
                    "method": method, "layer": layer, "temporal_columns": temporal,
                    "mean_macro_accuracy": float(accuracy.mean()),
                    "std_macro_accuracy_across_head_seeds": float(accuracy.std(ddof=1)),
                    "min_macro_accuracy": float(accuracy.min()), "max_macro_accuracy": float(accuracy.max()),
                    "mean_accuracy": float(np.mean([row["accuracy_mean_over_repeats"] for row in rows])),
                    **{f"mean_class{cls}_accuracy": float(np.mean([row[f"class{cls}_accuracy"] for row in rows]))
                       for cls in range(num_classes)},
                    "mean_absolute_degradation": float(degradation.mean()),
                    "std_absolute_degradation_across_head_seeds": float(degradation.std(ddof=1)),
                    "mean_repeat_std": float(np.mean([row["macro_accuracy_std_over_repeats"] for row in rows])),
                    "max_repeat_prediction_disagreement": int(max(disagreements, default=0)),
                })
    return raw_rows, per_seed_rows, aggregate_rows


def _hierarchical_paired_bootstrap_impl(
    predictions: np.ndarray,
    labels: np.ndarray,
    head_seeds: Sequence[int],
    *,
    replicates: int,
    seed: int,
    point_estimates: Mapping[tuple[str, int, int, str], float] | None,
    num_classes: int | None = None,
) -> list[dict[str, Any]]:
    """Resample test examples within class and head seeds, paired across methods."""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels, dtype=np.int64)
    num_classes = int(labels.max()) + 1 if num_classes is None and labels.size else int(num_classes or 0)
    if num_classes < 2 or not np.array_equal(np.unique(labels), np.arange(num_classes)):
        raise ValueError("bootstrap labels must include all contiguous classes")
    if predictions.ndim == 6:
        predictions = predictions[:, :, 0]
    if predictions.ndim != 5 or predictions.shape[:2] != (len(METHODS), len(head_seeds)):
        raise ValueError("predictions must have shape [M,H,L,T,N] or [M,H,R,L,T,N]")
    if predictions.shape[2:4] != (len(LAYERS), len(TEMPORAL_LEVELS)) or predictions.shape[-1] != len(labels):
        raise ValueError("prediction dimensions do not match layers, temporal levels, and labels")
    if replicates < 1 or len(head_seeds) < 2:
        raise ValueError("bootstrap requires positive replicates and at least two head seeds")
    class_positions = [np.flatnonzero(labels == cls) for cls in range(num_classes)]
    if any(len(values) == 0 for values in class_positions):
        raise ValueError("paired bootstrap requires every class")

    rng = np.random.default_rng(seed)
    head_draws = rng.integers(0, len(head_seeds), size=(replicates, len(head_seeds)))
    class_draws = [rng.choice(values, size=(replicates, len(values)), replace=True)
                   for values in class_positions]
    # [replicate, method, layer, temporal, metric=(macro accuracy, full-minus-current drop)]
    distributions = np.empty((replicates, len(METHODS), len(LAYERS), len(TEMPORAL_LEVELS), 2), dtype=np.float64)
    full_ti = TEMPORAL_LEVELS.index(14)
    for mi in range(len(METHODS)):
        for li in range(len(LAYERS)):
            for ti in range(len(TEMPORAL_LEVELS)):
                class_accuracies: list[np.ndarray] = []
                class_drops: list[np.ndarray] = []
                for cls in range(num_classes):
                    sample_ids = class_draws[cls]
                    current = predictions[mi, head_draws[:, :, None], li, ti, sample_ids[:, None, :]]
                    current_correct = (current == cls).mean(axis=-1)
                    class_accuracies.append(current_correct.mean(axis=-1))
                    full = predictions[mi, head_draws[:, :, None], li, full_ti, sample_ids[:, None, :]]
                    full_correct = (full == cls).mean(axis=-1)
                    class_drops.append((full_correct - current_correct).mean(axis=-1))
                distributions[:, mi, li, ti, 0] = np.mean(class_accuracies, axis=0)
                distributions[:, mi, li, ti, 1] = np.mean(class_drops, axis=0)

    metric_names = ("macro_accuracy", "absolute_degradation")
    if point_estimates is None:
        point_estimates = {}
        for mi, method in enumerate(METHODS):
            for li, layer in enumerate(LAYERS):
                for ti, temporal in enumerate(TEMPORAL_LEVELS):
                    for ki, metric in enumerate(metric_names):
                        point_estimates[(method, layer, temporal, metric)] = float(
                            distributions[:, mi, li, ti, ki].mean())
    rows: list[dict[str, Any]] = []
    for li, layer in enumerate(LAYERS):
        for ti, temporal in enumerate(TEMPORAL_LEVELS):
            for ki, metric in enumerate(metric_names):
                values = {method: distributions[:, mi, li, ti, ki] for mi, method in enumerate(METHODS)}
                for method in METHODS:
                    estimate = point_estimates[(method, layer, temporal, metric)]
                    low, high = np.quantile(values[method], [0.025, 0.975])
                    rows.append({
                        "record_type": "method", "method": method, "reference_method": "",
                        "layer": layer, "temporal_columns": temporal, "metric": metric, "class_id": "",
                        "estimate": estimate, "ci_low": float(low), "ci_high": float(high),
                        "difference": "", "difference_low": "", "difference_high": "", "excludes_zero": "",
                        "bootstrap_replicates": replicates, "bootstrap_seed": seed,
                    })
                for method, reference in COMPARISONS:
                    difference = values[method] - values[reference]
                    low, high = np.quantile(difference, [0.025, 0.975])
                    estimate_difference = (point_estimates[(method, layer, temporal, metric)]
                                           - point_estimates[(reference, layer, temporal, metric)])
                    rows.append({
                        "record_type": "comparison", "method": method, "reference_method": reference,
                        "layer": layer, "temporal_columns": temporal, "metric": metric, "class_id": "",
                        "estimate": point_estimates[(method, layer, temporal, metric)],
                        "ci_low": "", "ci_high": "", "difference": estimate_difference,
                        "difference_low": float(low), "difference_high": float(high),
                        "excludes_zero": bool(low > 0 or high < 0),
                        "bootstrap_replicates": replicates, "bootstrap_seed": seed,
                    })
    return rows
def join_geometry_metrics(geometry_rows: Sequence[Mapping[str, Any]],
                         aggregate_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    geometry = {(str(row["method"]), int(row["layer"]), int(row["temporal_columns"])): row
                for row in geometry_rows}
    output = []
    for row in aggregate_rows:
        key = (str(row["method"]), int(row["layer"]), int(row["temporal_columns"]))
        if key not in geometry:
            raise ValueError(f"missing geometry metrics for {key}")
        geo = geometry[key]
        output.append({
            "method": key[0], "layer": key[1], "temporal_columns": key[2],
            "class_separation_ratio": float(geo["class_separation_ratio"]),
            "normalized_distance": float(geo["normalized_distance"]),
            "cosine_similarity": float(geo["cosine_similarity"]),
            "mean_macro_accuracy": float(row["mean_macro_accuracy"]),
            "mean_absolute_degradation": float(row["mean_absolute_degradation"]),
        })
    if len(output) != len(METHODS) * len(LAYERS) * len(TEMPORAL_LEVELS):
        raise ValueError("geometry/function join is incomplete")
    return output


def write_figures(aggregate_rows: Sequence[Mapping[str, Any]], report_dir: Path,
                  *, task: str = "interference", prefix_semantics: str = "temporal") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(row["method"], int(row["layer"]), int(row["temporal_columns"])): row for row in aggregate_rows}
    levels = np.asarray(TEMPORAL_LEVELS)
    prefix_label = "temporal columns" if prefix_semantics == "temporal" else "input columns"
    prefix_name = "temporal-prefix" if prefix_semantics == "temporal" else "column-prefix"
    for metric, std_metric, filename, ylabel, title in (
        ("mean_macro_accuracy", "std_macro_accuracy_across_head_seeds",
         "temporal_head_robustness", "Macro accuracy", f"Frozen full-information heads under {prefix_name} availability"),
        ("mean_absolute_degradation", "std_absolute_degradation_across_head_seeds",
         "temporal_head_degradation", "Accuracy loss from Tp14", f"Degradation relative to full {prefix_label}"),
    ):
        fig, axes = plt.subplots(1, len(LAYERS), figsize=(18, 4.5), sharey=True, constrained_layout=True)
        for li, layer in enumerate(LAYERS):
            ax = axes[li]
            for method in METHODS:
                rows = [lookup[(method, layer, temporal)] for temporal in TEMPORAL_LEVELS]
                values = np.asarray([row[metric] for row in rows], dtype=float)
                errors = np.asarray([row[std_metric] for row in rows], dtype=float)
                ax.errorbar(levels, values, yerr=errors, color=METHOD_COLORS[method],
                            marker="o", capsize=3, linewidth=1.8, label=METHOD_LABELS[method])
            ax.set_title(f"L{layer}")
            ax.set_xticks(levels, [f"P{value}" for value in levels])
            ax.set_xlabel(f"Visible {prefix_label}")
            ax.grid(alpha=0.25)
            if li == 0:
                ax.set_ylabel(ylabel)
        axes[0].legend(fontsize=9)
        fig.suptitle(f"{task}: {title}")
        for ext in ("png", "pdf"):
            fig.savefig(report_dir / f"{filename}.{ext}", dpi=220)
        plt.close(fig)


def _format_ci(row: Mapping[str, Any], *, comparison: bool = False) -> str:
    if comparison:
        return (f'{float(row["difference"]):+.3f} '
                f'[{float(row["difference_low"]):+.3f}, {float(row["difference_high"]):+.3f}]')
    return f'{float(row["estimate"]):.3f} [{float(row["ci_low"]):.3f}, {float(row["ci_high"]):.3f}]'


def write_summary(path: Path, aggregate_rows: Sequence[Mapping[str, Any]],
                  bootstrap_rows: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any],
                  *, task: str = "interference") -> None:
    aggregate = {(row["method"], int(row["layer"]), int(row["temporal_columns"])): row
                 for row in aggregate_rows}
    bootstrap = {(row["record_type"], row["method"], row["reference_method"],
                  int(row["layer"]), int(row["temporal_columns"]), row["metric"]): row
                 for row in bootstrap_rows}
    semantics = provenance.get("prefix_semantics", "column")
    prefix_word = "temporal" if semantics == "temporal" else "input-column"
    counts = provenance.get("split", {}).get("class_counts", {})
    count_text = ", ".join(f"class {cls}: {n}" for cls, n in sorted(counts.items(), key=lambda item: int(item[0])))
    lines = [
        f"# Frozen-head robustness under {task}", "", "## Protocol", "",
        f"- Same full-information L2/L4/L6/L8 heads are evaluated under Tp2/Tp4/Tp8/Tp14 {prefix_word} prefixes.",
        "- Methods: P0, P1, P2, and P2-Reverse; each encoder remains frozen during evaluation.",
        "- P2-Reverse uses epoch 49 / step 11,300 from a 50-epoch run, so its comparison with 60-epoch methods is exploratory.",
        "- P2-Reverse source split provenance is unverifiable; extraction, head training, and evaluation use the canonical seed-2025 split.",
        f"- Test class counts: {count_text}.",
        "- Five head seeds are summarized with mean and standard deviation. Repeated inference passes are a determinism audit, not independent trials.",
        "- Paired hierarchical bootstrap resamples head seeds and test examples within each class. Uncertainty is conditional on pretraining seed 0.", "",
        f"## {'Tp4' if semantics == 'temporal' else 'P4'} paired method comparisons", "",
        "Differences are method minus reference; intervals are 95% hierarchical-bootstrap intervals for macro accuracy.", "",
        "| Layer | P1−P0 | P2−P0 | P2-Reverse−P0 | P2−P1 | P2−P2-Reverse |", "|---|---:|---:|---:|---:|---:|"]
    for layer in LAYERS:
        cells = [bootstrap[("comparison", method, reference, layer, 4, "macro_accuracy")]
                 for method, reference in COMPARISONS]
        rendered = [_format_ci(row, comparison=True) for row in cells]
        lines.append(f"| L{layer} | " + " | ".join(rendered) + " |")
    lines += ["", "Accuracy degradation is Tp14 macro accuracy minus current-prefix macro accuracy; Tp14 degradation is zero by construction.",
              "Repeated inference passes are not counted as independent evidence.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
