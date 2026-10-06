"""Matched multi-method representation trajectories for Phase 5 tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

LAYERS = (2, 4, 6, 8)
TEMPORAL_LEVELS = (2, 4, 8, 14)
GRID_SIZE = 14
EMBED_DIM = 256


METHODS = ("p0", "p1", "p2", "p2_reverse")
METHOD_LABELS = {"p0": "P0", "p1": "P1", "p2": "P2", "p2_reverse": "P2-Reverse"}
COMPARISONS = (("p1", "p0"), ("p2", "p0"), ("p2_reverse", "p0"), ("p2", "p1"), ("p2", "p2_reverse"))
HIERARCHY = {"p0": (False, None), "p1": (True, "same_scale"), "p2": (True, "mask_relative_fine_to_coarse"), "p2_reverse": (True, "mask_relative_coarse_to_fine")}
CHECKPOINT_HORIZONS = {"p0": (59, 13560), "p1": (59, 13560), "p2": (59, 13560), "p2_reverse": (49, 11300)}
TASK_SEMANTICS = {
    "interference": {"prefix_axis": "width", "prefix_semantics": "temporal"},
    "human-activity": {"prefix_axis": "width", "prefix_semantics": "temporal"},
    "rfs": {"prefix_axis": "width", "prefix_semantics": "column"},
    "jamming": {"prefix_axis": "width", "prefix_semantics": "column"},
}
TASK_LABELS = {
    "interference": "Interference", "human-activity": "Human activity sensing",
    "rfs": "RF signal classification", "jamming": "Jamming classification",
}


def canonical_mask_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    masks = config.get("masks", {})
    block = masks.get("multi_block", {})
    return {"type": masks.get("type"), "allow_overlap": bool(block.get("allow_overlap", False)),
            "aspect_ratio_type": block.get("aspect_ratio_type"), "aspect_ratio": list(block.get("aspect_ratio", [])),
            "enc_mask_scale": list(block.get("enc_mask_scale", [])), "min_keep": int(block.get("min_keep", -1)),
            "num_enc_masks": int(block.get("num_enc_masks", -1)), "num_pred_masks": int(block.get("num_pred_masks", -1)),
            "pred_mask_scale": list(block.get("pred_mask_scale", []))}

class IndexedSamples(Dataset):
    """Return samples with source indices so persisted split order is auditable."""

    def __init__(self, base: Dataset, indices: list[int]):
        self.base = base
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position: int):
        index = self.indices[position]
        sample = self.base[index][0]
        return sample, index


class IndexedTaskSamples(Dataset):
    """Return canonical dataset samples, mapped labels, and original indices."""
    def __init__(self, base: Dataset, indices: list[int]):
        self.base = base
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position: int):
        index = self.indices[position]
        sample, label = self.base[index][:2]
        return sample, label, index


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_indices(indices: list[int]) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def task_class_names(dataset: Dataset, num_classes: int) -> list[str]:
    """Use canonical cache label names when their ordering matches the task IDs."""
    names = getattr(dataset, "labels", None)
    if names is None and hasattr(dataset, "dataset"):
        names = getattr(dataset.dataset, "labels", None)
    if names is not None and len(names) == int(num_classes):
        return [str(name) for name in names]
    return [f"Class {cls}" for cls in range(int(num_classes))]


def temporal_token_indices(columns: int, *, grid_size: int = GRID_SIZE) -> torch.Tensor:
    """Row-major indices for leading columns on an unchanged positional grid."""
    columns = int(columns)
    if not 1 <= columns <= grid_size:
        raise ValueError(f"temporal columns must be in [1, {grid_size}], got {columns}")
    return torch.tensor(
        [row * grid_size + column for row in range(grid_size) for column in range(columns)],
        dtype=torch.long,
    )


def validate_checkpoint_metadata(
    metadata: Mapping[str, Any], *, method: str, expected_epoch: int | None = None, expected_step: int | None = None
) -> dict[str, Any]:
    """Check matched checkpoint provenance for a canonical method."""
    if method not in METHODS:
        raise ValueError(f"unsupported method {method!r}; expected one of {METHODS}")
    phase5 = metadata.get("phase5", {})
    horizon_epoch, horizon_step = CHECKPOINT_HORIZONS[method]
    expected_epoch = horizon_epoch if expected_epoch is None else expected_epoch
    expected_step = horizon_step if expected_step is None else expected_step
    config = metadata.get("config", {})
    masks = config.get("masks", {})
    block = masks.get("multi_block", {})
    hierarchy = config.get("hierarchy", {})
    expected_hierarchy, expected_mode = HIERARCHY[method]
    errors = []
    if metadata.get("seed") != 0 or phase5.get("seed") != 0:
        errors.append("checkpoint seed must be 0")
    if phase5.get("method") != method:
        errors.append(f"checkpoint method must be {method}")
    if int(phase5.get("epoch", expected_epoch)) != expected_epoch:
        errors.append(f"checkpoint metadata epoch must be {expected_epoch}")
    expected_max_epochs = 50 if method == "p2_reverse" else None
    if expected_max_epochs is not None and int(phase5.get("max_epochs", -1)) != expected_max_epochs:
        errors.append(f"{METHOD_LABELS[method]} checkpoint run must declare max_epochs={expected_max_epochs}")
    signature = canonical_mask_signature(config)
    expected_signature = {"type": "multi_block", "allow_overlap": False, "aspect_ratio_type": "span_time",
                          "aspect_ratio": [0.75, 1.5], "enc_mask_scale": [1.0, 1.0], "min_keep": 10,
                          "num_enc_masks": 1, "num_pred_masks": 6, "pred_mask_scale": [0.15, 0.2]}
    if signature != expected_signature:
        errors.append(f"mask signature must match canonical multi_block/span_time six-mask recipe: {signature}")
    if bool(hierarchy.get("enabled", False)) != expected_hierarchy:
        errors.append(f"hierarchy enabled must be {expected_hierarchy}")
    if expected_hierarchy and (hierarchy.get("mode") != expected_mode or tuple(hierarchy.get("layers", ())) != (2, 4, 6)):
        errors.append(f"{METHOD_LABELS[method]} hierarchy must be {expected_mode} at L2/L4/L6")
    if errors:
        raise ValueError(f"invalid {method} checkpoint metadata: " + "; ".join(errors))
    return {
        "method": method,
        "epoch": expected_epoch,
        "global_step": expected_step,
        "max_epochs": phase5.get("max_epochs"),
        "mask_type": masks["type"],
        "mask_aspect_ratio_type": block["aspect_ratio_type"],
        "num_pred_masks": int(block["num_pred_masks"]),
        "hierarchy_enabled": expected_hierarchy,
        "hierarchy_mode": hierarchy.get("mode"),
        "mask_signature": signature,
    }


def validate_source_metrics(path: Path, *, method: str, manifest_path: Path, task: str = "interference",
                            canonical_indices: Mapping[str, list[int]] | None = None,
                            allow_unverifiable_manifest: bool = False) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("method") != method or int(payload.get("pretrain_seed", -1)) != 0:
        raise ValueError(f"{path} must describe method={method}, pretrain_seed=0")
    if task not in payload.get("tasks", {}):
        raise ValueError(f"{path} has no {task} task")
    source_manifest = Path(payload.get("split_manifest", ""))
    if not source_manifest.is_absolute():
        relative_manifest = source_manifest
        source_manifest = _source_root(Path(path)) / relative_manifest
        if not source_manifest.is_file():
            repository_root = Path(__file__).resolve().parents[1]
            source_manifest = repository_root / relative_manifest
    if not source_manifest.is_file():
        if not allow_unverifiable_manifest:
            raise ValueError(f"{METHOD_LABELS[method]} source split manifest is unavailable: {source_manifest}")
        payload["_split_verification"] = {"status": "unverifiable", "recorded_path": str(source_manifest),
                                           "reason": "Source manifest is missing; analysis uses the canonical manifest for all extracted samples."}
        return payload
    source_payload = json.loads(source_manifest.read_text())
    source_indices = source_payload.get("tasks", {}).get(task, {}).get("indices", {})
    canonical = canonical_indices or json.loads(Path(manifest_path).read_text()).get("tasks", {}).get(task, {}).get("indices", {})
    for split in ("train", "validation", "test"):
        if [int(i) for i in source_indices.get(split, [])] != [int(i) for i in canonical.get(split, [])]:
            raise ValueError(f"{METHOD_LABELS[method]} source metrics {split} indices differ from canonical split")
    if int(source_payload.get("seed", -1)) != 2025:
        raise ValueError(f"{METHOD_LABELS[method]} source split must have seed 2025")
    payload["_split_verification"] = {"status": "verified", "manifest_path": str(source_manifest.resolve()),
                                       "indices_equal": True}
    return payload


def _source_root(metrics_path: Path) -> Path:
    for parent in metrics_path.resolve().parents:
        if parent.name == "runs":
            return parent.parent
    return metrics_path.resolve().parent


def load_task_split(manifest_path: Path, task_name: str, labels: np.ndarray,
                    num_classes: int) -> tuple[dict[str, list[int]], np.ndarray]:
    """Validate a primary v3 split and return exact persisted order plus canonical labels."""
    manifest = json.loads(Path(manifest_path).read_text())
    if int(manifest.get("schema_version", -1)) != 3 or int(manifest.get("seed", -1)) != 2025:
        raise ValueError("expected primary split manifest schema v3, seed 2025")
    task = manifest.get("tasks", {}).get(task_name)
    if not isinstance(task, dict):
        raise ValueError(f"split manifest is missing task {task_name!r}")
    mapped = np.asarray(labels, dtype=np.int64).reshape(-1)
    if mapped.size == 0 or not np.array_equal(np.unique(mapped), np.arange(num_classes)):
        raise ValueError(f"{task_name} dataset labels must cover contiguous IDs 0..{num_classes - 1}")
    if task.get("spec", {}).get("target_type", "classification") != "classification":
        raise ValueError(f"{task_name} is not a classification task")
    declared_classes = task.get("spec", {}).get("num_outputs")
    if declared_classes is not None and int(declared_classes) != int(num_classes):
        raise ValueError(f"{task_name} split class count differs from dataset TaskSpec")
    indices = {key: [int(i) for i in task["indices"][key]] for key in ("train", "validation", "test")}
    for split, selected in indices.items():
        if len(selected) != len(set(selected)) or (selected and (min(selected) < 0 or max(selected) >= len(mapped))):
            raise ValueError(f"invalid {split} indices in manifest")
        declared = task.get("counts", {}).get(split)
        if declared is not None and int(declared) != len(selected):
            raise ValueError(f"manifest {split} count does not match its indices")
    all_indices = [index for selected in indices.values() for index in selected]
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("persisted train/validation/test indices overlap")
    for split in ("validation", "test"):
        counts = np.bincount(mapped[indices[split]], minlength=num_classes)
        if np.any(counts == 0):
            raise ValueError(f"{task_name} {split} split must contain every class; got {counts.tolist()}")
    return indices, mapped


def load_split_and_labels(manifest_path: Path, raw_labels: np.ndarray) -> tuple[dict[str, list[int]], np.ndarray]:
    """Backward-compatible Interference split loader with its historical balance check."""
    raw_labels = np.asarray(raw_labels).reshape(-1)
    values = np.unique(raw_labels)
    label_map = {int(value): index for index, value in enumerate(values.tolist())}
    mapped = np.asarray([label_map[int(value)] for value in raw_labels], dtype=np.int64)
    indices, mapped = load_task_split(manifest_path, "interference", mapped, 2)
    for split in ("validation", "test"):
        counts = np.bincount(mapped[indices[split]], minlength=2)
        if not np.array_equal(counts, np.full(2, 700)):
            raise ValueError(f"Interference {split} must contain 700 samples per class; got {counts.tolist()}")
    return indices, mapped


def extract_features(encoder: torch.nn.Module, loader: DataLoader, labels_by_index: np.ndarray,
                     *, device: torch.device, limit: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract normalized mean-pooled vectors, [N,T,L,D], without gradients."""
    collected: list[np.ndarray] = []
    seen_indices: list[int] = []
    seen_labels: list[int] = []
    encoder.eval()
    with torch.inference_mode():
        for samples, source_indices in loader:
            if limit is not None:
                remain = limit - len(seen_indices)
                if remain <= 0:
                    break
                samples, source_indices = samples[:remain], source_indices[:remain]
            samples = samples.to(device=device, dtype=torch.float32, non_blocking=True)
            source_indices_np = source_indices.cpu().numpy().astype(np.int64)
            per_sample = []
            for temporal_columns in TEMPORAL_LEVELS:
                indices = temporal_token_indices(temporal_columns).to(device)
                keep = indices.unsqueeze(0).expand(samples.shape[0], -1)
                outputs = encoder(samples, masks=[keep], return_layers=LAYERS, normalize_exits=True)
                level = [outputs[layer].mean(dim=1).float().cpu().numpy() for layer in LAYERS]
                per_sample.append(np.stack(level, axis=1))
            collected.append(np.stack(per_sample, axis=1))
            seen_indices.extend(source_indices_np.tolist())
            seen_labels.extend(labels_by_index[source_indices_np].tolist())
    if not collected:
        raise RuntimeError("no samples were extracted")
    features = np.concatenate(collected, axis=0).astype(np.float32, copy=False)
    labels = np.asarray(seen_labels, dtype=np.int64)
    indices = np.asarray(seen_indices, dtype=np.int64)
    if features.shape != (len(indices), len(TEMPORAL_LEVELS), len(LAYERS), EMBED_DIM):
        raise ValueError(f"unexpected feature shape {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("encoder produced non-finite features")
    return features, labels, indices


def fit_pca(features: np.ndarray) -> PCA:
    """Fit a single 2D PCA over all temporal-level/layer validation points."""
    flat = np.asarray(features, dtype=np.float64).reshape(-1, features.shape[-1])
    return PCA(n_components=2, svd_solver="full").fit(flat)


def transform_pca(pca: PCA, features: np.ndarray) -> np.ndarray:
    shape = features.shape[:-1] + (2,)
    return pca.transform(np.asarray(features).reshape(-1, features.shape[-1])).reshape(shape)


def trajectory_metrics(features_by_method: Mapping[str, np.ndarray], labels: np.ndarray) -> list[dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    classes = np.unique(labels)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("labels must be contiguous class IDs starting at zero")
    rows: list[dict[str, Any]] = []
    for method, features in features_by_method.items():
        if features.shape[0] != len(labels):
            raise ValueError(f"{method} features do not align with labels")
        for li, layer in enumerate(LAYERS):
            full = features[:, -1, li, :].astype(np.float64)
            for ti, temporal in enumerate(TEMPORAL_LEVELS):
                current = features[:, ti, li, :].astype(np.float64)
                distances = np.linalg.norm(current - full, axis=1) / (np.linalg.norm(full, axis=1) + 1e-12)
                similarities = np.sum(current * full, axis=1) / (
                    np.linalg.norm(current, axis=1) * np.linalg.norm(full, axis=1) + 1e-12
                )
                per_class_distance = {int(cls): float(distances[labels == cls].mean()) for cls in classes}
                per_class_cosine = {int(cls): float(similarities[labels == cls].mean()) for cls in classes}
                centers = {int(cls): current[labels == cls].mean(axis=0) for cls in classes}
                radii = {int(cls): float(np.sqrt(np.mean(np.sum(
                    (current[labels == cls] - centers[int(cls)]) ** 2, axis=1)))) for cls in classes}
                pair_distances = [float(np.linalg.norm(centers[int(a)] - centers[int(b)]))
                                  for i, a in enumerate(classes) for b in classes[i + 1:]]
                separation = float(np.mean(pair_distances) / (np.mean(list(radii.values())) + 1e-12))
                row = {
                    "method": method, "layer": layer, "temporal_columns": temporal,
                    "normalized_distance": float(np.mean(list(per_class_distance.values()))),
                    "cosine_similarity": float(np.mean(list(per_class_cosine.values()))),
                    "class_separation_ratio": separation,
                }
                for cls in classes:
                    row[f"normalized_distance_class{cls}"] = per_class_distance[int(cls)]
                    row[f"cosine_similarity_class{cls}"] = per_class_cosine[int(cls)]
                    row[f"class_radius_class{cls}"] = radii[int(cls)]
                rows.append(row)
    return rows


def paired_bootstrap(features_by_method: Mapping[str, np.ndarray], labels: np.ndarray,
                     *, replicates: int = 2000, seed: int = 2025) -> list[dict[str, Any]]:
    """Fast class-stratified paired bootstrap; all methods reuse each sampled identity draw."""
    if set(features_by_method) != set(METHODS):
        raise ValueError(f"paired bootstrap requires exactly {METHODS}")
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    classes = np.unique(labels)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("labels must be contiguous class IDs starting at zero")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == cls) for cls in classes]
    if any(not len(group) for group in groups):
        raise ValueError("every class must be represented in the bootstrap sample")
    # Precompute per-example trajectory distances/cosines and squared feature norms.
    terms = {}
    for method, features in features_by_method.items():
        values = np.asarray(features, dtype=np.float64)
        if values.shape[0] != len(labels):
            raise ValueError(f"{method} features do not align with labels")
        distances = np.empty((len(labels), len(TEMPORAL_LEVELS), len(LAYERS)), dtype=np.float64)
        cosines = np.empty_like(distances)
        squared_norms = np.sum(values * values, axis=-1)
        for li in range(len(LAYERS)):
            full = values[:, -1, li]
            full_norm = np.linalg.norm(full, axis=1)
            for ti in range(len(TEMPORAL_LEVELS)):
                current = values[:, ti, li]
                distances[:, ti, li] = np.linalg.norm(current - full, axis=1) / (full_norm + 1e-12)
                cosines[:, ti, li] = np.sum(current * full, axis=1) / (
                    np.linalg.norm(current, axis=1) * full_norm + 1e-12)
        terms[method] = {"features": values, "distances": distances, "cosines": cosines,
                         "squared_norms": squared_norms}

    keys = [(metric, cls) for metric in ("normalized_distance", "cosine_similarity")
            for cls in [None, *[int(c) for c in classes]]]
    keys += [("class_separation_ratio", None)]
    boot = {(kind, method, reference, layer, temporal, metric, cls): np.empty(replicates, dtype=np.float64)
            for kind in ("method", "comparison")
            for method, reference in ([(m, "") for m in METHODS] if kind == "method" else list(COMPARISONS))
            for layer in LAYERS for temporal in TEMPORAL_LEVELS for metric, cls in keys}
    for replicate in range(replicates):
        sampled_by_class = [rng.choice(group, size=len(group), replace=True) for group in groups]
        sampled_all = np.concatenate(sampled_by_class)
        stats: dict[tuple[str, int, int], dict[str, float]] = {}
        for method in METHODS:
            record = terms[method]
            feature = record["features"]
            for li, layer in enumerate(LAYERS):
                for ti, temporal in enumerate(TEMPORAL_LEVELS):
                    class_distance = [float(record["distances"][draw, ti, li].mean()) for draw in sampled_by_class]
                    class_cosine = [float(record["cosines"][draw, ti, li].mean()) for draw in sampled_by_class]
                    centers = np.stack([feature[draw, ti, li].mean(axis=0) for draw in sampled_by_class])
                    radii = []
                    for cls_index, draw in enumerate(sampled_by_class):
                        mean_sq = float(record["squared_norms"][draw, ti, li].mean())
                        center_sq = float(np.dot(centers[cls_index], centers[cls_index]))
                        radii.append(float(np.sqrt(max(0.0, mean_sq - center_sq))))
                    if len(classes) == 2:
                        # Preserve the historical binary equation exactly.
                        pair_mean = float(np.linalg.norm(centers[0] - centers[1]))
                    else:
                        gram = centers @ centers.T
                        norms = np.diag(gram)
                        pairwise = np.sqrt(np.maximum(norms[:, None] + norms[None, :] - 2 * gram, 0.0))
                        pair_mean = float(pairwise[np.triu_indices(len(classes), k=1)].mean())
                    stats[(method, layer, temporal)] = {
                        "normalized_distance": float(np.mean(class_distance)),
                        "cosine_similarity": float(np.mean(class_cosine)),
                        "class_separation_ratio": pair_mean / (float(np.mean(radii)) + 1e-12),
                    }
                    for cls_index, cls in enumerate(classes):
                        stats[(method, layer, temporal)][f"normalized_distance_class{cls}"] = class_distance[cls_index]
                        stats[(method, layer, temporal)][f"cosine_similarity_class{cls}"] = class_cosine[cls_index]
        for layer in LAYERS:
            for temporal in TEMPORAL_LEVELS:
                for metric, cls in keys:
                    field = metric if cls is None else f"{metric}_class{cls}"
                    values = {method: stats[(method, layer, temporal)][field] for method in METHODS}
                    for method, value in values.items():
                        boot[("method", method, "", layer, temporal, metric, cls)][replicate] = value
                    for method, reference in COMPARISONS:
                        boot[("comparison", method, reference, layer, temporal, metric, cls)][replicate] = values[method] - values[reference]
    point_rows = trajectory_metrics(features_by_method, labels)
    point = {(row["method"], row["layer"], row["temporal_columns"]): row for row in point_rows}
    rows = []
    for layer in LAYERS:
        for temporal in TEMPORAL_LEVELS:
            for metric, cls in keys:
                field = metric if cls is None else f"{metric}_class{cls}"
                for method in METHODS:
                    sample = boot[("method", method, "", layer, temporal, metric, cls)]
                    lo, hi = np.quantile(sample, [.025, .975])
                    rows.append({"record_type": "method", "method": method, "reference_method": "", "layer": layer,
                                 "temporal_columns": temporal, "metric": metric, "class_id": "" if cls is None else cls,
                                 "estimate": point[(method, layer, temporal)][field], "ci_low": float(lo), "ci_high": float(hi),
                                 "difference": "", "difference_low": "", "difference_high": "", "excludes_zero": "",
                                 "replicates": replicates, "seed": seed})
                for method, reference in COMPARISONS:
                    sample = boot[("comparison", method, reference, layer, temporal, metric, cls)]
                    lo, hi = np.quantile(sample, [.025, .975])
                    delta = point[(method, layer, temporal)][field] - point[(reference, layer, temporal)][field]
                    rows.append({"record_type": "comparison", "method": method, "reference_method": reference,
                                 "layer": layer, "temporal_columns": temporal, "metric": metric,
                                 "class_id": "" if cls is None else cls, "estimate": point[(method, layer, temporal)][field],
                                 "ci_low": "", "ci_high": "", "difference": delta, "difference_low": float(lo),
                                 "difference_high": float(hi), "excludes_zero": bool(lo > 0 or hi < 0),
                                 "replicates": replicates, "seed": seed})
    return rows


def per_class_metric_rows(metric_rows: list[dict[str, Any]], num_classes: int) -> list[dict[str, Any]]:
    rows = []
    for row in metric_rows:
        for cls in range(num_classes):
            rows.append({"method": row["method"], "layer": row["layer"],
                         "temporal_columns": row["temporal_columns"], "class_id": cls,
                         "normalized_distance": row[f"normalized_distance_class{cls}"],
                         "cosine_similarity": row[f"cosine_similarity_class{cls}"],
                         "class_radius": row[f"class_radius_class{cls}"]})
    return rows


def write_figures(features_by_method: Mapping[str, np.ndarray], coordinates: Mapping[str, np.ndarray],
                  labels: np.ndarray, metric_rows: list[dict[str, Any]], interval_rows: list[dict[str, Any]],
                  output_dir: Path, pca_explained_variance: Mapping[str, list[float]] | None = None,
                  *, task: str = "interference", prefix_semantics: str = "column") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    classes = np.unique(labels).astype(int).tolist()
    cmap = plt.get_cmap("tab20", max(20, len(classes)))
    colors = {cls: cmap(cls % 20) for cls in classes}
    pca_explained_variance = pca_explained_variance or {}
    method_colors = {"p0": "#277da1", "p1": "#f9c74f", "p2": "#f94144", "p2_reverse": "#7209b7"}
    fig, axes = plt.subplots(4, 4, figsize=(16, 13), constrained_layout=True, sharex="row", sharey="row")
    for mi, method in enumerate(METHODS):
        coords = coordinates[method]
        for li, layer in enumerate(LAYERS):
            ax = axes[mi, li]
            for cls in classes:
                centroids = np.stack([coords[labels == cls, ti, li].mean(axis=0) for ti in range(len(TEMPORAL_LEVELS))])
                ax.plot(centroids[:, 0], centroids[:, 1], color=colors[cls], marker="o", linewidth=1.2,
                        alpha=.7 if len(classes) > 8 else .95,
                        label=f"Class {cls}" if mi == 0 and li == 0 else None)
                if len(classes) > 8:
                    ax.annotate(str(cls), centroids[-1], xytext=(2, 2), textcoords="offset points", fontsize=6)
                else:
                    for ti, point in enumerate(centroids):
                        ax.annotate(str(TEMPORAL_LEVELS[ti]), point, xytext=(3, 3), textcoords="offset points", fontsize=7)
            ax.set_title(f"{METHOD_LABELS[method]} · L{layer}")
            if li == 0:
                ratios = pca_explained_variance.get(method, [])
                captured = sum(ratios[:2]) if ratios else float("nan")
                ax.set_ylabel(f"PC 2 ({captured:.1%} variance)")
            if mi == 3:
                ax.set_xlabel("PC 1")
            ax.grid(alpha=.2)
    if len(classes) <= 8:
        fig.legend(loc="outside lower center", ncol=min(len(classes), 6))
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"class_centroid_trajectories.{ext}", dpi=220)
    plt.close(fig)

    metrics = ("normalized_distance", "cosine_similarity", "class_separation_ratio")
    titles = {"normalized_distance": "Normalized distance to full-input feature",
              "cosine_similarity": "Cosine similarity to full-input feature",
              "class_separation_ratio": "Between/within-class separation ratio"}
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    lookup = {(r["method"], r["layer"], r["temporal_columns"]): r for r in metric_rows}
    ci = {(r["method"], r["layer"], r["temporal_columns"], r["metric"]): r for r in interval_rows
          if r["record_type"] == "method" and r["class_id"] == ""}
    for ri, metric in enumerate(metrics):
        for li, layer in enumerate(LAYERS):
            ax = axes[ri, li]
            for method in METHODS:
                values, low, high = [], [], []
                for temporal in TEMPORAL_LEVELS:
                    row = lookup[(method, layer, temporal)]
                    interval = ci[(method, layer, temporal, metric)]
                    values.append(row[metric])
                    low.append(max(0.0, row[metric] - interval["ci_low"]))
                    high.append(max(0.0, interval["ci_high"] - row[metric]))
                ax.errorbar(TEMPORAL_LEVELS, values, yerr=np.vstack((low, high)), color=method_colors[method],
                            marker="o", capsize=2, label=METHOD_LABELS[method])
            ax.set_title(f"{titles[metric]} · L{layer}" if li == 0 else f"L{layer}")
            ax.set_xticks(TEMPORAL_LEVELS)
            ax.grid(alpha=.25)
            if li == 0:
                ax.set_ylabel(titles[metric])
            if ri == 2:
                ax.set_xlabel("Visible temporal columns" if prefix_semantics == "temporal" else "Visible input columns")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{TASK_LABELS.get(task, task)} representation trajectories")
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"trajectory_metrics.{ext}", dpi=220)
    plt.close(fig)


def write_summary(path: Path, metrics: list[dict[str, Any]], intervals: list[dict[str, Any]],
                  provenance: dict[str, Any], *, task: str = "interference") -> None:
    semantics = provenance.get("prefix_semantics", "column")
    class_counts = provenance.get("class_counts", {})
    count_text = ", ".join(f"class {key}: {value}" for key, value in sorted(class_counts.items(), key=lambda item: int(item[0])))
    prefix_word = "temporal" if semantics == "temporal" else "input-column"
    lines = [f"# Four-method {TASK_LABELS.get(task, task)} representation trajectories", "",
             f"Inference-only comparison of {prefix_word} prefix trajectories. Each method has its own PCA fit on validation data; PCA coordinates are not comparable across methods. Primary scalar metrics use the original 256-D feature space.", "",
             "## Protocol", "",
             "- Methods: P0, P1 (same-scale intermediate supervision), P2 (fine-to-coarse), and P2-Reverse (coarse-to-fine).",
             "- Checkpoints: P0/P1/P2 epoch 59 / step 13,560; P2-Reverse epoch 49 / step 11,300. All pretraining seeds are 0 and use the canonical six-mask multi_block/span_time signature.",
             "- P2-Reverse is from a 50-epoch run while the other methods use 60-epoch runs; interpret its comparison as exploratory.",
             "- P2-Reverse source split provenance is unverifiable because its original manifest is unavailable. All extracted representations and newly trained heads use the canonical seed-2025 split.",
             f"- Test samples: {sum(int(value) for value in class_counts.values())} ({count_text}).",
             "- Prefixes: 2, 4, 8, and 14 columns (28, 56, 112, and 196 tokens), retaining the original 14×14 positional grid.",
             "- Features: frozen context-encoder normalized tokens mean-pooled to 256 dimensions at L2/L4/L6/L8.",
             "- Intervals: 2,000 class-stratified paired bootstrap resamples (seed 2025); uncertainty is conditional on pretraining seed 0.", "",
             f"## Predeclared {'Tp4' if semantics == 'temporal' else 'P4'} comparisons", "",
             "Paired differences are method minus reference; intervals below are 95% bootstrap intervals for class-separation ratio.", "",
             "| Layer | P1−P0 | P2−P0 | P2-Reverse−P0 | P2−P1 | P2−P2-Reverse |", "|---|---:|---:|---:|---:|---:|"]
    comparison = {(r["method"], r["reference_method"], r["layer"]): r for r in intervals
                  if r["record_type"] == "comparison" and r["temporal_columns"] == 4
                  and r["metric"] == "class_separation_ratio" and r["class_id"] == ""}
    for layer in LAYERS:
        cells = [comparison[(method, reference, layer)] for method, reference in COMPARISONS]
        rendered = ["{:.3f} [{:.3f}, {:.3f}]".format(r["difference"], r["difference_low"], r["difference_high"])
                    for r in cells]
        lines.append(f"| L{layer} | " + " | ".join(rendered) + " |")
    lines += ["", "These are descriptive estimates; no automatic winner or causal attribution is implied. Intervals quantify resampling uncertainty for pretraining seed 0 and do not capture across-seed variation.", "",
              "See `class_centroid_trajectories` for PCA-projected class centroids and `trajectory_metrics` for original-space metric trajectories.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
