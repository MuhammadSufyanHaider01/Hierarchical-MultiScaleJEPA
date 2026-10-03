"""Matched P0/P2 intermediate representation trajectories for Interference."""

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_indices(indices: list[int]) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


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
    metadata: Mapping[str, Any], *, method: str, expected_epoch: int = 59, expected_step: int = 13560
) -> dict[str, Any]:
    """Check real Phase 5 checkpoint provenance and P0/P2 mask signatures."""
    phase5 = metadata.get("phase5", {})
    config = metadata.get("config", {})
    masks = config.get("masks", {})
    block = masks.get("multi_block", {})
    hierarchy = config.get("hierarchy", {})
    expected_hierarchy = method == "p2"
    errors = []
    if metadata.get("seed") != 0 or phase5.get("seed") != 0:
        errors.append("checkpoint seed must be 0")
    if phase5.get("method") != method:
        errors.append(f"checkpoint method must be {method}")
    if int(phase5.get("epoch", expected_epoch)) != expected_epoch:
        errors.append(f"checkpoint metadata epoch must be {expected_epoch}")
    if masks.get("type") != "multi_block":
        errors.append("mask type must be multi_block")
    if block.get("aspect_ratio_type") != "span_time":
        errors.append("multi_block aspect_ratio_type must be span_time")
    if int(block.get("num_pred_masks", -1)) != 6:
        errors.append("multi_block num_pred_masks must be 6")
    if bool(hierarchy.get("enabled", False)) != expected_hierarchy:
        errors.append(f"hierarchy enabled must be {expected_hierarchy}")
    if expected_hierarchy and (
        hierarchy.get("mode") != "mask_relative_fine_to_coarse"
        or tuple(hierarchy.get("layers", ())) != (2, 4, 6)
    ):
        errors.append("P2 hierarchy must be fine-to-coarse at L2/L4/L6")
    if errors:
        raise ValueError(f"invalid {method} checkpoint metadata: " + "; ".join(errors))
    return {
        "method": method,
        "epoch": expected_epoch,
        "global_step": expected_step,
        "mask_type": masks["type"],
        "mask_aspect_ratio_type": block["aspect_ratio_type"],
        "num_pred_masks": int(block["num_pred_masks"]),
        "hierarchy_enabled": expected_hierarchy,
        "hierarchy_mode": hierarchy.get("mode"),
    }


def validate_source_metrics(path: Path, *, method: str, manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("method") != method or int(payload.get("pretrain_seed", -1)) != 0:
        raise ValueError(f"{path} must describe method={method}, pretrain_seed=0")
    if "interference" not in payload.get("tasks", {}):
        raise ValueError(f"{path} has no Interference task")
    source_manifest = Path(payload.get("split_manifest", ""))
    if not source_manifest.is_absolute():
        source_manifest = _source_root(Path(path)) / source_manifest
    if source_manifest.resolve() != Path(manifest_path).resolve():
        raise ValueError(f"{method} metrics use split {source_manifest}, not {manifest_path}")
    return payload


def _source_root(metrics_path: Path) -> Path:
    for parent in metrics_path.resolve().parents:
        if parent.name == "runs":
            return parent.parent
    return metrics_path.resolve().parent


def load_split_and_labels(manifest_path: Path, raw_labels: np.ndarray) -> tuple[dict[str, list[int]], np.ndarray]:
    manifest = json.loads(Path(manifest_path).read_text())
    if int(manifest.get("schema_version", -1)) != 3 or int(manifest.get("seed", -1)) != 2025:
        raise ValueError("expected primary split manifest schema v3, seed 2025")
    task = manifest.get("tasks", {}).get("interference")
    if not isinstance(task, dict):
        raise ValueError("split manifest is missing the Interference task")
    raw_labels = np.asarray(raw_labels).reshape(-1)
    values = np.unique(raw_labels)
    label_map = {int(value): index for index, value in enumerate(values.tolist())}
    mapped = np.asarray([label_map[int(v)] for v in raw_labels], dtype=np.int64)
    indices = {key: [int(i) for i in task["indices"][key]] for key in ("train", "validation", "test")}
    for split, selected in indices.items():
        if len(selected) != len(set(selected)) or (selected and (min(selected) < 0 or max(selected) >= len(mapped))):
            raise ValueError(f"invalid {split} indices in manifest")
        declared = task.get("counts", {}).get(split)
        if declared is not None and declared != len(selected):
            raise ValueError(f"manifest {split} count does not match its indices")
    for split in ("validation", "test"):
        labels = mapped[indices[split]]
        counts = np.bincount(labels, minlength=len(values))
        if len(values) != 2 or not np.array_equal(counts, np.full(2, 700)):
            raise ValueError(f"{split} must contain 700 samples per class; got {counts.tolist()}")
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
    rows: list[dict[str, Any]] = []
    for method, features in features_by_method.items():
        for li, layer in enumerate(LAYERS):
            full = features[:, -1, li, :].astype(np.float64)
            for ti, temporal in enumerate(TEMPORAL_LEVELS):
                current = features[:, ti, li, :].astype(np.float64)
                distances = np.linalg.norm(current - full, axis=1) / (np.linalg.norm(full, axis=1) + 1e-12)
                similarities = np.sum(current * full, axis=1) / (
                    np.linalg.norm(current, axis=1) * np.linalg.norm(full, axis=1) + 1e-12
                )
                per_class_distance = [float(distances[labels == cls].mean()) for cls in (0, 1)]
                per_class_cosine = [float(similarities[labels == cls].mean()) for cls in (0, 1)]
                centers = [current[labels == cls].mean(axis=0) for cls in (0, 1)]
                within = [np.sqrt(np.mean(np.sum((current[labels == cls] - centers[cls]) ** 2, axis=1))) for cls in (0, 1)]
                separation = float(np.linalg.norm(centers[0] - centers[1]) / (np.mean(within) + 1e-12))
                rows.append({
                    "method": method, "layer": layer, "temporal_columns": temporal,
                    "normalized_distance": float(np.mean(per_class_distance)),
                    "cosine_similarity": float(np.mean(per_class_cosine)),
                    "class_separation_ratio": separation,
                    "normalized_distance_class0": per_class_distance[0],
                    "normalized_distance_class1": per_class_distance[1],
                    "cosine_similarity_class0": per_class_cosine[0],
                    "cosine_similarity_class1": per_class_cosine[1],
                })
    return rows


def paired_bootstrap(features_by_method: Mapping[str, np.ndarray], labels: np.ndarray,
                     *, replicates: int = 2000, seed: int = 2025) -> list[dict[str, Any]]:
    """Class-stratified paired bootstrap; identical sample draws for P0/P2."""
    if set(features_by_method) != {"p0", "p2"}:
        raise ValueError("paired bootstrap requires p0 and p2 features")
    rng = np.random.default_rng(seed)
    grouped = {cls: np.flatnonzero(labels == cls) for cls in (0, 1)}
    boot: dict[tuple[str, int, int, str], list[float]] = {}
    for rep in range(replicates):
        sampled = np.concatenate([rng.choice(grouped[cls], size=len(grouped[cls]), replace=True) for cls in (0, 1)])
        stats = trajectory_metrics({m: arr[sampled] for m, arr in features_by_method.items()}, labels[sampled])
        for row in stats:
            for metric in ("normalized_distance", "cosine_similarity", "class_separation_ratio"):
                boot.setdefault((row["method"], row["layer"], row["temporal_columns"], metric), []).append(row[metric])
                if metric != "class_separation_ratio":
                    for cls in (0, 1):
                        key = f"{metric}_class{cls}"
                        boot.setdefault((row["method"], row["layer"], row["temporal_columns"], metric, cls), []).append(row[key])
    rows = []
    for layer in LAYERS:
        for temporal in TEMPORAL_LEVELS:
            for metric in ("normalized_distance", "cosine_similarity", "class_separation_ratio"):
                p0 = np.asarray(boot[("p0", layer, temporal, metric)])
                p2 = np.asarray(boot[("p2", layer, temporal, metric)])
                delta = p2 - p0
                point_rows = trajectory_metrics(features_by_method, labels)
                point = {(r["method"], r["layer"], r["temporal_columns"]): r[metric] for r in point_rows}
                rows.append({
                    "layer": layer, "temporal_columns": temporal, "metric": metric,
                    "p0_low": float(np.quantile(p0, .025)), "p0_high": float(np.quantile(p0, .975)),
                    "p2_low": float(np.quantile(p2, .025)), "p2_high": float(np.quantile(p2, .975)),
                    "p2_minus_p0": float(point[("p2", layer, temporal)] - point[("p0", layer, temporal)]),
                    "difference_low": float(np.quantile(delta, .025)),
                    "difference_high": float(np.quantile(delta, .975)),
                    "replicates": replicates, "seed": seed,
                })
                if metric != "class_separation_ratio":
                    for cls in (0, 1):
                        p0_class = np.asarray(boot[("p0", layer, temporal, metric, cls)])
                        p2_class = np.asarray(boot[("p2", layer, temporal, metric, cls)])
                        delta_class = p2_class - p0_class
                        key = f"{metric}_class{cls}"
                        point_class = {(r["method"], r["layer"], r["temporal_columns"]): r[key] for r in point_rows}
                        rows.append({
                            "layer": layer, "temporal_columns": temporal, "metric": metric, "class_id": cls,
                            "p0_low": float(np.quantile(p0_class, .025)), "p0_high": float(np.quantile(p0_class, .975)),
                            "p2_low": float(np.quantile(p2_class, .025)), "p2_high": float(np.quantile(p2_class, .975)),
                            "p2_minus_p0": float(point_class[("p2", layer, temporal)] - point_class[("p0", layer, temporal)]),
                            "difference_low": float(np.quantile(delta_class, .025)),
                            "difference_high": float(np.quantile(delta_class, .975)),
                            "replicates": replicates, "seed": seed,
                        })
    return rows


def write_figures(features_by_method: Mapping[str, np.ndarray], coordinates: Mapping[str, np.ndarray],
                  labels: np.ndarray, metric_rows: list[dict[str, Any]], interval_rows: list[dict[str, Any]],
                  output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {0: "#277da1", 1: "#f94144"}
    markers = ["o", "s", "^", "D"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for mi, method in enumerate(("p0", "p2")):
        coords = coordinates[method]
        for li, layer in enumerate(LAYERS):
            ax = axes[mi, li]
            for cls in (0, 1):
                centroids = np.stack([coords[labels == cls, ti, li].mean(axis=0) for ti in range(4)])
                ax.plot(centroids[:, 0], centroids[:, 1], color=colors[cls], marker="o", linewidth=1.8,
                        label=f"Class {cls}" if li == 0 and mi == 0 else None)
                ax.scatter(centroids[:, 0], centroids[:, 1], color=colors[cls], marker="o", s=45)
                for ti, point in enumerate(centroids):
                    ax.annotate(f"{TEMPORAL_LEVELS[ti]}", point, xytext=(3, 3), textcoords="offset points", fontsize=8)
            ax.set_title(f"{method.upper()} · L{layer}")
            ax.set_xlabel("PCA 1")
            if li == 0:
                ax.set_ylabel("PCA 2\n" + method.upper())
            ax.grid(alpha=.2)
    fig.legend(loc="outside lower center", ncol=2)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"class_centroid_trajectories.{ext}", dpi=220)
    plt.close(fig)

    metrics = ("normalized_distance", "cosine_similarity", "class_separation_ratio")
    labels_text = {
        "normalized_distance": "Normalized distance to full-input feature",
        "cosine_similarity": "Cosine similarity to full-input feature",
        "class_separation_ratio": "Between/within-class separation ratio",
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    lookup = {(r["method"], r["layer"], r["temporal_columns"]): r for r in metric_rows}
    ci_lookup = {(r["layer"], r["temporal_columns"], r["metric"]): r for r in interval_rows if r.get("class_id") is None}
    for ax, metric in zip(axes, metrics):
        for method, color in (("p0", "#277da1"), ("p2", "#f94144")):
            for layer, linestyle in zip(LAYERS, ("-", "--", ":", "-.")):
                values, lower, upper = [], [], []
                for temporal in TEMPORAL_LEVELS:
                    row = lookup[(method, layer, temporal)]
                    ci = ci_lookup[(layer, temporal, metric)]
                    values.append(row[metric])
                    lower.append(max(0.0, row[metric] - ci[f"{method}_low"]))
                    upper.append(max(0.0, ci[f"{method}_high"] - row[metric]))
                ax.errorbar(TEMPORAL_LEVELS, values, yerr=np.vstack((lower, upper)), color=color,
                            linestyle=linestyle, marker="o", capsize=2,
                            label=f"{method.upper()} L{layer}")
        ax.set_title(labels_text[metric])
        ax.set_xlabel("Visible temporal patch columns")
        ax.set_xticks(TEMPORAL_LEVELS)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8, ncol=2)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"trajectory_metrics.{ext}", dpi=220)
    plt.close(fig)


def write_summary(path: Path, metrics: list[dict[str, Any]], intervals: list[dict[str, Any]], provenance: dict[str, Any]) -> None:
    lines = ["# P0/P2 Interference representation trajectories", "",
             "Inference-only comparison of fixed-grid temporal-prefix features. PCA is fit on validation features separately for each model; test metrics are computed in the original 256-D feature space.", "",
             "## Protocol", "",
             f"- Checkpoints: P0 epoch {provenance['checkpoints']['p0']['epoch']} / P2 epoch {provenance['checkpoints']['p2']['epoch']} (step 13560).",
             "- Samples: 1,400 held-out test examples, balanced at 700 per class.",
             "- Prefix token counts: 28, 56, 112, and 196; original 14×14 positional grid retained.",
             "- Features: normalized encoder tokens mean-pooled to 256 dimensions at L2/L4/L6/L8.",
             "- Uncertainty: class-stratified paired bootstrap, 2,000 replicates; interval table includes P2−P0 CIs.", "",
             "## Key test-set findings", ""]
    lookup = {(r["method"], r["layer"], r["temporal_columns"]): r for r in metrics}
    for metric, human in (("normalized_distance", "distance"), ("cosine_similarity", "cosine"), ("class_separation_ratio", "separation ratio")):
        p0_short, p0_full = lookup[("p0", 8, 2)][metric], lookup[("p0", 8, 14)][metric]
        p2_short, p2_full = lookup[("p2", 8, 2)][metric], lookup[("p2", 8, 14)][metric]
        lines.append(f"- At L8, {human}: P0 Tp2→Tp14 {p0_short:.4f}→{p0_full:.4f}; P2 {p2_short:.4f}→{p2_full:.4f}.")
    lines += ["", "See `class_centroid_trajectories` for the projected class paths and `trajectory_metrics` for estimates with 95% bootstrap intervals.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
