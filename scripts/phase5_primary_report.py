#!/usr/bin/env python3
"""Create a fair, single-seed masking-condition comparison result packet.

All exit-method manuscript rows use frozen context-encoder representations and the
same persisted 80/10/10 downstream split.  The original LatentWave row is its
full-depth L8 context representation; the post-hoc early-exit row reuses that
same P0 checkpoint, and the hierarchical row uses its own P2 checkpoint.

The historical EMA-teacher/source-style probe is deliberately not plotted or
tabulated here: it is a useful provenance check, but not a fair comparator for
the frozen-head evaluation protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from masking_profiles import load_mask_profile  # noqa: E402

EXIT_LAYERS = (2, 4, 6, 8)
TASK_ORDER = (
    "rfs",
    "jamming",
    "human-activity",
    "interference",
    "nrpos-outdoor",
    "deepmimo-beam",
    "deepmimo-los",
)
# Updated by ``_discover_tasks`` before any result packet is loaded. Keeping
# the historical LoS/NLoS task in TASK_ORDER lets existing multiblock reports
# remain reproducible while new packets use the six-task suite.
TASKS = TASK_ORDER[:6]
METHODS = ("b1", "p2")
METHOD_LABELS = {
    "b0": "B0 — Original LatentWave",
    "b1": "P0 — LatentWave with Post-hoc Frozen Early Exits",
    "p1": "P1 — Same-scale Multi-depth JEPA",
    "p2": "P2 — Proposed Hierarchical Multi-Scale JEPA",
    "p3": "P3 — Pair-local Hierarchical Multi-Scale JEPA",
    "p2_reverse": "P2-Reverse — Coarse-to-fine Hierarchical JEPA",
    "cnn_p0": "C0 — CNN Original L8 JEPA Objective",
    "cnn_p1": "CNN-HierJEPA P1 — Same-scale Multi-depth",
    "cnn_p2": "C2 — CNN Mask-relative Fine-to-coarse JEPA",
    "cnn_p2_reverse": "CNN-HierJEPA P2-Reverse — Coarse-to-fine",
}
FIGURE_METHOD_LABELS = {
    "b0": "B0: Original LatentWave (full-depth)",
    "b1": "P0: LatentWave + post-hoc frozen exits",
    "p1": "P1: Same-scale multi-depth JEPA",
    "p2": "P2: Proposed hierarchical JEPA",
    "p3": "P3: Pair-local hierarchical JEPA",
    "p2_reverse": "P2-Reverse: Coarse-to-fine hierarchical JEPA",
    "cnn_p0": "C0: CNN original L8 objective",
    "cnn_p1": "CNN-HierJEPA P1: Same-scale",
    "cnn_p2": "C2: CNN mask-relative fine-to-coarse",
    "cnn_p2_reverse": "CNN-HierJEPA P2-Reverse: Coarse-to-fine",
}
FIGURE_METHOD_SHORT = {
    "b0": "B0\nOriginal",
    "b1": "P0\nPost-hoc exits",
    "p1": "P1\nSame-scale",
    "p2": "P2\nProposed",
    "p3": "P3\nPair-local",
    "p2_reverse": "P2-Reverse\nCoarse-to-fine",
    "cnn_p0": "C0\nOriginal L8",
    "cnn_p1": "CNN-P1\nSame-scale",
    "cnn_p2": "C2\nFine-to-coarse",
    "cnn_p2_reverse": "CNN-P2-Reverse\nCoarse-to-fine",
}
TASK_LABELS = {
    "rfs": "RF sensing",
    "jamming": "Jamming classification",
    "human-activity": "Human activity sensing",
    "interference": "Interference classification",
    "nrpos-outdoor": "NR positioning",
    "deepmimo-beam": "DeepMIMO beam",
    "deepmimo-los": "DeepMIMO LoS/NLoS",
}
TASK_METRICS = {
    "rfs": ("macro_accuracy", "Macro accuracy", True),
    "jamming": ("macro_accuracy", "Macro accuracy", True),
    "human-activity": ("macro_accuracy", "Macro accuracy", True),
    "interference": ("macro_accuracy", "Macro accuracy", True),
    "nrpos-outdoor": ("mean_distance_error", "Mean distance error (m)", False),
    "deepmimo-beam": ("macro_accuracy", "Macro accuracy", True),
    "deepmimo-los": ("macro_accuracy", "Macro accuracy", True),
}
SOURCE_METRICS = {
    "rfs": ("rfs_pca", "Balanced accuracy (source probe)", True),
    "nrpos-outdoor": ("pos_error", "Mean distance error (m)", False),
    "deepmimo-beam": ("beam_pca", "Balanced accuracy (source probe)", True),
    "deepmimo-los": ("los_pca", "Balanced accuracy (source probe)", True),
}
CNN_MASK_PROFILE_DETAILS = {
    "frequency_n6": {
        "geometry": "six span-time blocks (contiguous frequency bands across the full time axis)",
        "repeated_geometry_caveat": (
            "frequency_n6 is a separately trained historical-style repeat of "
            "multi_block_n6 with the same nominal span-time geometry; the two "
            "packets are not independent mask-family evidence."
        ),
    },
    "region_n6": {
        "geometry": "six random contiguous two-dimensional time-frequency blocks",
        "repeated_geometry_caveat": None,
    },
    "time_n6": {
        "geometry": "six span-frequency blocks (contiguous time bands across the full frequency axis)",
        "repeated_geometry_caveat": None,
    },
    "multi_block_n6": {
        "geometry": "six span-time blocks (contiguous frequency bands across the full time axis)",
        "repeated_geometry_caveat": None,
    },
}
CNN_MASK_PROFILE_NAMES = tuple(CNN_MASK_PROFILE_DETAILS)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs/phase5_formal")
    parser.add_argument(
        "--p2-run-root",
        type=Path,
        default=None,
        help=(
            "Optional root containing p2/seed_000 artifacts. Use when the proposed "
            "checkpoint was evaluated in a separate, checkpoint-specific run root."
        ),
    )
    parser.add_argument(
        "--p1-run-root",
        type=Path,
        default=None,
        help=(
            "Optional root containing p1/seed_000 artifacts. When supplied, the "
            "same-scale multi-depth JEPA control is included in all unified figures and tables."
        ),
    )
    parser.add_argument(
        "--p3-run-root",
        type=Path,
        default=None,
        help="Optional root containing p3/seed_000 artifacts for the pair-local routing ablation.",
    )
    parser.add_argument("--cnn-run-root", type=Path, default=None, help="Root containing cnn_p0, cnn_p1, cnn_p2, and cnn_p2_reverse result directories.")
    parser.add_argument(
        "--comparison-mode",
        choices=("legacy", "cnn_only"),
        default="legacy",
        help="cnn_only combines matching-mask ViT B0/P0/P2 and selected CNN method packets.",
    )
    parser.add_argument(
        "--cnn-methods",
        nargs="+",
        choices=("cnn_p0", "cnn_p1", "cnn_p2", "cnn_p2_reverse"),
        default=None,
        help="Ordered CNN methods included by --comparison-mode cnn_only.",
    )
    parser.add_argument(
        "--mask-profile",
        choices=CNN_MASK_PROFILE_NAMES,
        default=None,
        help=(
            "Canonical registered profile for a CNN-only packet. When supplied, "
            "the report verifies matching C0/C2 pretraining provenance and uses "
            "the per-profile ViT P0 reference."
        ),
    )
    parser.add_argument(
        "--vit-p0-run-root",
        type=Path,
        default=None,
        help=(
            "Root for the per-mask ViT P0 frozen-head packet. Expected files are "
            "head_seed_000/p0/seed_000/downstream/exit_metrics.json and "
            "benchmarks/p0/seed_000/benchmarks.json. Defaults to "
            "runs/masking_head_seeds/<mask-profile> (multi_block_n6 when omitted)."
        ),
    )
    parser.add_argument(
        "--vit-p2-run-root",
        type=Path,
        default=None,
        help=(
            "Root containing the matching-mask ViT P2 packet at "
            "p2/seed_000/{downstream/exit_metrics.json,benchmarks.json}. "
            "Defaults to runs/masking_study/<mask-profile>."
        ),
    )
    parser.add_argument(
        "--p2-reverse-import-root",
        type=Path,
        default=None,
        help=(
            "Optional imported P2-Reverse packet directory containing downstream/exit_metrics.json "
            "and, where available, benchmarks.json."
        ),
    )
    parser.add_argument("--source-root", type=Path, default=REPOSITORY_ROOT / "runs/phase5_pilot")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "reports/phase5_primary")
    parser.add_argument(
        "--masking-label",
        default="multi-block/span_time",
        help="Human-readable masking condition recorded in the report manifest and README.",
    )
    parser.add_argument(
        "--report-title",
        default="Phase 5 primary single-seed pilot packet",
        help="Top-level Markdown title for this report packet.",
    )
    parser.add_argument(
        "--provenance-note",
        default=None,
        help="Optional run-specific limitation or provenance note recorded in the README and manifest.",
    )
    return parser.parse_args()


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _split_manifest_signature(path: str) -> str:
    """Compare split assignments while ignoring machine-specific dataset paths."""

    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = REPOSITORY_ROOT / manifest_path
    manifest = _read(manifest_path)
    for task_record in manifest.get("tasks", {}).values():
        spec = task_record.get("spec", {})
        if isinstance(spec, dict):
            spec.pop("path", None)
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cnn_pretraining_experiment(packet: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    checkpoint = Path(packet["checkpoint"])
    experiment = checkpoint.parents[1] / "experiment.json"
    if not experiment.is_file():
        raise FileNotFoundError(
            f"{packet['method']} checkpoint has no experiment metadata: {experiment}"
        )
    return experiment, _read(experiment)


def _validate_cnn_profile_provenance(
    packets: dict[str, dict[str, Any]], profile_name: str
) -> dict[str, Any]:
    """Require comparable CNN continuations before emitting a named packet."""

    profile, profile_path, profile_sha256 = load_mask_profile(profile_name)
    expected_profile = {
        "name": profile_name,
        "path": str(profile_path.relative_to(REPOSITORY_ROOT)),
        "sha256": profile_sha256,
        "overrides": profile["masks"],
    }
    runtime_keys = (
        "seed", "microbatch_size", "accumulate_grad_batches", "effective_batch_size",
        "max_epochs", "schedule_horizon_epochs", "precision", "attention_backend",
        "epoch_metrics_only", "performance_safe", "data_loader_workers",
        "hdf5_worker_budget", "data_loader_workers_per_cache",
        "data_loader_prefetch_factor", "early_stopping", "validation_fraction",
        "validation_seed", "early_stopping_patience", "early_stopping_min_delta",
        "resume_schedule_horizon_extended",
    )
    runtime_reference = None
    resolved_masks_reference = None
    split_hashes = set()
    for method, packet in packets.items():
        experiment_path, payload = _cnn_pretraining_experiment(packet)
        metadata = payload.get("metadata")
        runtime = payload.get("runtime")
        if not isinstance(metadata, dict) or not isinstance(runtime, dict):
            raise ValueError(f"{method} experiment metadata is incomplete: {experiment_path}")
        if metadata.get("seed") != 0 or runtime.get("seed") != 0:
            raise ValueError(f"{method} does not use the required pretraining seed 0")
        if runtime.get("method") != method:
            raise ValueError(f"{method} experiment identifies {runtime.get('method')!r}")
        if metadata.get("config", {}).get("encoder", {}).get("backbone") != "convnext_v2_hier":
            raise ValueError(f"{method} experiment is not a ConvNeXt V2 CNN run")
        actual_profile = metadata.get("mask_profile")
        if not isinstance(actual_profile, dict):
            raise ValueError(f"{method} experiment does not record a mask profile")
        for key, expected in expected_profile.items():
            if actual_profile.get(key) != expected:
                raise ValueError(
                    f"{method} mask profile {key!r} does not match {profile_name!r}"
                )
        resolved_masks = actual_profile.get("resolved")
        if not isinstance(resolved_masks, dict):
            raise ValueError(f"{method} experiment does not record resolved mask settings")
        if resolved_masks_reference is None:
            resolved_masks_reference = resolved_masks
        elif resolved_masks != resolved_masks_reference:
            raise ValueError("CNN-P0 and CNN-P2 resolved mask settings differ")
        if runtime.get("mask_profile") != actual_profile:
            raise ValueError(f"{method} runtime and checkpoint mask-profile metadata differ")
        runtime_contract = {key: runtime.get(key) for key in runtime_keys}
        if runtime_reference is None:
            runtime_reference = runtime_contract
        elif runtime_contract != runtime_reference:
            raise ValueError("CNN-P0 and CNN-P2 pretraining execution profiles differ")
        validation_manifest = experiment_path.parent / "pretrain_split.json"
        if not validation_manifest.is_file():
            raise FileNotFoundError(
                f"{method} continuation has no pretraining validation split: {validation_manifest}"
            )
        split_hashes.add(_sha256(validation_manifest))
    if len(split_hashes) != 1:
        raise ValueError("CNN-P0 and CNN-P2 pretraining validation splits differ")
    detail = CNN_MASK_PROFILE_DETAILS.get(profile_name, {})
    return {
        "name": profile_name,
        "path": expected_profile["path"],
        "sha256": profile_sha256,
        "overrides": profile["masks"],
        "resolved": resolved_masks_reference,
        "geometry": detail.get("geometry"),
        "repeated_geometry_caveat": detail.get("repeated_geometry_caveat"),
        "pretraining_validation_split_sha256": next(iter(split_hashes)),
        "execution_profile": runtime_reference,
    }


def _discover_tasks_from_packets(packets: dict[str, dict[str, Any]]) -> tuple[tuple[str, ...], str]:
    """Validate task-suite and persisted-split compatibility for local packets."""
    if not packets:
        raise ValueError("report requires at least one result packet")
    task_sets = {method: set(packet.get("tasks", {})) for method, packet in packets.items()}
    if len(set(map(frozenset, task_sets.values()))) != 1:
        raise ValueError(f"included methods have different downstream task suites: {task_sets}")
    first_method = next(iter(packets))
    unknown = task_sets[first_method].difference(TASK_ORDER)
    split_manifests = {method: packet.get("split_manifest") for method, packet in packets.items()}
    if None in split_manifests.values():
        raise ValueError(f"included methods are missing a persisted split manifest: {split_manifests}")
    split_signatures = {method: _split_manifest_signature(path) for method, path in split_manifests.items()}
    if len(set(split_signatures.values())) != 1:
        raise ValueError(f"included methods have different persisted split assignments: {split_manifests}")
    if unknown:
        raise ValueError(f"report does not recognise downstream tasks: {sorted(unknown)}")
    tasks = tuple(task for task in TASK_ORDER if task in task_sets[first_method])
    if not tasks:
        raise ValueError("report requires at least one downstream task")
    return tasks, next(iter(split_signatures.values()))


def _discover_tasks(
    p0_run_root: Path,
    p2_run_root: Path,
    p1_run_root: Path | None = None,
    p3_run_root: Path | None = None,
    p2_reverse_metrics_path: Path | None = None,
    cnn_run_root: Path | None = None,
) -> tuple[str, ...]:
    """Require all included methods to contain the same recognised task suite."""

    packets = {"p0": _read(p0_run_root / "p0/seed_000/downstream/exit_metrics.json")}
    if cnn_run_root is not None:
        for method in ("cnn_p0", "cnn_p1", "cnn_p2", "cnn_p2_reverse"):
            packets[method] = _read(cnn_run_root / method / "seed_000/downstream/exit_metrics.json")
    else:
        packets["p2"] = _read(p2_run_root / "p2/seed_000/downstream/exit_metrics.json")
    if cnn_run_root is None and p1_run_root is not None:
        packets["p1"] = _read(p1_run_root / "p1/seed_000/downstream/exit_metrics.json")
    if cnn_run_root is None and p3_run_root is not None:
        packets["p3"] = _read(p3_run_root / "p3/seed_000/downstream/exit_metrics.json")
    if cnn_run_root is None and p2_reverse_metrics_path is not None:
        packets["p2_reverse"] = _read(p2_reverse_metrics_path)
    task_sets = {method: set(packet.get("tasks", {})) for method, packet in packets.items()}
    if len(set(map(frozenset, task_sets.values()))) != 1:
        raise ValueError(f"included methods have different downstream task suites: {task_sets}")
    unknown = task_sets["p0"].difference(TASK_ORDER)
    # Imported P2-Reverse metrics preserve the source-machine split-manifest
    # path, but do not bundle its index assignments. Locally generated methods
    # still require an exact persisted-manifest match.
    local_packets = {method: packet for method, packet in packets.items() if method != "p2_reverse"}
    split_manifests = {method: packet.get("split_manifest") for method, packet in local_packets.items()}
    if None in split_manifests.values():
        raise ValueError(f"included methods are missing a persisted split manifest: {split_manifests}")
    split_signatures = {method: _split_manifest_signature(path) for method, path in split_manifests.items()}
    if len(set(split_signatures.values())) != 1:
        raise ValueError(f"included methods have different persisted split assignments: {split_manifests}")
    if unknown:
        raise ValueError(f"report does not recognise downstream tasks: {sorted(unknown)}")
    tasks = tuple(task for task in TASK_ORDER if task in task_sets["p0"])
    if not tasks:
        raise ValueError("report requires at least one downstream task")
    return tasks


def _task_grid() -> tuple[int, int]:
    """Return a compact panel layout for either four or six task packets."""

    columns = 2 if len(TASKS) <= 4 else 3
    rows = (len(TASKS) + columns - 1) // columns
    return rows, columns


def _hide_unused_axes(axes, populated: int) -> None:
    for axis in axes.flat[populated:]:
        axis.set_visible(False)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_metric(task: str, value: float) -> str:
    if TASK_METRICS[task][2]:
        return f"{value * 100:.2f}%"
    return f"{value:.3f} m"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_source_reference(source_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = [source_root / "p0_source_probe.json", source_root / "p0_source_probe_has.json"]
    records = [_read(path) for path in paths if path.is_file()]
    if not records:
        raise FileNotFoundError(paths[0])
    tasks: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for record in records:
        if record.get("protocol", {}).get("name") != "pinned_latentwave_source_style_l8_probe":
            raise ValueError("source probe does not identify the pinned LatentWave protocol")
        for task, value in record.get("tasks", {}).items():
            if task in tasks:
                raise ValueError(f"duplicate source-probe task: {task}")
            if task not in TASKS:
                if task == "deepmimo-los":
                    # Ignore the legacy task if an older source probe was
                    # reused while assembling the revised report packet.
                    continue
                raise ValueError(f"unexpected source-probe task: {task}")
            tasks[task] = value
    missing = [task for task in TASKS if task not in tasks]
    if missing:
        raise ValueError(f"source-probe results are incomplete: {missing}")
    first = records[0]
    reference = {
        "method": "b0",
        "label": METHOD_LABELS["b0"],
        "checkpoint": first["checkpoint"],
        "config": first["config"],
        "device": first["device"],
        "protocol": first["protocol"],
        "tasks": tasks,
    }
    rows = []
    for task in TASKS:
        result = tasks[task]
        metric_key, metric_label, higher_is_better = SOURCE_METRICS[task]
        if result["metric_name"] != metric_key:
            raise ValueError(
                f"source metric mismatch for {task}: expected {metric_key}, got {result['metric_name']}"
            )
        rows.append(
            {
                "method": "b0",
                "method_label": METHOD_LABELS["b0"],
                "task": task,
                "task_label": TASK_LABELS[task],
                "metric": metric_key,
                "metric_label": metric_label,
                "value": float(result["value"]),
                "display_value": _fmt_metric(task, float(result["value"])),
                "higher_is_better": higher_is_better,
                "examples": int(result["examples"]),
                "protocol": "EMA target encoder, full-input L8, source-style 80/20 probe",
            }
        )
    return reference, rows


def _load_context_l8_reference(baseline_packet: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Use P0's frozen context-encoder L8 heads as Original LatentWave.

    This is intentionally derived from the same recorded P0 artifact used by
    the post-hoc-exit baseline.  It changes only the deployment depth, not the
    encoder, checkpoint, split, head-training protocol, or test set.
    """

    rows = []
    for task in TASKS:
        fixed_l8 = next(
            row
            for row in baseline_packet["metric_rows"]
            if row["task"] == task
            and row["evaluation"] == "fixed_exit"
            and row["exit_layer"] == 8
        )
        row = dict(fixed_l8)
        row.update(
            {
                "method": "b0",
                "method_label": METHOD_LABELS["b0"],
                "protocol": "frozen context encoder, full-input L8, shared 80/10/10 split",
                "examples": None,
            }
        )
        rows.append(row)
    return (
        {
            "method": "b0",
            "label": METHOD_LABELS["b0"],
            "checkpoint": baseline_packet["checkpoint"],
            "protocol": "frozen context encoder, full-input L8, shared 80/10/10 split",
        },
        rows,
    )


def _load_exit_results(
    run_root: Path,
    method: str,
    *,
    source_method: str | None = None,
    metrics_path: Path | None = None,
    benchmark_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_method = source_method or {"b1": "p0", "p1": "p1", "p2": "p2", "p3": "p3", "cnn_p0": "cnn_p0", "cnn_p1": "cnn_p1", "cnn_p2": "cnn_p2", "cnn_p2_reverse": "cnn_p2_reverse"}[method]
    base = run_root / source_method / "seed_000"
    metrics_path = metrics_path or (base / "downstream/exit_metrics.json")
    benchmark_path = benchmark_path or (base / "benchmarks.json")
    metrics = _read(metrics_path)
    if method == "p2_reverse" and metrics.get("method") != "p2_reverse":
        raise ValueError("imported packet does not identify P2-Reverse")
    if metrics.get("dataset_suite") != "primary":
        raise ValueError(f"{method} packet is not the primary downstream suite")
    if not metrics.get("protocol", {}).get("test_evaluation"):
        raise ValueError(f"{method} packet does not contain a held-out test evaluation")
    rows = []
    for task in TASKS:
        if task not in metrics.get("tasks", {}):
            raise ValueError(f"{method} downstream metrics missing task: {task}")
        task_record = metrics["tasks"][task]
        metric_key, metric_label, higher_is_better = TASK_METRICS[task]
        if task_record["spec"]["metric_name"] != metric_key:
            raise ValueError(
                f"{method} metric mismatch for {task}: expected {metric_key}, "
                f"got {task_record['spec']['metric_name']}"
            )
        for layer in EXIT_LAYERS:
            exit_record = task_record["exits"][str(layer)]
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "exit_layer": layer,
                    "evaluation": "fixed_exit",
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "value": float(exit_record["test"][metric_key]),
                    "display_value": _fmt_metric(task, float(exit_record["test"][metric_key])),
                    "validation_value": float(exit_record["validation"][metric_key]),
                    "higher_is_better": higher_is_better,
                    "best_epoch": int(exit_record["best_epoch"]),
                    "head_epochs_completed": int(exit_record["epochs_completed"]),
                    "head_stopped_early": bool(exit_record["stopped_early"]),
                    "protocol": "frozen context encoder, independent head, shared 80/10/10 split",
                }
            )
        for policy, value_key, layer_key in (
            ("best_validation", "selected_test_best_validation", "selected_exit_best_validation"),
            ("earliest_within_tolerance", "selected_test_earliest_within_tolerance", "selected_exit_earliest_within_tolerance"),
        ):
            selected = task_record[value_key]
            layer = int(task_record[layer_key])
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "exit_layer": layer,
                    "evaluation": policy,
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "value": float(selected[metric_key]),
                    "display_value": _fmt_metric(task, float(selected[metric_key])),
                    "validation_value": None,
                    "higher_is_better": higher_is_better,
                    "protocol": "validation-selected from the same frozen-head evaluation",
                }
            )

    benchmark_rows = []
    benchmark_available = benchmark_path.is_file()
    if benchmark_available:
        benchmarks = _read(benchmark_path)
        entries = {(entry["task"], int(entry["exit_layer"])): entry for entry in benchmarks.get("entries", [])}
        for task in TASKS:
            for layer in EXIT_LAYERS:
                key = (task, layer)
                if key not in entries:
                    raise ValueError(f"{method} benchmark missing task/exit: {task}/L{layer}")
                entry = entries[key]
                benchmark_rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "task": task,
                        "task_label": TASK_LABELS[task],
                        "exit_layer": layer,
                        "flops": int(entry["flops"]),
                        "latency_mean_ms": float(entry["latency"]["mean_ms"]),
                        "latency_median_ms": float(entry["latency"]["median_ms"]),
                        "latency_p95_ms": float(entry["latency"]["p95_ms"]),
                        "latency_std_ms": float(entry["latency"]["standard_deviation_ms"]),
                        "sample_shape": entry["sample_shape"],
                        "protocol": "batch-1 synchronized CUDA timing after warm-up",
                    }
                )
        for task in TASKS:
            l8 = next(row["latency_mean_ms"] for row in benchmark_rows if row["task"] == task and row["exit_layer"] == 8)
            l8_flops = next(row["flops"] for row in benchmark_rows if row["task"] == task and row["exit_layer"] == 8)
            for row in benchmark_rows:
                if row["task"] == task:
                    row["latency_savings_vs_l8"] = 1.0 - row["latency_mean_ms"] / l8
                    row["flops_savings_vs_l8"] = 1.0 - row["flops"] / l8_flops

    packet = {
        "method": method,
        "label": METHOD_LABELS[method],
        "pretraining_method": source_method,
        "checkpoint": metrics["checkpoint"],
        "metrics_path": str(metrics_path),
        "benchmark_path": str(benchmark_path) if benchmark_available else None,
        "benchmark_available": benchmark_available,
        "split_manifest": metrics.get("split_manifest"),
        "metrics_protocol": metrics["protocol"],
        "metric_rows": rows,
        "benchmark_rows": benchmark_rows,
    }
    return packet, rows + benchmark_rows

def _plot_performance(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = {"b0": "#009E73", "b1": "#0072B2", "p1": "#CC79A7", "p2": "#D55E00", "p3": "#000000", "p2_reverse": "#6A3D9A", "cnn_p0": "#44AA99", "cnn_p1": "#F0E442", "cnn_p2": "#E69F00", "cnn_p2_reverse": "#56B4E9"}
    paths = []
    rows_count, columns_count = _task_grid()
    fig, axes = plt.subplots(rows_count, columns_count, figsize=(4.8 * columns_count, 3.7 * rows_count), squeeze=False)
    for axis, task in zip(axes.flat, TASKS):
        metric_key, metric_label, higher_is_better = TASK_METRICS[task]
        available_methods = [method for method in ("b0", *METHODS) if any(row["method"] == method and row["task"] == task and row.get("evaluation") == "fixed_exit" for row in rows)]
        for method in available_methods:
            method_rows = sorted(
                (row for row in rows if row["method"] == method and row["task"] == task and row.get("evaluation") == "fixed_exit"),
                key=lambda row: row["exit_layer"],
            )
            axis.plot(
                [row["exit_layer"] for row in method_rows],
                [row["value"] for row in method_rows],
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                color=colors[method],
                label=FIGURE_METHOD_LABELS[method],
            )
        axis.set_title(TASK_LABELS[task], fontsize=11, weight="bold")
        axis.set_xlabel("Encoder exit layer")
        axis.set_ylabel(metric_label)
        axis.set_xticks(EXIT_LAYERS)
        axis.grid(axis="y", alpha=0.25, linewidth=0.8)
        if higher_is_better:
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        else:
            axis.text(
                0.98,
                0.05,
                "lower is better",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#555555",
            )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    _hide_unused_axes(axes, len(TASKS))
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Frozen-head downstream performance by encoder depth", fontsize=14, weight="bold", y=1.045)
    fig.text(0.5, -0.015, "Single-seed pilot; test metrics are reported after validation-only head/exit selection.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"early_exit_performance.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _plot_selected(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    """Plot held-out performance at the validation-best exit for every method."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = {"b0": "#009E73", "b1": "#0072B2", "p1": "#CC79A7", "p2": "#D55E00", "p3": "#000000", "p2_reverse": "#6A3D9A", "cnn_p0": "#44AA99", "cnn_p1": "#F0E442", "cnn_p2": "#E69F00", "cnn_p2_reverse": "#56B4E9"}
    rows_count, columns_count = _task_grid()
    fig, axes = plt.subplots(rows_count, columns_count, figsize=(4.8 * columns_count, 3.7 * rows_count), squeeze=False)
    for axis, task in zip(axes.flat, TASKS):
        _, metric_label, higher_is_better = TASK_METRICS[task]
        plot_methods = [method for method in ("b0", *METHODS) if any(row["method"] == method and row["task"] == task for row in rows)]
        selected = []
        for method in plot_methods:
            if method == "b0":
                selected.append(next(row for row in rows if row["method"] == method and row["task"] == task and row.get("evaluation") == "fixed_exit" and row["exit_layer"] == 8))
            else:
                selected.append(next(row for row in rows if row["method"] == method and row["task"] == task and row.get("evaluation") == "best_validation"))
        bars = axis.bar(range(len(plot_methods)), [row["value"] for row in selected], color=[colors[row["method"]] for row in selected], edgecolor="#222222", linewidth=0.7)
        axis.set_xticks(range(len(plot_methods)), [f"{FIGURE_METHOD_SHORT[row['method']]}\nL{row['exit_layer']}" for row in selected])
        axis.set_title(TASK_LABELS[task], fontsize=11, weight="bold")
        axis.set_ylabel(metric_label)
        axis.grid(axis="y", alpha=0.25, linewidth=0.8)
        if higher_is_better:
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        else:
            axis.text(0.98, 0.05, "lower is better", transform=axis.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")
        for bar, row in zip(bars, selected):
            axis.annotate(f"L{row['exit_layer']}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3 if higher_is_better else -12), textcoords="offset points", ha="center", va="bottom" if higher_is_better else "top", fontsize=8)
    plot_methods = [method for method in ("b0", *METHODS) if any(row["method"] == method for row in rows)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[method], label=FIGURE_METHOD_LABELS[method]) for method in plot_methods]
    _hide_unused_axes(axes, len(TASKS))
    fig.legend(handles=handles, loc="upper center", ncol=len(plot_methods), frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=9)
    fig.suptitle("Validation-best exits and held-out test performance", fontsize=14, weight="bold", y=1.045)
    fig.text(0.5, -0.015, "Layer labels show the validation-best encoder depth; selection never uses the test split.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"selected_exit_performance.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths

def _plot_efficiency(output_dir: Path, benchmark_rows: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"b0": "#009E73", "b1": "#0072B2", "p1": "#CC79A7", "p2": "#D55E00", "p3": "#000000", "p2_reverse": "#6A3D9A", "cnn_p0": "#44AA99", "cnn_p1": "#F0E442", "cnn_p2": "#E69F00", "cnn_p2_reverse": "#56B4E9"}
    fig, axes = plt.subplots(2, len(TASKS), figsize=(3.6 * len(TASKS), 6.3), squeeze=False)
    for column, task in enumerate(TASKS):
        available_methods = [method for method in ("b0", *METHODS) if any(row["method"] == method and row["task"] == task for row in benchmark_rows)]
        for method in available_methods:
            task_rows = [row for row in benchmark_rows if row["method"] == method and row["task"] == task]
            task_rows.sort(key=lambda row: row["exit_layer"])
            if not task_rows:
                continue
            axes[0, column].plot(
                [row["exit_layer"] for row in task_rows],
                [row["flops"] / 1e9 for row in task_rows],
                marker="o",
                linewidth=2,
                color=colors[method],
                linestyle={"b0": "-", "b1": "-", "p1": ":", "p2": "--", "p3": "-", "p2_reverse": "-.", "cnn_p0": "-.", "cnn_p1": "-", "cnn_p2": ":", "cnn_p2_reverse": "--"}[method],
                label=FIGURE_METHOD_LABELS[method],
            )
            axes[1, column].plot(
                [row["exit_layer"] for row in task_rows],
                [row["latency_mean_ms"] for row in task_rows],
                marker="o",
                linewidth=2,
                color=colors[method],
                label=FIGURE_METHOD_LABELS[method],
            )
        axes[0, column].set_title(TASK_LABELS[task], fontsize=10, weight="bold")
        axes[1, column].set_title(TASK_LABELS[task], fontsize=10, weight="bold")
        for axis in (axes[0, column], axes[1, column]):
            axis.set_xticks(EXIT_LAYERS)
            axis.set_xlabel("Exit layer")
            axis.grid(axis="y", alpha=0.25, linewidth=0.8)
        axes[0, column].set_ylabel("FLOPs (billions)" if column == 0 else "")
        axes[1, column].set_ylabel("Latency (ms)" if column == 0 else "")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.text(0.5, 0.03, "Batch-1 synchronized CUDA timing; lower is better.", ha="center", fontsize=9)
    fig.suptitle("Inference compute and latency by exit depth", fontsize=14, weight="bold", y=1.065)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96), h_pad=2.2, w_pad=1.3)
    paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"inference_efficiency.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _plot_context_l8_reference(output_dir: Path, source_rows: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    classification = [row for row in source_rows if row["higher_is_better"]]
    position = next(row for row in source_rows if not row["higher_is_better"])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), gridspec_kw={"width_ratios": [3, 1]})
    axes[0].bar(
        [row["task_label"] for row in classification],
        [row["value"] for row in classification],
        color="#009E73",
        edgecolor="#222222",
        linewidth=0.7,
    )
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylabel("Macro accuracy")
    axes[0].set_title("Classification", weight="bold")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar([position["task_label"]], [position["value"]], color="#009E73", edgecolor="#222222", linewidth=0.7)
    axes[1].set_ylabel("Mean distance error (m)")
    axes[1].set_title("Positioning", weight="bold")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    fig.suptitle("Original LatentWave full-depth context-encoder reference", fontsize=14, weight="bold", y=1.04)
    fig.text(0.5, -0.025, "Frozen context encoder at L8; shared 80/10/10 split and validation-selected task heads.", ha="center", fontsize=8.5)
    fig.tight_layout()
    paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"b0_context_l8_reference.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _build_headline_rows(
    source_rows: list[dict[str, Any]],
    comparable_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one deployment-oriented row per method and downstream task."""

    rows = []
    for task in TASKS:
        source = next(row for row in source_rows if row["task"] == task)
        source_benchmark = next(
            row
            for row in benchmark_rows
            if row["method"] == "b0" and row["task"] == task and row["exit_layer"] == 8
        )
        rows.append(
            {
                "method": "b0", "method_label": METHOD_LABELS["b0"], "task": task,
                "task_label": TASK_LABELS[task], "metric": source["metric"],
                "metric_label": source["metric_label"], "value": source["value"],
                "display_value": source["display_value"], "exit_layer": 8,
                "selection": "full-depth L8", "protocol": "frozen context encoder, shared 80/10/10 split",
                "flops": source_benchmark["flops"], "latency_mean_ms": source_benchmark["latency_mean_ms"],
                "latency_p95_ms": source_benchmark["latency_p95_ms"],
            }
        )
        for method in METHODS:
            selected = next(
                row for row in comparable_rows
                if row["method"] == method and row["task"] == task
                and row["evaluation"] == "best_validation"
            )
            benchmark = next(
                (row for row in benchmark_rows
                 if row["method"] == method and row["task"] == task
                 and row["exit_layer"] == selected["exit_layer"]),
                None,
            )
            rows.append(
                {
                    "method": method, "method_label": METHOD_LABELS[method], "task": task,
                    "task_label": TASK_LABELS[task], "metric": selected["metric"],
                    "metric_label": selected["metric_label"], "value": selected["value"],
                    "display_value": selected["display_value"], "exit_layer": selected["exit_layer"],
                    "selection": "best validation-performing exit",
                    "protocol": "frozen context encoder, shared 80/10/10 split",
                    "flops": None if benchmark is None else benchmark["flops"],
                    "latency_mean_ms": None if benchmark is None else benchmark["latency_mean_ms"],
                    "latency_p95_ms": None if benchmark is None else benchmark["latency_p95_ms"],
                }
            )
    return rows


def _build_selected_headline_rows(
    comparable_rows: list[dict[str, Any]], benchmark_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build validation-best deployment rows without injecting a source baseline."""
    rows = []
    for task in TASKS:
        for method in METHODS:
            selected = next(
                row for row in comparable_rows
                if row["method"] == method and row["task"] == task
                and row["evaluation"] == "best_validation"
            )
            benchmark = next(
                (row for row in benchmark_rows
                 if row["method"] == method and row["task"] == task
                 and row["exit_layer"] == selected["exit_layer"]),
                None,
            )
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "metric": selected["metric"],
                    "metric_label": selected["metric_label"],
                    "value": selected["value"],
                    "display_value": selected["display_value"],
                    "exit_layer": selected["exit_layer"],
                    "selection": "best validation-performing exit",
                    "protocol": "frozen context encoder, shared 80/10/10 split",
                    "flops": None if benchmark is None else benchmark["flops"],
                    "latency_mean_ms": None if benchmark is None else benchmark["latency_mean_ms"],
                    "latency_p95_ms": None if benchmark is None else benchmark["latency_p95_ms"],
                }
            )
    return rows

def _plot_all_method_performance(
    output_dir: Path,
    source_rows: list[dict[str, Any]],
    comparable_rows: list[dict[str, Any]],
) -> list[str]:
    """Plot the full-depth reference and all included exit approaches."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = {"b0": "#009E73", "b1": "#0072B2", "p1": "#CC79A7", "p2": "#D55E00", "p3": "#000000", "p2_reverse": "#6A3D9A", "cnn_p0": "#44AA99", "cnn_p1": "#F0E442", "cnn_p2": "#E69F00", "cnn_p2_reverse": "#56B4E9"}
    rows_count, columns_count = _task_grid()
    fig, axes = plt.subplots(rows_count, columns_count, figsize=(5.0 * columns_count, 3.9 * rows_count), squeeze=False)
    for axis, task in zip(axes.flat, TASKS):
        _, _, higher_is_better = TASK_METRICS[task]
        source = next(row for row in source_rows if row["task"] == task)
        axis.plot(
            [8],
            [source["value"]],
            marker="D",
            markersize=7,
            linestyle="None",
            color=colors["b0"],
            label=FIGURE_METHOD_LABELS["b0"],
        )
        for method in METHODS:
            values = [
                next(
                    row["value"]
                    for row in comparable_rows
                    if row["method"] == method
                    and row["task"] == task
                    and row["evaluation"] == "fixed_exit"
                    and row["exit_layer"] == layer
                )
                for layer in EXIT_LAYERS
            ]
            axis.plot(
                EXIT_LAYERS,
                values,
                marker="o",
                linewidth=2.1,
                markersize=5,
                color=colors[method],
                label=FIGURE_METHOD_LABELS[method],
            )
        axis.set_title(TASK_LABELS[task], fontsize=11, weight="bold")
        axis.set_xlabel("Encoder exit layer")
        axis.set_ylabel("Accuracy" if higher_is_better else "Mean distance error (m)")
        axis.set_xticks(EXIT_LAYERS)
        axis.grid(axis="y", alpha=0.25, linewidth=0.8)
        if higher_is_better:
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        else:
            axis.text(0.98, 0.05, "lower is better", transform=axis.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    _hide_unused_axes(axes, len(TASKS))
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS) + 1, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.suptitle("Downstream performance across the approaches", fontsize=14, weight="bold", y=1.045)
    fig.text(
        0.5,
        -0.012,
        "Diamond: Original LatentWave at full-depth L8. Circles: frozen-head exits. All rows use the shared 80/10/10 split and frozen context encoders.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.93))
    paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"all_methods_downstream_performance.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _plot_all_method_selected(output_dir: Path, headline_rows: list[dict[str, Any]]) -> list[str]:
    """Plot the headline deployment choice for all included approaches."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = {"b0": "#009E73", "b1": "#0072B2", "p1": "#CC79A7", "p2": "#D55E00", "p3": "#000000", "p2_reverse": "#6A3D9A", "cnn_p0": "#44AA99", "cnn_p1": "#F0E442", "cnn_p2": "#E69F00", "cnn_p2_reverse": "#56B4E9"}
    methods = ("b0", *METHODS)
    rows_count, columns_count = _task_grid()
    fig, axes = plt.subplots(rows_count, columns_count, figsize=(5.0 * columns_count, 3.9 * rows_count), squeeze=False)
    for axis, task in zip(axes.flat, TASKS):
        _, _, higher_is_better = TASK_METRICS[task]
        task_rows = [row for row in headline_rows if row["task"] == task]
        bars = axis.bar(
            range(len(methods)),
            [next(row["value"] for row in task_rows if row["method"] == method) for method in methods],
            color=[colors[method] for method in methods],
            edgecolor="#222222",
            linewidth=0.7,
        )
        for bar, method in zip(bars, methods):
            row = next(row for row in task_rows if row["method"] == method)
            bar.set_label(FIGURE_METHOD_LABELS[method])
            axis.annotate(
                f"L{row['exit_layer']}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3 if higher_is_better else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if higher_is_better else "top",
                fontsize=8,
            )
        axis.set_xticks(range(len(methods)), [FIGURE_METHOD_SHORT[method] for method in methods])
        axis.set_title(TASK_LABELS[task], fontsize=11, weight="bold")
        axis.set_ylabel("Accuracy" if higher_is_better else "Mean distance error (m)")
        axis.grid(axis="y", alpha=0.25, linewidth=0.8)
        if higher_is_better:
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        else:
            axis.text(0.98, 0.05, "lower is better", transform=axis.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[method], label=FIGURE_METHOD_LABELS[method]) for method in methods]
    _hide_unused_axes(axes, len(TASKS))
    fig.legend(handles=handles, loc="upper center", ncol=len(methods), frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.suptitle("Headline downstream result at each approach's validation-best exit", fontsize=14, weight="bold", y=1.045)
    fig.text(0.5, -0.012, "Layer labels show full-depth L8 for B0 and the validation-best exit for every exit-capable approach.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.025, 1, 0.93))
    paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"all_methods_selected_performance.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _plot_all_method_inference(output_dir: Path, headline_rows: list[dict[str, Any]]) -> list[str]:
    """Plot deployment inference cost; annotate any deliberately unavailable rows."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"b0": "#009E73", "b1": "#0072B2", "p1": "#CC79A7", "p2": "#D55E00", "p3": "#000000", "p2_reverse": "#6A3D9A", "cnn_p0": "#44AA99", "cnn_p1": "#F0E442", "cnn_p2": "#E69F00", "cnn_p2_reverse": "#56B4E9"}
    methods = ("b0", *METHODS)
    fig, axes = plt.subplots(2, len(TASKS), figsize=(3.45 * len(TASKS), 7.1), squeeze=False)
    unavailable_any = False
    for column, task in enumerate(TASKS):
        task_rows = [row for row in headline_rows if row["task"] == task]
        available = [method for method in methods if next(row for row in task_rows if row["method"] == method)["flops"] is not None]
        unavailable = [method for method in methods if method not in available]
        unavailable_any = unavailable_any or bool(unavailable)
        labels = [f"{FIGURE_METHOD_SHORT[method]}\nL{next(row['exit_layer'] for row in task_rows if row['method'] == method)}" for method in available]
        for axis, measure, ylabel in (
            (axes[0, column], "flops", "FLOPs (billions)"),
            (axes[1, column], "latency_mean_ms", "Latency (ms)"),
        ):
            values = [float(next(row[measure] for row in task_rows if row["method"] == method)) for method in available]
            if measure == "flops":
                values = [value / 1e9 for value in values]
            bars = axis.bar(range(len(available)), values, color=[colors[method] for method in available], edgecolor="#222222", linewidth=0.7)
            axis.set_xticks(range(len(available)), labels)
            axis.tick_params(axis="x", labelsize=7.2)
            axis.set_title(TASK_LABELS[task], fontsize=10, weight="bold")
            axis.set_ylabel(ylabel if column == 0 else "")
            axis.grid(axis="y", alpha=0.25, linewidth=0.8)
            for bar, value in zip(bars, values):
                axis.annotate(f"{value:.3f}", xy=(bar.get_x() + bar.get_width() / 2, value), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
            if unavailable:
                axis.text(0.5, 0.97, "Unavailable: " + ", ".join("P2-Reverse" if method == "p2_reverse" else method for method in unavailable), transform=axis.transAxes, ha="center", va="top", fontsize=7, color="#555555")
    legend_methods = [method for method in methods if any(row["method"] == method and row["flops"] is not None for row in headline_rows)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[method], label=FIGURE_METHOD_LABELS[method]) for method in legend_methods]
    fig.legend(handles=handles, loc="upper center", ncol=len(legend_methods), frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=9)
    fig.suptitle("Inference cost at each approach's validation-best exit", fontsize=14, weight="bold", y=1.065)
    note = "Batch-1 synchronized CUDA timing on the RTX 5070 Ti; lower is better. Layer labels are printed below each method."
    if unavailable_any:
        note += " P2-Reverse has no supplied benchmark for this masking condition and is not plotted there."
    fig.text(0.5, 0.03, note, ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96), h_pad=2.2, w_pad=1.3)
    paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"all_methods_inference_cost.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _write_cnn_only_report(
    output_dir: Path,
    packets: dict[str, dict[str, Any]],
    headline_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    figure_paths: list[str],
    *,
    masking_label: str,
    report_title: str,
    provenance_note: str | None,
    split_signature: str,
    condition: dict[str, Any] | None,
) -> None:
    """Write ViT B0/P0/P2 and CNN C0/C2 results in one mask-specific packet."""
    comparable_rows = [row for packet in packets.values() for row in packet["metric_rows"]]
    fixed_rows = [row for row in comparable_rows if row["evaluation"] == "fixed_exit"]
    selected_rows = [row for row in comparable_rows if row["evaluation"] == "best_validation"]
    fixed_table = [[row["method_label"], row["task_label"], f"L{row['exit_layer']}", row["metric_label"], row["display_value"], _fmt_metric(row["task"], row["validation_value"]), row["best_epoch"]] for row in fixed_rows]
    selected_table = [[row["method_label"], row["task_label"], f"L{row['exit_layer']}", row["display_value"]] for row in selected_rows]
    headline_table = [[row["method_label"], row["task_label"], f"L{row['exit_layer']}", row["display_value"]] for row in headline_rows]
    headline_inference_table = [[row["method_label"], row["task_label"], f"L{row['exit_layer']}", f"{row['flops'] / 1e9:.3f}", f"{row['latency_mean_ms']:.4f}", f"{row['latency_p95_ms']:.4f}"] for row in headline_rows]
    inference_table = [[row["method_label"], row["task_label"], f"L{row['exit_layer']}", f"{row['flops'] / 1e9:.3f}", f"{row['latency_mean_ms']:.4f}", f"{row['latency_p95_ms']:.4f}", f"{row['flops_savings_vs_l8'] * 100:.1f}%", f"{row['latency_savings_vs_l8'] * 100:.1f}%"] for row in benchmark_rows]
    scope = [
        ["B0 — Original LatentWave (full-depth)", "ViT P0 checkpoint; full-input L8", "Frozen ViT context encoder at L8", "fixed-depth reference derived from P0"],
        ["P0 — LatentWave + post-hoc frozen exits", "ViT LatentWave P0 objective", "Frozen ViT context encoder at L2/L4/L6/L8", "ViT baseline; head_seed_000"],
        ["P2 — Proposed Hierarchical Multi-Scale JEPA", "ViT P2 objective with multi-scale auxiliary reconstruction", "Frozen ViT context encoder at L2/L4/L6/L8", "ViT hierarchical baseline"],
        ["C0 — CNN original L8 objective", "ConvNeXt V2; one L8 predictor and masked-token MSE", "Frozen CNN context encoder at S1/S2/S3/S4", "single-objective CNN baseline"],
        ["C2 — CNN mask-relative fine-to-coarse", "ConvNeXt V2; L8 anchor plus L2/L4/L6 mask-relative losses", "Frozen CNN context encoder at S1/S2/S3/S4", "hierarchical CNN method"],
    ]
    condition_sentence = (
        f"This {masking_label} packet compares ViT B0/P0/P2 with CNN C0/C2 at seed 0. "
        "B0 is the fixed L8 result from the same ViT P0 checkpoint, not a separate pretraining run."
    )
    geometry_note = ""
    if condition is not None and condition.get("geometry"):
        geometry_note += f"\nThe canonical mask geometry is {condition['geometry']}.\n"
    if condition is not None and condition.get("repeated_geometry_caveat"):
        geometry_note += f"\n{condition['repeated_geometry_caveat']}\n"
    provenance_section = (
        f"## Run-specific provenance\n\n{provenance_note}\n"
        if provenance_note else ""
    )
    report = f"""# {report_title}

{condition_sentence} It contains single-seed point estimates, not a statistical
superiority claim.{geometry_note}

## Scope and comparability

{_markdown_table(['Approach', 'Pretraining', 'Downstream representation', 'Role'], scope)}

CNN stages S1/S2/S3/S4 map to logical exits L2/L4/L6/L8. All learned approaches
use the same persisted 80/10/10 downstream split and validation-only exit
selection before the held-out test split is read. ViT P0 uses the existing
per-profile `head_seed_000` packet; ViT P2 uses the matching-profile hierarchical
packet; B0 reuses P0's fixed L8 result. CNN C0 and C2 use the matched staged
pretraining protocol described in the provenance note.

{provenance_section}

## Task suite

{", ".join(TASK_LABELS[task] for task in TASKS)}. None of these downstream
caches participates in pretraining.

## Validation-best held-out performance

{_markdown_table(['Approach', 'Task', 'Exit', 'Test value'], headline_table)}

## Validation-best inference cost

{_markdown_table(['Approach', 'Task', 'Exit', 'FLOPs (B)', 'Mean ms', 'P95 ms'], headline_inference_table)}

## Fixed-exit downstream metrics

{_markdown_table(['Approach', 'Task', 'Exit', 'Metric', 'Test value', 'Validation value', 'Head epoch'], fixed_table)}

## Detailed validation-best exits

{_markdown_table(['Approach', 'Task', 'Exit', 'Test value'], selected_table)}

## Detailed batch-1 inference cost

{_markdown_table(['Approach', 'Task', 'Exit', 'FLOPs (B)', 'Mean ms', 'P95 ms', 'FLOPs saving', 'Latency saving'], inference_table)}

## Figures

{chr(10).join(f'- `{Path(path).name}`' for path in figure_paths)}
"""
    (output_dir / "README.md").write_text(report)
    markdown_tables = {
        "table_primary_fixed_exits.md": ("# Fixed-exit downstream metrics\n\n", ['Approach', 'Task', 'Exit', 'Metric', 'Test value', 'Validation value', 'Head epoch'], fixed_table),
        "table_selected_exits.md": ("# Validation-best exits\n\n", ['Approach', 'Task', 'Exit', 'Test value'], selected_table),
        "table_inference_cost.md": ("# Batch-1 inference cost\n\n", ['Approach', 'Task', 'Exit', 'FLOPs (B)', 'Mean ms', 'P95 ms', 'FLOPs saving', 'Latency saving'], inference_table),
        "table_all_methods_downstream.md": ("# ViT and CNN validation-best held-out performance\n\n", ['Approach', 'Task', 'Exit', 'Test value'], headline_table),
        "table_all_methods_inference.md": ("# ViT and CNN validation-best inference cost\n\n", ['Approach', 'Task', 'Exit', 'FLOPs (B)', 'Mean ms', 'P95 ms'], headline_inference_table),
    }
    for filename, (title, headers, rows) in markdown_tables.items():
        (output_dir / filename).write_text(title + _markdown_table(headers, rows) + "\n")
    csv_rows = {
        "table_primary_fixed_exits.csv": fixed_rows,
        "table_selected_exits.csv": selected_rows,
        "table_inference_cost.csv": benchmark_rows,
        "table_all_methods_downstream.csv": headline_rows,
        "table_all_methods_inference.csv": headline_rows,
    }
    for filename, rows in csv_rows.items():
        fields = sorted({key for row in rows for key in row})
        with (output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    provenance = {}
    for method, packet in packets.items():
        checkpoint = Path(packet["checkpoint"])
        experiment = checkpoint.parents[1] / "experiment.json"
        experiment_payload = _read(experiment) if experiment.is_file() else None
        provenance[method] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "pretraining_experiment": str(experiment) if experiment_payload is not None else None,
            "pretraining_metadata": None if experiment_payload is None else experiment_payload.get("metadata"),
            "pretraining_runtime": None if experiment_payload is None else experiment_payload.get("runtime"),
            "metrics_path": packet["metrics_path"],
            "benchmark_path": packet["benchmark_path"],
            "metrics_protocol": packet.get("metrics_protocol"),
            "pretraining_method": packet.get("pretraining_method"),
        }
    _write_json(output_dir / "manifest.json", {
        "report": (
            f"vit_b0_p0_p2_cnn_c0_c2_{condition['name']}_single_seed"
            if condition is not None else "vit_b0_p0_p2_cnn_c0_c2_multi_block_n6_single_seed"
        ),
        "comparison_mode": "vit_b0_p0_p2_with_cnn_c0_c2",
        "git_revision": _git_revision(),
        "methods": [METHOD_LABELS[method] for method in ("b0", *METHODS)],
        "b0_reference": {
            "derived_from_method": "b1",
            "checkpoint": packets["b1"]["checkpoint"],
            "exit_layer": 8,
            "interpretation": "fixed full-depth reference using the same ViT P0 checkpoint and held-out head results",
        },
        "vit_p0_baseline": {
            "pretrain_seed": 0,
            "head_seed": 0,
            "metrics_path": packets["b1"]["metrics_path"],
            "benchmark_path": packets["b1"]["benchmark_path"],
            "checkpoint": packets["b1"]["checkpoint"],
        },
        "vit_p2_baseline": {
            "pretrain_seed": 0,
            "head_seed": 0,
            "metrics_path": packets["p2"]["metrics_path"],
            "benchmark_path": packets["p2"]["benchmark_path"],
            "checkpoint": packets["p2"]["checkpoint"],
        },
        "seed": 0,
        "masking": masking_label,
        "canonical_mask_profile": condition,
        "task_suite": list(TASKS),
        "headline_exit_policy": "best_validation",
        "downstream_split_signature": hashlib.sha256(split_signature.encode()).hexdigest(),
        "method_provenance": provenance,
        "figures": [Path(path).name for path in figure_paths],
        "provenance_note": provenance_note,
    })


def _resolve_vit_p0_packet_paths(vit_run_root: Path) -> tuple[Path, Path]:
    """Resolve supported layouts for the matching-mask ViT P0 packet."""

    candidates = (
        (
            vit_run_root / "head_seed_000/p0/seed_000/downstream/exit_metrics.json",
            vit_run_root / "benchmarks/p0/seed_000/benchmarks.json",
        ),
        (
            vit_run_root / "p0/seed_000/downstream/exit_metrics.json",
            vit_run_root / "p0/seed_000/benchmarks.json",
        ),
    )
    for metrics_path, benchmark_path in candidates:
        if metrics_path.is_file() and benchmark_path.is_file():
            return metrics_path, benchmark_path
    expected = "\n".join(f"- {metrics}\n  {benchmark}" for metrics, benchmark in candidates)
    raise FileNotFoundError(
        f"No complete ViT P0 metrics/benchmark packet under {vit_run_root}. Expected:\n{expected}"
    )


def _run_cnn_only_report(args: argparse.Namespace, cnn_run_root: Path, output_dir: Path) -> int:
    if tuple(args.cnn_methods or ()) != ("cnn_p0", "cnn_p2"):
        raise ValueError("cnn_only currently requires exactly: --cnn-methods cnn_p0 cnn_p2")
    global TASKS, METHODS
    cnn_methods = tuple(args.cnn_methods)
    METHODS = ("b1", "p2", *cnn_methods)
    profile_name = getattr(args, "mask_profile", None) or "multi_block_n6"
    vit_run_root = getattr(args, "vit_p0_run_root", None)
    if vit_run_root is not None:
        vit_run_root = _repo_path(vit_run_root)
        vit_metrics_path, vit_benchmark_path = _resolve_vit_p0_packet_paths(vit_run_root)
    else:
        candidates = (
            REPOSITORY_ROOT / "runs/masking_head_seeds" / profile_name,
            REPOSITORY_ROOT / "runs/masking_study" / profile_name,
        )
        resolved = None
        for candidate in candidates:
            try:
                resolved = (candidate, *_resolve_vit_p0_packet_paths(candidate))
                break
            except FileNotFoundError:
                continue
        if resolved is None:
            raise FileNotFoundError(
                f"No ViT P0 packet found for mask profile {profile_name!r} in: "
                + ", ".join(str(path) for path in candidates)
            )
        vit_run_root, vit_metrics_path, vit_benchmark_path = resolved
    if vit_run_root.name != profile_name:
        raise ValueError(
            f"ViT P0 root {vit_run_root} does not match mask profile {profile_name!r}"
        )
    vit_p2_run_root = getattr(args, "vit_p2_run_root", None)
    if vit_p2_run_root is None:
        p2_candidates = (
            REPOSITORY_ROOT / "runs/masking_study" / profile_name,
            REPOSITORY_ROOT / "runs/masking_head_seeds" / profile_name,
        )
        vit_p2_run_root = next(
            (candidate for candidate in p2_candidates
             if (candidate / "p2/seed_000/downstream/exit_metrics.json").is_file()
             and (candidate / "p2/seed_000/benchmarks.json").is_file()),
            None,
        )
        if vit_p2_run_root is None:
            raise FileNotFoundError(
                f"No matching-mask ViT P2 packet found for profile {profile_name!r}"
            )
    else:
        vit_p2_run_root = _repo_path(vit_p2_run_root)
    if vit_p2_run_root.name != profile_name:
        raise ValueError(
            f"ViT P2 root {vit_p2_run_root} does not match mask profile {profile_name!r}"
        )
    vit_p2_metrics_path = vit_p2_run_root / "p2/seed_000/downstream/exit_metrics.json"
    vit_p2_benchmark_path = vit_p2_run_root / "p2/seed_000/benchmarks.json"
    raw_packets = {"b1": _read(vit_metrics_path), "p2": _read(vit_p2_metrics_path)}
    raw_packets.update({
        method: _read(cnn_run_root / method / "seed_000/downstream/exit_metrics.json")
        for method in cnn_methods
    })
    for method, packet in raw_packets.items():
        expected_method = "p0" if method == "b1" else method
        if packet.get("method") != expected_method:
            raise ValueError(f"{method} metrics packet identifies {packet.get('method')!r}")
    TASKS, split_signature = _discover_tasks_from_packets(raw_packets)
    packets = {
        "b1": _load_exit_results(
            vit_run_root, "b1", source_method="p0",
            metrics_path=vit_metrics_path, benchmark_path=vit_benchmark_path,
        )[0],
        "p2": _load_exit_results(
            vit_p2_run_root, "p2", source_method="p2",
            metrics_path=vit_p2_metrics_path, benchmark_path=vit_p2_benchmark_path,
        )[0],
        **{
            method: _load_exit_results(cnn_run_root, method, source_method=method)[0]
            for method in cnn_methods
        },
    }
    profile_name = getattr(args, "mask_profile", None)
    condition = (
        _validate_cnn_profile_provenance(
            {method: packets[method] for method in cnn_methods}, profile_name
        )
        if profile_name is not None else None
    )
    b0_reference, b0_metric_rows = _load_context_l8_reference(packets["b1"])
    b0_benchmark_rows = []
    for row in packets["b1"]["benchmark_rows"]:
        if row["exit_layer"] == 8:
            b0_row = dict(row)
            b0_row.update({
                "method": "b0",
                "method_label": METHOD_LABELS["b0"],
                "protocol": "ViT P0 checkpoint, fixed full-depth L8 benchmark",
            })
            b0_benchmark_rows.append(b0_row)
    if len(b0_benchmark_rows) != len(TASKS):
        raise ValueError("ViT P0 baseline requires a complete L8 benchmark for every task")
    b0_packet = {
        **b0_reference,
        "metrics_path": packets["b1"]["metrics_path"],
        "benchmark_path": packets["b1"]["benchmark_path"],
        "metric_rows": b0_metric_rows,
        "benchmark_rows": b0_benchmark_rows,
    }
    report_packets = {"b0": b0_packet, **packets}
    comparable_rows = [row for packet in report_packets.values() for row in packet["metric_rows"]]
    benchmark_rows = [row for packet in report_packets.values() for row in packet["benchmark_rows"]]
    expected_benchmarks = len(packets) * len(TASKS) * len(EXIT_LAYERS)
    actual_benchmarks = sum(len(packet["benchmark_rows"]) for packet in packets.values())
    if actual_benchmarks != expected_benchmarks:
        raise ValueError("ViT P0/P2 and CNN C0/C2 require complete benchmarks for every task and exit")
    headline_rows = _build_headline_rows(b0_metric_rows, comparable_rows, benchmark_rows)
    figure_paths = []
    figure_paths.extend(_plot_performance(output_dir, comparable_rows))
    figure_paths.extend(_plot_selected(output_dir, comparable_rows))
    figure_paths.extend(_plot_efficiency(output_dir, benchmark_rows))
    _write_cnn_only_report(
        output_dir, report_packets, headline_rows, benchmark_rows, figure_paths,
        masking_label=args.masking_label, report_title=args.report_title,
        provenance_note=args.provenance_note, split_signature=split_signature,
        condition=condition,
    )
    print(json.dumps({"output_dir": str(output_dir), "figures": len(figure_paths), "methods": [METHOD_LABELS[method] for method in ("b0", *METHODS)]}, indent=2))
    return 0

def _build_report(
    output_dir: Path,
    source_reference: dict[str, Any],
    source_rows: list[dict[str, Any]],
    b1_packet: dict[str, Any],
    p2_packet: dict[str, Any],
    headline_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    figure_paths: list[str],
    *,
    report_title: str,
    masking_label: str,
    provenance_note: str | None,
    p1_packet: dict[str, Any] | None = None,
    p3_packet: dict[str, Any] | None = None,
    p2_reverse_packet: dict[str, Any] | None = None,
) -> None:
    packets = [b1_packet] + ([p1_packet] if p1_packet is not None else []) + [p2_packet] + ([p3_packet] if p3_packet is not None else []) + ([p2_reverse_packet] if p2_reverse_packet is not None else [])
    comparable_rows = [row for packet in packets for row in packet["metric_rows"]]
    detailed_benchmark_rows = [row for row in benchmark_rows if row["method"] in METHODS]
    fixed_rows = [row for row in comparable_rows if row["evaluation"] == "fixed_exit"]
    selected_rows = [row for row in comparable_rows if row["evaluation"] == "best_validation"]
    fixed_table = []
    for row in fixed_rows:
        fixed_table.append(
            [
                row["method_label"],
                row["task_label"],
                f"L{row['exit_layer']}",
                row["metric_label"],
                row["display_value"],
                _fmt_metric(row["task"], row["validation_value"]),
                row["best_epoch"],
            ]
        )
    selected_table = []
    for row in selected_rows:
        selected_table.append(
            [
                row["method_label"],
                row["task_label"],
                row["evaluation"].replace("_", " "),
                f"L{row['exit_layer']}",
                row["display_value"],
            ]
        )
    source_table = [[row["task_label"], row["metric_label"], row["display_value"], row["examples"]] for row in source_rows]
    headline_table = []
    for row in headline_rows:
        headline_table.append(
            [
                row["method_label"],
                row["task_label"],
                f"L{row['exit_layer']}",
                row["selection"],
                row["display_value"],
            ]
        )
    headline_inference_table = []
    for row in headline_rows:
        headline_inference_table.append(
            [
                row["method_label"],
                row["task_label"],
                f"L{row['exit_layer']}",
                "n/a" if row["flops"] is None else f"{row['flops'] / 1e9:.3f}",
                "n/a" if row["latency_mean_ms"] is None else f"{row['latency_mean_ms']:.4f}",
                "n/a" if row["latency_p95_ms"] is None else f"{row['latency_p95_ms']:.4f}",
            ]
        )
    compute_table = []
    for row in detailed_benchmark_rows:
        compute_table.append(
            [
                row["method_label"],
                row["task_label"],
                f"L{row['exit_layer']}",
                f"{row['flops'] / 1e9:.3f}",
                f"{row['latency_mean_ms']:.4f}",
                f"{row['latency_p95_ms']:.4f}",
                f"{row['flops_savings_vs_l8'] * 100:.1f}%",
                f"{row['latency_savings_vs_l8'] * 100:.1f}%",
            ]
        )
    approach_scope = [
        ["Original LatentWave", "Original LatentWave training", "Frozen context encoder, full-input L8", "Full-depth baseline"],
        ["LatentWave with Post-hoc Frozen Early Exits", "Reuses the Original LatentWave checkpoint", "Frozen context encoder at L2/L4/L6/L8", "Early-exit baseline"],
    ]
    cnn_methods = p2_packet.get("method") == "cnn_p2"
    if cnn_methods:
        approach_scope.append(["CNN-HierJEPA P1 — Same-scale Multi-depth", "ConvNeXt V2 CNN pretraining with same-scale intermediate targets", "Frozen CNN context encoder at S1/S2/S3/S4 (L2/L4/L6/L8)", "Same-scale auxiliary-loss control"])
        approach_scope.append(["CNN-HierJEPA P2 — Mask-relative Fine-to-coarse", "ConvNeXt V2 CNN pretraining with mask-relative hierarchical targets", "Frozen CNN context encoder at S1/S2/S3/S4 (L2/L4/L6/L8)", "Proposed method"])
    else:
        if p1_packet is not None:
            approach_scope.append(["Same-scale Multi-depth JEPA", "P1 pretraining with same-scale intermediate targets", "Frozen context encoder at L2/L4/L6/L8", "Same-scale auxiliary-loss control"])
        approach_scope.append(["P2 — Proposed Hierarchical Multi-Scale JEPA", "Separately pretrained fine-to-coarse hierarchical JEPA", "Frozen context encoder at L2/L4/L6/L8", "Proposed method"])
        if p3_packet is not None:
            approach_scope.append(["P3 — Pair-local Hierarchical Multi-Scale JEPA", "P2 objective with L2/L4/L6 auxiliary gradients limited to block pairs 1–2, 3–4, and 5–6", "Frozen context encoder at L2/L4/L6/L8", "Gradient-routing ablation"])
    if p2_reverse_packet is not None:
        if cnn_methods:
            approach_scope.append(["CNN-HierJEPA P2-Reverse — Coarse-to-fine", "ConvNeXt V2 CNN pretraining with reversed mask-relative target scales", "Frozen CNN context encoder at S1/S2/S3/S4 (L2/L4/L6/L8)", "Hierarchy-direction ablation"])
        else:
            approach_scope.append(["P2-Reverse — Coarse-to-fine Hierarchical JEPA", "Imported reverse-hierarchy pretraining artifact", "Frozen context encoder at L2/L4/L6/L8", "Exploratory hierarchy-direction ablation"])
    provenance_section = (
        f"## Run-specific provenance\n\n{provenance_note}\n"
        if provenance_note else ""
    )
    p1_provenance_section = ""
    if p1_packet is not None:
        p1_checkpoint = p1_packet["checkpoint"]
        p1_experiment = Path(p1_checkpoint).parents[1] / "experiment.json"
        p1_provenance_section = (
            "## P1 checkpoint provenance\n\n"
            f"{p1_packet['label']} checkpoint: `{p1_checkpoint}`.\n\n"
            f"Pretraining experiment metadata: `{p1_experiment}`.\n"
        )
    report = f"""# {report_title}

This directory contains the submission-oriented result tables and figures for
the completed single-seed pilot. It is an aggregation of recorded
artifacts, not a statistical claim. The report was generated from repository
revision `{_git_revision()}`.

## Scope and comparability

{_markdown_table(['Approach', 'Pretraining', 'Downstream representation', 'Role'], approach_scope)}

All rows use independent frozen task heads, the same persisted 80/10/10
split, and validation-only head/exit selection. Original LatentWave is the
P0 context encoder evaluated at L8; the post-hoc early-exit baseline reuses
that exact P0 checkpoint at L2/L4/L6/L8. CNN-HierJEPA maps stages S1/S2/S3/S4
to logical exits L2/L4/L6/L8. S4 is nearest-neighbor upsampled from 7x7 to the
canonical 14x14 grid, duplicating tokens and changing final-stage token
weighting. Locally generated CNN methods require an exact shared split-manifest
match; imported ViT P2-Reverse artifacts may lack that manifest.

All values are single-seed pilot point estimates.

{provenance_section}

## Task-suite note

{p1_provenance_section}
This packet covers **{masking_label}** and the following downstream task suite:
{", ".join(TASK_LABELS[task] for task in TASKS)}. Jamming and interference
are labelled frozen-head tasks only; none of the six downstream caches is used
for pretraining.

## All-approach headline downstream comparison

The manuscript-facing rows use full-depth L8 B0 and the validation-best
performing exit for every exit-capable approach. The layer is chosen using the
validation metric only (maximum accuracy or minimum positioning error); the
held-out test split is never used for exit selection.

{_markdown_table(['Approach', 'Task', 'Exit', 'Selection', 'Test value'], headline_table)}

## All-approach headline inference comparison

{_markdown_table(['Approach', 'Task', 'Exit', 'FLOPs (B)', 'Mean ms', 'P95 ms'], headline_inference_table)}

## Original LatentWave full-depth context-encoder reference

{_markdown_table(['Task', 'Metric', 'Test value'], [row[:3] for row in source_table])}

## Detailed frozen-exit downstream metrics

Test values are shown after each head's validation-selected checkpoint. The
validation column is included for auditability; it is not a test result.

{_markdown_table(['Method', 'Task', 'Exit', 'Metric', 'Test value', 'Validation value', 'Head epoch'], fixed_table)}

## Detailed validation-best exits

The manuscript-facing exit is selected from validation only. For positioning,
lower mean distance error is better; for classification, higher macro
accuracy is better.

{_markdown_table(['Method', 'Task', 'Policy', 'Exit', 'Test value'], selected_table)}

## Detailed batch-1 inference cost

Latency is the synchronized CUDA mean/p95 measurement after warm-up on the
RTX 5070 Ti. Savings are relative to the same method/task at L8. A missing P2-Reverse
benchmark is reported as unavailable, not estimated.

{_markdown_table(['Method', 'Task', 'Exit', 'FLOPs (B)', 'Mean ms', 'P95 ms', 'FLOPs saving', 'Latency saving'], compute_table)}

## Figures

The PNG files are 300-DPI raster exports and the PDF files are vector exports.

{chr(10).join(f'- `{Path(path).name}`' for path in figure_paths)}

## Exclusions

Diagnostic early-stopped pretraining runs, extra transfer datasets, extra
seeds, and alternate masking regimes are excluded. The same-scale auxiliary JEPA control is included when its completed P1 artifacts are supplied.
"""
    (output_dir / "README.md").write_text(report)

    (output_dir / "table_b0_context_l8_reference.md").write_text(
        "# Original LatentWave full-depth context-encoder reference\n\n"
        + _markdown_table(["Task", "Metric", "Test value"], [row[:3] for row in source_table])
        + "\n"
    )
    (output_dir / "table_primary_fixed_exits.md").write_text(
        "# Detailed frozen-exit downstream metrics\n\n" + _markdown_table(["Approach", "Task", "Exit", "Metric", "Test value", "Validation value", "Head epoch"], fixed_table) + "\n"
    )
    (output_dir / "table_selected_exits.md").write_text(
        "# Detailed validation-best exits\n\n" + _markdown_table(["Approach", "Task", "Policy", "Exit", "Test value"], selected_table) + "\n"
    )
    (output_dir / "table_inference_cost.md").write_text(
        "# Detailed batch-1 inference cost\n\n" + _markdown_table(["Approach", "Task", "Exit", "FLOPs (B)", "Mean ms", "P95 ms", "FLOPs saving", "Latency saving"], compute_table) + "\n"
    )
    (output_dir / "table_all_methods_downstream.md").write_text(
        "# All-approach headline downstream comparison\n\n"
        + _markdown_table(["Approach", "Task", "Exit", "Selection", "Test value"], headline_table)
        + "\n"
    )
    (output_dir / "table_all_methods_inference.md").write_text(
        "# All-approach headline inference comparison\n\n"
        + _markdown_table(["Approach", "Task", "Exit", "FLOPs (B)", "Mean ms", "P95 ms"], headline_inference_table)
        + "\n"
    )

    import csv

    with (output_dir / "table_primary_fixed_exits.csv").open("w", newline="") as handle:
        fields = ["method", "task", "exit_layer", "metric", "value", "validation_value", "best_epoch", "head_epochs_completed", "head_stopped_early"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in fixed_rows)
    with (output_dir / "table_selected_exits.csv").open("w", newline="") as handle:
        fields = ["method", "task", "evaluation", "exit_layer", "metric", "value", "higher_is_better"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in selected_rows)
    with (output_dir / "table_b0_context_l8_reference.csv").open("w", newline="") as handle:
        fields = ["method", "task", "metric", "value", "protocol"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in source_rows)
    with (output_dir / "table_inference_cost.csv").open("w", newline="") as handle:
        fields = ["method", "task", "exit_layer", "flops", "latency_mean_ms", "latency_median_ms", "latency_p95_ms", "latency_std_ms", "flops_savings_vs_l8", "latency_savings_vs_l8"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in detailed_benchmark_rows)
    with (output_dir / "table_all_methods_downstream.csv").open("w", newline="") as handle:
        fields = ["method", "method_label", "task", "exit_layer", "selection", "metric", "value", "protocol"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in headline_rows)
    with (output_dir / "table_all_methods_inference.csv").open("w", newline="") as handle:
        fields = ["method", "method_label", "task", "exit_layer", "flops", "latency_mean_ms", "latency_p95_ms", "protocol"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in headline_rows)


def main() -> int:
    args = _arguments()
    run_root = _repo_path(args.run_root)
    p2_run_root = _repo_path(args.p2_run_root) if args.p2_run_root is not None else run_root
    p1_run_root = _repo_path(args.p1_run_root) if args.p1_run_root is not None else None
    p3_run_root = _repo_path(args.p3_run_root) if args.p3_run_root is not None else None
    p2_reverse_root = _repo_path(args.p2_reverse_import_root) if args.p2_reverse_import_root is not None else None
    p2_reverse_metrics_path = p2_reverse_root / "downstream/exit_metrics.json" if p2_reverse_root is not None else None
    source_root = _repo_path(args.source_root)
    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(output_dir / ".matplotlib")
    cnn_run_root = _repo_path(args.cnn_run_root) if args.cnn_run_root is not None else None
    if args.comparison_mode == "cnn_only":
        if cnn_run_root is None:
            raise ValueError("--comparison-mode cnn_only requires --cnn-run-root")
        if args.cnn_methods is None:
            raise ValueError("--comparison-mode cnn_only requires --cnn-methods cnn_p0 cnn_p2")
        return _run_cnn_only_report(args, cnn_run_root, output_dir)
    if args.cnn_methods is not None:
        raise ValueError("--cnn-methods is only valid with --comparison-mode cnn_only")
    if cnn_run_root is not None and (p1_run_root is not None or p3_run_root is not None or p2_reverse_root is not None):
        raise ValueError("--cnn-run-root supplies CNN P1/P2/P2-Reverse; do not combine it with ViT P1, P3, or P2-Reverse inputs")
    global TASKS, METHODS
    METHODS = ("b1", "cnn_p0", "cnn_p1", "cnn_p2", "cnn_p2_reverse") if cnn_run_root is not None else (("b1", "p1", "p2") if p1_run_root is not None else ("b1", "p2"))
    if cnn_run_root is None and p3_run_root is not None:
        METHODS = (*METHODS, "p3")
    if cnn_run_root is None and p2_reverse_root is not None:
        METHODS = (*METHODS, "p2_reverse")
    TASKS = _discover_tasks(run_root, p2_run_root, p1_run_root, p3_run_root, p2_reverse_metrics_path, cnn_run_root)

    b1_packet, _ = _load_exit_results(run_root, "b1")
    source_reference, source_rows = _load_context_l8_reference(b1_packet)
    if cnn_run_root is not None:
        cnn_p0_packet = _load_exit_results(cnn_run_root, "cnn_p0", source_method="cnn_p0")[0]
        p1_packet = _load_exit_results(cnn_run_root, "cnn_p1", source_method="cnn_p1")[0]
        p2_packet = _load_exit_results(cnn_run_root, "cnn_p2", source_method="cnn_p2")[0]
        p2_reverse_packet = _load_exit_results(cnn_run_root, "cnn_p2_reverse", source_method="cnn_p2_reverse")[0]
        p3_packet = None
    else:
        cnn_p0_packet = None
        p2_packet, _ = _load_exit_results(p2_run_root, "p2")
        p1_packet = _load_exit_results(p1_run_root, "p1")[0] if p1_run_root is not None else None
        p3_packet = _load_exit_results(p3_run_root, "p3")[0] if p3_run_root is not None else None
        p2_reverse_packet = (_load_exit_results(p2_reverse_root, "p2_reverse", source_method="p2_reverse", metrics_path=p2_reverse_metrics_path, benchmark_path=p2_reverse_root / "benchmarks.json")[0] if p2_reverse_root is not None else None)
    comparable_packets = [b1_packet] + ([cnn_p0_packet] if cnn_p0_packet is not None else []) + ([p1_packet] if p1_packet is not None else []) + [p2_packet] + ([p3_packet] if p3_packet is not None else []) + ([p2_reverse_packet] if p2_reverse_packet is not None else [])
    comparable_rows = [row for packet in comparable_packets for row in packet["metric_rows"]]
    b0_benchmark_rows = []
    for row in b1_packet["benchmark_rows"]:
        if row["exit_layer"] != 8:
            continue
        b0_row = dict(row)
        b0_row["method"] = "b0"
        b0_row["method_label"] = METHOD_LABELS["b0"]
        b0_row["protocol"] = "Original LatentWave frozen context encoder, full-depth L8 benchmark"
        b0_benchmark_rows.append(b0_row)
    benchmark_rows = b0_benchmark_rows + [row for packet in comparable_packets for row in packet["benchmark_rows"]]
    headline_rows = _build_headline_rows(source_rows, comparable_rows, benchmark_rows)
    figure_paths = []
    figure_paths.extend(_plot_all_method_performance(output_dir, source_rows, comparable_rows))
    figure_paths.extend(_plot_all_method_selected(output_dir, headline_rows))
    figure_paths.extend(_plot_all_method_inference(output_dir, headline_rows))
    figure_paths.extend(_plot_performance(output_dir, comparable_rows))
    figure_paths.extend(_plot_selected(output_dir, comparable_rows))
    figure_paths.extend(_plot_efficiency(output_dir, [row for row in benchmark_rows if row["method"] in METHODS]))
    figure_paths.extend(_plot_context_l8_reference(output_dir, source_rows))
    _build_report(
        output_dir, source_reference, source_rows, b1_packet, p2_packet, headline_rows,
        benchmark_rows, figure_paths, report_title=args.report_title,
        masking_label=args.masking_label, provenance_note=args.provenance_note,
        p1_packet=p1_packet, p3_packet=p3_packet, p2_reverse_packet=p2_reverse_packet,
    )
    manifest = {
        "report": "phase5_primary_single_seed",
        "git_revision": _git_revision(),
        "run_root": str(run_root), "p2_run_root": str(p2_run_root),
        "p1_checkpoint": p1_packet.get("checkpoint") if p1_packet is not None else None,
        "p1_pretraining_experiment": str(Path(p1_packet["checkpoint"]).parents[1] / "experiment.json") if p1_packet is not None else None,
        "p1_run_root": str(p1_run_root) if p1_run_root is not None else None,
        "p3_run_root": str(p3_run_root) if p3_run_root is not None else None,
        "p3_checkpoint": p3_packet.get("checkpoint") if p3_packet is not None else None,
        "cnn_run_root": str(cnn_run_root) if cnn_run_root is not None else None,
        "p2_reverse_import_root": str(p2_reverse_root) if p2_reverse_root is not None else None,
        "p2_reverse_benchmark_available": p2_reverse_packet["benchmark_available"] if p2_reverse_packet is not None else None,
        "p2_reverse_split_manifest_verification": ("verified locally" if cnn_run_root is not None else "not available in imported archive") if p2_reverse_packet is not None else None,
        "p2_reverse_archive": {"source_commit_claimed": "90c4793ac9ce6a174bb39138c7fc324e742a9147", "sha256": "83aeed0d9292cf89be5f1ed63e3b897bfce5dee3b41a050cba2bdf68c5c38ad6"} if p2_reverse_packet is not None and cnn_run_root is None else None,
        "source_root": None, "output_dir": str(output_dir),
        "methods": [METHOD_LABELS[method] for method in ("b0", *METHODS)],
        "seed": 0, "masking": args.masking_label, "task_suite": list(TASKS),
        "headline_exit_policy": "best_validation",
        "included_inputs": {
            "b0_context_l8": str(run_root / "p0/seed_000/downstream/exit_metrics.json"),
            "b1_metrics": str(run_root / "p0/seed_000/downstream/exit_metrics.json"),
            "b1_benchmark": str(run_root / "p0/seed_000/benchmarks.json"),
            "p2_metrics": str((cnn_run_root / "cnn_p2/seed_000/downstream/exit_metrics.json") if cnn_run_root is not None else (p2_run_root / "p2/seed_000/downstream/exit_metrics.json")),
            "p2_benchmark": str((cnn_run_root / "cnn_p2/seed_000/benchmarks.json") if cnn_run_root is not None else (p2_run_root / "p2/seed_000/benchmarks.json")),
            **({"p1_metrics": str(p1_run_root / "p1/seed_000/downstream/exit_metrics.json"), "p1_benchmark": str(p1_run_root / "p1/seed_000/benchmarks.json")} if p1_run_root is not None else {}),
            **({"p3_metrics": str(p3_run_root / "p3/seed_000/downstream/exit_metrics.json"), "p3_benchmark": str(p3_run_root / "p3/seed_000/benchmarks.json")} if p3_run_root is not None else {}),
            **({"cnn_p0_metrics": str(cnn_run_root / "cnn_p0/seed_000/downstream/exit_metrics.json"), "cnn_p0_benchmark": str(cnn_run_root / "cnn_p0/seed_000/benchmarks.json"), "cnn_p1_metrics": str(cnn_run_root / "cnn_p1/seed_000/downstream/exit_metrics.json"), "cnn_p1_benchmark": str(cnn_run_root / "cnn_p1/seed_000/benchmarks.json"), "cnn_p2_reverse_metrics": str(cnn_run_root / "cnn_p2_reverse/seed_000/downstream/exit_metrics.json"), "cnn_p2_reverse_benchmark": str(cnn_run_root / "cnn_p2_reverse/seed_000/benchmarks.json")} if cnn_run_root is not None else {}),
            **({"p2_reverse_metrics": str(p2_reverse_metrics_path), "p2_reverse_benchmark": str(p2_reverse_root / "benchmarks.json") if p2_reverse_packet["benchmark_available"] else None} if p2_reverse_packet is not None else {}),
        },
        "excluded": ["EMA-teacher source-style probe (archived provenance only)", "diagnostic early-stopped runs", "extra transfer datasets", "seeds 1 and 2", "alternate masking regimes"],
        "figures": [Path(path).name for path in figure_paths],
        "provenance_note": args.provenance_note,
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "figures": len(figure_paths), "methods": manifest["methods"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
