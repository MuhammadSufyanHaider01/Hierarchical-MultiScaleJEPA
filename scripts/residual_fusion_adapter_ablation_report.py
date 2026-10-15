#!/usr/bin/env python3
"""Build a fail-closed P0/P2, single-seed Fusion V2 adapter-ablation report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("p0", "p2")
VARIANTS = ("bottleneck_linear", "bottleneck_sigmoid", "single_linear", "single_gelu", "single_sigmoid")
ALGORITHMS = ("current", "bottleneck_gelu", *VARIANTS)
TASKS = ("rfs", "jamming", "human-activity", "interference", "nrpos-outdoor", "deepmimo-beam")
EXITS = (2, 4, 6, 8)
CHECKPOINT_HASHES = {
    "p0": "8faa7245b6692e2d74d753fde2268fb712264c02c1da19632c045d022d192f1f",
    "p2": "7e1e27d96183b8f9d1c0d6b7493b93945633ccfb1514555135b4d04f22bf18ac",
}
CURRENT_ROOT = ROOT / "runs/masking_head_seeds/multi_block_n6"
GELU_ROOT = ROOT / "runs/ee_optimization/multi_block_n6/residual_dynamic_fusion"
ABLATION_ROOT = ROOT / "runs/ee_optimization/multi_block_n6/residual_fusion_adapter_ablation"
OUT = ROOT / "reports/early_exit_optimization/multi_block_n6/residual_fusion_adapter_ablation_single_seed"
LABELS = {
    "current": "Current EE", "bottleneck_gelu": "Bottleneck GELU — existing V2",
    "bottleneck_linear": "Bottleneck Linear", "bottleneck_sigmoid": "Bottleneck Sigmoid",
    "single_linear": "Single Linear", "single_gelu": "Single GELU", "single_sigmoid": "Single Sigmoid (centered)",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packet_path(method: str, algorithm: str, *, current_root=CURRENT_ROOT, gelu_root=GELU_ROOT, ablation_root=ABLATION_ROOT) -> Path:
    if algorithm == "current":
        return current_root / "head_seed_000" / method / "seed_000/downstream/exit_metrics.json"
    if algorithm == "bottleneck_gelu":
        return gelu_root / "head_seed_000" / method / "seed_000/downstream/exit_metrics.json"
    return ablation_root / "head_seed_000" / algorithm / method / "seed_000/downstream/exit_metrics.json"


def _validate_packet(method: str, algorithm: str, packet: dict, path: Path) -> None:
    if packet.get("method") != method or int(packet.get("pretrain_seed", -1)) != 0 or int(packet.get("head_seed", -1)) != 0:
        raise ValueError(f"packet provenance mismatch: {path}")
    if packet.get("split_seed") not in (None, 2025) or packet.get("protocol", {}).get("test_evaluation") is not True:
        raise ValueError(f"packet protocol incomplete: {path}")
    if algorithm == "bottleneck_gelu" and packet.get("head_algorithm") != "residual_projected_dynamic_fusion":
        raise ValueError(f"GELU baseline identity mismatch: {path}")
    if algorithm in VARIANTS and (packet.get("head_algorithm") != "residual_fusion_adapter_ablation" or packet.get("adapter_variant") != algorithm):
        raise ValueError(f"ablation identity mismatch: {path}")
    checkpoint = Path(packet.get("checkpoint", ""))
    if not checkpoint.is_file() or sha(checkpoint) != CHECKPOINT_HASHES[method]:
        raise ValueError(f"locked checkpoint identity invalid: {path}")
    if packet.get("checkpoint_sha256") not in (None, CHECKPOINT_HASHES[method]):
        raise ValueError(f"checkpoint hash mismatch: {path}")
    split = Path(packet.get("split_manifest", ""))
    if not split.is_file() or int(json.loads(split.read_text()).get("seed", -1)) != 2025:
        raise ValueError(f"locked split invalid: {path}")
    if packet.get("split_manifest_sha256") not in (None, sha(split)):
        raise ValueError(f"split hash mismatch: {path}")
    if set(packet.get("tasks", {})) != set(TASKS):
        raise ValueError(f"task coverage incomplete: {path}")
    for task in TASKS:
        row = packet["tasks"][task]
        if set(row.get("exits", {})) != {str(layer) for layer in EXITS}:
            raise ValueError(f"exit coverage incomplete: {path}:{task}")
        if int(row.get("selected_exit_best_validation", -1)) not in EXITS:
            raise ValueError(f"validation-selected exit missing: {path}:{task}")
        metric = row.get("spec", {}).get("metric_name")
        for layer in EXITS:
            entry = row["exits"][str(layer)]
            if metric not in entry.get("validation", {}) or metric not in entry.get("test", {}):
                raise ValueError(f"metrics incomplete: {path}:{task}:L{layer}")
            head = Path(entry.get("head_path", ""))
            if not head.is_file():
                raise FileNotFoundError(f"checkpointed head missing: {path}:{task}:L{layer}")
            if algorithm != "current":
                if entry.get("checkpoint_sha256") != sha(head):
                    raise ValueError(f"head identity mismatch: {path}:{task}:L{layer}")
                expected_sources = {str(source) for source in EXITS if source < layer}
                if set(entry.get("fusion_diagnostics", {})) != expected_sources:
                    raise ValueError(f"fusion diagnostics incomplete: {path}:{task}:L{layer}")
            if algorithm in VARIANTS:
                counts = entry.get("trainable_parameters", {})
                if set(counts) != {"total", "adapter", "gate", "output"} or int(counts["total"]) != sum(int(counts[key]) for key in ("adapter", "gate", "output")):
                    raise ValueError(f"parameter counts incomplete: {path}:{task}:L{layer}")


def load_packet(method: str, algorithm: str, **roots) -> tuple[Path, dict]:
    path = packet_path(method, algorithm, **roots)
    if not path.is_file():
        raise FileNotFoundError(f"required {algorithm} packet missing: {path}")
    packet = json.loads(path.read_text())
    _validate_packet(method, algorithm, packet, path)
    return path, packet


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing empty output: {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def gain(value: float, baseline: float, higher: bool) -> float:
    return value - baseline if higher else baseline - value


def reference_parameter_counts(algorithm: str, num_outputs: int, exit_layer: int) -> dict[str, int]:
    """Exact trainable-head counts for imported non-ablation references."""
    output = 2 * 256 + 256 * num_outputs + num_outputs
    if algorithm == "current":
        return {"total": output, "adapter": 0, "gate": 0, "output": output}
    sources = sum(source < exit_layer for source in EXITS)
    adapter = 33600 * sources  # LN + 256->64 + 64->256 for each source.
    gate = 0 if not sources else 16960 + 4225 * sources  # anchor gate + source gate/score.
    return {"total": adapter + gate + output, "adapter": adapter, "gate": gate, "output": output}


def plot_figures(fixed: list[dict], output: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    indexed = {(r["method"], r["algorithm"], r["task"], r["exit"]): r for r in fixed}
    colors = {"current": "#666666", "bottleneck_gelu": "#0072B2", "bottleneck_linear": "#E69F00",
              "bottleneck_sigmoid": "#CC79A7", "single_linear": "#D55E00", "single_gelu": "#009E73", "single_sigmoid": "#56B4E9"}
    figures = []
    for method in METHODS:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9)); fig.suptitle(f"{method.upper()} — Fusion V2 adapter ablation", y=0.99)
        for axis, task in zip(axes.flat, TASKS):
            for algorithm in ALGORITHMS:
                rows = [indexed[method, algorithm, task, layer] for layer in EXITS]
                axis.plot(EXITS, [row["test_value"] for row in rows], marker="o", linewidth=1.4, color=colors[algorithm], label=LABELS[algorithm])
            axis.set_title(task); axis.set_xticks(EXITS, [f"L{layer}" for layer in EXITS]); axis.grid(axis="y", alpha=.25)
            axis.set_ylabel("Mean distance error (lower is better)" if task == "nrpos-outdoor" else "Macro accuracy")
        handles, labels = axes.flat[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
        fig.text(.5, .015, "L2 has no fusion sources and is displayed as a reproducibility control, not an adapter effect.", ha="center")
        fig.tight_layout(rect=(.01, .045, .99, .87))
        for suffix, kwargs in (("png", {"dpi": 220}), ("pdf", {})):
            name = f"{method}_adapter_ablation_layerwise_performance.{suffix}"; fig.savefig(output / name, **kwargs); figures.append(name)
        plt.close(fig)
    return figures


def build_report(current_root=CURRENT_ROOT, gelu_root=GELU_ROOT, ablation_root=ABLATION_ROOT, output=OUT) -> dict:
    roots = {"current_root": Path(current_root), "gelu_root": Path(gelu_root), "ablation_root": Path(ablation_root)}
    sources = {(method, algorithm): load_packet(method, algorithm, **roots) for method in METHODS for algorithm in ALGORITHMS}
    split_hashes = {sha(Path(packet["split_manifest"])) for _, packet in sources.values()}
    if len(split_hashes) != 1:
        raise ValueError("comparison split identities differ")
    fixed, selected, gains, counts, gates, provenance, l2 = [], [], [], [], [], [], []
    for method in METHODS:
        for algorithm in ALGORITHMS:
            path, packet = sources[method, algorithm]
            provenance.append({"method": method, "algorithm": algorithm, "metrics_path": str(path), "metrics_sha256": sha(path),
                               "checkpoint_sha256": sha(Path(packet["checkpoint"])), "split_manifest_sha256": sha(Path(packet["split_manifest"]))})
            for task in TASKS:
                row, spec = packet["tasks"][task], packet["tasks"][task]["spec"]
                metric, higher = spec["metric_name"], bool(spec["higher_is_better"])
                chosen = int(row["selected_exit_best_validation"]); entry = row["exits"][str(chosen)]
                selected.append({"method": method, "algorithm": algorithm, "label": LABELS[algorithm], "task": task, "metric": metric,
                                 "selected_exit_best_validation": chosen, "validation_value": entry["validation"][metric], "selected_test_value": entry["test"][metric]})
                for layer in EXITS:
                    entry = row["exits"][str(layer)]
                    fixed.append({"method": method, "algorithm": algorithm, "label": LABELS[algorithm], "task": task, "exit": layer,
                                  "metric": metric, "higher_is_better": higher, "validation_value": entry["validation"][metric],
                                  "test_value": entry["test"][metric], "selected_by_validation": layer == chosen})
                    if layer == 2:
                        l2.append({"method": method, "algorithm": algorithm, "task": task, "metric": metric, "test_value": entry["test"][metric]})
                    row_counts = entry["trainable_parameters"] if algorithm in VARIANTS else reference_parameter_counts(
                        algorithm, int(spec["num_outputs"]), layer)
                    counts.append({"method": method, "adapter_variant": algorithm, "label": LABELS[algorithm],
                                   "task": task, "exit": layer, **row_counts})
                    if algorithm != "current":
                        for source, diagnostic in entry["fusion_diagnostics"].items():
                            gates.append({"method": method, "algorithm": algorithm, "task": task, "target_exit": layer, "source_exit": int(source), **diagnostic})
        gelu = sources[method, "bottleneck_gelu"][1]
        for variant in VARIANTS:
            packet = sources[method, variant][1]
            for task in TASKS:
                spec = packet["tasks"][task]["spec"]; metric, higher = spec["metric_name"], bool(spec["higher_is_better"])
                comparisons = [(f"fixed_L{layer}", layer, layer) for layer in EXITS]
                comparisons.append(("validation_selected", int(packet["tasks"][task]["selected_exit_best_validation"]), int(gelu["tasks"][task]["selected_exit_best_validation"])))
                for comparison, layer, reference in comparisons:
                    value = float(packet["tasks"][task]["exits"][str(layer)]["test"][metric])
                    base = float(gelu["tasks"][task]["exits"][str(reference)]["test"][metric])
                    gains.append({"method": method, "adapter_variant": variant, "task": task, "comparison": comparison,
                                  "variant_exit": layer, "bottleneck_gelu_exit": reference, "metric": metric,
                                  "positive_is_improvement": True, "direction_aware_gain": gain(value, base, higher)})
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "fixed_exit_results.csv", fixed); write_csv(output / "validation_selected_results.csv", selected)
    write_csv(output / "direction_aware_gains_vs_bottleneck_gelu.csv", gains); write_csv(output / "parameter_counts.csv", counts)
    write_csv(output / "gate_diagnostics.csv", gates); write_csv(output / "l2_reproducibility_control.csv", l2)
    figures = plot_figures(fixed, output)
    manifest = {"study": "residual_fusion_adapter_ablation_single_seed", "methods": METHODS, "algorithms": ALGORITHMS,
                "head_seed": 0, "pretrain_seed": 0, "split_seed": 2025, "tasks": TASKS, "exits": EXITS,
                "primary_reference": "bottleneck_gelu", "test_used_for_selection": False, "figures": figures, "sources": provenance}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "README.md").write_text("# Fusion V2 adapter ablation — single-seed pilot\n\n"
        "P0/P2 head seed 0 comparison of Current EE, the immutable Fusion V2 Bottleneck GELU reference, and five new adapter variants. "
        "Validation selects exits; held-out test evaluation follows selection. Positive direction-aware gains favor the new variant over Bottleneck GELU. "
        "L2 has no earlier source representation: its results are a reproducibility control, not a fusion effect.\n")
    return {"output": str(output), "fixed_rows": len(fixed), "selected_rows": len(selected), "gain_rows": len(gains), "sources": len(provenance), "figures": len(figures)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-root", type=Path, default=CURRENT_ROOT); parser.add_argument("--gelu-root", type=Path, default=GELU_ROOT)
    parser.add_argument("--ablation-root", type=Path, default=ABLATION_ROOT); parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(build_report(args.current_root, args.gelu_root, args.ablation_root, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
