"""Render masking report figures in the established single-run visual style.

The plotted values come from the report aggregates, but figures intentionally
remain simple point/line/bar summaries: seed-level variability belongs in the
tables and provenance manifests, not in these established report panels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from masking_head_study import EXITS, METHODS, TASKS

REPORT_METHODS = ("b0", "p0", "p1", "p2", "p2_reverse")
LABELS = {
    "b0": "B0 (P0 L8)",
    "p0": "P0",
    "p1": "P1",
    "p2": "P2",
    "p2_reverse": "P2-Reverse",
}

COLORS = {
    "b0": "#009E73",
    "p0": "#0072B2",
    "p1": "#CC79A7",
    "p2": "#D55E00",
    "p2_reverse": "#6A3D9A",
}
SHORT_LABELS = {"b0": "B0", "p0": "P0", "p1": "P1", "p2": "P2", "p2_reverse": "P2-Rev"}
TASK_LABELS = {
    "rfs": "RF sensing",
    "jamming": "Jamming classification",
    "human-activity": "Human activity sensing",
    "interference": "Interference classification",
    "nrpos-outdoor": "NR positioning",
    "deepmimo-beam": "DeepMIMO beam",
}


def write_figures(output: Path, mask: str, aggregates: dict[str, Any], benchmarks: dict[str, Any]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    produced = []

    def save(fig, stem: str) -> None:
        fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        produced.extend([f"{stem}.png", f"{stem}.pdf"])

    def performance(stem: str, title: str, *, selected: bool, methods: tuple[str, ...]) -> None:
        from matplotlib.ticker import PercentFormatter

        fig, axes = plt.subplots(2, 3, figsize=(4.8 * 3, 3.7 * 2), squeeze=False)
        for task, ax in zip(TASKS, axes.flat):
            task_meta = aggregates["p0"]["tasks"][task]
            for method in methods:
                row = aggregates[method]["tasks"][task]
                if selected:
                    stat = row["selected"]
                    value = stat["mean"]
                    ax.bar(
                        [method], [value], color=COLORS[method],
                        edgecolor="#222222", linewidth=0.7,
                        label=LABELS[method],
                    )
                elif method == "b0":
                    stat = row["fixed_l8"]
                    ax.plot(
                        [8], [stat["mean"]], marker="D", markersize=6,
                        linewidth=2.2, color=COLORS[method], label=LABELS[method],
                    )
                else:
                    ax.plot(
                        EXITS,
                        [row["fixed"][layer]["mean"] for layer in EXITS],
                        marker="o", linewidth=2.2, markersize=5.5,
                        color=COLORS[method], label=LABELS[method],
                    )
            ax.set_title(TASK_LABELS[task], fontsize=11, weight="bold")
            ax.set_ylabel(
                "Macro accuracy" if task_meta["higher_is_better"]
                else "Mean distance error (m)"
            )
            ax.grid(axis="y", alpha=0.25, linewidth=0.8)
            if task_meta["higher_is_better"]:
                ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            else:
                ax.text(
                    0.98, 0.05, "lower is better", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="#555555",
                )
            if not selected:
                ax.set_xlabel("Encoder exit layer")
                ax.set_xticks(EXITS)
            else:
                ax.set_xticks(range(len(methods)), [SHORT_LABELS[m] for m in methods])
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", ncol=min(len(methods), 5),
                   frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=9)
        fig.suptitle(title, fontsize=14, weight="bold", y=1.045)
        fig.tight_layout(rect=(0, 0.02, 1, 0.94))
        save(fig, stem)

    performance(
        "all_methods_downstream_performance",
        "Frozen-head downstream performance by encoder depth",
        selected=False,
        methods=("p0", "p1", "p2", "p2_reverse"),
    )
    performance(
        "all_methods_selected_performance",
        "Validation-best exit performance across the approaches",
        selected=True,
        methods=REPORT_METHODS,
    )
    performance(
        "selected_exit_performance",
        "Validation-best exit performance across the approaches",
        selected=True,
        methods=REPORT_METHODS,
    )
    performance(
        "early_exit_performance",
        "P0 frozen-head performance by encoder depth",
        selected=False,
        methods=("p0",),
    )
    performance(
        "b0_context_l8_reference",
        "Original LatentWave full-depth reference (P0 L8)",
        selected=True,
        methods=("b0",),
    )

    for stem, key, metric, ylabel in (
        ("all_methods_inference_cost", "latency", "mean_ms", "Latency (ms, batch 1)"),
        ("inference_efficiency", "flops", None, "FLOPs (billions)"),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(4.8 * 3, 3.7 * 2), squeeze=False)
        for task, ax in zip(TASKS, axes.flat):
            for method in REPORT_METHODS:
                source_method = "p0" if method == "b0" else method
                records = [
                    r for r in benchmarks[source_method]["entries"]
                    if r["task"] == task and (method != "b0" or int(r["exit_layer"]) == 8)
                ]
                records.sort(key=lambda r: int(r["exit_layer"]))
                y = [r[key][metric] if metric else r[key] for r in records]
                if key == "flops":
                    y = [value / 1e9 for value in y]
                if method == "b0":
                    ax.plot([8], y[-1:], marker="D", markersize=6, linewidth=2.2,
                            color=COLORS[method], label=LABELS[method])
                else:
                    ax.plot(EXITS, y, marker="o", linewidth=2.0, markersize=5,
                            color=COLORS[method], label=LABELS[method])
            ax.set_title(TASK_LABELS[task], fontsize=10, weight="bold")
            ax.set_xlabel("Exit layer")
            ax.set_xticks(EXITS)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", ncol=5,
                   frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=9)
        fig.suptitle(ylabel, fontsize=14, weight="bold", y=1.045)
        fig.tight_layout(rect=(0, 0.02, 1, 0.94))
        save(fig, stem)

    return produced
