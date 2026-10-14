# Residual Dynamic Fusion V2 — single-seed pilot

This report compares Current EE and Residual Projected Dynamic Fusion V2 for P0/P2, head seed 0. All headline selected-exit rows use validation-only selection; the held-out test split is used only for evaluation. Direction-aware gains are positive when V2 improves the task metric. The single seed is a pilot, not an estimate of across-seed variability.

## Figures

Per-method/task PNG and PDF plots (`p0_*_exit_performance.*` and `p2_*_exit_performance.*`) mirror the masking-study packet. `algorithm_layerwise_performance.png`/`.pdf` provides the compact P2 comparison across all six tasks. All points are held-out test results for head seed 0; they do not show across-seed error bars.
