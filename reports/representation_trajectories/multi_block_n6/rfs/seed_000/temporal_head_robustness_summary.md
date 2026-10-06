# Frozen-head robustness under rfs

## Protocol

- Same full-information L2/L4/L6/L8 heads are evaluated under Tp2/Tp4/Tp8/Tp14 input-column prefixes.
- Methods: P0, P1, P2, and P2-Reverse; each encoder remains frozen during evaluation.
- P2-Reverse uses epoch 49 / step 11,300 from a 50-epoch run, so its comparison with 60-epoch methods is exploratory.
- P2-Reverse source split provenance is unverifiable; extraction, head training, and evaluation use the canonical seed-2025 split.
- Test class counts: class 0: 3, class 1: 7, class 2: 11, class 3: 5, class 4: 21, class 5: 40, class 6: 7, class 7: 5, class 8: 72, class 9: 59, class 10: 4, class 11: 4, class 12: 30, class 13: 4, class 14: 3, class 15: 12, class 16: 28, class 17: 10, class 18: 8, class 19: 29.
- Five head seeds are summarized with mean and standard deviation. Repeated inference passes are a determinism audit, not independent trials.
- Paired hierarchical bootstrap resamples head seeds and test examples within each class. Uncertainty is conditional on pretraining seed 0.

## P4 paired method comparisons

Differences are method minus reference; intervals are 95% hierarchical-bootstrap intervals for macro accuracy.

| Layer | P1−P0 | P2−P0 | P2-Reverse−P0 | P2−P1 | P2−P2-Reverse |
|---|---:|---:|---:|---:|---:|
| L2 | -0.083 [-0.133, -0.037] | +0.079 [+0.019, +0.145] | -0.015 [-0.064, +0.033] | +0.162 [+0.104, +0.228] | +0.094 [+0.031, +0.162] |
| L4 | -0.020 [-0.065, +0.024] | +0.106 [+0.050, +0.161] | +0.018 [-0.028, +0.066] | +0.126 [+0.070, +0.182] | +0.088 [+0.029, +0.149] |
| L6 | -0.037 [-0.083, +0.008] | +0.108 [+0.057, +0.164] | -0.013 [-0.064, +0.040] | +0.145 [+0.089, +0.198] | +0.122 [+0.071, +0.175] |
| L8 | -0.089 [-0.140, -0.042] | +0.036 [-0.011, +0.088] | -0.010 [-0.061, +0.040] | +0.125 [+0.069, +0.185] | +0.047 [-0.001, +0.100] |

Accuracy degradation is Tp14 macro accuracy minus current-prefix macro accuracy; Tp14 degradation is zero by construction.
Repeated inference passes are not counted as independent evidence.
