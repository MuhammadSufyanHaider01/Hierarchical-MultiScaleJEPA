# Frozen-head robustness under jamming

## Protocol

- Same full-information L2/L4/L6/L8 heads are evaluated under Tp2/Tp4/Tp8/Tp14 input-column prefixes.
- Methods: P0, P1, P2, and P2-Reverse; each encoder remains frozen during evaluation.
- P2-Reverse uses epoch 49 / step 11,300 from a 50-epoch run, so its comparison with 60-epoch methods is exploratory.
- P2-Reverse source split provenance is unverifiable; extraction, head training, and evaluation use the canonical seed-2025 split.
- Test class counts: class 0: 125, class 1: 125, class 2: 125, class 3: 125, class 4: 125, class 5: 125.
- Five head seeds are summarized with mean and standard deviation. Repeated inference passes are a determinism audit, not independent trials.
- Paired hierarchical bootstrap resamples head seeds and test examples within each class. Uncertainty is conditional on pretraining seed 0.

## P4 paired method comparisons

Differences are method minus reference; intervals are 95% hierarchical-bootstrap intervals for macro accuracy.

| Layer | P1−P0 | P2−P0 | P2-Reverse−P0 | P2−P1 | P2−P2-Reverse |
|---|---:|---:|---:|---:|---:|
| L2 | -0.001 [-0.030, +0.031] | -0.009 [-0.043, +0.025] | -0.030 [-0.061, +0.001] | -0.007 [-0.041, +0.024] | +0.021 [-0.010, +0.052] |
| L4 | +0.032 [-0.002, +0.067] | +0.016 [-0.015, +0.049] | +0.032 [+0.001, +0.064] | -0.016 [-0.048, +0.016] | -0.015 [-0.046, +0.017] |
| L6 | +0.073 [+0.041, +0.105] | +0.030 [-0.004, +0.064] | +0.043 [+0.011, +0.076] | -0.043 [-0.076, -0.010] | -0.014 [-0.045, +0.020] |
| L8 | -0.059 [-0.092, -0.025] | -0.112 [-0.143, -0.081] | +0.004 [-0.025, +0.034] | -0.053 [-0.087, -0.019] | -0.115 [-0.145, -0.085] |

Accuracy degradation is Tp14 macro accuracy minus current-prefix macro accuracy; Tp14 degradation is zero by construction.
Repeated inference passes are not counted as independent evidence.
