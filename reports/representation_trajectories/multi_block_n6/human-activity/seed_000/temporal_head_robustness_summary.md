# Frozen-head robustness under human-activity

## Protocol

- Same full-information L2/L4/L6/L8 heads are evaluated under Tp2/Tp4/Tp8/Tp14 temporal prefixes.
- Methods: P0, P1, P2, and P2-Reverse; each encoder remains frozen during evaluation.
- P2-Reverse uses epoch 49 / step 11,300 from a 50-epoch run, so its comparison with 60-epoch methods is exploratory.
- P2-Reverse source split provenance is unverifiable; extraction, head training, and evaluation use the canonical seed-2025 split.
- Test class counts: class 0: 20, class 1: 20, class 2: 20, class 3: 20, class 4: 20, class 5: 20.
- Five head seeds are summarized with mean and standard deviation. Repeated inference passes are a determinism audit, not independent trials.
- Paired hierarchical bootstrap resamples head seeds and test examples within each class. Uncertainty is conditional on pretraining seed 0.

## Tp4 paired method comparisons

Differences are method minus reference; intervals are 95% hierarchical-bootstrap intervals for macro accuracy.

| Layer | P1−P0 | P2−P0 | P2-Reverse−P0 | P2−P1 | P2−P2-Reverse |
|---|---:|---:|---:|---:|---:|
| L2 | +0.092 [+0.030, +0.160] | +0.063 [-0.003, +0.135] | -0.003 [-0.068, +0.062] | -0.028 [-0.097, +0.043] | +0.067 [+0.002, +0.135] |
| L4 | +0.065 [+0.012, +0.122] | +0.075 [+0.002, +0.150] | +0.050 [-0.012, +0.115] | +0.010 [-0.057, +0.078] | +0.025 [-0.037, +0.093] |
| L6 | +0.110 [+0.040, +0.178] | +0.097 [+0.030, +0.160] | +0.085 [+0.027, +0.145] | -0.013 [-0.068, +0.043] | +0.012 [-0.037, +0.058] |
| L8 | -0.080 [-0.137, -0.028] | -0.042 [-0.090, +0.007] | +0.018 [-0.040, +0.075] | +0.038 [-0.003, +0.087] | -0.060 [-0.102, -0.020] |

Accuracy degradation is Tp14 macro accuracy minus current-prefix macro accuracy; Tp14 degradation is zero by construction.
Repeated inference passes are not counted as independent evidence.
