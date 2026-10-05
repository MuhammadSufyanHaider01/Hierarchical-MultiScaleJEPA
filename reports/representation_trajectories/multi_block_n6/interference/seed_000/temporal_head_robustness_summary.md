# Temporal-head robustness under Interference

## Protocol

- Frozen P0, P1, P2, and P2-Reverse encoders; binary Interference classification.
- Heads were trained on Tp14 only with the backbone frozen and validation-only early stopping. The same saved head was evaluated at Tp2/Tp4/Tp8/Tp14.
- Five independent head-seed runs (2025–2029) are summarized as mean ± sample standard deviation. Five inference passes are a reproducibility audit, not independent trials.
- Each temporal condition retains the leading 2/4/8/14 columns of the fixed 14×14 patch grid (28/56/112/196 tokens). All L2/L4/L6/L8 exits use normalized context-encoder tokens mean-pooled to 256 dimensions.
- Test split: 1,400 persisted samples, balanced at 700 per class. Macro accuracy is primary; degradation is Tp14 macro accuracy minus partial-prefix macro accuracy.
- Bootstrap: 2,000 paired hierarchical resamples (head seeds and examples resampled within class), seed 2025. Intervals do not measure backbone pretraining-seed variability.

## Tp4 test metrics (mean ± SD across head seeds)

| Layer | P0 accuracy / drop | P1 accuracy / drop | P2 accuracy / drop | P2-Reverse accuracy / drop |
|---|---:|---:|---:|---:|
| L2 | 0.432 ± 0.005 / 0.292 ± 0.010 | 0.416 ± 0.003 / 0.314 ± 0.004 | 0.410 ± 0.001 / 0.340 ± 0.010 | 0.397 ± 0.002 / 0.321 ± 0.008 |
| L4 | 0.568 ± 0.009 / 0.156 ± 0.005 | 0.424 ± 0.004 / 0.305 ± 0.008 | 0.422 ± 0.003 / 0.327 ± 0.006 | 0.420 ± 0.004 / 0.317 ± 0.008 |
| L6 | 0.481 ± 0.002 / 0.247 ± 0.005 | 0.391 ± 0.001 / 0.343 ± 0.004 | 0.454 ± 0.004 / 0.298 ± 0.003 | 0.500 ± 0.005 / 0.233 ± 0.015 |
| L8 | 0.524 ± 0.011 / 0.203 ± 0.011 | 0.466 ± 0.007 / 0.282 ± 0.007 | 0.453 ± 0.003 / 0.302 ± 0.004 | 0.536 ± 0.012 / 0.192 ± 0.009 |

## Paired method comparisons

The following table reports method-minus-reference differences in macro accuracy at Tp4 with 95% hierarchical paired-bootstrap intervals.

| Layer | P1−P0 | P2−P0 | P2-Reverse−P0 | P2−P1 | P2−P2-Reverse |
|---|---:|---:|---:|---:|---:|
| L2 | -0.017 [-0.037, +0.002] | -0.022 [-0.041, -0.003] | -0.035 [-0.050, -0.021] | -0.005 [-0.022, +0.012] | +0.013 [-0.003, +0.030] |
| L4 | -0.144 [-0.174, -0.115] | -0.146 [-0.174, -0.117] | -0.148 [-0.174, -0.120] | -0.002 [-0.023, +0.018] | +0.002 [-0.019, +0.022] |
| L6 | -0.090 [-0.121, -0.058] | -0.027 [-0.058, +0.005] | +0.019 [-0.013, +0.050] | +0.063 [+0.037, +0.089] | -0.046 [-0.071, -0.019] |
| L8 | -0.058 [-0.090, -0.025] | -0.072 [-0.103, -0.037] | +0.011 [-0.021, +0.044] | -0.014 [-0.041, +0.012] | -0.083 [-0.109, -0.055] |

Interpret comparisons descriptively and alongside degradation and per-head-seed spread; do not select a winner from point estimates alone. Head-seed/test-sample intervals are conditional on the single pretraining seed 0.

P2-Reverse uses epoch 49 / step 11,300 from a 50-epoch run, whereas P1/P2 use epoch 59 / step 13,560 from 60-epoch runs. P2-Reverse comparisons are exploratory and may be affected by this horizon mismatch.

Repeated inference prediction disagreements are required to be zero. Small probability differences are retained in the raw-run table as a numerical stability diagnostic.

See temporal_head_robustness.csv, temporal_head_degradation.png, and geometry_functional_alignment.csv for the full results.
