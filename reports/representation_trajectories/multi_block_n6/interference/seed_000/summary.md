# Four-method Interference representation trajectories

Inference-only comparison of temporal-prefix trajectories. Each method has its own PCA fit on validation data; PCA coordinates are not comparable across methods. Primary scalar metrics are in the original 256-D feature space.

## Protocol

- Methods: P0, P1 (same-scale intermediate supervision), P2 (fine-to-coarse), and P2-Reverse (coarse-to-fine).
- Selected checkpoints: P0 epoch 59 / step 13560 (run configured for 240 epochs); P1 epoch 59 / step 13560 (60 epochs); P2 epoch 59 / step 13560 (60 epochs); P2-Reverse epoch 49 / step 11300 (50 epochs). All are seed 0 and use the canonical six-mask multi_block/span_time signature.
- P0/P1/P2 are matched by selected update step, but their configured training schedules differ; P2-Reverse is also 10 epochs shorter than P1/P2. Method differences are descriptive and cannot isolate hierarchy/objective effects from schedule and horizon differences.
- The P2-Reverse source packet's split manifest is unavailable. All extracted features use the canonical seed-2025 split, but equality with the missing source packet manifest cannot be verified.
- Samples: 1,400 held-out test examples, balanced at 700 per class.
- Prefixes: 2, 4, 8, and 14 temporal columns (28, 56, 112, and 196 tokens), retaining the 14×14 positional grid.
- Features: frozen context-encoder normalized tokens mean-pooled to 256 dimensions at L2/L4/L6/L8.
- Intervals: 2,000 class-stratified paired bootstrap resamples (seed 2025); uncertainty is conditional on seed 0.

## Predeclared Tp4 comparisons

Paired differences are method minus reference; the intervals below are 95% bootstrap intervals for class-separation ratio.

| Layer | P1−P0 | P2−P0 | P2-Reverse−P0 | P2−P1 | P2−P2-Reverse |
|---|---:|---:|---:|---:|---:|
| L2 | 0.062 [0.040, 0.077] | 0.037 [0.016, 0.052] | 0.049 [0.026, 0.064] | -0.025 [-0.035, -0.014] | -0.012 [-0.022, -0.003] |
| L4 | 0.033 [0.016, 0.047] | 0.019 [0.002, 0.033] | 0.042 [0.024, 0.055] | -0.014 [-0.028, 0.001] | -0.023 [-0.036, -0.008] |
| L6 | 0.040 [0.020, 0.055] | 0.026 [0.006, 0.041] | 0.052 [0.033, 0.066] | -0.013 [-0.030, 0.005] | -0.026 [-0.041, -0.010] |
| L8 | 0.018 [-0.001, 0.034] | 0.005 [-0.017, 0.023] | 0.022 [0.003, 0.037] | -0.013 [-0.030, 0.003] | -0.017 [-0.032, -0.001] |

These are descriptive estimates; no automatic winner or causal attribution is implied. The intervals quantify resampling uncertainty for this seed and do not capture across-seed variation.

See `class_centroid_trajectories` for PCA-projected class centroids and `trajectory_metrics` for original-space metric trajectories.
