# Four-method Human activity sensing representation trajectories

Inference-only comparison of temporal prefix trajectories. Each method has its own PCA fit on validation data; PCA coordinates are not comparable across methods. Primary scalar metrics use the original 256-D feature space.

## Protocol

- Methods: P0, P1 (same-scale intermediate supervision), P2 (fine-to-coarse), and P2-Reverse (coarse-to-fine).
- Checkpoints: P0/P1/P2 epoch 59 / step 13,560; P2-Reverse epoch 49 / step 11,300. All pretraining seeds are 0 and use the canonical six-mask multi_block/span_time signature.
- P2-Reverse is from a 50-epoch run while the other methods use 60-epoch runs; interpret its comparison as exploratory.
- P2-Reverse source split provenance is unverifiable because its original manifest is unavailable. All extracted representations and newly trained heads use the canonical seed-2025 split.
- Test samples: 120 (class 0: 20, class 1: 20, class 2: 20, class 3: 20, class 4: 20, class 5: 20).
- Prefixes: 2, 4, 8, and 14 temporal columns (28, 56, 112, and 196 tokens), retaining the original 14×14 positional grid.
- Features: frozen context-encoder normalized tokens mean-pooled to 256 dimensions at L2/L4/L6/L8.
- Intervals: 2,000 class-stratified paired bootstrap resamples (seed 2025); uncertainty is conditional on pretraining seed 0.

## Predeclared Tp4 comparisons

Paired differences are method minus reference; intervals below are 95% bootstrap intervals for class-separation ratio.

| Layer | P1−P0 | P2−P0 | P2-Reverse−P0 | P2−P1 | P2−P2-Reverse |
|---|---:|---:|---:|---:|---:|
| L2 | 0.071 [0.022, 0.127] | 0.116 [0.020, 0.203] | 0.014 [-0.045, 0.076] | 0.045 [-0.018, 0.105] | 0.102 [0.043, 0.161] |
| L4 | 0.165 [0.107, 0.228] | 0.106 [0.032, 0.179] | 0.115 [0.033, 0.197] | -0.059 [-0.112, -0.011] | -0.009 [-0.059, 0.042] |
| L6 | 0.232 [0.162, 0.301] | 0.253 [0.181, 0.326] | 0.221 [0.138, 0.307] | 0.020 [-0.050, 0.088] | 0.032 [-0.033, 0.103] |
| L8 | 0.077 [-0.010, 0.157] | 0.058 [-0.024, 0.139] | 0.186 [0.090, 0.278] | -0.019 [-0.109, 0.070] | -0.128 [-0.204, -0.041] |

These are descriptive estimates; no automatic winner or causal attribution is implied. Intervals quantify resampling uncertainty for pretraining seed 0 and do not capture across-seed variation.

See `class_centroid_trajectories` for PCA-projected class centroids and `trajectory_metrics` for original-space metric trajectories.
