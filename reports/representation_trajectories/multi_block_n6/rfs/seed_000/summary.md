# Four-method RF signal classification representation trajectories

Inference-only comparison of input-column prefix trajectories. Each method has its own PCA fit on validation data; PCA coordinates are not comparable across methods. Primary scalar metrics use the original 256-D feature space.

## Protocol

- Methods: P0, P1 (same-scale intermediate supervision), P2 (fine-to-coarse), and P2-Reverse (coarse-to-fine).
- Checkpoints: P0/P1/P2 epoch 59 / step 13,560; P2-Reverse epoch 49 / step 11,300. All pretraining seeds are 0 and use the canonical six-mask multi_block/span_time signature.
- P2-Reverse is from a 50-epoch run while the other methods use 60-epoch runs; interpret its comparison as exploratory.
- P2-Reverse source split provenance is unverifiable because its original manifest is unavailable. All extracted representations and newly trained heads use the canonical seed-2025 split.
- Test samples: 362 (class 0: 3, class 1: 7, class 2: 11, class 3: 5, class 4: 21, class 5: 40, class 6: 7, class 7: 5, class 8: 72, class 9: 59, class 10: 4, class 11: 4, class 12: 30, class 13: 4, class 14: 3, class 15: 12, class 16: 28, class 17: 10, class 18: 8, class 19: 29).
- Prefixes: 2, 4, 8, and 14 input-column columns (28, 56, 112, and 196 tokens), retaining the original 14×14 positional grid.
- Features: frozen context-encoder normalized tokens mean-pooled to 256 dimensions at L2/L4/L6/L8.
- Intervals: 2,000 class-stratified paired bootstrap resamples (seed 2025); uncertainty is conditional on pretraining seed 0.

## Predeclared Tp4 comparisons

Paired differences are method minus reference; intervals below are 95% bootstrap intervals for class-separation ratio.

| Layer | P1−P0 | P2−P0 | P2-Reverse−P0 | P2−P1 | P2−P2-Reverse |
|---|---:|---:|---:|---:|---:|
| L2 | 0.057 [0.022, 0.084] | 0.059 [0.008, 0.099] | 0.039 [0.001, 0.070] | 0.002 [-0.034, 0.039] | 0.020 [-0.016, 0.057] |
| L4 | 0.036 [-0.007, 0.076] | 0.050 [0.002, 0.098] | 0.039 [-0.006, 0.076] | 0.014 [-0.029, 0.057] | 0.011 [-0.028, 0.055] |
| L6 | 0.046 [0.002, 0.089] | 0.032 [-0.016, 0.073] | 0.027 [-0.017, 0.068] | -0.013 [-0.055, 0.022] | 0.005 [-0.040, 0.046] |
| L8 | -0.005 [-0.045, 0.042] | -0.010 [-0.055, 0.036] | 0.004 [-0.039, 0.052] | -0.004 [-0.052, 0.037] | -0.014 [-0.065, 0.033] |

These are descriptive estimates; no automatic winner or causal attribution is implied. Intervals quantify resampling uncertainty for pretraining seed 0 and do not capture across-seed variation.

See `class_centroid_trajectories` for PCA-projected class centroids and `trajectory_metrics` for original-space metric trajectories.
