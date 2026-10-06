# Four-method Jamming classification representation trajectories

Inference-only comparison of input-column prefix trajectories. Each method has its own PCA fit on validation data; PCA coordinates are not comparable across methods. Primary scalar metrics use the original 256-D feature space.

## Protocol

- Methods: P0, P1 (same-scale intermediate supervision), P2 (fine-to-coarse), and P2-Reverse (coarse-to-fine).
- Checkpoints: P0/P1/P2 epoch 59 / step 13,560; P2-Reverse epoch 49 / step 11,300. All pretraining seeds are 0 and use the canonical six-mask multi_block/span_time signature.
- P2-Reverse is from a 50-epoch run while the other methods use 60-epoch runs; interpret its comparison as exploratory.
- P2-Reverse source split provenance is unverifiable because its original manifest is unavailable. All extracted representations and newly trained heads use the canonical seed-2025 split.
- Test samples: 750 (class 0: 125, class 1: 125, class 2: 125, class 3: 125, class 4: 125, class 5: 125).
- Prefixes: 2, 4, 8, and 14 input-column columns (28, 56, 112, and 196 tokens), retaining the original 14×14 positional grid.
- Features: frozen context-encoder normalized tokens mean-pooled to 256 dimensions at L2/L4/L6/L8.
- Intervals: 2,000 class-stratified paired bootstrap resamples (seed 2025); uncertainty is conditional on pretraining seed 0.

## Predeclared Tp4 comparisons

Paired differences are method minus reference; intervals below are 95% bootstrap intervals for class-separation ratio.

| Layer | P1−P0 | P2−P0 | P2-Reverse−P0 | P2−P1 | P2−P2-Reverse |
|---|---:|---:|---:|---:|---:|
| L2 | 0.028 [0.019, 0.036] | 0.038 [0.025, 0.050] | 0.039 [0.028, 0.049] | 0.010 [0.004, 0.017] | -0.001 [-0.008, 0.005] |
| L4 | 0.041 [0.027, 0.054] | 0.038 [0.020, 0.056] | 0.049 [0.032, 0.064] | -0.004 [-0.012, 0.009] | -0.012 [-0.020, -0.000] |
| L6 | 0.013 [-0.003, 0.031] | 0.027 [0.007, 0.048] | 0.034 [0.016, 0.052] | 0.014 [0.002, 0.026] | -0.007 [-0.018, 0.005] |
| L8 | -0.010 [-0.026, 0.008] | 0.001 [-0.017, 0.023] | 0.043 [0.024, 0.061] | 0.011 [-0.003, 0.027] | -0.042 [-0.053, -0.027] |

These are descriptive estimates; no automatic winner or causal attribution is implied. Intervals quantify resampling uncertainty for pretraining seed 0 and do not capture across-seed variation.

See `class_centroid_trajectories` for PCA-projected class centroids and `trajectory_metrics` for original-space metric trajectories.
