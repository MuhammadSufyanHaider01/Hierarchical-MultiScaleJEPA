# P0/P2 Interference representation trajectories

Inference-only comparison of fixed-grid temporal-prefix features. PCA is fit on validation features separately for each model; test metrics are computed in the original 256-D feature space.

## Protocol

- Checkpoints: P0 epoch 59 / P2 epoch 59 (step 13560).
- Samples: 1,400 held-out test examples, balanced at 700 per class.
- Prefix token counts: 28, 56, 112, and 196; original 14×14 positional grid retained.
- Features: normalized encoder tokens mean-pooled to 256 dimensions at L2/L4/L6/L8.
- Uncertainty: class-stratified paired bootstrap, 2,000 replicates; interval table includes P2−P0 CIs.

## Key test-set findings

- At L8, distance: P0 Tp2→Tp14 0.5013→0.0000; P2 0.7439→0.0000.
- At L8, cosine: P0 Tp2→Tp14 0.8677→1.0000; P2 0.7376→1.0000.
- At L8, separation ratio: P0 Tp2→Tp14 0.2722→0.1076; P2 0.1937→0.1107.

See `class_centroid_trajectories` for the projected class paths and `trajectory_metrics` for estimates with 95% bootstrap intervals.
