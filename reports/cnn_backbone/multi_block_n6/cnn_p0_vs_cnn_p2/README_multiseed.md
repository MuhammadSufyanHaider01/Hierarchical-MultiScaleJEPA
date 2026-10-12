# Seven-head-seed M-block comparison: CNN C0 and C2

This supplement adds seven independently initialized frozen-head runs (head seeds 0–6) for C0 and C2 to the existing M-block CNN report. The pretrained CNN checkpoint is held fixed at pretraining seed 0 for each method. All six downstream tasks use the persisted CNN split manifest (seed 2025); exits are selected by validation performance separately for each task and head seed, and each saved selection is evaluated once on the held-out test split.

B0, P0, and P2 are reused from `reports/masking_study/multi_block_n6`; their seven-head-seed outcomes are not retrained here. **Split comparability caveat:** The existing ViT multiseed report and the CNN head runs record different split-manifest SHA-256 values (c617ee71716d273e5bef8318ec0d50d7c0b6e7e44d2c1bd6c3514e6909b571ae vs 5f48b317da917f2f76f26f8e0bf12808e60b8c6d94bd048b3b59e902ed3d575a). Both use split seed 2025 and the same six task files, but exact sample-level split identity is not verified; treat cross-backbone comparisons as descriptive, not paired.

The CNN inference summary uses the already-recorded per-exit FLOPs and synchronized batch-1 latency, weighted over each head seed's validation-selected exit. Thus the selected-inference spread reflects exit choice across head seeds, not repeated timing runs.

## Seven-seed validation-selected held-out results

See `multiseed_validation_selected.csv` for mean, sample standard deviation, min/max, all per-seed test results, chosen exits, and exit-selection frequency.

## Fixed-exit results

See `multiseed_fixed_exits.csv` for mean ± sample standard deviation at L2/L4/L6/L8 (B0 is full-depth L8 only).

## Selected-exit inference

See `multiseed_selected_inference.csv` for per-task FLOPs and latency averaged over the seven validation-selected exits. Latency timing itself is reused from the existing benchmark packets, not remeasured for each head seed.

## Figures

- `multiseed_selected_performance.png`
- `multiseed_selected_performance.pdf`
- `multiseed_fixed_exits.png`
- `multiseed_fixed_exits.pdf`
- `multiseed_selected_inference.png`
- `multiseed_selected_inference.pdf`
