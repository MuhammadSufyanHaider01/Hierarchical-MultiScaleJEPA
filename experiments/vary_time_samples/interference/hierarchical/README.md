# Proposed hierarchical approach — Wireless interference classification

This is an immutable, validation-only input-availability selection packet. The foundation checkpoint, frozen exit heads, source HDF5 files, and split manifest were not modified; the test split was not evaluated.

## Run definition

- Availability axis: Available time patches (Tₚ)
- Levels: `2, 4, 8, 14`
- Full input shape: `1 × 224 × 224`
- Primary validation metric: macro accuracy
- Candidate exits: L2, L4, L6, L8; selection is earliest within the source 5% validation tolerance.
- Matmul precision: `highest`
- TF32 enabled: `False`

## Results files

- `input_information_temporal.md` / `.csv`: every fixed-exit validation metric, L8-relative gap, input shape, token count, selected exit, FLOPs, and latency.
- `input_information_selected_exits.md` / `.csv`: compact selected-exit summary.
- `interference_temporal_exit_vs_time_patches.png` / `.pdf`: this method's selected-metric curve; annotations identify the selected exit layer.
- `input_information_exit_metrics.json`: complete machine-readable metrics and immutable provenance.

## Provenance

- Source full-information metrics: `/home/waves-lab/LatentWave_Data/preprocessed_datasets/MultiScaleJEPA-main/runs/masking_study/time_n6/checkpoint_epoch029_six_tasks/p2/seed_000/downstream/exit_metrics.json`
- Frozen checkpoint: `/home/waves-lab/LatentWave_Data/preprocessed_datasets/MultiScaleJEPA-main/runs/masking_study/time_n6/proposed_hierarchical_performance/seed_000/checkpoints/epoch=epoch=029-step=step=6780.ckpt`
- Split manifest: `/home/waves-lab/LatentWave_Data/preprocessed_datasets/MultiScaleJEPA-main/runs/masking_study/time_n6/checkpoint_epoch029_six_tasks/splits_seed_2025_primary_v3.json`
