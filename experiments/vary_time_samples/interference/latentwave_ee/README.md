# Time-sample availability vs. early-exit depth — Wireless interference classification

## Run definition

| Field | Value |
|---|---|
| Downstream task | Wireless interference classification (`interference`) |
| Cache | `../interference.h5`, dataset `cls_spectrograms` |
| Full input shape | `[N, 1, 224, 224]` |
| Target | Two-class interference label |
| Primary metric | Macro accuracy; higher is better |
| Foundation model | `time_n6` P0, seed 0, frozen context encoder |
| Checkpoint | `runs/masking_study/time_n6/original_latentwave/seed_000/checkpoints/last.ckpt` |
| Encoder configuration | `configs/p0.yml` |
| Candidate exits | L2, L4, L6, L8 |
| Head protocol | Existing frozen full-information heads; no head or backbone updates |
| Split | Persisted 80/10/10 manifest `runs/masking_study/time_n6/splits_seed_2025_primary_v3.json` |

The availability factor is the fixed leading processed-cache time window
`x[..., :16·Tₚ]`, evaluated at `Tₚ ∈ {2,4,8,14}`. These correspond to input
widths of 32, 64, 128, and 224 columns, respectively. No resizing,
renormalization, reordering, source-HDF5 modification, or split generation is
performed. Exit selection uses validation only: select the earliest exit whose
macro accuracy is within 5% of the best validation exit.

## Validation results

| Tₚ | Input width | Tokens | L2 accuracy | L4 accuracy | L6 accuracy | L8 accuracy | Best validation exit | 5%-rule selected exit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 32 | 28 | 0.416 | **0.519** | 0.501 | 0.519 | L4 | L4 |
| 4 | 64 | 56 | 0.455 | 0.521 | 0.478 | **0.522** | L8 | L4 |
| 8 | 128 | 112 | 0.603 | 0.673 | **0.689** | 0.638 | L6 | L4 |
| 14 | 224 | 196 | **0.749** | 0.745 | 0.739 | 0.714 | L2 | L2 |

## Selected-exit inference cost

FLOPs and synchronized batch-1 latency were measured only for the
validation-selected exit, with 50 warm-up and 200 measurement iterations.

| Tₚ | Selected exit | Selected validation accuracy | FLOPs | Median latency (ms) | P95 latency (ms) |
|---:|---:|---:|---:|---:|---:|
| 2 | L4 | 0.519 | 0.183B | 4.830 | 5.148 |
| 4 | L4 | 0.521 | 0.373B | 4.846 | 5.159 |
| 8 | L4 | 0.673 | 0.771B | 3.068 | 5.102 |
| 14 | L2 | 0.749 | 0.721B | 2.492 | 2.688 |

This task responds strongly to temporal availability: the selected validation
macro accuracy rises from about 0.52 at `Tₚ=2–4` to 0.749 at full input.
The selected exit is L4 through eight patches, then becomes L2 at full input.
At `Tₚ=4` and `Tₚ=8`, L4 is selected because it is within the 5% validation
tolerance of the best L8 and L6 exits, respectively.

## Status and artifacts

This packet contains **validation-selection results only**. The test split has
not been accessed, so no test metric or final test-selected claim is reported.

- `input_information_exit_metrics.json` is the complete machine-readable
  selection artifact, including source checkpoint/head provenance and all
  fixed-exit metrics.
- `interference_temporal_exit_vs_time_patches.png` and `.pdf` plot selected
  validation accuracy against Tₚ; the annotation beside each point gives the
  selected exit layer.
- The CSV and Markdown files provide tabular copies of the recorded results.
