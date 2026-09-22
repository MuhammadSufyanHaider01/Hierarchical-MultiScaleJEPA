# Time-sample availability vs. early-exit depth — WiFi human activity recognition

## Run definition

| Field | Value |
|---|---|
| Downstream task | WiFi human activity recognition (`human-activity`, HAS) |
| Cache | `../has.h5`, dataset `csi` |
| Full input shape | `[N, 3, 224, 224]` |
| Target | Six-class human-activity label |
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
| 2 | 32 | 84 | **0.842** | 0.817 | 0.725 | 0.708 | L2 | L2 |
| 4 | 64 | 168 | **0.842** | 0.825 | 0.808 | 0.767 | L2 | L2 |
| 8 | 128 | 336 | 0.875 | 0.858 | 0.875 | **0.883** | L8 | L2 |
| 14 | 224 | 588 | **1.000** | 0.992 | 0.992 | **1.000** | L2/L8 | L2 |

## Selected-exit inference cost

FLOPs and synchronized batch-1 latency were measured only for the
validation-selected exit, with 50 warm-up and 200 measurement iterations.

| Tₚ | Selected exit | Selected validation accuracy | FLOPs | Median latency (ms) | P95 latency (ms) |
|---:|---:|---:|---:|---:|
| 2 | L2 | 0.842 | 0.290B | 4.765 | 5.035 |
| 4 | L2 | 0.842 | 0.608B | 4.748 | 5.007 |
| 8 | L2 | 0.875 | 1.332B | 4.780 | 5.051 |
| 14 | L2 | 1.000 | 2.635B | 2.547 | 2.801 |

HAS is comparatively easy under this protocol: L2 is selected at every
availability level, and the full-information validation accuracy reaches 1.0.
This ceiling effect is why interference is retained as the more informative
time-sample sensitivity task.

## Status and artifacts

This packet contains **validation-selection results only**. The test split has
not been accessed, so no test metric or final test-selected claim is reported.

- `input_information_exit_metrics.json` is the complete machine-readable
  selection artifact, including source checkpoint/head provenance and all
  fixed-exit metrics.
- `human-activity_temporal_exit_vs_time_patches.png` and `.pdf` plot selected
  validation accuracy against Tₚ; the annotation beside each point gives the
  selected exit layer.
- The CSV and Markdown files provide tabular copies of the recorded results.
