# Antenna availability vs. early-exit depth — WiFi human activity recognition

## Run definition

| Field | Value |
|---|---|
| Downstream task | WiFi human activity recognition (`human-activity`) |
| Cache | `../has.h5` |
| Full input shape | `[N, 3, 224, 224]` |
| Target | Six activity classes |
| Primary metric | Macro accuracy; higher is better |
| Foundation model | `time_n6` P0, seed 0, frozen context encoder |
| Checkpoint | `runs/masking_study/time_n6/original_latentwave/seed_000/checkpoints/last.ckpt` |
| Encoder configuration | `configs/p0.yml` |
| Candidate exits | L2, L4, L6, L8 |
| Head protocol | Existing frozen full-information heads; no head or backbone updates |
| Split | Persisted 80/10/10 manifest `runs/masking_study/time_n6/splits_seed_2025_primary_v3.json` |

The availability factor is the fixed ordered channel prefix `x[:C]`, evaluated
at `C ∈ {1,2,3}`. No channel is synthesized, duplicated, reordered, or
renormalized. Exit selection uses validation only: select the earliest exit
whose macro accuracy is within 5% of the best validation exit.

## Validation results

| C | Tokens | L2 macro accuracy | L4 macro accuracy | L6 macro accuracy | L8 macro accuracy | Best validation exit | 5%-rule selected exit |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 196 | 0.700 | **0.867** | 0.758 | 0.400 | L4 | L4 |
| 2 | 392 | 0.858 | 0.858 | **0.867** | 0.675 | L6 | L2 |
| 3 | 588 | **1.000** | 0.992 | 0.992 | **1.000** | L2 | L2 |

## Selected-exit inference cost

FLOPs and synchronized batch-1 latency were measured only for the
validation-selected exit, with 50 warm-up and 200 measurement iterations.

| C | Selected exit | Selected validation macro accuracy | FLOPs | Median latency (ms) | P95 latency (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | L4 | 0.867 | 1.416B | 2.506 | 2.799 |
| 2 | L2 | 0.858 | 1.599B | 2.434 | 2.723 |
| 3 | L2 | 1.000 | 2.635B | 0.462 | 2.753 |

The validation-selected exit becomes shallower as more CSI channels are
available: L4 at C=1, then L2 at C=2 and C=3. At C=2, L6 attains the best
macro accuracy, but the earliest eligible L2 is within the 5% tolerance.

## Status and artifacts

This packet contains **validation-selection results only**. The test split has
not been accessed, so no test metric or final test-selected claim is reported.

- `input_information_exit_metrics.json` is the complete machine-readable
  selection artifact, including source checkpoint/head provenance and all
  fixed-exit metrics.
- `human-activity_antenna_exit_vs_channels.png` and `.pdf` plot selected
  validation macro accuracy against C; the annotation beside each point gives
  the selected exit layer.
- The CSV and Markdown files provide tabular copies of the recorded results.
