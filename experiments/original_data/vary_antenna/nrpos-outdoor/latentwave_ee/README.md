# Antenna availability vs. early-exit depth — NR outdoor positioning

## Run definition

| Field | Value |
|---|---|
| Downstream task | 5G NR outdoor positioning (`nrpos-outdoor`) |
| Cache | `data/downstream/nrpos-outdoor.h5` |
| Full input shape | `[N, 4, 224, 224]` |
| Target | Normalized 3-D position |
| Primary metric | Mean physical distance error (m); lower is better |
| Foundation model | `time_n6` P0, seed 0, frozen context encoder |
| Checkpoint | `runs/masking_study/time_n6/original_latentwave/seed_000/checkpoints/last.ckpt` |
| Encoder configuration | `configs/p0.yml` |
| Candidate exits | L2, L4, L6, L8 |
| Head protocol | Existing frozen full-information heads; no head or backbone updates |
| Split | Persisted 80/10/10 manifest `runs/masking_study/time_n6/splits_seed_2025_primary_v3.json` |

The availability factor is the fixed ordered channel prefix `x[:C]`, evaluated
at `C ∈ {1,2,3,4}`. No channel is synthesized, duplicated, reordered, or
renormalized. Exit selection uses validation only: select the earliest exit
whose mean distance error is within 5% of the best validation exit.

## Validation results

| C | Tokens | L2 error (m) | L4 error (m) | L6 error (m) | L8 error (m) | Best validation exit | 5%-rule selected exit |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 196 | 9.763 | 9.464 | **8.348** | 8.830 | L6 | L6 |
| 2 | 392 | 7.969 | 7.441 | **6.626** | 7.240 | L6 | L6 |
| 3 | 588 | 4.533 | 4.300 | **4.167** | 4.265 | L6 | L4 |
| 4 | 784 | 3.745 | 3.443 | 3.447 | **3.421** | L8 | L4 |

## Selected-exit inference cost

FLOPs and synchronized batch-1 latency were measured only for the
validation-selected exit, with 50 warm-up and 200 measurement iterations.

| C | Selected exit | Selected validation error (m) | FLOPs | Median latency (ms) | P95 latency (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | L6 | 8.348 | 2.111B | 2.589 | 3.075 |
| 2 | L6 | 6.626 | 4.695B | 2.716 | 3.143 |
| 3 | L4 | 4.300 | 5.193B | 2.760 | 3.245 |
| 4 | L4 | 3.443 | 7.553B | 2.840 | 3.398 |

The validation-selected exit moves from L6 at one and two available channels
to L4 at three and four channels. The C=3 and C=4 choices are shallower
because L4 remains within the 5% validation tolerance of the respective best
exit.

## Status and artifacts

This packet contains **validation-selection results only**. The test split has
not been accessed, so no test metric or final test-selected claim is reported.

- `input_information_exit_metrics.json` is the complete machine-readable
  selection artifact, including source checkpoint/head provenance and all
  fixed-exit metrics.
- `antenna_exit_vs_channels.png` and `.pdf` plot selected validation error
  against C; the annotation beside each point gives the selected exit layer.
- The CSV and Markdown files provide tabular copies of the recorded results.
