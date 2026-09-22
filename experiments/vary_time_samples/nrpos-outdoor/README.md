# Time-sample availability vs. early-exit depth — NR outdoor positioning

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

The availability factor is the fixed leading processed-cache time window
`x[..., :16·Tₚ]`, evaluated at `Tₚ ∈ {2,4,8,14}`. These correspond to input
widths of 32, 64, 128, and 224 columns, respectively. No resizing,
renormalization, reordering, source-HDF5 modification, or split generation is
performed. Exit selection uses validation only: select the earliest exit whose
mean distance error is within 5% of the best validation exit.

## Validation results

| Tₚ | Input width | Tokens | L2 error (m) | L4 error (m) | L6 error (m) | L8 error (m) | Best validation exit | 5%-rule selected exit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 32 | 112 | 10.663 | 9.037 | **5.398** | 6.804 | L6 | L6 |
| 4 | 64 | 224 | 7.565 | 6.424 | **5.431** | 6.053 | L6 | L6 |
| 8 | 128 | 448 | 5.404 | **3.919** | 4.044 | 4.616 | L4 | L4 |
| 14 | 224 | 784 | 3.745 | 3.443 | 3.447 | **3.421** | L8 | L4 |

## Selected-exit inference cost

FLOPs and synchronized batch-1 latency were measured only for the
validation-selected exit, with 50 warm-up and 200 measurement iterations.

| Tₚ | Selected exit | Selected validation error (m) | FLOPs | Median latency (ms) | P95 latency (ms) |
|---:|---:|---:|---:|---:|---:|
| 2 | L6 | 5.398 | 1.149B | 4.798 | 5.315 |
| 4 | L6 | 5.431 | 2.452B | 4.824 | 5.363 |
| 8 | L4 | 3.919 | 3.699B | 4.859 | 5.359 |
| 14 | L4 | 3.443 | 7.553B | 3.079 | 3.261 |

The selected exit changes from L6 for the shorter two and four-patch windows
to L4 at eight and fourteen patches. At full information (`Tₚ=14`), L8 has the
lowest error, but L4 is selected because its 3.443 m error is within 5% of
L8's 3.421 m validation error.

## Status and artifacts

This packet contains **validation-selection results only**. The test split has
not been accessed, so no test metric or final test-selected claim is reported.

- `input_information_exit_metrics.json` is the complete machine-readable
  selection artifact, including source checkpoint/head provenance and all
  fixed-exit metrics.
- `temporal_exit_vs_time_patches.png` and `.pdf` plot selected validation error
  against Tₚ; the annotation beside each point gives the selected exit layer.
- The CSV and Markdown files provide tabular copies of the recorded results.
