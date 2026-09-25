# Frequency masking (N_m=6) single-seed comparison

This directory contains the submission-oriented result tables and figures for
the completed single-seed pilot. It is an aggregation of recorded
artifacts, not a statistical claim. The report was generated from repository
revision `cea1d9a34463a984f6cde90f0243610c39b778f3`.

## Scope and comparability

| Approach | Pretraining | Downstream representation | Role |
|---|---|---|---|
| Original LatentWave | Original LatentWave training | Frozen context encoder, full-input L8 | Full-depth baseline |
| LatentWave with Post-hoc Frozen Early Exits | Reuses the Original LatentWave checkpoint | Frozen context encoder at L2/L4/L6/L8 | Early-exit baseline |
| Proposed Hierarchical Multi-Scale JEPA | Separately pretrained hierarchical JEPA | Frozen context encoder at L2/L4/L6/L8 | Proposed method |

All rows use independent frozen task heads, the same persisted 80/10/10
split, and validation-only head/exit selection. Original LatentWave is the
P0 context encoder evaluated at L8; the post-hoc early-exit baseline reuses
that exact P0 checkpoint at L2/L4/L6/L8. Therefore the three approaches are
directly comparable for this single-seed packet.

All values are single-seed pilot point estimates.

## Run-specific provenance

The proposed hierarchical method uses checkpoint epoch index 54 (55 completed pretraining epochs); pretraining was intentionally stopped after this checkpoint was saved, before the 60-epoch maximum. Results are a single-seed comparison. Exact train/validation/test indices match the Original LatentWave run; only environment-specific HDF5 path strings differ.


## Task-suite note

This packet covers **Frequency masking (N_m=6)** and the following downstream task suite:
RF sensing, Jamming classification, Human activity sensing, Interference classification, NR positioning, DeepMIMO beam. Jamming and interference
are labelled frozen-head tasks only; none of the six downstream caches is used
for pretraining.

## All-approach headline downstream comparison

The deployment rows use full-depth L8 Original LatentWave and the
earliest exit within 5% of validation-best for the two exit approaches.

| Approach | Task | Exit | Selection | Test value |
| --- | --- | --- | --- | --- |
| Original LatentWave | RF sensing | L8 | full-depth L8 | 68.32% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | earliest within 5% of validation best | 68.32% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | earliest within 5% of validation best | 71.72% |
| Original LatentWave | Jamming classification | L8 | full-depth L8 | 89.73% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | earliest within 5% of validation best | 87.07% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | earliest within 5% of validation best | 86.53% |
| Original LatentWave | Human activity sensing | L8 | full-depth L8 | 100.00% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | earliest within 5% of validation best | 95.83% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | earliest within 5% of validation best | 96.67% |
| Original LatentWave | Interference classification | L8 | full-depth L8 | 71.64% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | earliest within 5% of validation best | 71.43% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | earliest within 5% of validation best | 73.21% |
| Original LatentWave | NR positioning | L8 | full-depth L8 | 3.157 m |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | earliest within 5% of validation best | 3.291 m |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | earliest within 5% of validation best | 2.891 m |
| Original LatentWave | DeepMIMO beam | L8 | full-depth L8 | 42.84% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | earliest within 5% of validation best | 39.80% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | earliest within 5% of validation best | 41.17% |

## All-approach headline inference comparison

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| Original LatentWave | RF sensing | L8 | 2.807 | 1.1259 | 1.2477 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.1259 | 1.2477 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.5170 | 0.7086 |
| Original LatentWave | Jamming classification | L8 | 2.807 | 1.1180 | 1.2667 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.9049 | 1.0111 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | 1.416 | 0.7376 | 0.8430 |
| Original LatentWave | Human activity sensing | L8 | 10.308 | 1.1939 | 1.3596 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4585 | 0.6210 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.4931 | 0.6314 |
| Original LatentWave | Interference classification | L8 | 2.807 | 1.1184 | 1.2317 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4493 | 0.4954 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.5121 | 0.6582 |
| Original LatentWave | NR positioning | L8 | 15.003 | 1.5560 | 1.7544 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2147 | 1.3986 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.6759 | 2.7483 |
| Original LatentWave | DeepMIMO beam | L8 | 0.102 | 1.2222 | 1.2982 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.5903 | 0.7066 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.6414 | 0.7627 |

## Original LatentWave full-depth context-encoder reference

| Task | Metric | Test value |
| --- | --- | --- |
| RF sensing | Macro accuracy | 68.32% |
| Jamming classification | Macro accuracy | 89.73% |
| Human activity sensing | Macro accuracy | 100.00% |
| Interference classification | Macro accuracy | 71.64% |
| NR positioning | Mean distance error (m) | 3.157 m |
| DeepMIMO beam | Macro accuracy | 42.84% |

## Detailed frozen-exit downstream metrics

Test values are shown after each head's validation-selected checkpoint. The
validation column is included for auditability; it is not a test result.

| Method | Task | Exit | Metric | Test value | Validation value | Head epoch |
| --- | --- | --- | --- | --- | --- | --- |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | Macro accuracy | 56.14% | 62.07% | 79 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | Macro accuracy | 67.59% | 73.35% | 153 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | Macro accuracy | 64.84% | 67.47% | 58 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | Macro accuracy | 68.32% | 79.85% | 123 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | Macro accuracy | 79.60% | 77.73% | 62 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | Macro accuracy | 82.13% | 80.67% | 76 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | Macro accuracy | 87.07% | 86.80% | 71 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | Macro accuracy | 89.73% | 89.60% | 59 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | Macro accuracy | 95.83% | 99.17% | 26 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | Macro accuracy | 95.00% | 100.00% | 21 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | Macro accuracy | 96.67% | 100.00% | 14 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | Macro accuracy | 100.00% | 99.17% | 21 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | Macro accuracy | 71.43% | 72.29% | 26 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | Macro accuracy | 71.00% | 72.00% | 42 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | Macro accuracy | 72.14% | 72.64% | 50 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | Macro accuracy | 71.64% | 71.79% | 47 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | Mean distance error (m) | 3.709 m | 3.673 m | 73 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | Mean distance error (m) | 3.478 m | 3.401 m | 26 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | Mean distance error (m) | 3.291 m | 3.247 m | 75 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | Mean distance error (m) | 3.157 m | 3.110 m | 80 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | Macro accuracy | 39.80% | 42.75% | 61 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | Macro accuracy | 42.16% | 43.19% | 34 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | Macro accuracy | 44.18% | 44.38% | 68 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | Macro accuracy | 42.84% | 43.56% | 24 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | Macro accuracy | 71.72% | 72.01% | 131 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | Macro accuracy | 65.42% | 69.53% | 74 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | Macro accuracy | 66.25% | 72.68% | 56 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | Macro accuracy | 71.13% | 72.71% | 76 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | Macro accuracy | 81.87% | 82.67% | 59 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | Macro accuracy | 86.53% | 88.67% | 84 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | Macro accuracy | 85.20% | 86.80% | 52 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | Macro accuracy | 88.27% | 88.67% | 78 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | Macro accuracy | 96.67% | 99.17% | 26 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | Macro accuracy | 99.17% | 100.00% | 38 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | Macro accuracy | 97.50% | 100.00% | 8 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | Macro accuracy | 96.67% | 99.17% | 6 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | Macro accuracy | 73.21% | 72.64% | 18 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | Macro accuracy | 74.21% | 74.57% | 10 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | Macro accuracy | 73.57% | 73.07% | 15 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | Macro accuracy | 73.00% | 73.21% | 23 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | Mean distance error (m) | 3.619 m | 3.554 m | 35 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | Mean distance error (m) | 3.317 m | 3.312 m | 44 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | Mean distance error (m) | 3.046 m | 2.998 m | 31 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | Mean distance error (m) | 2.891 m | 2.853 m | 47 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | Macro accuracy | 41.17% | 44.10% | 42 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | Macro accuracy | 41.44% | 43.19% | 35 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | Macro accuracy | 41.94% | 43.38% | 39 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | Macro accuracy | 41.38% | 43.31% | 35 |

## Detailed selected exits

"Earliest within 5%" is selected from validation only. For positioning,
lower mean distance error is better; for classification, higher macro
accuracy is better.

| Method | Task | Policy | Exit | Test value |
| --- | --- | --- | --- | --- |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | best validation | L8 | 68.32% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | earliest within tolerance | L8 | 68.32% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | best validation | L8 | 89.73% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | earliest within tolerance | L6 | 87.07% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | best validation | L4 | 95.00% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | earliest within tolerance | L2 | 95.83% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | best validation | L6 | 72.14% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | earliest within tolerance | L2 | 71.43% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | best validation | L8 | 3.157 m |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | earliest within tolerance | L6 | 3.291 m |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | best validation | L6 | 44.18% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | earliest within tolerance | L2 | 39.80% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | best validation | L8 | 71.13% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | earliest within tolerance | L2 | 71.72% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | best validation | L8 | 88.27% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | earliest within tolerance | L4 | 86.53% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | best validation | L4 | 99.17% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | earliest within tolerance | L2 | 96.67% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | best validation | L4 | 74.21% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | earliest within tolerance | L2 | 73.21% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | best validation | L8 | 2.891 m |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | earliest within tolerance | L8 | 2.891 m |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | best validation | L2 | 41.17% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | earliest within tolerance | L2 | 41.17% |

## Detailed batch-1 inference cost

Latency is the synchronized CUDA mean/p95 measurement after warm-up on the
RTX 5070 Ti. Savings are relative to the same method/task at L8.

| Method | Task | Exit | FLOPs (B) | Mean ms | P95 ms | FLOPs saving | Latency saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.4533 | 0.4842 | 74.3% | 59.7% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.6903 | 0.7766 | 49.5% | 38.7% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | 2.111 | 0.8981 | 0.9808 | 24.8% | 20.2% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.1259 | 1.2477 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | 0.721 | 0.4559 | 0.4861 | 74.3% | 59.2% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | 1.416 | 0.6809 | 0.7705 | 49.5% | 39.1% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.9049 | 1.0111 | 24.8% | 19.1% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 2.807 | 1.1180 | 1.2667 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4585 | 0.6210 | 74.4% | 61.6% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | 5.193 | 0.6990 | 0.8777 | 49.6% | 41.5% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | 7.750 | 0.9481 | 1.1484 | 24.8% | 20.6% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | 10.308 | 1.1939 | 1.3596 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4493 | 0.4954 | 74.3% | 59.8% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | 1.416 | 0.6744 | 0.7613 | 49.5% | 39.7% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | 2.111 | 0.9005 | 1.0228 | 24.8% | 19.5% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | 2.807 | 1.1184 | 1.2317 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | 3.828 | 0.5577 | 0.7190 | 74.5% | 64.2% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | 7.553 | 0.8867 | 1.0755 | 49.7% | 43.0% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2147 | 1.3986 | 24.8% | 21.9% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5560 | 1.7544 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.5903 | 0.7066 | 74.2% | 51.7% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | 0.052 | 0.8039 | 0.9136 | 49.5% | 34.2% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | 0.077 | 1.0210 | 1.1488 | 24.7% | 16.5% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 0.102 | 1.2222 | 1.2982 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.5170 | 0.7086 | 74.3% | 56.9% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 0.7326 | 0.8937 | 49.5% | 38.9% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 0.9761 | 1.1960 | 24.8% | 18.6% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.1988 | 1.4799 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.4821 | 0.5706 | 74.3% | 59.5% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | 1.416 | 0.7376 | 0.8430 | 49.5% | 38.1% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | 2.111 | 0.9890 | 1.2134 | 24.8% | 17.0% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 2.807 | 1.1914 | 1.4280 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.4931 | 0.6314 | 74.4% | 62.5% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 5.193 | 0.7665 | 1.0774 | 49.6% | 41.7% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | 7.750 | 1.0181 | 1.3125 | 24.8% | 22.5% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | 10.308 | 1.3140 | 2.3789 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.5121 | 0.6582 | 74.3% | 57.9% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | 1.416 | 0.7602 | 0.9142 | 49.5% | 37.4% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | 2.111 | 0.9814 | 1.2573 | 24.8% | 19.2% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | 2.807 | 1.2151 | 1.7694 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | 3.828 | 0.6167 | 0.9022 | 74.5% | 63.2% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | 7.553 | 0.9776 | 1.2880 | 49.7% | 41.7% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.3338 | 2.4615 | 24.8% | 20.4% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.6759 | 2.7483 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.6414 | 0.7627 | 74.2% | 51.1% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 0.8844 | 1.0191 | 49.5% | 32.6% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | 0.077 | 1.0857 | 1.2107 | 24.7% | 17.2% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | 0.102 | 1.3115 | 1.5230 | 0.0% | 0.0% |

## Figures

The PNG files are 300-DPI raster exports and the PDF files are vector exports.

- `all_methods_downstream_performance.png`
- `all_methods_downstream_performance.pdf`
- `all_methods_selected_performance.png`
- `all_methods_selected_performance.pdf`
- `all_methods_inference_cost.png`
- `all_methods_inference_cost.pdf`
- `early_exit_performance.png`
- `early_exit_performance.pdf`
- `selected_exit_performance.png`
- `selected_exit_performance.pdf`
- `inference_efficiency.png`
- `inference_efficiency.pdf`
- `b0_context_l8_reference.png`
- `b0_context_l8_reference.pdf`

## Exclusions

Diagnostic early-stopped pretraining runs, the same-scale auxiliary JEPA
ablation, extra transfer datasets, extra seeds, and alternate masking regimes
are excluded.
They require separate approval and are not part of this primary pilot packet.
