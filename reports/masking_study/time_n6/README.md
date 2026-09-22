# Time-masking (N_m=6) single-seed comparison packet

This directory contains the submission-oriented result tables and figures for
the completed single-seed pilot. It is an aggregation of recorded
artifacts, not a statistical claim. The report was generated from repository
revision `95451bbbaac68c61bfb84edd4e8112c4df587a45`.

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

The proposed method uses the recoverable epoch-29 checkpoint (step 6,780). Pretraining was interrupted by an NVIDIA driver reset after epoch 34 completed; epoch 34 was not checkpointed. These results are therefore a preliminary single-seed checkpoint comparison, not the planned 60-epoch final result.


## Task-suite note

This packet covers **time masking, N_m=6** and the following downstream task suite:
RF sensing, Jamming classification, Human activity sensing, Interference classification, NR positioning, DeepMIMO beam. Jamming and interference
are labelled frozen-head tasks only; none of the six downstream caches is used
for pretraining.

## All-approach headline downstream comparison

The deployment rows use full-depth L8 Original LatentWave and the
earliest exit within 5% of validation-best for the two exit approaches.

| Approach | Task | Exit | Selection | Test value |
| --- | --- | --- | --- | --- |
| Original LatentWave | RF sensing | L8 | full-depth L8 | 72.76% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | earliest within 5% of validation best | 68.27% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | earliest within 5% of validation best | 71.24% |
| Original LatentWave | Jamming classification | L8 | full-depth L8 | 86.80% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | earliest within 5% of validation best | 89.07% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | earliest within 5% of validation best | 89.87% |
| Original LatentWave | Human activity sensing | L8 | full-depth L8 | 99.17% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | earliest within 5% of validation best | 100.00% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | earliest within 5% of validation best | 99.17% |
| Original LatentWave | Interference classification | L8 | full-depth L8 | 70.14% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | earliest within 5% of validation best | 73.64% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | earliest within 5% of validation best | 74.50% |
| Original LatentWave | NR positioning | L8 | full-depth L8 | 3.318 m |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | earliest within 5% of validation best | 3.408 m |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | earliest within 5% of validation best | 3.167 m |
| Original LatentWave | DeepMIMO beam | L8 | full-depth L8 | 35.24% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | earliest within 5% of validation best | 41.71% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | earliest within 5% of validation best | 41.03% |

## All-approach headline inference comparison

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| Original LatentWave | RF sensing | L8 | 2.807 | 1.1221 | 1.2569 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.4529 | 0.6066 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.4631 | 0.5396 |
| Original LatentWave | Jamming classification | L8 | 2.807 | 1.1183 | 1.2613 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | 0.721 | 0.4636 | 0.5966 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.4611 | 0.5266 |
| Original LatentWave | Human activity sensing | L8 | 10.308 | 1.2337 | 1.4207 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4702 | 0.7052 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.4592 | 0.5599 |
| Original LatentWave | Interference classification | L8 | 2.807 | 1.1167 | 1.3175 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4514 | 0.5666 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.4580 | 0.5294 |
| Original LatentWave | NR positioning | L8 | 15.003 | 1.5814 | 1.8182 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | 7.553 | 0.9480 | 1.2454 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.2237 | 1.5212 |
| Original LatentWave | DeepMIMO beam | L8 | 0.102 | 1.1969 | 1.3786 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.5901 | 0.7504 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.6226 | 0.6861 |

## Original LatentWave full-depth context-encoder reference

| Task | Metric | Test value |
| --- | --- | --- |
| RF sensing | Macro accuracy | 72.76% |
| Jamming classification | Macro accuracy | 86.80% |
| Human activity sensing | Macro accuracy | 99.17% |
| Interference classification | Macro accuracy | 70.14% |
| NR positioning | Mean distance error (m) | 3.318 m |
| DeepMIMO beam | Macro accuracy | 35.24% |

## Detailed frozen-exit downstream metrics

Test values are shown after each head's validation-selected checkpoint. The
validation column is included for auditability; it is not a test result.

| Method | Task | Exit | Metric | Test value | Validation value | Head epoch |
| --- | --- | --- | --- | --- | --- | --- |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | Macro accuracy | 68.27% | 72.79% | 92 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | Macro accuracy | 72.98% | 70.02% | 67 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | Macro accuracy | 68.58% | 70.12% | 69 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | Macro accuracy | 72.76% | 67.36% | 130 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | Macro accuracy | 89.07% | 88.27% | 94 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | Macro accuracy | 88.93% | 89.07% | 45 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | Macro accuracy | 90.53% | 90.27% | 89 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | Macro accuracy | 86.80% | 88.00% | 41 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | Macro accuracy | 100.00% | 100.00% | 27 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | Macro accuracy | 100.00% | 99.17% | 12 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | Macro accuracy | 99.17% | 99.17% | 6 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | Macro accuracy | 99.17% | 100.00% | 18 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | Macro accuracy | 73.64% | 74.93% | 29 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | Macro accuracy | 72.21% | 74.50% | 60 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | Macro accuracy | 73.64% | 73.86% | 35 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | Macro accuracy | 70.14% | 71.36% | 37 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | Mean distance error (m) | 3.710 m | 3.745 m | 68 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | Mean distance error (m) | 3.408 m | 3.443 m | 57 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | Mean distance error (m) | 3.418 m | 3.447 m | 75 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | Mean distance error (m) | 3.318 m | 3.421 m | 57 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | Macro accuracy | 41.71% | 43.43% | 56 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | Macro accuracy | 39.41% | 43.33% | 43 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | Macro accuracy | 43.03% | 41.04% | 31 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | Macro accuracy | 35.24% | 39.43% | 14 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | Macro accuracy | 71.24% | 72.14% | 146 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | Macro accuracy | 69.60% | 71.59% | 73 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | Macro accuracy | 72.41% | 73.35% | 130 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | Macro accuracy | 68.31% | 71.29% | 79 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | Macro accuracy | 89.87% | 87.87% | 73 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | Macro accuracy | 90.27% | 90.13% | 104 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | Macro accuracy | 89.87% | 89.47% | 96 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | Macro accuracy | 88.13% | 88.13% | 58 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | Macro accuracy | 99.17% | 100.00% | 22 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | Macro accuracy | 100.00% | 100.00% | 13 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | Macro accuracy | 99.17% | 100.00% | 8 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | Macro accuracy | 98.33% | 100.00% | 10 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | Macro accuracy | 74.50% | 72.29% | 37 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | Macro accuracy | 73.43% | 72.57% | 20 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | Macro accuracy | 73.29% | 73.00% | 22 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | Macro accuracy | 72.71% | 71.07% | 39 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | Mean distance error (m) | 3.613 m | 3.626 m | 61 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | Mean distance error (m) | 3.262 m | 3.338 m | 49 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | Mean distance error (m) | 3.167 m | 3.144 m | 76 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | Mean distance error (m) | 3.104 m | 3.123 m | 63 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | Macro accuracy | 41.03% | 42.43% | 38 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | Macro accuracy | 39.16% | 42.56% | 61 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | Macro accuracy | 38.23% | 41.72% | 32 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | Macro accuracy | 39.67% | 41.07% | 51 |

## Detailed selected exits

"Earliest within 5%" is selected from validation only. For positioning,
lower mean distance error is better; for classification, higher macro
accuracy is better.

| Method | Task | Policy | Exit | Test value |
| --- | --- | --- | --- | --- |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | best validation | L2 | 68.27% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | earliest within tolerance | L2 | 68.27% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | best validation | L6 | 90.53% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | earliest within tolerance | L2 | 89.07% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | best validation | L2 | 100.00% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | earliest within tolerance | L2 | 100.00% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | best validation | L2 | 73.64% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | earliest within tolerance | L2 | 73.64% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | best validation | L8 | 3.318 m |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | earliest within tolerance | L4 | 3.408 m |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | best validation | L2 | 41.71% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | earliest within tolerance | L2 | 41.71% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | best validation | L6 | 72.41% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | earliest within tolerance | L2 | 71.24% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | best validation | L4 | 90.27% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | earliest within tolerance | L2 | 89.87% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | best validation | L2 | 99.17% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | earliest within tolerance | L2 | 99.17% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | best validation | L6 | 73.29% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | earliest within tolerance | L2 | 74.50% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | best validation | L8 | 3.104 m |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | earliest within tolerance | L6 | 3.167 m |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | best validation | L4 | 39.16% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | earliest within tolerance | L2 | 41.03% |

## Detailed batch-1 inference cost

Latency is the synchronized CUDA mean/p95 measurement after warm-up on the
RTX 5070 Ti. Savings are relative to the same method/task at L8.

| Method | Task | Exit | FLOPs (B) | Mean ms | P95 ms | FLOPs saving | Latency saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.4529 | 0.6066 | 74.3% | 59.6% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.6796 | 0.7927 | 49.5% | 39.4% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | 2.111 | 0.9041 | 1.0494 | 24.8% | 19.4% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.1221 | 1.2569 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | 0.721 | 0.4636 | 0.5966 | 74.3% | 58.5% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | 1.416 | 0.6783 | 0.7795 | 49.5% | 39.3% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.8836 | 1.0219 | 24.8% | 21.0% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 2.807 | 1.1183 | 1.2613 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4702 | 0.7052 | 74.4% | 61.9% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | 5.193 | 0.7166 | 0.9296 | 49.6% | 41.9% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | 7.750 | 0.9719 | 1.1741 | 24.8% | 21.2% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | 10.308 | 1.2337 | 1.4207 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4514 | 0.5666 | 74.3% | 59.6% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | 1.416 | 0.6712 | 0.7938 | 49.5% | 39.9% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | 2.111 | 0.8860 | 1.0409 | 24.8% | 20.7% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | 2.807 | 1.1167 | 1.3175 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | 3.828 | 0.5620 | 0.7873 | 74.5% | 64.5% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | 7.553 | 0.9480 | 1.2454 | 49.7% | 40.1% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2453 | 1.4504 | 24.8% | 21.2% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5814 | 1.8182 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.5901 | 0.7504 | 74.2% | 50.7% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | 0.052 | 0.7879 | 0.9513 | 49.5% | 34.2% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | 0.077 | 1.0136 | 1.2005 | 24.7% | 15.3% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 0.102 | 1.1969 | 1.3786 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.4631 | 0.5396 | 74.3% | 59.2% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 0.6946 | 0.7709 | 49.5% | 38.8% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 0.9192 | 1.0527 | 24.8% | 19.0% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.1344 | 1.2625 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.4611 | 0.5266 | 74.3% | 59.4% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | 1.416 | 0.6907 | 0.7629 | 49.5% | 39.3% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | 2.111 | 0.9175 | 1.0668 | 24.8% | 19.3% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 2.807 | 1.1371 | 1.2618 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.4592 | 0.5599 | 74.4% | 61.6% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 5.193 | 0.7036 | 0.8015 | 49.6% | 41.2% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | 7.750 | 0.9468 | 1.1245 | 24.8% | 20.8% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | 10.308 | 1.1959 | 1.3647 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.4580 | 0.5294 | 74.3% | 59.7% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | 1.416 | 0.6919 | 0.7929 | 49.5% | 39.1% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | 2.111 | 0.9212 | 1.0489 | 24.8% | 18.9% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | 2.807 | 1.1355 | 1.2613 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | 3.828 | 0.5529 | 0.6479 | 74.5% | 64.6% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | 7.553 | 0.8921 | 1.0742 | 49.7% | 42.9% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.2237 | 1.5212 | 24.8% | 21.7% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.5631 | 1.8933 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.6226 | 0.6861 | 74.2% | 50.5% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 0.8755 | 1.0413 | 49.5% | 30.3% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | 0.077 | 1.0986 | 1.2802 | 24.7% | 12.6% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | 0.102 | 1.2570 | 1.4723 | 0.0% | 0.0% |

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
