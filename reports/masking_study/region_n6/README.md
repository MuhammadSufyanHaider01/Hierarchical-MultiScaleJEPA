# Region-masking single-seed comparison packet

This directory contains the submission-oriented result tables and figures for
the completed single-seed pilot. It is an aggregation of recorded
artifacts, not a statistical claim. The report was generated from repository
revision `c4bc99187c694263c7ef800a3ffda7c4775f0975`.

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

## Task-suite note

This packet covers **Region masking (N_m=6)** and the following downstream task suite:
RF sensing, Jamming classification, Human activity sensing, Interference classification, NR positioning, DeepMIMO beam. Jamming and interference
are labelled frozen-head tasks only; none of the six downstream caches is used
for pretraining.

## All-approach headline downstream comparison

The deployment rows use full-depth L8 Original LatentWave and the
earliest exit within 5% of validation-best for the two exit approaches.

| Approach | Task | Exit | Selection | Test value |
| --- | --- | --- | --- | --- |
| Original LatentWave | RF sensing | L8 | full-depth L8 | 78.97% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | earliest within 5% of validation best | 75.16% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | earliest within 5% of validation best | 75.40% |
| Original LatentWave | Jamming classification | L8 | full-depth L8 | 85.47% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | earliest within 5% of validation best | 87.60% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | earliest within 5% of validation best | 88.40% |
| Original LatentWave | Human activity sensing | L8 | full-depth L8 | 96.67% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | earliest within 5% of validation best | 97.50% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | earliest within 5% of validation best | 97.50% |
| Original LatentWave | Interference classification | L8 | full-depth L8 | 71.29% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | earliest within 5% of validation best | 73.43% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | earliest within 5% of validation best | 74.00% |
| Original LatentWave | NR positioning | L8 | full-depth L8 | 2.939 m |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | earliest within 5% of validation best | 3.012 m |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | earliest within 5% of validation best | 2.759 m |
| Original LatentWave | DeepMIMO beam | L8 | full-depth L8 | 39.09% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | earliest within 5% of validation best | 42.52% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | earliest within 5% of validation best | 44.46% |

## All-approach headline inference comparison

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| Original LatentWave | RF sensing | L8 | 2.807 | 1.2750 | 1.4085 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.7627 | 0.7963 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 0.9112 | 0.9645 |
| Original LatentWave | Jamming classification | L8 | 2.807 | 1.1578 | 1.2586 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.9318 | 1.0112 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.5394 | 1.2924 |
| Original LatentWave | Human activity sensing | L8 | 10.308 | 1.2174 | 1.3959 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4697 | 0.6069 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.4440 | 0.4584 |
| Original LatentWave | Interference classification | L8 | 2.807 | 1.1557 | 1.2556 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4762 | 0.5561 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.4551 | 0.4799 |
| Original LatentWave | NR positioning | L8 | 15.003 | 1.5557 | 1.7371 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2303 | 1.4191 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.5602 | 2.1005 |
| Original LatentWave | DeepMIMO beam | L8 | 0.102 | 1.2556 | 1.3944 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.6012 | 0.7444 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.5712 | 0.6024 |

## Original LatentWave full-depth context-encoder reference

| Task | Metric | Test value |
| --- | --- | --- |
| RF sensing | Macro accuracy | 78.97% |
| Jamming classification | Macro accuracy | 85.47% |
| Human activity sensing | Macro accuracy | 96.67% |
| Interference classification | Macro accuracy | 71.29% |
| NR positioning | Mean distance error (m) | 2.939 m |
| DeepMIMO beam | Macro accuracy | 39.09% |

## Detailed frozen-exit downstream metrics

Test values are shown after each head's validation-selected checkpoint. The
validation column is included for auditability; it is not a test result.

| Method | Task | Exit | Metric | Test value | Validation value | Head epoch |
| --- | --- | --- | --- | --- | --- | --- |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | Macro accuracy | 69.78% | 73.48% | 82 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | Macro accuracy | 75.16% | 75.42% | 52 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | Macro accuracy | 75.05% | 76.97% | 41 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | Macro accuracy | 78.97% | 78.75% | 64 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | Macro accuracy | 82.00% | 79.33% | 53 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | Macro accuracy | 84.00% | 80.80% | 39 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | Macro accuracy | 87.60% | 86.80% | 83 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | Macro accuracy | 85.47% | 85.33% | 72 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | Macro accuracy | 97.50% | 98.33% | 54 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | Macro accuracy | 98.33% | 99.17% | 32 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | Macro accuracy | 99.17% | 99.17% | 22 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | Macro accuracy | 96.67% | 100.00% | 26 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | Macro accuracy | 73.43% | 72.14% | 23 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | Macro accuracy | 71.57% | 74.21% | 33 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | Macro accuracy | 70.07% | 72.43% | 43 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | Macro accuracy | 71.29% | 72.14% | 64 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | Mean distance error (m) | 3.546 m | 3.529 m | 73 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | Mean distance error (m) | 3.264 m | 3.297 m | 62 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | Mean distance error (m) | 3.012 m | 3.043 m | 89 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | Mean distance error (m) | 2.939 m | 3.007 m | 55 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | Macro accuracy | 42.52% | 43.48% | 66 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | Macro accuracy | 39.38% | 42.24% | 33 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | Macro accuracy | 41.46% | 43.85% | 34 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | Macro accuracy | 39.09% | 41.59% | 54 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | Macro accuracy | 67.92% | 72.81% | 84 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | Macro accuracy | 75.34% | 78.65% | 141 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | Macro accuracy | 75.40% | 81.36% | 114 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | Macro accuracy | 76.30% | 82.80% | 87 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | Macro accuracy | 88.40% | 86.00% | 58 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | Macro accuracy | 86.27% | 87.60% | 62 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | Macro accuracy | 85.60% | 87.20% | 46 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | Macro accuracy | 86.67% | 87.07% | 56 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | Macro accuracy | 97.50% | 100.00% | 23 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | Macro accuracy | 99.17% | 99.17% | 10 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | Macro accuracy | 99.17% | 100.00% | 12 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | Macro accuracy | 99.17% | 100.00% | 19 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | Macro accuracy | 74.00% | 73.36% | 29 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | Macro accuracy | 72.57% | 72.14% | 32 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | Macro accuracy | 74.93% | 74.93% | 39 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | Macro accuracy | 73.57% | 74.93% | 37 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | Mean distance error (m) | 3.332 m | 3.287 m | 61 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | Mean distance error (m) | 3.003 m | 3.122 m | 71 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | Mean distance error (m) | 2.877 m | 3.037 m | 53 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | Mean distance error (m) | 2.759 m | 2.874 m | 67 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | Macro accuracy | 44.46% | 44.48% | 32 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | Macro accuracy | 43.20% | 43.98% | 45 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | Macro accuracy | 43.01% | 42.99% | 42 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | Macro accuracy | 42.52% | 41.90% | 43 |

## Detailed selected exits

"Earliest within 5%" is selected from validation only. For positioning,
lower mean distance error is better; for classification, higher macro
accuracy is better.

| Method | Task | Policy | Exit | Test value |
| --- | --- | --- | --- | --- |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | best validation | L8 | 78.97% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | earliest within tolerance | L4 | 75.16% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | best validation | L6 | 87.60% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | earliest within tolerance | L6 | 87.60% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | best validation | L8 | 96.67% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | earliest within tolerance | L2 | 97.50% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | best validation | L4 | 71.57% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | earliest within tolerance | L2 | 73.43% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | best validation | L8 | 2.939 m |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | earliest within tolerance | L6 | 3.012 m |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | best validation | L6 | 41.46% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | earliest within tolerance | L2 | 42.52% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | best validation | L8 | 76.30% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | earliest within tolerance | L6 | 75.40% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | best validation | L4 | 86.27% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | earliest within tolerance | L2 | 88.40% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | best validation | L2 | 97.50% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | earliest within tolerance | L2 | 97.50% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | best validation | L6 | 74.93% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | earliest within tolerance | L2 | 74.00% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | best validation | L8 | 2.759 m |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | earliest within tolerance | L8 | 2.759 m |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | best validation | L2 | 44.46% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | earliest within tolerance | L2 | 44.46% |

## Detailed batch-1 inference cost

Latency is the synchronized CUDA mean/p95 measurement after warm-up on the
RTX 5070 Ti. Savings are relative to the same method/task at L8.

| Method | Task | Exit | FLOPs (B) | Mean ms | P95 ms | FLOPs saving | Latency saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.5118 | 0.5423 | 74.3% | 59.9% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.7627 | 0.7963 | 49.5% | 40.2% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | 2.111 | 1.0237 | 1.1079 | 24.8% | 19.7% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.2750 | 1.4085 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | 0.721 | 0.4787 | 0.5254 | 74.3% | 58.7% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | 1.416 | 0.7003 | 0.7470 | 49.5% | 39.5% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.9318 | 1.0112 | 24.8% | 19.5% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 2.807 | 1.1578 | 1.2586 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4697 | 0.6069 | 74.4% | 61.4% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | 5.193 | 0.7255 | 0.9150 | 49.6% | 40.4% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | 7.750 | 0.9743 | 1.2232 | 24.8% | 20.0% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | 10.308 | 1.2174 | 1.3959 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4762 | 0.5561 | 74.3% | 58.8% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | 1.416 | 0.7035 | 0.8086 | 49.5% | 39.1% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | 2.111 | 0.9384 | 1.0155 | 24.8% | 18.8% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | 2.807 | 1.1557 | 1.2556 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | 3.828 | 0.5717 | 0.7766 | 74.5% | 63.3% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | 7.553 | 0.9008 | 1.0954 | 49.7% | 42.1% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2303 | 1.4191 | 24.8% | 20.9% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5557 | 1.7371 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.6012 | 0.7444 | 74.2% | 52.1% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | 0.052 | 0.8241 | 0.8939 | 49.5% | 34.4% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | 0.077 | 1.0366 | 1.1626 | 24.7% | 17.4% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 0.102 | 1.2556 | 1.3944 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.4542 | 0.4795 | 74.3% | 63.8% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 0.6617 | 0.7098 | 49.5% | 47.3% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 0.9112 | 0.9645 | 24.8% | 27.4% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.2549 | 1.9018 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.5394 | 1.2924 | 74.3% | 51.6% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | 1.416 | 0.7819 | 1.3971 | 49.5% | 29.8% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | 2.111 | 0.9049 | 0.9886 | 24.8% | 18.7% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 2.807 | 1.1135 | 1.1969 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.4440 | 0.4584 | 74.4% | 62.7% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 5.193 | 0.8331 | 1.6843 | 49.6% | 30.0% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | 7.750 | 1.1514 | 1.9251 | 24.8% | 3.3% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | 10.308 | 1.1904 | 1.1970 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.4551 | 0.4799 | 74.3% | 61.6% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | 1.416 | 0.7229 | 0.7344 | 49.5% | 39.0% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | 2.111 | 0.9504 | 0.9825 | 24.8% | 19.8% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | 2.807 | 1.1845 | 1.2172 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | 3.828 | 0.5491 | 0.5564 | 74.5% | 64.8% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | 7.553 | 0.8700 | 0.8815 | 49.7% | 44.2% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.1954 | 1.2354 | 24.8% | 23.4% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.5602 | 2.1005 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.5712 | 0.6024 | 74.2% | 53.4% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 0.8007 | 0.8962 | 49.5% | 34.7% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | 0.077 | 0.9932 | 1.0634 | 24.7% | 19.0% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | 0.102 | 1.2269 | 1.2999 | 0.0% | 0.0% |

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
