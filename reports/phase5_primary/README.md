# Phase 5 primary single-seed pilot packet

This directory contains the submission-oriented result tables and figures for
the completed single-seed pilot. It is an aggregation of recorded
artifacts, not a statistical claim. The report was generated from repository
revision `5e8d8cbfc32e469f94a87022791307f642d11ef8`.

## Scope and comparability

| Approach | Pretraining | Downstream representation | Role |
|---|---|---|---|
| Original LatentWave | Original LatentWave training | EMA target encoder, full-input L8 | Source-faithful full-depth reference |
| LatentWave with Post-hoc Frozen Early Exits | Reuses the Original LatentWave checkpoint | Frozen context encoder at L2/L4/L6/L8 | Early-exit baseline |
| Proposed Hierarchical Multi-Scale JEPA | Separately pretrained hierarchical JEPA | Frozen context encoder at L2/L4/L6/L8 | Proposed method |

The Original LatentWave reference uses the pinned source-style 80/20
scikit-learn probe. The two exit approaches use independent frozen heads, a
shared persisted 80/10/10 split, and validation-only exit selection. The
single-figure all-approach plots show this protocol distinction explicitly;
the Original LatentWave point is a full-depth source reference, not a claim
that its metric was obtained with the frozen-head split.

All values are single-seed pilot point estimates.

## All-approach headline downstream comparison

The deployment rows use the full-depth Original LatentWave reference and the
earliest exit within 5% of validation-best for the two exit approaches.

| Approach | Task | Exit | Selection | Test value |
| --- | --- | --- | --- | --- |
| Original LatentWave | RF sensing | L8 | full-depth source reference | 75.00% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | earliest within 5% of validation best | 68.26% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | earliest within 5% of validation best | 74.09% |
| Original LatentWave | NR positioning | L8 | full-depth source reference | 2.680 m |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | earliest within 5% of validation best | 3.204 m |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | earliest within 5% of validation best | 2.813 m |
| Original LatentWave | DeepMIMO beam | L8 | full-depth source reference | 50.18% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | earliest within 5% of validation best | 39.23% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | earliest within 5% of validation best | 44.69% |
| Original LatentWave | DeepMIMO LoS/NLoS | L8 | full-depth source reference | 92.92% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO LoS/NLoS | L2 | earliest within 5% of validation best | 92.73% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO LoS/NLoS | L2 | earliest within 5% of validation best | 92.70% |

## All-approach headline inference comparison

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| Original LatentWave | RF sensing | L8 | 2.807 | 1.1463 | 1.2914 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.1463 | 1.2914 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 0.9070 | 0.9555 |
| Original LatentWave | NR positioning | L8 | 15.003 | 1.5568 | 1.8755 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2160 | 1.4982 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.1879 | 1.3249 |
| Original LatentWave | DeepMIMO beam | L8 | 0.102 | 1.3618 | 1.5644 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.6291 | 0.8062 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.5839 | 0.6922 |
| Original LatentWave | DeepMIMO LoS/NLoS | L8 | 0.102 | 1.2372 | 1.3776 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO LoS/NLoS | L2 | 0.026 | 0.5931 | 0.6949 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO LoS/NLoS | L2 | 0.026 | 0.5752 | 0.5976 |

## Original LatentWave source-faithful reference

| Task | Metric | Test value | Examples |
| --- | --- | --- | --- |
| RF sensing | Balanced accuracy (source probe) | 75.00% | 3624 |
| NR positioning | Mean distance error (m) | 2.680 m | 11628 |
| DeepMIMO beam | Balanced accuracy (source probe) | 50.18% | 14840 |
| DeepMIMO LoS/NLoS | Balanced accuracy (source probe) | 92.92% | 14840 |

## Detailed frozen-exit downstream metrics

Test values are shown after each head's validation-selected checkpoint. The
validation column is included for auditability; it is not a test result.

| Method | Task | Exit | Metric | Test value | Validation value | Head epoch |
| --- | --- | --- | --- | --- | --- | --- |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | Macro accuracy | 61.65% | 64.81% | 133 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | Macro accuracy | 63.95% | 66.08% | 91 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | Macro accuracy | 61.95% | 66.63% | 56 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | Macro accuracy | 68.26% | 71.68% | 80 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | Mean distance error (m) | 3.606 m | 3.592 m | 53 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | Mean distance error (m) | 3.330 m | 3.347 m | 51 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | Mean distance error (m) | 3.204 m | 3.249 m | 27 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | Mean distance error (m) | 3.104 m | 3.102 m | 55 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | Macro accuracy | 39.23% | 42.78% | 40 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | Macro accuracy | 42.37% | 43.75% | 52 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | Macro accuracy | 44.52% | 44.14% | 75 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | Macro accuracy | 44.85% | 44.25% | 32 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO LoS/NLoS | L2 | Macro accuracy | 92.73% | 91.44% | 61 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO LoS/NLoS | L4 | Macro accuracy | 92.63% | 91.63% | 24 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO LoS/NLoS | L6 | Macro accuracy | 93.12% | 91.29% | 51 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO LoS/NLoS | L8 | Macro accuracy | 92.13% | 91.49% | 45 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | Macro accuracy | 72.25% | 68.00% | 80 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | Macro accuracy | 65.53% | 68.05% | 50 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | Macro accuracy | 74.09% | 76.11% | 98 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | Macro accuracy | 72.26% | 76.49% | 78 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | Mean distance error (m) | 3.312 m | 3.297 m | 42 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | Mean distance error (m) | 3.101 m | 3.073 m | 45 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | Mean distance error (m) | 2.813 m | 2.788 m | 38 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | Mean distance error (m) | 2.732 m | 2.761 m | 65 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | Macro accuracy | 44.69% | 45.96% | 61 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | Macro accuracy | 45.21% | 46.42% | 64 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | Macro accuracy | 43.96% | 45.06% | 18 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | Macro accuracy | 43.23% | 45.23% | 21 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO LoS/NLoS | L2 | Macro accuracy | 92.70% | 92.81% | 68 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO LoS/NLoS | L4 | Macro accuracy | 92.89% | 92.38% | 14 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO LoS/NLoS | L6 | Macro accuracy | 92.90% | 92.15% | 34 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO LoS/NLoS | L8 | Macro accuracy | 92.78% | 91.76% | 50 |

## Detailed selected exits

"Earliest within 5%" is selected from validation only. For positioning,
lower mean distance error is better; for classification, higher macro
accuracy is better.

| Method | Task | Policy | Exit | Test value |
| --- | --- | --- | --- | --- |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | best validation | L8 | 68.26% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | earliest within tolerance | L8 | 68.26% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | best validation | L8 | 3.104 m |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | earliest within tolerance | L6 | 3.204 m |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | best validation | L8 | 44.85% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | earliest within tolerance | L2 | 39.23% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO LoS/NLoS | best validation | L4 | 92.63% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO LoS/NLoS | earliest within tolerance | L2 | 92.73% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | best validation | L8 | 72.26% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | earliest within tolerance | L6 | 74.09% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | best validation | L8 | 2.732 m |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | earliest within tolerance | L6 | 2.813 m |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | best validation | L4 | 45.21% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | earliest within tolerance | L2 | 44.69% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO LoS/NLoS | best validation | L2 | 92.70% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO LoS/NLoS | earliest within tolerance | L2 | 92.70% |

## Detailed batch-1 inference cost

Latency is the synchronized CUDA mean/p95 measurement after warm-up on the
RTX 5070 Ti. Savings are relative to the same method/task at L8.

| Method | Task | Exit | FLOPs (B) | Mean ms | P95 ms | FLOPs saving | Latency saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.4639 | 0.5663 | 74.3% | 59.5% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.6879 | 0.8330 | 49.5% | 40.0% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | 2.111 | 0.9028 | 1.0622 | 24.8% | 21.2% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.1463 | 1.2914 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | 3.828 | 0.5532 | 0.7015 | 74.5% | 64.5% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | 7.553 | 0.8855 | 1.1308 | 49.7% | 43.1% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2160 | 1.4982 | 24.8% | 21.9% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5568 | 1.8755 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.6291 | 0.8062 | 74.2% | 53.8% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | 0.052 | 0.8745 | 1.0327 | 49.5% | 35.8% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | 0.077 | 1.0911 | 1.2416 | 24.7% | 19.9% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 0.102 | 1.3618 | 1.5644 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO LoS/NLoS | L2 | 0.026 | 0.5931 | 0.6949 | 74.2% | 52.1% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO LoS/NLoS | L4 | 0.052 | 0.8020 | 0.9087 | 49.5% | 35.2% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO LoS/NLoS | L6 | 0.077 | 1.0026 | 1.0962 | 24.7% | 19.0% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO LoS/NLoS | L8 | 0.102 | 1.2372 | 1.3776 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.4578 | 0.4751 | 74.3% | 60.4% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 0.7052 | 0.7232 | 49.5% | 38.9% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 0.9070 | 0.9555 | 24.8% | 21.5% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.1549 | 1.1967 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | 3.828 | 0.5304 | 0.5416 | 74.5% | 65.8% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | 7.553 | 0.8517 | 0.8649 | 49.7% | 45.1% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.1879 | 1.3249 | 24.8% | 23.4% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.5508 | 1.7468 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.5839 | 0.6922 | 74.2% | 52.6% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 0.8011 | 0.8718 | 49.5% | 35.0% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | 0.077 | 1.0150 | 1.0703 | 24.7% | 17.6% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | 0.102 | 1.2316 | 1.2973 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO LoS/NLoS | L2 | 0.026 | 0.5752 | 0.5976 | 74.2% | 53.6% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO LoS/NLoS | L4 | 0.052 | 0.7857 | 0.8373 | 49.5% | 36.7% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO LoS/NLoS | L6 | 0.077 | 1.0030 | 1.1679 | 24.7% | 19.2% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO LoS/NLoS | L8 | 0.102 | 1.2409 | 1.3166 | 0.0% | 0.0% |

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
- `b0_source_reference.png`
- `b0_source_reference.pdf`

## Exclusions

Diagnostic early-stopped pretraining runs, the same-scale auxiliary JEPA
ablation, OOD transfer results, extra seeds, and alternate masking regimes are
excluded.
They require separate approval and are not part of this primary pilot packet.
