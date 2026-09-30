# Multi-block masking single-seed comparison packet

This directory contains the submission-oriented result tables and figures for
the completed single-seed pilot. It is an aggregation of recorded
artifacts, not a statistical claim. The report was generated from repository
revision `42796dd56e19ee470c42d575500592c5bd4af417`.

## Scope and comparability

| Approach | Pretraining | Downstream representation | Role |
| --- | --- | --- | --- |
| Original LatentWave | Original LatentWave training | Frozen context encoder, full-input L8 | Full-depth baseline |
| LatentWave with Post-hoc Frozen Early Exits | Reuses the Original LatentWave checkpoint | Frozen context encoder at L2/L4/L6/L8 | Early-exit baseline |
| Same-scale Multi-depth JEPA | P1 pretraining with same-scale intermediate targets | Frozen context encoder at L2/L4/L6/L8 | Same-scale auxiliary-loss control |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Separately pretrained fine-to-coarse hierarchical JEPA | Frozen context encoder at L2/L4/L6/L8 | Proposed method |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Imported reverse-hierarchy pretraining artifact | Frozen context encoder at L2/L4/L6/L8 | Exploratory hierarchy-direction ablation |

All rows use independent frozen task heads, the same persisted 80/10/10
split, and validation-only head/exit selection. Original LatentWave is the
P0 context encoder evaluated at L8; the post-hoc early-exit baseline reuses
that exact P0 checkpoint at L2/L4/L6/L8. B0, P0, P1, and P2 are directly
comparable for this single-seed packet. When included, P2-Reverse is an
exploratory imported ablation: its task suite and protocol match, but its
persisted split-index manifest was not packaged, so exact split identity could
not be re-verified locally.

All values are single-seed pilot point estimates.

## Run-specific provenance

P2-Reverse was imported from the verified artifact archive (claimed source commit 90c4793). Its six-task held-out metrics and synchronized batch-1 benchmark are available; the M-block benchmark packet was separately attached and validated (SHA-256: 078d2715eed558cc2b6e7262e292a88517c01b07f261f6e2709dff40f61dbee7). Its persisted split-index file was not included, so exact split identity cannot be re-verified locally.


## Task-suite note

## P1 checkpoint provenance

Same-scale Multi-depth JEPA checkpoint: `/home/waves-lab/LatentWave_Data/preprocessed_datasets/MultiScaleJEPA-main/runs/masking_study/frequency_n6/p1_same_scale_frequency_p2matched/seed_000/checkpoints/last.ckpt`.

Pretraining experiment metadata: `/home/waves-lab/LatentWave_Data/preprocessed_datasets/MultiScaleJEPA-main/runs/masking_study/frequency_n6/p1_same_scale_frequency_p2matched/seed_000/experiment.json`.

This packet covers **Multi-block span-time masking (6 prediction masks)** and the following downstream task suite:
RF sensing, Jamming classification, Human activity sensing, Interference classification, NR positioning, DeepMIMO beam. Jamming and interference
are labelled frozen-head tasks only; none of the six downstream caches is used
for pretraining.

## All-approach headline downstream comparison

The manuscript-facing rows use full-depth L8 B0 and the validation-best
performing exit for every exit-capable approach. The layer is chosen using the
validation metric only (maximum accuracy or minimum positioning error); the
held-out test split is never used for exit selection.

| Approach | Task | Exit | Selection | Test value |
| --- | --- | --- | --- | --- |
| B0 — Original LatentWave | RF sensing | L8 | full-depth L8 | 69.46% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | best validation-performing exit | 63.30% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L8 | best validation-performing exit | 72.00% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | best validation-performing exit | 72.26% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L6 | best validation-performing exit | 71.81% |
| B0 — Original LatentWave | Jamming classification | L8 | full-depth L8 | 88.00% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | best validation-performing exit | 88.00% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L8 | best validation-performing exit | 85.33% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | best validation-performing exit | 88.27% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L8 | best validation-performing exit | 88.67% |
| B0 — Original LatentWave | Human activity sensing | L8 | full-depth L8 | 98.33% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | best validation-performing exit | 96.67% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L2 | best validation-performing exit | 97.50% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | best validation-performing exit | 100.00% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L8 | best validation-performing exit | 97.50% |
| B0 — Original LatentWave | Interference classification | L8 | full-depth L8 | 74.79% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | best validation-performing exit | 75.21% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L6 | best validation-performing exit | 73.43% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | best validation-performing exit | 75.86% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L6 | best validation-performing exit | 73.93% |
| B0 — Original LatentWave | NR positioning | L8 | full-depth L8 | 3.025 m |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | best validation-performing exit | 3.025 m |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L8 | best validation-performing exit | 2.974 m |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | best validation-performing exit | 2.760 m |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L8 | best validation-performing exit | 2.874 m |
| B0 — Original LatentWave | DeepMIMO beam | L8 | full-depth L8 | 45.74% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | best validation-performing exit | 45.74% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L8 | best validation-performing exit | 44.28% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | best validation-performing exit | 43.47% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L2 | best validation-performing exit | 40.84% |

## All-approach headline inference comparison

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| B0 — Original LatentWave | RF sensing | L8 | 2.807 | 1.1175 | 1.2204 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.6789 | 0.7893 |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L8 | 2.807 | 1.0941 | 1.2155 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.1080 | 1.1874 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L6 | 2.111 | 2.5394 | 2.5530 |
| B0 — Original LatentWave | Jamming classification | L8 | 2.807 | 1.1267 | 1.2456 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 2.807 | 1.1267 | 1.2456 |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L8 | 2.807 | 1.1047 | 1.1961 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 2.807 | 1.1244 | 1.2091 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L8 | 2.807 | 3.2167 | 3.2959 |
| B0 — Original LatentWave | Human activity sensing | L8 | 10.308 | 1.1870 | 1.3541 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | 5.193 | 0.7012 | 0.8600 |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L2 | 2.635 | 0.4445 | 0.4498 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 5.193 | 0.6998 | 0.8424 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L8 | 10.308 | 2.9847 | 3.0025 |
| B0 — Original LatentWave | Interference classification | L8 | 2.807 | 1.1186 | 1.2273 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | 2.111 | 0.8998 | 1.0251 |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L6 | 2.111 | 0.8828 | 0.9540 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | 2.807 | 1.1191 | 1.2016 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L6 | 2.111 | 2.5350 | 2.5488 |
| B0 — Original LatentWave | NR positioning | L8 | 15.003 | 1.5272 | 1.6933 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5272 | 1.6933 |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L8 | 15.003 | 1.5204 | 1.7843 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.2052 | 1.3663 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L8 | 15.003 | 3.0212 | 3.0368 |
| B0 — Original LatentWave | DeepMIMO beam | L8 | 0.102 | 1.2205 | 1.3524 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 0.102 | 1.2205 | 1.3524 |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L8 | 0.102 | 1.3086 | 1.3859 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 0.8035 | 0.9121 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L2 | 0.026 | 1.3979 | 1.4220 |

## Original LatentWave full-depth context-encoder reference

| Task | Metric | Test value |
| --- | --- | --- |
| RF sensing | Macro accuracy | 69.46% |
| Jamming classification | Macro accuracy | 88.00% |
| Human activity sensing | Macro accuracy | 98.33% |
| Interference classification | Macro accuracy | 74.79% |
| NR positioning | Mean distance error (m) | 3.025 m |
| DeepMIMO beam | Macro accuracy | 45.74% |

## Detailed frozen-exit downstream metrics

Test values are shown after each head's validation-selected checkpoint. The
validation column is included for auditability; it is not a test result.

| Method | Task | Exit | Metric | Test value | Validation value | Head epoch |
| --- | --- | --- | --- | --- | --- | --- |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | Macro accuracy | 60.87% | 65.01% | 110 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | Macro accuracy | 63.30% | 71.70% | 95 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | Macro accuracy | 66.98% | 70.50% | 75 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | Macro accuracy | 69.46% | 71.08% | 163 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | Macro accuracy | 81.07% | 79.87% | 144 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | Macro accuracy | 80.53% | 78.80% | 74 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | Macro accuracy | 81.87% | 83.20% | 86 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | Macro accuracy | 88.00% | 87.73% | 21 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | Macro accuracy | 95.83% | 99.17% | 24 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | Macro accuracy | 96.67% | 100.00% | 24 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | Macro accuracy | 100.00% | 100.00% | 35 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | Macro accuracy | 98.33% | 100.00% | 26 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | Macro accuracy | 71.64% | 72.43% | 40 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | Macro accuracy | 74.07% | 73.50% | 43 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | Macro accuracy | 75.21% | 73.86% | 32 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | Macro accuracy | 74.79% | 73.79% | 34 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | Mean distance error (m) | 3.734 m | 3.644 m | 17 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | Mean distance error (m) | 3.307 m | 3.335 m | 35 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | Mean distance error (m) | 3.104 m | 3.108 m | 93 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | Mean distance error (m) | 3.025 m | 3.039 m | 59 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | Macro accuracy | 42.72% | 43.97% | 42 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | Macro accuracy | 42.14% | 43.54% | 17 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | Macro accuracy | 43.01% | 43.82% | 24 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | Macro accuracy | 45.74% | 44.69% | 54 |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L2 | Macro accuracy | 65.53% | 69.23% | 79 |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L4 | Macro accuracy | 68.16% | 69.47% | 78 |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L6 | Macro accuracy | 71.63% | 69.19% | 67 |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L8 | Macro accuracy | 72.00% | 70.47% | 101 |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L2 | Macro accuracy | 85.60% | 83.73% | 67 |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L4 | Macro accuracy | 86.00% | 86.67% | 39 |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L6 | Macro accuracy | 87.87% | 88.00% | 70 |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L8 | Macro accuracy | 85.33% | 88.40% | 26 |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L2 | Macro accuracy | 97.50% | 99.17% | 21 |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L4 | Macro accuracy | 97.50% | 98.33% | 16 |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L6 | Macro accuracy | 98.33% | 99.17% | 20 |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L8 | Macro accuracy | 98.33% | 99.17% | 15 |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L2 | Macro accuracy | 73.29% | 72.79% | 24 |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L4 | Macro accuracy | 72.14% | 73.07% | 17 |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L6 | Macro accuracy | 73.43% | 74.64% | 54 |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L8 | Macro accuracy | 74.71% | 72.93% | 47 |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L2 | Mean distance error (m) | 3.521 m | 3.485 m | 55 |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L4 | Mean distance error (m) | 3.243 m | 3.289 m | 64 |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L6 | Mean distance error (m) | 2.993 m | 3.053 m | 86 |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L8 | Mean distance error (m) | 2.974 m | 2.974 m | 38 |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | Macro accuracy | 43.26% | 45.23% | 52 |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L4 | Macro accuracy | 41.66% | 45.12% | 25 |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L6 | Macro accuracy | 45.23% | 46.19% | 49 |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L8 | Macro accuracy | 44.28% | 46.39% | 68 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | Macro accuracy | 72.25% | 68.00% | 80 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | Macro accuracy | 65.53% | 68.05% | 50 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | Macro accuracy | 74.09% | 76.11% | 98 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | Macro accuracy | 72.26% | 76.49% | 78 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | Macro accuracy | 82.67% | 84.00% | 51 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | Macro accuracy | 84.00% | 84.13% | 52 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | Macro accuracy | 86.53% | 86.93% | 80 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | Macro accuracy | 88.27% | 89.60% | 47 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | Macro accuracy | 100.00% | 99.17% | 42 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | Macro accuracy | 100.00% | 100.00% | 19 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | Macro accuracy | 100.00% | 99.17% | 17 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | Macro accuracy | 100.00% | 100.00% | 26 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | Macro accuracy | 75.29% | 74.57% | 38 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | Macro accuracy | 75.29% | 75.29% | 25 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | Macro accuracy | 75.00% | 75.29% | 18 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | Macro accuracy | 75.86% | 75.57% | 23 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | Mean distance error (m) | 3.296 m | 3.284 m | 43 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | Mean distance error (m) | 3.101 m | 3.061 m | 89 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | Mean distance error (m) | 2.760 m | 2.759 m | 62 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | Mean distance error (m) | 2.725 m | 2.771 m | 52 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | Macro accuracy | 42.68% | 44.24% | 27 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | Macro accuracy | 43.47% | 46.86% | 32 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | Macro accuracy | 41.45% | 44.57% | 19 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | Macro accuracy | 44.45% | 45.82% | 28 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L2 | Macro accuracy | 61.59% | 66.59% | 85 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L4 | Macro accuracy | 61.13% | 72.93% | 84 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L6 | Macro accuracy | 71.81% | 76.95% | 115 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L8 | Macro accuracy | 69.93% | 73.58% | 92 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L2 | Macro accuracy | 83.07% | 81.73% | 95 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L4 | Macro accuracy | 86.00% | 85.60% | 97 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L6 | Macro accuracy | 85.73% | 87.47% | 86 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L8 | Macro accuracy | 88.67% | 90.40% | 38 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L2 | Macro accuracy | 97.50% | 99.17% | 19 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L4 | Macro accuracy | 95.83% | 99.17% | 19 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L6 | Macro accuracy | 100.00% | 99.17% | 16 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L8 | Macro accuracy | 97.50% | 100.00% | 12 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L2 | Macro accuracy | 72.29% | 70.93% | 21 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L4 | Macro accuracy | 74.00% | 73.21% | 46 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L6 | Macro accuracy | 73.93% | 73.93% | 54 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L8 | Macro accuracy | 73.14% | 73.21% | 47 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L2 | Mean distance error (m) | 3.607 m | 3.591 m | 37 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L4 | Mean distance error (m) | 3.276 m | 3.286 m | 33 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L6 | Mean distance error (m) | 3.072 m | 3.055 m | 38 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L8 | Mean distance error (m) | 2.874 m | 2.902 m | 69 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L2 | Macro accuracy | 40.84% | 44.16% | 95 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L4 | Macro accuracy | 43.08% | 43.07% | 69 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L6 | Macro accuracy | 43.05% | 42.92% | 61 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L8 | Macro accuracy | 42.05% | 42.18% | 28 |

## Detailed validation-best exits

The manuscript-facing exit is selected from validation only. For positioning,
lower mean distance error is better; for classification, higher macro
accuracy is better.

| Method | Task | Policy | Exit | Test value |
| --- | --- | --- | --- | --- |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | best validation | L4 | 63.30% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | best validation | L8 | 88.00% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | best validation | L4 | 96.67% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | best validation | L6 | 75.21% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | best validation | L8 | 3.025 m |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | best validation | L8 | 45.74% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | best validation | L8 | 72.00% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | best validation | L8 | 85.33% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | best validation | L2 | 97.50% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | best validation | L6 | 73.43% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | best validation | L8 | 2.974 m |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | best validation | L8 | 44.28% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | best validation | L8 | 72.26% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | best validation | L8 | 88.27% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | best validation | L4 | 100.00% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | best validation | L8 | 75.86% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | best validation | L6 | 2.760 m |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | best validation | L4 | 43.47% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | best validation | L6 | 71.81% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | best validation | L8 | 88.67% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | best validation | L8 | 97.50% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | best validation | L6 | 73.93% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | best validation | L8 | 2.874 m |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | best validation | L2 | 40.84% |

## Detailed batch-1 inference cost

Latency is the synchronized CUDA mean/p95 measurement after warm-up on the
RTX 5070 Ti. Savings are relative to the same method/task at L8. A missing P2-Reverse
benchmark is reported as unavailable, not estimated.

| Method | Task | Exit | FLOPs (B) | Mean ms | P95 ms | FLOPs saving | Latency saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.4502 | 0.5238 | 74.3% | 59.7% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.6789 | 0.7893 | 49.5% | 39.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | 2.111 | 0.9145 | 0.9972 | 24.8% | 18.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.1175 | 1.2204 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | 0.721 | 0.4564 | 0.4830 | 74.3% | 59.5% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | 1.416 | 0.6853 | 0.7551 | 49.5% | 39.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.8979 | 0.9835 | 24.8% | 20.3% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 2.807 | 1.1267 | 1.2456 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4521 | 0.5502 | 74.4% | 61.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | 5.193 | 0.7012 | 0.8600 | 49.6% | 40.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | 7.750 | 0.9470 | 1.1033 | 24.8% | 20.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | 10.308 | 1.1870 | 1.3541 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4463 | 0.4776 | 74.3% | 60.1% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | 1.416 | 0.6719 | 0.7731 | 49.5% | 39.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | 2.111 | 0.8998 | 1.0251 | 24.8% | 19.6% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | 2.807 | 1.1186 | 1.2273 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | 3.828 | 0.5451 | 0.6899 | 74.5% | 64.3% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | 7.553 | 0.8722 | 1.0278 | 49.7% | 42.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2051 | 1.3760 | 24.8% | 21.1% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5272 | 1.6933 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.5806 | 0.6564 | 74.2% | 52.4% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | 0.052 | 0.7985 | 0.9120 | 49.5% | 34.6% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | 0.077 | 0.9947 | 1.0930 | 24.7% | 18.5% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 0.102 | 1.2205 | 1.3524 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L2 | 0.721 | 0.4453 | 0.4697 | 74.3% | 59.3% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L4 | 1.416 | 0.6706 | 0.7150 | 49.5% | 38.7% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L6 | 2.111 | 0.8748 | 0.9622 | 24.8% | 20.0% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L8 | 2.807 | 1.0941 | 1.2155 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L2 | 0.721 | 0.4554 | 0.4805 | 74.3% | 58.8% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L4 | 1.416 | 0.6757 | 0.7234 | 49.5% | 38.8% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L6 | 2.111 | 0.8852 | 0.9649 | 24.8% | 19.9% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L8 | 2.807 | 1.1047 | 1.1961 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L2 | 2.635 | 0.4445 | 0.4498 | 74.4% | 62.4% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L4 | 5.193 | 0.6872 | 0.7957 | 49.6% | 41.9% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L6 | 7.750 | 0.9297 | 1.1955 | 24.8% | 21.4% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L8 | 10.308 | 1.1826 | 1.4365 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L2 | 0.721 | 0.4480 | 0.4726 | 74.3% | 59.5% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L4 | 1.416 | 0.6692 | 0.7237 | 49.5% | 39.5% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L6 | 2.111 | 0.8828 | 0.9540 | 24.8% | 20.2% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L8 | 2.807 | 1.1057 | 1.1944 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L2 | 3.828 | 0.5411 | 0.5403 | 74.5% | 64.4% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L4 | 7.553 | 0.8697 | 0.9019 | 49.7% | 42.8% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L6 | 11.278 | 1.1907 | 1.4632 | 24.8% | 21.7% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L8 | 15.003 | 1.5204 | 1.7843 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | 0.026 | 0.6164 | 0.6534 | 74.2% | 52.9% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L4 | 0.052 | 0.8457 | 0.9367 | 49.5% | 35.4% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L6 | 0.077 | 1.0978 | 1.2236 | 24.7% | 16.1% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L8 | 0.102 | 1.3086 | 1.3859 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.4524 | 0.4835 | 74.3% | 59.2% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 0.6872 | 0.7562 | 49.5% | 38.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 0.9098 | 0.9708 | 24.8% | 17.9% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.1080 | 1.1874 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.4580 | 0.4683 | 74.3% | 59.3% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | 1.416 | 0.6880 | 0.8003 | 49.5% | 38.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | 2.111 | 0.9102 | 1.0096 | 24.8% | 19.1% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 2.807 | 1.1244 | 1.2091 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.4588 | 0.5644 | 74.4% | 61.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 5.193 | 0.6998 | 0.8424 | 49.6% | 41.5% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | 7.750 | 0.9446 | 1.0969 | 24.8% | 21.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | 10.308 | 1.1955 | 1.3379 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.4588 | 0.5002 | 74.3% | 59.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | 1.416 | 0.6708 | 0.7306 | 49.5% | 40.1% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | 2.111 | 0.9171 | 0.9938 | 24.8% | 18.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | 2.807 | 1.1191 | 1.2016 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | 3.828 | 0.5494 | 0.6407 | 74.5% | 64.3% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | 7.553 | 0.8822 | 1.0498 | 49.7% | 42.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.2052 | 1.3663 | 24.8% | 21.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.5370 | 1.6954 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.5830 | 0.6852 | 74.2% | 52.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 0.8035 | 0.9121 | 49.5% | 34.9% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | 0.077 | 1.0219 | 1.1496 | 24.7% | 17.3% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | 0.102 | 1.2350 | 1.3468 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L2 | 0.721 | 1.2285 | 1.2381 | 74.3% | 61.5% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L4 | 1.416 | 1.8834 | 1.8948 | 49.5% | 41.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L6 | 2.111 | 2.5394 | 2.5530 | 24.8% | 20.5% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L8 | 2.807 | 3.1947 | 3.2115 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L2 | 0.721 | 1.2344 | 1.2443 | 74.3% | 61.6% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L4 | 1.416 | 1.8935 | 1.9022 | 49.5% | 41.1% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L6 | 2.111 | 2.5520 | 2.5697 | 24.8% | 20.7% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L8 | 2.807 | 3.2167 | 3.2959 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L2 | 2.635 | 1.1848 | 1.1928 | 74.4% | 60.3% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L4 | 5.193 | 1.7866 | 1.7976 | 49.6% | 40.1% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L6 | 7.750 | 2.3893 | 2.4028 | 24.8% | 19.9% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L8 | 10.308 | 2.9847 | 3.0025 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L2 | 0.721 | 1.2289 | 1.2376 | 74.3% | 61.5% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L4 | 1.416 | 1.8929 | 1.9018 | 49.5% | 40.7% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L6 | 2.111 | 2.5350 | 2.5488 | 24.8% | 20.6% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L8 | 2.807 | 3.1917 | 3.2063 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L2 | 3.828 | 1.2079 | 1.2185 | 74.5% | 60.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L4 | 7.553 | 1.8119 | 1.8211 | 49.7% | 40.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L6 | 11.278 | 2.4155 | 2.4298 | 24.8% | 20.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L8 | 15.003 | 3.0212 | 3.0368 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L2 | 0.026 | 1.3979 | 1.4220 | 74.2% | 56.6% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L4 | 0.052 | 2.0119 | 2.0342 | 49.5% | 37.6% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L6 | 0.077 | 2.6093 | 2.6393 | 24.7% | 19.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L8 | 0.102 | 3.2228 | 3.2523 | 0.0% | 0.0% |

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

Diagnostic early-stopped pretraining runs, extra transfer datasets, extra
seeds, and alternate masking regimes are excluded. The same-scale auxiliary JEPA control is included when its completed P1 artifacts are supplied.
