# Frequency-masking single-seed comparison packet

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

P2-Reverse was imported from the verified artifact archive (claimed source commit 90c4793). Its six-task held-out metrics and synchronized batch-1 benchmark are available; its persisted split-index file was not included, so exact split identity cannot be re-verified locally.


## Task-suite note

## P1 checkpoint provenance

Same-scale Multi-depth JEPA checkpoint: `/home/waves-lab/LatentWave_Data/preprocessed_datasets/MultiScaleJEPA-main/runs/masking_study/frequency_n6/p1_same_scale_frequency_p2matched/seed_000/checkpoints/last.ckpt`.

Pretraining experiment metadata: `/home/waves-lab/LatentWave_Data/preprocessed_datasets/MultiScaleJEPA-main/runs/masking_study/frequency_n6/p1_same_scale_frequency_p2matched/seed_000/experiment.json`.

This packet covers **Frequency masking (N_m=6)** and the following downstream task suite:
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
| B0 — Original LatentWave | RF sensing | L8 | full-depth L8 | 68.32% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | best validation-performing exit | 68.32% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L8 | best validation-performing exit | 72.00% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | best validation-performing exit | 71.13% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L8 | best validation-performing exit | 69.92% |
| B0 — Original LatentWave | Jamming classification | L8 | full-depth L8 | 89.73% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | best validation-performing exit | 89.73% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L8 | best validation-performing exit | 85.33% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | best validation-performing exit | 88.27% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L8 | best validation-performing exit | 87.73% |
| B0 — Original LatentWave | Human activity sensing | L8 | full-depth L8 | 100.00% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | best validation-performing exit | 95.00% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L2 | best validation-performing exit | 97.50% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | best validation-performing exit | 99.17% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L2 | best validation-performing exit | 97.50% |
| B0 — Original LatentWave | Interference classification | L8 | full-depth L8 | 71.64% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | best validation-performing exit | 72.14% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L6 | best validation-performing exit | 73.43% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | best validation-performing exit | 74.21% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L4 | best validation-performing exit | 73.57% |
| B0 — Original LatentWave | NR positioning | L8 | full-depth L8 | 3.157 m |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | best validation-performing exit | 3.157 m |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L8 | best validation-performing exit | 2.974 m |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | best validation-performing exit | 2.891 m |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L8 | best validation-performing exit | 2.877 m |
| B0 — Original LatentWave | DeepMIMO beam | L8 | full-depth L8 | 42.84% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | best validation-performing exit | 44.18% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L8 | best validation-performing exit | 44.28% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | best validation-performing exit | 41.17% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L6 | best validation-performing exit | 42.87% |

## All-approach headline inference comparison

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| B0 — Original LatentWave | RF sensing | L8 | 2.807 | 1.1259 | 1.2477 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.1259 | 1.2477 |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L8 | 2.807 | 1.0941 | 1.2155 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.1988 | 1.4799 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L8 | 2.807 | 3.2274 | 3.2449 |
| B0 — Original LatentWave | Jamming classification | L8 | 2.807 | 1.1180 | 1.2667 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 2.807 | 1.1180 | 1.2667 |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L8 | 2.807 | 1.1047 | 1.1961 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 2.807 | 1.1914 | 1.4280 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L8 | 2.807 | 3.2528 | 3.2569 |
| B0 — Original LatentWave | Human activity sensing | L8 | 10.308 | 1.1939 | 1.3596 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | 5.193 | 0.6990 | 0.8777 |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L2 | 2.635 | 0.4445 | 0.4498 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 5.193 | 0.7665 | 1.0774 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L2 | 2.635 | 1.1999 | 1.2101 |
| B0 — Original LatentWave | Interference classification | L8 | 2.807 | 1.1184 | 1.2317 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | 2.111 | 0.9005 | 1.0228 |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L6 | 2.111 | 0.8828 | 0.9540 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | 1.416 | 0.7602 | 0.9142 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L4 | 1.416 | 1.9190 | 1.9285 |
| B0 — Original LatentWave | NR positioning | L8 | 15.003 | 1.5560 | 1.7544 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5560 | 1.7544 |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L8 | 15.003 | 1.5204 | 1.7843 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.6759 | 2.7483 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L8 | 15.003 | 3.0338 | 3.0496 |
| B0 — Original LatentWave | DeepMIMO beam | L8 | 0.102 | 1.2222 | 1.2982 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | 0.077 | 1.0210 | 1.1488 |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L8 | 0.102 | 1.3086 | 1.3859 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.6414 | 0.7627 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L6 | 0.077 | 2.6288 | 2.6589 |

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
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | Macro accuracy | 56.14% | 62.07% | 79 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | Macro accuracy | 67.59% | 73.35% | 153 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | Macro accuracy | 64.84% | 67.47% | 58 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | Macro accuracy | 68.32% | 79.85% | 123 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | Macro accuracy | 79.60% | 77.73% | 62 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | Macro accuracy | 82.13% | 80.67% | 76 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | Macro accuracy | 87.07% | 86.80% | 71 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | Macro accuracy | 89.73% | 89.60% | 59 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | Macro accuracy | 95.83% | 99.17% | 26 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | Macro accuracy | 95.00% | 100.00% | 21 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | Macro accuracy | 96.67% | 100.00% | 14 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | Macro accuracy | 100.00% | 99.17% | 21 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | Macro accuracy | 71.43% | 72.29% | 26 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | Macro accuracy | 71.00% | 72.00% | 42 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | Macro accuracy | 72.14% | 72.64% | 50 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | Macro accuracy | 71.64% | 71.79% | 47 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | Mean distance error (m) | 3.709 m | 3.673 m | 73 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | Mean distance error (m) | 3.478 m | 3.401 m | 26 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | Mean distance error (m) | 3.291 m | 3.247 m | 75 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | Mean distance error (m) | 3.157 m | 3.110 m | 80 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | Macro accuracy | 39.80% | 42.75% | 61 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | Macro accuracy | 42.16% | 43.19% | 34 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | Macro accuracy | 44.18% | 44.38% | 68 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | Macro accuracy | 42.84% | 43.56% | 24 |
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
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | Macro accuracy | 71.72% | 72.01% | 131 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | Macro accuracy | 65.42% | 69.53% | 74 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | Macro accuracy | 66.25% | 72.68% | 56 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | Macro accuracy | 71.13% | 72.71% | 76 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | Macro accuracy | 81.87% | 82.67% | 59 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | Macro accuracy | 86.53% | 88.67% | 84 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | Macro accuracy | 85.20% | 86.80% | 52 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | Macro accuracy | 88.27% | 88.67% | 78 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | Macro accuracy | 96.67% | 99.17% | 26 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | Macro accuracy | 99.17% | 100.00% | 38 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | Macro accuracy | 97.50% | 100.00% | 8 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | Macro accuracy | 96.67% | 99.17% | 6 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | Macro accuracy | 73.21% | 72.64% | 18 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | Macro accuracy | 74.21% | 74.57% | 10 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | Macro accuracy | 73.57% | 73.07% | 15 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | Macro accuracy | 73.00% | 73.21% | 23 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | Mean distance error (m) | 3.619 m | 3.554 m | 35 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | Mean distance error (m) | 3.317 m | 3.312 m | 44 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | Mean distance error (m) | 3.046 m | 2.998 m | 31 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | Mean distance error (m) | 2.891 m | 2.853 m | 47 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | Macro accuracy | 41.17% | 44.10% | 42 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | Macro accuracy | 41.44% | 43.19% | 35 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | Macro accuracy | 41.94% | 43.38% | 39 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | Macro accuracy | 41.38% | 43.31% | 35 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L2 | Macro accuracy | 66.47% | 68.81% | 127 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L4 | Macro accuracy | 66.54% | 70.65% | 101 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L6 | Macro accuracy | 66.32% | 69.38% | 44 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L8 | Macro accuracy | 69.92% | 76.30% | 114 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L2 | Macro accuracy | 84.53% | 83.60% | 69 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L4 | Macro accuracy | 83.60% | 84.13% | 60 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L6 | Macro accuracy | 84.80% | 85.87% | 97 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L8 | Macro accuracy | 87.73% | 89.87% | 40 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L2 | Macro accuracy | 97.50% | 99.17% | 25 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L4 | Macro accuracy | 99.17% | 99.17% | 29 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L6 | Macro accuracy | 95.83% | 99.17% | 15 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L8 | Macro accuracy | 99.17% | 99.17% | 50 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L2 | Macro accuracy | 72.64% | 71.86% | 17 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L4 | Macro accuracy | 73.57% | 73.71% | 49 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L6 | Macro accuracy | 73.07% | 73.07% | 34 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L8 | Macro accuracy | 73.93% | 73.14% | 18 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L2 | Mean distance error (m) | 3.547 m | 3.485 m | 37 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L4 | Mean distance error (m) | 3.245 m | 3.239 m | 68 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L6 | Mean distance error (m) | 3.010 m | 3.029 m | 61 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L8 | Mean distance error (m) | 2.877 m | 2.860 m | 58 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L2 | Macro accuracy | 45.80% | 45.87% | 50 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L4 | Macro accuracy | 45.44% | 45.73% | 58 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L6 | Macro accuracy | 42.87% | 46.11% | 69 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L8 | Macro accuracy | 42.28% | 43.49% | 22 |

## Detailed validation-best exits

The manuscript-facing exit is selected from validation only. For positioning,
lower mean distance error is better; for classification, higher macro
accuracy is better.

| Method | Task | Policy | Exit | Test value |
| --- | --- | --- | --- | --- |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | best validation | L8 | 68.32% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | best validation | L8 | 89.73% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | best validation | L4 | 95.00% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | best validation | L6 | 72.14% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | best validation | L8 | 3.157 m |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | best validation | L6 | 44.18% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | best validation | L8 | 72.00% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | best validation | L8 | 85.33% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | best validation | L2 | 97.50% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | best validation | L6 | 73.43% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | best validation | L8 | 2.974 m |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | best validation | L8 | 44.28% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | best validation | L8 | 71.13% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | best validation | L8 | 88.27% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | best validation | L4 | 99.17% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | best validation | L4 | 74.21% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | best validation | L8 | 2.891 m |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | best validation | L2 | 41.17% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | best validation | L8 | 69.92% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | best validation | L8 | 87.73% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | best validation | L2 | 97.50% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | best validation | L4 | 73.57% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | best validation | L8 | 2.877 m |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | best validation | L6 | 42.87% |

## Detailed batch-1 inference cost

Latency is the synchronized CUDA mean/p95 measurement after warm-up on the
RTX 5070 Ti. Savings are relative to the same method/task at L8. A missing P2-Reverse
benchmark is reported as unavailable, not estimated.

| Method | Task | Exit | FLOPs (B) | Mean ms | P95 ms | FLOPs saving | Latency saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.4533 | 0.4842 | 74.3% | 59.7% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.6903 | 0.7766 | 49.5% | 38.7% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | 2.111 | 0.8981 | 0.9808 | 24.8% | 20.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.1259 | 1.2477 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | 0.721 | 0.4559 | 0.4861 | 74.3% | 59.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | 1.416 | 0.6809 | 0.7705 | 49.5% | 39.1% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.9049 | 1.0111 | 24.8% | 19.1% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 2.807 | 1.1180 | 1.2667 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4585 | 0.6210 | 74.4% | 61.6% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | 5.193 | 0.6990 | 0.8777 | 49.6% | 41.5% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | 7.750 | 0.9481 | 1.1484 | 24.8% | 20.6% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | 10.308 | 1.1939 | 1.3596 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4493 | 0.4954 | 74.3% | 59.8% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | 1.416 | 0.6744 | 0.7613 | 49.5% | 39.7% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | 2.111 | 0.9005 | 1.0228 | 24.8% | 19.5% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | 2.807 | 1.1184 | 1.2317 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | 3.828 | 0.5577 | 0.7190 | 74.5% | 64.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | 7.553 | 0.8867 | 1.0755 | 49.7% | 43.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2147 | 1.3986 | 24.8% | 21.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5560 | 1.7544 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.5903 | 0.7066 | 74.2% | 51.7% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | 0.052 | 0.8039 | 0.9136 | 49.5% | 34.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | 0.077 | 1.0210 | 1.1488 | 24.7% | 16.5% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 0.102 | 1.2222 | 1.2982 | 0.0% | 0.0% |
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
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.5170 | 0.7086 | 74.3% | 56.9% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 0.7326 | 0.8937 | 49.5% | 38.9% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 0.9761 | 1.1960 | 24.8% | 18.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.1988 | 1.4799 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.4821 | 0.5706 | 74.3% | 59.5% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | 1.416 | 0.7376 | 0.8430 | 49.5% | 38.1% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | 2.111 | 0.9890 | 1.2134 | 24.8% | 17.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 2.807 | 1.1914 | 1.4280 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.4931 | 0.6314 | 74.4% | 62.5% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 5.193 | 0.7665 | 1.0774 | 49.6% | 41.7% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | 7.750 | 1.0181 | 1.3125 | 24.8% | 22.5% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | 10.308 | 1.3140 | 2.3789 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.5121 | 0.6582 | 74.3% | 57.9% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | 1.416 | 0.7602 | 0.9142 | 49.5% | 37.4% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | 2.111 | 0.9814 | 1.2573 | 24.8% | 19.2% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | 2.807 | 1.2151 | 1.7694 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | 3.828 | 0.6167 | 0.9022 | 74.5% | 63.2% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | 7.553 | 0.9776 | 1.2880 | 49.7% | 41.7% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.3338 | 2.4615 | 24.8% | 20.4% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.6759 | 2.7483 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.6414 | 0.7627 | 74.2% | 51.1% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 0.8844 | 1.0191 | 49.5% | 32.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | 0.077 | 1.0857 | 1.2107 | 24.7% | 17.2% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | 0.102 | 1.3115 | 1.5230 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L2 | 0.721 | 1.2471 | 1.2593 | 74.3% | 61.4% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L4 | 1.416 | 1.9101 | 1.9214 | 49.5% | 40.8% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L6 | 2.111 | 2.5769 | 2.5938 | 24.8% | 20.2% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L8 | 2.807 | 3.2274 | 3.2449 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L2 | 0.721 | 1.2533 | 1.2643 | 74.3% | 61.5% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L4 | 1.416 | 1.9184 | 1.9334 | 49.5% | 41.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L6 | 2.111 | 2.5819 | 2.5981 | 24.8% | 20.6% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L8 | 2.807 | 3.2528 | 3.2569 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L2 | 2.635 | 1.1999 | 1.2101 | 74.4% | 60.4% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L4 | 5.193 | 1.8140 | 1.8255 | 49.6% | 40.2% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L6 | 7.750 | 2.4230 | 2.4384 | 24.8% | 20.1% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L8 | 10.308 | 3.0324 | 3.0479 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L2 | 0.721 | 1.2507 | 1.2613 | 74.3% | 61.4% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L4 | 1.416 | 1.9190 | 1.9285 | 49.5% | 40.8% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L6 | 2.111 | 2.5835 | 2.5951 | 24.8% | 20.3% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L8 | 2.807 | 3.2403 | 3.2641 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L2 | 3.828 | 1.2181 | 1.2285 | 74.5% | 59.9% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L4 | 7.553 | 1.8313 | 1.8449 | 49.7% | 39.6% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L6 | 11.278 | 2.4354 | 2.4493 | 24.8% | 19.7% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L8 | 15.003 | 3.0338 | 3.0496 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L2 | 0.026 | 1.4228 | 1.4517 | 74.2% | 56.3% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L4 | 0.052 | 2.0450 | 2.0886 | 49.5% | 37.2% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L6 | 0.077 | 2.6288 | 2.6589 | 24.7% | 19.3% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L8 | 0.102 | 3.2590 | 3.3209 | 0.0% | 0.0% |

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
