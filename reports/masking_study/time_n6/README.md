# Time-masking single-seed comparison packet

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

Same-scale Multi-depth JEPA checkpoint: `/home/waves-lab/LatentWave_Data/preprocessed_datasets/MultiScaleJEPA-main/runs/masking_study/time_n6/p1_same_scale_time_p2matched/seed_000/checkpoints/last.ckpt`.

Pretraining experiment metadata: `/home/waves-lab/LatentWave_Data/preprocessed_datasets/MultiScaleJEPA-main/runs/masking_study/time_n6/p1_same_scale_time_p2matched/seed_000/experiment.json`.

This packet covers **Time masking (N_m=6)** and the following downstream task suite:
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
| B0 — Original LatentWave | RF sensing | L8 | full-depth L8 | 72.76% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | best validation-performing exit | 68.27% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L6 | best validation-performing exit | 69.07% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | best validation-performing exit | 74.48% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L4 | best validation-performing exit | 73.75% |
| B0 — Original LatentWave | Jamming classification | L8 | full-depth L8 | 86.80% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | best validation-performing exit | 90.53% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L2 | best validation-performing exit | 88.80% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | best validation-performing exit | 87.87% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L4 | best validation-performing exit | 88.93% |
| B0 — Original LatentWave | Human activity sensing | L8 | full-depth L8 | 99.17% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | best validation-performing exit | 100.00% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L2 | best validation-performing exit | 99.17% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | best validation-performing exit | 98.33% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L4 | best validation-performing exit | 100.00% |
| B0 — Original LatentWave | Interference classification | L8 | full-depth L8 | 70.14% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | best validation-performing exit | 73.64% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L4 | best validation-performing exit | 72.64% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | best validation-performing exit | 76.21% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L2 | best validation-performing exit | 75.50% |
| B0 — Original LatentWave | NR positioning | L8 | full-depth L8 | 3.318 m |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | best validation-performing exit | 3.318 m |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L8 | best validation-performing exit | 3.366 m |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | best validation-performing exit | 3.032 m |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L4 | best validation-performing exit | 3.346 m |
| B0 — Original LatentWave | DeepMIMO beam | L8 | full-depth L8 | 35.24% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | best validation-performing exit | 41.71% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L4 | best validation-performing exit | 41.98% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | best validation-performing exit | 42.62% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L4 | best validation-performing exit | 42.28% |

## All-approach headline inference comparison

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| B0 — Original LatentWave | RF sensing | L8 | 2.807 | 1.1221 | 1.2569 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.4529 | 0.6066 |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L6 | 2.111 | 0.9441 | 1.0313 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 1.1986 | 2.7153 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L4 | 1.416 | 1.9556 | 1.9648 |
| B0 — Original LatentWave | Jamming classification | L8 | 2.807 | 1.1183 | 1.2613 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.8836 | 1.0219 |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L2 | 0.721 | 0.4751 | 0.4966 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | 1.416 | 1.2106 | 2.7608 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L4 | 1.416 | 1.9629 | 1.9778 |
| B0 — Original LatentWave | Human activity sensing | L8 | 10.308 | 1.2337 | 1.4207 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4702 | 0.7052 |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L2 | 2.635 | 0.4663 | 0.6237 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.9278 | 2.6623 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L4 | 5.193 | 1.8496 | 1.8587 |
| B0 — Original LatentWave | Interference classification | L8 | 2.807 | 1.1167 | 1.3175 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4514 | 0.5666 |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L4 | 1.416 | 0.7236 | 0.8257 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | 2.111 | 1.4776 | 2.9724 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L2 | 0.721 | 1.2704 | 1.2818 |
| B0 — Original LatentWave | NR positioning | L8 | 15.003 | 1.5814 | 1.8182 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5814 | 1.8182 |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L8 | 15.003 | 1.5291 | 1.8344 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 2.1171 | 3.6676 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L4 | 7.553 | 1.8560 | 1.8669 |
| B0 — Original LatentWave | DeepMIMO beam | L8 | 0.102 | 1.1969 | 1.3786 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.5901 | 0.7504 |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L4 | 0.052 | 0.8417 | 0.9653 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 1.1577 | 4.7483 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L4 | 0.052 | 2.0724 | 2.0989 |

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
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | Macro accuracy | 68.27% | 72.79% | 92 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | Macro accuracy | 72.98% | 70.02% | 67 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | Macro accuracy | 68.58% | 70.12% | 69 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | Macro accuracy | 72.76% | 67.36% | 130 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | Macro accuracy | 89.07% | 88.27% | 94 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | Macro accuracy | 88.93% | 89.07% | 45 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | Macro accuracy | 90.53% | 90.27% | 89 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | Macro accuracy | 86.80% | 88.00% | 41 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | Macro accuracy | 100.00% | 100.00% | 27 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | Macro accuracy | 100.00% | 99.17% | 12 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | Macro accuracy | 99.17% | 99.17% | 6 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | Macro accuracy | 99.17% | 100.00% | 18 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | Macro accuracy | 73.64% | 74.93% | 29 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | Macro accuracy | 72.21% | 74.50% | 60 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | Macro accuracy | 73.64% | 73.86% | 35 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | Macro accuracy | 70.14% | 71.36% | 37 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | Mean distance error (m) | 3.710 m | 3.745 m | 68 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | Mean distance error (m) | 3.408 m | 3.443 m | 57 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | Mean distance error (m) | 3.418 m | 3.447 m | 75 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | Mean distance error (m) | 3.318 m | 3.421 m | 57 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | Macro accuracy | 41.71% | 43.43% | 56 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | Macro accuracy | 39.41% | 43.33% | 43 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | Macro accuracy | 43.03% | 41.04% | 31 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | Macro accuracy | 35.24% | 39.43% | 14 |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L2 | Macro accuracy | 67.31% | 69.03% | 60 |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L4 | Macro accuracy | 77.15% | 70.04% | 68 |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L6 | Macro accuracy | 69.07% | 75.75% | 100 |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L8 | Macro accuracy | 69.97% | 72.90% | 71 |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L2 | Macro accuracy | 88.80% | 89.87% | 84 |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L4 | Macro accuracy | 88.53% | 89.73% | 79 |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L6 | Macro accuracy | 86.00% | 87.33% | 27 |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L8 | Macro accuracy | 84.40% | 87.07% | 60 |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L2 | Macro accuracy | 99.17% | 100.00% | 16 |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L4 | Macro accuracy | 99.17% | 100.00% | 9 |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L6 | Macro accuracy | 100.00% | 99.17% | 6 |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L8 | Macro accuracy | 100.00% | 99.17% | 16 |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L2 | Macro accuracy | 72.93% | 72.36% | 18 |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L4 | Macro accuracy | 72.64% | 75.50% | 47 |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L6 | Macro accuracy | 73.00% | 71.14% | 53 |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L8 | Macro accuracy | 72.50% | 72.14% | 34 |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L2 | Mean distance error (m) | 3.552 m | 3.507 m | 61 |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L4 | Mean distance error (m) | 3.349 m | 3.368 m | 49 |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L6 | Mean distance error (m) | 3.337 m | 3.316 m | 58 |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L8 | Mean distance error (m) | 3.366 m | 3.313 m | 67 |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | Macro accuracy | 38.13% | 39.69% | 25 |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L4 | Macro accuracy | 41.98% | 43.24% | 74 |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L6 | Macro accuracy | 40.66% | 41.36% | 21 |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L8 | Macro accuracy | 39.93% | 42.03% | 111 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | Macro accuracy | 69.87% | 66.89% | 27 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | Macro accuracy | 74.48% | 79.14% | 101 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | Macro accuracy | 71.50% | 75.35% | 62 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | Macro accuracy | 73.44% | 74.44% | 118 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | Macro accuracy | 86.93% | 86.67% | 71 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | Macro accuracy | 87.87% | 88.40% | 22 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | Macro accuracy | 86.93% | 87.33% | 44 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | Macro accuracy | 86.27% | 84.53% | 65 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | Macro accuracy | 98.33% | 100.00% | 14 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | Macro accuracy | 100.00% | 100.00% | 16 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | Macro accuracy | 98.33% | 100.00% | 33 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | Macro accuracy | 97.50% | 100.00% | 32 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | Macro accuracy | 74.64% | 73.07% | 70 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | Macro accuracy | 75.07% | 76.07% | 27 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | Macro accuracy | 76.21% | 76.14% | 65 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | Macro accuracy | 73.50% | 74.36% | 31 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | Mean distance error (m) | 3.547 m | 3.515 m | 61 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | Mean distance error (m) | 3.245 m | 3.285 m | 74 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | Mean distance error (m) | 3.032 m | 3.088 m | 59 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | Mean distance error (m) | 3.099 m | 3.120 m | 61 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | Macro accuracy | 42.62% | 45.86% | 79 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | Macro accuracy | 40.40% | 44.24% | 20 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | Macro accuracy | 43.99% | 44.39% | 21 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | Macro accuracy | 41.71% | 44.02% | 93 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L2 | Macro accuracy | 70.27% | 73.68% | 95 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L4 | Macro accuracy | 73.75% | 78.29% | 87 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L6 | Macro accuracy | 75.69% | 75.28% | 124 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L8 | Macro accuracy | 69.12% | 76.63% | 110 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L2 | Macro accuracy | 87.60% | 86.93% | 66 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L4 | Macro accuracy | 88.93% | 89.47% | 72 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L6 | Macro accuracy | 89.47% | 88.53% | 43 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L8 | Macro accuracy | 86.67% | 87.60% | 75 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L2 | Macro accuracy | 99.17% | 99.17% | 26 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L4 | Macro accuracy | 100.00% | 100.00% | 24 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L6 | Macro accuracy | 100.00% | 100.00% | 24 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L8 | Macro accuracy | 99.17% | 99.17% | 16 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L2 | Macro accuracy | 75.50% | 74.43% | 62 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L4 | Macro accuracy | 74.93% | 73.00% | 33 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L6 | Macro accuracy | 72.79% | 73.64% | 37 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L8 | Macro accuracy | 72.93% | 72.57% | 32 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L2 | Mean distance error (m) | 3.623 m | 3.637 m | 49 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L4 | Mean distance error (m) | 3.346 m | 3.349 m | 65 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L6 | Mean distance error (m) | 3.333 m | 3.373 m | 42 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L8 | Mean distance error (m) | 3.446 m | 3.393 m | 48 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L2 | Macro accuracy | 41.11% | 42.75% | 58 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L4 | Macro accuracy | 42.28% | 43.75% | 49 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L6 | Macro accuracy | 39.82% | 43.38% | 30 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L8 | Macro accuracy | 38.75% | 41.41% | 54 |

## Detailed validation-best exits

The manuscript-facing exit is selected from validation only. For positioning,
lower mean distance error is better; for classification, higher macro
accuracy is better.

| Method | Task | Policy | Exit | Test value |
| --- | --- | --- | --- | --- |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | best validation | L2 | 68.27% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | best validation | L6 | 90.53% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | best validation | L2 | 100.00% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | best validation | L2 | 73.64% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | best validation | L8 | 3.318 m |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | best validation | L2 | 41.71% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | best validation | L6 | 69.07% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | best validation | L2 | 88.80% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | best validation | L2 | 99.17% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | best validation | L4 | 72.64% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | best validation | L8 | 3.366 m |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | best validation | L4 | 41.98% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | best validation | L4 | 74.48% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | best validation | L4 | 87.87% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | best validation | L2 | 98.33% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | best validation | L6 | 76.21% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | best validation | L6 | 3.032 m |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | best validation | L2 | 42.62% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | best validation | L4 | 73.75% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | best validation | L4 | 88.93% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | best validation | L4 | 100.00% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | best validation | L2 | 75.50% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | best validation | L4 | 3.346 m |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | best validation | L4 | 42.28% |

## Detailed batch-1 inference cost

Latency is the synchronized CUDA mean/p95 measurement after warm-up on the
RTX 5070 Ti. Savings are relative to the same method/task at L8. A missing P2-Reverse
benchmark is reported as unavailable, not estimated.

| Method | Task | Exit | FLOPs (B) | Mean ms | P95 ms | FLOPs saving | Latency saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.4529 | 0.6066 | 74.3% | 59.6% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.6796 | 0.7927 | 49.5% | 39.4% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | 2.111 | 0.9041 | 1.0494 | 24.8% | 19.4% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.1221 | 1.2569 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | 0.721 | 0.4636 | 0.5966 | 74.3% | 58.5% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | 1.416 | 0.6783 | 0.7795 | 49.5% | 39.3% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.8836 | 1.0219 | 24.8% | 21.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 2.807 | 1.1183 | 1.2613 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4702 | 0.7052 | 74.4% | 61.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | 5.193 | 0.7166 | 0.9296 | 49.6% | 41.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | 7.750 | 0.9719 | 1.1741 | 24.8% | 21.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | 10.308 | 1.2337 | 1.4207 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4514 | 0.5666 | 74.3% | 59.6% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | 1.416 | 0.6712 | 0.7938 | 49.5% | 39.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | 2.111 | 0.8860 | 1.0409 | 24.8% | 20.7% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | 2.807 | 1.1167 | 1.3175 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | 3.828 | 0.5620 | 0.7873 | 74.5% | 64.5% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | 7.553 | 0.9480 | 1.2454 | 49.7% | 40.1% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2453 | 1.4504 | 24.8% | 21.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5814 | 1.8182 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.5901 | 0.7504 | 74.2% | 50.7% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | 0.052 | 0.7879 | 0.9513 | 49.5% | 34.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | 0.077 | 1.0136 | 1.2005 | 24.7% | 15.3% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 0.102 | 1.1969 | 1.3786 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L2 | 0.721 | 0.5063 | 0.5239 | 74.3% | 56.5% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L4 | 1.416 | 0.7739 | 0.8237 | 49.5% | 33.5% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L6 | 2.111 | 0.9441 | 1.0313 | 24.8% | 18.9% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L8 | 2.807 | 1.1638 | 1.2671 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L2 | 0.721 | 0.4751 | 0.4966 | 74.3% | 59.8% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L4 | 1.416 | 0.7149 | 0.7781 | 49.5% | 39.5% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L6 | 2.111 | 0.9386 | 1.0185 | 24.8% | 20.5% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L8 | 2.807 | 1.1811 | 1.2827 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L2 | 2.635 | 0.4663 | 0.6237 | 74.4% | 61.0% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L4 | 5.193 | 0.7069 | 0.8740 | 49.6% | 40.9% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L6 | 7.750 | 0.9575 | 1.2173 | 24.8% | 19.9% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L8 | 10.308 | 1.1958 | 1.4901 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L2 | 0.721 | 0.4736 | 0.5000 | 74.3% | 59.5% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L4 | 1.416 | 0.7236 | 0.8257 | 49.5% | 38.1% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L6 | 2.111 | 0.9613 | 1.1099 | 24.8% | 17.8% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L8 | 2.807 | 1.1693 | 1.2758 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L2 | 3.828 | 0.5499 | 0.6950 | 74.5% | 64.0% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L4 | 7.553 | 0.8730 | 1.0426 | 49.7% | 42.9% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L6 | 11.278 | 1.2014 | 1.5147 | 24.8% | 21.4% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L8 | 15.003 | 1.5291 | 1.8344 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | 0.026 | 0.6172 | 0.7605 | 74.2% | 51.5% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L4 | 0.052 | 0.8417 | 0.9653 | 49.5% | 33.8% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L6 | 0.077 | 1.0553 | 1.1988 | 24.7% | 17.0% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L8 | 0.102 | 1.2715 | 1.4212 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.8352 | 2.5321 | 74.3% | 52.4% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 1.1986 | 2.7153 | 49.5% | 31.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 1.4615 | 2.7603 | 24.8% | 16.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.7530 | 3.0962 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.9171 | 2.5556 | 74.3% | 47.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | 1.416 | 1.2106 | 2.7608 | 49.5% | 30.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | 2.111 | 1.4744 | 2.8694 | 24.8% | 15.7% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 2.807 | 1.7492 | 3.0914 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.9278 | 2.6623 | 74.4% | 55.3% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 5.193 | 1.2912 | 2.9072 | 49.6% | 37.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | 7.750 | 1.6255 | 3.3916 | 24.8% | 21.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | 10.308 | 2.0746 | 3.6463 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.9485 | 2.5553 | 74.3% | 45.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | 1.416 | 1.2102 | 2.6611 | 49.5% | 30.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | 2.111 | 1.4776 | 2.9724 | 24.8% | 15.5% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | 2.807 | 1.7494 | 3.0894 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | 3.828 | 0.9660 | 2.7893 | 74.5% | 61.4% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | 7.553 | 1.5646 | 3.1305 | 49.7% | 37.4% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 2.1171 | 3.6676 | 24.8% | 15.3% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 2.5004 | 4.0001 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 1.1577 | 4.7483 | 74.2% | 48.1% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 1.5087 | 4.8161 | 49.5% | 32.3% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | 0.077 | 1.8778 | 4.9617 | 24.7% | 15.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | 0.102 | 2.2290 | 5.2559 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L2 | 0.721 | 1.2731 | 1.2851 | 74.3% | 61.4% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L4 | 1.416 | 1.9556 | 1.9648 | 49.5% | 40.7% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L6 | 2.111 | 2.6314 | 2.6440 | 24.8% | 20.3% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L8 | 2.807 | 3.3005 | 3.3198 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L2 | 0.721 | 1.2789 | 1.2895 | 74.3% | 61.6% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L4 | 1.416 | 1.9629 | 1.9778 | 49.5% | 41.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L6 | 2.111 | 2.6396 | 2.6545 | 24.8% | 20.7% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L8 | 2.807 | 3.3285 | 3.3395 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L2 | 2.635 | 1.2231 | 1.2333 | 74.4% | 60.7% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L4 | 5.193 | 1.8496 | 1.8587 | 49.6% | 40.5% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L6 | 7.750 | 2.4784 | 2.4937 | 24.8% | 20.3% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L8 | 10.308 | 3.1105 | 3.1168 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L2 | 0.721 | 1.2704 | 1.2818 | 74.3% | 61.6% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L4 | 1.416 | 1.9519 | 1.9641 | 49.5% | 40.9% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L6 | 2.111 | 2.6354 | 2.6451 | 24.8% | 20.3% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L8 | 2.807 | 3.3050 | 3.3219 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L2 | 3.828 | 1.2390 | 1.2487 | 74.5% | 59.9% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L4 | 7.553 | 1.8560 | 1.8669 | 49.7% | 39.9% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L6 | 11.278 | 2.4762 | 2.4949 | 24.8% | 19.8% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L8 | 15.003 | 3.0884 | 3.1045 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L2 | 0.026 | 1.4456 | 1.4732 | 74.2% | 56.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L4 | 0.052 | 2.0724 | 2.0989 | 49.5% | 37.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L6 | 0.077 | 2.6677 | 2.7023 | 24.7% | 18.9% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L8 | 0.102 | 3.2881 | 3.3197 | 0.0% | 0.0% |

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
