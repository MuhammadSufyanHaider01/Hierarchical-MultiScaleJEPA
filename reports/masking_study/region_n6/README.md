# Region-masking single-seed comparison packet

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

P2-Reverse was imported from the verified artifact archive (claimed source commit 90c4793). Its six-task held-out metrics and benchmark are available; its persisted split-index file was not included, so exact split identity cannot be re-verified locally. Region P2-Reverse pretraining metadata is incomplete: the supplied summary describes downstream evaluation rather than an original pretraining summary.


## Task-suite note

## P1 checkpoint provenance

Same-scale Multi-depth JEPA checkpoint: `/home/waves-lab/LatentWave_Data/preprocessed_datasets/MultiScaleJEPA-main/runs/masking_study/region_n6/p1_same_scale_region_p2matched/seed_000/checkpoints/last.ckpt`.

Pretraining experiment metadata: `/home/waves-lab/LatentWave_Data/preprocessed_datasets/MultiScaleJEPA-main/runs/masking_study/region_n6/p1_same_scale_region_p2matched/seed_000/experiment.json`.

This packet covers **Region masking (N_m=6)** and the following downstream task suite:
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
| B0 — Original LatentWave | RF sensing | L8 | full-depth L8 | 78.97% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | best validation-performing exit | 78.97% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L4 | best validation-performing exit | 82.56% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | best validation-performing exit | 76.30% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L6 | best validation-performing exit | 75.76% |
| B0 — Original LatentWave | Jamming classification | L8 | full-depth L8 | 85.47% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | best validation-performing exit | 87.60% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L8 | best validation-performing exit | 86.67% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | best validation-performing exit | 86.27% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L6 | best validation-performing exit | 88.13% |
| B0 — Original LatentWave | Human activity sensing | L8 | full-depth L8 | 96.67% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | best validation-performing exit | 96.67% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L4 | best validation-performing exit | 96.67% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | best validation-performing exit | 97.50% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L2 | best validation-performing exit | 97.50% |
| B0 — Original LatentWave | Interference classification | L8 | full-depth L8 | 71.29% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | best validation-performing exit | 71.57% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L4 | best validation-performing exit | 74.71% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | best validation-performing exit | 74.93% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L4 | best validation-performing exit | 74.36% |
| B0 — Original LatentWave | NR positioning | L8 | full-depth L8 | 2.939 m |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | best validation-performing exit | 2.939 m |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L8 | best validation-performing exit | 2.694 m |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | best validation-performing exit | 2.759 m |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L8 | best validation-performing exit | 2.776 m |
| B0 — Original LatentWave | DeepMIMO beam | L8 | full-depth L8 | 39.09% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | best validation-performing exit | 41.46% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | best validation-performing exit | 49.27% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | best validation-performing exit | 44.46% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L2 | best validation-performing exit | 44.41% |

## All-approach headline inference comparison

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| B0 — Original LatentWave | RF sensing | L8 | 2.807 | 1.2750 | 1.4085 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.2750 | 1.4085 |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L4 | 1.416 | 0.6742 | 0.6855 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.2549 | 1.9018 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L6 | 2.111 | 1.7043 | 1.7146 |
| B0 — Original LatentWave | Jamming classification | L8 | 2.807 | 1.1578 | 1.2586 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.9318 | 1.0112 |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L8 | 2.807 | 1.0937 | 1.1360 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | 1.416 | 0.7819 | 1.3971 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L6 | 2.111 | 1.6786 | 1.7061 |
| B0 — Original LatentWave | Human activity sensing | L8 | 10.308 | 1.2174 | 1.3959 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | 10.308 | 1.2174 | 1.3959 |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L4 | 5.193 | 0.6922 | 0.6885 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.4440 | 0.4584 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L2 | 2.635 | 0.8883 | 0.9066 |
| B0 — Original LatentWave | Interference classification | L8 | 2.807 | 1.1557 | 1.2556 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | 1.416 | 0.7035 | 0.8086 |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L4 | 1.416 | 0.6644 | 0.6785 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | 2.111 | 0.9504 | 0.9825 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L4 | 1.416 | 1.2959 | 1.3233 |
| B0 — Original LatentWave | NR positioning | L8 | 15.003 | 1.5557 | 1.7371 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5557 | 1.7371 |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L8 | 15.003 | 1.5327 | 1.5290 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.5602 | 2.1005 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L8 | 15.003 | 2.0336 | 2.0608 |
| B0 — Original LatentWave | DeepMIMO beam | L8 | 0.102 | 1.2556 | 1.3944 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | 0.077 | 1.0366 | 1.1626 |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | 0.026 | 0.5739 | 0.5849 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.5712 | 0.6024 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L2 | 0.026 | 1.3808 | 1.4593 |

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
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | Macro accuracy | 69.78% | 73.48% | 82 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | Macro accuracy | 75.16% | 75.42% | 52 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | Macro accuracy | 75.05% | 76.97% | 41 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | Macro accuracy | 78.97% | 78.75% | 64 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | Macro accuracy | 82.00% | 79.33% | 53 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | Macro accuracy | 84.00% | 80.80% | 39 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | Macro accuracy | 87.60% | 86.80% | 83 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | Macro accuracy | 85.47% | 85.33% | 72 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | Macro accuracy | 97.50% | 98.33% | 54 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | Macro accuracy | 98.33% | 99.17% | 32 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | Macro accuracy | 99.17% | 99.17% | 22 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | Macro accuracy | 96.67% | 100.00% | 26 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | Macro accuracy | 73.43% | 72.14% | 23 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | Macro accuracy | 71.57% | 74.21% | 33 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | Macro accuracy | 70.07% | 72.43% | 43 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | Macro accuracy | 71.29% | 72.14% | 64 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | Mean distance error (m) | 3.546 m | 3.529 m | 73 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | Mean distance error (m) | 3.264 m | 3.297 m | 62 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | Mean distance error (m) | 3.012 m | 3.043 m | 89 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | Mean distance error (m) | 2.939 m | 3.007 m | 55 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | Macro accuracy | 42.52% | 43.48% | 66 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | Macro accuracy | 39.38% | 42.24% | 33 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | Macro accuracy | 41.46% | 43.85% | 34 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | Macro accuracy | 39.09% | 41.59% | 54 |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L2 | Macro accuracy | 75.93% | 78.13% | 113 |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L4 | Macro accuracy | 82.56% | 82.35% | 102 |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L6 | Macro accuracy | 73.88% | 78.04% | 88 |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L8 | Macro accuracy | 78.19% | 76.02% | 94 |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L2 | Macro accuracy | 87.33% | 87.47% | 69 |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L4 | Macro accuracy | 86.93% | 86.53% | 54 |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L6 | Macro accuracy | 86.13% | 87.73% | 70 |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L8 | Macro accuracy | 86.67% | 89.33% | 51 |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L2 | Macro accuracy | 95.83% | 98.33% | 24 |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L4 | Macro accuracy | 96.67% | 100.00% | 13 |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L6 | Macro accuracy | 99.17% | 100.00% | 19 |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L8 | Macro accuracy | 100.00% | 100.00% | 37 |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L2 | Macro accuracy | 73.00% | 71.93% | 61 |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L4 | Macro accuracy | 74.71% | 73.29% | 51 |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L6 | Macro accuracy | 74.36% | 73.21% | 24 |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L8 | Macro accuracy | 74.50% | 73.14% | 23 |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L2 | Mean distance error (m) | 3.259 m | 3.229 m | 61 |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L4 | Mean distance error (m) | 2.945 m | 2.964 m | 82 |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L6 | Mean distance error (m) | 2.785 m | 2.825 m | 92 |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L8 | Mean distance error (m) | 2.694 m | 2.713 m | 75 |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | Macro accuracy | 49.27% | 47.26% | 66 |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L4 | Macro accuracy | 43.47% | 45.00% | 30 |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L6 | Macro accuracy | 45.00% | 43.98% | 34 |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L8 | Macro accuracy | 42.04% | 44.00% | 35 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | Macro accuracy | 67.92% | 72.81% | 84 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | Macro accuracy | 75.34% | 78.65% | 141 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | Macro accuracy | 75.40% | 81.36% | 114 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | Macro accuracy | 76.30% | 82.80% | 87 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | Macro accuracy | 88.40% | 86.00% | 58 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | Macro accuracy | 86.27% | 87.60% | 62 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | Macro accuracy | 85.60% | 87.20% | 46 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | Macro accuracy | 86.67% | 87.07% | 56 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | Macro accuracy | 97.50% | 100.00% | 23 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | Macro accuracy | 99.17% | 99.17% | 10 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | Macro accuracy | 99.17% | 100.00% | 12 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | Macro accuracy | 99.17% | 100.00% | 19 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | Macro accuracy | 74.00% | 73.36% | 29 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | Macro accuracy | 72.57% | 72.14% | 32 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | Macro accuracy | 74.93% | 74.93% | 39 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | Macro accuracy | 73.57% | 74.93% | 37 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | Mean distance error (m) | 3.332 m | 3.287 m | 61 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | Mean distance error (m) | 3.003 m | 3.122 m | 71 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | Mean distance error (m) | 2.877 m | 3.037 m | 53 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | Mean distance error (m) | 2.759 m | 2.874 m | 67 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | Macro accuracy | 44.46% | 44.48% | 32 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | Macro accuracy | 43.20% | 43.98% | 45 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | Macro accuracy | 43.01% | 42.99% | 42 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | Macro accuracy | 42.52% | 41.90% | 43 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L2 | Macro accuracy | 73.02% | 74.67% | 73 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L4 | Macro accuracy | 77.52% | 78.50% | 89 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L6 | Macro accuracy | 75.76% | 81.00% | 60 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L8 | Macro accuracy | 81.33% | 80.28% | 163 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L2 | Macro accuracy | 87.20% | 86.67% | 109 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L4 | Macro accuracy | 87.33% | 88.13% | 38 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L6 | Macro accuracy | 88.13% | 88.80% | 71 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L8 | Macro accuracy | 87.47% | 86.53% | 40 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L2 | Macro accuracy | 97.50% | 99.17% | 46 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L4 | Macro accuracy | 97.50% | 99.17% | 19 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L6 | Macro accuracy | 99.17% | 99.17% | 13 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L8 | Macro accuracy | 99.17% | 99.17% | 23 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L2 | Macro accuracy | 73.36% | 72.64% | 37 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L4 | Macro accuracy | 74.36% | 73.14% | 42 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L6 | Macro accuracy | 70.64% | 72.21% | 35 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L8 | Macro accuracy | 72.86% | 72.57% | 21 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L2 | Mean distance error (m) | 3.345 m | 3.332 m | 50 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L4 | Mean distance error (m) | 2.952 m | 2.923 m | 83 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L6 | Mean distance error (m) | 2.893 m | 2.900 m | 81 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L8 | Mean distance error (m) | 2.776 m | 2.833 m | 59 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L2 | Macro accuracy | 44.41% | 45.14% | 71 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L4 | Macro accuracy | 43.36% | 42.66% | 69 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L6 | Macro accuracy | 42.26% | 42.47% | 51 |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L8 | Macro accuracy | 40.92% | 40.10% | 27 |

## Detailed validation-best exits

The manuscript-facing exit is selected from validation only. For positioning,
lower mean distance error is better; for classification, higher macro
accuracy is better.

| Method | Task | Policy | Exit | Test value |
| --- | --- | --- | --- | --- |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | best validation | L8 | 78.97% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | best validation | L6 | 87.60% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | best validation | L8 | 96.67% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | best validation | L4 | 71.57% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | best validation | L8 | 2.939 m |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | best validation | L6 | 41.46% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | best validation | L4 | 82.56% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | best validation | L8 | 86.67% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | best validation | L4 | 96.67% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | best validation | L4 | 74.71% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | best validation | L8 | 2.694 m |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | best validation | L2 | 49.27% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | best validation | L8 | 76.30% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | best validation | L4 | 86.27% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | best validation | L2 | 97.50% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | best validation | L6 | 74.93% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | best validation | L8 | 2.759 m |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | best validation | L2 | 44.46% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | best validation | L6 | 75.76% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | best validation | L6 | 88.13% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | best validation | L2 | 97.50% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | best validation | L4 | 74.36% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | best validation | L8 | 2.776 m |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | best validation | L2 | 44.41% |

## Detailed batch-1 inference cost

Latency is the synchronized CUDA mean/p95 measurement after warm-up on the
RTX 5070 Ti. Savings are relative to the same method/task at L8. A missing P2-Reverse
benchmark is reported as unavailable, not estimated.

| Method | Task | Exit | FLOPs (B) | Mean ms | P95 ms | FLOPs saving | Latency saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.5118 | 0.5423 | 74.3% | 59.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.7627 | 0.7963 | 49.5% | 40.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | 2.111 | 1.0237 | 1.1079 | 24.8% | 19.7% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.2750 | 1.4085 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | 0.721 | 0.4787 | 0.5254 | 74.3% | 58.7% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | 1.416 | 0.7003 | 0.7470 | 49.5% | 39.5% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.9318 | 1.0112 | 24.8% | 19.5% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 2.807 | 1.1578 | 1.2586 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4697 | 0.6069 | 74.4% | 61.4% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | 5.193 | 0.7255 | 0.9150 | 49.6% | 40.4% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | 7.750 | 0.9743 | 1.2232 | 24.8% | 20.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | 10.308 | 1.2174 | 1.3959 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4762 | 0.5561 | 74.3% | 58.8% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | 1.416 | 0.7035 | 0.8086 | 49.5% | 39.1% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | 2.111 | 0.9384 | 1.0155 | 24.8% | 18.8% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | 2.807 | 1.1557 | 1.2556 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | 3.828 | 0.5717 | 0.7766 | 74.5% | 63.3% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | 7.553 | 0.9008 | 1.0954 | 49.7% | 42.1% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2303 | 1.4191 | 24.8% | 20.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5557 | 1.7371 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.6012 | 0.7444 | 74.2% | 52.1% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | 0.052 | 0.8241 | 0.8939 | 49.5% | 34.4% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | 0.077 | 1.0366 | 1.1626 | 24.7% | 17.4% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 0.102 | 1.2556 | 1.3944 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L2 | 0.721 | 0.4509 | 0.4844 | 74.3% | 58.9% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L4 | 1.416 | 0.6742 | 0.6855 | 49.5% | 38.6% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L6 | 2.111 | 0.8783 | 0.9020 | 24.8% | 20.0% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L8 | 2.807 | 1.0981 | 1.1954 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L2 | 0.721 | 0.4403 | 0.4629 | 74.3% | 59.7% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L4 | 1.416 | 0.6550 | 0.6869 | 49.5% | 40.1% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L6 | 2.111 | 0.8768 | 0.9461 | 24.8% | 19.8% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L8 | 2.807 | 1.0937 | 1.1360 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L2 | 2.635 | 0.4563 | 0.4555 | 74.4% | 61.3% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L4 | 5.193 | 0.6922 | 0.6885 | 49.6% | 41.3% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L6 | 7.750 | 0.9386 | 0.9344 | 24.8% | 20.5% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L8 | 10.308 | 1.1800 | 1.1767 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L2 | 0.721 | 0.4509 | 0.4593 | 74.3% | 58.9% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L4 | 1.416 | 0.6644 | 0.6785 | 49.5% | 39.5% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L6 | 2.111 | 0.8790 | 0.9048 | 24.8% | 20.0% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L8 | 2.807 | 1.0981 | 1.1746 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L2 | 3.828 | 0.5431 | 0.5460 | 74.5% | 64.6% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L4 | 7.553 | 0.8711 | 0.8734 | 49.7% | 43.2% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L6 | 11.278 | 1.1942 | 1.1946 | 24.8% | 22.1% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L8 | 15.003 | 1.5327 | 1.5290 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | 0.026 | 0.5739 | 0.5849 | 74.2% | 52.2% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L4 | 0.052 | 0.7966 | 0.8066 | 49.5% | 33.7% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L6 | 0.077 | 0.9974 | 1.0167 | 24.7% | 17.0% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L8 | 0.102 | 1.2012 | 1.2282 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.4542 | 0.4795 | 74.3% | 63.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 0.6617 | 0.7098 | 49.5% | 47.3% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 0.9112 | 0.9645 | 24.8% | 27.4% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.2549 | 1.9018 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.5394 | 1.2924 | 74.3% | 51.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | 1.416 | 0.7819 | 1.3971 | 49.5% | 29.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | 2.111 | 0.9049 | 0.9886 | 24.8% | 18.7% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 2.807 | 1.1135 | 1.1969 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.4440 | 0.4584 | 74.4% | 62.7% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 5.193 | 0.8331 | 1.6843 | 49.6% | 30.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | 7.750 | 1.1514 | 1.9251 | 24.8% | 3.3% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | 10.308 | 1.1904 | 1.1970 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.4551 | 0.4799 | 74.3% | 61.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | 1.416 | 0.7229 | 0.7344 | 49.5% | 39.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | 2.111 | 0.9504 | 0.9825 | 24.8% | 19.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | 2.807 | 1.1845 | 1.2172 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | 3.828 | 0.5491 | 0.5564 | 74.5% | 64.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | 7.553 | 0.8700 | 0.8815 | 49.7% | 44.2% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.1954 | 1.2354 | 24.8% | 23.4% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.5602 | 2.1005 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.5712 | 0.6024 | 74.2% | 53.4% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 0.8007 | 0.8962 | 49.5% | 34.7% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | 0.077 | 0.9932 | 1.0634 | 24.7% | 19.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | 0.102 | 1.2269 | 1.2999 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L2 | 0.721 | 0.8645 | 0.8956 | 74.3% | 58.2% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L4 | 1.416 | 1.2882 | 1.3171 | 49.5% | 37.6% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L6 | 2.111 | 1.7043 | 1.7146 | 24.8% | 17.5% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L8 | 2.807 | 2.0660 | 2.0891 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L2 | 0.721 | 0.8860 | 0.9066 | 74.3% | 57.4% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L4 | 1.416 | 1.2963 | 1.3242 | 49.5% | 37.6% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L6 | 2.111 | 1.6786 | 1.7061 | 24.8% | 19.2% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L8 | 2.807 | 2.0781 | 2.0970 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L2 | 2.635 | 0.8883 | 0.9066 | 74.4% | 57.3% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L4 | 5.193 | 1.2911 | 1.3209 | 49.6% | 37.9% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L6 | 7.750 | 1.6746 | 1.7057 | 24.8% | 19.5% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L8 | 10.308 | 2.0793 | 2.0879 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L2 | 0.721 | 0.8834 | 0.9037 | 74.3% | 57.1% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L4 | 1.416 | 1.2959 | 1.3233 | 49.5% | 37.1% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L6 | 2.111 | 1.6780 | 1.6986 | 24.8% | 18.6% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L8 | 2.807 | 2.0605 | 2.0825 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L2 | 3.828 | 0.8767 | 0.8960 | 74.5% | 56.9% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L4 | 7.553 | 1.2775 | 1.3060 | 49.7% | 37.2% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L6 | 11.278 | 1.6512 | 1.6749 | 24.8% | 18.8% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L8 | 15.003 | 2.0336 | 2.0608 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L2 | 0.026 | 1.3808 | 1.4593 | 74.2% | 46.1% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L4 | 0.052 | 1.7808 | 1.8083 | 49.5% | 30.5% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L6 | 0.077 | 2.1777 | 2.2041 | 24.7% | 15.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L8 | 0.102 | 2.5630 | 2.5923 | 0.0% | 0.0% |

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
