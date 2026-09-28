# Time-masking comparison

This directory contains the submission-oriented result tables and figures for
the completed single-seed pilot. It is an aggregation of recorded
artifacts, not a statistical claim. The report was generated from repository
revision `50563739260ab3448f1b45a9a054d23e839a351c`.

## Scope and comparability

| Approach | Pretraining | Downstream representation | Role |
| --- | --- | --- | --- |
| Original LatentWave | Original LatentWave training | Frozen context encoder, full-input L8 | Full-depth baseline |
| LatentWave with Post-hoc Frozen Early Exits | Reuses the Original LatentWave checkpoint | Frozen context encoder at L2/L4/L6/L8 | Early-exit baseline |
| Same-scale Multi-depth JEPA | P1 pretraining with same-scale intermediate targets | Frozen context encoder at L2/L4/L6/L8 | Same-scale auxiliary-loss control |
| Proposed Hierarchical Multi-Scale JEPA | Separately pretrained hierarchical JEPA | Frozen context encoder at L2/L4/L6/L8 | Proposed method |

All rows use independent frozen task heads, the same persisted 80/10/10
split, and validation-only head/exit selection. Original LatentWave is the
P0 context encoder evaluated at L8; the post-hoc early-exit baseline reuses
that exact P0 checkpoint at L2/L4/L6/L8. Therefore the included approaches are
directly comparable for this single-seed packet.

All values are single-seed pilot point estimates.

## Run-specific provenance

P1 same-scale multi-depth JEPA uses the selected validation checkpoint from the seed-0 time-masking run; validation-selected heads were evaluated once on held-out test splits. All approaches use the same time-masking pretraining regime and six-task suite.


## Task-suite note

## P1 checkpoint provenance

Same-scale Multi-depth JEPA checkpoint: `/home/waves-lab/LatentWave_Data/preprocessed_datasets/MultiScaleJEPA-main/runs/masking_study/time_n6/p1_same_scale_time_p2matched/seed_000/checkpoints/last.ckpt`.

Pretraining experiment metadata: `/home/waves-lab/LatentWave_Data/preprocessed_datasets/MultiScaleJEPA-main/runs/masking_study/time_n6/p1_same_scale_time_p2matched/seed_000/experiment.json`.

This packet covers **Time masking (N_m=6)** and the following downstream task suite:
RF sensing, Jamming classification, Human activity sensing, Interference classification, NR positioning, DeepMIMO beam. Jamming and interference
are labelled frozen-head tasks only; none of the six downstream caches is used
for pretraining.

## All-approach headline downstream comparison

The deployment rows use full-depth L8 Original LatentWave and the
earliest exit within 5% of validation-best for the exit approaches.

| Approach | Task | Exit | Selection | Test value |
| --- | --- | --- | --- | --- |
| Original LatentWave | RF sensing | L8 | full-depth L8 | 72.76% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | earliest within 5% of validation best | 68.27% |
| Same-scale Multi-depth JEPA | RF sensing | L6 | earliest within 5% of validation best | 69.07% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | earliest within 5% of validation best | 74.48% |
| Original LatentWave | Jamming classification | L8 | full-depth L8 | 86.80% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | earliest within 5% of validation best | 89.07% |
| Same-scale Multi-depth JEPA | Jamming classification | L2 | earliest within 5% of validation best | 88.80% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | earliest within 5% of validation best | 86.93% |
| Original LatentWave | Human activity sensing | L8 | full-depth L8 | 99.17% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | earliest within 5% of validation best | 100.00% |
| Same-scale Multi-depth JEPA | Human activity sensing | L2 | earliest within 5% of validation best | 99.17% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | earliest within 5% of validation best | 98.33% |
| Original LatentWave | Interference classification | L8 | full-depth L8 | 70.14% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | earliest within 5% of validation best | 73.64% |
| Same-scale Multi-depth JEPA | Interference classification | L2 | earliest within 5% of validation best | 72.93% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | earliest within 5% of validation best | 74.64% |
| Original LatentWave | NR positioning | L8 | full-depth L8 | 3.318 m |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | earliest within 5% of validation best | 3.408 m |
| Same-scale Multi-depth JEPA | NR positioning | L4 | earliest within 5% of validation best | 3.349 m |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | earliest within 5% of validation best | 3.032 m |
| Original LatentWave | DeepMIMO beam | L8 | full-depth L8 | 35.24% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | earliest within 5% of validation best | 41.71% |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L4 | earliest within 5% of validation best | 41.98% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | earliest within 5% of validation best | 42.62% |

## All-approach headline inference comparison

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| Original LatentWave | RF sensing | L8 | 2.807 | 1.1221 | 1.2569 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.4529 | 0.6066 |
| Same-scale Multi-depth JEPA | RF sensing | L6 | 2.111 | 0.9441 | 1.0313 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 1.1986 | 2.7153 |
| Original LatentWave | Jamming classification | L8 | 2.807 | 1.1183 | 1.2613 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | 0.721 | 0.4636 | 0.5966 |
| Same-scale Multi-depth JEPA | Jamming classification | L2 | 0.721 | 0.4751 | 0.4966 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.9171 | 2.5556 |
| Original LatentWave | Human activity sensing | L8 | 10.308 | 1.2337 | 1.4207 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4702 | 0.7052 |
| Same-scale Multi-depth JEPA | Human activity sensing | L2 | 2.635 | 0.4663 | 0.6237 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.9278 | 2.6623 |
| Original LatentWave | Interference classification | L8 | 2.807 | 1.1167 | 1.3175 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4514 | 0.5666 |
| Same-scale Multi-depth JEPA | Interference classification | L2 | 0.721 | 0.4736 | 0.5000 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.9485 | 2.5553 |
| Original LatentWave | NR positioning | L8 | 15.003 | 1.5814 | 1.8182 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | 7.553 | 0.9480 | 1.2454 |
| Same-scale Multi-depth JEPA | NR positioning | L4 | 7.553 | 0.8730 | 1.0426 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 2.1171 | 3.6676 |
| Original LatentWave | DeepMIMO beam | L8 | 0.102 | 1.1969 | 1.3786 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.5901 | 0.7504 |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L4 | 0.052 | 0.8417 | 0.9653 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 1.1577 | 4.7483 |

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
| Same-scale Multi-depth JEPA | RF sensing | L2 | Macro accuracy | 67.31% | 69.03% | 60 |
| Same-scale Multi-depth JEPA | RF sensing | L4 | Macro accuracy | 77.15% | 70.04% | 68 |
| Same-scale Multi-depth JEPA | RF sensing | L6 | Macro accuracy | 69.07% | 75.75% | 100 |
| Same-scale Multi-depth JEPA | RF sensing | L8 | Macro accuracy | 69.97% | 72.90% | 71 |
| Same-scale Multi-depth JEPA | Jamming classification | L2 | Macro accuracy | 88.80% | 89.87% | 84 |
| Same-scale Multi-depth JEPA | Jamming classification | L4 | Macro accuracy | 88.53% | 89.73% | 79 |
| Same-scale Multi-depth JEPA | Jamming classification | L6 | Macro accuracy | 86.00% | 87.33% | 27 |
| Same-scale Multi-depth JEPA | Jamming classification | L8 | Macro accuracy | 84.40% | 87.07% | 60 |
| Same-scale Multi-depth JEPA | Human activity sensing | L2 | Macro accuracy | 99.17% | 100.00% | 16 |
| Same-scale Multi-depth JEPA | Human activity sensing | L4 | Macro accuracy | 99.17% | 100.00% | 9 |
| Same-scale Multi-depth JEPA | Human activity sensing | L6 | Macro accuracy | 100.00% | 99.17% | 6 |
| Same-scale Multi-depth JEPA | Human activity sensing | L8 | Macro accuracy | 100.00% | 99.17% | 16 |
| Same-scale Multi-depth JEPA | Interference classification | L2 | Macro accuracy | 72.93% | 72.36% | 18 |
| Same-scale Multi-depth JEPA | Interference classification | L4 | Macro accuracy | 72.64% | 75.50% | 47 |
| Same-scale Multi-depth JEPA | Interference classification | L6 | Macro accuracy | 73.00% | 71.14% | 53 |
| Same-scale Multi-depth JEPA | Interference classification | L8 | Macro accuracy | 72.50% | 72.14% | 34 |
| Same-scale Multi-depth JEPA | NR positioning | L2 | Mean distance error (m) | 3.552 m | 3.507 m | 61 |
| Same-scale Multi-depth JEPA | NR positioning | L4 | Mean distance error (m) | 3.349 m | 3.368 m | 49 |
| Same-scale Multi-depth JEPA | NR positioning | L6 | Mean distance error (m) | 3.337 m | 3.316 m | 58 |
| Same-scale Multi-depth JEPA | NR positioning | L8 | Mean distance error (m) | 3.366 m | 3.313 m | 67 |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | Macro accuracy | 38.13% | 39.69% | 25 |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L4 | Macro accuracy | 41.98% | 43.24% | 74 |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L6 | Macro accuracy | 40.66% | 41.36% | 21 |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L8 | Macro accuracy | 39.93% | 42.03% | 111 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | Macro accuracy | 69.87% | 66.89% | 27 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | Macro accuracy | 74.48% | 79.14% | 101 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | Macro accuracy | 71.50% | 75.35% | 62 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | Macro accuracy | 73.44% | 74.44% | 118 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | Macro accuracy | 86.93% | 86.67% | 71 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | Macro accuracy | 87.87% | 88.40% | 22 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | Macro accuracy | 86.93% | 87.33% | 44 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | Macro accuracy | 86.27% | 84.53% | 65 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | Macro accuracy | 98.33% | 100.00% | 14 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | Macro accuracy | 100.00% | 100.00% | 16 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | Macro accuracy | 98.33% | 100.00% | 33 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | Macro accuracy | 97.50% | 100.00% | 32 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | Macro accuracy | 74.64% | 73.07% | 70 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | Macro accuracy | 75.07% | 76.07% | 27 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | Macro accuracy | 76.21% | 76.14% | 65 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | Macro accuracy | 73.50% | 74.36% | 31 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | Mean distance error (m) | 3.547 m | 3.515 m | 61 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | Mean distance error (m) | 3.245 m | 3.285 m | 74 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | Mean distance error (m) | 3.032 m | 3.088 m | 59 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | Mean distance error (m) | 3.099 m | 3.120 m | 61 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | Macro accuracy | 42.62% | 45.86% | 79 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | Macro accuracy | 40.40% | 44.24% | 20 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | Macro accuracy | 43.99% | 44.39% | 21 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | Macro accuracy | 41.71% | 44.02% | 93 |

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
| Same-scale Multi-depth JEPA | RF sensing | best validation | L6 | 69.07% |
| Same-scale Multi-depth JEPA | RF sensing | earliest within tolerance | L6 | 69.07% |
| Same-scale Multi-depth JEPA | Jamming classification | best validation | L2 | 88.80% |
| Same-scale Multi-depth JEPA | Jamming classification | earliest within tolerance | L2 | 88.80% |
| Same-scale Multi-depth JEPA | Human activity sensing | best validation | L2 | 99.17% |
| Same-scale Multi-depth JEPA | Human activity sensing | earliest within tolerance | L2 | 99.17% |
| Same-scale Multi-depth JEPA | Interference classification | best validation | L4 | 72.64% |
| Same-scale Multi-depth JEPA | Interference classification | earliest within tolerance | L2 | 72.93% |
| Same-scale Multi-depth JEPA | NR positioning | best validation | L8 | 3.366 m |
| Same-scale Multi-depth JEPA | NR positioning | earliest within tolerance | L4 | 3.349 m |
| Same-scale Multi-depth JEPA | DeepMIMO beam | best validation | L4 | 41.98% |
| Same-scale Multi-depth JEPA | DeepMIMO beam | earliest within tolerance | L4 | 41.98% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | best validation | L4 | 74.48% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | earliest within tolerance | L4 | 74.48% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | best validation | L4 | 87.87% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | earliest within tolerance | L2 | 86.93% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | best validation | L2 | 98.33% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | earliest within tolerance | L2 | 98.33% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | best validation | L6 | 76.21% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | earliest within tolerance | L2 | 74.64% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | best validation | L6 | 3.032 m |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | earliest within tolerance | L6 | 3.032 m |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | best validation | L2 | 42.62% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | earliest within tolerance | L2 | 42.62% |

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
| Same-scale Multi-depth JEPA | RF sensing | L2 | 0.721 | 0.5063 | 0.5239 | 74.3% | 56.5% |
| Same-scale Multi-depth JEPA | RF sensing | L4 | 1.416 | 0.7739 | 0.8237 | 49.5% | 33.5% |
| Same-scale Multi-depth JEPA | RF sensing | L6 | 2.111 | 0.9441 | 1.0313 | 24.8% | 18.9% |
| Same-scale Multi-depth JEPA | RF sensing | L8 | 2.807 | 1.1638 | 1.2671 | 0.0% | 0.0% |
| Same-scale Multi-depth JEPA | Jamming classification | L2 | 0.721 | 0.4751 | 0.4966 | 74.3% | 59.8% |
| Same-scale Multi-depth JEPA | Jamming classification | L4 | 1.416 | 0.7149 | 0.7781 | 49.5% | 39.5% |
| Same-scale Multi-depth JEPA | Jamming classification | L6 | 2.111 | 0.9386 | 1.0185 | 24.8% | 20.5% |
| Same-scale Multi-depth JEPA | Jamming classification | L8 | 2.807 | 1.1811 | 1.2827 | 0.0% | 0.0% |
| Same-scale Multi-depth JEPA | Human activity sensing | L2 | 2.635 | 0.4663 | 0.6237 | 74.4% | 61.0% |
| Same-scale Multi-depth JEPA | Human activity sensing | L4 | 5.193 | 0.7069 | 0.8740 | 49.6% | 40.9% |
| Same-scale Multi-depth JEPA | Human activity sensing | L6 | 7.750 | 0.9575 | 1.2173 | 24.8% | 19.9% |
| Same-scale Multi-depth JEPA | Human activity sensing | L8 | 10.308 | 1.1958 | 1.4901 | 0.0% | 0.0% |
| Same-scale Multi-depth JEPA | Interference classification | L2 | 0.721 | 0.4736 | 0.5000 | 74.3% | 59.5% |
| Same-scale Multi-depth JEPA | Interference classification | L4 | 1.416 | 0.7236 | 0.8257 | 49.5% | 38.1% |
| Same-scale Multi-depth JEPA | Interference classification | L6 | 2.111 | 0.9613 | 1.1099 | 24.8% | 17.8% |
| Same-scale Multi-depth JEPA | Interference classification | L8 | 2.807 | 1.1693 | 1.2758 | 0.0% | 0.0% |
| Same-scale Multi-depth JEPA | NR positioning | L2 | 3.828 | 0.5499 | 0.6950 | 74.5% | 64.0% |
| Same-scale Multi-depth JEPA | NR positioning | L4 | 7.553 | 0.8730 | 1.0426 | 49.7% | 42.9% |
| Same-scale Multi-depth JEPA | NR positioning | L6 | 11.278 | 1.2014 | 1.5147 | 24.8% | 21.4% |
| Same-scale Multi-depth JEPA | NR positioning | L8 | 15.003 | 1.5291 | 1.8344 | 0.0% | 0.0% |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | 0.026 | 0.6172 | 0.7605 | 74.2% | 51.5% |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L4 | 0.052 | 0.8417 | 0.9653 | 49.5% | 33.8% |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L6 | 0.077 | 1.0553 | 1.1988 | 24.7% | 17.0% |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L8 | 0.102 | 1.2715 | 1.4212 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.8352 | 2.5321 | 74.3% | 52.4% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 1.1986 | 2.7153 | 49.5% | 31.6% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 1.4615 | 2.7603 | 24.8% | 16.6% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.7530 | 3.0962 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.9171 | 2.5556 | 74.3% | 47.6% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | 1.416 | 1.2106 | 2.7608 | 49.5% | 30.8% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | 2.111 | 1.4744 | 2.8694 | 24.8% | 15.7% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 2.807 | 1.7492 | 3.0914 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.9278 | 2.6623 | 74.4% | 55.3% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 5.193 | 1.2912 | 2.9072 | 49.6% | 37.8% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | 7.750 | 1.6255 | 3.3916 | 24.8% | 21.6% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | 10.308 | 2.0746 | 3.6463 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.9485 | 2.5553 | 74.3% | 45.8% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | 1.416 | 1.2102 | 2.6611 | 49.5% | 30.8% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | 2.111 | 1.4776 | 2.9724 | 24.8% | 15.5% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | 2.807 | 1.7494 | 3.0894 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | 3.828 | 0.9660 | 2.7893 | 74.5% | 61.4% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | 7.553 | 1.5646 | 3.1305 | 49.7% | 37.4% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 2.1171 | 3.6676 | 24.8% | 15.3% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 2.5004 | 4.0001 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 1.1577 | 4.7483 | 74.2% | 48.1% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 1.5087 | 4.8161 | 49.5% | 32.3% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | 0.077 | 1.8778 | 4.9617 | 24.7% | 15.8% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | 0.102 | 2.2290 | 5.2559 | 0.0% | 0.0% |

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
