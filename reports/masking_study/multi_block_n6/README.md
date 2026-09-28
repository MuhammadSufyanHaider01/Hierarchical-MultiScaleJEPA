# Multi-block masking comparison

This directory contains the submission-oriented result tables and figures for
the completed single-seed pilot. It is an aggregation of recorded
artifacts, not a statistical claim. The report was generated from repository
revision `869f22b8bf3e5ee416392b8a2dcf0d1c031f2da1`.

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

The M-block and frequency_n6 profiles resolve to identical multi-block sampler settings (span_time, six prediction masks, and all remaining geometry parameters); all six persisted downstream train/validation/test index arrays were verified identical. Therefore the frequency_n6 P1 seed-0 checkpoint, selected heads, one-time held-out results, and synchronized batch-1 benchmark are reused here instead of redundantly repeating the same P1 condition. The historical M-block P2 run used FP32 with microbatch 4 and accumulation 16; the reused P1 run used BF16 mixed precision with microbatch 16 and accumulation 4 (effective batch 64), so this training-profile difference is a comparability limitation.


## Task-suite note

## P1 checkpoint provenance

Same-scale Multi-depth JEPA checkpoint: `/home/waves-lab/LatentWave_Data/preprocessed_datasets/MultiScaleJEPA-main/runs/masking_study/frequency_n6/p1_same_scale_frequency_p2matched/seed_000/checkpoints/last.ckpt`.

Pretraining experiment metadata: `/home/waves-lab/LatentWave_Data/preprocessed_datasets/MultiScaleJEPA-main/runs/masking_study/frequency_n6/p1_same_scale_frequency_p2matched/seed_000/experiment.json`.

This packet covers **Multi-block span-time masking (6 prediction masks)** and the following downstream task suite:
RF sensing, Jamming classification, Human activity sensing, Interference classification, NR positioning, DeepMIMO beam. Jamming and interference
are labelled frozen-head tasks only; none of the six downstream caches is used
for pretraining.

## All-approach headline downstream comparison

The deployment rows use full-depth L8 Original LatentWave and the
earliest exit within 5% of validation-best for the exit approaches.

| Approach | Task | Exit | Selection | Test value |
| --- | --- | --- | --- | --- |
| Original LatentWave | RF sensing | L8 | full-depth L8 | 69.46% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | earliest within 5% of validation best | 63.30% |
| Same-scale Multi-depth JEPA | RF sensing | L2 | earliest within 5% of validation best | 65.53% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | earliest within 5% of validation best | 74.09% |
| Original LatentWave | Jamming classification | L8 | full-depth L8 | 88.00% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | earliest within 5% of validation best | 88.00% |
| Same-scale Multi-depth JEPA | Jamming classification | L4 | earliest within 5% of validation best | 86.00% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | earliest within 5% of validation best | 86.53% |
| Original LatentWave | Human activity sensing | L8 | full-depth L8 | 98.33% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | earliest within 5% of validation best | 95.83% |
| Same-scale Multi-depth JEPA | Human activity sensing | L2 | earliest within 5% of validation best | 97.50% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | earliest within 5% of validation best | 100.00% |
| Original LatentWave | Interference classification | L8 | full-depth L8 | 74.79% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | earliest within 5% of validation best | 71.64% |
| Same-scale Multi-depth JEPA | Interference classification | L2 | earliest within 5% of validation best | 73.29% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | earliest within 5% of validation best | 75.29% |
| Original LatentWave | NR positioning | L8 | full-depth L8 | 3.025 m |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | earliest within 5% of validation best | 3.104 m |
| Same-scale Multi-depth JEPA | NR positioning | L6 | earliest within 5% of validation best | 2.993 m |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | earliest within 5% of validation best | 2.760 m |
| Original LatentWave | DeepMIMO beam | L8 | full-depth L8 | 45.74% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | earliest within 5% of validation best | 42.72% |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | earliest within 5% of validation best | 43.26% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | earliest within 5% of validation best | 43.47% |

## All-approach headline inference comparison

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| Original LatentWave | RF sensing | L8 | 2.807 | 1.1175 | 1.2204 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.6789 | 0.7893 |
| Same-scale Multi-depth JEPA | RF sensing | L2 | 0.721 | 0.4453 | 0.4697 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 0.9098 | 0.9708 |
| Original LatentWave | Jamming classification | L8 | 2.807 | 1.1267 | 1.2456 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 2.807 | 1.1267 | 1.2456 |
| Same-scale Multi-depth JEPA | Jamming classification | L4 | 1.416 | 0.6757 | 0.7234 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | 2.111 | 0.9102 | 1.0096 |
| Original LatentWave | Human activity sensing | L8 | 10.308 | 1.1870 | 1.3541 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4521 | 0.5502 |
| Same-scale Multi-depth JEPA | Human activity sensing | L2 | 2.635 | 0.4445 | 0.4498 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.4588 | 0.5644 |
| Original LatentWave | Interference classification | L8 | 2.807 | 1.1186 | 1.2273 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4463 | 0.4776 |
| Same-scale Multi-depth JEPA | Interference classification | L2 | 0.721 | 0.4480 | 0.4726 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.4588 | 0.5002 |
| Original LatentWave | NR positioning | L8 | 15.003 | 1.5272 | 1.6933 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2051 | 1.3760 |
| Same-scale Multi-depth JEPA | NR positioning | L6 | 11.278 | 1.1907 | 1.4632 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.2052 | 1.3663 |
| Original LatentWave | DeepMIMO beam | L8 | 0.102 | 1.2205 | 1.3524 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.5806 | 0.6564 |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | 0.026 | 0.6164 | 0.6534 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 0.8035 | 0.9121 |

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
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | Macro accuracy | 60.87% | 65.01% | 110 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | Macro accuracy | 63.30% | 71.70% | 95 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | Macro accuracy | 66.98% | 70.50% | 75 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | Macro accuracy | 69.46% | 71.08% | 163 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | Macro accuracy | 81.07% | 79.87% | 144 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | Macro accuracy | 80.53% | 78.80% | 74 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | Macro accuracy | 81.87% | 83.20% | 86 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | Macro accuracy | 88.00% | 87.73% | 21 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | Macro accuracy | 95.83% | 99.17% | 24 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | Macro accuracy | 96.67% | 100.00% | 24 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | Macro accuracy | 100.00% | 100.00% | 35 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | Macro accuracy | 98.33% | 100.00% | 26 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | Macro accuracy | 71.64% | 72.43% | 40 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | Macro accuracy | 74.07% | 73.50% | 43 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | Macro accuracy | 75.21% | 73.86% | 32 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | Macro accuracy | 74.79% | 73.79% | 34 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | Mean distance error (m) | 3.734 m | 3.644 m | 17 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | Mean distance error (m) | 3.307 m | 3.335 m | 35 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | Mean distance error (m) | 3.104 m | 3.108 m | 93 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | Mean distance error (m) | 3.025 m | 3.039 m | 59 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | Macro accuracy | 42.72% | 43.97% | 42 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | Macro accuracy | 42.14% | 43.54% | 17 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | Macro accuracy | 43.01% | 43.82% | 24 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | Macro accuracy | 45.74% | 44.69% | 54 |
| Same-scale Multi-depth JEPA | RF sensing | L2 | Macro accuracy | 65.53% | 69.23% | 79 |
| Same-scale Multi-depth JEPA | RF sensing | L4 | Macro accuracy | 68.16% | 69.47% | 78 |
| Same-scale Multi-depth JEPA | RF sensing | L6 | Macro accuracy | 71.63% | 69.19% | 67 |
| Same-scale Multi-depth JEPA | RF sensing | L8 | Macro accuracy | 72.00% | 70.47% | 101 |
| Same-scale Multi-depth JEPA | Jamming classification | L2 | Macro accuracy | 85.60% | 83.73% | 67 |
| Same-scale Multi-depth JEPA | Jamming classification | L4 | Macro accuracy | 86.00% | 86.67% | 39 |
| Same-scale Multi-depth JEPA | Jamming classification | L6 | Macro accuracy | 87.87% | 88.00% | 70 |
| Same-scale Multi-depth JEPA | Jamming classification | L8 | Macro accuracy | 85.33% | 88.40% | 26 |
| Same-scale Multi-depth JEPA | Human activity sensing | L2 | Macro accuracy | 97.50% | 99.17% | 21 |
| Same-scale Multi-depth JEPA | Human activity sensing | L4 | Macro accuracy | 97.50% | 98.33% | 16 |
| Same-scale Multi-depth JEPA | Human activity sensing | L6 | Macro accuracy | 98.33% | 99.17% | 20 |
| Same-scale Multi-depth JEPA | Human activity sensing | L8 | Macro accuracy | 98.33% | 99.17% | 15 |
| Same-scale Multi-depth JEPA | Interference classification | L2 | Macro accuracy | 73.29% | 72.79% | 24 |
| Same-scale Multi-depth JEPA | Interference classification | L4 | Macro accuracy | 72.14% | 73.07% | 17 |
| Same-scale Multi-depth JEPA | Interference classification | L6 | Macro accuracy | 73.43% | 74.64% | 54 |
| Same-scale Multi-depth JEPA | Interference classification | L8 | Macro accuracy | 74.71% | 72.93% | 47 |
| Same-scale Multi-depth JEPA | NR positioning | L2 | Mean distance error (m) | 3.521 m | 3.485 m | 55 |
| Same-scale Multi-depth JEPA | NR positioning | L4 | Mean distance error (m) | 3.243 m | 3.289 m | 64 |
| Same-scale Multi-depth JEPA | NR positioning | L6 | Mean distance error (m) | 2.993 m | 3.053 m | 86 |
| Same-scale Multi-depth JEPA | NR positioning | L8 | Mean distance error (m) | 2.974 m | 2.974 m | 38 |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | Macro accuracy | 43.26% | 45.23% | 52 |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L4 | Macro accuracy | 41.66% | 45.12% | 25 |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L6 | Macro accuracy | 45.23% | 46.19% | 49 |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L8 | Macro accuracy | 44.28% | 46.39% | 68 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | Macro accuracy | 72.25% | 68.00% | 80 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | Macro accuracy | 65.53% | 68.05% | 50 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | Macro accuracy | 74.09% | 76.11% | 98 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | Macro accuracy | 72.26% | 76.49% | 78 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | Macro accuracy | 82.67% | 84.00% | 51 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | Macro accuracy | 84.00% | 84.13% | 52 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | Macro accuracy | 86.53% | 86.93% | 80 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | Macro accuracy | 88.27% | 89.60% | 47 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | Macro accuracy | 100.00% | 99.17% | 42 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | Macro accuracy | 100.00% | 100.00% | 19 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | Macro accuracy | 100.00% | 99.17% | 17 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | Macro accuracy | 100.00% | 100.00% | 26 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | Macro accuracy | 75.29% | 74.57% | 38 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | Macro accuracy | 75.29% | 75.29% | 25 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | Macro accuracy | 75.00% | 75.29% | 18 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | Macro accuracy | 75.86% | 75.57% | 23 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | Mean distance error (m) | 3.296 m | 3.284 m | 43 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | Mean distance error (m) | 3.101 m | 3.061 m | 89 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | Mean distance error (m) | 2.760 m | 2.759 m | 62 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | Mean distance error (m) | 2.725 m | 2.771 m | 52 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | Macro accuracy | 42.68% | 44.24% | 27 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | Macro accuracy | 43.47% | 46.86% | 32 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | Macro accuracy | 41.45% | 44.57% | 19 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | Macro accuracy | 44.45% | 45.82% | 28 |

## Detailed selected exits

"Earliest within 5%" is selected from validation only. For positioning,
lower mean distance error is better; for classification, higher macro
accuracy is better.

| Method | Task | Policy | Exit | Test value |
| --- | --- | --- | --- | --- |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | best validation | L4 | 63.30% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | earliest within tolerance | L4 | 63.30% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | best validation | L8 | 88.00% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | earliest within tolerance | L8 | 88.00% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | best validation | L4 | 96.67% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | earliest within tolerance | L2 | 95.83% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | best validation | L6 | 75.21% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | earliest within tolerance | L2 | 71.64% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | best validation | L8 | 3.025 m |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | earliest within tolerance | L6 | 3.104 m |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | best validation | L8 | 45.74% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | earliest within tolerance | L2 | 42.72% |
| Same-scale Multi-depth JEPA | RF sensing | best validation | L8 | 72.00% |
| Same-scale Multi-depth JEPA | RF sensing | earliest within tolerance | L2 | 65.53% |
| Same-scale Multi-depth JEPA | Jamming classification | best validation | L8 | 85.33% |
| Same-scale Multi-depth JEPA | Jamming classification | earliest within tolerance | L4 | 86.00% |
| Same-scale Multi-depth JEPA | Human activity sensing | best validation | L2 | 97.50% |
| Same-scale Multi-depth JEPA | Human activity sensing | earliest within tolerance | L2 | 97.50% |
| Same-scale Multi-depth JEPA | Interference classification | best validation | L6 | 73.43% |
| Same-scale Multi-depth JEPA | Interference classification | earliest within tolerance | L2 | 73.29% |
| Same-scale Multi-depth JEPA | NR positioning | best validation | L8 | 2.974 m |
| Same-scale Multi-depth JEPA | NR positioning | earliest within tolerance | L6 | 2.993 m |
| Same-scale Multi-depth JEPA | DeepMIMO beam | best validation | L8 | 44.28% |
| Same-scale Multi-depth JEPA | DeepMIMO beam | earliest within tolerance | L2 | 43.26% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | best validation | L8 | 72.26% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | earliest within tolerance | L6 | 74.09% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | best validation | L8 | 88.27% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | earliest within tolerance | L6 | 86.53% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | best validation | L4 | 100.00% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | earliest within tolerance | L2 | 100.00% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | best validation | L8 | 75.86% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | earliest within tolerance | L2 | 75.29% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | best validation | L6 | 2.760 m |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | earliest within tolerance | L6 | 2.760 m |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | best validation | L4 | 43.47% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | earliest within tolerance | L4 | 43.47% |

## Detailed batch-1 inference cost

Latency is the synchronized CUDA mean/p95 measurement after warm-up on the
RTX 5070 Ti. Savings are relative to the same method/task at L8.

| Method | Task | Exit | FLOPs (B) | Mean ms | P95 ms | FLOPs saving | Latency saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.4502 | 0.5238 | 74.3% | 59.7% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.6789 | 0.7893 | 49.5% | 39.2% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | 2.111 | 0.9145 | 0.9972 | 24.8% | 18.2% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.1175 | 1.2204 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | 0.721 | 0.4564 | 0.4830 | 74.3% | 59.5% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | 1.416 | 0.6853 | 0.7551 | 49.5% | 39.2% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.8979 | 0.9835 | 24.8% | 20.3% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 2.807 | 1.1267 | 1.2456 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4521 | 0.5502 | 74.4% | 61.9% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | 5.193 | 0.7012 | 0.8600 | 49.6% | 40.9% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | 7.750 | 0.9470 | 1.1033 | 24.8% | 20.2% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | 10.308 | 1.1870 | 1.3541 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4463 | 0.4776 | 74.3% | 60.1% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | 1.416 | 0.6719 | 0.7731 | 49.5% | 39.9% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | 2.111 | 0.8998 | 1.0251 | 24.8% | 19.6% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | 2.807 | 1.1186 | 1.2273 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | 3.828 | 0.5451 | 0.6899 | 74.5% | 64.3% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | 7.553 | 0.8722 | 1.0278 | 49.7% | 42.9% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2051 | 1.3760 | 24.8% | 21.1% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5272 | 1.6933 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.5806 | 0.6564 | 74.2% | 52.4% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | 0.052 | 0.7985 | 0.9120 | 49.5% | 34.6% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | 0.077 | 0.9947 | 1.0930 | 24.7% | 18.5% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 0.102 | 1.2205 | 1.3524 | 0.0% | 0.0% |
| Same-scale Multi-depth JEPA | RF sensing | L2 | 0.721 | 0.4453 | 0.4697 | 74.3% | 59.3% |
| Same-scale Multi-depth JEPA | RF sensing | L4 | 1.416 | 0.6706 | 0.7150 | 49.5% | 38.7% |
| Same-scale Multi-depth JEPA | RF sensing | L6 | 2.111 | 0.8748 | 0.9622 | 24.8% | 20.0% |
| Same-scale Multi-depth JEPA | RF sensing | L8 | 2.807 | 1.0941 | 1.2155 | 0.0% | 0.0% |
| Same-scale Multi-depth JEPA | Jamming classification | L2 | 0.721 | 0.4554 | 0.4805 | 74.3% | 58.8% |
| Same-scale Multi-depth JEPA | Jamming classification | L4 | 1.416 | 0.6757 | 0.7234 | 49.5% | 38.8% |
| Same-scale Multi-depth JEPA | Jamming classification | L6 | 2.111 | 0.8852 | 0.9649 | 24.8% | 19.9% |
| Same-scale Multi-depth JEPA | Jamming classification | L8 | 2.807 | 1.1047 | 1.1961 | 0.0% | 0.0% |
| Same-scale Multi-depth JEPA | Human activity sensing | L2 | 2.635 | 0.4445 | 0.4498 | 74.4% | 62.4% |
| Same-scale Multi-depth JEPA | Human activity sensing | L4 | 5.193 | 0.6872 | 0.7957 | 49.6% | 41.9% |
| Same-scale Multi-depth JEPA | Human activity sensing | L6 | 7.750 | 0.9297 | 1.1955 | 24.8% | 21.4% |
| Same-scale Multi-depth JEPA | Human activity sensing | L8 | 10.308 | 1.1826 | 1.4365 | 0.0% | 0.0% |
| Same-scale Multi-depth JEPA | Interference classification | L2 | 0.721 | 0.4480 | 0.4726 | 74.3% | 59.5% |
| Same-scale Multi-depth JEPA | Interference classification | L4 | 1.416 | 0.6692 | 0.7237 | 49.5% | 39.5% |
| Same-scale Multi-depth JEPA | Interference classification | L6 | 2.111 | 0.8828 | 0.9540 | 24.8% | 20.2% |
| Same-scale Multi-depth JEPA | Interference classification | L8 | 2.807 | 1.1057 | 1.1944 | 0.0% | 0.0% |
| Same-scale Multi-depth JEPA | NR positioning | L2 | 3.828 | 0.5411 | 0.5403 | 74.5% | 64.4% |
| Same-scale Multi-depth JEPA | NR positioning | L4 | 7.553 | 0.8697 | 0.9019 | 49.7% | 42.8% |
| Same-scale Multi-depth JEPA | NR positioning | L6 | 11.278 | 1.1907 | 1.4632 | 24.8% | 21.7% |
| Same-scale Multi-depth JEPA | NR positioning | L8 | 15.003 | 1.5204 | 1.7843 | 0.0% | 0.0% |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | 0.026 | 0.6164 | 0.6534 | 74.2% | 52.9% |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L4 | 0.052 | 0.8457 | 0.9367 | 49.5% | 35.4% |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L6 | 0.077 | 1.0978 | 1.2236 | 24.7% | 16.1% |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L8 | 0.102 | 1.3086 | 1.3859 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.4524 | 0.4835 | 74.3% | 59.2% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 0.6872 | 0.7562 | 49.5% | 38.0% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 0.9098 | 0.9708 | 24.8% | 17.9% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.1080 | 1.1874 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.4580 | 0.4683 | 74.3% | 59.3% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | 1.416 | 0.6880 | 0.8003 | 49.5% | 38.8% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | 2.111 | 0.9102 | 1.0096 | 24.8% | 19.1% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 2.807 | 1.1244 | 1.2091 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.4588 | 0.5644 | 74.4% | 61.6% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 5.193 | 0.6998 | 0.8424 | 49.6% | 41.5% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | 7.750 | 0.9446 | 1.0969 | 24.8% | 21.0% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | 10.308 | 1.1955 | 1.3379 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.4588 | 0.5002 | 74.3% | 59.0% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | 1.416 | 0.6708 | 0.7306 | 49.5% | 40.1% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | 2.111 | 0.9171 | 0.9938 | 24.8% | 18.0% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | 2.807 | 1.1191 | 1.2016 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | 3.828 | 0.5494 | 0.6407 | 74.5% | 64.3% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | 7.553 | 0.8822 | 1.0498 | 49.7% | 42.6% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.2052 | 1.3663 | 24.8% | 21.6% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.5370 | 1.6954 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.5830 | 0.6852 | 74.2% | 52.8% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 0.8035 | 0.9121 | 49.5% | 34.9% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | 0.077 | 1.0219 | 1.1496 | 24.7% | 17.3% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | 0.102 | 1.2350 | 1.3468 | 0.0% | 0.0% |

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
