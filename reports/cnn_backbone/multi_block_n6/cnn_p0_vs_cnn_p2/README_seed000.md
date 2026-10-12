# M-block: ViT B0/P0/P2 vs CNN C0/C2, single-seed packet

This M-block / span-time, N_m=6 packet compares ViT B0/P0/P2 with CNN C0/C2 at seed 0. B0 is the fixed L8 result from the same ViT P0 checkpoint, not a separate pretraining run. It contains single-seed point estimates, not a statistical
superiority claim.

## Scope and comparability

| Approach | Pretraining | Downstream representation | Role |
| --- | --- | --- | --- |
| B0 — Original LatentWave (full-depth) | ViT P0 checkpoint; full-input L8 | Frozen ViT context encoder at L8 | fixed-depth reference derived from P0 |
| P0 — LatentWave + post-hoc frozen exits | ViT LatentWave P0 objective | Frozen ViT context encoder at L2/L4/L6/L8 | ViT baseline; head_seed_000 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | ViT P2 objective with multi-scale auxiliary reconstruction | Frozen ViT context encoder at L2/L4/L6/L8 | ViT hierarchical baseline |
| C0 — CNN original L8 objective | ConvNeXt V2; one L8 predictor and masked-token MSE | Frozen CNN context encoder at S1/S2/S3/S4 | single-objective CNN baseline |
| C2 — CNN mask-relative fine-to-coarse | ConvNeXt V2; L8 anchor plus L2/L4/L6 mask-relative losses | Frozen CNN context encoder at S1/S2/S3/S4 | hierarchical CNN method |

CNN stages S1/S2/S3/S4 map to logical exits L2/L4/L6/L8. All learned approaches
use the same persisted 80/10/10 downstream split and validation-only exit
selection before the held-out test split is read. ViT P0 uses the existing
per-profile `head_seed_000` packet; ViT P2 uses the matching-profile hierarchical
packet; B0 reuses P0's fixed L8 result. CNN C0 and C2 use the matched staged
pretraining protocol described in the provenance note.

## Run-specific provenance

C0 was trained directly with a deterministic 90/10 pretraining split and validation-loss early stopping (100-epoch maximum); its planned five-epoch full-data prefix was explicitly skipped. C2 reuses its already-trained frozen heads and matching epoch-69 checkpoint (pretraining validation loss 0.466410); only its missing held-out test evaluation and inference benchmark are added. A later C2 continuation reached a better epoch-93 pretraining checkpoint (validation loss 0.431457), but its heads were not retrained on that checkpoint and it is not the C2 representation evaluated in this report. Training histories differ; all results are seed-0 descriptive comparisons.


## Task suite

RF sensing, Jamming classification, Human activity sensing, Interference classification, NR positioning, DeepMIMO beam. None of these downstream
caches participates in pretraining.

## Validation-best held-out performance

| Approach | Task | Exit | Test value |
| --- | --- | --- | --- |
| B0 — Original LatentWave | RF sensing | L8 | 69.46% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 63.30% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 72.26% |
| C0 — CNN Original L8 JEPA Objective | RF sensing | L6 | 48.61% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | RF sensing | L4 | 73.93% |
| B0 — Original LatentWave | Jamming classification | L8 | 88.00% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 88.00% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 88.27% |
| C0 — CNN Original L8 JEPA Objective | Jamming classification | L6 | 70.40% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Jamming classification | L4 | 68.13% |
| B0 — Original LatentWave | Human activity sensing | L8 | 98.33% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | 96.67% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 100.00% |
| C0 — CNN Original L8 JEPA Objective | Human activity sensing | L6 | 94.17% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Human activity sensing | L2 | 93.33% |
| B0 — Original LatentWave | Interference classification | L8 | 74.79% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | 75.21% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | 75.86% |
| C0 — CNN Original L8 JEPA Objective | Interference classification | L4 | 72.86% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Interference classification | L4 | 73.00% |
| B0 — Original LatentWave | NR positioning | L8 | 3.025 m |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 3.025 m |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 2.760 m |
| C0 — CNN Original L8 JEPA Objective | NR positioning | L6 | 3.423 m |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | NR positioning | L4 | 3.708 m |
| B0 — Original LatentWave | DeepMIMO beam | L8 | 45.74% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 45.74% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 43.47% |
| C0 — CNN Original L8 JEPA Objective | DeepMIMO beam | L6 | 57.05% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | DeepMIMO beam | L4 | 62.77% |

## Validation-best inference cost

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| B0 — Original LatentWave | RF sensing | L8 | 2.807 | 1.1175 | 1.2204 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.6789 | 0.7893 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.1080 | 1.1874 |
| C0 — CNN Original L8 JEPA Objective | RF sensing | L6 | 1.742 | 1.6358 | 1.7402 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | RF sensing | L4 | 1.732 | 1.7056 | 1.8553 |
| B0 — Original LatentWave | Jamming classification | L8 | 2.807 | 1.1267 | 1.2456 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 2.807 | 1.1267 | 1.2456 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 2.807 | 1.1244 | 1.2091 |
| C0 — CNN Original L8 JEPA Objective | Jamming classification | L6 | 1.742 | 1.6279 | 1.7566 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Jamming classification | L4 | 1.732 | 1.7075 | 1.8769 |
| B0 — Original LatentWave | Human activity sensing | L8 | 10.308 | 1.1870 | 1.3541 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | 5.193 | 0.7012 | 0.8600 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 5.193 | 0.6998 | 0.8424 |
| C0 — CNN Original L8 JEPA Objective | Human activity sensing | L6 | 5.225 | 1.6027 | 1.7706 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Human activity sensing | L2 | 5.181 | 1.6712 | 1.8222 |
| B0 — Original LatentWave | Interference classification | L8 | 2.807 | 1.1186 | 1.2273 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | 2.111 | 0.8998 | 1.0251 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | 2.807 | 1.1191 | 1.2016 |
| C0 — CNN Original L8 JEPA Objective | Interference classification | L4 | 1.732 | 1.8303 | 1.9041 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Interference classification | L4 | 1.732 | 1.7161 | 1.8839 |
| B0 — Original LatentWave | NR positioning | L8 | 15.003 | 1.5272 | 1.6933 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5272 | 1.6933 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.2052 | 1.3663 |
| C0 — CNN Original L8 JEPA Objective | NR positioning | L6 | 6.966 | 1.6433 | 1.7965 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | NR positioning | L4 | 6.928 | 1.7400 | 1.9296 |
| B0 — Original LatentWave | DeepMIMO beam | L8 | 0.102 | 1.2205 | 1.3524 |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 0.102 | 1.2205 | 1.3524 |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 0.8035 | 0.9121 |
| C0 — CNN Original L8 JEPA Objective | DeepMIMO beam | L6 | 0.109 | 1.6551 | 1.8244 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | DeepMIMO beam | L4 | 0.090 | 1.7112 | 1.8795 |

## Fixed-exit downstream metrics

| Approach | Task | Exit | Metric | Test value | Validation value | Head epoch |
| --- | --- | --- | --- | --- | --- | --- |
| B0 — Original LatentWave | RF sensing | L8 | Macro accuracy | 69.46% | 71.08% | 163 |
| B0 — Original LatentWave | Jamming classification | L8 | Macro accuracy | 88.00% | 87.73% | 21 |
| B0 — Original LatentWave | Human activity sensing | L8 | Macro accuracy | 98.33% | 100.00% | 26 |
| B0 — Original LatentWave | Interference classification | L8 | Macro accuracy | 74.79% | 73.79% | 34 |
| B0 — Original LatentWave | NR positioning | L8 | Mean distance error (m) | 3.025 m | 3.039 m | 59 |
| B0 — Original LatentWave | DeepMIMO beam | L8 | Macro accuracy | 45.74% | 44.69% | 54 |
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
| C0 — CNN Original L8 JEPA Objective | RF sensing | L2 | Macro accuracy | 29.13% | 32.48% | 134 |
| C0 — CNN Original L8 JEPA Objective | RF sensing | L4 | Macro accuracy | 33.67% | 35.05% | 46 |
| C0 — CNN Original L8 JEPA Objective | RF sensing | L6 | Macro accuracy | 48.61% | 46.77% | 68 |
| C0 — CNN Original L8 JEPA Objective | RF sensing | L8 | Macro accuracy | 41.01% | 34.36% | 36 |
| C0 — CNN Original L8 JEPA Objective | Jamming classification | L2 | Macro accuracy | 54.80% | 58.27% | 84 |
| C0 — CNN Original L8 JEPA Objective | Jamming classification | L4 | Macro accuracy | 58.93% | 64.40% | 40 |
| C0 — CNN Original L8 JEPA Objective | Jamming classification | L6 | Macro accuracy | 70.40% | 72.80% | 108 |
| C0 — CNN Original L8 JEPA Objective | Jamming classification | L8 | Macro accuracy | 62.53% | 60.40% | 31 |
| C0 — CNN Original L8 JEPA Objective | Human activity sensing | L2 | Macro accuracy | 84.17% | 92.50% | 64 |
| C0 — CNN Original L8 JEPA Objective | Human activity sensing | L4 | Macro accuracy | 78.33% | 87.50% | 21 |
| C0 — CNN Original L8 JEPA Objective | Human activity sensing | L6 | Macro accuracy | 94.17% | 96.67% | 55 |
| C0 — CNN Original L8 JEPA Objective | Human activity sensing | L8 | Macro accuracy | 78.33% | 89.17% | 60 |
| C0 — CNN Original L8 JEPA Objective | Interference classification | L2 | Macro accuracy | 69.50% | 71.00% | 36 |
| C0 — CNN Original L8 JEPA Objective | Interference classification | L4 | Macro accuracy | 72.86% | 74.57% | 58 |
| C0 — CNN Original L8 JEPA Objective | Interference classification | L6 | Macro accuracy | 72.43% | 73.79% | 12 |
| C0 — CNN Original L8 JEPA Objective | Interference classification | L8 | Macro accuracy | 68.29% | 72.64% | 37 |
| C0 — CNN Original L8 JEPA Objective | NR positioning | L2 | Mean distance error (m) | 4.793 m | 4.703 m | 97 |
| C0 — CNN Original L8 JEPA Objective | NR positioning | L4 | Mean distance error (m) | 4.604 m | 4.525 m | 29 |
| C0 — CNN Original L8 JEPA Objective | NR positioning | L6 | Mean distance error (m) | 3.423 m | 3.348 m | 84 |
| C0 — CNN Original L8 JEPA Objective | NR positioning | L8 | Mean distance error (m) | 5.180 m | 5.024 m | 48 |
| C0 — CNN Original L8 JEPA Objective | DeepMIMO beam | L2 | Macro accuracy | 55.59% | 54.95% | 58 |
| C0 — CNN Original L8 JEPA Objective | DeepMIMO beam | L4 | Macro accuracy | 48.04% | 47.99% | 81 |
| C0 — CNN Original L8 JEPA Objective | DeepMIMO beam | L6 | Macro accuracy | 57.05% | 58.12% | 36 |
| C0 — CNN Original L8 JEPA Objective | DeepMIMO beam | L8 | Macro accuracy | 47.02% | 49.18% | 70 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | RF sensing | L2 | Macro accuracy | 57.89% | 62.00% | 77 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | RF sensing | L4 | Macro accuracy | 73.93% | 73.81% | 80 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | RF sensing | L6 | Macro accuracy | 57.50% | 63.39% | 33 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | RF sensing | L8 | Macro accuracy | 38.33% | 40.35% | 75 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Jamming classification | L2 | Macro accuracy | 68.13% | 69.20% | 28 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Jamming classification | L4 | Macro accuracy | 68.13% | 72.67% | 49 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Jamming classification | L6 | Macro accuracy | 69.47% | 69.87% | 27 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Jamming classification | L8 | Macro accuracy | 60.80% | 60.27% | 52 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Human activity sensing | L2 | Macro accuracy | 93.33% | 98.33% | 105 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Human activity sensing | L4 | Macro accuracy | 91.67% | 95.83% | 66 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Human activity sensing | L6 | Macro accuracy | 92.50% | 90.83% | 26 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Human activity sensing | L8 | Macro accuracy | 75.00% | 70.83% | 50 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Interference classification | L2 | Macro accuracy | 70.57% | 71.36% | 40 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Interference classification | L4 | Macro accuracy | 73.00% | 74.71% | 20 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Interference classification | L6 | Macro accuracy | 70.29% | 72.79% | 24 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Interference classification | L8 | Macro accuracy | 70.36% | 72.79% | 22 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | NR positioning | L2 | Mean distance error (m) | 3.926 m | 3.999 m | 56 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | NR positioning | L4 | Mean distance error (m) | 3.708 m | 3.653 m | 37 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | NR positioning | L6 | Mean distance error (m) | 3.809 m | 3.819 m | 65 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | NR positioning | L8 | Mean distance error (m) | 4.958 m | 4.949 m | 50 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | DeepMIMO beam | L2 | Macro accuracy | 57.59% | 60.43% | 28 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | DeepMIMO beam | L4 | Macro accuracy | 62.77% | 63.08% | 60 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | DeepMIMO beam | L6 | Macro accuracy | 59.28% | 60.72% | 125 |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | DeepMIMO beam | L8 | Macro accuracy | 40.56% | 42.77% | 83 |

## Detailed validation-best exits

| Approach | Task | Exit | Test value |
| --- | --- | --- | --- |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 63.30% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 88.00% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | 96.67% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | 75.21% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 3.025 m |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 45.74% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 72.26% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 88.27% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 100.00% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | 75.86% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 2.760 m |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 43.47% |
| C0 — CNN Original L8 JEPA Objective | RF sensing | L6 | 48.61% |
| C0 — CNN Original L8 JEPA Objective | Jamming classification | L6 | 70.40% |
| C0 — CNN Original L8 JEPA Objective | Human activity sensing | L6 | 94.17% |
| C0 — CNN Original L8 JEPA Objective | Interference classification | L4 | 72.86% |
| C0 — CNN Original L8 JEPA Objective | NR positioning | L6 | 3.423 m |
| C0 — CNN Original L8 JEPA Objective | DeepMIMO beam | L6 | 57.05% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | RF sensing | L4 | 73.93% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Jamming classification | L4 | 68.13% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Human activity sensing | L2 | 93.33% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Interference classification | L4 | 73.00% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | NR positioning | L4 | 3.708 m |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | DeepMIMO beam | L4 | 62.77% |

## Detailed batch-1 inference cost

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms | FLOPs saving | Latency saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 — Original LatentWave | RF sensing | L8 | 2.807 | 1.1175 | 1.2204 | 0.0% | 0.0% |
| B0 — Original LatentWave | Jamming classification | L8 | 2.807 | 1.1267 | 1.2456 | 0.0% | 0.0% |
| B0 — Original LatentWave | Human activity sensing | L8 | 10.308 | 1.1870 | 1.3541 | 0.0% | 0.0% |
| B0 — Original LatentWave | Interference classification | L8 | 2.807 | 1.1186 | 1.2273 | 0.0% | 0.0% |
| B0 — Original LatentWave | NR positioning | L8 | 15.003 | 1.5272 | 1.6933 | 0.0% | 0.0% |
| B0 — Original LatentWave | DeepMIMO beam | L8 | 0.102 | 1.2205 | 1.3524 | 0.0% | 0.0% |
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
| C0 — CNN Original L8 JEPA Objective | RF sensing | L2 | 1.727 | 1.6450 | 1.7579 | 1.9% | -0.0% |
| C0 — CNN Original L8 JEPA Objective | RF sensing | L4 | 1.732 | 1.6568 | 1.8132 | 1.6% | -0.8% |
| C0 — CNN Original L8 JEPA Objective | RF sensing | L6 | 1.742 | 1.6358 | 1.7402 | 1.1% | 0.5% |
| C0 — CNN Original L8 JEPA Objective | RF sensing | L8 | 1.761 | 1.6445 | 1.8048 | 0.0% | 0.0% |
| C0 — CNN Original L8 JEPA Objective | Jamming classification | L2 | 1.727 | 1.6533 | 1.8070 | 1.9% | -0.8% |
| C0 — CNN Original L8 JEPA Objective | Jamming classification | L4 | 1.732 | 1.6406 | 1.7818 | 1.6% | -0.1% |
| C0 — CNN Original L8 JEPA Objective | Jamming classification | L6 | 1.742 | 1.6279 | 1.7566 | 1.1% | 0.7% |
| C0 — CNN Original L8 JEPA Objective | Jamming classification | L8 | 1.761 | 1.6395 | 1.8061 | 0.0% | 0.0% |
| C0 — CNN Original L8 JEPA Objective | Human activity sensing | L2 | 5.181 | 1.6116 | 1.7585 | 1.9% | 0.3% |
| C0 — CNN Original L8 JEPA Objective | Human activity sensing | L4 | 5.196 | 1.6192 | 1.7754 | 1.6% | -0.1% |
| C0 — CNN Original L8 JEPA Objective | Human activity sensing | L6 | 5.225 | 1.6027 | 1.7706 | 1.1% | 0.9% |
| C0 — CNN Original L8 JEPA Objective | Human activity sensing | L8 | 5.282 | 1.6172 | 1.7762 | 0.0% | 0.0% |
| C0 — CNN Original L8 JEPA Objective | Interference classification | L2 | 1.727 | 1.6446 | 1.7034 | 1.9% | 10.1% |
| C0 — CNN Original L8 JEPA Objective | Interference classification | L4 | 1.732 | 1.8303 | 1.9041 | 1.6% | -0.1% |
| C0 — CNN Original L8 JEPA Objective | Interference classification | L6 | 1.742 | 1.8439 | 1.9004 | 1.1% | -0.8% |
| C0 — CNN Original L8 JEPA Objective | Interference classification | L8 | 1.761 | 1.8284 | 1.9046 | 0.0% | 0.0% |
| C0 — CNN Original L8 JEPA Objective | NR positioning | L2 | 6.908 | 1.6639 | 1.8305 | 1.9% | -0.4% |
| C0 — CNN Original L8 JEPA Objective | NR positioning | L4 | 6.928 | 1.6594 | 1.8153 | 1.6% | -0.1% |
| C0 — CNN Original L8 JEPA Objective | NR positioning | L6 | 6.966 | 1.6433 | 1.7965 | 1.1% | 0.8% |
| C0 — CNN Original L8 JEPA Objective | NR positioning | L8 | 7.043 | 1.6570 | 1.8086 | 0.0% | 0.0% |
| C0 — CNN Original L8 JEPA Objective | DeepMIMO beam | L2 | 0.080 | 1.6540 | 1.8170 | 45.8% | -1.4% |
| C0 — CNN Original L8 JEPA Objective | DeepMIMO beam | L4 | 0.090 | 1.6463 | 1.8142 | 39.2% | -1.0% |
| C0 — CNN Original L8 JEPA Objective | DeepMIMO beam | L6 | 0.109 | 1.6551 | 1.8244 | 26.1% | -1.5% |
| C0 — CNN Original L8 JEPA Objective | DeepMIMO beam | L8 | 0.147 | 1.6307 | 1.7901 | 0.0% | 0.0% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | RF sensing | L2 | 1.727 | 1.7124 | 1.8884 | 1.9% | -0.1% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | RF sensing | L4 | 1.732 | 1.7056 | 1.8553 | 1.6% | 0.3% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | RF sensing | L6 | 1.742 | 1.6888 | 1.8389 | 1.1% | 1.2% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | RF sensing | L8 | 1.761 | 1.7102 | 1.8757 | 0.0% | 0.0% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Jamming classification | L2 | 1.727 | 1.6967 | 1.8134 | 1.9% | 0.1% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Jamming classification | L4 | 1.732 | 1.7075 | 1.8769 | 1.6% | -0.6% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Jamming classification | L6 | 1.742 | 1.6997 | 1.8782 | 1.1% | -0.1% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Jamming classification | L8 | 1.761 | 1.6979 | 1.8646 | 0.0% | 0.0% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Human activity sensing | L2 | 5.181 | 1.6712 | 1.8222 | 1.9% | 1.0% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Human activity sensing | L4 | 5.196 | 1.6850 | 1.8463 | 1.6% | 0.2% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Human activity sensing | L6 | 5.225 | 1.6601 | 1.8273 | 1.1% | 1.6% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Human activity sensing | L8 | 5.282 | 1.6877 | 1.8569 | 0.0% | 0.0% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Interference classification | L2 | 1.727 | 1.7097 | 1.8489 | 1.9% | -0.0% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Interference classification | L4 | 1.732 | 1.7161 | 1.8839 | 1.6% | -0.4% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Interference classification | L6 | 1.742 | 1.6983 | 1.8648 | 1.1% | 0.6% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Interference classification | L8 | 1.761 | 1.7091 | 1.8708 | 0.0% | 0.0% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | NR positioning | L2 | 6.908 | 1.7142 | 1.8771 | 1.9% | 9.0% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | NR positioning | L4 | 6.928 | 1.7400 | 1.9296 | 1.6% | 7.6% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | NR positioning | L6 | 6.966 | 1.8785 | 2.0018 | 1.1% | 0.3% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | NR positioning | L8 | 7.043 | 1.8840 | 2.0077 | 0.0% | 0.0% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | DeepMIMO beam | L2 | 0.080 | 1.6979 | 1.8591 | 45.8% | -1.3% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | DeepMIMO beam | L4 | 0.090 | 1.7112 | 1.8795 | 39.2% | -2.1% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | DeepMIMO beam | L6 | 0.109 | 1.6901 | 1.8666 | 26.1% | -0.8% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | DeepMIMO beam | L8 | 0.147 | 1.6768 | 1.8478 | 0.0% | 0.0% |

## Figures

- `early_exit_performance.png`
- `early_exit_performance.pdf`
- `selected_exit_performance.png`
- `selected_exit_performance.pdf`
- `inference_efficiency.png`
- `inference_efficiency.pdf`
