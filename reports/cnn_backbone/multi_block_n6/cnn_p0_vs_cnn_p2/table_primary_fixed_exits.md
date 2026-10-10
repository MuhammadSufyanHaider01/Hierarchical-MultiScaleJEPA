# Fixed-exit downstream metrics

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
