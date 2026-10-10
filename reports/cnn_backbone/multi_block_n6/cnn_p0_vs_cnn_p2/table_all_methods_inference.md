# ViT and CNN validation-best inference cost

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
