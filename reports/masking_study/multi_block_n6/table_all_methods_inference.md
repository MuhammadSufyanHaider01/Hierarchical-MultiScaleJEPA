# All-approach headline inference comparison

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
