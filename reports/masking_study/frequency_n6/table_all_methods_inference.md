# All-approach headline inference comparison

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
