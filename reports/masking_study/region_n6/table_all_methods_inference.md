# All-approach headline inference comparison

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
