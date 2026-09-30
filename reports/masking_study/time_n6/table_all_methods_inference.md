# All-approach headline inference comparison

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
