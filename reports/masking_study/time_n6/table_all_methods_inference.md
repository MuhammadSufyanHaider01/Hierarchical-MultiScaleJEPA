# All-approach headline inference comparison

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| Original LatentWave | RF sensing | L8 | 2.807 | 1.1221 | 1.2569 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.4529 | 0.6066 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 1.1986 | 2.7153 |
| Original LatentWave | Jamming classification | L8 | 2.807 | 1.1183 | 1.2613 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | 0.721 | 0.4636 | 0.5966 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.9171 | 2.5556 |
| Original LatentWave | Human activity sensing | L8 | 10.308 | 1.2337 | 1.4207 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4702 | 0.7052 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.9278 | 2.6623 |
| Original LatentWave | Interference classification | L8 | 2.807 | 1.1167 | 1.3175 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4514 | 0.5666 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.9485 | 2.5553 |
| Original LatentWave | NR positioning | L8 | 15.003 | 1.5814 | 1.8182 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | 7.553 | 0.9480 | 1.2454 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 2.1171 | 3.6676 |
| Original LatentWave | DeepMIMO beam | L8 | 0.102 | 1.1969 | 1.3786 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.5901 | 0.7504 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 1.1577 | 4.7483 |
