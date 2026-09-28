# All-approach headline inference comparison

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| Original LatentWave | RF sensing | L8 | 2.807 | 1.1259 | 1.2477 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.1259 | 1.2477 |
| Same-scale Multi-depth JEPA | RF sensing | L2 | 0.721 | 0.4453 | 0.4697 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.5170 | 0.7086 |
| Original LatentWave | Jamming classification | L8 | 2.807 | 1.1180 | 1.2667 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.9049 | 1.0111 |
| Same-scale Multi-depth JEPA | Jamming classification | L4 | 1.416 | 0.6757 | 0.7234 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | 1.416 | 0.7376 | 0.8430 |
| Original LatentWave | Human activity sensing | L8 | 10.308 | 1.1939 | 1.3596 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4585 | 0.6210 |
| Same-scale Multi-depth JEPA | Human activity sensing | L2 | 2.635 | 0.4445 | 0.4498 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.4931 | 0.6314 |
| Original LatentWave | Interference classification | L8 | 2.807 | 1.1184 | 1.2317 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4493 | 0.4954 |
| Same-scale Multi-depth JEPA | Interference classification | L2 | 0.721 | 0.4480 | 0.4726 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.5121 | 0.6582 |
| Original LatentWave | NR positioning | L8 | 15.003 | 1.5560 | 1.7544 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2147 | 1.3986 |
| Same-scale Multi-depth JEPA | NR positioning | L6 | 11.278 | 1.1907 | 1.4632 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.6759 | 2.7483 |
| Original LatentWave | DeepMIMO beam | L8 | 0.102 | 1.2222 | 1.2982 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.5903 | 0.7066 |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | 0.026 | 0.6164 | 0.6534 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.6414 | 0.7627 |
