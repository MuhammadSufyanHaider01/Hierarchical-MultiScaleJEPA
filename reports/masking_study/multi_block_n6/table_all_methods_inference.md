# All-approach headline inference comparison

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
