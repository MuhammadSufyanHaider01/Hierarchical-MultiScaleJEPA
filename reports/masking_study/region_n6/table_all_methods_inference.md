# All-approach headline inference comparison

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| Original LatentWave | RF sensing | L8 | 2.807 | 1.2750 | 1.4085 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.7627 | 0.7963 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 0.9112 | 0.9645 |
| Original LatentWave | Jamming classification | L8 | 2.807 | 1.1578 | 1.2586 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.9318 | 1.0112 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.5394 | 1.2924 |
| Original LatentWave | Human activity sensing | L8 | 10.308 | 1.2174 | 1.3959 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4697 | 0.6069 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.4440 | 0.4584 |
| Original LatentWave | Interference classification | L8 | 2.807 | 1.1557 | 1.2556 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4762 | 0.5561 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.4551 | 0.4799 |
| Original LatentWave | NR positioning | L8 | 15.003 | 1.5557 | 1.7371 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2303 | 1.4191 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.5602 | 2.1005 |
| Original LatentWave | DeepMIMO beam | L8 | 0.102 | 1.2556 | 1.3944 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.6012 | 0.7444 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.5712 | 0.6024 |
