# All-approach headline inference comparison

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| Original LatentWave | RF sensing | L8 | 2.807 | 1.1463 | 1.2914 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.1463 | 1.2914 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 0.9070 | 0.9555 |
| Original LatentWave | NR positioning | L8 | 15.003 | 1.5568 | 1.8755 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2160 | 1.4982 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.1879 | 1.3249 |
| Original LatentWave | DeepMIMO beam | L8 | 0.102 | 1.3618 | 1.5644 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.6291 | 0.8062 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.5839 | 0.6922 |
| Original LatentWave | DeepMIMO LoS/NLoS | L8 | 0.102 | 1.2372 | 1.3776 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO LoS/NLoS | L2 | 0.026 | 0.5931 | 0.6949 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO LoS/NLoS | L2 | 0.026 | 0.5752 | 0.5976 |
