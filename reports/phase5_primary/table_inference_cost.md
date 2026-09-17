# Detailed batch-1 inference cost

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms | FLOPs saving | Latency saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.4639 | 0.5663 | 74.3% | 59.5% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.6879 | 0.8330 | 49.5% | 40.0% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | 2.111 | 0.9028 | 1.0622 | 24.8% | 21.2% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.1463 | 1.2914 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | 3.828 | 0.5532 | 0.7015 | 74.5% | 64.5% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | 7.553 | 0.8855 | 1.1308 | 49.7% | 43.1% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2160 | 1.4982 | 24.8% | 21.9% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5568 | 1.8755 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.6291 | 0.8062 | 74.2% | 53.8% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | 0.052 | 0.8745 | 1.0327 | 49.5% | 35.8% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | 0.077 | 1.0911 | 1.2416 | 24.7% | 19.9% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 0.102 | 1.3618 | 1.5644 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO LoS/NLoS | L2 | 0.026 | 0.5931 | 0.6949 | 74.2% | 52.1% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO LoS/NLoS | L4 | 0.052 | 0.8020 | 0.9087 | 49.5% | 35.2% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO LoS/NLoS | L6 | 0.077 | 1.0026 | 1.0962 | 24.7% | 19.0% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO LoS/NLoS | L8 | 0.102 | 1.2372 | 1.3776 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.4578 | 0.4751 | 74.3% | 60.4% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 0.7052 | 0.7232 | 49.5% | 38.9% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 0.9070 | 0.9555 | 24.8% | 21.5% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.1549 | 1.1967 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | 3.828 | 0.5304 | 0.5416 | 74.5% | 65.8% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | 7.553 | 0.8517 | 0.8649 | 49.7% | 45.1% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.1879 | 1.3249 | 24.8% | 23.4% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.5508 | 1.7468 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.5839 | 0.6922 | 74.2% | 52.6% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 0.8011 | 0.8718 | 49.5% | 35.0% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | 0.077 | 1.0150 | 1.0703 | 24.7% | 17.6% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | 0.102 | 1.2316 | 1.2973 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO LoS/NLoS | L2 | 0.026 | 0.5752 | 0.5976 | 74.2% | 53.6% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO LoS/NLoS | L4 | 0.052 | 0.7857 | 0.8373 | 49.5% | 36.7% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO LoS/NLoS | L6 | 0.077 | 1.0030 | 1.1679 | 24.7% | 19.2% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO LoS/NLoS | L8 | 0.102 | 1.2409 | 1.3166 | 0.0% | 0.0% |
