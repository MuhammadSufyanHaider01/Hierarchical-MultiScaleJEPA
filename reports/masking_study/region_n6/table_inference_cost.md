# Detailed batch-1 inference cost

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms | FLOPs saving | Latency saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.5118 | 0.5423 | 74.3% | 59.9% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.7627 | 0.7963 | 49.5% | 40.2% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | 2.111 | 1.0237 | 1.1079 | 24.8% | 19.7% |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.2750 | 1.4085 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | 0.721 | 0.4787 | 0.5254 | 74.3% | 58.7% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | 1.416 | 0.7003 | 0.7470 | 49.5% | 39.5% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.9318 | 1.0112 | 24.8% | 19.5% |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 2.807 | 1.1578 | 1.2586 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4697 | 0.6069 | 74.4% | 61.4% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | 5.193 | 0.7255 | 0.9150 | 49.6% | 40.4% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | 7.750 | 0.9743 | 1.2232 | 24.8% | 20.0% |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | 10.308 | 1.2174 | 1.3959 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4762 | 0.5561 | 74.3% | 58.8% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | 1.416 | 0.7035 | 0.8086 | 49.5% | 39.1% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | 2.111 | 0.9384 | 1.0155 | 24.8% | 18.8% |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | 2.807 | 1.1557 | 1.2556 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | 3.828 | 0.5717 | 0.7766 | 74.5% | 63.3% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | 7.553 | 0.9008 | 1.0954 | 49.7% | 42.1% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2303 | 1.4191 | 24.8% | 20.9% |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5557 | 1.7371 | 0.0% | 0.0% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.6012 | 0.7444 | 74.2% | 52.1% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | 0.052 | 0.8241 | 0.8939 | 49.5% | 34.4% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | 0.077 | 1.0366 | 1.1626 | 24.7% | 17.4% |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 0.102 | 1.2556 | 1.3944 | 0.0% | 0.0% |
| Same-scale Multi-depth JEPA | RF sensing | L2 | 0.721 | 0.4509 | 0.4844 | 74.3% | 58.9% |
| Same-scale Multi-depth JEPA | RF sensing | L4 | 1.416 | 0.6742 | 0.6855 | 49.5% | 38.6% |
| Same-scale Multi-depth JEPA | RF sensing | L6 | 2.111 | 0.8783 | 0.9020 | 24.8% | 20.0% |
| Same-scale Multi-depth JEPA | RF sensing | L8 | 2.807 | 1.0981 | 1.1954 | 0.0% | 0.0% |
| Same-scale Multi-depth JEPA | Jamming classification | L2 | 0.721 | 0.4403 | 0.4629 | 74.3% | 59.7% |
| Same-scale Multi-depth JEPA | Jamming classification | L4 | 1.416 | 0.6550 | 0.6869 | 49.5% | 40.1% |
| Same-scale Multi-depth JEPA | Jamming classification | L6 | 2.111 | 0.8768 | 0.9461 | 24.8% | 19.8% |
| Same-scale Multi-depth JEPA | Jamming classification | L8 | 2.807 | 1.0937 | 1.1360 | 0.0% | 0.0% |
| Same-scale Multi-depth JEPA | Human activity sensing | L2 | 2.635 | 0.4563 | 0.4555 | 74.4% | 61.3% |
| Same-scale Multi-depth JEPA | Human activity sensing | L4 | 5.193 | 0.6922 | 0.6885 | 49.6% | 41.3% |
| Same-scale Multi-depth JEPA | Human activity sensing | L6 | 7.750 | 0.9386 | 0.9344 | 24.8% | 20.5% |
| Same-scale Multi-depth JEPA | Human activity sensing | L8 | 10.308 | 1.1800 | 1.1767 | 0.0% | 0.0% |
| Same-scale Multi-depth JEPA | Interference classification | L2 | 0.721 | 0.4509 | 0.4593 | 74.3% | 58.9% |
| Same-scale Multi-depth JEPA | Interference classification | L4 | 1.416 | 0.6644 | 0.6785 | 49.5% | 39.5% |
| Same-scale Multi-depth JEPA | Interference classification | L6 | 2.111 | 0.8790 | 0.9048 | 24.8% | 20.0% |
| Same-scale Multi-depth JEPA | Interference classification | L8 | 2.807 | 1.0981 | 1.1746 | 0.0% | 0.0% |
| Same-scale Multi-depth JEPA | NR positioning | L2 | 3.828 | 0.5431 | 0.5460 | 74.5% | 64.6% |
| Same-scale Multi-depth JEPA | NR positioning | L4 | 7.553 | 0.8711 | 0.8734 | 49.7% | 43.2% |
| Same-scale Multi-depth JEPA | NR positioning | L6 | 11.278 | 1.1942 | 1.1946 | 24.8% | 22.1% |
| Same-scale Multi-depth JEPA | NR positioning | L8 | 15.003 | 1.5327 | 1.5290 | 0.0% | 0.0% |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | 0.026 | 0.5739 | 0.5849 | 74.2% | 52.2% |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L4 | 0.052 | 0.7966 | 0.8066 | 49.5% | 33.7% |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L6 | 0.077 | 0.9974 | 1.0167 | 24.7% | 17.0% |
| Same-scale Multi-depth JEPA | DeepMIMO beam | L8 | 0.102 | 1.2012 | 1.2282 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.4542 | 0.4795 | 74.3% | 63.8% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 0.6617 | 0.7098 | 49.5% | 47.3% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 0.9112 | 0.9645 | 24.8% | 27.4% |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.2549 | 1.9018 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.5394 | 1.2924 | 74.3% | 51.6% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | 1.416 | 0.7819 | 1.3971 | 49.5% | 29.8% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | 2.111 | 0.9049 | 0.9886 | 24.8% | 18.7% |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 2.807 | 1.1135 | 1.1969 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.4440 | 0.4584 | 74.4% | 62.7% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 5.193 | 0.8331 | 1.6843 | 49.6% | 30.0% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | 7.750 | 1.1514 | 1.9251 | 24.8% | 3.3% |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | 10.308 | 1.1904 | 1.1970 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.4551 | 0.4799 | 74.3% | 61.6% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | 1.416 | 0.7229 | 0.7344 | 49.5% | 39.0% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | 2.111 | 0.9504 | 0.9825 | 24.8% | 19.8% |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | 2.807 | 1.1845 | 1.2172 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | 3.828 | 0.5491 | 0.5564 | 74.5% | 64.8% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | 7.553 | 0.8700 | 0.8815 | 49.7% | 44.2% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.1954 | 1.2354 | 24.8% | 23.4% |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.5602 | 2.1005 | 0.0% | 0.0% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.5712 | 0.6024 | 74.2% | 53.4% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 0.8007 | 0.8962 | 49.5% | 34.7% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | 0.077 | 0.9932 | 1.0634 | 24.7% | 19.0% |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | 0.102 | 1.2269 | 1.2999 | 0.0% | 0.0% |
