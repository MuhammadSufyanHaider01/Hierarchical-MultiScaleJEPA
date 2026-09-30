# Detailed batch-1 inference cost

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms | FLOPs saving | Latency saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.5118 | 0.5423 | 74.3% | 59.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.7627 | 0.7963 | 49.5% | 40.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | 2.111 | 1.0237 | 1.1079 | 24.8% | 19.7% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.2750 | 1.4085 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | 0.721 | 0.4787 | 0.5254 | 74.3% | 58.7% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | 1.416 | 0.7003 | 0.7470 | 49.5% | 39.5% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.9318 | 1.0112 | 24.8% | 19.5% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 2.807 | 1.1578 | 1.2586 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4697 | 0.6069 | 74.4% | 61.4% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | 5.193 | 0.7255 | 0.9150 | 49.6% | 40.4% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | 7.750 | 0.9743 | 1.2232 | 24.8% | 20.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | 10.308 | 1.2174 | 1.3959 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4762 | 0.5561 | 74.3% | 58.8% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | 1.416 | 0.7035 | 0.8086 | 49.5% | 39.1% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | 2.111 | 0.9384 | 1.0155 | 24.8% | 18.8% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | 2.807 | 1.1557 | 1.2556 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | 3.828 | 0.5717 | 0.7766 | 74.5% | 63.3% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | 7.553 | 0.9008 | 1.0954 | 49.7% | 42.1% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2303 | 1.4191 | 24.8% | 20.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5557 | 1.7371 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.6012 | 0.7444 | 74.2% | 52.1% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | 0.052 | 0.8241 | 0.8939 | 49.5% | 34.4% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | 0.077 | 1.0366 | 1.1626 | 24.7% | 17.4% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 0.102 | 1.2556 | 1.3944 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L2 | 0.721 | 0.4509 | 0.4844 | 74.3% | 58.9% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L4 | 1.416 | 0.6742 | 0.6855 | 49.5% | 38.6% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L6 | 2.111 | 0.8783 | 0.9020 | 24.8% | 20.0% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L8 | 2.807 | 1.0981 | 1.1954 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L2 | 0.721 | 0.4403 | 0.4629 | 74.3% | 59.7% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L4 | 1.416 | 0.6550 | 0.6869 | 49.5% | 40.1% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L6 | 2.111 | 0.8768 | 0.9461 | 24.8% | 19.8% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L8 | 2.807 | 1.0937 | 1.1360 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L2 | 2.635 | 0.4563 | 0.4555 | 74.4% | 61.3% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L4 | 5.193 | 0.6922 | 0.6885 | 49.6% | 41.3% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L6 | 7.750 | 0.9386 | 0.9344 | 24.8% | 20.5% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L8 | 10.308 | 1.1800 | 1.1767 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L2 | 0.721 | 0.4509 | 0.4593 | 74.3% | 58.9% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L4 | 1.416 | 0.6644 | 0.6785 | 49.5% | 39.5% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L6 | 2.111 | 0.8790 | 0.9048 | 24.8% | 20.0% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L8 | 2.807 | 1.0981 | 1.1746 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L2 | 3.828 | 0.5431 | 0.5460 | 74.5% | 64.6% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L4 | 7.553 | 0.8711 | 0.8734 | 49.7% | 43.2% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L6 | 11.278 | 1.1942 | 1.1946 | 24.8% | 22.1% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L8 | 15.003 | 1.5327 | 1.5290 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | 0.026 | 0.5739 | 0.5849 | 74.2% | 52.2% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L4 | 0.052 | 0.7966 | 0.8066 | 49.5% | 33.7% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L6 | 0.077 | 0.9974 | 1.0167 | 24.7% | 17.0% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L8 | 0.102 | 1.2012 | 1.2282 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.4542 | 0.4795 | 74.3% | 63.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 0.6617 | 0.7098 | 49.5% | 47.3% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 0.9112 | 0.9645 | 24.8% | 27.4% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.2549 | 1.9018 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.5394 | 1.2924 | 74.3% | 51.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | 1.416 | 0.7819 | 1.3971 | 49.5% | 29.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | 2.111 | 0.9049 | 0.9886 | 24.8% | 18.7% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 2.807 | 1.1135 | 1.1969 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.4440 | 0.4584 | 74.4% | 62.7% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 5.193 | 0.8331 | 1.6843 | 49.6% | 30.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | 7.750 | 1.1514 | 1.9251 | 24.8% | 3.3% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | 10.308 | 1.1904 | 1.1970 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.4551 | 0.4799 | 74.3% | 61.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | 1.416 | 0.7229 | 0.7344 | 49.5% | 39.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | 2.111 | 0.9504 | 0.9825 | 24.8% | 19.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | 2.807 | 1.1845 | 1.2172 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | 3.828 | 0.5491 | 0.5564 | 74.5% | 64.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | 7.553 | 0.8700 | 0.8815 | 49.7% | 44.2% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.1954 | 1.2354 | 24.8% | 23.4% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.5602 | 2.1005 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.5712 | 0.6024 | 74.2% | 53.4% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 0.8007 | 0.8962 | 49.5% | 34.7% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | 0.077 | 0.9932 | 1.0634 | 24.7% | 19.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | 0.102 | 1.2269 | 1.2999 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L2 | 0.721 | 0.8645 | 0.8956 | 74.3% | 58.2% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L4 | 1.416 | 1.2882 | 1.3171 | 49.5% | 37.6% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L6 | 2.111 | 1.7043 | 1.7146 | 24.8% | 17.5% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L8 | 2.807 | 2.0660 | 2.0891 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L2 | 0.721 | 0.8860 | 0.9066 | 74.3% | 57.4% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L4 | 1.416 | 1.2963 | 1.3242 | 49.5% | 37.6% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L6 | 2.111 | 1.6786 | 1.7061 | 24.8% | 19.2% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L8 | 2.807 | 2.0781 | 2.0970 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L2 | 2.635 | 0.8883 | 0.9066 | 74.4% | 57.3% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L4 | 5.193 | 1.2911 | 1.3209 | 49.6% | 37.9% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L6 | 7.750 | 1.6746 | 1.7057 | 24.8% | 19.5% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L8 | 10.308 | 2.0793 | 2.0879 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L2 | 0.721 | 0.8834 | 0.9037 | 74.3% | 57.1% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L4 | 1.416 | 1.2959 | 1.3233 | 49.5% | 37.1% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L6 | 2.111 | 1.6780 | 1.6986 | 24.8% | 18.6% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L8 | 2.807 | 2.0605 | 2.0825 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L2 | 3.828 | 0.8767 | 0.8960 | 74.5% | 56.9% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L4 | 7.553 | 1.2775 | 1.3060 | 49.7% | 37.2% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L6 | 11.278 | 1.6512 | 1.6749 | 24.8% | 18.8% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L8 | 15.003 | 2.0336 | 2.0608 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L2 | 0.026 | 1.3808 | 1.4593 | 74.2% | 46.1% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L4 | 0.052 | 1.7808 | 1.8083 | 49.5% | 30.5% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L6 | 0.077 | 2.1777 | 2.2041 | 24.7% | 15.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L8 | 0.102 | 2.5630 | 2.5923 | 0.0% | 0.0% |
