# Detailed batch-1 inference cost

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms | FLOPs saving | Latency saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.4529 | 0.6066 | 74.3% | 59.6% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.6796 | 0.7927 | 49.5% | 39.4% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | 2.111 | 0.9041 | 1.0494 | 24.8% | 19.4% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.1221 | 1.2569 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | 0.721 | 0.4636 | 0.5966 | 74.3% | 58.5% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | 1.416 | 0.6783 | 0.7795 | 49.5% | 39.3% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.8836 | 1.0219 | 24.8% | 21.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 2.807 | 1.1183 | 1.2613 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4702 | 0.7052 | 74.4% | 61.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | 5.193 | 0.7166 | 0.9296 | 49.6% | 41.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | 7.750 | 0.9719 | 1.1741 | 24.8% | 21.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | 10.308 | 1.2337 | 1.4207 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4514 | 0.5666 | 74.3% | 59.6% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | 1.416 | 0.6712 | 0.7938 | 49.5% | 39.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | 2.111 | 0.8860 | 1.0409 | 24.8% | 20.7% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | 2.807 | 1.1167 | 1.3175 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | 3.828 | 0.5620 | 0.7873 | 74.5% | 64.5% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | 7.553 | 0.9480 | 1.2454 | 49.7% | 40.1% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2453 | 1.4504 | 24.8% | 21.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5814 | 1.8182 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.5901 | 0.7504 | 74.2% | 50.7% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | 0.052 | 0.7879 | 0.9513 | 49.5% | 34.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | 0.077 | 1.0136 | 1.2005 | 24.7% | 15.3% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 0.102 | 1.1969 | 1.3786 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L2 | 0.721 | 0.5063 | 0.5239 | 74.3% | 56.5% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L4 | 1.416 | 0.7739 | 0.8237 | 49.5% | 33.5% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L6 | 2.111 | 0.9441 | 1.0313 | 24.8% | 18.9% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L8 | 2.807 | 1.1638 | 1.2671 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L2 | 0.721 | 0.4751 | 0.4966 | 74.3% | 59.8% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L4 | 1.416 | 0.7149 | 0.7781 | 49.5% | 39.5% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L6 | 2.111 | 0.9386 | 1.0185 | 24.8% | 20.5% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L8 | 2.807 | 1.1811 | 1.2827 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L2 | 2.635 | 0.4663 | 0.6237 | 74.4% | 61.0% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L4 | 5.193 | 0.7069 | 0.8740 | 49.6% | 40.9% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L6 | 7.750 | 0.9575 | 1.2173 | 24.8% | 19.9% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L8 | 10.308 | 1.1958 | 1.4901 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L2 | 0.721 | 0.4736 | 0.5000 | 74.3% | 59.5% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L4 | 1.416 | 0.7236 | 0.8257 | 49.5% | 38.1% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L6 | 2.111 | 0.9613 | 1.1099 | 24.8% | 17.8% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L8 | 2.807 | 1.1693 | 1.2758 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L2 | 3.828 | 0.5499 | 0.6950 | 74.5% | 64.0% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L4 | 7.553 | 0.8730 | 1.0426 | 49.7% | 42.9% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L6 | 11.278 | 1.2014 | 1.5147 | 24.8% | 21.4% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L8 | 15.003 | 1.5291 | 1.8344 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | 0.026 | 0.6172 | 0.7605 | 74.2% | 51.5% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L4 | 0.052 | 0.8417 | 0.9653 | 49.5% | 33.8% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L6 | 0.077 | 1.0553 | 1.1988 | 24.7% | 17.0% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L8 | 0.102 | 1.2715 | 1.4212 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.8352 | 2.5321 | 74.3% | 52.4% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 1.1986 | 2.7153 | 49.5% | 31.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 1.4615 | 2.7603 | 24.8% | 16.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.7530 | 3.0962 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.9171 | 2.5556 | 74.3% | 47.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | 1.416 | 1.2106 | 2.7608 | 49.5% | 30.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | 2.111 | 1.4744 | 2.8694 | 24.8% | 15.7% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 2.807 | 1.7492 | 3.0914 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.9278 | 2.6623 | 74.4% | 55.3% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 5.193 | 1.2912 | 2.9072 | 49.6% | 37.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | 7.750 | 1.6255 | 3.3916 | 24.8% | 21.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | 10.308 | 2.0746 | 3.6463 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.9485 | 2.5553 | 74.3% | 45.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | 1.416 | 1.2102 | 2.6611 | 49.5% | 30.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | 2.111 | 1.4776 | 2.9724 | 24.8% | 15.5% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | 2.807 | 1.7494 | 3.0894 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | 3.828 | 0.9660 | 2.7893 | 74.5% | 61.4% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | 7.553 | 1.5646 | 3.1305 | 49.7% | 37.4% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 2.1171 | 3.6676 | 24.8% | 15.3% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 2.5004 | 4.0001 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 1.1577 | 4.7483 | 74.2% | 48.1% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 1.5087 | 4.8161 | 49.5% | 32.3% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | 0.077 | 1.8778 | 4.9617 | 24.7% | 15.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | 0.102 | 2.2290 | 5.2559 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L2 | 0.721 | 1.2731 | 1.2851 | 74.3% | 61.4% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L4 | 1.416 | 1.9556 | 1.9648 | 49.5% | 40.7% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L6 | 2.111 | 2.6314 | 2.6440 | 24.8% | 20.3% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L8 | 2.807 | 3.3005 | 3.3198 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L2 | 0.721 | 1.2789 | 1.2895 | 74.3% | 61.6% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L4 | 1.416 | 1.9629 | 1.9778 | 49.5% | 41.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L6 | 2.111 | 2.6396 | 2.6545 | 24.8% | 20.7% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L8 | 2.807 | 3.3285 | 3.3395 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L2 | 2.635 | 1.2231 | 1.2333 | 74.4% | 60.7% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L4 | 5.193 | 1.8496 | 1.8587 | 49.6% | 40.5% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L6 | 7.750 | 2.4784 | 2.4937 | 24.8% | 20.3% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L8 | 10.308 | 3.1105 | 3.1168 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L2 | 0.721 | 1.2704 | 1.2818 | 74.3% | 61.6% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L4 | 1.416 | 1.9519 | 1.9641 | 49.5% | 40.9% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L6 | 2.111 | 2.6354 | 2.6451 | 24.8% | 20.3% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L8 | 2.807 | 3.3050 | 3.3219 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L2 | 3.828 | 1.2390 | 1.2487 | 74.5% | 59.9% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L4 | 7.553 | 1.8560 | 1.8669 | 49.7% | 39.9% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L6 | 11.278 | 2.4762 | 2.4949 | 24.8% | 19.8% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L8 | 15.003 | 3.0884 | 3.1045 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L2 | 0.026 | 1.4456 | 1.4732 | 74.2% | 56.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L4 | 0.052 | 2.0724 | 2.0989 | 49.5% | 37.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L6 | 0.077 | 2.6677 | 2.7023 | 24.7% | 18.9% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L8 | 0.102 | 3.2881 | 3.3197 | 0.0% | 0.0% |
