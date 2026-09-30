# Detailed batch-1 inference cost

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms | FLOPs saving | Latency saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.4533 | 0.4842 | 74.3% | 59.7% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.6903 | 0.7766 | 49.5% | 38.7% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | 2.111 | 0.8981 | 0.9808 | 24.8% | 20.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.1259 | 1.2477 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | 0.721 | 0.4559 | 0.4861 | 74.3% | 59.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | 1.416 | 0.6809 | 0.7705 | 49.5% | 39.1% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.9049 | 1.0111 | 24.8% | 19.1% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 2.807 | 1.1180 | 1.2667 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4585 | 0.6210 | 74.4% | 61.6% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | 5.193 | 0.6990 | 0.8777 | 49.6% | 41.5% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | 7.750 | 0.9481 | 1.1484 | 24.8% | 20.6% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | 10.308 | 1.1939 | 1.3596 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4493 | 0.4954 | 74.3% | 59.8% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | 1.416 | 0.6744 | 0.7613 | 49.5% | 39.7% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | 2.111 | 0.9005 | 1.0228 | 24.8% | 19.5% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | 2.807 | 1.1184 | 1.2317 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | 3.828 | 0.5577 | 0.7190 | 74.5% | 64.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | 7.553 | 0.8867 | 1.0755 | 49.7% | 43.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2147 | 1.3986 | 24.8% | 21.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5560 | 1.7544 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.5903 | 0.7066 | 74.2% | 51.7% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | 0.052 | 0.8039 | 0.9136 | 49.5% | 34.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | 0.077 | 1.0210 | 1.1488 | 24.7% | 16.5% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 0.102 | 1.2222 | 1.2982 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L2 | 0.721 | 0.4453 | 0.4697 | 74.3% | 59.3% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L4 | 1.416 | 0.6706 | 0.7150 | 49.5% | 38.7% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L6 | 2.111 | 0.8748 | 0.9622 | 24.8% | 20.0% |
| P1 — Same-scale Multi-depth JEPA | RF sensing | L8 | 2.807 | 1.0941 | 1.2155 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L2 | 0.721 | 0.4554 | 0.4805 | 74.3% | 58.8% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L4 | 1.416 | 0.6757 | 0.7234 | 49.5% | 38.8% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L6 | 2.111 | 0.8852 | 0.9649 | 24.8% | 19.9% |
| P1 — Same-scale Multi-depth JEPA | Jamming classification | L8 | 2.807 | 1.1047 | 1.1961 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L2 | 2.635 | 0.4445 | 0.4498 | 74.4% | 62.4% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L4 | 5.193 | 0.6872 | 0.7957 | 49.6% | 41.9% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L6 | 7.750 | 0.9297 | 1.1955 | 24.8% | 21.4% |
| P1 — Same-scale Multi-depth JEPA | Human activity sensing | L8 | 10.308 | 1.1826 | 1.4365 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L2 | 0.721 | 0.4480 | 0.4726 | 74.3% | 59.5% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L4 | 1.416 | 0.6692 | 0.7237 | 49.5% | 39.5% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L6 | 2.111 | 0.8828 | 0.9540 | 24.8% | 20.2% |
| P1 — Same-scale Multi-depth JEPA | Interference classification | L8 | 2.807 | 1.1057 | 1.1944 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L2 | 3.828 | 0.5411 | 0.5403 | 74.5% | 64.4% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L4 | 7.553 | 0.8697 | 0.9019 | 49.7% | 42.8% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L6 | 11.278 | 1.1907 | 1.4632 | 24.8% | 21.7% |
| P1 — Same-scale Multi-depth JEPA | NR positioning | L8 | 15.003 | 1.5204 | 1.7843 | 0.0% | 0.0% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L2 | 0.026 | 0.6164 | 0.6534 | 74.2% | 52.9% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L4 | 0.052 | 0.8457 | 0.9367 | 49.5% | 35.4% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L6 | 0.077 | 1.0978 | 1.2236 | 24.7% | 16.1% |
| P1 — Same-scale Multi-depth JEPA | DeepMIMO beam | L8 | 0.102 | 1.3086 | 1.3859 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.5170 | 0.7086 | 74.3% | 56.9% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 0.7326 | 0.8937 | 49.5% | 38.9% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 0.9761 | 1.1960 | 24.8% | 18.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.1988 | 1.4799 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.4821 | 0.5706 | 74.3% | 59.5% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | 1.416 | 0.7376 | 0.8430 | 49.5% | 38.1% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | 2.111 | 0.9890 | 1.2134 | 24.8% | 17.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 2.807 | 1.1914 | 1.4280 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.4931 | 0.6314 | 74.4% | 62.5% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 5.193 | 0.7665 | 1.0774 | 49.6% | 41.7% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | 7.750 | 1.0181 | 1.3125 | 24.8% | 22.5% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | 10.308 | 1.3140 | 2.3789 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.5121 | 0.6582 | 74.3% | 57.9% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | 1.416 | 0.7602 | 0.9142 | 49.5% | 37.4% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | 2.111 | 0.9814 | 1.2573 | 24.8% | 19.2% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | 2.807 | 1.2151 | 1.7694 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | 3.828 | 0.6167 | 0.9022 | 74.5% | 63.2% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | 7.553 | 0.9776 | 1.2880 | 49.7% | 41.7% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.3338 | 2.4615 | 24.8% | 20.4% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.6759 | 2.7483 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.6414 | 0.7627 | 74.2% | 51.1% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 0.8844 | 1.0191 | 49.5% | 32.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | 0.077 | 1.0857 | 1.2107 | 24.7% | 17.2% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | 0.102 | 1.3115 | 1.5230 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L2 | 0.721 | 1.2471 | 1.2593 | 74.3% | 61.4% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L4 | 1.416 | 1.9101 | 1.9214 | 49.5% | 40.8% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L6 | 2.111 | 2.5769 | 2.5938 | 24.8% | 20.2% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | RF sensing | L8 | 2.807 | 3.2274 | 3.2449 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L2 | 0.721 | 1.2533 | 1.2643 | 74.3% | 61.5% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L4 | 1.416 | 1.9184 | 1.9334 | 49.5% | 41.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L6 | 2.111 | 2.5819 | 2.5981 | 24.8% | 20.6% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Jamming classification | L8 | 2.807 | 3.2528 | 3.2569 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L2 | 2.635 | 1.1999 | 1.2101 | 74.4% | 60.4% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L4 | 5.193 | 1.8140 | 1.8255 | 49.6% | 40.2% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L6 | 7.750 | 2.4230 | 2.4384 | 24.8% | 20.1% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Human activity sensing | L8 | 10.308 | 3.0324 | 3.0479 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L2 | 0.721 | 1.2507 | 1.2613 | 74.3% | 61.4% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L4 | 1.416 | 1.9190 | 1.9285 | 49.5% | 40.8% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L6 | 2.111 | 2.5835 | 2.5951 | 24.8% | 20.3% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | Interference classification | L8 | 2.807 | 3.2403 | 3.2641 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L2 | 3.828 | 1.2181 | 1.2285 | 74.5% | 59.9% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L4 | 7.553 | 1.8313 | 1.8449 | 49.7% | 39.6% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L6 | 11.278 | 2.4354 | 2.4493 | 24.8% | 19.7% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | NR positioning | L8 | 15.003 | 3.0338 | 3.0496 | 0.0% | 0.0% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L2 | 0.026 | 1.4228 | 1.4517 | 74.2% | 56.3% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L4 | 0.052 | 2.0450 | 2.0886 | 49.5% | 37.2% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L6 | 0.077 | 2.6288 | 2.6589 | 24.7% | 19.3% |
| P2-Reverse — Coarse-to-fine Hierarchical JEPA | DeepMIMO beam | L8 | 0.102 | 3.2590 | 3.3209 | 0.0% | 0.0% |
