# Batch-1 inference cost

| Approach | Task | Exit | FLOPs (B) | Mean ms | P95 ms | FLOPs saving | Latency saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 — Original LatentWave | RF sensing | L8 | 2.807 | 1.1175 | 1.2204 | 0.0% | 0.0% |
| B0 — Original LatentWave | Jamming classification | L8 | 2.807 | 1.1267 | 1.2456 | 0.0% | 0.0% |
| B0 — Original LatentWave | Human activity sensing | L8 | 10.308 | 1.1870 | 1.3541 | 0.0% | 0.0% |
| B0 — Original LatentWave | Interference classification | L8 | 2.807 | 1.1186 | 1.2273 | 0.0% | 0.0% |
| B0 — Original LatentWave | NR positioning | L8 | 15.003 | 1.5272 | 1.6933 | 0.0% | 0.0% |
| B0 — Original LatentWave | DeepMIMO beam | L8 | 0.102 | 1.2205 | 1.3524 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | 0.721 | 0.4502 | 0.5238 | 74.3% | 59.7% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | 1.416 | 0.6789 | 0.7893 | 49.5% | 39.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | 2.111 | 0.9145 | 0.9972 | 24.8% | 18.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | 2.807 | 1.1175 | 1.2204 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | 0.721 | 0.4564 | 0.4830 | 74.3% | 59.5% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | 1.416 | 0.6853 | 0.7551 | 49.5% | 39.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | 2.111 | 0.8979 | 0.9835 | 24.8% | 20.3% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | 2.807 | 1.1267 | 1.2456 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | 2.635 | 0.4521 | 0.5502 | 74.4% | 61.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | 5.193 | 0.7012 | 0.8600 | 49.6% | 40.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | 7.750 | 0.9470 | 1.1033 | 24.8% | 20.2% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | 10.308 | 1.1870 | 1.3541 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | 0.721 | 0.4463 | 0.4776 | 74.3% | 60.1% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | 1.416 | 0.6719 | 0.7731 | 49.5% | 39.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | 2.111 | 0.8998 | 1.0251 | 24.8% | 19.6% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | 2.807 | 1.1186 | 1.2273 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | 3.828 | 0.5451 | 0.6899 | 74.5% | 64.3% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | 7.553 | 0.8722 | 1.0278 | 49.7% | 42.9% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | 11.278 | 1.2051 | 1.3760 | 24.8% | 21.1% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | 15.003 | 1.5272 | 1.6933 | 0.0% | 0.0% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | 0.026 | 0.5806 | 0.6564 | 74.2% | 52.4% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | 0.052 | 0.7985 | 0.9120 | 49.5% | 34.6% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | 0.077 | 0.9947 | 1.0930 | 24.7% | 18.5% |
| P0 — LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | 0.102 | 1.2205 | 1.3524 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | 0.721 | 0.4524 | 0.4835 | 74.3% | 59.2% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | 1.416 | 0.6872 | 0.7562 | 49.5% | 38.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | 2.111 | 0.9098 | 0.9708 | 24.8% | 17.9% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | 2.807 | 1.1080 | 1.1874 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | 0.721 | 0.4580 | 0.4683 | 74.3% | 59.3% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | 1.416 | 0.6880 | 0.8003 | 49.5% | 38.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | 2.111 | 0.9102 | 1.0096 | 24.8% | 19.1% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | 2.807 | 1.1244 | 1.2091 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | 2.635 | 0.4588 | 0.5644 | 74.4% | 61.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | 5.193 | 0.6998 | 0.8424 | 49.6% | 41.5% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | 7.750 | 0.9446 | 1.0969 | 24.8% | 21.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | 10.308 | 1.1955 | 1.3379 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | 0.721 | 0.4588 | 0.5002 | 74.3% | 59.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | 1.416 | 0.6708 | 0.7306 | 49.5% | 40.1% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | 2.111 | 0.9171 | 0.9938 | 24.8% | 18.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | 2.807 | 1.1191 | 1.2016 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | 3.828 | 0.5494 | 0.6407 | 74.5% | 64.3% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | 7.553 | 0.8822 | 1.0498 | 49.7% | 42.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | 11.278 | 1.2052 | 1.3663 | 24.8% | 21.6% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | 15.003 | 1.5370 | 1.6954 | 0.0% | 0.0% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | 0.026 | 0.5830 | 0.6852 | 74.2% | 52.8% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | 0.052 | 0.8035 | 0.9121 | 49.5% | 34.9% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | 0.077 | 1.0219 | 1.1496 | 24.7% | 17.3% |
| P2 — Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | 0.102 | 1.2350 | 1.3468 | 0.0% | 0.0% |
| C0 — CNN Original L8 JEPA Objective | RF sensing | L2 | 1.727 | 1.6450 | 1.7579 | 1.9% | -0.0% |
| C0 — CNN Original L8 JEPA Objective | RF sensing | L4 | 1.732 | 1.6568 | 1.8132 | 1.6% | -0.8% |
| C0 — CNN Original L8 JEPA Objective | RF sensing | L6 | 1.742 | 1.6358 | 1.7402 | 1.1% | 0.5% |
| C0 — CNN Original L8 JEPA Objective | RF sensing | L8 | 1.761 | 1.6445 | 1.8048 | 0.0% | 0.0% |
| C0 — CNN Original L8 JEPA Objective | Jamming classification | L2 | 1.727 | 1.6533 | 1.8070 | 1.9% | -0.8% |
| C0 — CNN Original L8 JEPA Objective | Jamming classification | L4 | 1.732 | 1.6406 | 1.7818 | 1.6% | -0.1% |
| C0 — CNN Original L8 JEPA Objective | Jamming classification | L6 | 1.742 | 1.6279 | 1.7566 | 1.1% | 0.7% |
| C0 — CNN Original L8 JEPA Objective | Jamming classification | L8 | 1.761 | 1.6395 | 1.8061 | 0.0% | 0.0% |
| C0 — CNN Original L8 JEPA Objective | Human activity sensing | L2 | 5.181 | 1.6116 | 1.7585 | 1.9% | 0.3% |
| C0 — CNN Original L8 JEPA Objective | Human activity sensing | L4 | 5.196 | 1.6192 | 1.7754 | 1.6% | -0.1% |
| C0 — CNN Original L8 JEPA Objective | Human activity sensing | L6 | 5.225 | 1.6027 | 1.7706 | 1.1% | 0.9% |
| C0 — CNN Original L8 JEPA Objective | Human activity sensing | L8 | 5.282 | 1.6172 | 1.7762 | 0.0% | 0.0% |
| C0 — CNN Original L8 JEPA Objective | Interference classification | L2 | 1.727 | 1.6446 | 1.7034 | 1.9% | 10.1% |
| C0 — CNN Original L8 JEPA Objective | Interference classification | L4 | 1.732 | 1.8303 | 1.9041 | 1.6% | -0.1% |
| C0 — CNN Original L8 JEPA Objective | Interference classification | L6 | 1.742 | 1.8439 | 1.9004 | 1.1% | -0.8% |
| C0 — CNN Original L8 JEPA Objective | Interference classification | L8 | 1.761 | 1.8284 | 1.9046 | 0.0% | 0.0% |
| C0 — CNN Original L8 JEPA Objective | NR positioning | L2 | 6.908 | 1.6639 | 1.8305 | 1.9% | -0.4% |
| C0 — CNN Original L8 JEPA Objective | NR positioning | L4 | 6.928 | 1.6594 | 1.8153 | 1.6% | -0.1% |
| C0 — CNN Original L8 JEPA Objective | NR positioning | L6 | 6.966 | 1.6433 | 1.7965 | 1.1% | 0.8% |
| C0 — CNN Original L8 JEPA Objective | NR positioning | L8 | 7.043 | 1.6570 | 1.8086 | 0.0% | 0.0% |
| C0 — CNN Original L8 JEPA Objective | DeepMIMO beam | L2 | 0.080 | 1.6540 | 1.8170 | 45.8% | -1.4% |
| C0 — CNN Original L8 JEPA Objective | DeepMIMO beam | L4 | 0.090 | 1.6463 | 1.8142 | 39.2% | -1.0% |
| C0 — CNN Original L8 JEPA Objective | DeepMIMO beam | L6 | 0.109 | 1.6551 | 1.8244 | 26.1% | -1.5% |
| C0 — CNN Original L8 JEPA Objective | DeepMIMO beam | L8 | 0.147 | 1.6307 | 1.7901 | 0.0% | 0.0% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | RF sensing | L2 | 1.727 | 1.7124 | 1.8884 | 1.9% | -0.1% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | RF sensing | L4 | 1.732 | 1.7056 | 1.8553 | 1.6% | 0.3% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | RF sensing | L6 | 1.742 | 1.6888 | 1.8389 | 1.1% | 1.2% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | RF sensing | L8 | 1.761 | 1.7102 | 1.8757 | 0.0% | 0.0% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Jamming classification | L2 | 1.727 | 1.6967 | 1.8134 | 1.9% | 0.1% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Jamming classification | L4 | 1.732 | 1.7075 | 1.8769 | 1.6% | -0.6% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Jamming classification | L6 | 1.742 | 1.6997 | 1.8782 | 1.1% | -0.1% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Jamming classification | L8 | 1.761 | 1.6979 | 1.8646 | 0.0% | 0.0% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Human activity sensing | L2 | 5.181 | 1.6712 | 1.8222 | 1.9% | 1.0% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Human activity sensing | L4 | 5.196 | 1.6850 | 1.8463 | 1.6% | 0.2% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Human activity sensing | L6 | 5.225 | 1.6601 | 1.8273 | 1.1% | 1.6% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Human activity sensing | L8 | 5.282 | 1.6877 | 1.8569 | 0.0% | 0.0% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Interference classification | L2 | 1.727 | 1.7097 | 1.8489 | 1.9% | -0.0% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Interference classification | L4 | 1.732 | 1.7161 | 1.8839 | 1.6% | -0.4% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Interference classification | L6 | 1.742 | 1.6983 | 1.8648 | 1.1% | 0.6% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | Interference classification | L8 | 1.761 | 1.7091 | 1.8708 | 0.0% | 0.0% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | NR positioning | L2 | 6.908 | 1.7142 | 1.8771 | 1.9% | 9.0% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | NR positioning | L4 | 6.928 | 1.7400 | 1.9296 | 1.6% | 7.6% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | NR positioning | L6 | 6.966 | 1.8785 | 2.0018 | 1.1% | 0.3% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | NR positioning | L8 | 7.043 | 1.8840 | 2.0077 | 0.0% | 0.0% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | DeepMIMO beam | L2 | 0.080 | 1.6979 | 1.8591 | 45.8% | -1.3% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | DeepMIMO beam | L4 | 0.090 | 1.7112 | 1.8795 | 39.2% | -2.1% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | DeepMIMO beam | L6 | 0.109 | 1.6901 | 1.8666 | 26.1% | -0.8% |
| C2 — CNN Mask-relative Fine-to-coarse JEPA | DeepMIMO beam | L8 | 0.147 | 1.6768 | 1.8478 | 0.0% | 0.0% |
