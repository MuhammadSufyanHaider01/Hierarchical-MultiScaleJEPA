# Available-method headline comparison

FLOPs are hardware-independent per-example operation counts; latency is hardware-dependent and is not reported.

| Method | Task | Exit | Selection | Test value | FLOPs/example |
| --- | --- | --- | --- | --- | --- |
| P0 — Original LatentWave (L8) | RF sensing | L8 | full-depth L8 | 0.7275857329368591 | 2806654976 |
| P0 — Original LatentWave (L8) | Jamming classification | L8 | full-depth L8 | 0.8680000305175781 | 2806647808 |
| P0 — Original LatentWave (L8) | Human activity sensing | L8 | full-depth L8 | 0.9916666746139526 | 10308160512 |
| P0 — Original LatentWave (L8) | Interference classification | L8 | full-depth L8 | 0.7014285326004028 | 2806645760 |
| P0 — Original LatentWave (L8) | NR positioning | L8 | full-depth L8 | 3.3183515071868896 | 15003026944 |
| P0 — Original LatentWave (L8) | DeepMIMO beam | L8 | full-depth L8 | 0.3524077236652374 | 102243840 |
| P2 — Hierarchical Multi-Scale JEPA | RF sensing | L2 | earliest within 5% of validation best | 0.7123818397521973 | 720939008 |
| P2 — Hierarchical Multi-Scale JEPA | Jamming classification | L2 | earliest within 5% of validation best | 0.8986667394638062 | 720931840 |
| P2 — Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | earliest within 5% of validation best | 0.9916666746139526 | 2634845184 |
| P2 — Hierarchical Multi-Scale JEPA | Interference classification | L2 | earliest within 5% of validation best | 0.7450000047683716 | 720929792 |
| P2 — Hierarchical Multi-Scale JEPA | NR positioning | L6 | earliest within 5% of validation best | 3.1671366691589355 | 11277960704 |
| P2 — Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | earliest within 5% of validation best | 0.4102822244167328 | 26353152 |
| P2-Reverse — Coarse-to-Fine JEPA | DeepMIMO beam | L2 | earliest within 5% of validation best | 0.4110841453075409 | 26353152 |
| P2-Reverse — Coarse-to-Fine JEPA | Human activity sensing | L2 | earliest within 5% of validation best | 0.9916666746139526 | 2634845184 |
| P2-Reverse — Coarse-to-Fine JEPA | Interference classification | L2 | earliest within 5% of validation best | 0.7549999952316284 | 720929792 |
| P2-Reverse — Coarse-to-Fine JEPA | Jamming classification | L2 | earliest within 5% of validation best | 0.8760000467300415 | 720931840 |
| P2-Reverse — Coarse-to-Fine JEPA | NR positioning | L4 | earliest within 5% of validation best | 3.346201181411743 | 7552894464 |
| P2-Reverse — Coarse-to-Fine JEPA | RF sensing | L4 | earliest within 5% of validation best | 0.7375330328941345 | 1416177664 |
