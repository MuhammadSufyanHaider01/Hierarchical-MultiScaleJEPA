# Available-method headline comparison

FLOPs are hardware-independent per-example operation counts; latency is hardware-dependent and is not reported.

| Method | Task | Exit | Selection | Test value | FLOPs/example |
| --- | --- | --- | --- | --- | --- |
| P0 — Original LatentWave (L8) | RF sensing | L8 | full-depth L8 | 0.7896893620491028 | 2806654976 |
| P0 — Original LatentWave (L8) | Jamming classification | L8 | full-depth L8 | 0.8546666502952576 | 2806647808 |
| P0 — Original LatentWave (L8) | Human activity sensing | L8 | full-depth L8 | 0.9666666984558105 | 10308160512 |
| P0 — Original LatentWave (L8) | Interference classification | L8 | full-depth L8 | 0.7128571271896362 | 2806645760 |
| P0 — Original LatentWave (L8) | NR positioning | L8 | full-depth L8 | 2.93912935256958 | 15003026944 |
| P0 — Original LatentWave (L8) | DeepMIMO beam | L8 | full-depth L8 | 0.3909437656402588 | 102243840 |
| P2 — Hierarchical Multi-Scale JEPA | RF sensing | L6 | earliest within 5% of validation best | 0.7540153861045837 | 2111416320 |
| P2 — Hierarchical Multi-Scale JEPA | Jamming classification | L2 | earliest within 5% of validation best | 0.8840000033378601 | 720931840 |
| P2 — Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | earliest within 5% of validation best | 0.9750000834465027 | 2634845184 |
| P2 — Hierarchical Multi-Scale JEPA | Interference classification | L2 | earliest within 5% of validation best | 0.7400000095367432 | 720929792 |
| P2 — Hierarchical Multi-Scale JEPA | NR positioning | L8 | earliest within 5% of validation best | 2.759456157684326 | 15003026944 |
| P2 — Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | earliest within 5% of validation best | 0.44455403089523315 | 26353152 |
| P2-Reverse — Coarse-to-Fine JEPA | DeepMIMO beam | L2 | earliest within 5% of validation best | 0.4441002905368805 | 26353152 |
| P2-Reverse — Coarse-to-Fine JEPA | Human activity sensing | L2 | earliest within 5% of validation best | 0.9750000834465027 | 2634845184 |
| P2-Reverse — Coarse-to-Fine JEPA | Interference classification | L2 | earliest within 5% of validation best | 0.7335714101791382 | 720929792 |
| P2-Reverse — Coarse-to-Fine JEPA | Jamming classification | L2 | earliest within 5% of validation best | 0.8720000982284546 | 720931840 |
| P2-Reverse — Coarse-to-Fine JEPA | NR positioning | L4 | earliest within 5% of validation best | 2.951751947402954 | 7552894464 |
| P2-Reverse — Coarse-to-Fine JEPA | RF sensing | L4 | earliest within 5% of validation best | 0.7751945853233337 | 1416177664 |
