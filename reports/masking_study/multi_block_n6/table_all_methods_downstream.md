# All-method headline downstream comparison

FLOPs are per example; latency is hardware-dependent and not reported here.

| Method | Task | Exit | Selection | Test value | FLOPs/example |
| --- | --- | --- | --- | --- | --- |
| Original LatentWave | RF sensing | L8 | full-depth L8 | 0.694596529006958 | 2806654976 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | earliest within 5% of validation best | 0.6330236196517944 | 1416177664 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | earliest within 5% of validation best | 0.7408782839775085 | 2111416320 |
| Original LatentWave | Jamming classification | L8 | full-depth L8 | 0.8799999952316284 | 2806647808 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | earliest within 5% of validation best | 0.8799999952316284 | 2806647808 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | earliest within 5% of validation best | 0.8653333187103271 | 2111409152 |
| Original LatentWave | Human activity sensing | L8 | full-depth L8 | 0.9833333492279053 | 10308160512 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | earliest within 5% of validation best | 0.9583333730697632 | 2634845184 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | earliest within 5% of validation best | 1.0 | 2634845184 |
| Original LatentWave | Interference classification | L8 | full-depth L8 | 0.7478571534156799 | 2806645760 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | earliest within 5% of validation best | 0.7164285778999329 | 720929792 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | earliest within 5% of validation best | 0.7528571486473083 | 720929792 |
| Original LatentWave | NR positioning | L8 | full-depth L8 | 3.0250096321105957 | 15003026944 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | earliest within 5% of validation best | 3.103659152984619 | 11277960704 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | earliest within 5% of validation best | 2.760051965713501 | 11277960704 |
| Original LatentWave | DeepMIMO beam | L8 | full-depth L8 | 0.457409143447876 | 102243840 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | earliest within 5% of validation best | 0.42720386385917664 | 26353152 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | earliest within 5% of validation best | 0.43465596437454224 | 51650048 |
| P1 Same-scale Intermediate Supervision | DeepMIMO beam | L2 | earliest within 5% of validation best | 0.4338372051715851 | 26353152 |
| P1 Same-scale Intermediate Supervision | Human activity sensing | L2 | earliest within 5% of validation best | 0.9916666746139526 | 2634845184 |
| P1 Same-scale Intermediate Supervision | Interference classification | L2 | earliest within 5% of validation best | 0.7421428561210632 | 720929792 |
| P1 Same-scale Intermediate Supervision | Jamming classification | L4 | earliest within 5% of validation best | 0.8586666584014893 | 1416170496 |
| P1 Same-scale Intermediate Supervision | NR positioning | L6 | earliest within 5% of validation best | 3.0587894916534424 | 11277960704 |
| P1 Same-scale Intermediate Supervision | RF sensing | L4 | earliest within 5% of validation best | 0.675300121307373 | 1416177664 |
| P2-Reverse Coarse-to-Fine JEPA | DeepMIMO beam | L2 | earliest within 5% of validation best | 0.4084385633468628 | 26353152 |
| P2-Reverse Coarse-to-Fine JEPA | Human activity sensing | L2 | earliest within 5% of validation best | 0.9750000834465027 | 2634845184 |
| P2-Reverse Coarse-to-Fine JEPA | Interference classification | L2 | earliest within 5% of validation best | 0.7228571176528931 | 720929792 |
| P2-Reverse Coarse-to-Fine JEPA | Jamming classification | L6 | earliest within 5% of validation best | 0.8573333621025085 | 2111409152 |
| P2-Reverse Coarse-to-Fine JEPA | NR positioning | L8 | earliest within 5% of validation best | 2.873737335205078 | 15003026944 |
| P2-Reverse Coarse-to-Fine JEPA | RF sensing | L6 | earliest within 5% of validation best | 0.718097448348999 | 2111416320 |
