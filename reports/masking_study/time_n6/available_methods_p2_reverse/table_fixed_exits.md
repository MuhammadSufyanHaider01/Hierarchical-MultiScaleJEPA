# Fixed-exit downstream results

Validation results are shown only where available; test values are from the held-out split. FLOPs are per example through the listed exit and task head.

| Method | Task | Exit | Metric | Test | Validation | FLOPs/example |
| --- | --- | --- | --- | --- | --- | --- |
| P0 — Original LatentWave (L8) | RF sensing | L8 | macro_accuracy | 0.7275857329368591 |  | 2806654976 |
| P0 — Original LatentWave (L8) | Jamming classification | L8 | macro_accuracy | 0.8680000305175781 |  | 2806647808 |
| P0 — Original LatentWave (L8) | Human activity sensing | L8 | macro_accuracy | 0.9916666746139526 |  | 10308160512 |
| P0 — Original LatentWave (L8) | Interference classification | L8 | macro_accuracy | 0.7014285326004028 |  | 2806645760 |
| P0 — Original LatentWave (L8) | NR positioning | L8 | mean_distance_error | 3.3183515071868896 |  | 15003026944 |
| P0 — Original LatentWave (L8) | DeepMIMO beam | L8 | macro_accuracy | 0.3524077236652374 |  | 102243840 |
| P2 — Hierarchical Multi-Scale JEPA | RF sensing | L2 | macro_accuracy | 0.7123818397521973 | 0.7214015126228333 | 720939008 |
| P2 — Hierarchical Multi-Scale JEPA | RF sensing | L4 | macro_accuracy | 0.6960070729255676 | 0.715930163860321 | 1416177664 |
| P2 — Hierarchical Multi-Scale JEPA | RF sensing | L6 | macro_accuracy | 0.7241429686546326 | 0.7335072755813599 | 2111416320 |
| P2 — Hierarchical Multi-Scale JEPA | RF sensing | L8 | macro_accuracy | 0.6831163167953491 | 0.712942898273468 | 2806654976 |
| P2 — Hierarchical Multi-Scale JEPA | Jamming classification | L2 | macro_accuracy | 0.8986667394638062 | 0.8786667585372925 | 720931840 |
| P2 — Hierarchical Multi-Scale JEPA | Jamming classification | L4 | macro_accuracy | 0.9026666879653931 | 0.9013333320617676 | 1416170496 |
| P2 — Hierarchical Multi-Scale JEPA | Jamming classification | L6 | macro_accuracy | 0.8986667394638062 | 0.8946666717529297 | 2111409152 |
| P2 — Hierarchical Multi-Scale JEPA | Jamming classification | L8 | macro_accuracy | 0.8813333511352539 | 0.8813333511352539 | 2806647808 |
| P2 — Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | macro_accuracy | 0.9916666746139526 | 1.0 | 2634845184 |
| P2 — Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | macro_accuracy | 1.0 | 1.0 | 5192616960 |
| P2 — Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | macro_accuracy | 0.9916666746139526 | 1.0 | 7750388736 |
| P2 — Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | macro_accuracy | 0.9833333492279053 | 1.0 | 10308160512 |
| P2 — Hierarchical Multi-Scale JEPA | Interference classification | L2 | macro_accuracy | 0.7450000047683716 | 0.7228571176528931 | 720929792 |
| P2 — Hierarchical Multi-Scale JEPA | Interference classification | L4 | macro_accuracy | 0.7342857122421265 | 0.7257143259048462 | 1416168448 |
| P2 — Hierarchical Multi-Scale JEPA | Interference classification | L6 | macro_accuracy | 0.7328571081161499 | 0.7300000190734863 | 2111407104 |
| P2 — Hierarchical Multi-Scale JEPA | Interference classification | L8 | macro_accuracy | 0.7271428108215332 | 0.7107142806053162 | 2806645760 |
| P2 — Hierarchical Multi-Scale JEPA | NR positioning | L2 | mean_distance_error | 3.612790584564209 | 3.6256442070007324 | 3827828224 |
| P2 — Hierarchical Multi-Scale JEPA | NR positioning | L4 | mean_distance_error | 3.2616496086120605 | 3.338240385055542 | 7552894464 |
| P2 — Hierarchical Multi-Scale JEPA | NR positioning | L6 | mean_distance_error | 3.1671366691589355 | 3.144455909729004 | 11277960704 |
| P2 — Hierarchical Multi-Scale JEPA | NR positioning | L8 | mean_distance_error | 3.103607416152954 | 3.122920274734497 | 15003026944 |
| P2 — Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | macro_accuracy | 0.4102822244167328 | 0.42427608370780945 | 26353152 |
| P2 — Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | macro_accuracy | 0.3915551006793976 | 0.42557254433631897 | 51650048 |
| P2 — Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | macro_accuracy | 0.3822903335094452 | 0.41724875569343567 | 76946944 |
| P2 — Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | macro_accuracy | 0.3966853618621826 | 0.41071900725364685 | 102243840 |
| P2-Reverse — Coarse-to-Fine JEPA | DeepMIMO beam | L2 | macro_accuracy | 0.4110841453075409 | 0.42754626274108887 | 26353152 |
| P2-Reverse — Coarse-to-Fine JEPA | DeepMIMO beam | L4 | macro_accuracy | 0.42282748222351074 | 0.437482625246048 | 51650048 |
| P2-Reverse — Coarse-to-Fine JEPA | DeepMIMO beam | L6 | macro_accuracy | 0.3981703519821167 | 0.43382665514945984 | 76946944 |
| P2-Reverse — Coarse-to-Fine JEPA | DeepMIMO beam | L8 | macro_accuracy | 0.3875464200973511 | 0.4141084551811218 | 102243840 |
| P2-Reverse — Coarse-to-Fine JEPA | Human activity sensing | L2 | macro_accuracy | 0.9916666746139526 | 0.9916666746139526 | 2634845184 |
| P2-Reverse — Coarse-to-Fine JEPA | Human activity sensing | L4 | macro_accuracy | 1.0 | 1.0 | 5192616960 |
| P2-Reverse — Coarse-to-Fine JEPA | Human activity sensing | L6 | macro_accuracy | 1.0 | 1.0 | 7750388736 |
| P2-Reverse — Coarse-to-Fine JEPA | Human activity sensing | L8 | macro_accuracy | 0.9916666746139526 | 0.9916666746139526 | 10308160512 |
| P2-Reverse — Coarse-to-Fine JEPA | Interference classification | L2 | macro_accuracy | 0.7549999952316284 | 0.7442857027053833 | 720929792 |
| P2-Reverse — Coarse-to-Fine JEPA | Interference classification | L4 | macro_accuracy | 0.7492856979370117 | 0.7300000190734863 | 1416168448 |
| P2-Reverse — Coarse-to-Fine JEPA | Interference classification | L6 | macro_accuracy | 0.7278571128845215 | 0.7364285588264465 | 2111407104 |
| P2-Reverse — Coarse-to-Fine JEPA | Interference classification | L8 | macro_accuracy | 0.729285717010498 | 0.7257143259048462 | 2806645760 |
| P2-Reverse — Coarse-to-Fine JEPA | Jamming classification | L2 | macro_accuracy | 0.8760000467300415 | 0.8693333864212036 | 720931840 |
| P2-Reverse — Coarse-to-Fine JEPA | Jamming classification | L4 | macro_accuracy | 0.8893334269523621 | 0.8946666717529297 | 1416170496 |
| P2-Reverse — Coarse-to-Fine JEPA | Jamming classification | L6 | macro_accuracy | 0.8946666717529297 | 0.8853334188461304 | 2111409152 |
| P2-Reverse — Coarse-to-Fine JEPA | Jamming classification | L8 | macro_accuracy | 0.8666666746139526 | 0.8760000467300415 | 2806647808 |
| P2-Reverse — Coarse-to-Fine JEPA | NR positioning | L2 | mean_distance_error | 3.622772455215454 | 3.637192964553833 | 3827828224 |
| P2-Reverse — Coarse-to-Fine JEPA | NR positioning | L4 | mean_distance_error | 3.346201181411743 | 3.3487138748168945 | 7552894464 |
| P2-Reverse — Coarse-to-Fine JEPA | NR positioning | L6 | mean_distance_error | 3.333043336868286 | 3.3734281063079834 | 11277960704 |
| P2-Reverse — Coarse-to-Fine JEPA | NR positioning | L8 | mean_distance_error | 3.4456372261047363 | 3.393275499343872 | 15003026944 |
| P2-Reverse — Coarse-to-Fine JEPA | RF sensing | L2 | macro_accuracy | 0.7027408480644226 | 0.7367825508117676 | 720939008 |
| P2-Reverse — Coarse-to-Fine JEPA | RF sensing | L4 | macro_accuracy | 0.7375330328941345 | 0.7829279899597168 | 1416177664 |
| P2-Reverse — Coarse-to-Fine JEPA | RF sensing | L6 | macro_accuracy | 0.7568814158439636 | 0.7528130412101746 | 2111416320 |
| P2-Reverse — Coarse-to-Fine JEPA | RF sensing | L8 | macro_accuracy | 0.6912025809288025 | 0.7662854790687561 | 2806654976 |
