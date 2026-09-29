# Detailed frozen-exit downstream metrics

Validation is used for head and exit selection; test is evaluated after selection. FLOPs are per example for the encoder through the listed exit and that task's head.

| Method | Task | Exit | Metric | Test | Validation | Best epoch | FLOPs/example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L2 | macro_accuracy | 0.6086896657943726 | 0.6500607132911682 | 110 | 720939008 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L4 | macro_accuracy | 0.6330236196517944 | 0.717022716999054 | 95 | 1416177664 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L6 | macro_accuracy | 0.6698223948478699 | 0.7049547433853149 | 75 | 2111416320 |
| LatentWave with Post-hoc Frozen Early Exits | RF sensing | L8 | macro_accuracy | 0.694596529006958 | 0.7107575535774231 | 163 | 2806654976 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L2 | macro_accuracy | 0.8106666803359985 | 0.7986667156219482 | 144 | 720931840 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L4 | macro_accuracy | 0.8053333163261414 | 0.7879999876022339 | 74 | 1416170496 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L6 | macro_accuracy | 0.8186666369438171 | 0.8320000171661377 | 86 | 2111409152 |
| LatentWave with Post-hoc Frozen Early Exits | Jamming classification | L8 | macro_accuracy | 0.8799999952316284 | 0.8773333430290222 | 21 | 2806647808 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L2 | macro_accuracy | 0.9583333730697632 | 0.9916666746139526 | 24 | 2634845184 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L4 | macro_accuracy | 0.9666666984558105 | 1.0 | 24 | 5192616960 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L6 | macro_accuracy | 1.0 | 1.0 | 35 | 7750388736 |
| LatentWave with Post-hoc Frozen Early Exits | Human activity sensing | L8 | macro_accuracy | 0.9833333492279053 | 1.0 | 26 | 10308160512 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L2 | macro_accuracy | 0.7164285778999329 | 0.7242857217788696 | 40 | 720929792 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L4 | macro_accuracy | 0.7407143115997314 | 0.7350000143051147 | 43 | 1416168448 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L6 | macro_accuracy | 0.7521428465843201 | 0.7385714054107666 | 32 | 2111407104 |
| LatentWave with Post-hoc Frozen Early Exits | Interference classification | L8 | macro_accuracy | 0.7478571534156799 | 0.7378571629524231 | 34 | 2806645760 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L2 | mean_distance_error | 3.7335476875305176 | 3.6436097621917725 | 17 | 3827828224 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L4 | mean_distance_error | 3.3069732189178467 | 3.3349170684814453 | 35 | 7552894464 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L6 | mean_distance_error | 3.103659152984619 | 3.107639789581299 | 93 | 11277960704 |
| LatentWave with Post-hoc Frozen Early Exits | NR positioning | L8 | mean_distance_error | 3.0250096321105957 | 3.0388526916503906 | 59 | 15003026944 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L2 | macro_accuracy | 0.42720386385917664 | 0.43974044919013977 | 42 | 26353152 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L4 | macro_accuracy | 0.4214315414428711 | 0.4353559613227844 | 17 | 51650048 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L6 | macro_accuracy | 0.43013834953308105 | 0.43822842836380005 | 24 | 76946944 |
| LatentWave with Post-hoc Frozen Early Exits | DeepMIMO beam | L8 | macro_accuracy | 0.457409143447876 | 0.44694578647613525 | 54 | 102243840 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L2 | macro_accuracy | 0.7224922180175781 | 0.6799823641777039 | 80 | 720939008 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L4 | macro_accuracy | 0.6553102731704712 | 0.6805490255355835 | 50 | 1416177664 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L6 | macro_accuracy | 0.7408782839775085 | 0.7611322402954102 | 98 | 2111416320 |
| Proposed Hierarchical Multi-Scale JEPA | RF sensing | L8 | macro_accuracy | 0.7225880026817322 | 0.7649297118186951 | 78 | 2806654976 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L2 | macro_accuracy | 0.8266667127609253 | 0.8400000333786011 | 51 | 720931840 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L4 | macro_accuracy | 0.8400000333786011 | 0.8413333296775818 | 52 | 1416170496 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L6 | macro_accuracy | 0.8653333187103271 | 0.8693333864212036 | 80 | 2111409152 |
| Proposed Hierarchical Multi-Scale JEPA | Jamming classification | L8 | macro_accuracy | 0.8826667070388794 | 0.8960000276565552 | 47 | 2806647808 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L2 | macro_accuracy | 1.0 | 0.9916666746139526 | 42 | 2634845184 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L4 | macro_accuracy | 1.0 | 1.0 | 19 | 5192616960 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L6 | macro_accuracy | 1.0 | 0.9916666746139526 | 17 | 7750388736 |
| Proposed Hierarchical Multi-Scale JEPA | Human activity sensing | L8 | macro_accuracy | 1.0 | 1.0 | 26 | 10308160512 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L2 | macro_accuracy | 0.7528571486473083 | 0.7457143068313599 | 38 | 720929792 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L4 | macro_accuracy | 0.7528571486473083 | 0.7528571486473083 | 25 | 1416168448 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L6 | macro_accuracy | 0.75 | 0.7528571486473083 | 18 | 2111407104 |
| Proposed Hierarchical Multi-Scale JEPA | Interference classification | L8 | macro_accuracy | 0.7585713863372803 | 0.7557142972946167 | 23 | 2806645760 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L2 | mean_distance_error | 3.2955331802368164 | 3.2839748859405518 | 43 | 3827828224 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L4 | mean_distance_error | 3.101229190826416 | 3.0609498023986816 | 89 | 7552894464 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L6 | mean_distance_error | 2.760051965713501 | 2.7590718269348145 | 62 | 11277960704 |
| Proposed Hierarchical Multi-Scale JEPA | NR positioning | L8 | mean_distance_error | 2.7252769470214844 | 2.7710092067718506 | 52 | 15003026944 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L2 | macro_accuracy | 0.4267839789390564 | 0.44238680601119995 | 27 | 26353152 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L4 | macro_accuracy | 0.43465596437454224 | 0.4686080515384674 | 32 | 51650048 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L6 | macro_accuracy | 0.4145088195800781 | 0.44565439224243164 | 19 | 76946944 |
| Proposed Hierarchical Multi-Scale JEPA | DeepMIMO beam | L8 | macro_accuracy | 0.4445035755634308 | 0.45818132162094116 | 28 | 102243840 |
| P1 Same-scale Intermediate Supervision | DeepMIMO beam | L2 | macro_accuracy | 0.4338372051715851 | 0.4696016013622284 | 72 | 26353152 |
| P1 Same-scale Intermediate Supervision | DeepMIMO beam | L4 | macro_accuracy | 0.43197140097618103 | 0.45724374055862427 | 33 | 51650048 |
| P1 Same-scale Intermediate Supervision | DeepMIMO beam | L6 | macro_accuracy | 0.43679121136665344 | 0.44491609930992126 | 72 | 76946944 |
| P1 Same-scale Intermediate Supervision | DeepMIMO beam | L8 | macro_accuracy | 0.4231424033641815 | 0.4271644949913025 | 48 | 102243840 |
| P1 Same-scale Intermediate Supervision | Human activity sensing | L2 | macro_accuracy | 0.9916666746139526 | 1.0 | 31 | 2634845184 |
| P1 Same-scale Intermediate Supervision | Human activity sensing | L4 | macro_accuracy | 0.9833333492279053 | 1.0 | 14 | 5192616960 |
| P1 Same-scale Intermediate Supervision | Human activity sensing | L6 | macro_accuracy | 1.0 | 1.0 | 9 | 7750388736 |
| P1 Same-scale Intermediate Supervision | Human activity sensing | L8 | macro_accuracy | 0.9833333492279053 | 1.0 | 11 | 10308160512 |
| P1 Same-scale Intermediate Supervision | Interference classification | L2 | macro_accuracy | 0.7421428561210632 | 0.7328571081161499 | 58 | 720929792 |
| P1 Same-scale Intermediate Supervision | Interference classification | L4 | macro_accuracy | 0.7392857074737549 | 0.7300000190734863 | 51 | 1416168448 |
| P1 Same-scale Intermediate Supervision | Interference classification | L6 | macro_accuracy | 0.729285717010498 | 0.7257143259048462 | 22 | 2111407104 |
| P1 Same-scale Intermediate Supervision | Interference classification | L8 | macro_accuracy | 0.7285714149475098 | 0.7350000143051147 | 55 | 2806645760 |
| P1 Same-scale Intermediate Supervision | Jamming classification | L2 | macro_accuracy | 0.8226666450500488 | 0.8173334002494812 | 57 | 720931840 |
| P1 Same-scale Intermediate Supervision | Jamming classification | L4 | macro_accuracy | 0.8586666584014893 | 0.8373333811759949 | 49 | 1416170496 |
| P1 Same-scale Intermediate Supervision | Jamming classification | L6 | macro_accuracy | 0.8706666827201843 | 0.8693333864212036 | 64 | 2111409152 |
| P1 Same-scale Intermediate Supervision | Jamming classification | L8 | macro_accuracy | 0.8573333621025085 | 0.8573333621025085 | 38 | 2806647808 |
| P1 Same-scale Intermediate Supervision | NR positioning | L2 | mean_distance_error | 3.4157588481903076 | 3.4019293785095215 | 37 | 3827828224 |
| P1 Same-scale Intermediate Supervision | NR positioning | L4 | mean_distance_error | 3.167616128921509 | 3.2160484790802 | 66 | 7552894464 |
| P1 Same-scale Intermediate Supervision | NR positioning | L6 | mean_distance_error | 3.0587894916534424 | 3.114600896835327 | 99 | 11277960704 |
| P1 Same-scale Intermediate Supervision | NR positioning | L8 | mean_distance_error | 2.9429073333740234 | 3.0197176933288574 | 63 | 15003026944 |
| P1 Same-scale Intermediate Supervision | RF sensing | L2 | macro_accuracy | 0.6357453465461731 | 0.6440702676773071 | 42 | 720939008 |
| P1 Same-scale Intermediate Supervision | RF sensing | L4 | macro_accuracy | 0.675300121307373 | 0.7137924432754517 | 73 | 1416177664 |
| P1 Same-scale Intermediate Supervision | RF sensing | L6 | macro_accuracy | 0.7169788479804993 | 0.7174023985862732 | 82 | 2111416320 |
| P1 Same-scale Intermediate Supervision | RF sensing | L8 | macro_accuracy | 0.6881622672080994 | 0.7131338119506836 | 55 | 2806654976 |
| P2-Reverse Coarse-to-Fine JEPA | DeepMIMO beam | L2 | macro_accuracy | 0.4084385633468628 | 0.44161805510520935 | 95 | 26353152 |
| P2-Reverse Coarse-to-Fine JEPA | DeepMIMO beam | L4 | macro_accuracy | 0.4307680130004883 | 0.4307299554347992 | 69 | 51650048 |
| P2-Reverse Coarse-to-Fine JEPA | DeepMIMO beam | L6 | macro_accuracy | 0.4304586946964264 | 0.42923662066459656 | 61 | 76946944 |
| P2-Reverse Coarse-to-Fine JEPA | DeepMIMO beam | L8 | macro_accuracy | 0.42046719789505005 | 0.42182376980781555 | 28 | 102243840 |
| P2-Reverse Coarse-to-Fine JEPA | Human activity sensing | L2 | macro_accuracy | 0.9750000834465027 | 0.9916666746139526 | 19 | 2634845184 |
| P2-Reverse Coarse-to-Fine JEPA | Human activity sensing | L4 | macro_accuracy | 0.9583333730697632 | 0.9916666746139526 | 19 | 5192616960 |
| P2-Reverse Coarse-to-Fine JEPA | Human activity sensing | L6 | macro_accuracy | 1.0 | 0.9916666746139526 | 16 | 7750388736 |
| P2-Reverse Coarse-to-Fine JEPA | Human activity sensing | L8 | macro_accuracy | 0.9750000238418579 | 1.0 | 12 | 10308160512 |
| P2-Reverse Coarse-to-Fine JEPA | Interference classification | L2 | macro_accuracy | 0.7228571176528931 | 0.7092857360839844 | 21 | 720929792 |
| P2-Reverse Coarse-to-Fine JEPA | Interference classification | L4 | macro_accuracy | 0.7400000095367432 | 0.7321428060531616 | 46 | 1416168448 |
| P2-Reverse Coarse-to-Fine JEPA | Interference classification | L6 | macro_accuracy | 0.7392857074737549 | 0.7392857074737549 | 54 | 2111407104 |
| P2-Reverse Coarse-to-Fine JEPA | Interference classification | L8 | macro_accuracy | 0.7314285635948181 | 0.7321428060531616 | 47 | 2806645760 |
| P2-Reverse Coarse-to-Fine JEPA | Jamming classification | L2 | macro_accuracy | 0.830666720867157 | 0.8173333406448364 | 95 | 720931840 |
| P2-Reverse Coarse-to-Fine JEPA | Jamming classification | L4 | macro_accuracy | 0.8600000143051147 | 0.8559999465942383 | 97 | 1416170496 |
| P2-Reverse Coarse-to-Fine JEPA | Jamming classification | L6 | macro_accuracy | 0.8573333621025085 | 0.874666690826416 | 86 | 2111409152 |
| P2-Reverse Coarse-to-Fine JEPA | Jamming classification | L8 | macro_accuracy | 0.8866666555404663 | 0.9039999842643738 | 38 | 2806647808 |
| P2-Reverse Coarse-to-Fine JEPA | NR positioning | L2 | mean_distance_error | 3.6070046424865723 | 3.590766191482544 | 37 | 3827828224 |
| P2-Reverse Coarse-to-Fine JEPA | NR positioning | L4 | mean_distance_error | 3.275991916656494 | 3.285541296005249 | 33 | 7552894464 |
| P2-Reverse Coarse-to-Fine JEPA | NR positioning | L6 | mean_distance_error | 3.072230100631714 | 3.0547921657562256 | 38 | 11277960704 |
| P2-Reverse Coarse-to-Fine JEPA | NR positioning | L8 | mean_distance_error | 2.873737335205078 | 2.901618480682373 | 69 | 15003026944 |
| P2-Reverse Coarse-to-Fine JEPA | RF sensing | L2 | macro_accuracy | 0.615882158279419 | 0.6659142374992371 | 85 | 720939008 |
| P2-Reverse Coarse-to-Fine JEPA | RF sensing | L4 | macro_accuracy | 0.6113244295120239 | 0.7292972803115845 | 84 | 1416177664 |
| P2-Reverse Coarse-to-Fine JEPA | RF sensing | L6 | macro_accuracy | 0.718097448348999 | 0.7695488333702087 | 115 | 2111416320 |
| P2-Reverse Coarse-to-Fine JEPA | RF sensing | L8 | macro_accuracy | 0.699284553527832 | 0.7357760071754456 | 92 | 2806654976 |
