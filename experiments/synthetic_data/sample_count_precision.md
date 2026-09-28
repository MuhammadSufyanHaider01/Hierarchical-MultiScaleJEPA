# Formal sample-count and precision note

These synthetic OFDMA factory-scene results are supporting stress-test evidence, not a reconstruction of the physically collected LatentWave datasets. The formal master is shared across tasks, sweeps, methods, and condition-specific diagnostics (`master_sha256=e228979706684d6374b689fe1b4f630cc05b01ae34c027a6fd4accb7c6096778`).

The locked total is 7,500 realizations, with disjoint realization-level splits: 4,500 head-train and 750 each for head validation, calibration, exit selection, and held-out test. The count was fixed from pilot anchors before formal selection; it was not extended based on test results.

The pilot-based precision planner estimated that the source-count cap would not meet all desired test-interval half-widths:

| Task proxy | Desired 95% test half-width | Projected at cap | Test examples at cap | Estimated required test examples | Shortfall |
|---|---:|---:|---:|---:|---:|
| NR positioning / factory localization | 0.25 m | 0.4524 m | 750 | 2,457 | 1,707 |
| Interference / jammer presence | 0.030 macro accuracy | 0.0323 | 750 | 868 | 118 |
| Jamming / jammer mode | 0.030 macro accuracy | 0.0325 | 750 | 879 | 129 |

These are pilot-based projections, not observed test confidence intervals. Formal exit selection and all current diagnostic reports use only the dedicated 750-realization `exit_selection` split. No held-out test predictions or metrics have been produced. The test stage remains gated on manual review of the selection artifacts and should report the realized test intervals alongside this projected shortfall.

Fixed-exit tables also include an exact argbest-layer diagnostic on the `exit_selection` split. It is descriptive and must not replace the prespecified earliest-within-5%-of-best selector or be presented as an independent test result.
