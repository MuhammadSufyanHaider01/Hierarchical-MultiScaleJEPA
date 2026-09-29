# jamming — antenna condition-specific study fixed-exit metrics

Validation-only results on the dedicated `exit_selection` split. `selected_exit_5pct_rule` is the established selector; `exact_argbest_exit_diagnostic` is descriptive and does not replace selection. Paired realization/master/split provenance and artifact hashes are in the CSV.

| Level | Method | Head strategy | Selected exit / metric | Exact argbest (diagnostic) | L2 | L4 | L6 | L8 | FLOPs | Latency ms |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | p0 | full_information_trained | L2 / 0.468889 | L4 | 0.468889 | 0.476889 | 0.471556 | 0.455111 | 720931840 | 0.703658 |
| 1 | p0 | condition_specific | L4 / 0.484 | L4 | 0.451556 | 0.484 | 0.478222 | 0.455556 | 1416170496 | 0.902235 |
| 2 | p0 | full_information_trained | L2 / 0.483111 | L4 | 0.483111 | 0.496889 | 0.479111 | 0.466667 | 1599212544 | 0.827563 |
| 2 | p0 | condition_specific | L4 / 0.488889 | L4 | 0.463556 | 0.488889 | 0.467556 | 0.458667 | 3147041792 | 1.37364 |
| 4 | p0 | full_information_trained | L2 / 0.490222 | L4 | 0.490222 | 0.503111 | 0.493778 | 0.491556 | 3827829760 | 1.09105 |
| 4 | p0 | condition_specific | L2 / 0.481778 | L4 | 0.481778 | 0.504 | 0.488 | 0.477778 | 3827829760 | 1.31266 |
| 6 | p0 | full_information_trained | L2 / 0.480444 | L4 | 0.480444 | 0.501778 | 0.493778 | 0.492889 | 6685854720 | 1.62801 |
| 6 | p0 | condition_specific | L2 / 0.495111 | L4 | 0.495111 | 0.499556 | 0.497333 | 0.490667 | 6685854720 | 2.68853 |
| 8 | p0 | full_information_trained | L2 / 0.488444 | L8 | 0.488444 | 0.502667 | 0.490667 | 0.503556 | 10173287424 | 3.16048 |
| 8 | p0 | condition_specific | L4 / 0.508444 | L4 | 0.481778 | 0.508444 | 0.493778 | 0.490667 | 20141050880 | 7.84286 |
| 10 | p0 | full_information_trained | L2 / 0.492889 | L8 | 0.492889 | 0.504889 | 0.502667 | 0.511111 | 14290127872 | 6.24816 |
| 10 | p0 | condition_specific | L2 / 0.492 | L6 | 0.492 | 0.515111 | 0.515556 | 0.502222 | 14290127872 | 6.05201 |
| 12 | p0 | full_information_trained | L2 / 0.494667 | L8 | 0.494667 | 0.501333 | 0.503556 | 0.506222 | 19036376064 | 7.49022 |
| 12 | p0 | condition_specific | L2 / 0.487556 | L4 | 0.487556 | 0.511111 | 0.504 | 0.495556 | 19036376064 | 8.6974 |
| 16 | p0 | full_information_trained | L2 / 0.494667 | L8 | 0.494667 | 0.495556 | 0.502222 | 0.510667 | 30417095680 | 13.2731 |
| 16 | p0 | condition_specific | L2 / 0.494667 | L8 | 0.494667 | 0.495556 | 0.502222 | 0.510667 | 30417095680 | 13.7755 |
| 1 | p2 | full_information_trained | L2 / 0.460889 | L4 | 0.460889 | 0.473778 | 0.468889 | 0.454222 | 720931840 | 0.72463 |
| 1 | p2 | condition_specific | L2 / 0.450222 | L4 | 0.450222 | 0.472 | 0.468 | 0.449333 | 720931840 | 0.679954 |
| 2 | p2 | full_information_trained | L2 / 0.485333 | L4 | 0.485333 | 0.497778 | 0.490667 | 0.488889 | 1599212544 | 0.773689 |
| 2 | p2 | condition_specific | L4 / 0.492444 | L4 | 0.463111 | 0.492444 | 0.485333 | 0.479111 | 3147041792 | 1.30636 |
| 4 | p2 | full_information_trained | L4 / 0.507556 | L6 | 0.484889 | 0.507556 | 0.512 | 0.492889 | 7552896000 | 2.56265 |
| 4 | p2 | condition_specific | L2 / 0.488 | L6 | 0.488 | 0.492889 | 0.495111 | 0.484 | 3827829760 | 1.31682 |
| 6 | p2 | full_information_trained | L2 / 0.490667 | L4 | 0.490667 | 0.514222 | 0.507556 | 0.499556 | 6685854720 | 1.39419 |
| 6 | p2 | condition_specific | L2 / 0.484889 | L4 | 0.484889 | 0.503111 | 0.5 | 0.496889 | 6685854720 | 2.66938 |
| 8 | p2 | full_information_trained | L4 / 0.517778 | L4 | 0.478222 | 0.517778 | 0.505778 | 0.502667 | 20141050880 | 6.41491 |
| 8 | p2 | condition_specific | L2 / 0.492889 | L4 | 0.492889 | 0.517778 | 0.515556 | 0.507111 | 10173287424 | 4.01814 |
| 10 | p2 | full_information_trained | L4 / 0.534667 | L4 | 0.493333 | 0.534667 | 0.509778 | 0.512444 | 28323351552 | 10.4737 |
| 10 | p2 | condition_specific | L2 / 0.502667 | L4 | 0.502667 | 0.521333 | 0.515111 | 0.512889 | 14290127872 | 6.07961 |
| 12 | p2 | full_information_trained | L4 / 0.533333 | L4 | 0.495556 | 0.533333 | 0.515111 | 0.512 | 37764467712 | 15.8263 |
| 12 | p2 | condition_specific | L4 / 0.516 | L6 | 0.495556 | 0.516 | 0.540444 | 0.528 | 37764467712 | 17.205 |
| 16 | p2 | full_information_trained | L4 / 0.532444 | L4 | 0.495556 | 0.532444 | 0.514667 | 0.511111 | 60423146496 | 16.3569 |
| 16 | p2 | condition_specific | L4 / 0.532444 | L4 | 0.495556 | 0.532444 | 0.514667 | 0.511111 | 60423146496 | 27.1577 |
