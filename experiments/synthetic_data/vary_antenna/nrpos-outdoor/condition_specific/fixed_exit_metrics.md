# nrpos-outdoor — antenna condition-specific study fixed-exit metrics

Validation-only results on the dedicated `exit_selection` split. `selected_exit_5pct_rule` is the established selector; `exact_argbest_exit_diagnostic` is descriptive and does not replace selection. Paired realization/master/split provenance and artifact hashes are in the CSV.

| Level | Method | Head strategy | Selected exit / metric | Exact argbest (diagnostic) | L2 | L4 | L6 | L8 | FLOPs | Latency ms |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | p0 | full_information_trained | L2 / 9.78313 | L8 | 9.78313 | 9.32892 | 9.35964 | 9.32376 | 720930304 | 0.360977 |
| 1 | p0 | condition_specific | L2 / 9.28843 | L8 | 9.28843 | 9.03069 | 8.87296 | 8.86009 | 720930304 | 0.668888 |
| 2 | p0 | full_information_trained | L4 / 8.64068 | L6 | 9.25604 | 8.64068 | 8.42437 | 8.54076 | 3147040256 | 0.66731 |
| 2 | p0 | condition_specific | L4 / 8.59494 | L6 | 9.13379 | 8.59494 | 8.42407 | 8.43798 | 3147040256 | 1.30799 |
| 4 | p0 | full_information_trained | L4 / 7.95078 | L6 | 8.49815 | 7.95078 | 7.79701 | 7.82816 | 7552894464 | 1.13124 |
| 4 | p0 | condition_specific | L4 / 8.00901 | L6 | 8.5461 | 8.00901 | 7.84083 | 7.8979 | 7552894464 | 2.58196 |
| 6 | p0 | full_information_trained | L4 / 7.79615 | L6 | 8.35771 | 7.79615 | 7.62639 | 7.66974 | 13217564160 | 2.49036 |
| 6 | p0 | condition_specific | L4 / 7.82179 | L6 | 8.37161 | 7.82179 | 7.70166 | 7.74696 | 13217564160 | 5.06225 |
| 8 | p0 | full_information_trained | L4 / 7.67431 | L6 | 8.19966 | 7.67431 | 7.46466 | 7.55231 | 20141049344 | 3.78167 |
| 8 | p0 | condition_specific | L4 / 7.63189 | L6 | 8.2124 | 7.63189 | 7.55529 | 7.6297 | 20141049344 | 8.05862 |
| 10 | p0 | full_information_trained | L4 / 7.71734 | L6 | 8.26418 | 7.71734 | 7.46973 | 7.56834 | 28323350016 | 5.77367 |
| 10 | p0 | condition_specific | L4 / 7.68222 | L6 | 8.26924 | 7.68222 | 7.54907 | 7.63955 | 28323350016 | 11.9907 |
| 12 | p0 | full_information_trained | L4 / 7.71892 | L6 | 8.24553 | 7.71892 | 7.46548 | 7.5206 | 37764466176 | 8.2798 |
| 12 | p0 | condition_specific | L4 / 7.6958 | L6 | 8.23251 | 7.6958 | 7.54333 | 7.56772 | 37764466176 | 17.149 |
| 16 | p0 | full_information_trained | L4 / 7.6564 | L6 | 8.14004 | 7.6564 | 7.42561 | 7.42606 | 60423144960 | 13.0321 |
| 16 | p0 | condition_specific | L4 / 7.6564 | L6 | 8.14004 | 7.6564 | 7.42561 | 7.42606 | 60423144960 | 27.0271 |
| 1 | p2 | full_information_trained | L2 / 9.69829 | L6 | 9.69829 | 9.50962 | 9.49621 | 9.68839 | 720930304 | 0.36657 |
| 1 | p2 | condition_specific | L4 / 8.97339 | L6 | 9.42997 | 8.97339 | 8.97277 | 9.27695 | 1416168960 | 0.907893 |
| 2 | p2 | full_information_trained | L2 / 9.13176 | L6 | 9.13176 | 8.87596 | 8.79152 | 8.98899 | 1599211008 | 0.419961 |
| 2 | p2 | condition_specific | L2 / 8.93495 | L6 | 8.93495 | 8.60516 | 8.59898 | 8.83804 | 1599211008 | 0.819945 |
| 4 | p2 | full_information_trained | L2 / 8.38973 | L6 | 8.38973 | 8.15918 | 8.09527 | 8.28712 | 3827828224 | 0.570622 |
| 4 | p2 | condition_specific | L4 / 8.13196 | L6 | 8.4967 | 8.13196 | 8.0848 | 8.28178 | 7552894464 | 2.53554 |
| 6 | p2 | full_information_trained | L2 / 8.28184 | L6 | 8.28184 | 8.03846 | 7.94844 | 8.07086 | 6685853184 | 1.30161 |
| 6 | p2 | condition_specific | L2 / 8.27446 | L6 | 8.27446 | 7.98145 | 7.93896 | 8.10054 | 6685853184 | 2.67152 |
| 8 | p2 | full_information_trained | L2 / 8.15063 | L6 | 8.15063 | 7.92442 | 7.82667 | 7.95163 | 10173285888 | 1.9594 |
| 8 | p2 | condition_specific | L4 / 7.94973 | L6 | 8.14411 | 7.94973 | 7.71195 | 8.01817 | 20141049344 | 7.81846 |
| 10 | p2 | full_information_trained | L2 / 8.1047 | L6 | 8.1047 | 7.88541 | 7.7954 | 7.93984 | 14290126336 | 2.95299 |
| 10 | p2 | condition_specific | L2 / 8.10487 | L6 | 8.10487 | 7.93688 | 7.78748 | 7.99823 | 14290126336 | 5.99645 |
| 12 | p2 | full_information_trained | L2 / 8.07975 | L6 | 8.07975 | 7.88123 | 7.8247 | 7.92343 | 19036374528 | 4.17368 |
| 12 | p2 | condition_specific | L2 / 8.07115 | L6 | 8.07115 | 7.92259 | 7.80829 | 7.99173 | 19036374528 | 8.75251 |
| 16 | p2 | full_information_trained | L2 / 7.98023 | L6 | 7.98023 | 7.78151 | 7.74381 | 7.86196 | 30417094144 | 6.55349 |
| 16 | p2 | condition_specific | L2 / 7.98023 | L6 | 7.98023 | 7.78151 | 7.74381 | 7.86196 | 30417094144 | 13.811 |
