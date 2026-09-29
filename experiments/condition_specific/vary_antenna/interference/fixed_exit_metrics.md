# interference — antenna condition-specific study fixed-exit metrics

Validation-only results on the dedicated `exit_selection` split. `selected_exit_5pct_rule` is the established selector; `exact_argbest_exit_diagnostic` is descriptive and does not replace selection. Paired realization/master/split provenance and artifact hashes are in the CSV.

| Level | Method | Head strategy | Selected exit / metric | Exact argbest (diagnostic) | L2 | L4 | L6 | L8 | FLOPs | Latency ms |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | p0 | full_information_trained | L2 / 0.661333 | L8 | 0.661333 | 0.668 | 0.669333 | 0.681333 | 720929792 | 0.699903 |
| 1 | p0 | condition_specific | L2 / 0.690667 | L2 | 0.690667 | 0.672 | 0.669333 | 0.690667 | 720929792 | 0.681376 |
| 2 | p0 | full_information_trained | L2 / 0.677333 | L8 | 0.677333 | 0.656 | 0.68 | 0.688 | 1599210496 | 0.406748 |
| 2 | p0 | condition_specific | L2 / 0.685333 | L8 | 0.685333 | 0.692 | 0.68 | 0.698667 | 1599210496 | 0.833694 |
| 4 | p0 | full_information_trained | L2 / 0.686667 | L8 | 0.686667 | 0.673333 | 0.68 | 0.692 | 3827827712 | 0.886506 |
| 4 | p0 | condition_specific | L2 / 0.685333 | L8 | 0.685333 | 0.689333 | 0.686667 | 0.696 | 3827827712 | 1.26232 |
| 6 | p0 | full_information_trained | L2 / 0.686667 | L8 | 0.686667 | 0.678667 | 0.684 | 0.690667 | 6685852672 | 2.70483 |
| 6 | p0 | condition_specific | L2 / 0.685333 | L8 | 0.685333 | 0.690667 | 0.690667 | 0.705333 | 6685852672 | 2.73186 |
| 8 | p0 | full_information_trained | L2 / 0.681333 | L8 | 0.681333 | 0.677333 | 0.688 | 0.698667 | 10173285376 | 2.06513 |
| 8 | p0 | condition_specific | L2 / 0.678667 | L8 | 0.678667 | 0.690667 | 0.690667 | 0.698667 | 10173285376 | 3.81395 |
| 10 | p0 | full_information_trained | L2 / 0.690667 | L8 | 0.690667 | 0.684 | 0.693333 | 0.702667 | 14290125824 | 4.84463 |
| 10 | p0 | condition_specific | L4 / 0.693333 | L8 | 0.670667 | 0.693333 | 0.692 | 0.708 | 28323349504 | 11.9665 |
| 12 | p0 | full_information_trained | L2 / 0.693333 | L8 | 0.693333 | 0.686667 | 0.688 | 0.705333 | 19036374016 | 7.73586 |
| 12 | p0 | condition_specific | L2 / 0.689333 | L8 | 0.689333 | 0.689333 | 0.689333 | 0.709333 | 19036374016 | 8.72095 |
| 16 | p0 | full_information_trained | L2 / 0.701333 | L8 | 0.701333 | 0.685333 | 0.688 | 0.708 | 30417093632 | 12.1549 |
| 16 | p0 | condition_specific | L2 / 0.701333 | L8 | 0.701333 | 0.685333 | 0.688 | 0.708 | 30417093632 | 14.0043 |
| 1 | p2 | full_information_trained | L2 / 0.669333 | L4 | 0.669333 | 0.670667 | 0.661333 | 0.669333 | 720929792 | 0.396426 |
| 1 | p2 | condition_specific | L2 / 0.674667 | L6 | 0.674667 | 0.684 | 0.705333 | 0.685333 | 720929792 | 0.726727 |
| 2 | p2 | full_information_trained | L2 / 0.688 | L6 | 0.688 | 0.692 | 0.697333 | 0.689333 | 1599210496 | 0.36227 |
| 2 | p2 | condition_specific | L4 / 0.7 | L6 | 0.681333 | 0.7 | 0.717333 | 0.692 | 3147039744 | 1.3184 |
| 4 | p2 | full_information_trained | L2 / 0.688 | L4 | 0.688 | 0.702667 | 0.692 | 0.689333 | 3827827712 | 1.31216 |
| 4 | p2 | condition_specific | L2 / 0.674667 | L8 | 0.674667 | 0.7 | 0.7 | 0.706667 | 3827827712 | 1.25455 |
| 6 | p2 | full_information_trained | L2 / 0.686667 | L4 | 0.686667 | 0.701333 | 0.694667 | 0.694667 | 6685852672 | 2.71088 |
| 6 | p2 | condition_specific | L2 / 0.690667 | L4 | 0.690667 | 0.708 | 0.694667 | 0.696 | 6685852672 | 2.67446 |
| 8 | p2 | full_information_trained | L2 / 0.677333 | L4 | 0.677333 | 0.704 | 0.697333 | 0.697333 | 10173285376 | 3.27654 |
| 8 | p2 | condition_specific | L2 / 0.689333 | L6 | 0.689333 | 0.701333 | 0.714667 | 0.7 | 10173285376 | 4.11167 |
| 10 | p2 | full_information_trained | L2 / 0.684 | L4 | 0.684 | 0.704 | 0.698667 | 0.702667 | 14290125824 | 4.38999 |
| 10 | p2 | condition_specific | L2 / 0.690667 | L8 | 0.690667 | 0.694667 | 0.697333 | 0.712 | 14290125824 | 5.97428 |
| 12 | p2 | full_information_trained | L2 / 0.688 | L4 | 0.688 | 0.710667 | 0.698667 | 0.709333 | 19036374016 | 8.95062 |
| 12 | p2 | condition_specific | L2 / 0.694667 | L8 | 0.694667 | 0.698667 | 0.704 | 0.712 | 19036374016 | 8.75417 |
| 16 | p2 | full_information_trained | L2 / 0.697333 | L8 | 0.697333 | 0.710667 | 0.708 | 0.717333 | 30417093632 | 10.7791 |
| 16 | p2 | condition_specific | L2 / 0.697333 | L8 | 0.697333 | 0.710667 | 0.708 | 0.717333 | 30417093632 | 13.7738 |
