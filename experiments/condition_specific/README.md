# Synthetic condition-specific availability reports

This report family contains the synthetic-data condition-specific antenna and time-sample runs and mirrors the task/axis hierarchy of the synthetic study. It holds the packets previously nested under each synthetic task’s `condition_specific/` folder.

Figures contain only two series: LatentWave + EE with condition-specific heads and the proposed hierarchical method with condition-specific heads. Full-information-head curves are omitted; their paired numbers remain in the diagnostic tables for audit. Standard comparison plots show the official earliest-within-5%-of-best exits; separate `best_layer_*` plots show strict argbest exits among L2/L4/L6/L8. Each plotted point is labeled with its exit layer.

The strict argbest score is measured on the same synthetic exit-selection split used to choose that layer, so it is descriptive rather than a held-out estimate. No test evaluation was performed. All values come from the shared synthetic factory/OFDMA master (`master_sha256=e228979706684d6374b689fe1b4f630cc05b01ae34c027a6fd4accb7c6096778`).
