# Seven-seed downstream masking report: multi_block_n6

Frozen pretrained checkpoints; downstream head seeds 0–6; fixed split seed 2025.
Each aggregate is mean ± sample SD and includes per-seed values in the CSV tables.
NR positioning error is lower-is-better. B0 is P0 at L8; P0 is reported at each exit.
No backbone pretraining was run for this study.
The imported P2-Reverse head_seed=2025 metrics are excluded from the head seeds 0–6 aggregate.

frequency_n6 and multi_block_n6 are separate repeated historical conditions with the same nominal span-time geometry; they are not independent mask-family evidence.
p0: Recovered phase5_formal checkpoint, epoch 64 of configured 240; differs from the report-referenced M-block P0 checkpoint.
p1: M-block P1 reuses the frequency_n6 P1 checkpoint, as recorded in the checkpoint mapping manifest.
p2: Checkpoint source/mapping is documented in checkpoint_bundle/README.md.
p2_reverse: Checkpoint source/mapping is documented in checkpoint_bundle/README.md.

| Method | Task | Validation-selected test mean ± SD | Selected exits by seed |
|---|---|---:|---|
| p0 | rfs | 0.645844 ± 0.01202 | {"0": 4, "1": 4, "2": 4, "3": 6, "4": 6, "5": 6, "6": 4} |
| p0 | jamming | 0.89581 ± 0.007829 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p0 | human-activity | 0.980952 ± 0.007927 | {"0": 4, "1": 4, "2": 4, "3": 4, "4": 4, "5": 4, "6": 4} |
| p0 | interference | 0.747347 ± 0.003317 | {"0": 6, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 6} |
| p0 | nrpos-outdoor | 3.02089 ± 0.01645 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p0 | deepmimo-beam | 0.450519 ± 0.01664 | {"0": 8, "1": 6, "2": 4, "3": 6, "4": 6, "5": 6, "6": 6} |
| p1 | rfs | 0.716226 ± 0.01609 | {"0": 8, "1": 6, "2": 6, "3": 6, "4": 6, "5": 4, "6": 6} |
| p1 | jamming | 0.867619 ± 0.008458 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p1 | human-activity | 0.975 ± 0.01667 | {"0": 2, "1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2} |
| p1 | interference | 0.737143 ± 0.003376 | {"0": 6, "1": 6, "2": 6, "3": 6, "4": 6, "5": 6, "6": 6} |
| p1 | nrpos-outdoor | 2.96772 ± 0.01964 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p1 | deepmimo-beam | 0.450628 ± 0.008335 | {"0": 8, "1": 6, "2": 6, "3": 6, "4": 2, "5": 4, "6": 6} |
| p2 | rfs | 0.728337 ± 0.02046 | {"0": 8, "1": 8, "2": 8, "3": 6, "4": 6, "5": 8, "6": 8} |
| p2 | jamming | 0.884191 ± 0.005934 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p2 | human-activity | 0.99881 ± 0.00315 | {"0": 4, "1": 4, "2": 4, "3": 4, "4": 8, "5": 4, "6": 4} |
| p2 | interference | 0.75398 ± 0.003243 | {"0": 8, "1": 6, "2": 6, "3": 8, "4": 8, "5": 8, "6": 6} |
| p2 | nrpos-outdoor | 2.75663 ± 0.01936 | {"0": 6, "1": 8, "2": 6, "3": 8, "4": 6, "5": 8, "6": 8} |
| p2 | deepmimo-beam | 0.445695 ± 0.009773 | {"0": 4, "1": 6, "2": 2, "3": 4, "4": 2, "5": 4, "6": 6} |
| p2_reverse | rfs | 0.720144 ± 0.01593 | {"0": 6, "1": 6, "2": 6, "3": 6, "4": 8, "5": 6, "6": 6} |
| p2_reverse | jamming | 0.897905 ± 0.002417 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p2_reverse | human-activity | 0.989286 ± 0.01045 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p2_reverse | interference | 0.739796 ± 0.006823 | {"0": 6, "1": 6, "2": 6, "3": 6, "4": 6, "5": 6, "6": 6} |
| p2_reverse | nrpos-outdoor | 2.87738 ± 0.02789 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p2_reverse | deepmimo-beam | 0.421849 ± 0.01359 | {"0": 2, "1": 2, "2": 2, "3": 8, "4": 2, "5": 8, "6": 2} |
| b0 | rfs | 0.678702 ± 0.02244 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | jamming | 0.89581 ± 0.007829 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | human-activity | 0.980952 ± 0.01045 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | interference | 0.745918 ± 0.003266 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | nrpos-outdoor | 3.02089 ± 0.01645 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | deepmimo-beam | 0.451712 ± 0.01283 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |

CSV/Markdown tables and the existing figure panels regenerated as PNG and PDF are adjacent.
Artifact hashes and source provenance are recorded in manifest.json.
