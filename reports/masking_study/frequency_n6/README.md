# Seven-seed downstream masking report: frequency_n6

Frozen pretrained checkpoints; downstream head seeds 0–6; fixed split seed 2025.
Each aggregate is mean ± sample SD and includes per-seed values in the CSV tables.
NR positioning error is lower-is-better. B0 is P0 at L8; P0 is reported at each exit.
No backbone pretraining was run for this study.
The imported P2-Reverse head_seed=2025 metrics are excluded from the head seeds 0–6 aggregate.

frequency_n6 and multi_block_n6 are separate repeated historical conditions with the same nominal span-time geometry; they are not independent mask-family evidence.
p0: Checkpoint source/mapping is documented in checkpoint_bundle/README.md.
p1: Checkpoint source/mapping is documented in checkpoint_bundle/README.md.
p2: Checkpoint source/mapping is documented in checkpoint_bundle/README.md.
p2_reverse: Checkpoint source/mapping is documented in checkpoint_bundle/README.md.

| Method | Task | Validation-selected test mean ± SD | Selected exits by seed |
|---|---|---:|---|
| p0 | rfs | 0.667025 ± 0.01072 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p0 | jamming | 0.889333 ± 0.006799 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p0 | human-activity | 0.964286 ± 0.009272 | {"0": 4, "1": 4, "2": 4, "3": 4, "4": 4, "5": 4, "6": 4} |
| p0 | interference | 0.716939 ± 0.009163 | {"0": 6, "1": 6, "2": 6, "3": 6, "4": 6, "5": 8, "6": 4} |
| p0 | nrpos-outdoor | 3.16007 ± 0.02169 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p0 | deepmimo-beam | 0.448249 ± 0.01448 | {"0": 6, "1": 8, "2": 8, "3": 8, "4": 8, "5": 6, "6": 6} |
| p1 | rfs | 0.716226 ± 0.01609 | {"0": 8, "1": 6, "2": 6, "3": 6, "4": 6, "5": 4, "6": 6} |
| p1 | jamming | 0.867429 ± 0.00884 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p1 | human-activity | 0.975 ± 0.01667 | {"0": 2, "1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2} |
| p1 | interference | 0.736837 ± 0.003422 | {"0": 6, "1": 6, "2": 6, "3": 6, "4": 6, "5": 6, "6": 6} |
| p1 | nrpos-outdoor | 2.96772 ± 0.01973 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p1 | deepmimo-beam | 0.450439 ± 0.008262 | {"0": 8, "1": 6, "2": 6, "3": 6, "4": 2, "5": 4, "6": 6} |
| p2 | rfs | 0.688119 ± 0.0171 | {"0": 8, "1": 6, "2": 6, "3": 6, "4": 6, "5": 6, "6": 6} |
| p2 | jamming | 0.878095 ± 0.01 | {"0": 8, "1": 4, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p2 | human-activity | 0.97381 ± 0.03383 | {"0": 4, "1": 4, "2": 4, "3": 6, "4": 2, "5": 4, "6": 2} |
| p2 | interference | 0.743367 ± 0.003368 | {"0": 4, "1": 4, "2": 4, "3": 4, "4": 4, "5": 4, "6": 4} |
| p2 | nrpos-outdoor | 2.85102 ± 0.02014 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p2 | deepmimo-beam | 0.423428 ± 0.01361 | {"0": 2, "1": 8, "2": 8, "3": 4, "4": 8, "5": 4, "6": 2} |
| p2_reverse | rfs | 0.68044 ± 0.009708 | {"0": 8, "1": 8, "2": 6, "3": 8, "4": 8, "5": 8, "6": 8} |
| p2_reverse | jamming | 0.878667 ± 0.003443 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p2_reverse | human-activity | 0.988095 ± 0.004454 | {"0": 2, "1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2} |
| p2_reverse | interference | 0.734796 ± 0.005528 | {"0": 4, "1": 4, "2": 4, "3": 4, "4": 4, "5": 4, "6": 4} |
| p2_reverse | nrpos-outdoor | 2.88958 ± 0.009341 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p2_reverse | deepmimo-beam | 0.434989 ± 0.009951 | {"0": 2, "1": 2, "2": 2, "3": 4, "4": 6, "5": 2, "6": 6} |
| b0 | rfs | 0.667025 ± 0.01072 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | jamming | 0.889333 ± 0.006799 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | human-activity | 0.988095 ± 0.008132 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | interference | 0.732245 ± 0.008743 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | nrpos-outdoor | 3.16007 ± 0.02169 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | deepmimo-beam | 0.449929 ± 0.01373 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |

CSV/Markdown tables and the existing figure panels regenerated as PNG and PDF are adjacent.
Artifact hashes and source provenance are recorded in manifest.json.
