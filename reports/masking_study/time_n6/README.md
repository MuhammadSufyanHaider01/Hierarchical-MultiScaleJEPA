# Seven-seed downstream masking report: time_n6

Frozen pretrained checkpoints; downstream head seeds 0–6; fixed split seed 2025.
Each aggregate is mean ± sample SD and includes per-seed values in the CSV tables.
NR positioning error is lower-is-better. B0 is P0 at L8; P0 is reported at each exit.
No backbone pretraining was run for this study.
The imported P2-Reverse head_seed=2025 metrics are excluded from the head seeds 0–6 aggregate.

p0: Checkpoint source/mapping is documented in checkpoint_bundle/README.md.
p1: Checkpoint source/mapping is documented in checkpoint_bundle/README.md.
p2: Checkpoint source/mapping is documented in checkpoint_bundle/README.md.
p2_reverse: Checkpoint source/mapping is documented in checkpoint_bundle/README.md.

| Method | Task | Validation-selected test mean ± SD | Selected exits by seed |
|---|---|---:|---|
| p0 | rfs | 0.721958 ± 0.03451 | {"0": 2, "1": 4, "2": 4, "3": 6, "4": 4, "5": 2, "6": 4} |
| p0 | jamming | 0.901333 ± 0.006971 | {"0": 6, "1": 4, "2": 6, "3": 6, "4": 4, "5": 6, "6": 4} |
| p0 | human-activity | 1 ± 0 | {"0": 2, "1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2} |
| p0 | interference | 0.739898 ± 0.00487 | {"0": 2, "1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2} |
| p0 | nrpos-outdoor | 3.34215 ± 0.03957 | {"0": 8, "1": 8, "2": 8, "3": 4, "4": 4, "5": 4, "6": 8} |
| p0 | deepmimo-beam | 0.406164 ± 0.01083 | {"0": 2, "1": 4, "2": 2, "3": 4, "4": 4, "5": 4, "6": 4} |
| p1 | rfs | 0.730504 ± 0.04835 | {"0": 6, "1": 4, "2": 6, "3": 6, "4": 4, "5": 6, "6": 6} |
| p1 | jamming | 0.883429 ± 0.007654 | {"0": 2, "1": 2, "2": 4, "3": 4, "4": 2, "5": 2, "6": 2} |
| p1 | human-activity | 0.990476 ± 0.005751 | {"0": 2, "1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2} |
| p1 | interference | 0.732755 ± 0.005509 | {"0": 4, "1": 4, "2": 4, "3": 4, "4": 4, "5": 4, "6": 4} |
| p1 | nrpos-outdoor | 3.33596 ± 0.01535 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 6, "5": 8, "6": 6} |
| p1 | deepmimo-beam | 0.416 ± 0.008191 | {"0": 4, "1": 4, "2": 4, "3": 4, "4": 4, "5": 4, "6": 4} |
| p2 | rfs | 0.744219 ± 0.01222 | {"0": 4, "1": 4, "2": 4, "3": 6, "4": 6, "5": 4, "6": 6} |
| p2 | jamming | 0.895238 ± 0.006525 | {"0": 4, "1": 4, "2": 4, "3": 4, "4": 4, "5": 4, "6": 4} |
| p2 | human-activity | 0.988095 ± 0.008132 | {"0": 2, "1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2} |
| p2 | interference | 0.757041 ± 0.003878 | {"0": 6, "1": 4, "2": 6, "3": 4, "4": 4, "5": 4, "6": 4} |
| p2 | nrpos-outdoor | 3.05002 ± 0.01786 | {"0": 6, "1": 6, "2": 6, "3": 6, "4": 6, "5": 6, "6": 6} |
| p2 | deepmimo-beam | 0.428392 ± 0.007704 | {"0": 2, "1": 4, "2": 6, "3": 4, "4": 2, "5": 6, "6": 2} |
| p2_reverse | rfs | 0.74324 ± 0.01018 | {"0": 4, "1": 4, "2": 4, "3": 4, "4": 4, "5": 2, "6": 4} |
| p2_reverse | jamming | 0.886286 ± 0.005848 | {"0": 4, "1": 4, "2": 4, "3": 4, "4": 4, "5": 4, "6": 4} |
| p2_reverse | human-activity | 1 ± 0 | {"0": 4, "1": 4, "2": 4, "3": 4, "4": 4, "5": 4, "6": 4} |
| p2_reverse | interference | 0.748061 ± 0.01099 | {"0": 2, "1": 6, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2} |
| p2_reverse | nrpos-outdoor | 3.31246 ± 0.01986 | {"0": 6, "1": 6, "2": 6, "3": 6, "4": 6, "5": 4, "6": 6} |
| p2_reverse | deepmimo-beam | 0.4126 ± 0.008131 | {"0": 4, "1": 2, "2": 6, "3": 4, "4": 6, "5": 6, "6": 6} |
| b0 | rfs | 0.698831 ± 0.03059 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | jamming | 0.86781 ± 0.008792 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | human-activity | 0.990476 ± 0.00315 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | interference | 0.704796 ± 0.008813 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | nrpos-outdoor | 3.31666 ± 0.02421 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | deepmimo-beam | 0.379183 ± 0.01755 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |

CSV/Markdown tables and the existing figure panels regenerated as PNG and PDF are adjacent.
Artifact hashes and source provenance are recorded in manifest.json.
