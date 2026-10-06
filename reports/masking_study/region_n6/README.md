# Seven-seed downstream masking report: region_n6

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
| p0 | rfs | 0.788691 ± 0.01121 | {"0": 8, "1": 8, "2": 8, "3": 6, "4": 8, "5": 6, "6": 6} |
| p0 | jamming | 0.871238 ± 0.003838 | {"0": 6, "1": 6, "2": 6, "3": 6, "4": 6, "5": 6, "6": 6} |
| p0 | human-activity | 0.971429 ± 0.008132 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p0 | interference | 0.722245 ± 0.003922 | {"0": 4, "1": 4, "2": 4, "3": 4, "4": 4, "5": 4, "6": 4} |
| p0 | nrpos-outdoor | 2.93362 ± 0.01036 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p0 | deepmimo-beam | 0.408418 ± 0.007448 | {"0": 6, "1": 4, "2": 6, "3": 4, "4": 6, "5": 6, "6": 4} |
| p1 | rfs | 0.793711 ± 0.02185 | {"0": 4, "1": 4, "2": 4, "3": 2, "4": 4, "5": 4, "6": 4} |
| p1 | jamming | 0.872571 ± 0.0062 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p1 | human-activity | 0.982143 ± 0.01397 | {"0": 4, "1": 4, "2": 4, "3": 4, "4": 4, "5": 6, "6": 4} |
| p1 | interference | 0.744082 ± 0.005277 | {"0": 4, "1": 4, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p1 | nrpos-outdoor | 2.71395 ± 0.02018 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p1 | deepmimo-beam | 0.4728 ± 0.01459 | {"0": 2, "1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2} |
| p2 | rfs | 0.75015 ± 0.01177 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p2 | jamming | 0.863048 ± 0.004727 | {"0": 4, "1": 6, "2": 4, "3": 4, "4": 4, "5": 6, "6": 4} |
| p2 | human-activity | 0.982143 ± 0.007498 | {"0": 2, "1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2} |
| p2 | interference | 0.740408 ± 0.006972 | {"0": 6, "1": 6, "2": 6, "3": 6, "4": 8, "5": 6, "6": 6} |
| p2 | nrpos-outdoor | 2.765 ± 0.0234 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p2 | deepmimo-beam | 0.444113 ± 0.005874 | {"0": 2, "1": 2, "2": 2, "3": 2, "4": 4, "5": 2, "6": 2} |
| p2_reverse | rfs | 0.785112 ± 0.01307 | {"0": 6, "1": 6, "2": 6, "3": 6, "4": 8, "5": 6, "6": 6} |
| p2_reverse | jamming | 0.882667 ± 0.008503 | {"0": 4, "1": 4, "2": 4, "3": 4, "4": 6, "5": 6, "6": 6} |
| p2_reverse | human-activity | 0.983333 ± 0.006804 | {"0": 4, "1": 2, "2": 2, "3": 4, "4": 2, "5": 2, "6": 2} |
| p2_reverse | interference | 0.73602 ± 0.01203 | {"0": 2, "1": 4, "2": 4, "3": 8, "4": 4, "5": 8, "6": 4} |
| p2_reverse | nrpos-outdoor | 2.80198 ± 0.01487 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| p2_reverse | deepmimo-beam | 0.438394 ± 0.01384 | {"0": 2, "1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2} |
| b0 | rfs | 0.780556 ± 0.02214 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | jamming | 0.856381 ± 0.005317 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | human-activity | 0.971429 ± 0.008132 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | interference | 0.720408 ± 0.005165 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | nrpos-outdoor | 2.93362 ± 0.01036 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |
| b0 | deepmimo-beam | 0.398667 ± 0.006798 | {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8} |

CSV/Markdown tables and the existing figure panels regenerated as PNG and PDF are adjacent.
Artifact hashes and source provenance are recorded in manifest.json.
