# time_n6 available-method downstream comparison

This downstream-only packet includes only methods with complete comparable results: P0 — Original LatentWave (L8), P2 — Hierarchical Multi-Scale JEPA, P2-Reverse — Coarse-to-Fine JEPA. P1 and the post-hoc B1 approach are excluded. P0/P2 are reused only when a complete historical per-mask result packet is present; P2-Reverse is loaded from its validated frozen-head test packet. FLOPs/example count the matched encoder through each exit and task head; counts use `torch.utils.flop_counter.FlopCounterMode` supported operations only and do not indicate wall-clock speed.

For P2-Reverse, head checkpoints and exits are selected on validation only; held-out test metrics are reported after selection. Consult `manifest.json` for exact input files and hashes.
