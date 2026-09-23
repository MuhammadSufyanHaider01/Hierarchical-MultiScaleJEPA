# Paired availability study — 5G NR outdoor positioning

This packet compares validation-only, frozen-head early-exit selection for the LatentWave + EE P0 baseline and the proposed hierarchical P2 approach. No foundation weights, downstream heads, manifest, or test split were changed.

- Availability axis: Available antenna channels (C)
- Primary validation metric: mean distance error
- P0 source artifact: `experiments/vary_antenna/nrpos-outdoor/latentwave_ee/input_information_exit_metrics.json`
- P2 source artifact: `experiments/vary_antenna/nrpos-outdoor/hierarchical/input_information_exit_metrics.json`

`latentwave_ee/` and `hierarchical/` contain immutable per-method metric artifacts; the root PNG/PDF is the single two-line comparison figure.
