# CNN-HierJEPA implementation and run sequence

This branch adds an optional CNN backbone while preserving the default
LatentWave ViT path and its existing checkpoints. This document records the
implemented interfaces and the approved first experiment. It does not authorize
or launch formal GPU jobs.

The CNN implementation is now also reachable from `main`; the executable
M-block experiment sequence and existing-artifact status are maintained in
[`cnn_hierjepa_main_execution_plan.md`](cnn_hierjepa_main_execution_plan.md).

Human-facing CNN approach labels are **C0** (`cnn_p0`) and **C2** (`cnn_p2`);
ViT P0/P2 labels remain unchanged. The internal CNN method keys and artifact
paths retain the `cnn_p0`/`cnn_p2` spelling.

## Backbone and representation interface

- The ConvNeXt V2-style encoder uses stage depths `[2, 2, 6, 2]` and widths
  `[48, 96, 192, 384]`, with depthwise 7x7 blocks, LayerNorm, GELU, GRN, and
  residual connections.
- Each physical input channel uses the same CNN weights. A learned residual
  fusion module receives each channel feature and the mean across channels
  after each stage.
- Stages S1/S2/S3/S4 expose logical exits L2/L4/L6/L8. Stage adapters produce
  256-dimensional tokens on the existing 14x14 predictor grid. S1/S2 are
  block-average pooled, S3 remains native, and S4 is nearest-neighbor upsampled
  from 7x7; this duplicates tokens and changes final-stage token weighting.
- Context masks replace hidden 16x16 input patches with the configured fixed
  value before CNN processing. The EMA target sees the original unmasked input. CNN stage blocks use activation checkpointing during training; this trades recomputation for lower memory without changing inference.
- CNN P1, P2, and P2-Reverse use the existing method objectives. Intermediate
  CNN auxiliary losses route gradients only through their supervised stage
  and its input transition; the original L8 anchor remains end-to-end.

## Result-generation interface

`scripts/phase5_primary_report.py --cnn-run-root PATH` expects:

```text
PATH/cnn_p1/seed_000/{downstream/exit_metrics.json,benchmarks.json}
PATH/cnn_p2/seed_000/{downstream/exit_metrics.json,benchmarks.json}
PATH/cnn_p2_reverse/seed_000/{downstream/exit_metrics.json,benchmarks.json}
```
