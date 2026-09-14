# Phase 2: frozen P0 representation exits

Phase 2 adds downstream exits to the pinned LatentWave encoder without
changing the P0 pretraining computation. The reference implementation is the
official LatentWave repository at commit
[`330e9ef5f6eb69e728ba43b3755036cffdd086cd`](https://github.com/AhmedEssam19/latent-wave/tree/330e9ef5f6eb69e728ba43b3755036cffdd086cd).
Any regenerated P0 result must use that source path and report this commit;
the exit API is an additive local wrapper, not a replacement for the paper's
baseline.

## Representation API

`Encoder.forward(x, masks=None)` remains the original full-depth path. Passing
`return_layers=[2, 4, 6, 8]` returns a dictionary of full-input token tensors
after those transformer blocks. `forward_to_layer(x, exit_layer)` returns one
full-input representation for downstream use. Every optional
representation applies the encoder's final `LayerNorm` to the selected block
output, while the normalized tensor is never fed into later blocks.

## Heads and freezing

`FrozenEncoderExit` attaches an independent head to one exit. Each head uses
mean pooling over tokens, trainable `LayerNorm`, and a linear task projection.
The whole context backbone is frozen, including patch embedding, fixed
positional embeddings, all transformer blocks, and final normalization. Only
head parameters are optimized. Under the Version 1 hierarchical protocol,
exits wrap the frozen pretrained `context_encoder.*` checkpoint namespace; the
EMA target encoder and every JEPA predictor are discarded after pretraining.

`load_context_encoder_checkpoint` explicitly prefers `context_encoder.*` when
both branches are present in a Lightning checkpoint. It retains a
`target_encoder.*` fallback only for legacy checkpoints that contain no context
state; the target branch is never the preferred downstream source.

`load_target_encoder_checkpoint` is intentionally distinct: it requires the
EMA `target_encoder.*` namespace (or a standalone encoder checkpoint) and
never silently substitutes `context_encoder.*`. It exists only for the pinned
source-style P0 L8 probe, not for the Version 1 early-exit table.

The older `LatentWFM` fine-tuning entrypoint now gives `freeze_encoder()` the
same full-backbone semantics: it freezes patch embedding, every transformer
block, and final normalization while leaving the task head trainable. Its
explicit partial-block mode remains a distinct fine-tuning ablation rather
than a frozen-exit evaluation.

## Tasks and selection

The configured Phase 2 tasks are RF classification, NR outdoor positioning,
DeepMIMO 16-beam prediction, and DeepMIMO LoS/NLoS classification. Exit
selection is performed on validation metrics only. The best validation exit
breaks ties toward the earliest layer. The earliest-within-tolerance rule uses
the first exit within 5% of the best validation metric, with the direction
chosen by task type. Test data is reserved for final reporting.

For formal Phase 5 evaluation, each frozen head has a 200-epoch maximum and
validation-metric early stopping with patience 20. This controls head training
on the fixed labelled validation split; it is unrelated to, and must not be
confused with, raw latent-loss pretraining early stopping.

Run the real-data CPU smoke test with:

```bash
PYTHONPATH=src .venv/bin/python scripts/exit_smoke.py --device cpu --task rfs --steps 1 --batch-size 1
```

On the configured CUDA host, use `--device cuda`.
