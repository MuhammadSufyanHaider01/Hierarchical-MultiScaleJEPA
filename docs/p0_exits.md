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
full-input representation for target-encoder downstream use. Every optional
representation applies the encoder's final `LayerNorm` to the selected block
output, while the normalized tensor is never fed into later blocks.

## Heads and freezing

`FrozenEncoderExit` attaches an independent head to one exit. Each head uses
mean pooling over tokens, trainable `LayerNorm`, and a linear task projection.
The whole target backbone is frozen, including patch embedding, fixed
positional embeddings, all transformer blocks, and final normalization. Only
head parameters are optimized. Target checkpoints are loaded from the
`target_encoder.*` prefix produced by the P0 Lightning module.

## Tasks and selection

The configured Phase 2 tasks are RF classification, NR outdoor positioning,
DeepMIMO 16-beam prediction, and DeepMIMO LoS/NLoS classification. Exit
selection is performed on validation metrics only. The best validation exit
breaks ties toward the earliest layer. The earliest-within-tolerance rule uses
the first exit within 5% of the best validation metric, with the direction
chosen by task type. Test data is reserved for final reporting.

Run the real-data CPU smoke test with:

```bash
PYTHONPATH=src .venv/bin/python scripts/exit_smoke.py --device cpu --task rfs --steps 1 --batch-size 1
```

On the configured CUDA host, use `--device cuda`; the script can also load a
P0 target checkpoint with `--checkpoint`.
