# Phase 4: P2 mask-relative fine-to-coarse JEPA

P2 (B3) is the proposed hierarchical multi-scale extension of the pinned
LatentWave baseline. P0 and P1 remain available without behavior changes.
P2 keeps the original L8 LatentWave objective as its only final-layer anchor
and changes only the L2/L4/L6 auxiliary target construction.

## Objective

```text
L_P2 = L_LW + lambda_H(epoch) / 3 * (L_L2 + L_L4 + L_L6)
```

`L_LW` is the original masked latent-prediction loss. The three auxiliary
losses are mean MSE terms over direct region predictions and their EMA targets.
The default uses `lambda_H=1.0` with a 12-epoch linear warm-up.

## Mask-relative target geometry

P2 derives every target region from each actual prediction mask; it does not
use a global target grid or adaptive pooling. The accepted P2 mask is a
contiguous, complete rectangular block whose height and width are both even.
Every member of every pooled region is asserted to be in that prediction mask.

For a prediction block of height `H` and width `W`:

| Auxiliary depth | Region definition | Vectors per channel |
|---|---|---:|
| L2 | one fine patch per region | `H * W` |
| L4 | non-overlapping local `2x2` regions | `(H / 2) * (W / 2)` |
| L6 | one whole-block region | `1` |

The real-data P2 configuration uses the existing span-time `2x14` block on
the `14x14` token grid, giving 28, 7, and 1 target vectors at L2, L4, and L6.
A `4x4` block gives 16, 4, and 1. Non-rectangular, duplicate, out-of-range,
or odd-sided masks fail loudly rather than silently changing the experiment.

## Target and predictor contracts

The EMA target encoder receives the full input and always runs through all
eight blocks. Selected L2/L4/L6 pre-normalization token grids are reshaped as
`[batch, channel, fine_patch, embedding]`; exact mask-relative members are
gathered, mean pooled, and then functionally LayerNormed. Pooling never occurs
inside the target backbone.

The region layout retains explicit
`[prediction_mask, batch, region, fine_patch]` axes. Pooled targets and direct
predictor outputs retain explicit
`[prediction_mask, context_mask, batch, channel, region, embedding]` axes
until MSE matching. The P2 predictor reuses the P0 context-token path but
builds one query per region from its learned mask token plus the mean of that
region's fine predictor positional embeddings. It predicts one vector per
region directly; predicted fine vectors are never pooled afterward.

The target encoder remains under `no_grad` and is EMA-updated only. The
context encoder, original P0 predictor, and all three auxiliary predictors are
optimized. After pretraining, retain only the trained context encoder for the
frozen downstream exits.

## Configuration and checks

Use `configs/p2.yml`, whose hierarchy mode is
`mask_relative_fine_to_coarse`. Run CPU and RTX 5070 Ti smoke checks with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_p2_regions -v
PYTHONPATH=src .venv/bin/python scripts/p2_smoke.py --device cpu --steps 1
PYTHONPATH=src .venv/bin/python scripts/p2_smoke.py --device cuda --steps 1
```

The P2 unit suite covers hand-computed pooling, canonical `2x14` geometry,
positional-query averaging, malformed-mask rejection, prediction/target axis
matching, auxiliary gradients, and the absence of target gradients.

Phase 5 pilot P2 uses the same fixed 60-epoch horizon as P0, without raw
latent-loss early stopping. Its RTX 5070 Ti profile is FP32, microbatch 4,
accumulation 16, effective batch 64, and
`PYTORCH_ALLOC_CONF=expandable_segments:True` set before Python starts with
the runner's `--memory-safe` check enabled.
