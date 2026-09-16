# Hierarchical JEPA Version 1 design amendment

This is the authoritative design amendment for the Hierarchical LatentWave
implementation plan. It incorporates the refinements supplied in
`hierarchical_latentwave_update_notes.md` on 2026-09-11. Where it conflicts
with an older global-grid, four-auxiliary-loss, or target-backbone downstream
description, this document takes precedence.

## Terminology and research claim

Use **Hierarchical** as the umbrella term:

- **Deep supervision**: JEPA losses at multiple encoder depths.
- **Multi-scale**: different latent target-region scales at different depths.
- **Hierarchical**: deep supervision plus progressive fine-to-coarse latent
  target prediction.

The preferred method name is **Hierarchical Multi-Scale JEPA** or
**Hierarchical JEPA with Fine-to-Coarse Latent Supervision**. It predicts
EMA-target latent features; it is not input-space or "multi-stage"
reconstruction.

The appropriate novelty framing is:

> We introduce hierarchical fine-to-coarse latent-region supervision for a
> wireless JEPA backbone and evaluate whether it makes intermediate
> representations intrinsically more early-exit-ready.

Do not claim novelty merely from multi-depth JEPA supervision, because
intermediate JEPA self-supervision has prior art.

## Fixed-mask, progressive-target invariant

The context encoder uses one masking pattern for its entire depth. The masked
physical area does not shrink and deeper layers do not see less-hidden input.
Only the latent representation used as an auxiliary target becomes coarser.

For a contiguous `4 x 4` masked region, the canonical Version 1 example is:

| Depth | Target scale for the same physical area | Target vectors per channel |
|---|---|---:|
| L2 | `4 x 4` fine latent grid | 16 |
| L4 | `2 x 2` grid after local `2 x 2` pooling | 4 |
| L6 | `1 x 1` region after pooling the full masked block | 1 |
| L8 | Original LatentWave fine objective | Original mask-token count |

For other rectangular masks, the implementation derives regions from each
sample's actual prediction-mask coordinates. The canonical L4 rule is local
`2 x 2` pooling when both dimensions are divisible by two; otherwise it uses
contiguous balanced quadrants with explicit per-region valid member counts.
The canonical L6 rule pools a valid complete masked block into one region.
Unsupported non-rectangular geometry must fail loudly rather than silently use
global/adaptive pooling.

## P1 and P2 objectives

P1 (B2) is the same-scale deep-supervision control. Its L2/L4/L6 targets all
retain the fine representation of the same prediction-mask patches:

```text
L_P1 = L_LW + lambda_H(epoch) / 3 * (L2 + L4 + L6)
```

P2 (B3) is the proposed hierarchical multi-scale method. L2 is fine, L4 uses
mask-aligned local pooling, and L6 uses a mask-aligned coarse region target:

```text
L_P2 = L_LW + lambda_H(epoch) / 3 * (L2 + L4 + L6)
```

`L_LW` is the unchanged original LatentWave L8 objective. There is no primary
coarse L8 auxiliary loss; it is an ablation only. Each `L_l` is mean MSE over
the valid mask-aligned latent regions at that depth, with the EMA target
stop-gradient applied.

The default warm-up remains `lambda_H = 1.0` over 12 epochs. The normalization
by three keeps the effective hierarchy weight independent of the number of
auxiliary depths.

## Target construction and predictor contract

The target encoder remains the original full-depth LatentWave ViT. It is never
pooled internally and is not truncated during pretraining. At selected taps,
its full latent grid is passed to a target-construction side branch:

```text
full target encoder -> tapped latent grid -> extract exact masked area
                   -> mask-aligned pooling -> stop-gradient target
```

Target tapping is not target early exiting. Pooling does not add a projection:
the feature width remains 256 and only spatial target resolution changes.

Each auxiliary predictor directly predicts one vector per target region at its
own scale. It must not predict fine tokens and pool its predictions afterward.
Its query is the learned mask token plus the mean predictor positional
embedding of the fine patches that make up that region.

Keep batch, prediction-mask, channel, region, and embedding axes explicit
until predictions and targets are paired. Every fine patch in a pooled target
region must belong to that sample's prediction mask.

## Gradient and downstream contracts

`L2`, `L4`, and `L6` update their corresponding predictors and all context
blocks up to their tap. `L_LW` updates the original LatentWave predictor and
the complete context encoder. The target encoder receives no gradients and is
updated only by EMA.

After pretraining, discard the target encoder, all auxiliary predictors, the
original JEPA predictor, and all pooling helpers. The primary downstream
backbone is the **pretrained context encoder**, frozen in full. Train identical
mean-pool -> LayerNorm -> Linear task heads at L2/L4/L6/L8. Deployment runs
only through the selected context-encoder exit; later blocks are skipped.

## Baselines and evaluation

| ID | Method | Purpose |
|---|---|---|
| B0 | Original LatentWave | Full-depth self-supervised baseline. |
| B1 | Original LatentWave + frozen early-exit heads | Post-hoc early-exit baseline derived from the P0 checkpoint. |
| B2 | Multi-depth same-scale JEPA | Controlled ablation isolating deep supervision. |
| B3 | Hierarchical Multi-Scale JEPA | Primary proposed method; tests fine-to-coarse target aggregation. |
| B4 | WavesFM | Contextual/deferred unless source, checkpoints, and protocol are comparable. |
| B5 | V-JEPA-style deep supervision | Optional only if it materially differs from B2. |

The central question is whether B3 reaches near-full downstream performance at
an earlier frozen context-encoder depth than B1. B2 is a separate control for
same-scale deep supervision. Report pretraining
compute/memory overhead separately from inference FLOP and latency reduction.
Use fixed validation-selected exits as the primary deployment policy; dynamic
entropy exits remain classification-only secondary evaluation.

## Conceptual lineage

| Source | Contribution to this project |
|---|---|
| LatentWave | Wireless JEPA backbone, shared per-antenna tokenizer, context/EMA-target framework, and final objective. |
| LocalMIM | Multi-depth supervision and fine-to-coarse target-scale philosophy. |
| V-JEPA 2.1 | Prior-art precedent for deep/intermediate JEPA supervision. |
| Fast Wireless Foundation Models with Early-Exits | Frozen-backbone heads and task-specific exit evaluation. |
| Proposed method | Fine-to-coarse EMA latent-region prediction intended to strengthen intermediate wireless features. |

## Documentation and implementation alignment

The Version 1 P1 objective is implemented as the mean of the three raw
auxiliary losses before the epoch-dependent hierarchy weight is applied. A
numerical regression test locks this `/3` factor, so reported P1 results use
the Version 1 objective rather than scaling the auxiliary contribution with
the number of branches. Downstream exits and validation representations now
use the learned `context_encoder`; checkpoint loading explicitly prefers its
namespace and falls back to target-encoder weights only for legacy checkpoints
that lack a context branch. The legacy full-depth `LatentWFM` frozen-head path
also freezes patch embedding and final normalization by default, matching the
complete-backbone frozen-evaluation contract. P2 now derives L2/L4/L6 target
regions from each actual prediction mask, uses target-side pooling only, and
directly predicts one vector per region. Its tests cover hand-computed pooling,
mask leakage, malformed geometry, target gradient absence, and full-depth EMA
target execution; CPU and RTX 5070 Ti smoke runs are finite.
