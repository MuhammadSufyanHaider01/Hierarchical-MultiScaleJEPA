# Phase 3: P1 same-scale multi-depth JEPA

P1 adds auxiliary JEPA losses at L2, L4, and L6 while keeping the pinned
LatentWave L8 objective as the final-layer anchor. The original LatentWave
repository at commit
[`330e9ef5f6eb69e728ba43b3755036cffdd086cd`](https://github.com/AhmedEssam19/latent-wave/tree/330e9ef5f6eb69e728ba43b3755036cffdd086cd)
remains the P0 reference. P1 is an additive extension and must not be used to
rewrite regenerated P0 results.

The preferred Version 1 protocol is defined by
[`hierarchical_design_amendment.md`](hierarchical_design_amendment.md). It
defines P1 as the deep-supervision control for the proposed hierarchical
multi-scale method, not as a claim of standalone novelty.

## Objective

For a training example, the P1 objective is:

```text
L_P1 = L_LW + lambda_H(epoch) / 3 * (L_L2 + L_L4 + L_L6)
```

Each auxiliary predictor is independent and has the same predictor structure
as the pinned P0 predictor. It receives the corresponding masked context
representation from the L2, L4, or L6 context encoder depth and predicts the
same fine-resolution prediction-mask tokens as P0.

The matching target features come from the same depth of the frozen EMA target
encoder on the full input. They use the original prediction masks and retain
the original batch, channel, mask, and embedding ordering. The same physical
mask remains fixed throughout the context path. No region pooling, coarse
geometry, or mask-relative grouping is used in P1; those are Phase 4 P2
behavior.

## Warm-up and logging

The default `configs/p1.yml` uses `lambda_H=1.0` with a 12-epoch linear
warm-up. The first epoch uses `1/12` of the auxiliary weight, and the full
weight is reached after the twelfth epoch. The three raw auxiliary losses are
averaged before the hierarchy weight is applied, so the L8 anchor remains
unscaled and `lambda_H` has the same meaning independent of branch count.

Training logs include the anchor loss, each auxiliary loss, effective
hierarchy weight, valid target-token count, target and predictor RMS, auxiliary
predictor gradient norms, and EMA momentum.

For formal Phase 5 comparison, P1 runs the fixed 240-epoch source horizon with
no raw latent-loss early stopping. On the RTX 5070 Ti it uses the required
memory-safe profile: FP32, microbatch 4, accumulation 16, effective batch 64,
and `PYTORCH_ALLOC_CONF=expandable_segments:True` set before Python starts.
This changes only the microbatch partitioning; the effective batch and
optimizer-step schedule remain aligned with P0.

## Gradient contract

The target encoder is always under `torch.no_grad()` and receives no gradient.
The context encoder, original predictor, and three auxiliary predictors are
optimized. When hierarchy is disabled, no auxiliary predictors are
instantiated and P0's default forward/loss/optimizer path is preserved.

## Version 1 implementation-alignment status

The Version 1 objective is implemented in `JEPA._p1_total_loss`: the three raw
auxiliary losses are averaged before `lambda_H(epoch)` is applied. The P1 smoke
script uses the same helper, and the unit suite includes a numerical regression
test for the `/3` factor. This preserves the unchanged L8 anchor and keeps the
meaning of `lambda_H` independent of the number of auxiliary layers.

## Configuration

Use `configs/p1.yml` for P1 pretraining. The P0 configuration remains
`configs/p0.yml`; it has no hierarchy section and therefore defaults to the
disabled path.
