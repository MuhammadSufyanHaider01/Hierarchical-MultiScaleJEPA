# Phase 3: P1 same-scale multi-depth JEPA

P1 adds auxiliary JEPA losses at L2, L4, and L6 while keeping the pinned
LatentWave L8 objective as the final-layer anchor. The original LatentWave
repository at commit
[`330e9ef5f6eb69e728ba43b3755036cffdd086cd`](https://github.com/AhmedEssam19/latent-wave/tree/330e9ef5f6eb69e728ba43b3755036cffdd086cd)
remains the P0 reference. P1 is an additive extension and must not be used to
rewrite regenerated P0 results.

## Objective

For a training example, the P1 objective is:

```text
L_P1 = L_L8 + lambda_H(epoch) * (L_L2 + L_L4 + L_L6)
```

Each auxiliary predictor is independent and has the same predictor structure
as the pinned P0 predictor. It receives the corresponding masked context
representation from the L2, L4, or L6 context encoder depth and predicts the
same fine-resolution prediction-mask tokens as P0.

The matching target features come from the same depth of the frozen EMA target
encoder on the full input. They use the original prediction masks and retain
the original batch, channel, mask, and embedding ordering. No region pooling,
coarse geometry, or mask-relative grouping is used in P1; those are Phase 4
P2 behavior.

## Warm-up and logging

The default `configs/p1.yml` uses `lambda_H=1.0` with a 12-epoch linear
warm-up. The first epoch uses `1/12` of the auxiliary weight, and the full
weight is reached after the twelfth epoch. The L8 anchor remains unscaled.

Training logs include the anchor loss, each auxiliary loss, effective
hierarchy weight, valid target-token count, target and predictor RMS, auxiliary
predictor gradient norms, and EMA momentum.

## Gradient contract

The target encoder is always under `torch.no_grad()` and receives no gradient.
The context encoder, original predictor, and three auxiliary predictors are
optimized. When hierarchy is disabled, no auxiliary predictors are
instantiated and P0's default forward/loss/optimizer path is preserved.

## Configuration

Use `configs/p1.yml` for P1 pretraining. The P0 configuration remains
`configs/p0.yml`; it has no hierarchy section and therefore defaults to the
disabled path.
