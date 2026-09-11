# Hierarchical Multi-Scale LatentWave with Early Exits
## Standalone Implementation Specification and Experimental Plan

**Purpose of this document:** This file is intended to be sufficient on its own for implementing the proposed work without referring back to any chat history. It records the motivation, exact architecture, objective-function design, tensor flow, masking/scale logic, downstream early-exit adaptation, baselines, ablations, evaluation protocol, implementation order, sanity checks, and paper/repository references.

**Status:** Proposed research design; implementation has not yet been validated experimentally.

**Last updated:** 2026-09-10

---

## 1. Core research question

The project asks:

> **Can we modify JEPA pretraining so that useful wireless representations emerge earlier in a Transformer encoder, enabling stronger and cheaper task-aware early exits while preserving full-depth performance?**

The base model is **LatentWave**, which learns wireless representations by predicting masked regions in latent space using a JEPA context encoder, an EMA target encoder, and a predictor.

The proposed method adds two ideas:

1. **Multi-depth JEPA supervision:** apply latent-prediction losses at several intermediate encoder depths instead of relying only on the final encoder representation.
2. **Fine-to-coarse latent targets:** inspired by LocalMIM, make shallow layers predict fine/local masked latent regions while deeper layers predict increasingly coarse/larger latent regions.

After self-supervised pretraining, discard the target encoder and all JEPA predictors, freeze the retained context encoder, attach lightweight task heads at several depths, and perform early-exit evaluation following the methodology of **Fast Wireless Foundation Models with Early-Exits**.

The desired causal chain is:

```text
Hierarchical JEPA pretraining
        ↓
stronger intermediate representations
        ↓
frozen lightweight downstream heads at multiple depths
        ↓
earlier task-specific exits
        ↓
lower FLOPs / latency at similar downstream performance
```

---

# 2. Papers and repositories

## 2.1 Primary base paper: LatentWave

**Paper**
- Ahmed Mohamed, Ahmed Aboulfotouh, Hatem Abou-Zeid
- **LatentWave: JEPA Pretraining for Wireless Foundation Models**
- arXiv:2606.06373
- https://arxiv.org/abs/2606.06373
- PDF: https://arxiv.org/pdf/2606.06373

**Official repository**
- https://github.com/AhmedEssam19/latent-wave

**Important current repository files**
- `configs/jepa.yml`
- `src/models/encoder.py`
- `src/models/predictor.py`
- `src/models/jepa.py`
- `src/models/latentwfm.py`
- `src/models/utils.py`
- `src/finetune.py`
- `src/pretrain.py`
- `src/collators/` (mask generation)
- `src/evaluation/`

The current `configs/jepa.yml` defines:
- input resolution: `224 x 224`
- patch size: `16`
- encoder embedding dimension: `256`
- encoder depth: `8`
- encoder attention heads: `8`
- predictor embedding dimension: `128`
- predictor depth: `4`
- predictor attention heads: `4`
- pretraining epochs: `240`
- warmup: `12`
- EMA momentum schedule: `0.996 → 1.0`
- optimizer reference LR: `1e-3`
- starting/final LR: `1e-6`
- weight decay: `0.04`
- current masking config includes multi-block masking.

The current LatentWave JEPA code:
- builds the target encoder as a deep copy of the context encoder;
- freezes target-encoder gradients;
- computes final-layer target representations under `torch.no_grad()`;
- applies LayerNorm to target features;
- extracts prediction-mask locations;
- uses MSE between predictor output and target latent vectors;
- updates the target encoder with EMA after each training batch.

These behaviors should be preserved unless deliberately changed in an ablation.

---

## 2.2 Early-exit paper

**Paper**
- Omar Mashaal, Hatem Abou-Zeid
- **Fast Wireless Foundation Models with Early-Exits**
- arXiv:2606.29640
- https://arxiv.org/abs/2606.29640
- PDF: https://arxiv.org/pdf/2606.29640

Key ideas used in this project:
- freeze the foundation-model encoder during downstream task adaptation;
- attach identical lightweight task heads to multiple encoder exits;
- use global/mean pooling before the task head;
- train the task heads while the backbone remains fixed;
- compare fixed task-specific exits against dynamic entropy-based exits;
- fixed task-specific exit selection can be competitive or better than dynamic sample-wise routing.

The paper's Algorithm 2 is a dynamic entropy-based early-exit policy for classification.

The paper's Algorithm 3 selects one fixed exit per task offline, using either:
- few-shot labeled validation accuracy; or
- average confidence on an unlabeled set.

---

## 2.3 LocalMIM

**Paper**
- Haoqing Wang, Yehui Tang, Yunhe Wang, Jianyuan Guo, Zhi-Hong Deng, Kai Han
- **Masked Image Modeling with Local Multi-Scale Reconstruction**
- CVPR 2023
- arXiv:2303.05251
- https://arxiv.org/abs/2303.05251
- PDF: https://arxiv.org/pdf/2303.05251

**Official repository**
- https://github.com/Haoqing-Wang/LocalMIM

Idea borrowed conceptually:
- shallow layers receive finer/local supervision;
- deeper layers receive coarser/more global supervision;
- intermediate layers receive direct self-supervised objectives.

Important distinction:
- LocalMIM reconstructs **input-space targets** such as HOG/pixels.
- Our method predicts **JEPA latent-space targets** from the EMA target encoder.

---

## 2.4 V-JEPA 2.1 — prior art that affects novelty framing

**Paper**
- Lorenzo Mur-Labadia et al.
- **V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning**
- arXiv:2603.14482
- https://arxiv.org/abs/2603.14482

V-JEPA 2.1 already uses **deep self-supervision at multiple intermediate encoder layers**. Therefore, the novelty of this project must **not** be stated simply as "we apply JEPA loss at intermediate layers."

The stronger framing is:

> **Wireless-specific hierarchical latent prediction co-designed for early-exit-ready representations, with fine-to-coarse target regions and controlled frozen-backbone early-exit evaluation.**

---

# 3. What must remain unchanged from LatentWave

For the main controlled experiment, avoid changing unrelated components.

Keep the following unchanged unless explicitly running an ablation:

- same pretraining datasets;
- same data preprocessing;
- same shared per-channel patch embedding;
- same stochastic antenna/channel sampling;
- same encoder architecture;
- same embedding dimension;
- same Transformer depth;
- same attention-head count;
- same positional encoding style;
- same base masking scheme;
- same EMA target-encoder mechanism;
- same optimizer family and LR schedule;
- same training epochs/budget where feasible;
- same downstream datasets and splits;
- same lightweight head architecture across exits.

The core experimental principle is:

> **Change the self-supervised objective and intermediate prediction branches, not the underlying backbone capacity.**

---

# 4. Base LatentWave architecture

## 4.1 Input

A wireless sample is represented as:

\[
x \in \mathbb{R}^{B \times C \times 224 \times 224}
\]

where:

- \(B\): batch size;
- \(C\): number of selected antenna/channels;
- the last two dimensions represent the 2D wireless map (e.g., frequency × time).

Both spectrograms and CSI are preprocessed into this common image-like form in LatentWave.

---

## 4.2 Shared per-channel patch embedding

Each antenna/channel is patchified separately using a shared single-channel patch embedder.

Patch size:

\[
16 \times 16
\]

Therefore each antenna produces:

\[
224/16 = 14
\]

patches per spatial dimension:

\[
14 \times 14 = 196
\]

patches.

Each patch becomes a 256-dimensional token.

Before flattening across antennas, think of the representation as:

\[
[B, C, 14, 14, 256].
\]

Equivalent flattened token count:

\[
N = 196C.
\]

The same patch projection weights are shared across antennas.

---

## 4.3 Context encoder

The context encoder is an 8-block ViT:

- embedding dimension: `256`
- depth: `8`
- attention heads: `8`
- MLP ratio: `4`
- fixed 2D sine-cosine positional embeddings.

The context encoder sees only visible/context tokens after masking.

For this project, define intermediate context representations:

\[
h_2, h_4, h_6, h_8
\]

where \(h_l\) means the context encoder's output after Transformer block \(l\).

---

## 4.4 Target encoder

The target encoder has the same architecture as the context encoder and begins as a deep copy.

It:
- receives the full/unmasked input;
- does not receive gradient updates;
- is updated using EMA from the context encoder.

Define:

\[
t_2, t_4, t_6, t_8
\]

as target-encoder outputs at the corresponding depths.

The EMA update remains:

\[
W_t \leftarrow \tau W_t + (1-\tau) W_c
\]

where:
- \(W_t\): target-encoder parameters;
- \(W_c\): context-encoder parameters;
- \(\tau\): EMA momentum.

Use the existing LatentWave EMA schedule unless running an ablation.

---

## 4.5 Original LatentWave predictor

Current LatentWave predictor:

```text
encoder latent dimension: 256
        ↓
Linear(256 → 128)
        ↓
mask tokens + positional encoding
        ↓
4 Transformer blocks, 4 heads
        ↓
LayerNorm
        ↓
Linear(128 → 256)
        ↓
predicted masked latent vectors
```

Original final-depth loss is MSE between predicted masked latent vectors and the corresponding target-encoder latent vectors.

---

# 5. Proposed architecture

## 5.1 Supervised depths

Primary candidate depths:

\[
\boxed{\mathcal{D} = \{2,4,6,8\}}
\]

This matches the natural quarter-depth progression of the 8-layer encoder.

Do not claim these layers are optimal until tested. Include depth-choice ablations later if resources permit.

---

## 5.2 Intermediate output requirement

Modify `Encoder.forward()` so it can optionally return selected intermediate representations.

Recommended interface:

```python
final_out, intermediates = encoder(
    x,
    masks=masks,
    return_layers=(2, 4, 6, 8),
)
```

where:

```python
intermediates = {
    2: h2,
    4: h4,
    6: h6,
    8: h8,
}
```

Important:
- preserve the existing default behavior for compatibility;
- do not change standard LatentWave runs;
- apply the same final/intermediate normalization policy consistently.

A practical design is to save the tensor immediately after each selected Transformer block, then apply a dedicated LayerNorm before sending it into the auxiliary JEPA branch.

---

## 5.3 Separate auxiliary predictor per selected depth

Recommended first implementation:

\[
P_2, P_4, P_6, P_8
\]

Use independent predictors rather than sharing parameters.

Reason:
- easiest to interpret;
- each depth solves an independent latent-prediction problem;
- avoids confounding depth-specific behavior with shared predictor capacity;
- all predictors are discarded after pretraining, so they do not affect downstream inference cost.

Start with the same predictor architecture as LatentWave at every depth:

```text
input dimension 256
Linear(256 → 128)
4 Transformer blocks
4 heads
MLP ratio 4
LayerNorm
Linear(128 → 256)
```

Later ablation:
- smaller auxiliary predictors;
- shared predictor with depth embeddings.

Do **not** make shared-predictor complexity part of the first main experiment.

---

# 6. Critical concept: mask geometry vs target scale

These are different.

## 6.1 Mask geometry

The mask specifies:

> **Which original wireless patches are hidden from the context encoder?**

Keep LatentWave's masking policy fixed across all depths in the main experiment.

Example original mask on the `14 × 14` grid:

```text
visible visible masked masked ...
visible visible masked masked ...
...
```

The same time-frequency positions are masked consistently across antenna channels, as in LatentWave.

---

## 6.2 Target scale

The target scale specifies:

> **How large a masked region is represented by one JEPA prediction at a given depth?**

This is where the LocalMIM-inspired fine-to-coarse design enters.

Important:

> **Pooling is not required for dimensional compatibility.**

If all depths use fine targets, every depth can directly predict the normal `14 × 14` latent grid.

Pooling is introduced only because we deliberately want deeper layers to predict **larger latent regions**.

---

# 7. Baseline multi-depth JEPA: same-scale targets

Before implementing multi-scale targets, implement a simpler multi-depth baseline.

At every selected layer:

\[
h_l \rightarrow P_l \rightarrow \hat{t}_l
\]

and compare against target features at the same original patch resolution.

All depths therefore use the original fine target scale:

```text
Layer 2 → 14 × 14 targets
Layer 4 → 14 × 14 targets
Layer 6 → 14 × 14 targets
Layer 8 → 14 × 14 targets
```

For masked location \(j\), layer-\(l\) loss:

\[
L_l^{\text{fine}} =
\frac{1}{N_m}
\sum_{j \in M}
\left\|
\hat{t}_{l,j} -
\operatorname{sg}(t_{l,j})
\right\|_2^2
\]

where:
- \(M\): masked positions;
- \(N_m = |M|\);
- `sg` means stop-gradient.

This baseline isolates the effect of **multi-depth/deep JEPA supervision** without multi-scale pooling.

It is essential because V-JEPA 2.1 already motivates deep self-supervision.

---

# 8. Proposed multi-depth, multi-scale JEPA

## 8.1 Central idea

Use progressively larger latent target regions at deeper layers.

Initial recommended target grids:

```text
Layer 2 → 14 × 14 target grid
Layer 4 →  7 × 7 target grid
Layer 6 →  4 × 4 target grid
Layer 8 →  2 × 2 target grid
```

This sequence is an **initial engineering choice**, not a theoretical requirement.

Use `AdaptiveAvgPool2d` for non-divisible cases such as `14 → 4` and `14 → 2`.

The purpose is:

```text
shallow depth → fine/local latent prediction
deeper depth  → larger/coarser latent-region prediction
```

Do not describe this as "making dimensions compatible." The correct rationale is **depth-dependent abstraction**.

---

## 8.2 Target-side pooling only

The target encoder naturally produces one 256-D vector per original patch.

For a `2 × 2` group of target vectors:

\[
t_a,\; t_b,\; t_c,\; t_d
\]

construct a single coarse target:

\[
\bar t = \frac{t_a+t_b+t_c+t_d}{4}.
\]

The predictor should directly output **one vector for the coarse region**.

Preferred design:

```text
Target side:
4 fine target vectors
        ↓ average
1 coarse target vector

Predictor side:
context features + 1 coarse-region query
        ↓
1 predicted coarse vector

Loss:
MSE(predicted coarse vector, pooled target vector)
```

Do **not** make four fine predictions and pool them afterward in the main design. That would still make the predictor solve a fine-scale problem.

---

## 8.3 Pooling functions

Define a target pooling function for each selected layer:

\[
A_2(t_2) = t_2
\]

\[
A_4(t_4) = \text{AdaptiveAvgPool}_{7\times7}(t_4)
\]

\[
A_6(t_6) = \text{AdaptiveAvgPool}_{4\times4}(t_6)
\]

\[
A_8(t_8) = \text{AdaptiveAvgPool}_{2\times2}(t_8)
\]

Implementation requirement:

Before pooling, reshape target features from flattened antenna/token form back to something equivalent to:

```python
[B, C, H_grid, W_grid, D]
```

For the base model:

```text
H_grid = W_grid = 14
D = 256
```

A convenient PyTorch layout for spatial pooling is:

```python
[B*C, D, 14, 14]
```

Then:

```python
pooled = F.adaptive_avg_pool2d(features, output_size=(g_l, g_l))
```

and reshape as needed.

---

# 9. Coarse mask construction

The original mask is defined on the fine `14 × 14` grid.

Let:

\[
m \in \{0,1\}^{14\times14}
\]

where:
- `1` = masked/target patch;
- `0` = visible/non-target patch.

For each target scale, pool the mask using the same spatial mapping used for the target features.

Recommended conservative rule for first implementation:

> A coarse region contributes to the auxiliary loss only if **all** fine patches represented by that coarse region are masked.

This prevents visible information from leaking into a target region.

For clean divisible pooling, this is easy.

For adaptive pooling (`14 → 4`, `14 → 2`), be careful: adaptive pooling bins may not all contain equal numbers of original cells. Implement region membership explicitly or derive the adaptive pooling boundaries so that a coarse region is valid only if all underlying fine-mask values are `1`.

A simpler first implementation option is to choose only target scales that divide 14 cleanly, e.g.:

```text
14 × 14
7 × 7
2 × 2
1 × 1
```

However, the `14, 7, 4, 2` sequence gives a smoother scale progression. If using it, test mask-region mapping carefully.

**Recommendation:** implement target-region mapping explicitly rather than relying on thresholding the adaptively pooled binary mask.

---

# 10. Coarse positional queries

The predictor needs a query representing the location of each target region.

For fine targets:
- use the standard masked-patch positional encoding.

For a coarse target region:
- obtain the set of fine-grid positions belonging to that region;
- average their predictor positional embeddings.

If region \(r\) contains positions \(p_1, p_2, \dots, p_n\):

\[
p_r = \frac{1}{n}\sum_i p_i
\]

Construct the predictor query:

\[
q_r = q_{\text{mask}} + p_r
\]

where \(q_{\text{mask}}\) is the learned predictor mask token.

The auxiliary predictor then receives:
- the context features at layer \(l\);
- coarse-region query tokens \(q_r\).

It directly outputs one 256-D vector per coarse region.

---

# 11. Intermediate representation normalization

Recommended:

Before each auxiliary predictor branch:

```python
h_l_norm = F.layer_norm(h_l, (h_l.size(-1),))
```

On the target side:

```python
t_l_norm = F.layer_norm(t_l, (t_l.size(-1),))
```

Then pool the normalized target features:

```python
target_l = A_l(t_l_norm)
```

Reason:
- target distributions may differ across depth;
- normalization makes the auxiliary regression objective more stable;
- this is consistent with LatentWave's existing final target LayerNorm behavior.

Ablation later:
- normalization before vs after target pooling;
- per-layer learned LayerNorm modules vs functional LayerNorm.

For the first implementation, use a consistent simple policy and document it.

---

# 12. Per-layer auxiliary objective

For selected layer \(l\):

\[
\boxed{
L_l =
\frac{1}{N_l}
\sum_{r \in M_l}
\left\|
P_l(h_l, q_r)
-
\operatorname{sg}\left[A_l(t_l)_r\right]
\right\|_2^2
}
\]

Interpretation:
- \(h_l\): context representation at depth \(l\);
- \(P_l\): depth-specific predictor;
- \(q_r\): query for masked target region \(r\);
- \(A_l(t_l)_r\): pooled target-encoder representation for region \(r\);
- \(M_l\): valid target regions at depth \(l\);
- \(N_l = |M_l|\);
- target is stop-gradient.

If implementing across batch and channels, average over all:
- batch items;
- selected antenna channels;
- valid target regions;
- feature dimensions.

Use MSE, as LatentWave already does.

---

# 13. Preserve original LatentWave loss as an anchor

Do **not** replace the original final-layer LatentWave loss in the main model.

Let:

\[
L_{\text{LW}}
\]

be the unmodified original LatentWave final-layer JEPA MSE at the standard fine target scale.

Reasons:
1. preserves full-depth LatentWave training behavior;
2. reduces risk of hurting the final representation;
3. gives a cleaner causal story;
4. allows the hierarchical objective to be interpreted as an auxiliary addition;
5. makes comparison against original LatentWave more defensible.

The proposed layer-8 coarse auxiliary loss is **additional** to the original fine final-layer LatentWave loss.

Therefore layer 8 gets:
- original fine LatentWave supervision; and
- proposed coarse hierarchical supervision.

---

# 14. Hierarchical auxiliary loss

Average the auxiliary losses:

\[
\boxed{
L_{\text{H}} =
\frac{1}{4}
\left(
L_2 + L_4 + L_6 + L_8
\right)
}
\]

Use an average rather than a raw sum so that adding four branches does not automatically scale the total loss by approximately 4×.

---

# 15. Final pretraining objective

The recommended main objective is:

\[
\boxed{
L_{\text{total}}
=
L_{\text{LW}}
+
\lambda_{\text{H}} L_{\text{H}}
}
\]

Expanded:

\[
\boxed{
L_{\text{total}}
=
L_{\text{LW}}
+
\frac{\lambda_{\text{H}}}{4}
\left(
L_2 + L_4 + L_6 + L_8
\right)
}
\]

Start with:

```text
lambda_H = 1.0
```

Recommended ablation:

```text
lambda_H ∈ {0.25, 0.5, 1.0, 2.0}
```

Do not add cross-antenna relational losses, compute-aware loss weights, distillation, VICReg, SIGReg, etc. in the first core implementation. Add them only later as controlled extensions if needed.

---

# 16. Optional hierarchical-loss warm-up

Potential stability improvement:

During the first `T_w` epochs, ramp the hierarchical auxiliary weight from 0 to `lambda_H_max`.

\[
\lambda_{\text{H}}(e)
=
\lambda_{\max}
\cdot
\min\left(1, \frac{e}{T_w}\right)
\]

Suggested first test:

```text
T_w = 12 epochs
```

because LatentWave already uses a 12-epoch LR warm-up.

This is an implementation/stability mechanism, **not** the core novelty.

Run an ablation if it materially affects results.

---

# 17. Gradient flow

Target encoder:
- no gradient;
- updated only by EMA.

Context encoder:
- receives gradients from the original LatentWave loss;
- also receives gradients from every auxiliary loss whose tap is at or deeper than a given layer.

Conceptually:

```text
Context layer 1 gets:
L_LW + L2 + L4 + L6 + L8

Context layer 3 gets:
L_LW + L4 + L6 + L8

Context layer 5 gets:
L_LW + L6 + L8

Context layer 7 gets:
L_LW + L8
```

Predictors:
- `P2` receives gradient only from `L2`;
- `P4` only from `L4`;
- `P6` only from `L6`;
- `P8` only from `L8`;
- original LatentWave predictor receives gradient from `L_LW`.

This is exactly why early/intermediate context layers should receive stronger direct learning signals.

---

# 18. Architecture diagram

```text
                           FULL INPUT x
                               │
                       ┌───────┴────────┐
                       │                │
                Context branch     Target branch
                  (masked)           (full)
                       │                │
                    Block 1          Block 1
                       │                │
                    Block 2          Block 2
                       │ h2             │ t2
                       ├──→ P2 ─────────┤→ fine target
                       │                │
                    Block 3          Block 3
                       │                │
                    Block 4          Block 4
                       │ h4             │ t4
                       ├──→ P4 ─────────┤→ pooled 7×7 target
                       │                │
                    Block 5          Block 5
                       │                │
                    Block 6          Block 6
                       │ h6             │ t6
                       ├──→ P6 ─────────┤→ pooled 4×4 target
                       │                │
                    Block 7          Block 7
                       │                │
                    Block 8          Block 8
                       │ h8             │ t8
                       ├──→ P8 ─────────┤→ pooled 2×2 target
                       │                │
                       └→ original LatentWave fine predictor/target
```

Target-branch outputs are stop-gradient.

---

# 19. Repository implementation plan

## 19.1 `src/models/encoder.py`

Current behavior:
- patch embedding;
- positional embedding;
- mask application;
- flatten tokens across antennas;
- run blocks;
- final LayerNorm;
- return final output.

Required changes:
1. add optional `return_layers`;
2. capture outputs after selected blocks;
3. preserve current final output behavior;
4. ensure intermediate features retain enough metadata to restore antenna/spatial layout when needed.

Recommended signature:

```python
def forward(
    self,
    x,
    masks=None,
    return_layers=None,
):
    ...
    return final_out
```

or, when requested:

```python
return final_out, intermediates
```

Alternative cleaner API:

```python
def forward_features(...):
    ...
```

and keep `forward()` backward-compatible.

Important test:
- with `return_layers=None`, output must numerically match original LatentWave.

---

## 19.2 `src/models/predictor.py`

Do **not** immediately rewrite the existing predictor.

Create either:
- `HierarchicalPredictor`, or
- reuse `Predictor` with support for externally supplied target-query positions.

Need functionality to:
1. accept context features from arbitrary encoder depth;
2. accept a custom set of coarse query positional embeddings;
3. generate exactly one output vector per coarse target region;
4. project back to 256 dimensions.

Recommended first implementation:
- instantiate four independent `Predictor`-like modules;
- add a method allowing explicit query embeddings instead of only standard fine masks.

Do not pool predictor outputs after prediction in the proposed multi-scale method.

---

## 19.3 `src/models/jepa.py`

This is the main modification point.

Current `JEPA.forward()`:
- context encoder on masked input;
- predictor from final context output;
- target encoder under `no_grad`;
- LayerNorm target output;
- select target masks;
- return predictor and target features.

Proposed new flow:

```python
context_final, context_intermediates = context_encoder(
    x,
    masks_enc,
    return_layers=(2,4,6,8),
)

with torch.no_grad():
    target_final, target_intermediates = target_encoder(
        x,
        masks=None,
        return_layers=(2,4,6,8),
    )

# original LatentWave loss path remains unchanged
pred_lw, target_lw = original_final_jepa_path(...)

# auxiliary hierarchical paths
losses_aux = {}
for l in (2,4,6,8):
    h = normalize(context_intermediates[l])
    t = normalize(target_intermediates[l])

    target_regions, region_queries, valid_regions = build_scaled_targets(
        t,
        original_prediction_mask,
        target_grid=grid_by_layer[l],
    )

    pred_regions = predictors[str(l)](
        h,
        region_queries=region_queries,
        ...
    )

    losses_aux[l] = mse_on_valid_regions(
        pred_regions,
        target_regions.detach(),
    )

L_H = mean(losses_aux.values())
L_total = L_LW + lambda_H * L_H
```

EMA update stays unchanged.

---

## 19.4 New helper module recommended

Create something such as:

```text
src/models/hierarchical_targets.py
```

Functions:

```python
reshape_tokens_to_grid(...)
adaptive_pool_latents(...)
build_region_membership(...)
build_coarse_mask(...)
build_coarse_positional_queries(...)
build_scaled_target(...)
```

Keeping this logic outside `jepa.py` will make debugging much easier.

---

## 19.5 Configuration additions

Create a new config, e.g.:

```text
configs/jepa_hierarchical.yml
```

Possible section:

```yaml
hierarchical:
  enabled: true
  layers: [2, 4, 6, 8]
  target_grids:
    2: 14
    4: 7
    6: 4
    8: 2
  lambda: 1.0
  warmup_epochs: 12
  predictor:
    embed_dim: 128
    depth: 4
    num_heads: 4
  require_fully_masked_region: true
```

For same-scale ablation:

```yaml
target_grids:
  2: 14
  4: 14
  6: 14
  8: 14
```

---

# 20. Pretraining experiments

## Experiment P0 — Original LatentWave

Exactly reproduce original JEPA.

Purpose:
- verify code/data pipeline;
- obtain baseline checkpoint and metrics;
- confirm published trends.

Objective:

\[
L = L_{\text{LW}}
\]

No auxiliary predictors.

---

## Experiment P1 — Multi-depth JEPA, same scale

Use intermediate losses at layers 2/4/6/8.

All targets stay on the fine `14 × 14` grid.

Objective:

\[
L = L_{\text{LW}} +
\frac{\lambda_H}{4}(L_2+L_4+L_6+L_8)
\]

with:

```text
target grids = [14,14,14,14]
```

Purpose:
- isolate benefit of deep JEPA supervision.

---

## Experiment P2 — Proposed multi-depth + multi-scale JEPA

Same objective form, but target scales:

```text
[14,7,4,2]
```

Purpose:
- test whether fine-to-coarse latent targets improve intermediate representation quality beyond deep supervision alone.

---

## Experiment P3 — Optional scale ablation

Examples:
- `[14, 14, 7, 7]`
- `[14, 7, 7, 2]`
- `[14, 7, 2, 1]`
- `[14, 14, 14, 14]`
- reverse coarse-to-fine (negative-control experiment).

Do this only after P0/P1/P2 work reliably.

---

# 21. Downstream early-exit protocol

## 21.1 What is retained after pretraining?

Discard:
- target encoder;
- original JEPA predictor;
- auxiliary predictors `P2/P4/P6/P8`.

Keep:
- pretrained **context encoder only**.

This is the wireless foundation-model backbone.

---

## 21.2 Freeze the backbone

For the main early-exit evaluation:

\[
\boxed{\text{freeze the entire pretrained context encoder}}
\]

Only task heads are optimized.

This follows the early-exit paper's evaluation philosophy and cleanly isolates representation quality.

Do **not** use backbone fine-tuning as the primary result.

Optional secondary result:
- full or partial backbone fine-tuning;
- clearly label it as a separate experiment.

---

## 21.3 Candidate exits

Primary exits:

```text
Layer 2
Layer 4
Layer 6
Layer 8
```

For each downstream task, attach an identical lightweight head at each exit.

---

## 21.4 Exit head architecture

To stay close to LatentWave and early-exit methodology:

```text
intermediate token representation
        ↓
mean pooling / GAP over tokens
        ↓
LayerNorm
        ↓
Linear(256 → task output dimension)
```

For classification:
- output dimension = number of classes;
- cross-entropy training.

For regression:
- output dimension = regression target dimension;
- MSE or Huber loss depending on the LatentWave task protocol.

Keep head capacity identical across exits.

---

## 21.5 Train heads independently

For strict methodological cleanliness, train each exit head independently while the same encoder checkpoint remains frozen.

For task \(q\) and exit \(l\):

\[
u_l = \text{MeanPool}(h_l)
\]

Classification logits:

\[
z_{q,l} = H_{q,l}(u_l)
\]

Cross-entropy:

\[
L^{\text{task}}_{q,l}
=
-\frac{1}{N}
\sum_i \log p_{q,l}^{(i)}(y_i)
\]

Regression:

\[
L^{\text{task}}_{q,l}
=
\frac{1}{N}\sum_i
\left\|
\hat y_{q,l}^{(i)} - y_i
\right\|^2
\]

The backbone has `requires_grad=False`.

Joint head training is mathematically possible because the heads do not share parameters, but independent training more closely matches the early-exit paper and simplifies fairness.

---

# 22. Fixed task-specific early exit

This should be the **primary deployment strategy**.

## 22.1 Exact early-exit-paper-style selector

For a labeled few-shot/validation set:

\[
l_q^\star = \arg\max_l A_q(l)
\]

where \(A_q(l)\) is validation performance at exit \(l\).

Break performance ties toward the smaller/earlier exit.

For an unlabeled set on classification tasks:
- compute average maximum softmax confidence at each exit;
- select the exit with highest average confidence;
- equivalently compare average uncertainty/entropy as in the early-exit paper.

---

## 22.2 Recommended compute-aware variant for this project

In addition to reproducing the paper's selector, report an **earliest-within-tolerance** rule:

For a higher-is-better metric:

\[
\boxed{
l_q^\star =
\min l
\quad
\text{s.t.}
\quad
Q_q(l) \ge Q_q(8)-\epsilon_q
}
\]

For a lower-is-better error metric:

\[
\boxed{
l_q^\star =
\min l
\quad
\text{s.t.}
\quad
E_q(l) \le E_q(8)+\epsilon_q
}
\]

Interpretation:
- find the earliest/cheapest layer whose task performance is within an allowed degradation of the full model.

This directly matches the efficiency goal.

Report multiple tolerances, e.g.:

```text
epsilon = 0
epsilon = 0.5 percentage points
epsilon = 1 percentage point
epsilon = 2 percentage points
```

For regression use task-appropriate absolute or relative error tolerances.

---

# 23. Dynamic sample-wise early exit (secondary)

For classification only, reproduce the entropy policy from the early-exit paper.

At exit \(l\):

\[
H_l = -\sum_c p_l(c)\log p_l(c)
\]

Maximum entropy for \(K\) classes:

\[
H_{\max} = \ln K
\]

Normalized threshold:

\[
T_l = \bar\tau \ln K
\]

If:

\[
H_l \le T_l
\]

exit; otherwise continue deeper.

Sweep \(\bar\tau\) on validation data.

Report:
- accuracy;
- average depth;
- FLOPs;
- measured latency;
- exit distribution.

Do **not** make dynamic routing the main contribution unless it clearly improves the fixed policy.

For regression tasks, entropy is not directly applicable. Prefer the fixed task-specific exit as the primary method. Dynamic regression uncertainty can be a future extension.

---

# 24. Baselines

## Baseline B0 — Published LatentWave numbers

Use the LatentWave paper's reported downstream results only as external context.

Do not use published numbers as the sole direct baseline because:
- training seed/splits may differ;
- our early-exit heads may use a different protocol;
- intermediate-layer results are not necessarily published for every task.

---

## Baseline B1 — Re-run original LatentWave, full depth

Use the official repository and same datasets/config.

Evaluate the standard final encoder representation.

This verifies reproduction and gives an apples-to-apples final-depth baseline.

---

## Baseline B2 — Original LatentWave + frozen early-exit heads

This is the **most important baseline**.

Procedure:
1. pretrain or load original LatentWave;
2. freeze its context encoder;
3. attach identical heads at layers 2/4/6/8;
4. train heads with identical downstream hyperparameters;
5. evaluate performance and compute at every exit.

This tests **post-hoc early exit on standard LatentWave**.

Our proposed method must ideally dominate this baseline in the performance-vs-compute tradeoff.

---

## Baseline B3 — Multi-depth JEPA, same scale

Pretraining:
- direct JEPA losses at layers 2/4/6/8;
- no coarse target pooling;
- all auxiliary targets use original `14 × 14` scale.

Purpose:
- determine how much of the improvement comes simply from deep supervision.

---

## Baseline B4 — Proposed multi-depth + multi-scale JEPA

This is the full proposed method.

Compare B4 against B3 to isolate the value of fine-to-coarse latent target construction.

---

## Baseline B5 — WavesFM / masked-modeling WFM

LatentWave already compares with WavesFM, a masked-modeling baseline pretrained on the same data.

If code/checkpoints and exact data protocol are available, include it.

At minimum, report its published final-depth values for context.

If early-exit instrumentation is feasible, add the same frozen intermediate heads; otherwise clearly state that only full-depth results are compared.

---

## Optional baseline B6 — V-JEPA-style deep supervision variant

If feasible, implement a variant closer to generic deep JEPA supervision (without LocalMIM-style coarse target regions).

This is especially useful for novelty defense because V-JEPA 2.1 already uses deep self-supervision.

---

# 25. Required ablations

Minimum ablation set:

1. **Original LatentWave**
2. **Multi-depth, same scale**
3. **Multi-depth, multi-scale**
4. `lambda_H` sweep
5. target-scale schedule sweep (resources permitting)
6. selected depth set (resources permitting)
7. auxiliary predictor capacity (only if reviewers/experiments indicate sensitivity)

Useful negative controls:
- reverse scale schedule (coarse early, fine deep);
- only one intermediate loss;
- remove original LatentWave anchor loss;
- pool prediction after fine prediction instead of direct coarse prediction.

Do not run every possible ablation before validating the basic pipeline.

---

# 26. Downstream tasks

Use the same LatentWave downstream tasks where possible:

- RF signal classification;
- 5G NR positioning;
- beam prediction;
- LoS/NLoS classification.

For every task and every pretrained backbone, evaluate each exit:

```text
L2
L4
L6
L8
```

Keep:
- train/validation/test splits fixed;
- head architecture fixed;
- optimizer fixed;
- epochs fixed;
- random seeds fixed;
- pooling operation fixed.

---

# 27. Main result tables

## 27.1 Representation quality table

Example structure:

| Pretraining Method | Task | L2 | L4 | L6 | L8 |
|---|---|---:|---:|---:|---:|
| LatentWave | RF classification | ... | ... | ... | ... |
| Multi-depth same-scale | RF classification | ... | ... | ... | ... |
| Proposed multi-scale | RF classification | ... | ... | ... | ... |

Repeat for every task.

This is arguably the most important table because the encoder is frozen.

---

## 27.2 Exit cost table

Measure cumulative cost to each exit:

| Exit | Encoder Blocks Used | FLOPs | Params Executed | Mean Latency |
|---|---:|---:|---:|---:|
| L2 | 2/8 | ... | ... | ... |
| L4 | 4/8 | ... | ... | ... |
| L6 | 6/8 | ... | ... | ... |
| L8 | 8/8 | ... | ... | ... |

Do not assume cost scales exactly linearly with depth; measure it.

---

## 27.3 Selected exits by task

| Method | Task | Selected Exit | Full Performance | Exit Performance | FLOP Saving | Latency Saving |
|---|---|---|---:|---:|---:|---:|
| LatentWave + EE | ... | ... | ... | ... | ... | ... |
| Proposed + EE | ... | ... | ... | ... | ... | ... |

---

# 28. Main figures

## Figure A — Performance vs encoder depth

For each downstream task:

```text
x-axis: encoder depth / exit
y-axis: task performance
curves:
- original LatentWave
- multi-depth same-scale
- proposed multi-scale
```

Desired observation:
- proposed curve rises earlier;
- final-depth performance is preserved.

---

## Figure B — Performance vs FLOPs

```text
x-axis: cumulative inference FLOPs
y-axis: task performance
```

Desired result:
- proposed model's curve shifts **up and left**.

This is likely the strongest figure for the paper.

---

## Figure C — Performance vs measured latency

Use actual hardware latency.

Measure:
- warm-up iterations;
- synchronized GPU timing;
- fixed batch size(s);
- mean and variance;
- multiple runs.

---

## Figure D — Exit selection across tasks

Bar/table showing which depth each task prefers.

This supports the claim that different tasks require different representation depth.

---

# 29. Metrics

## Pretraining diagnostics

Track:
- total loss;
- original LatentWave loss;
- each auxiliary loss `L2/L4/L6/L8`;
- hierarchical loss;
- EMA momentum;
- learning rate;
- effective rank (LatentWave already logs representation diagnostics);
- representation per-dimension standard deviation;
- gradient norm by encoder block if feasible.

Watch for:
- one auxiliary loss dominating;
- representation collapse;
- shallow losses going to zero while downstream probes do not improve;
- final LatentWave loss degrading significantly.

---

## Downstream metrics

Use the LatentWave paper's task-specific metrics where possible.

At minimum:
- classification accuracy for classification tasks;
- positioning error/RMSE or paper-equivalent metric for positioning;
- beam prediction metric consistent with LatentWave;
- LoS/NLoS accuracy.

Also report:
- FLOPs;
- measured latency;
- cumulative executed parameters/blocks;
- relative compute saving vs L8;
- exit depth.

---

# 30. Fairness rules

For every direct baseline comparison:

- same dataset;
- same split;
- same preprocessing;
- same random seed set;
- same backbone architecture;
- same embedding dimension;
- same head architecture;
- same frozen-backbone policy;
- same head optimizer;
- same head training epochs;
- same metric computation;
- same hardware for latency;
- same batch size for latency;
- same exit selection data;
- never use test data for exit selection.

Published-paper values should be labeled clearly as **reported**, while re-run values should be labeled **reproduced**.

---

# 31. Recommended seeds

At minimum:

```text
3 seeds
```

Prefer:

```text
5 seeds
```

for the most important downstream results.

Report:

\[
\text{mean} \pm \text{std}
\]

especially because small early-exit differences may otherwise be noise.

---

# 32. Pretraining implementation order

Implement in this order.

## Step 1 — Reproduce original LatentWave

Before changes:
- run a short pretraining sanity test;
- load/evaluate existing checkpoint if available;
- record baseline outputs/losses.

Do not proceed until reproduction is reliable.

---

## Step 2 — Add intermediate feature extraction only

Modify encoder to return layers 2/4/6/8.

Checks:
- original final output unchanged;
- intermediate tensor shapes correct;
- gradients work;
- masked and unmasked encoder modes work.

No new losses yet.

---

## Step 3 — Same-scale auxiliary JEPA

Add `P2/P4/P6/P8`.

Use original fine target masks at every depth.

Train for a short run.

Check:
- all auxiliary losses decrease;
- target encoder remains gradient-free;
- EMA update remains correct;
- original LatentWave loss still behaves normally.

---

## Step 4 — Implement target pooling helpers

Build:
- target grid reshape;
- region mapping;
- coarse target pooling;
- coarse valid-mask mapping;
- coarse positional-query construction.

Unit-test this separately.

---

## Step 5 — Multi-scale training

Turn on:

```text
[14,7,4,2]
```

Verify:
- exact number of target queries;
- predictor output shape matches target shape;
- no visible region accidentally participates as a masked coarse target;
- loss normalization does not depend strongly on number of regions.

---

## Step 6 — Full pretraining runs

Run:
- P0;
- P1;
- P2.

Only then expand to ablations.

---

# 33. Unit tests / sanity checks

## Encoder

- final output before/after code modification should match for identical weights/input when intermediate return is disabled;
- intermediate shapes should be correct at 2/4/6/8;
- target encoder should have `requires_grad=False`.

## Masking

- same time-frequency mask across selected antenna channels;
- context encoder never receives target patches;
- coarse target region includes only intended fine-grid patches;
- valid coarse target requires full masking under the main rule.

## Pooling

For synthetic known values, confirm manually.

Example:

```text
fine region:
1 2
3 4

pooled target = 2.5
```

Test separately for each grid transition.

## Predictor

- layer 4 should output one vector per valid 7×7 target region, not four fine vectors that are pooled later;
- output embedding dimension must be 256;
- each auxiliary predictor must only receive the corresponding `h_l`.

## Gradients

Check:
- `P2` gradient nonzero from `L2`;
- encoder block 2 gradient includes `L2`;
- encoder block 4 has no contribution from `L2`;
- target encoder gradient always `None`;
- original LatentWave predictor still trains from `L_LW`.

## EMA

Verify numerically on a toy parameter:

```text
new_target = tau * old_target + (1-tau) * context
```

---

# 34. Potential pitfalls

## 34.1 Target pooling does not automatically create "better" semantics

Fine-to-coarse targets are a hypothesis.

Do not claim coarser targets are richer until downstream experiments show it.

The correct language is:

> "We encourage increasingly spatially aggregated latent prediction with depth."

---

## 34.2 Multi-depth gains may come entirely from extra supervision

This is why P1 (same-scale) is mandatory.

If P1 ≈ P2, then the multi-scale component is not adding much.

That is still scientifically useful.

---

## 34.3 Final-layer performance may degrade

Possible causes:
- auxiliary losses too strong;
- competing gradient directions;
- overly aggressive coarse targets.

Mitigations:
- preserve `L_LW`;
- reduce `lambda_H`;
- auxiliary warm-up;
- change scale schedule;
- remove layer-8 coarse auxiliary if necessary and keep only original final loss at L8.

A legitimate alternative formulation if L8 double supervision hurts:

\[
L_{\text{total}}
=
L_{\text{LW}}
+
\frac{\lambda_H}{3}
(L_2+L_4+L_6)
\]

Then the full depth is anchored only by standard LatentWave.

This should be an ablation, not the first assumption.

---

## 34.4 Number of valid coarse regions may become too small

If the mask consists of small scattered regions, the "all fine patches must be masked" rule may leave very few coarse targets.

Solutions:
1. use LatentWave multi-block masks that naturally contain contiguous regions;
2. adjust target scale;
3. allow a high masked-fraction threshold (e.g. 0.75) only as a later ablation;
4. build coarse targets aligned to actual prediction-mask blocks.

Preferred: align coarse target construction with the multi-block mask geometry.

---

## 34.5 Adaptive pooling region definitions

`AdaptiveAvgPool2d(4)` on a 14-grid has uneven bin sizes.

For correctness, compute explicit mapping from fine indices to each output bin.

Never assume every `14 → 4` cell corresponds to exactly the same number of fine patches.

---

## 34.6 Antenna flattening

LatentWave flattens antenna tokens into one Transformer sequence.

When constructing pooled targets, restore the antenna axis first.

Do not accidentally pool across antennas unless explicitly designing a cross-antenna objective.

Spatial pooling should be within each antenna's `14 × 14` grid.

---

## 34.7 Positional encoding for coarse targets

Do not use arbitrary new coarse positions without checking consistency.

Recommended:
- average fine predictor positional embeddings belonging to the target region.

---

## 34.8 Pretraining compute increases

Four predictors add training cost and memory, even though inference cost is unchanged.

Report:
- pretraining throughput;
- GPU memory;
- training FLOPs or wall-clock;
- final inference cost.

A reviewer may ask whether inference savings justify extra pretraining cost.

---

# 35. Early-exit implementation details in LatentWave code

The existing `LatentWFM` in `src/models/latentwfm.py` applies:

```text
Encoder final output
→ token mean
→ LayerNorm
→ Linear head
```

For early exits, create a model that can stop after an arbitrary number of blocks.

Suggested class:

```python
class LatentWFMExit(nn.Module):
    def __init__(self, frozen_encoder, exit_layer, num_outputs):
        ...
```

Forward:

```python
tokens = patch_and_position(x)

for i, block in enumerate(encoder.blocks, start=1):
    tokens = block(tokens)
    if i == exit_layer:
        break

tokens = exit_norm(tokens)
pooled = tokens.mean(dim=1)
output = head(pooled)
return output
```

Important:
- handle encoder final norm consistently across exits;
- either give each exit its own LayerNorm or apply a consistent frozen normalization strategy;
- keep heads identical.

For strict representation probing, a trainable exit-specific LayerNorm can be considered part of the lightweight head, but use the same design at every exit.

---

# 36. FLOP and latency measurement

Do not estimate inference improvement only as:

```text
2/8 = 25%
4/8 = 50%
...
```

Self-attention/token count and patch embedding cost matter.

Compute actual cumulative FLOPs.

Tools:
- `fvcore.nn.FlopCountAnalysis`;
- `thop`;
- PyTorch profiler;
- custom block-level FLOP accounting.

Latency protocol:
1. put model in `.eval()`;
2. disable gradients;
3. warm up GPU;
4. use CUDA events or `torch.cuda.synchronize()`;
5. run enough repetitions;
6. report mean/std or median/p95;
7. test batch size 1 at minimum;
8. optionally report realistic deployment batch sizes.

Heads should be included in actual latency even if their cost is tiny.

---

# 37. Success criteria

The strongest outcome is **not** merely a slightly better L8 result.

The desired result is something like:

```text
Original LatentWave:
L2: weak
L4: moderate
L6: near-full
L8: best

Proposed:
L2: much stronger
L4: near-full
L6: full-equivalent
L8: preserved
```

Example only:

| Method | L2 | L4 | L6 | L8 |
|---|---:|---:|---:|---:|
| LatentWave | 75 | 84 | 90 | 91 |
| Proposed | 84 | 90 | 91 | 91 |

The key claim would then be:

> **Near-full downstream performance is available at an earlier encoder depth after hierarchical pretraining.**

That directly converts into lower inference cost.

---

# 38. Possible paper contribution framing

Avoid:

> "We combine LocalMIM, JEPA, and early exits."

Stronger framing:

> **We co-design JEPA pretraining and downstream variable-depth inference for wireless foundation models. The proposed objective directly supervises multiple encoder depths using depth-dependent latent target scales, explicitly encouraging useful predictive representations to emerge earlier in the backbone. With the pretrained encoder frozen, lightweight task heads then exploit these intermediate representations for task-aware early exiting.**

Potential contributions:

1. **Hierarchical wireless JEPA objective** that explicitly supervises intermediate encoder depths.
2. **Fine-to-coarse latent-region prediction** in wireless JEPA, transferring the multi-scale idea from reconstruction space to EMA latent target space.
3. **Early-exit-ready pretraining**, evaluated using frozen-backbone task heads so representation gains are attributable to self-supervised pretraining.
4. **Accuracy/latency/FLOP characterization** across heterogeneous wireless downstream tasks and depths.
5. **Controlled ablation** separating standard LatentWave, deep same-scale JEPA, and multi-scale hierarchical JEPA.

Do not claim first-of-kind until a systematic literature review is completed.

---

# 39. Research hypotheses

### H1 — Multi-depth supervision
Intermediate JEPA supervision will improve linear/frozen-head performance at early encoder depths compared with standard LatentWave.

### H2 — Fine-to-coarse supervision
Depth-dependent coarse latent targets will further improve the quality/abstraction of intermediate representations compared with same-scale multi-depth JEPA.

### H3 — Early-exit efficiency
The proposed pretraining will reduce the earliest depth required to achieve performance within a fixed tolerance of full-depth inference.

### H4 — Task dependence
Different wireless downstream tasks will continue to prefer different representation depths, but the proposed model will shift some tasks toward cheaper exits.

### H5 — Full-depth preservation
The original LatentWave anchor loss will prevent significant degradation of the final-depth representation.

---

# 40. Experimental decision tree

After P0/P1/P2:

## Case A: P1 improves early exits, P2 improves further
Best-case result.
Proceed with full early-exit paper.

## Case B: P1 improves, P2 does not
Conclusion:
- deep self-supervision helps;
- multi-scale pooling does not.
Then reconsider novelty relative to V-JEPA 2.1 and search for a more wireless-specific mechanism.

## Case C: P2 improves final performance but not early layers
The method is not yet an early-exit contribution.
Investigate target scales/loss weights.

## Case D: early layers improve but L8 degrades
Reduce `lambda_H`, use warm-up, or remove coarse L8 auxiliary.

## Case E: nothing improves
Check:
- correct target matching;
- mask mapping;
- intermediate LayerNorm;
- predictor capacity;
- gradient conflicts;
- auxiliary loss scaling.

---

# 41. Logging checklist

Every pretraining run should log:

```text
train/L_total
train/L_LW
train/L_H
train/L2
train/L4
train/L6
train/L8
train/ema_momentum
train/lr
train/weight_decay
train/effective_rank
train/mean_std
train/mask_percentage
train/num_channels
```

Optional:
```text
grad_norm/block1
grad_norm/block2
...
grad_cosine/L_LW_vs_L2
...
```

Every downstream exit run:

```text
task
pretraining_method
exit_layer
seed
train_loss
val_metric
test_metric
FLOPs
latency
parameter_count
checkpoint_id
```

---

# 42. Reproducibility metadata

Save with every checkpoint:

- git commit hash;
- complete YAML config;
- dataset version/path hash;
- random seed;
- PyTorch/CUDA versions;
- GPU model;
- exact target-scale schedule;
- `lambda_H`;
- warm-up;
- mask type;
- mask parameters;
- selected layers;
- predictor architecture;
- EMA schedule.

---

# 43. Suggested folder structure for the modified project

```text
latent-wave/
├── configs/
│   ├── jepa.yml
│   ├── jepa_hier_same_scale.yml
│   └── jepa_hier_multiscale.yml
├── src/
│   ├── models/
│   │   ├── encoder.py
│   │   ├── predictor.py
│   │   ├── jepa.py
│   │   ├── hierarchical_targets.py      # new
│   │   ├── latentwfm.py
│   │   └── early_exit_wfm.py            # new
│   ├── evaluation/
│   │   └── early_exit.py                # new
│   ├── pretrain.py
│   └── finetune.py
├── scripts/
│   ├── pretrain_baseline.sh
│   ├── pretrain_hier_same_scale.sh
│   ├── pretrain_hier_multiscale.sh
│   └── eval_early_exits.sh
└── results/
    ├── pretraining/
    ├── downstream/
    └── latency/
```

---

# 44. Pseudocode: proposed pretraining step

```python
def training_step(x, masks_enc, masks_pred):
    # ---------------------------------------------------------
    # 1. stochastic channel sampling -- preserve LatentWave
    # ---------------------------------------------------------
    x = sample_channels_as_in_latentwave(x)

    # ---------------------------------------------------------
    # 2. context encoder: masked input, capture layers 2/4/6/8
    # ---------------------------------------------------------
    h_final, h = context_encoder(
        x,
        masks=masks_enc,
        return_layers=(2, 4, 6, 8),
    )

    # ---------------------------------------------------------
    # 3. target encoder: full input, no gradients
    # ---------------------------------------------------------
    with torch.no_grad():
        t_final, t = target_encoder(
            x,
            masks=None,
            return_layers=(2, 4, 6, 8),
        )

    # ---------------------------------------------------------
    # 4. original LatentWave final-layer prediction
    # ---------------------------------------------------------
    pred_lw = original_predictor(
        h_final,
        C=x.size(1),
        masks_enc=masks_enc,
        masks_pred=masks_pred,
    )

    target_lw = build_original_latentwave_targets(
        t_final,
        masks_pred,
    )

    L_LW = mse(pred_lw, target_lw.detach())

    # ---------------------------------------------------------
    # 5. hierarchical auxiliary predictions
    # ---------------------------------------------------------
    grids = {2: 14, 4: 7, 6: 4, 8: 2}
    aux_losses = []

    for l in (2, 4, 6, 8):
        h_l = layer_norm(h[l])
        t_l = layer_norm(t[l])

        target_regions, queries, valid = build_scaled_targets(
            target_features=t_l,
            masks_pred=masks_pred,
            output_grid=grids[l],
            require_fully_masked=True,
        )

        pred_regions = predictors[str(l)](
            context_features=h_l,
            custom_queries=queries,
            masks_enc=masks_enc,
        )

        L_l = masked_mse(
            pred_regions,
            target_regions.detach(),
            valid,
        )

        aux_losses.append(L_l)

    L_H = torch.stack(aux_losses).mean()

    # optional warm-up schedule for lambda_H
    lambda_now = hierarchical_weight(current_epoch)

    L_total = L_LW + lambda_now * L_H

    return L_total
```

---

# 45. Pseudocode: frozen early-exit head training

```python
# load pretrained context encoder only
encoder = load_context_encoder(checkpoint)

# freeze EVERYTHING in backbone
for p in encoder.parameters():
    p.requires_grad = False

encoder.eval()

exit_layer = 4
head = ExitHead(
    in_dim=256,
    out_dim=num_outputs,
)

for x, y in train_loader:
    with torch.no_grad():
        h = encoder.forward_to_layer(
            x,
            exit_layer=exit_layer,
        )

    pooled = h.mean(dim=1)
    pred = head(pooled)

    loss = task_loss(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Repeat independently for exits:
- 2
- 4
- 6
- 8

and for each downstream task.

---

# 46. Pseudocode: fixed earliest-within-tolerance exit

```python
metrics = {
    2: val_metric_exit2,
    4: val_metric_exit4,
    6: val_metric_exit6,
    8: val_metric_exit8,
}

full = metrics[8]
epsilon = 0.01  # example for normalized metric

for l in [2, 4, 6, 8]:
    if metrics[l] >= full - epsilon:
        selected_exit = l
        break
```

For lower-is-better metrics, reverse the inequality appropriately.

Select on validation/few-shot data only.

---

# 47. Pseudocode: dynamic entropy exit

```python
for l in [2, 4, 6, 8]:
    h = encoder.forward_next_blocks(..., stop_at=l)
    logits = heads[l](mean_pool(h))
    probs = softmax(logits, dim=-1)

    entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)

    if l < 8 and entropy <= tau_bar * math.log(num_classes):
        return probs.argmax(dim=-1)

return probs.argmax(dim=-1)
```

This is secondary to the fixed exit.

---

# 48. Questions to answer before final paper submission

- Does intermediate JEPA supervision significantly improve frozen-head performance?
- Does multi-scale target pooling add value beyond deep supervision?
- Which target-scale schedule is best?
- Does the original LatentWave final loss need to remain?
- Is coarse supervision useful for all masking geometries?
- Do classification and channel-related tasks respond differently?
- Does task-preferred depth shift toward earlier layers?
- How much pretraining overhead is introduced?
- What is the measured inference saving?
- Does dynamic routing offer anything beyond fixed exits?
- Are gains statistically significant across seeds?
- Are gains preserved under OOD datasets?
- Does the proposed model preserve L8 performance?

---

# 49. Claims that are safe only after experiments

Do **not** write these as facts before results exist:

- "coarse targets learn richer features";
- "hierarchical JEPA improves generalization";
- "the proposed method reduces inference latency";
- "multi-scale prediction is better than same-scale prediction";
- "early layers become more semantic";
- "the method is the first hierarchical wireless JEPA";
- "the method is state-of-the-art."

Use hypothesis language until demonstrated.

---

# 50. Short implementation summary

If only one section is read during coding, use this:

1. Clone/reproduce official LatentWave.
2. Modify the 8-layer context and target encoders so they optionally return layers `2,4,6,8`.
3. Preserve the original final-layer LatentWave predictor and loss exactly.
4. Add four depth-specific auxiliary JEPA predictors.
5. Same-scale baseline: all four predictors target fine `14×14` latent regions.
6. Proposed model: target grids `14×14`, `7×7`, `4×4`, `2×2`.
7. Pool **target encoder latent features only** to construct larger latent target regions.
8. Predictors directly output one vector per coarse masked region; do not predict fine patches and pool afterward.
9. Use only coarse regions fully covered by the original mask in the first implementation.
10. Auxiliary loss at each layer is MSE between predicted region latent and stop-gradient EMA target latent.
11. Final objective:

\[
L_{\text{total}} =
L_{\text{LW}}
+
\frac{\lambda_H}{4}
(L_2+L_4+L_6+L_8)
\]

12. Start with `lambda_H = 1.0`; optionally warm it up.
13. Target encoder remains gradient-free and EMA-updated.
14. After pretraining, discard target encoder and all predictors.
15. Freeze the retained context encoder for the primary downstream evaluation.
16. Attach identical mean-pool + LayerNorm + Linear heads at exits `2,4,6,8`.
17. Train only the heads.
18. Primary baseline: original LatentWave with the same frozen exit heads.
19. Critical ablation: same-scale multi-depth JEPA vs multi-scale multi-depth JEPA.
20. Primary deployment: fixed task-specific exit; dynamic entropy routing is secondary.
21. Report accuracy/error, FLOPs, measured latency, selected exit, and savings.
22. The strongest success case is achieving near-L8 performance at L2/L4/L6 while preserving L8 quality.

---

# 51. Reference links

### LatentWave
- Paper: https://arxiv.org/abs/2606.06373
- PDF: https://arxiv.org/pdf/2606.06373
- Repo: https://github.com/AhmedEssam19/latent-wave
- Config: https://github.com/AhmedEssam19/latent-wave/blob/master/configs/jepa.yml
- JEPA model: https://github.com/AhmedEssam19/latent-wave/blob/master/src/models/jepa.py
- Encoder: https://github.com/AhmedEssam19/latent-wave/blob/master/src/models/encoder.py
- Predictor: https://github.com/AhmedEssam19/latent-wave/blob/master/src/models/predictor.py
- Downstream model: https://github.com/AhmedEssam19/latent-wave/blob/master/src/models/latentwfm.py
- Fine-tuning script: https://github.com/AhmedEssam19/latent-wave/blob/master/src/finetune.py

### Early Exit
- Paper: https://arxiv.org/abs/2606.29640
- PDF: https://arxiv.org/pdf/2606.29640

### LocalMIM
- Paper: https://arxiv.org/abs/2303.05251
- PDF: https://arxiv.org/pdf/2303.05251
- Repo: https://github.com/Haoqing-Wang/LocalMIM

### V-JEPA 2.1
- Paper: https://arxiv.org/abs/2603.14482
- PDF: https://arxiv.org/pdf/2603.14482

---

# 52. Final one-paragraph project definition

**Proposed method:** Starting from LatentWave, retain its shared per-antenna tokenizer, 8-layer ViT context/EMA-target encoders, original masking, and final latent-prediction loss. Add auxiliary JEPA predictors at encoder layers 2, 4, 6, and 8. Each auxiliary branch independently predicts the EMA target encoder's latent representation of masked regions at the same depth. A same-scale baseline uses the original 14×14 target granularity at every depth; the proposed multi-scale model uses progressively coarser target regions (initially 14×14, 7×7, 4×4, and 2×2) constructed by target-side latent pooling, with predictors directly generating one latent vector per coarse masked region. Train with the original LatentWave loss plus the average auxiliary loss, keep the target encoder stop-gradient/EMA-updated, then discard all predictors and the target encoder. For downstream tasks, freeze the learned context encoder, attach identical lightweight heads at layers 2/4/6/8, train only those heads, and compare original LatentWave versus the proposed pretrained model using fixed task-specific and secondary dynamic early-exit policies. The central evaluation is whether near-full downstream performance becomes available at earlier depths, thereby improving the performance–FLOP and performance–latency Pareto frontier.
