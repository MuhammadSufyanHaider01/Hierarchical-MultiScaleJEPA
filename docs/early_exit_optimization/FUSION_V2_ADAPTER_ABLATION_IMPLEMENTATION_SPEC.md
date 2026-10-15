# Residual Fusion V2 Adapter Ablation — Single-Seed Implementation Specification

## Repository and target branch

Repository:

```text
MuhammadSufyanHaider01/Hierarchical-MultiScaleJEPA
```

Target branch:

```text
main
```

This specification is written against the current `main` implementation, in particular:

```text
src/models/exit_heads.py
src/models/__init__.py
scripts/phase5_residual_fusion_exits.py
scripts/phase5_exits.py
src/phase5.py
```

The existing Residual Projected Dynamic Fusion V2 implementation and its committed P0/P2 single-seed results must remain unchanged and serve as the reference baseline.

---

# 1. Goal

Run a controlled adapter-design ablation around the existing Fusion V2 architecture.

The existing Fusion V2 source path is:

\[
u_j
\rightarrow
LN
\rightarrow
Linear(256,64)
\rightarrow
GELU
\rightarrow
Linear(64,256)
\rightarrow
c_{k,j},
\]

followed by the existing sample-dependent scalar gate:

\[
g_{k,j}(x)
=
\sigma(e_{k,j}),
\]

and residual fusion:

\[
h_k
=
u_k
+
\sum_{j<k}
g_{k,j}(x)c_{k,j}(x).
\]

The current `bottleneck + GELU` architecture has already been run. **Do not rerun it in this pilot.**

The purpose of this study is to determine whether the same downstream performance can be achieved with a simpler adapter, and whether the adapter nonlinearity itself is necessary.

---

# 2. Critical scope constraint: no pretraining

**Do not run pretraining.**

Reuse the exact already-existing M-block pretrained P0 and P2 ViT checkpoints used by the committed Fusion V2 experiment.

The experiment is only:

```text
existing frozen M-block P0/P2 encoder
        ↓
new downstream ablation head training
```

and not:

```text
new pretraining
        ↓
downstream evaluation
```

The committed Fusion V2 report records the checkpoint hashes used for the reference experiment:

```text
P0 checkpoint SHA256:
8faa7245b6692e2d74d753fde2268fb712264c02c1da19632c045d022d192f1f

P2 checkpoint SHA256:
7e1e27d96183b8f9d1c0d6b7493b93945633ccfb1514555135b4d04f22bf18ac
```

The ablation runner must verify that the supplied checkpoint hash matches the appropriate reference hash before training.

Do not silently substitute another checkpoint.

---

# 3. Pilot experiment scope

Run only:

```text
backbone: ViT
masking condition: multi_block_n6
methods: p0, p2
pretrain seed: 0
downstream head seed: 0
split seed: 2025
tasks: all 6 PRIMARY_TASKS
exit layers: [2, 4, 6, 8]
encoder fine-tuning: disabled
knowledge distillation: disabled
```

Primary tasks:

```text
rfs
jamming
human-activity
interference
nrpos-outdoor
deepmimo-beam
```

Only one downstream head seed is required in this initial pilot:

```text
head_seed = 0
```

Do not launch seeds 1–6 yet.

---

# 4. Existing reference architecture — do not rerun

The already-completed reference is:

## `bottleneck_gelu`

\[
\bar u_j = LN_{k,j}(u_j)
\]

\[
v_{k,j}
=
GELU
\left(
W^\downarrow_{k,j}\bar u_j+b^\downarrow_{k,j}
\right)
\]

\[
c_{k,j}
=
W^\uparrow_{k,j}v_{k,j}+b^\uparrow_{k,j}.
\]

Dimensions:

```text
256 -> 64 -> GELU -> 256
```

Its existing implementation is:

```text
ResidualProjectedFusionExitHead
FrozenEncoderResidualFusionExit
```

Its existing training script is:

```text
scripts/phase5_residual_fusion_exits.py
```

Its existing result packets are:

```text
runs/ee_optimization/multi_block_n6/residual_dynamic_fusion/
    head_seed_000/p0/seed_000/downstream/exit_metrics.json

runs/ee_optimization/multi_block_n6/residual_dynamic_fusion/
    head_seed_000/p2/seed_000/downstream/exit_metrics.json
```

Use these committed results directly as the baseline.

Do not modify or overwrite them.

---

# 5. New ablations to implement

Implement the following **five** new adapter variants.

The dynamic gate itself is not being ablated in this experiment.

The target anchor, source sets, mean pooling, gate structure, frozen encoder, final output LayerNorm/Linear head, training protocol, validation selection, and test protocol should remain as close as mechanically possible to the existing Fusion V2 implementation.

---

## Ablation A1 — Bottleneck Linear

Code identifier:

```text
bottleneck_linear
```

Architecture:

\[
\bar u_j=LN_{k,j}(u_j)
\]

\[
v_{k,j}
=
W^\downarrow_{k,j}\bar u_j+b^\downarrow_{k,j}
\]

\[
c_{k,j}
=
W^\uparrow_{k,j}v_{k,j}+b^\uparrow_{k,j}.
\]

Dimensions:

```text
256 -> 64 -> 256
```

There is **no activation** between the two linear layers.

This ablation answers:

> Is the GELU nonlinearity actually necessary, or is a low-rank linear cross-layer transformation sufficient?

Mathematically, ignoring biases:

\[
c_{k,j}
=
W^\uparrow_{k,j}
W^\downarrow_{k,j}
\bar u_j,
\]

which is a rank-at-most-64 linear transformation.

### Initialization

Keep the current V2 policy:

```python
down_projection = nn.Linear(256, 64)
up_projection = nn.Linear(64, 256)

nn.init.zeros_(up_projection.weight)
nn.init.zeros_(up_projection.bias)
```

Therefore initially:

\[
c_{k,j}=0
\]

and:

\[
h_k=u_k.
\]

### Gate source feature

Use:

\[
v_{k,j}\in\mathbb{R}^{64}
\]

as the source feature passed to the existing source gate projection.

This keeps the gate topology identical to the existing bottleneck-GELU implementation except for removing GELU.

---

## Ablation A2 — Bottleneck Sigmoid

Code identifier:

```text
bottleneck_sigmoid
```

Architecture:

\[
\bar u_j=LN_{k,j}(u_j)
\]

\[
v_{k,j}
=
\sigma
\left(
W^\downarrow_{k,j}\bar u_j+b^\downarrow_{k,j}
\right)
\]

\[
c_{k,j}
=
W^\uparrow_{k,j}v_{k,j}+b^\uparrow_{k,j}.
\]

Dimensions:

```text
256 -> 64 -> Sigmoid -> 256
```

This is the direct activation-controlled comparison against the existing:

```text
256 -> 64 -> GELU -> 256
```

Because the final `64 -> 256` up-projection remains unconstrained, the final correction vector can still contain positive and negative values after training even though the bottleneck feature lies in `(0,1)`.

### Initialization

Exactly as current V2:

```python
nn.init.zeros_(up_projection.weight)
nn.init.zeros_(up_projection.bias)
```

so:

\[
c_{k,j}=0
\]

at initialization.

### Gate source feature

Use the post-sigmoid bottleneck vector:

\[
v_{k,j}\in(0,1)^{64}
\]

as the input to the source gate projection.

Thus the only substantive difference from the existing V2 reference is:

```text
GELU -> Sigmoid
```

inside the adapter bottleneck.

---

## Ablation A3 — Single Linear

Code identifier:

```text
single_linear
```

Architecture:

\[
\bar u_j=LN_{k,j}(u_j)
\]

\[
c_{k,j}
=
W^{single}_{k,j}\bar u_j+b^{single}_{k,j}.
\]

Dimensions:

```text
256 -> 256
```

No adapter activation.

This removes the two-layer bottleneck entirely.

This ablation answers:

> Is one direct learned source-to-target correction sufficient?

### Initialization

To preserve the anchor-only starting point:

```python
nn.init.zeros_(single_projection.weight)
nn.init.zeros_(single_projection.bias)
```

so:

\[
c_{k,j}=0
\]

and:

\[
h_k=u_k
\]

at initialization.

### Important parameter-count note

A full `256 -> 256` linear layer has approximately twice as many adapter weights as the current low-rank `256 -> 64 -> 256` adapter:

\[
256\times256=65,536
\]

versus approximately:

\[
256\times64+64\times256=32,768.
\]

Therefore this variant is **shallower architecturally**, but not necessarily lighter in parameter count.

The report must include exact trainable parameter counts before making any efficiency claim.

### Gate source feature

For single-layer variants, use the processed 256-D source correction feature as the source input to the gate:

\[
s^{src}_{k,j}=c_{k,j}\in\mathbb{R}^{256}.
\]

The source gate projection therefore becomes:

```python
nn.Linear(256, 64)
```

instead of the bottleneck variant's:

```python
nn.Linear(64, 64)
```

The rest of the gate equation must remain unchanged.

---

## Ablation A4 — Single GELU

Code identifier:

```text
single_gelu
```

Architecture:

\[
\bar u_j=LN_{k,j}(u_j)
\]

\[
z_{k,j}
=
W^{single}_{k,j}\bar u_j+b^{single}_{k,j}
\]

\[
c_{k,j}
=
GELU(z_{k,j}).
\]

Dimensions:

```text
256 -> 256 -> GELU
```

This asks:

> Can one nonlinear full-dimensional transformation replace the two-layer bottleneck adapter?

### Initialization

Zero initialize:

```python
nn.init.zeros_(single_projection.weight)
nn.init.zeros_(single_projection.bias)
```

Since:

\[
GELU(0)=0,
\]

we retain:

\[
c_{k,j}=0
\]

and:

\[
h_k=u_k
\]

at initialization.

### Gate source feature

Use the post-GELU 256-D vector:

\[
c_{k,j}
\]

as the gate source feature.

The source gate projection is:

```python
nn.Linear(256, 64)
```

---

## Ablation A5 — Single Sigmoid, centered

Code identifier:

```text
single_sigmoid
```

Do **not** use raw:

\[
c=\sigma(z)
\]

as the primary implementation.

A raw sigmoid gives:

\[
\sigma(0)=0.5,
\]

which would destroy the exact zero-residual initialization and would also force every correction dimension to be positive.

Instead, implement the single-layer sigmoid ablation as a **centered sigmoid**:

\[
z_{k,j}
=
W^{single}_{k,j}\bar u_j+b^{single}_{k,j}
\]

\[
c_{k,j}
=
2\sigma(z_{k,j})-1.
\]

Dimensions:

```text
256 -> 256 -> centered Sigmoid
```

The output lies in:

\[
(-1,1)
\]

and is signed.

At zero initialization:

\[
z_{k,j}=0
\]

and:

\[
2\sigma(0)-1=0.
\]

Therefore:

\[
c_{k,j}=0
\]

and:

\[
h_k=u_k.
\]

This keeps the initialization protocol controlled across all variants.

The report must label this variant explicitly as:

```text
Single Sigmoid (centered)
```

rather than implying that it is an uncentered raw sigmoid.

### Gate source feature

Use the centered sigmoid 256-D correction vector as the gate source feature.

Source gate projection:

```python
nn.Linear(256, 64)
```

---

# 6. Gate mechanism — keep unchanged conceptually

Do not ablate the gate in this experiment.

For every target exit `k`, the anchor query remains:

\[
q_k
=
W_k^a
LN_k^a(u_k)
\]

with:

```text
256 -> 64
```

For source `j`, define an adapter-dependent gate feature:

\[
f_{k,j}.
\]

For bottleneck variants:

\[
f_{k,j}=v_{k,j}\in\mathbb{R}^{64}.
\]

For single-layer variants:

\[
f_{k,j}=c_{k,j}\in\mathbb{R}^{256}.
\]

Then:

\[
s_{k,j}
=
\tanh
\left(
q_k+
W^s_{k,j}f_{k,j}
\right),
\]

followed by:

\[
e_{k,j}
=
(w^g_{k,j})^Ts_{k,j}+b^g_{k,j},
\]

and:

\[
g_{k,j}(x)
=
\sigma(e_{k,j}).
\]

The gate output remains:

```text
[B, 1]
```

and is broadcast over the 256-D correction vector.

There must be:

```text
no softmax over sources
no sum-to-one constraint
no global alpha weights
```

Each source gate remains independent and sample-dependent.

---

# 7. Residual fusion equation — unchanged

For every variant:

\[
h_k
=
u_k+
\sum_{j<k}g_{k,j}(x)c_{k,j}(x).
\]

Source sets remain:

```text
L2: {}
L4: {2}
L6: {2, 4}
L8: {2, 4, 6}
```

Explicitly:

\[
h_2=u_2
\]

\[
h_4=u_4+g_{4,2}c_{4,2}
\]

\[
h_6=u_6+g_{6,2}c_{6,2}+g_{6,4}c_{6,4}
\]

\[
h_8=u_8+g_{8,2}c_{8,2}+g_{8,4}c_{8,4}+g_{8,6}c_{8,6}.
\]

The current target exit always remains the unit-coefficient anchor.

---

# 8. Final task head — unchanged

Keep the current Fusion V2 output head:

\[
\tilde h_k
=
LN^{out}_k(h_k)
\]

and:

\[
\hat y_k
=
W^{out}_k\tilde h_k+b^{out}_k.
\]

Implementation:

```python
self.output_norm = nn.LayerNorm(256)
self.output_projection = nn.Linear(256, num_outputs)
```

Do not add another downstream MLP.

---

# 9. Encoder behavior — unchanged

Use the current frozen prefix-forward implementation:

```python
encoder.forward_to_layers(x, requested_layers)
```

Requested layers:

```text
L2 -> [2]
L4 -> [2, 4]
L6 -> [2, 4, 6]
L8 -> [2, 4, 6, 8]
```

Do not separately forward the encoder once per source layer.

The encoder must remain:

```python
requires_grad = False
encoder.eval()
```

and feature extraction must remain under:

```python
torch.no_grad()
```

No changes to the ViT encoder architecture are needed.

---

# 10. Mean pooling — unchanged

For every encoder output:

\[
Z_j\in\mathbb{R}^{B\times N\times256},
\]

use:

\[
u_j
=
\frac{1}{N}
\sum_{n=1}^N Z_{j,n}.
\]

Do not add attention pooling or token-level fusion.

---

# 11. Recommended implementation design

Do not create five mostly duplicated copies of the Fusion V2 head.

Keep the existing classes unchanged:

```text
ResidualProjectedFusionExitHead
FrozenEncoderResidualFusionExit
```

Add a generic configurable ablation head, for example:

```python
class ResidualFusionAblationExitHead(nn.Module):
    ...
```

and:

```python
class FrozenEncoderResidualFusionAblationExit(nn.Module):
    ...
```

Recommended constructor:

```python
ResidualFusionAblationExitHead(
    embed_dim: int,
    num_outputs: int,
    exit_layer: int,
    source_layers,
    adapter_variant: str,
    bottleneck_dim: int = 64,
    gate_dim: int = 64,
)
```

Allowed values:

```python
ADAPTER_VARIANTS = (
    "bottleneck_linear",
    "bottleneck_sigmoid",
    "single_linear",
    "single_gelu",
    "single_sigmoid",
)
```

Validate the variant strictly.

---

# 12. Suggested module structure

For bottleneck variants:

```python
self.source_norms
self.down_projections
self.up_projections
self.anchor_gate_norm
self.anchor_gate_projection
self.source_gate_projections
self.gate_scores
self.output_norm
self.output_projection
```

For single-layer variants:

```python
self.source_norms
self.single_projections
self.anchor_gate_norm
self.anchor_gate_projection
self.source_gate_projections
self.gate_scores
self.output_norm
self.output_projection
```

Do not instantiate unused adapter modules.

At L2, all source-specific adapter and gate ModuleDicts should be empty.

---

# 13. Suggested internal adapter API

Use one internal helper to keep the main fusion loop common across all variants.

For example:

```python
def _source_correction_and_gate_feature(
    self,
    source_layer: int,
    source_vector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

It should return:

```text
correction: [B, 256]
gate_feature: [B, gate_input_dim]
```

Behavior:

```text
bottleneck_linear:
    hidden = down(LN(source))
    correction = up(hidden)
    gate_feature = hidden

bottleneck_sigmoid:
    hidden = sigmoid(down(LN(source)))
    correction = up(hidden)
    gate_feature = hidden

single_linear:
    correction = single(LN(source))
    gate_feature = correction

single_gelu:
    correction = GELU(single(LN(source)))
    gate_feature = correction

single_sigmoid:
    z = single(LN(source))
    correction = 2 * sigmoid(z) - 1
    gate_feature = correction
```

This avoids duplicating the residual and gate logic.

---

# 14. Preserve zero-residual initialization

Every new variant must satisfy numerically before training:

\[
c_{k,j}=0
\]

for every source.

Therefore:

\[
h_k=u_k.
\]

Required initialization:

```text
bottleneck_linear:
    zero-init up projection

bottleneck_sigmoid:
    zero-init up projection

single_linear:
    zero-init single projection

single_gelu:
    zero-init single projection

single_sigmoid:
    zero-init single projection
    use centered sigmoid
```

Add an explicit unit/smoke assertion for this.

---

# 15. Important first-step gradient behavior

Document and test the initialization behavior.

For bottleneck variants, because the up-projection is zero:

```text
first backward:
    up-projection can receive gradient
    down-projection/gate path may initially receive zero or negligible loss gradient through the correction branch
```

This matches the current Fusion V2 zero-initialized residual-adapter behavior.

For the single-layer variants, the zero-initialized projection is itself the final correction mapping, so it can receive gradient on the first step.

Do not treat this difference as a bug.

It is an optimization consequence of the different parameterizations and should be noted in the experiment metadata.

---

# 16. Export classes

Update:

```text
src/models/__init__.py
```

to export only the new generic ablation classes, e.g.:

```python
ResidualFusionAblationExitHead
FrozenEncoderResidualFusionAblationExit
```

Do not rename or remove existing exported classes.

---

# 17. New training runner

Add:

```text
scripts/phase5_residual_fusion_ablation_exits.py
```

Use:

```text
scripts/phase5_residual_fusion_exits.py
```

as the primary template.

Do not modify the existing baseline runner unless a very small reusable helper extraction is clearly safe.

Add required CLI argument:

```python
parser.add_argument(
    "--adapter-variant",
    choices=(
        "bottleneck_linear",
        "bottleneck_sigmoid",
        "single_linear",
        "single_gelu",
        "single_sigmoid",
    ),
    required=True,
)
```

Keep existing relevant arguments:

```text
--checkpoint
--method {p0,p2}
--pretrain-seed
--head-seed
--split-seed
--task
--exit-layer
--epochs
--batch-size
--num-workers
--learning-rate
--weight-decay
--head-early-stopping-patience
--head-early-stopping-min-delta
--resume
--selection-only
--test-only
--device
```

---

# 18. Locked split

Reuse exactly:

```text
runs/masking_head_seeds/multi_block_n6/head_seed_000/
splits_seed_2025_primary_v3.json
```

Verify:

```text
split seed = 2025
```

and fail if the manifest is absent or incompatible.

Do not create a new split.

---

# 19. Training protocol — match current Fusion V2

Preserve:

```text
epochs = 200
batch size = 32
num workers = 4
learning rate = 1e-3
AdamW
weight decay = 0.05
early-stopping patience = 20
early-stopping min_delta = 0
split seed = 2025
head seed = 0
```

Preserve the current task criteria:

```text
classification:
    CrossEntropyLoss
    existing DeepMIMO class weighting

nrpos-outdoor:
    MSE training loss
```

Preserve task evaluation:

```text
classification:
    macro_accuracy

nrpos-outdoor:
    mean_distance_error
```

No auxiliary losses.

Do not add:

```text
KD
feature matching
gate entropy loss
gate sparsity loss
residual norm loss
```

---

# 20. Deterministic downstream seeding

Preserve the current Phase-5 formulas.

Training-loader seed:

\[
s_{\text{loader}}
=
s_h+t
\]

where:

```text
s_h = head_seed = 0
t = task index
```

Per task/exit head seed:

\[
s_{t,k}
=
s_h+100t+k.
\]

All adapter variants should use the same task/exit seed formula.

---

# 21. L2 sanity control

L2 has no earlier source layers:

\[
\mathcal S_2=\varnothing.
\]

Therefore all five new variants must reduce to:

\[
h_2=u_2
\]

followed by the same output LayerNorm and Linear head.

Because all variants use the same head seed and no variant-specific source modules should be instantiated for L2, the L2 results should be numerically identical, or extremely close if unavoidable framework nondeterminism exists.

Treat a meaningful L2 discrepancy across variants as a bug or reproducibility warning.

This is an important built-in sanity check.

---

# 22. Optimizer parameter groups

Match the current Fusion V2 optimizer philosophy.

## Decay group: `weight_decay = 0.05`

For bottleneck variants:

```text
down_projection.weight
up_projection.weight
output_projection.weight
```

For single-layer variants:

```text
single_projection.weight
output_projection.weight
```

## No-decay group: `weight_decay = 0`

Keep no weight decay on:

```text
all LayerNorm parameters
all biases
anchor_gate_projection parameters
source_gate_projection parameters
gate_score parameters
```

The dynamic gate network must remain entirely in the no-decay group, matching current Fusion V2.

Add a parameter-group audit to ensure every trainable parameter belongs to exactly one optimizer group.

---

# 23. Frozen-backbone check

After each backward pass, preserve the current safety assertion:

```python
if any(p.grad is not None for p in encoder.parameters()):
    raise RuntimeError(...)
```

The encoder must never receive gradients.

---

# 24. Validation early stopping

For every task and exit:

```text
max epochs = 200
patience = 20
min_delta = 0
```

Use the task's current validation metric.

Save a head only when validation improves.

At completion, reload the best-validation checkpoint before diagnostics or test evaluation.

---

# 25. Validation-only exit selection

Use the existing:

```python
select_best_exit(...)
```

Headline selection:

```text
selected_exit_best_validation
```

Classification:

\[
k^*
=
\arg\max_k M_k^{val}.
\]

NR positioning:

\[
k^*
=
\arg\min_k M_k^{val}.
\]

Exact ties should preserve the existing shallower-exit rule.

The old tolerance selector may remain in the packet for compatibility but must not drive the main comparison.

---

# 26. Test discipline

Required sequence:

```text
train
    ↓
validation early stopping
    ↓
validation exit selection
    ↓
freeze selection
    ↓
held-out test evaluation
```

Never use test results to choose:

```text
adapter variant
epoch
exit
activation
bottleneck size
learning rate
```

For this initial pilot, inspect the five variants after all runs are completed rather than adaptively changing the architecture after each individual task result.

---

# 27. Diagnostics — preserve current Fusion V2 diagnostics

For the best validation checkpoint, run one validation diagnostic pass.

For each source, save:

```text
gate_mean
gate_sd
gate_median
gate_p10
gate_p90
residual_norm_mean
relative_residual_norm_mean
```

Use:

\[
R_{k,j}
=
\|g_{k,j}c_{k,j}\|_2
\]

and:

\[
\tilde R_{k,j}
=
\frac{
\|g_{k,j}c_{k,j}\|_2
}{
\|u_k\|_2+10^{-8}
}.
\]

Do not replace gate diagnostics with global alpha weights.

---

# 28. Add exact trainable parameter counts

For every variant and target exit, record:

```text
total_trainable_head_parameters
adapter_trainable_parameters
gate_trainable_parameters
output_head_trainable_parameters
```

This is required because the single `256 -> 256` variants may be shallower but can have more parameters than the low-rank bottleneck architecture.

Do not claim that a single-layer variant is "lighter" unless parameter count and measured inference cost support that claim.

---

# 29. Optional inference-cost measurement

If the repository's existing inference-cost utilities can be reused without changing the study substantially, also record for each variant/exit:

```text
head parameter count
head MACs/FLOPs estimate
end-to-end prefix + head inference latency
```

Keep this secondary to accuracy/error for the initial pilot.

At minimum, exact trainable parameter count is required.

---

# 30. Checkpoint schema

Recommended checkpoint packet:

```python
{
    "head_algorithm":
        "residual_fusion_adapter_ablation",

    "adapter_variant":
        args.adapter_variant,

    "exit_layer":
        layer,

    "source_layers":
        list(model.source_layers),

    "embed_dim":
        256,

    "bottleneck_dim":
        64 if variant.startswith("bottleneck_") else None,

    "gate_dim":
        64,

    "initialization":
        "zero_residual",

    "head":
        model.head.state_dict(),

    "epoch":
        best_epoch,

    "validation":
        validation_metrics,
}
```

Also save:

```text
checkpoint SHA256
pretrained encoder checkpoint SHA256
split-manifest SHA256
```

---

# 31. Result packet schema

Each `exit_metrics.json` should include:

```json
{
  "head_algorithm": "residual_fusion_adapter_ablation",
  "adapter_variant": "bottleneck_linear",
  "method": "p0",
  "pretrain_seed": 0,
  "head_seed": 0,
  "split_seed": 2025,
  "checkpoint_sha256": "...",
  "split_manifest_sha256": "...",
  "protocol": {},
  "tasks": {}
}
```

Each exit should contain:

```json
{
  "best_epoch": 0,
  "validation": {},
  "test": {},
  "head_path": "...",
  "source_layers": [],
  "fusion_diagnostics": {},
  "trainable_parameters": {
    "total": 0,
    "adapter": 0,
    "gate": 0,
    "output": 0
  }
}
```

Numbers above are placeholders.

---

# 32. New run directory

Use a new isolated hierarchy:

```text
runs/
└── ee_optimization/
    └── multi_block_n6/
        └── residual_fusion_adapter_ablation/
            └── head_seed_000/
                ├── bottleneck_linear/
                │   ├── p0/
                │   │   └── seed_000/downstream/
                │   └── p2/
                │       └── seed_000/downstream/
                │
                ├── bottleneck_sigmoid/
                │   ├── p0/
                │   └── p2/
                │
                ├── single_linear/
                │   ├── p0/
                │   └── p2/
                │
                ├── single_gelu/
                │   ├── p0/
                │   └── p2/
                │
                └── single_sigmoid/
                    ├── p0/
                    └── p2/
```

Do not overwrite:

```text
runs/ee_optimization/multi_block_n6/residual_dynamic_fusion/
```

or any previous Current/Fusion-V1 result.

---

# 33. Formal pilot job matrix

Five new variants:

```text
bottleneck_linear
bottleneck_sigmoid
single_linear
single_gelu
single_sigmoid
```

Two pretrained methods:

```text
p0
p2
```

Six tasks:

```text
6
```

Four exits:

```text
L2
L4
L6
L8
```

Total newly trained heads if every L2 is also run:

\[
5\times2\times6\times4=240.
\]

Although L2 is architecture-independent, running it for every variant is useful as a deterministic sanity control.

Do not run additional head seeds yet.

---

# 34. Smoke test before full jobs

Before launching the full matrix, test at least:

```text
method: p2
task: rfs
exit: L6
head seed: 0
```

for each of the five variants with limited train/validation batches.

Check:

```text
model construction
output shape
single prefix encoder forward
encoder frozen
source set correct
zero residual at initialization
gate range in (0,1)
no source softmax
checkpoint save/reload
diagnostic pass
parameter counting
optimizer grouping
```

Also run an L2 construction test for all variants and verify that no adapter/source-gate modules are instantiated.

---

# 35. Required implementation tests

The Codex implementation should include or execute checks for all of the following.

## 35.1 Zero-residual identity

Before training:

\[
h_k=u_k.
\]

Numerically compare fused and anchor features.

## 35.2 Correct source sets

```text
L2: ()
L4: (2,)
L6: (2,4)
L8: (2,4,6)
```

## 35.3 Gate range

Verify:

\[
0<g_{k,j}<1.
\]

## 35.4 Independent gates

There must be no cross-source normalization.

## 35.5 Single-sigmoid centering

For zero input/preactivation:

\[
2\sigma(0)-1=0.
\]

Verify the single-sigmoid correction is zero at initialization.

## 35.6 Frozen encoder

All encoder gradients remain `None`.

## 35.7 Prefix truncation

For L6, transformer blocks beyond 6 must not execute.

## 35.8 One encoder pass

All requested representations for one sample are returned through one `forward_to_layers(...)` call.

## 35.9 Optimizer coverage

Every trainable parameter appears in exactly one optimizer group.

## 35.10 L2 consistency

All five ablation variants should yield the same L2 architecture and deterministic result under the same head seed.

---

# 36. Report generation

Add a dedicated report script, preferably:

```text
scripts/residual_fusion_adapter_ablation_report.py
```

Report output:

```text
reports/
└── early_exit_optimization/
    └── multi_block_n6/
        └── residual_fusion_adapter_ablation_single_seed/
```

The report should import the existing committed Fusion V2 baseline results instead of rerunning them.

Reference baseline label:

```text
bottleneck_gelu
```

New labels:

```text
bottleneck_linear
bottleneck_sigmoid
single_linear
single_gelu
single_sigmoid
```

Also optionally include Current EE as contextual reference, but the main ablation comparison is against the existing Fusion V2 `bottleneck_gelu` baseline.

---

# 37. Required report files

Produce at least:

```text
README.md
manifest.json

fixed_exit_results.csv
validation_selected_results.csv
direction_aware_gains_vs_bottleneck_gelu.csv
parameter_counts.csv
gate_diagnostics.csv
```

Useful figures:

```text
p0_adapter_ablation_layerwise_performance.png/pdf
p2_adapter_ablation_layerwise_performance.png/pdf

per-task adapter-ablation plots
```

---

# 38. Fixed-exit comparison table

For every P0/P2 task, compare:

| Variant | L2 | L4 | L6 | L8 |
|---|---:|---:|---:|---:|
| Bottleneck GELU — existing V2 | ... | ... | ... | ... |
| Bottleneck Linear | ... | ... | ... | ... |
| Bottleneck Sigmoid | ... | ... | ... | ... |
| Single Linear | ... | ... | ... | ... |
| Single GELU | ... | ... | ... | ... |
| Single Sigmoid (centered) | ... | ... | ... | ... |

Use held-out test metrics for final fixed-exit reporting, but never for training or selection.

---

# 39. Validation-selected comparison

For every method/task/variant, report:

```text
selected_exit_best_validation
validation_value
selected_test_value
```

The selected test metric is evaluated only after the validation-selected exit is fixed.

---

# 40. Direction-aware gain relative to existing V2

For classification:

\[
\Delta
=
M^{variant}_k
-
M^{bottleneck\_gelu}_k.
\]

For NR positioning:

\[
\Delta
=
M^{bottleneck\_gelu}_k
-
M^{variant}_k.
\]

Positive means the new variant improves over the current Fusion V2 baseline.

Compute this for:

```text
fixed L2
fixed L4
fixed L6
fixed L8
validation-selected exit
```

---

# 41. Primary scientific comparisons

The report should explicitly answer these questions.

## Q1. Is nonlinearity required?

Compare:

```text
bottleneck_gelu
vs
bottleneck_linear
```

If performance is similar, the adapter can potentially be described as a low-rank linear feature-alignment layer.

---

## Q2. Does activation choice matter?

Compare:

```text
bottleneck_gelu
vs
bottleneck_sigmoid
```

This is the cleanest GELU-vs-Sigmoid comparison because topology and parameter count remain essentially fixed.

---

## Q3. Is the two-layer bottleneck required?

Compare:

```text
bottleneck_gelu
vs
single_linear
```

If single-linear matches performance, the proposed architecture can potentially be simplified to:

\[
h_k
=
u_k+
\sum_{j<k}
g_{k,j}(x)
W_{k,j}u_j.
\]

---

## Q4. Is one nonlinear full-dimensional layer sufficient?

Compare:

```text
single_linear
vs
single_gelu
vs
single_sigmoid
```

This determines whether a one-layer source transformation benefits from nonlinearity.

---

## Q5. Is the simpler architecture actually more efficient?

Use:

```text
trainable parameter count
optional measured latency/MACs
```

Do not infer efficiency only from number of layers.

---

# 42. Interpretation rules for the pilot

This is still only:

```text
head seed = 0
```

Therefore do not make final claims about variance or robustness.

The pilot is only for architecture triage.

A variant should be promoted to the full 7-head-seed study if it:

1. matches or improves current Fusion V2 on most meaningful fixed exits/tasks;
2. does not introduce systematic degradation;
3. offers a clear simplicity, parameter, or latency advantage.

If two variants are very close on one seed, do not use tiny test differences to declare a winner.

Prefer the simpler design and verify it over head seeds 0–6 next.

---

# 43. Do not redesign KD in this task

This ablation task is only about the Fusion V2 adapter.

Do not implement:

```text
knowledge distillation
cross-depth teacher selection
feature KD
logit KD
teacher-student projector
```

The KD stage should be considered only after the final fusion architecture is chosen.

---

# 44. Do not add additional architecture changes

For this pilot, do not alter:

```text
mean pooling
encoder taps
exit layers
gate activation
gate dimensionality
anchor residual rule
task head
training loss
early stopping
exit selector
split
pretrained checkpoints
```

This is an adapter topology/activation ablation only.

---

# 45. Expected code change set

Expected modifications/additions:

```text
src/models/exit_heads.py
    + ResidualFusionAblationExitHead
    + FrozenEncoderResidualFusionAblationExit

src/models/__init__.py
    + export the two new classes

scripts/phase5_residual_fusion_ablation_exits.py
    + configurable five-variant runner

scripts/residual_fusion_adapter_ablation_report.py
    + single-seed comparison/report generation
```

Existing files/classes to preserve:

```text
ResidualProjectedFusionExitHead
FrozenEncoderResidualFusionExit
scripts/phase5_residual_fusion_exits.py
existing V2 result packets
existing V2 reports
```

---

# 46. Final pilot matrix

The complete initial study is:

```text
NO NEW PRETRAINING

existing P0 multi_block_n6 checkpoint
existing P2 multi_block_n6 checkpoint

head seed:
    0 only

tasks:
    rfs
    jamming
    human-activity
    interference
    nrpos-outdoor
    deepmimo-beam

exits:
    L2
    L4
    L6
    L8

new variants:
    bottleneck_linear
    bottleneck_sigmoid
    single_linear
    single_gelu
    single_sigmoid

existing reference:
    bottleneck_gelu
```

The dynamic residual fusion rule remains:

\[
\boxed{
h_k
=
u_k+
\sum_{j<k}g_{k,j}(x)c_{k,j}(x)
}
\]

for every variant.

The only intended experimental variable is the architecture used to generate:

\[
c_{k,j}(x).
\]

The first decision after this pilot should be whether the current two-layer GELU adapter is actually necessary, or whether a simpler one-layer or linear transformation can preserve the same gains.
