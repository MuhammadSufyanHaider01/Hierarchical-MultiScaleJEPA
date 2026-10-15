# Fusion V2 Implementation Specification
## Residual Projected Dynamic Fusion for Early-Exit Evaluation

**Repository:** `MuhammadSufyanHaider01/Hierarchical-MultiScaleJEPA`  
**Target branch:** `main`

---

## 1. Objective

Implement a new early-exit fusion head that replaces the current scalar convex fusion with **Residual Projected Dynamic Fusion**.

The existing fusion performs mean pooling at each causal exit and combines the pooled vectors using global softmax weights whose sum is 1. The new method must instead:

1. keep the current target-exit representation as an unchanged anchor;
2. project every earlier exit through a `256 -> 64 -> 256` residual adapter;
3. compute an independent **sample-dependent sigmoid gate** for every earlier exit;
4. add the gated residual corrections to the anchor;
5. keep the ViT encoder fully frozen.

Do **not** overwrite Current EE or the existing Causal Fusion V1 implementation. This must be added as a new algorithm.

Use the identifier:

```text
residual_projected_dynamic_fusion
```

---

## 2. First experimental scope

The first run is a **single-head-seed pilot** only.

```text
backbone: ViT
masking condition: multi_block_n6
pretraining methods: p0, p2
pretrain seed: 0
head seed: 0
split seed: 2025
exit layers: [2, 4, 6, 8]
dataset suite: primary
distillation: disabled
encoder fine-tuning: disabled
```

Use all six existing `PRIMARY_TASKS` from `src/phase5.py`:

```text
rfs
jamming
human-activity
interference
nrpos-outdoor
deepmimo-beam
```

This produces:

```text
2 pretrained methods x 6 tasks x 4 exits = 48 Fusion-V2 heads
```

Do **not** launch the 7-head-seed experiment yet.

---

## 3. Current repository behavior to preserve

Relevant current files:

```text
src/models/encoder.py
src/models/exit_heads.py
src/phase5.py
scripts/phase5_exits.py
scripts/phase5_fusion_exits.py
```

Preserve the existing downstream protocol:

```text
epochs = 200
batch_size = 32
num_workers = 4
learning_rate = 1e-3
weight_decay = 0.05
head early-stopping patience = 20
head early-stopping min_delta = 0
split seed = 2025
optimizer = AdamW
```

Also preserve:

- current dataset construction;
- the existing persisted train/validation/test split;
- task-specific metrics;
- DeepMIMO class weighting;
- classification loss;
- NR-positioning loss;
- deterministic seeding;
- validation-based early stopping;
- validation-only exit selection;
- test evaluation only after exit selection.

The pretrained encoder must not be fine-tuned.

---

## 4. Keep Fusion V1 intact

Current classes in `src/models/exit_heads.py` include:

```python
ExitHead
FrozenEncoderExit
CausalFusionExitHead
FrozenEncoderFusionExit
```

Do not modify their behavior.

Add:

```python
ResidualProjectedFusionExitHead
FrozenEncoderResidualFusionExit
```

The repository should be able to distinguish:

```text
current
causal_fusion
residual_projected_dynamic_fusion
```

---

## 5. Encoder path

Use the existing:

```python
encoder.forward_to_layers(x, exit_layers)
```

Examples:

```text
L2 -> [2]
L4 -> [2, 4]
L6 -> [2, 4, 6]
L8 -> [2, 4, 6, 8]
```

For L6, execute the encoder only through block 6. Do not call `forward_to_layer()` independently for L2, L4, and L6.

The encoder must remain:

```python
requires_grad = False
encoder.eval()
```

and representation extraction must execute under:

```python
torch.no_grad()
```

---

## 6. Source sets

For target exit `k`, define:

\[
\mathcal{S}_k = \{j \in \{2,4,6,8\}: j < k\}.
\]

Therefore:

```text
L2: {}
L4: {2}
L6: {2, 4}
L8: {2, 4, 6}
```

The target exit itself is the **anchor** and is not included in the residual source set.

---

## 7. Mean pooling

For every requested encoder representation:

\[
Z_j \in \mathbb{R}^{B \times N \times 256},
\]

use the same mean pooling as Current EE:

\[
u_j = \frac{1}{N}\sum_{n=1}^{N} Z_{j,n}.
\]

Therefore:

\[
u_j \in \mathbb{R}^{B \times 256}.
\]

Do not introduce attention pooling in this first implementation.

---

## 8. Core Fusion-V2 equation

For target exit `k`:

\[
h_k
=
u_k
+
\sum_{j\in\mathcal{S}_k}
 g_{k,j}(x)c_{k,j}(x).
\]

Where:

- `u_k` is the pooled target-exit anchor;
- `c_{k,j}` is a learned projected correction from source exit `j`;
- `g_{k,j}(x)` is an independent sample-dependent sigmoid gate.

There is **no global softmax over source exits** and no condition that source gates sum to 1.

---

## 9. Residual adapter

For every source exit `j < k`, create a separate adapter.

First normalize:

\[
\bar u_j = LN_{k,j}(u_j).
\]

Down-project:

\[
v_{k,j}
=
GELU\left(W^{\downarrow}_{k,j}\bar u_j+b^{\downarrow}_{k,j}\right),
\]

with:

\[
W^{\downarrow}_{k,j}\in\mathbb{R}^{64\times256}.
\]

Thus:

\[
v_{k,j}\in\mathbb{R}^{64}.
\]

Then up-project:

\[
c_{k,j}
=
W^{\uparrow}_{k,j}v_{k,j}+b^{\uparrow}_{k,j},
\]

with:

\[
W^{\uparrow}_{k,j}\in\mathbb{R}^{256\times64}.
\]

Thus:

\[
c_{k,j}\in\mathbb{R}^{256}.
\]

Recommended modules:

```python
source_norms[j] = nn.LayerNorm(256)
down_projections[j] = nn.Linear(256, 64)
up_projections[j] = nn.Linear(64, 256)
```

Use `nn.ModuleDict` for source-specific modules.

---

## 10. Zero initialization

Zero initialize every source up-projection:

```python
nn.init.zeros_(up_projection.weight)
nn.init.zeros_(up_projection.bias)
```

This must guarantee initially:

\[
c_{k,j}=0
\]

and therefore:

\[
h_k=u_k.
\]

This identity behavior is required. Fusion V2 must begin from the current target-exit representation rather than from a uniform mixture of exits.

---

## 11. Dynamic gate

Create an anchor gate branch:

```python
anchor_gate_norm = nn.LayerNorm(256)
anchor_gate_projection = nn.Linear(256, 64)
```

Compute:

\[
q_k = W^g_k LN^g_k(u_k),
\]

where:

\[
q_k\in\mathbb{R}^{64}.
\]

For each source `j`, create:

```python
source_gate_projections[j] = nn.Linear(64, 64)
gate_scores[j] = nn.Linear(64, 1)
```

Then compute:

\[
s_{k,j}
=
\tanh\left(q_k+W^s_{k,j}v_{k,j}\right),
\]

\[
e_{k,j}
=
w^T_{k,j}s_{k,j}+b_{k,j},
\]

and finally:

\[
g_{k,j}(x)=\sigma(e_{k,j}).
\]

Gate shape:

```text
[B, 1]
```

and it should broadcast over the 256-dimensional correction vector.

---

## 12. Remove old alpha weighting

Fusion V2 must not use:

```python
fusion_logits
softmax(fusion_logits)
```

There is no cumulative source weight of 1.

For example, all of the following are valid for one L8 sample:

```text
g8,2 = 0.80
g8,4 = 0.65
g8,6 = 0.91
```

The gates are independent.

---

## 13. Residual fusion

For each source:

\[
r_{k,j}=g_{k,j}(x)c_{k,j}(x).
\]

Then:

\[
r_k=\sum_{j<k}r_{k,j},
\]

and:

\[
h_k=u_k+r_k.
\]

Explicitly:

### L2

\[
h_2=u_2.
\]

### L4

\[
h_4=u_4+g_{4,2}c_{4,2}.
\]

### L6

\[
h_6=u_6+g_{6,2}c_{6,2}+g_{6,4}c_{6,4}.
\]

### L8

\[
h_8=u_8+g_{8,2}c_{8,2}+g_{8,4}c_{8,4}+g_{8,6}c_{8,6}.
\]

The target-exit anchor always enters with coefficient 1.

---

## 14. Final task head

After fusion:

\[
\tilde h_k = LN^{out}_k(h_k),
\]

then:

\[
\hat y_k=W^{out}_k\tilde h_k+b^{out}_k.
\]

Recommended modules:

```python
self.output_norm = nn.LayerNorm(256)
self.output_projection = nn.Linear(256, num_outputs)
```

Keep the final classifier/regressor linear. Do not add an extra task MLP in this pilot.

---

## 15. Recommended class API

Add to `src/models/exit_heads.py`:

```python
class ResidualProjectedFusionExitHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_outputs: int,
        source_layers,
        bottleneck_dim: int = 64,
        gate_dim: int = 64,
    ):
        ...
```

and:

```python
class FrozenEncoderResidualFusionExit(nn.Module):
    ...
```

Suggested members:

```python
self.source_layers
self.bottleneck_dim
self.gate_dim

self.output_norm
self.output_projection

self.anchor_gate_norm
self.anchor_gate_projection

self.source_norms
self.down_projections
self.up_projections
self.source_gate_projections
self.gate_scores
```

For L2, source-specific modules and gate modules may be empty.

---

## 16. Forward-pass reference

Conceptually:

```python
def forward(self, representations, return_diagnostics=False):
    pooled = {
        layer: tokens.mean(dim=1)
        for layer, tokens in representations.items()
    }

    anchor = pooled[self.exit_layer]

    if not self.source_layers:
        fused = anchor
        diagnostics = {}
    else:
        anchor_query = self.anchor_gate_projection(
            self.anchor_gate_norm(anchor)
        )

        residual_sum = torch.zeros_like(anchor)
        diagnostics = {}

        for source_layer in self.source_layers:
            key = str(source_layer)
            source = pooled[source_layer]

            source_norm = self.source_norms[key](source)
            hidden = F.gelu(
                self.down_projections[key](source_norm)
            )

            correction = self.up_projections[key](hidden)

            gate_hidden = torch.tanh(
                anchor_query
                + self.source_gate_projections[key](hidden)
            )

            gate = torch.sigmoid(
                self.gate_scores[key](gate_hidden)
            )

            residual = gate * correction
            residual_sum = residual_sum + residual

            if return_diagnostics:
                anchor_norm = torch.linalg.vector_norm(anchor, dim=-1)
                residual_norm = torch.linalg.vector_norm(residual, dim=-1)
                diagnostics[source_layer] = {
                    "gate": gate.squeeze(-1),
                    "residual_norm": residual_norm,
                    "relative_residual_norm": residual_norm / (anchor_norm + 1e-8),
                }

        fused = anchor + residual_sum

    output = self.output_projection(
        self.output_norm(fused)
    )

    if return_diagnostics:
        return output, diagnostics

    return output
```

This is a reference structure; adapt naming/style to repository conventions.

---

## 17. Frozen wrapper behavior

`FrozenEncoderResidualFusionExit` should parallel the current `FrozenEncoderFusionExit`.

For target exit `k`:

```python
self.source_layers = tuple(
    layer for layer in (2, 4, 6, 8)
    if layer < self.exit_layer
)

self.requested_layers = self.source_layers + (self.exit_layer,)
```

Representation extraction:

```python
with torch.no_grad():
    representations = self.encoder.forward_to_layers(
        x,
        self.requested_layers,
    )
```

Keep the encoder in `.eval()` even when the head is in training mode.

---

## 18. New training script

Do not overwrite:

```text
scripts/phase5_fusion_exits.py
```

Add:

```text
scripts/phase5_residual_fusion_exits.py
```

Use Current EE and Fusion V1 as implementation templates.

Reuse existing helpers from `scripts/phase5_exits.py` where practical:

```python
EXIT_LAYERS
_criterion
_evaluate
_loader
_new_encoder
_seed_everything
```

The principal model replacement is:

```text
FrozenEncoderFusionExit
```

with:

```text
FrozenEncoderResidualFusionExit
```

Do not create a separate dataset/evaluation stack.

---

## 19. Locked split manifest

Reuse exactly:

```text
runs/masking_head_seeds/
multi_block_n6/
head_seed_000/
splits_seed_2025_primary_v3.json
```

The runner should fail if this manifest is missing.

Do not create a new split.

---

## 20. Seed behavior

Use explicitly:

```text
pretrain_seed = 0
head_seed = 0
```

Preserve the current task/exit head seed formula:

\[
s_{t,k}=s_h+100t+k.
\]

Preserve the current training-loader seed behavior:

\[
s_{loader}=s_h+t.
\]

This keeps the pilot paired with existing Current EE/Fusion-V1 runs.

---

## 21. Optimizer parameter groups

Use two AdamW groups.

### Weight decay = `0.05`

Apply to ordinary feature/task projection weights:

```text
down_projection weights
up_projection weights
output_projection weight
```

### Weight decay = `0.0`

Apply to:

```text
all LayerNorm parameters
all biases
anchor gate projection
source gate projections
gate score parameters
```

Conceptual optimizer:

```python
optimizer = torch.optim.AdamW(
    [
        {
            "params": decay_params,
            "weight_decay": args.weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ],
    lr=args.learning_rate,
)
```

Keep:

```text
learning_rate = 1e-3
```

for the first pilot.

---

## 22. Training objective

Do not introduce any auxiliary loss.

Classification:

\[
\mathcal{L}=CE(\hat y,y).
\]

Use the current `_criterion(...)`, including existing DeepMIMO class weighting.

NR positioning:

\[
\mathcal{L}=MSE(\hat y,y).
\]

Do not add:

```text
KD loss
feature matching loss
gate sparsity loss
gate entropy loss
residual regularization
```

in this pilot.

---

## 23. Early stopping

Preserve current behavior:

```text
max epochs = 200
patience = 20
min_delta = 0
```

Monitor the current task-specific validation metric.

Save only the best validation checkpoint.

---

## 24. Frozen-gradient check

After every backward pass, assert that all encoder gradients remain `None`.

If not, raise a runtime error.

---

## 25. Checkpoint format

Recommended checkpoint payload:

```python
{
    "head_algorithm":
        "residual_projected_dynamic_fusion",

    "exit_layer": layer,
    "source_layers": list(model.source_layers),

    "embed_dim": 256,
    "bottleneck_dim": 64,
    "gate_dim": 64,

    "head": model.head.state_dict(),
    "epoch": best_epoch,
    "validation": validation_metrics,
}
```

Do not save a Fusion-V1-style global `fusion_weights` field.

---

## 26. Diagnostic evaluation from the best checkpoint

After finding the best validation checkpoint:

1. reload the best head;
2. run the validation set once with `return_diagnostics=True`;
3. aggregate the dynamic-gate statistics.

For every source exit, save:

```text
gate_mean
gate_sd
gate_median
gate_p10
gate_p90
residual_norm_mean
relative_residual_norm_mean
```

Where:

\[
R_{k,j}(x)
=
\|g_{k,j}(x)c_{k,j}(x)\|_2
\]

and:

\[
\tilde R_{k,j}(x)
=
\frac{\|g_{k,j}(x)c_{k,j}(x)\|_2}
{\|u_k(x)\|_2+10^{-8}}.
\]

The diagnostics must correspond to the **best validation checkpoint**, not the final epoch.

---

## 27. `exit_metrics.json`

Top-level fields should include at least:

```json
{
  "head_algorithm": "residual_projected_dynamic_fusion",
  "method": "p0",
  "pretrain_seed": 0,
  "head_seed": 0,
  "split_seed": 2025,
  "tasks": {}
}
```

Per exit, include:

```json
{
  "best_epoch": 0,
  "validation": {},
  "head_path": "...",
  "source_layers": [2, 4],
  "fusion_diagnostics": {
    "2": {
      "gate_mean": 0.0,
      "gate_sd": 0.0,
      "gate_median": 0.0,
      "gate_p10": 0.0,
      "gate_p90": 0.0,
      "residual_norm_mean": 0.0,
      "relative_residual_norm_mean": 0.0
    }
  }
}
```

The numeric values shown above are schema placeholders only.

---

## 28. Exit selection

Use the existing validation-only best-exit selector.

For classification:

\[
k^*=\arg\max_k M_k^{val}.
\]

For positioning:

\[
k^*=\arg\min_k M_k^{val}.
\]

Use:

```text
selected_exit_best_validation
```

for the headline comparison.

The old tolerance field may remain for compatibility but must not drive the pilot conclusions.

---

## 29. Test discipline

Required sequence:

```text
train head
-> validation early stopping
-> validation exit selection
-> freeze selected exit
-> test evaluation
```

The test split must not influence:

```text
epoch choice
exit choice
gate design
bottleneck size
learning rate
```

---

## 30. Run directory

Use an isolated run path:

```text
runs/
└── ee_optimization/
    └── multi_block_n6/
        └── residual_dynamic_fusion/
            └── head_seed_000/
                ├── p0/
                │   └── seed_000/
                │       └── downstream/
                │           ├── exit_metrics.json
                │           └── heads/
                │               ├── rfs/
                │               │   ├── L2.pt
                │               │   ├── L4.pt
                │               │   ├── L6.pt
                │               │   └── L8.pt
                │               └── ...
                └── p2/
                    └── seed_000/
                        └── downstream/
```

Do not overwrite the existing Fusion-V1 run directory.

---

## 31. Pilot jobs

Only two full jobs are required initially:

```text
P0 / multi_block_n6 / pretrain_seed_0 / head_seed_0
P2 / multi_block_n6 / pretrain_seed_0 / head_seed_0
```

Each job runs:

```text
6 tasks x 4 exits
```

No multi-seed orchestration is required yet.

---

## 32. Smoke test

Before the two full pilot jobs, run a small smoke test, preferably:

```text
method: p2
task: rfs
exit: L6
head seed: 0
limited train batches
limited validation batches
```

Verify:

- model construction;
- one encoder prefix pass;
- L6 obtains L2/L4/L6 tensors;
- blocks 7–8 are not executed;
- encoder receives no gradients;
- initial residual corrections are zero;
- gates lie in `(0,1)`;
- no source softmax exists;
- checkpoint save/reload works;
- diagnostic aggregation works.

---

## 33. Required sanity tests

### L2 equivalence

At L2:

\[
h_2=u_2.
\]

There should be no source adapters or gates.

### Zero-initialization identity

Before the first optimizer step:

\[
c_{k,j}=0,
\]

hence:

\[
h_k=u_k.
\]

Test this numerically.

### Gate range

Verify:

\[
0<g_{k,j}<1.
\]

### No gate normalization

Do not enforce:

\[
\sum_j g_{k,j}=1.
\]

### Frozen backbone

All encoder gradients must remain `None`.

### Prefix truncation

L6 must stop after block 6.

### One encoder pass

All causal taps for one target exit must come from one `forward_to_layers(...)` execution.

### Best-checkpoint diagnostics

Gate statistics must come from the best validation checkpoint.

### Same split

Reuse the locked M-block split manifest exactly.

### Test after selection

No held-out result may influence the selected exit.

---

## 34. Single-seed pilot report

Create:

```text
reports/
└── early_exit_optimization/
    └── multi_block_n6/
        └── residual_dynamic_fusion_single_seed/
```

Do not overwrite the existing two-stage report.

Compare:

```text
P0:
    Current EE
    Causal Fusion V1
    Residual Dynamic Fusion V2

P2:
    Current EE
    Causal Fusion V1
    Residual Dynamic Fusion V2
```

Only head seed 0 is included.

---

## 35. Required pilot table

For every task, report all fixed exits:

| Algorithm | L2 | L4 | L6 | L8 |
|---|---:|---:|---:|---:|
| Current EE | ... | ... | ... | ... |
| Fusion V1 | ... | ... | ... | ... |
| Fusion V2 | ... | ... | ... | ... |

Produce separately for P0 and P2.

Also report:

```text
best validation exit
selected test metric
```

for every algorithm.

---

## 36. Direction-aware gains

For classification:

\[
\Delta_k^{V2}
=
M_k^{V2}-M_k^{Current}.
\]

For positioning:

\[
\Delta_k^{V2}
=
M_k^{Current}-M_k^{V2}.
\]

Positive always means improvement.

Also calculate:

\[
\Delta_k^{V2-V1}.
\]

The main pilot question is whether Fusion V2 materially improves fixed-exit performance over both Current EE and Fusion V1.

---

## 37. Gate diagnostic table

For L4/L6/L8, report:

| Method | Task | Target exit | Source exit | Gate mean | Gate SD | Relative residual norm mean |
|---|---|---:|---:|---:|---:|---:|

Produce separately for P0 and P2.

Do not report a global alpha for Fusion V2.

---

## 38. Decision after this pilot

Do **not** redesign or rerun distillation as part of this task.

First inspect the single-seed Fusion-V2 results.

If Fusion V2 produces meaningful fixed-exit gains, especially at L4/L6/L8, the next experiment will be:

```text
Residual Dynamic Fusion V2
P0 + P2
multi_block_n6
head seeds 0–6
```

Only after confirming the stronger teacher is reproducible should the KD stage be redesigned.

---

## 39. Explicit non-goals

Do not add:

```text
new pretraining
P1
P2-Reverse
ConvNeXt
other masking conditions
attention pooling
encoder fine-tuning
LoRA
ViT internal adapters
feature distillation
logit distillation
cross-depth KD
multi-teacher KD
gate sparsity loss
gate entropy loss
new exit-selection policy
seven-head-seed Fusion-V2 jobs
```

This task is only the Fusion-V2 single-seed pilot.

---

## 40. Expected implementation change set

Expected files:

```text
src/models/exit_heads.py
    + ResidualProjectedFusionExitHead
    + FrozenEncoderResidualFusionExit

scripts/phase5_residual_fusion_exits.py
    + Fusion-V2 train/evaluate runner

optional:
scripts/residual_fusion_single_seed_report.py
    + pilot report generator
```

Avoid modifying:

```text
src/models/encoder.py
scripts/phase5_exits.py
scripts/phase5_fusion_exits.py
```

unless a very small helper/compatibility change is strictly necessary.

---

## 41. Final intended experiment

The first formal Fusion-V2 experiment is exactly:

```text
ViT backbone
M-block masking only

P0 checkpoint:
    pretrain seed 0
    head seed 0

P2 checkpoint:
    pretrain seed 0
    head seed 0

6 primary tasks
4 exits: L2, L4, L6, L8

same frozen encoder
same split
same task losses
same metrics
same early stopping
same validation-only exit selection

comparison:
    Current EE
    Fusion V1
    Fusion V2
```

The only substantive model change should be:

```text
global convex scalar fusion
```

replaced by:

```text
target anchor
+
projected residual corrections
x
sample-dependent sigmoid gates
```

All previous Current EE and Fusion-V1 results must remain reproducible and untouched.
