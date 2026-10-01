# Stage 1 Implementation Plan
## P2-GR-LP: Stage-Localized Hierarchical Gradients + Lightweight Auxiliary Predictors

**Repository:** `MuhammadSufyanHaider01/Hierarchical-MultiScaleJEPA`  
**Target branch:** `main`  
**Current main revision inspected:** `606782ba67414ce796100bce6a7dc24a46fd3a0d`  
**Existing method being modified conceptually:** P2 / `mask_relative_fine_to_coarse`  
**New method ID:** `p2_gr_lp`  
**Backbone:** unchanged 8-block ViT  
**Hierarchy:** unchanged fine → local → coarse target hierarchy  
**Auxiliary gradient routing:** new stage-local routing  
**Auxiliary predictor depth:** 2 blocks  
**Main L8 LatentWave predictor depth:** unchanged at 4 blocks  
**Formal training horizon:** unchanged 60 epochs  
**Primary seed:** `0`  
**Primary GPU:** RTX 5070 Ti  
**Primary formal execution profile:** same BF16/fused-attention `--performance-safe` profile used by current P2, subject to the memory gate described below.

---

# 1. Purpose

The current P2 implementation attaches hierarchical auxiliary losses at:

\[
L2,\quad L4,\quad L6
\]

using:

\[
\text{fine}\rightarrow\text{local}\rightarrow\text{coarse}
\]

target regions.

However, the current gradients are cumulative:

\[
L_2\rightarrow B_1,B_2,
\]

\[
L_4\rightarrow B_1,\ldots,B_4,
\]

\[
L_6\rightarrow B_1,\ldots,B_6.
\]

This means early transformer blocks receive auxiliary gradients from several target scales simultaneously.

The new Stage-1 model is intended to correct two architectural issues in one controlled run:

1. **localize each auxiliary loss to its intended two-block encoder stage;**
2. **reduce the capacity of the auxiliary predictors from four transformer blocks to two**, so that the context encoder must carry more of the hierarchical representation burden.

The original full-depth LatentWave anchor remains fully end-to-end.

The resulting method is:

\[
\boxed{\text{P2-GR-LP}}
\]

where:

- `GR` = stage-local gradient routing;
- `LP` = lightweight auxiliary predictor.

---

# 2. What must remain unchanged

The experiment is useful only if the following remain identical to current P2:

```text
ViT context encoder architecture
8 encoder blocks
patch size = 16
embedding dimension = 256
number of attention heads = 8
main JEPA predictor = 4 blocks
main predictor embedding dimension = 128
EMA target encoder
EMA schedule
P2 target-region construction
fine/local/coarse pooling definitions
L2/L4/L6 hierarchy locations
L8 LatentWave anchor objective
lambda_H = 1.0
hierarchy warm-up = 12 epochs
pretraining datasets
channel-subset augmentation
masking configuration
optimizer
learning-rate schedule
weight-decay schedule
60-epoch formal horizon
effective training batch = 64
BF16 precision for the masking-study comparison
fused attention
downstream frozen-head architecture
downstream split seed
downstream head-training protocol
validation-based exit selection
held-out test discipline
inference benchmark protocol
```

Do **not** change pooling geometry, loss weights, masking geometry, downstream heads, or target EMA behavior in this Stage-1 experiment.

---

# 3. What changes

Only two representation-learning components change.

## 3.1 Auxiliary gradient routing

Current P2:

```text
L2 auxiliary gradient -> B1-B2
L4 auxiliary gradient -> B1-B4
L6 auxiliary gradient -> B1-B6
L8 anchor gradient    -> B1-B8
```

New P2-GR-LP:

```text
L2 auxiliary gradient -> B1-B2 only
L4 auxiliary gradient -> B3-B4 only
L6 auxiliary gradient -> B5-B6 only
L8 anchor gradient    -> B1-B8
```

The desired parameter gradients are therefore:

\[
\nabla_{\theta_{1:2}}
=
\nabla L_{\mathrm{anchor}}
+
\lambda_H\nabla L_2,
\]

\[
\nabla_{\theta_{3:4}}
=
\nabla L_{\mathrm{anchor}}
+
\lambda_H\nabla L_4,
\]

\[
\nabla_{\theta_{5:6}}
=
\nabla L_{\mathrm{anchor}}
+
\lambda_H\nabla L_6,
\]

\[
\nabla_{\theta_{7:8}}
=
\nabla L_{\mathrm{anchor}}.
\]

The L8 anchor is never detached.

---

## 3.2 Auxiliary predictor depth

Current P2:

```text
main L8 predictor: 4 transformer blocks
L2 predictor:      4 transformer blocks
L4 predictor:      4 transformer blocks
L6 predictor:      4 transformer blocks
```

P2-GR-LP:

```text
main L8 predictor: 4 transformer blocks  # unchanged
L2 predictor:      2 transformer blocks
L4 predictor:      2 transformer blocks
L6 predictor:      2 transformer blocks
```

No predictor width/head count is changed.

---

# 4. Why the implementation should use detached stage-boundary recomputation

Do **not** use gradient hooks as the primary implementation.

Do **not** detach the normal main encoder forward.

Do **not** manually overwrite `.grad` tensors after backward.

The cleanest implementation with the current Lightning training loop is:

1. run the normal context encoder through B1-B8 once;
2. retain raw outputs at stage boundaries;
3. use the normal L8 output for the original anchor;
4. for the L4 auxiliary branch, detach the raw L2 boundary and rerun only B3-B4;
5. for the L6 auxiliary branch, detach the raw L4 boundary and rerun only B5-B6;
6. use the raw L2 output directly for the L2 auxiliary branch.

Conceptually:

```text
MAIN PATH
x -> B1 -> B2 -> B3 -> B4 -> B5 -> B6 -> B7 -> B8 -> L8 anchor
          |           |           |
          |           |           |
          |           |           +-- main raw h6
          |           +-------------- main raw h4
          +-------------------------- main raw h2

AUX L2
main h2 -> norm -> P_L2 -> L2 loss

AUX L4
stopgrad(main h2) -> B3 -> B4 -> norm -> P_L4 -> L4 loss

AUX L6
stopgrad(main h4) -> B5 -> B6 -> norm -> P_L6 -> L6 loss
```

This gives the exact desired gradient paths without changing the full encoder path.

---

# 5. Important normalization detail

The current `Encoder.forward(..., return_layers=...)` applies the encoder's final LayerNorm to each returned intermediate layer by default.

Those normalized taps must **not** be fed back into later transformer blocks.

The normal transformer path feeds raw block output:

\[
h_2^{\rm raw}
\rightarrow B_3
\rightarrow B_4.
\]

Therefore Stage-1 must request raw stage outputs using:

```python
normalize_exits=False
```

for the main hierarchy context pass.

Then:

- normalize `h2_raw` only when feeding the L2 auxiliary predictor;
- feed `h2_raw.detach()` into B3-B4 for the L4 auxiliary branch;
- normalize the recomputed B4 result before the L4 predictor;
- feed `h4_raw.detach()` into B5-B6 for the L6 auxiliary branch;
- normalize the recomputed B6 result before the L6 predictor;
- normalize the raw B8 output before the unchanged main L8 predictor.

This preserves current P2 feature semantics while changing only the backward routing.

---

# 6. Target encoder behavior must not change

The target encoder remains:

```text
EMA copy of the full context encoder
no gradients
one full B1-B8 forward
raw taps at L2/L4/L6
normalized final L8
```

There is no reason to stage-localize the target branch because it is already under:

```python
torch.no_grad()
```

Continue to construct:

```text
L2 fine targets from target L2 features
L4 local targets from target L4 features
L6 coarse targets from target L6 features
L8 anchor targets from target L8 features
```

exactly as current P2 does.

---

# 7. P2 target-region hierarchy remains unchanged

Keep `src/models/regions.py` behavior unchanged.

For the standard `2 x 14` mask:

```text
L2: 28 fine regions
L4: 7 local regions
L6: 1 coarse region
```

The target counts remain:

\[
28\rightarrow7\rightarrow1.
\]

Do not alter `_local_regions()`, `pool_target_regions()`, positional region averaging, or region sizes as part of P2-GR-LP.

This Stage-1 experiment is not the place to fix or redesign region geometry.

---

# 8. New method/config design

Create a first-class method:

```text
p2_gr_lp
```

Do not run it as:

```text
--method p2 --run-name p2_gr_lp
```

because method identity is used throughout checkpoint metadata, downstream result paths, benchmark paths, and reports.

Create:

```text
configs/p2_gr_lp.yml
```

by copying `configs/p2.yml`.

Only the hierarchy section changes.

Recommended:

```yaml
hierarchy:
  enabled: true
  mode: "mask_relative_fine_to_coarse"
  layers: [2, 4, 6]
  lambda: 1.0
  warmup_epochs: 12
  gradient_routing: "stage_local"
  aux_predictor_depth: 2
```

Keep:

```yaml
predictor:
  embed_dim: 128
  depth: 4
  num_heads: 4
```

because this still defines the unchanged main L8 predictor.

---

# 9. Extend `HierarchyConfig`

File:

```text
src/configs/components.py
```

Extend the current config with backward-compatible fields:

```python
class HierarchyConfig(BaseSettings):
    enabled: bool = False
    mode: Literal[
        "same_scale",
        "mask_relative_fine_to_coarse",
    ] = "same_scale"

    layers: list[int] = [2, 4, 6]
    lambda_h: float = Field(default=1.0, validation_alias="lambda")
    warmup_epochs: int = 12
    include_l8_coarse: bool = False

    gradient_routing: Literal[
        "cumulative",
        "stage_local",
    ] = "cumulative"

    aux_predictor_depth: int | None = Field(default=None, ge=1)
```

The defaults are critical:

```text
gradient_routing = cumulative
aux_predictor_depth = None
```

so all existing P0/P1/P2 configs retain current behavior.

---

# 10. Parse new hierarchy fields in `JEPA.__init__`

File:

```text
src/models/jepa.py
```

After current hierarchy fields, add:

```python
self.hierarchy_gradient_routing = getattr(
    hierarchy,
    "gradient_routing",
    "cumulative",
)

configured_aux_depth = getattr(
    hierarchy,
    "aux_predictor_depth",
    None,
)

self.aux_predictor_depth = (
    CONFIG.predictor.depth
    if configured_aux_depth is None
    else int(configured_aux_depth)
)
```

Then call:

```python
self._validate_hierarchy_config()
```

as currently done.

---

# 11. Auxiliary predictor construction

Current code creates every auxiliary predictor with:

```python
depth=CONFIG.predictor.depth
```

Change only auxiliary predictor construction to:

```python
depth=self.aux_predictor_depth
```

Example:

```python
self.aux_predictors = nn.ModuleDict({
    str(layer): Predictor(
        num_patches=(CONFIG.encoder.img_size[0] // CONFIG.encoder.patch_size) ** 2,
        encoder_embed_dim=CONFIG.encoder.embed_dim,
        output_dim=CONFIG.encoder.embed_dim,
        predictor_embed_dim=CONFIG.predictor.embed_dim,
        depth=self.aux_predictor_depth,
        num_heads=CONFIG.predictor.num_heads,
    )
    for layer in active_hierarchy_layers
})
```

Do **not** modify construction of:

```python
self.predictor
```

which remains:

```python
depth=CONFIG.predictor.depth
```

and therefore remains four blocks in `p2_gr_lp.yml`.

---

# 12. Hierarchy validation rules

Extend `_validate_hierarchy_config()`.

For `stage_local`, require:

```text
hierarchy enabled
mode = mask_relative_fine_to_coarse
layers exactly (2,4,6)
layers strictly increasing
include_l8_coarse = false
aux predictor depth >= 1
```

Recommended checks:

```python
if self.hierarchy_gradient_routing not in {
    "cumulative",
    "stage_local",
}:
    raise ValueError(
        f"unsupported hierarchy gradient routing: "
        f"{self.hierarchy_gradient_routing!r}"
    )

if self.aux_predictor_depth < 1:
    raise ValueError("auxiliary predictor depth must be positive")

if self.hierarchy_gradient_routing == "stage_local":
    if self.hierarchy_mode != "mask_relative_fine_to_coarse":
        raise ValueError(
            "stage-local routing is currently defined only for "
            "mask_relative_fine_to_coarse"
        )

    if self.hierarchy_layers != P2_LAYERS:
        raise ValueError(
            f"stage-local P2 requires auxiliary layers "
            f"{P2_LAYERS}, got {self.hierarchy_layers}"
        )

    if tuple(sorted(self.hierarchy_layers)) != self.hierarchy_layers:
        raise ValueError(
            "stage-local hierarchy layers must be strictly increasing"
        )

    if self.include_l8_coarse:
        raise ValueError(
            "stage-local P2-GR-LP does not support the optional "
            "L8 coarse auxiliary ablation"
        )
```

This deliberately keeps Stage-1 scope narrow.

---

# 13. Add a reusable block-range function to `Encoder`

File:

```text
src/models/encoder.py
```

Add:

```python
def forward_block_range(
    self,
    tokens: torch.Tensor,
    *,
    start_layer: int,
    end_layer: int,
    normalize_output: bool = True,
) -> torch.Tensor:
```

Recommended implementation:

```python
def forward_block_range(
    self,
    tokens,
    *,
    start_layer,
    end_layer,
    normalize_output=True,
):
    if tokens.ndim != 3:
        raise ValueError(
            "forward_block_range expects token tensor [B,N,D], "
            f"got {tuple(tokens.shape)}"
        )

    start_layer = int(start_layer)
    end_layer = int(end_layer)

    if start_layer < 1:
        raise ValueError("start_layer must be >= 1")
    if end_layer > len(self.blocks):
        raise ValueError(
            f"end_layer must be <= {len(self.blocks)}"
        )
    if start_layer > end_layer:
        raise ValueError(
            f"start_layer {start_layer} exceeds "
            f"end_layer {end_layer}"
        )

    x = tokens

    for layer_index in range(start_layer - 1, end_layer):
        x = self.blocks[layer_index](x)

    if normalize_output:
        x = self.norm(x)

    return x
```

This method must **not** patchify, add positional embeddings, apply masks, or clone blocks.

---

# 14. Why `forward_block_range()` is needed

L4:

```python
l4_context = self.context_encoder.forward_block_range(
    raw[2].detach(),
    start_layer=3,
    end_layer=4,
    normalize_output=True,
)
```

L6:

```python
l6_context = self.context_encoder.forward_block_range(
    raw[4].detach(),
    start_layer=5,
    end_layer=6,
    normalize_output=True,
)
```

The detached input blocks gradient flow to earlier stages while the reused B3-B4/B5-B6 modules receive auxiliary gradients.

---

# 15. Add a helper in `JEPA` for context hierarchy features

Add a private helper such as:

```python
def _context_hierarchy_features(
    self,
    x,
    masks_enc,
):
    ...
```

Return:

```text
context_emb
auxiliary_context_outputs
```

---

# 16. Cumulative path must remain exactly compatible

For legacy/current routing:

```python
depth = len(self.context_encoder.blocks)

requested = tuple(
    sorted(
        set(
            self.hierarchy_layers
            + (depth,)
        )
    )
)

outputs = self.context_encoder(
    x,
    masks_enc,
    return_layers=requested,
    normalize_exits=True,
)

context_emb = outputs[depth]

auxiliary_context = {
    layer: outputs[layer]
    for layer in self.hierarchy_layers
}

return context_emb, auxiliary_context
```

This path preserves current P1/P2 behavior.

---

# 17. Stage-local path

For:

```python
self.hierarchy_gradient_routing == "stage_local"
```

request raw main-path taps:

```python
raw = self.context_encoder(
    x,
    masks_enc,
    return_layers=requested,
    normalize_exits=False,
)
```

Main L8:

```python
context_emb = self.context_encoder.norm(
    raw[depth]
)
```

L2:

```python
auxiliary_context = {
    2: self.context_encoder.norm(
        raw[2]
    ),
}
```

L4:

```python
auxiliary_context[4] = (
    self.context_encoder.forward_block_range(
        raw[2].detach(),
        start_layer=3,
        end_layer=4,
        normalize_output=True,
    )
)
```

L6:

```python
auxiliary_context[6] = (
    self.context_encoder.forward_block_range(
        raw[4].detach(),
        start_layer=5,
        end_layer=6,
        normalize_output=True,
    )
)
```

Return:

```python
return context_emb, auxiliary_context
```

---

# 18. Why L6 starts from main-path raw L4

Do not chain the auxiliary branches.

Incorrect:

```text
L4 auxiliary recompute -> detach -> L6 auxiliary recompute
```

Correct:

```text
main h4 raw -> detach -> B5-B6 auxiliary recompute
```

Each stage-local branch therefore starts from the real representation produced by the main context path at the preceding stage boundary.

---

# 19. Refactor `JEPA.forward()`

Hierarchy-requested context section:

```python
if hierarchy_requested:
    context_emb, auxiliary_context_outputs = (
        self._context_hierarchy_features(
            x,
            masks_enc,
        )
    )
else:
    auxiliary_context_outputs = None
    context_emb = self.context_encoder(
        x,
        masks_enc,
    )
```

Then keep:

```python
latent_mask_pred = self.predictor(
    context_emb,
    C,
    masks_enc,
    masks_pred,
)
```

unchanged.

---

# 20. Target branch stays unchanged

Do not modify target hierarchy construction.

Continue:

```python
target_outputs = self.target_encoder(
    x,
    return_layers=target_layers,
    normalize_exits=False,
)
```

and:

```python
target_out = self.target_encoder.norm(
    target_outputs[target_depth]
)
```

and the current P2 `pool_target_regions(...)` calls.

---

# 21. Auxiliary predictor input

Use:

```python
auxiliary_context_outputs[layer]
```

rather than the old direct `context_outputs[layer]`.

For mask-relative P2:

```python
auxiliary_predictions = {
    layer: self.aux_predictors[str(layer)].forward_regions(
        auxiliary_context_outputs[layer],
        C,
        masks_enc,
        p2_layout.for_layer(
            6 if layer == len(self.context_encoder.blocks)
            else layer
        ),
        p2_layout.sizes_for_layer(
            6 if layer == len(self.context_encoder.blocks)
            else layer
        ),
    )
    for layer in self.auxiliary_layers
}
```

---

# 22. P1 remains unchanged

P1 receives defaults:

```text
gradient_routing = cumulative
aux_predictor_depth = None -> 4
```

Do not make P1 stage-local in this experiment.

---

# 23. P2 remains unchanged

Leave:

```text
configs/p2.yml
```

unchanged.

Defaults preserve:

```text
cumulative routing
4-block auxiliary predictors
```

This is necessary for a clean current-P2 comparison.

---

# 24. Optimizer requires no redesign

Current optimizer already includes:

```python
self.predictor
* self.aux_predictors.values()
```

The new 2-block predictors are automatically optimized.

Keep one AdamW optimizer and the same schedules.

---

# 25. Loss requires no redesign

Keep:

\[
L
=
L_{\rm anchor}
+
\lambda_H
\frac{L_2+L_4+L_6}{3}.
\]

No new stage weights.

No new optimizer weights.

No new loss terms.

---

# 26. Warm-up remains unchanged

Keep the existing hierarchy warm-up to full:

```text
lambda_H = 1.0
over 12 epochs
```

Do not introduce staged loss activation.

---

# 27. Channel augmentation remains unchanged

Keep current channel-subset sampling before hierarchy forward.

Do not alter `sample_channel_subsets()`.

---

# 28. Attention backend requires no special handling

The recomputed stages reuse the same `Block` modules.

`set_attention_backend(model, "fused")` already reaches them.

No extra backend logic is required.

---

# 29. Parameter-count interpretation

The context encoder parameter count is identical to P2.

The deployed model parameter count is identical to P2.

Pretraining-only auxiliary predictors contain fewer blocks.

Do not claim reduced inference parameters because auxiliary predictors are not deployed in either method.

---

# 30. Register `p2_gr_lp` in `phase5_pretrain.py`

Add:

```python
"p2_gr_lp": REPOSITORY_ROOT / "configs/p2_gr_lp.yml",
```

to `METHOD_CONFIGS`.

---

# 31. Extend safe-profile method sets

Where current code checks:

```python
{"p1", "p2"}
```

change to:

```python
{"p1", "p2", "p2_gr_lp"}
```

for:

```text
--memory-safe
--throughput-safe
--performance-safe
```

Keep numerical profile requirements unchanged.

---

# 32. Keep `include_l8_coarse` P2-only

Do not enable it for `p2_gr_lp`.

The Stage-1 definition uses only:

```text
L2
L4
L6
```

auxiliary losses plus the unchanged L8 anchor.

---

# 33. Runtime metadata

Add explicit runtime fields:

```python
"hierarchy_mode": (
    config.hierarchy.mode
    if config.hierarchy.enabled
    else None
),

"hierarchy_gradient_routing": (
    config.hierarchy.gradient_routing
    if config.hierarchy.enabled
    else None
),

"aux_predictor_depth": (
    config.hierarchy.aux_predictor_depth
    if config.hierarchy.enabled
    else None
),
```

For Stage-1 expected:

```text
mask_relative_fine_to_coarse
stage_local
2
```

---

# 34. Add `scripts/p2_gr_lp_smoke.py`

Base it on `p2_smoke.py`.

Assert config identity:

```text
mode = mask_relative_fine_to_coarse
gradient_routing = stage_local
aux_predictor_depth = 2
```

Assert model structure:

```text
main predictor blocks = 4
L2 predictor blocks = 2
L4 predictor blocks = 2
L6 predictor blocks = 2
```

Run finite forward/backward on real pretraining data.

---

# 35. Add `scripts/p2_gr_lp_gradient_audit.py`

This script is mandatory because the central architectural change is gradient routing.

For each of:

```text
L2 loss only
L4 loss only
L6 loss only
L8 anchor only
```

run a fresh forward and backward.

Record per-block context gradient norm.

Expected routing:

| Loss | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 |
|---|---|---|---|---|---|---|---|---|
| L2 | ✓ | ✓ | 0 | 0 | 0 | 0 | 0 | 0 |
| L4 | 0 | 0 | ✓ | ✓ | 0 | 0 | 0 | 0 |
| L6 | 0 | 0 | 0 | 0 | ✓ | ✓ | 0 | 0 |
| Anchor | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Use a small numerical zero threshold.

---

# 36. Why the isolated audit is required

During total-loss training all eight blocks should have gradients because the anchor is end-to-end.

Therefore total-loss block norms alone cannot prove localization.

Isolated auxiliary backward is the correct test.

---

# 37. Encoder recomputation equivalence tests

At `eval()`:

```python
raw = encoder(
    x,
    masks,
    return_layers=(2, 4, 6),
    normalize_exits=False,
)

recomputed_l4 = encoder.forward_block_range(
    raw[2].detach(),
    start_layer=3,
    end_layer=4,
    normalize_output=True,
)

expected_l4 = encoder.norm(raw[4])
```

Assert close.

Repeat for L6:

```text
raw L4 -> B5-B6 == normalized raw L6
```

This proves recomputation changes gradient graph, not forward semantics.

---

# 38. Add `tests/test_p2_gr_lp.py`

Recommended tests:

```text
test_config_defaults_preserve_p2
test_aux_predictor_depth_override
test_forward_block_range_l4_matches_main_path
test_forward_block_range_l6_matches_main_path
test_stage_local_l2_gradient_scope
test_stage_local_l4_gradient_scope
test_stage_local_l6_gradient_scope
test_anchor_gradient_reaches_all_blocks
test_target_encoder_remains_frozen
test_main_predictor_depth_remains_four
test_stage_local_rejects_l8_coarse
test_stage_local_requires_p2_layers
test_stage_local_requires_fine_to_coarse_mode
```

---

# 39. Existing P2 tests must still pass

Do not change old P2 expected outputs.

Run:

```text
tests/test_p2_regions.py
```

as a regression test.

---

# 40. Register in GPU preflight

Change:

```python
for method in ("p0", "p1", "p2"):
```

to:

```python
for method in (
    "p0",
    "p1",
    "p2",
    "p2_gr_lp",
):
```

The naming convention resolves:

```text
scripts/p2_gr_lp_smoke.py
```

---

# 41. Register in downstream exit runner

Extend method choices in:

```text
scripts/phase5_exits.py
```

to include:

```text
p2_gr_lp
```

No downstream architecture change is needed.

---

# 42. Register in benchmark runner

Extend method choices in:

```text
scripts/phase5_benchmark.py
```

to include:

```text
p2_gr_lp
```

No benchmark computation change is needed.

---

# 43. Register in generic report

Add label in:

```text
scripts/phase5_report.py
```

for:

```text
p2_gr_lp
```

Suggested label:

```text
P2-GR-LP / Stage-local hierarchy
```

Do not change the existing dedicated primary report yet.

---

# 44. Update documentation

Add:

```text
docs/p2_gr_lp.md
```

and update:

```text
docs/gpu_run_protocol.md
```

with the new method and GPU protocol.

---

# 45. Files to add

```text
configs/p2_gr_lp.yml
scripts/p2_gr_lp_smoke.py
scripts/p2_gr_lp_gradient_audit.py
tests/test_p2_gr_lp.py
docs/p2_gr_lp.md
```

---

# 46. Files to modify

```text
src/configs/components.py
src/models/encoder.py
src/models/jepa.py
scripts/gpu_preflight.py
scripts/phase5_pretrain.py
scripts/phase5_exits.py
scripts/phase5_benchmark.py
scripts/phase5_report.py
docs/gpu_run_protocol.md
```

---

# 47. Files that should remain behaviorally unchanged

```text
src/models/regions.py
src/models/predictor.py
src/models/components.py
src/models/exit_heads.py
src/evaluation/exits.py
src/phase5.py
src/checkpointing.py
configs/p0.yml
configs/p1.yml
configs/p2.yml
scripts/p0_smoke.py
scripts/p1_smoke.py
scripts/p2_smoke.py
scripts/phase5_primary_report.py
```

---

# 48. Memory implications

Stage-local routing adds extra context-block activations for recomputed:

```text
B3-B4
B5-B6
```

but removes half the auxiliary predictor blocks.

Net memory must be measured.

Do not assume B16 fits.

---

# 49. Short memory diagnostic

Before the formal job:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
.venv/bin/python scripts/phase5_pretrain.py \
  --method p2_gr_lp \
  --seed 0 \
  --run-root runs/diagnostics \
  --run-name p2_gr_lp_perf_check \
  --run-kind diagnostic \
  --max-epochs 1 \
  --batch-size 16 \
  --accumulate-grad-batches 4 \
  --num-workers 3 \
  --hdf5-worker-budget 12 \
  --prefetch-factor 4 \
  --epoch-metrics-only \
  --attention-backend fused \
  --precision bf16-mixed \
  --device cuda \
  --performance-safe \
  --limit-train-batches 10
```

Required:

```text
no OOM
finite losses
reasonable peak memory
finite optimizer step
```

---

# 50. OOM policy

If B16 fails:

1. record failure;
2. do not silently change precision/profile;
3. prefer an explicitly documented BF16 B8 / accumulation-8 fallback preserving effective batch 64;
4. for strict comparison, current P2 should ideally use the same execution profile if Stage-1 cannot use B16.

Do not compare FP32 Stage-1 to BF16 P2 as a precision-controlled architectural result.

---

# 51. Formal pretraining command

Primary `multi_block_n6` Stage-1 run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
.venv/bin/python scripts/phase5_pretrain.py \
  --method p2_gr_lp \
  --seed 0 \
  --run-root runs/masking_study/multi_block_n6 \
  --run-kind formal \
  --max-epochs 60 \
  --batch-size 16 \
  --accumulate-grad-batches 4 \
  --num-workers 3 \
  --hdf5-worker-budget 12 \
  --prefetch-factor 4 \
  --epoch-metrics-only \
  --attention-backend fused \
  --precision bf16-mixed \
  --device cuda \
  --performance-safe
```

Expected directory:

```text
runs/masking_study/multi_block_n6/p2_gr_lp/seed_000/
```

---

# 52. Checkpoint selection

Use:

```text
checkpoints/last.ckpt
```

after the fixed 60-epoch horizon.

No latent-loss early stopping.

No downstream-selected pretraining checkpoint.

---

# 53. Downstream selection-only run

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_exits.py \
  --method p2_gr_lp \
  --pretrain-seed 0 \
  --split-seed 2025 \
  --run-root runs/masking_study/multi_block_n6 \
  --checkpoint runs/masking_study/multi_block_n6/p2_gr_lp/seed_000/checkpoints/last.ckpt \
  --dataset-suite primary \
  --selection-only \
  --epochs 200 \
  --head-early-stopping-patience 20 \
  --device cuda
```

---

# 54. Held-out test run

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_exits.py \
  --method p2_gr_lp \
  --pretrain-seed 0 \
  --split-seed 2025 \
  --run-root runs/masking_study/multi_block_n6 \
  --checkpoint runs/masking_study/multi_block_n6/p2_gr_lp/seed_000/checkpoints/last.ckpt \
  --dataset-suite primary \
  --test-only \
  --device cuda
```

---

# 55. Primary downstream suite

Unchanged:

```text
rfs
jamming
human-activity
interference
nrpos-outdoor
deepmimo-beam
```

Use the same split manifest and head protocol as current P2.

---

# 56. Inference benchmark

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_benchmark.py \
  --method p2_gr_lp \
  --pretrain-seed 0 \
  --run-root runs/masking_study/multi_block_n6 \
  --checkpoint runs/masking_study/multi_block_n6/p2_gr_lp/seed_000/checkpoints/last.ckpt \
  --dataset-suite primary \
  --warmup-iterations 50 \
  --measure-iterations 200 \
  --device cuda
```

---

# 57. Same-layer FLOP invariant

For each task and exit:

\[
F_{\mathrm{P2}}(L_k)
=
F_{\mathrm{P2-GR-LP}}(L_k).
\]

If not, treat it as a configuration error.

---

# 58. Stage-1 report

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_report.py \
  --run-root runs/masking_study/multi_block_n6 \
  --methods p2 p2_gr_lp \
  --seeds 0 \
  --output-dir reports/architecture_stage1/multi_block_n6 \
  --output-prefix p2_vs_p2_gr_lp \
  --title "P2 vs P2-GR-LP: stage-local gradients and lightweight predictors"
```

---

# 59. What to compare first

Primary fixed-exit comparisons:

```text
L2
L4
L6
L8
```

for every task.

Do not rely only on the selected best exit.

The main Stage-1 question is whether shallow/intermediate representations improve.

---

# 60. Representation gap

Classification:

\[
G_l=M_{L8}-M_l.
\]

Positioning:

\[
G_l=M_l-M_{L8}.
\]

Compare P2 and P2-GR-LP at L2/L4/L6.

Smaller gap is evidence of stronger early representation.

---

# 61. Masking-study extension order

Given limited compute:

```text
1. multi_block_n6
2. region_n6
3. time_n6
4. frequency_n6
```

Do not start all four before reviewing the first result.

---

# 62. Region run

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
.venv/bin/python scripts/phase5_pretrain.py \
  --method p2_gr_lp \
  --seed 0 \
  --run-root runs/masking_study/region_n6 \
  --mask-profile region_n6 \
  --run-kind formal \
  --max-epochs 60 \
  --batch-size 16 \
  --accumulate-grad-batches 4 \
  --num-workers 3 \
  --hdf5-worker-budget 12 \
  --prefetch-factor 4 \
  --epoch-metrics-only \
  --attention-backend fused \
  --precision bf16-mixed \
  --device cuda \
  --performance-safe
```

---

# 63. Time run

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
.venv/bin/python scripts/phase5_pretrain.py \
  --method p2_gr_lp \
  --seed 0 \
  --run-root runs/masking_study/time_n6 \
  --mask-profile time_n6 \
  --run-kind formal \
  --max-epochs 60 \
  --batch-size 16 \
  --accumulate-grad-batches 4 \
  --num-workers 3 \
  --hdf5-worker-budget 12 \
  --prefetch-factor 4 \
  --epoch-metrics-only \
  --attention-backend fused \
  --precision bf16-mixed \
  --device cuda \
  --performance-safe
```

---

# 64. Frequency run

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
.venv/bin/python scripts/phase5_pretrain.py \
  --method p2_gr_lp \
  --seed 0 \
  --run-root runs/masking_study/frequency_n6 \
  --mask-profile frequency_n6 \
  --run-kind formal \
  --max-epochs 60 \
  --batch-size 16 \
  --accumulate-grad-batches 4 \
  --num-workers 3 \
  --hdf5-worker-budget 12 \
  --prefetch-factor 4 \
  --epoch-metrics-only \
  --attention-backend fused \
  --precision bf16-mixed \
  --device cuda \
  --performance-safe
```

---

# 65. Do not add random masking in Stage-1

Keep the experiment to the mask-relative rectangular conditions already used by current P2.

---

# 66. Exact isolated gradient expectations

### L2 loss only

```text
B1-B2: gradient
B3-B8: no gradient
```

### L4 loss only

```text
B1-B2: no gradient
B3-B4: gradient
B5-B8: no gradient
```

### L6 loss only

```text
B1-B4: no gradient
B5-B6: gradient
B7-B8: no gradient
```

### L8 anchor only

```text
B1-B8: gradient
```

---

# 67. Total-loss expectations

Normal training total loss:

```text
B1-B2 = anchor + L2
B3-B4 = anchor + L4
B5-B6 = anchor + L6
B7-B8 = anchor only
```

All eight context blocks therefore have nonzero gradients during total-loss training.

---

# 68. Do not detach the main encoder stages

No detach is allowed on:

```text
main B2 -> B3
main B4 -> B5
main B6 -> B7
```

The L8 anchor must continue through all eight blocks.

---

# 69. No additional target encoder

Use the existing single EMA target.

---

# 70. No separate stage optimizers

Use the current single AdamW optimizer.

---

# 71. No per-stage lambda tuning

Keep equal auxiliary loss averaging.

---

# 72. No pooling change

Keep current fine/local/coarse regions unchanged.

---

# 73. No query-position change

Keep current region positional embedding averaging unchanged.

---

# 74. Predictor width/head count unchanged

Aux predictors remain:

```text
embed dim = 128
heads = 4
output dim = 256
```

Only depth changes to 2.

---

# 75. Checkpoint compatibility

Do not resume P2-GR-LP from a P2 checkpoint.

Auxiliary predictor shapes differ.

Start from the normal seeded initialization.

---

# 76. Downstream compatibility

Context encoder architecture is unchanged.

`load_context_encoder_checkpoint()` remains compatible.

---

# 77. Resume policy

Resume only from the same P2-GR-LP run/checkpoint.

---

# 78. Seed policy

First run seed 0 only.

Additional seeds are a follow-up if the Stage-1 signal is meaningful.

---

# 79. Success pattern

Most desirable:

```text
L2 improves
L4 improves
L6 improves
L8 stays similar
selected exit becomes earlier or equally early
same-layer FLOPs remain identical
```

---

# 80. If L8 also improves

That is valid.

Stage-local training still changes shared block optimization, so full-depth quality can change.

---

# 81. If Stage-1 is worse

Possible interpretation:

```text
cumulative multi-scale gradients provided useful regularization
or
2-block auxiliary predictors are too weak
```

Because the two corrections are intentionally combined, the single run will not separate those causes.

---

# 82. If Stage-1 is unchanged

That is strong evidence for proceeding to Stage 2: a genuinely hierarchical CNN backbone.

---

# 83. If Stage-1 clearly improves

That suggests the major fault was optimization/auxiliary design rather than the flat ViT alone.

---

# 84. Recommended coding order

```text
1. config fields
2. encoder block-range helper
3. JEPA hierarchy context helper
4. aux predictor depth override
5. P2-GR-LP config
6. unit tests
7. gradient audit
8. smoke script
9. runner registration
10. GPU preflight registration
11. report registration
12. CPU tests
13. CUDA audit/smoke
14. memory diagnostic
15. formal 60-epoch run
16. downstream heads
17. test
18. benchmark
19. comparison report
```

---

# 85. CPU test command

```bash
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_p2_regions \
  tests.test_p2_gr_lp \
  tests.test_phase5_exit_stopping \
  tests.test_phase5_splits -v
```

---

# 86. CPU smoke

```bash
PYTHONPATH=src .venv/bin/python scripts/p2_gr_lp_smoke.py \
  --device cpu \
  --steps 1
```

---

# 87. Gradient audit

CPU:

```bash
PYTHONPATH=src .venv/bin/python scripts/p2_gr_lp_gradient_audit.py \
  --device cpu
```

CUDA:

```bash
PYTHONPATH=src .venv/bin/python scripts/p2_gr_lp_gradient_audit.py \
  --device cuda
```

---

# 88. Mandatory GPU preflight

```bash
PYTHONPATH=src .venv/bin/python scripts/gpu_preflight.py
```

---

# 89. Clean output directory

Before formal launch:

```text
runs/masking_study/multi_block_n6/p2_gr_lp/seed_000/
```

must not contain unrelated artifacts.

---

# 90. Formal pretraining completion gate

```text
[ ] 60 epochs complete
[ ] last.ckpt exists
[ ] experiment.json exists
[ ] summary.json exists
[ ] all losses finite
[ ] routing recorded as stage_local
[ ] aux depth recorded as 2
[ ] main predictor depth is 4
[ ] no target gradients
[ ] no silent profile changes
```

---

# 91. Downstream completion gate

```text
[ ] all six tasks
[ ] all four exits
[ ] shared split
[ ] validation-selected heads
[ ] held-out test completed after selection
[ ] no backbone fine-tuning
```

---

# 92. Benchmark completion gate

```text
[ ] 6 tasks
[ ] 4 exits
[ ] 50 warm-up
[ ] 200 timed iterations
[ ] synchronized CUDA
[ ] same-layer FLOPs equal current P2
```

---

# 93. Comparison focus

The most important comparison is:

\[
\boxed{
\text{Current P2}
\quad \text{vs} \quad
\text{P2-GR-LP}
}
\]

at each fixed exit.

Do not judge the architecture only by one aggregate average.

---

# 94. Suggested method label

Internal:

```text
p2_gr_lp
```

Paper-facing:

```text
Stage-Localized Hierarchical JEPA
```

or:

```text
P2-GR-LP
```

until results justify a final name.

---

# 95. Final Stage-1 definition

\[
h_2=S_1(x)
\]

\[
h_4=S_2(h_2)
\]

\[
h_6=S_3(h_4)
\]

\[
h_8=S_4(h_6)
\]

Auxiliary representations:

\[
\tilde h_2=h_2
\]

\[
\tilde h_4=S_2(\operatorname{sg}(h_2))
\]

\[
\tilde h_6=S_3(\operatorname{sg}(h_4))
\]

Loss:

\[
L
=
L_{\mathrm{LW}}(h_8)
+
\frac{\lambda_H}{3}
\left[
L_{\mathrm{fine}}(\tilde h_2)
+
L_{\mathrm{local}}(\tilde h_4)
+
L_{\mathrm{coarse}}(\tilde h_6)
\right].
\]

Auxiliary predictor depth:

\[
4\rightarrow2
\]

while the main L8 predictor remains:

\[
4.
\]

This is the complete Stage-1 architectural definition.

---

# 96. Additional information needed

No additional information is required from the user before implementing this Stage-1 plan.

The current `main` branch already contains everything needed for compatibility:

```text
P2 hierarchy
region pooling
raw/normalized encoder taps
predictor implementation
optimizer registration
formal pretraining runner
GPU profiles
checkpoint metadata
six-task downstream evaluation
benchmarking
generic reporting
```

The only later decision is whether to run the remaining masking conditions after reviewing the first `multi_block_n6` result.
