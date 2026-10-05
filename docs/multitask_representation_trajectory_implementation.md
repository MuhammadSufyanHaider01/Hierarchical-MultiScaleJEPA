# Multi-task representation trajectory and temporal-head robustness implementation plan

## 1. Objective

Extend the current **Interference representation-trajectory experiment** to the other downstream **classification** tasks while preserving the exact scientific protocol, leakage controls, checkpoint provenance, temporal-prefix construction, PCA fitting strategy, 256-D geometry metrics, and frozen-head robustness evaluation already implemented on `main`.

The extension must be a **generalization of the existing implementation**, not a parallel task-specific copy. Interference remains the reference regression test: after the refactor, rerunning `--task interference` with the same inputs must reproduce the current artifacts and numerical behavior within deterministic floating-point tolerance.

The target classification tasks are:

- `rfs` — RF signal classification, 20 classes.
- `jamming` — jamming classification, number of classes inferred from the canonical cache via `build_task_dataset`.
- `human-activity` — EfficientFi human activity sensing, 6 classes.
- `deepmimo-beam` — DeepMIMO beam classification, effective output count from the canonical dataset wrapper (normally the configured 16-beam task).
- `interference` remains the existing binary reference task and is not redefined.

`nrpos-outdoor` is excluded from this extension because it is a regression task. A regression-specific representation study can be designed separately.

---

## 2. Current `main` contracts that must not change

The new implementation must preserve the following contracts from the existing Interference pipeline.

### 2.1 Methods and checkpoints

Use exactly:

- `p0`
- `p1`
- `p2`
- `p2_reverse`

with the existing `CHECKPOINT_HORIZONS`, hierarchy validation, canonical multi-block/span-time six-mask signature, pretraining seed 0, config/checkpoint compatibility checks, checkpoint SHA-256 recording, and the current P2-Reverse horizon caveat.

Do not substitute different checkpoints simply because a task has another downstream packet. The trajectory study is a matched-backbone analysis.

### 2.2 Frozen context encoder

All representation extraction uses the frozen **context encoder**, not the EMA target encoder.

At each temporal-prefix condition, request normalized exits:

- L2
- L4
- L6
- L8

using `normalize_exits=True`, then mean-pool the remaining normalized tokens to obtain one 256-D vector per sample.

### 2.3 Prefix conditions

Retain the current fixed 14x14 token grid and prefix levels:

- `Tp2` -> 28 retained tokens
- `Tp4` -> 56 retained tokens
- `Tp8` -> 112 retained tokens
- `Tp14` -> 196 retained tokens

The retained tokens are the **leading columns of the original 14x14 positional grid**. Positional coordinates are never compressed, resized, or renumbered.

`Tp14` must remain exactly equivalent to ordinary full-input encoder inference; this remains a tested invariant.

### 2.4 PCA protocol

For every **task x method** pair:

1. extract validation representations for all prefix levels and all four exits;
2. flatten the validation array over sample, prefix, and layer dimensions;
3. fit one 2-component PCA using `svd_solver="full"`;
4. save that PCA artifact;
5. apply the same frozen PCA basis to the task's held-out test representations.

There is one PCA basis per method per task. Do not fit PCA on test data and do not share a PCA basis across methods.

### 2.5 Functional follow-up

The temporal-head robustness follow-up remains separate from PCA. Heads are trained only from **full-information Tp14/full-input representations** through the existing frozen-head Phase-5 machinery, and the same saved heads are then evaluated without retraining under Tp2/Tp4/Tp8/Tp14.

The backbone always remains frozen.

---

## 3. Prefix-semantics audit: mandatory before calling a study "temporal"

The code-level operation is always a **leading width/column prefix on the fixed patch grid**. The paper-level word *temporal* is only valid when the downstream cache documents that the encoder-width axis represents time.

Before launching production runs, add a small task-analysis registry with explicit metadata such as:

```python
TRAJECTORY_TASKS = {
    "interference": {"prefix_axis": "width", "prefix_semantics": "temporal", ...},
    "human-activity": {"prefix_axis": "width", "prefix_semantics": "temporal", ...},
    "jamming": {"prefix_axis": "width", "prefix_semantics": "<audit-required>", ...},
    "rfs": {"prefix_axis": "width", "prefix_semantics": "<audit-required>", ...},
    "deepmimo-beam": {"prefix_axis": "width", "prefix_semantics": "<audit-required>", ...},
}
```

The current repository already uses leading temporal windows for the human-activity input-information study, so that task can reuse the same interpretation after the cache geometry is verified.

For RFS, jamming, and DeepMIMO beam, verify the preprocessing/cache convention before the production run. If width is not physically time, still allow the same **column-prefix representation trajectory** analysis, but record and report it as `prefix_semantics="column"` (or the correct physical axis), not as temporal robustness.

The runner must refuse a user-supplied `--prefix-semantics temporal` override when the task registry has not explicitly approved the temporal interpretation.

This semantic distinction is required for paper correctness; it must not be inferred from tensor shape alone.

---

## 4. Refactor strategy

### 4.1 `src/representation_trajectories.py`

Generalize the current binary Interference helpers while preserving all method/checkpoint constants.

#### Keep unchanged

- `LAYERS = (2, 4, 6, 8)`
- `TEMPORAL_LEVELS = (2, 4, 8, 14)`
- `GRID_SIZE = 14`
- `EMBED_DIM = 256`
- `METHODS`
- `METHOD_LABELS`
- `COMPARISONS`
- `HIERARCHY`
- `CHECKPOINT_HORIZONS`
- `canonical_mask_signature`
- `validate_checkpoint_metadata`
- `temporal_token_indices`
- PCA implementation
- SHA/provenance helpers

#### Replace task-specific sample plumbing

The current `IndexedSamples` drops labels and Interference later reloads/remaps labels directly from HDF5. Replace/generalize this with a task-agnostic dataset wrapper:

```python
class IndexedTaskSamples(Dataset):
    def __init__(self, base, indices): ...

    def __getitem__(self, position):
        source_index = self.indices[position]
        sample, label = self.base[source_index][:2]
        return sample, label, source_index
```

Use `phase5.build_task_dataset(task, repository_root)` as the authoritative dataset/task-spec entry point. This is important because it already handles:

- RFS task definition;
- contiguous in-memory remapping for arbitrary jamming/interference source labels;
- HAS paths and six-class output definition;
- DeepMIMO effective beam-output count;
- canonical `TaskSpec` metadata.

Do not reimplement task-specific HDF5 keys in the trajectory scripts.

#### Generic split loading

Replace binary-only `load_split_and_labels(...)` with a task-aware helper, for example:

```python
load_task_split(manifest_path, task, dataset, spec)
```

Requirements:

- split manifest schema version 3;
- split seed 2025;
- task must exist in the canonical primary manifest;
- indices must be unique, in range, and match declared counts;
- labels are read through the canonical dataset wrapper so any remapping is identical to Phase 5;
- classification labels must be contiguous `0..K-1` after dataset loading;
- validation and test must contain every class used by the task;
- **do not hard-code 700 samples/class** for multiclass tasks;
- persist exact per-class counts in provenance.

The existing 700/700 check remains an Interference-specific regression assertion, not a global assumption.

#### Generic feature extraction

Return:

```text
features: [N, 4 temporal/prefix levels, 4 layers, 256]
labels:   [N]
indices:  [N]
```

for every task.

Require:

- task sample spatial dimensions are compatible with the 224x224 / 16-pixel-patch -> 14x14 encoder grid used by this experiment;
- finite features;
- exact persisted split order;
- exact method-to-method sample alignment.

Fail loudly if a task cache does not match the expected encoder geometry. Do not silently resize a task only for this analysis.

---

## 5. Multiclass geometry metrics

The current Interference geometry code is binary-specific. Generalize it without changing the binary mathematical meaning.

### 5.1 Normalized distance to full input

For sample `i`, layer `l`, prefix `t`:

```text
||z_i(t,l) - z_i(Tp14,l)||_2 / (||z_i(Tp14,l)||_2 + eps)
```

For reporting, compute:

- per-class mean;
- macro mean across classes;
- optional all-sample mean for diagnostics.

Use the **macro class mean** as the manuscript-facing summary so unequal class counts do not dominate.

### 5.2 Cosine similarity to full input

Keep the existing per-sample cosine definition. Report:

- per-class mean;
- macro mean across classes.

### 5.3 Multiclass class-separation ratio

The current binary quantity is:

```text
distance between the two class centroids / mean within-class RMS radius
```

Generalize it to `K` classes as:

```text
mean pairwise centroid distance / mean class RMS radius
```

where

```text
mu_c = mean representation of class c
r_c  = sqrt(mean_i_in_c ||z_i - mu_c||_2^2)
```

and

```text
separation = mean_{a<b} ||mu_a - mu_b||_2 / (mean_c r_c + eps)
```

This definition **reduces exactly to the current binary definition when K=2**, so Interference remains comparable.

### 5.4 Output schema

Keep `metrics.csv` as the compact summary with at least:

```text
method, layer, prefix_columns,
normalized_distance,
cosine_similarity,
class_separation_ratio
```

Add a new long-form `per_class_metrics.csv`:

```text
method, layer, prefix_columns, class_id,
normalized_distance, cosine_similarity, class_radius
```

For backward compatibility, the Interference path may continue emitting the legacy `*_class0` / `*_class1` columns in `metrics.csv`, but all new analysis code should use the long-form file.

---

## 6. Paired bootstrap for K classes

Retain 2,000 replicates and seed 2025.

For each bootstrap replicate:

1. resample examples **within every class** with replacement;
2. reuse the same sampled example identities across all four methods;
3. compute task-level macro geometry metrics;
4. compute the same method contrasts already defined in `COMPARISONS`.

Do not force equal source class counts. Each class is resampled to its own observed test count, and the final metric is macro-averaged over classes.

The long-form bootstrap file must continue to report method estimates and paired method-minus-reference intervals. Add class-specific rows only where useful; do not create thousands of wide class columns.

---

## 7. PCA/centroid visualization for multiclass tasks

The conceptual plot remains unchanged:

- rows/method panels follow the current report design;
- columns/exits are L2/L4/L6/L8;
- each class has centroid points at P2/P4/P8/P14 (or Tp2/Tp4/Tp8/Tp14 where temporal semantics are validated);
- connect each class centroid through increasing prefix availability;
- PCA coordinates come only from the task/method validation-fit PCA basis.

### Legibility rules

For tasks with few classes (for example HAS with 6), show class legend and labels normally.

For high-cardinality tasks such as RFS (20 classes) and DeepMIMO beam (up to 16):

- still plot **all classes**; never choose classes based on test performance;
- use small markers/lines and reduced alpha;
- annotate class IDs at the full-input endpoint only;
- omit an oversized in-panel legend if necessary;
- write `class_key.csv` containing class IDs and any available canonical label names;
- optionally generate an additional `centroid_displacement_heatmap` for readability, but this is supplementary and must not replace the all-class trajectory plot.

Do not use separate PCA fits per class, layer, or prefix. That would destroy trajectory comparability.

---

## 8. `scripts/representation_trajectories.py`

Refactor the existing script to accept:

```bash
--task {interference,rfs,jamming,human-activity,deepmimo-beam}
```

with `interference` as the backward-compatible default initially.

### Required changes

- Resolve the dataset via `build_task_dataset`.
- Resolve `num_classes` from `TaskSpec.num_outputs` and verify observed labels are compatible.
- Replace `--interference-h5` with task-agnostic dataset handling; retain the old argument only as a deprecated compatibility alias if needed for exact historical reruns.
- Validate that every source `exit_metrics.json` contains the requested task, not specifically `interference`.
- Make `validate_source_metrics(...)` task-aware.
- Store `task`, `num_classes`, `class_counts`, `sample_shape`, `prefix_axis`, and `prefix_semantics` in provenance.
- Keep the same checkpoint/config/source-packet validation.
- Keep validation and test as separate invocations.
- Require `--pca-fit-dir` for the test run exactly as today.

### Output paths

Use:

```text
runs/representation_trajectories/multi_block_n6/<task>/seed_000/validation/
runs/representation_trajectories/multi_block_n6/<task>/seed_000/test/

reports/representation_trajectories/multi_block_n6/<task>/seed_000/
```

Do not place multiple tasks into a shared `features.npz` because task-specific PCA/provenance should remain independently auditable.

### Per-task report artifacts

At minimum:

```text
metrics.csv
per_class_metrics.csv
bootstrap_intervals.csv
class_centroid_trajectories.png
class_centroid_trajectories.pdf
trajectory_metrics.png
trajectory_metrics.pdf
summary.md
```

The run directories continue to hold:

```text
features.npz
coordinates.npz
provenance.json
pca_<method>.npz              # validation only
pca_metadata.json             # validation only
metrics.csv                   # test
bootstrap_intervals.csv       # test
```

---

## 9. Frozen-head robustness generalization

### 9.1 Train task heads using the existing Phase-5 runner

Do **not** add a second head-training implementation.

For every method and head seed, use `scripts/phase5_exits.py --selection-only` with the trajectory checkpoint and canonical split manifest.

Use head seeds:

```text
2025, 2026, 2027, 2028, 2029
```

For efficiency, one selection-only run can include all four new classification tasks:

```bash
--task rfs \
--task jamming \
--task human-activity \
--task deepmimo-beam
```

That gives 4 methods x 5 head seeds = 20 selection-only runner invocations, each producing the four L2/L4/L6/L8 heads for the requested tasks.

The existing Phase-5 logic remains authoritative for task-specific class weights (including DeepMIMO beam), early stopping, validation metric, and head checkpoint format.

### 9.2 Generalize `src/temporal_head_robustness.py`

Remove the global binary assumption:

```python
CLASS_COUNT = 2
```

and pass `num_classes`/`TaskSpec` through validation, head loading, probability allocation, metrics, and bootstrap.

Required changes:

- `validate_head_packet(..., task, num_classes)`
- instantiate `ExitHead(256, num_classes)`
- probability tensor final dimension is dynamic `K`
- `classification_metrics(..., num_classes)` computes accuracy, macro accuracy, and per-class accuracy for all present classes
- preserve the current deterministic inference audit
- preserve cached feature parity checks
- preserve Tp14/full-input logit parity
- preserve 5 repeated inference passes as reproducibility checks only

### 9.3 Multiclass hierarchical bootstrap

The current functional bootstrap already resamples head seeds and examples within class. Generalize the class loop from `range(2)` to `range(K)` and macro-average class accuracies.

Continue to report:

- macro accuracy;
- absolute degradation = Tp14 macro accuracy - current-prefix macro accuracy;
- paired method comparisons;
- 95% hierarchical bootstrap intervals;
- explicit statement that uncertainty is conditional on pretraining seed 0.

### 9.4 Performance figures

For every task, generate the same two main figures as Interference:

#### `temporal_head_robustness.{png,pdf}`

Four panels:

```text
L2 | L4 | L6 | L8
```

X-axis:

```text
P2 / P4 / P8 / P14
```

(or `Tp*` only when temporal semantics are validated).

Y-axis:

```text
Macro accuracy
```

Four method curves, with error bars = sample standard deviation across head seeds.

#### `temporal_head_degradation.{png,pdf}`

Same layout, with

```text
Tp14 macro accuracy - current-prefix macro accuracy
```

as the Y-axis.

Tp14 degradation must be exactly zero for each head before aggregation.

### 9.5 Geometry/functional alignment

Continue writing `geometry_functional_alignment.csv` with:

```text
method, layer, prefix_columns,
class_separation_ratio,
normalized_distance,
cosine_similarity,
mean_macro_accuracy,
mean_absolute_degradation
```

This is the key task-level file for testing whether visually stronger geometry actually aligns with functional downstream robustness.

---

## 10. Task-aware report naming

Use task-specific report titles, for example:

```text
Representation trajectories under RF sensing
Representation trajectories under Jamming classification
Representation trajectories under Human activity sensing
Representation trajectories under DeepMIMO beam prediction
```

Do not hard-code the word `Interference` in generated summaries.

For non-temporal prefix semantics, replace phrases such as:

```text
temporal-prefix robustness
visible temporal columns
```

with the correct physical description, for example:

```text
input-prefix robustness
visible input columns
```

The provenance file decides which wording the report generator may use.

---

## 11. Required tests

Keep all existing Interference tests passing unchanged, then add multiclass coverage.

### 11.1 Representation trajectory tests

Add synthetic 3-class or 6-class fixtures that verify:

- arbitrary class counts;
- unequal class sample counts;
- deterministic label mapping;
- fixed persisted split order;
- validation-only PCA fit;
- PCA output shape `[N, 4, 4, 2]`;
- feature output shape `[N, 4, 4, 256]`;
- `Tp14` equals full-grid inference exactly;
- generalized class-separation formula reduces to the existing binary result for K=2;
- paired bootstrap is deterministic and paired across methods;
- every class is resampled within class;
- all four methods use identical sample indices.

### 11.2 Frozen-head robustness tests

Add synthetic multiclass heads and verify:

- `ExitHead(256, K)` checkpoint validation;
- probability shape `[H,R,L,T,N,K]`;
- macro accuracy over K classes;
- per-class accuracy output;
- Tp14 degradation is exactly zero;
- hierarchical bootstrap resamples all K classes;
- full-input logit parity;
- cached/runtime feature parity;
- repeated inference predictions are identical;
- wrong task, wrong K, wrong split hash, wrong checkpoint hash, or wrong head shape fails before GPU evaluation.

### 11.3 Backward-compatibility test

Use the existing Interference artifacts as a regression fixture where practical. The refactored `--task interference` path must reproduce:

- sample count/order;
- feature shapes;
- PCA metadata semantics;
- geometry metrics within deterministic tolerance;
- report filenames;
- temporal-head accuracy/degradation values within floating-point tolerance.

Do not accept a refactor that changes the current Interference conclusions merely to support multiclass tasks.

---

## 12. Production run sequence

For each eligible task, run in this order.

### Stage A — semantic/data preflight

1. verify task exists in the primary split manifest;
2. verify classification target and `num_outputs`;
3. verify 224x224-compatible encoder geometry;
4. verify label coverage in validation/test;
5. verify prefix-axis physical semantics;
6. verify all four source metric packets contain the task;
7. verify all four matched checkpoints and configs.

No PCA or GPU extraction should begin if any preflight check fails.

### Stage B — validation representation extraction and PCA fit

Example:

```bash
PYTHONPATH=src python scripts/representation_trajectories.py \
  --task human-activity \
  --split validation \
  --device cuda
```

This stage saves features and PCA artifacts only. It does not read the held-out test representations for PCA fitting.

### Stage C — held-out test trajectory evaluation

```bash
PYTHONPATH=src python scripts/representation_trajectories.py \
  --task human-activity \
  --split test \
  --pca-fit-dir runs/representation_trajectories/multi_block_n6/human-activity/seed_000/validation \
  --device cuda
```

This stage creates the geometry metrics, bootstrap intervals, PCA trajectory figure, and task summary.

### Stage D — full-information head training

For each method and each head seed, run the existing Phase-5 selection-only path on the matched trajectory checkpoint. Prefer grouping the new tasks into one invocation per method/head seed.

The output root for these study-specific heads should live below the task study tree or another clearly named trajectory-head root; never overwrite the canonical Phase-5 packet.

### Stage E — prefix robustness evaluation

Generalized command form:

```bash
PYTHONPATH=src python scripts/temporal_head_robustness.py \
  --task human-activity \
  --trajectory-provenance runs/representation_trajectories/multi_block_n6/human-activity/seed_000/test/provenance.json \
  --trajectory-test-features runs/representation_trajectories/multi_block_n6/human-activity/seed_000/test/features.npz \
  --device cuda
```

This creates:

- raw inference rows;
- per-head-seed metrics;
- aggregate metrics;
- hierarchical bootstrap intervals;
- predictions/probabilities archive;
- robustness/degradation figures;
- geometry-functional alignment;
- provenance and summary.

### Stage F — cross-task summary (after all individual task runs are frozen)

Only after every task-level artifact passes provenance checks, add a separate aggregation script/report such as:

```text
reports/representation_trajectories/multi_block_n6/cross_task_summary/
```

Do not pool raw PCA coordinates across tasks. A cross-task summary may aggregate normalized geometry metrics, macro accuracy, degradation, and method rankings, but PCA spaces remain task- and method-specific.

---

## 13. Acceptance criteria

The extension is production-ready only when all of the following are true:

1. Existing Interference tests pass.
2. New multiclass tests pass.
3. `--task interference` reproduces the current report behavior.
4. Every task uses the same canonical persisted split manifest.
5. Validation fits PCA; test never refits it.
6. All four methods see exactly the same ordered examples for a task.
7. All geometry calculations use original 256-D representations; PCA is visualization only.
8. All frozen heads are trained on full-information inputs only.
9. Head selection uses validation only; test is never used for training or selection.
10. Error bars come from independent head seeds, not repeated inference passes.
11. Bootstrap pairing is preserved across methods and stratified across all classes.
12. Prefix semantics are explicitly recorded and scientifically correct for each task.
13. P2-Reverse provenance/horizon limitations remain visible in every relevant task summary.
14. No task-specific code path silently resizes, relabels, or changes a dataset outside the canonical Phase-5 loader.

---

## 14. Paper-facing interpretation rules

The new multi-task study should be used to answer two separate questions.

### Representation geometry

Does P2/P1/P2-Reverse change how class structure develops across encoder depth and increasing input availability relative to P0?

Use:

- PCA centroid trajectories as visualization;
- normalized distance, cosine similarity, and multiclass separation ratio in the original 256-D space as quantitative evidence.

Do not use visual PCA distances alone to rank methods.

### Functional robustness

Do the geometry changes remain compatible with a classifier learned under full information?

Use:

- macro accuracy;
- full-minus-prefix degradation;
- paired bootstrap method comparisons;
- geometry-functional alignment.

A task may show stronger centroid separation but worse frozen-head robustness; report that honestly rather than treating PCA separation as proof of superior downstream utility.

---

## 15. Recommended implementation order

1. Generalize dataset/split/label plumbing in `src/representation_trajectories.py`.
2. Add multiclass geometry metrics and bootstrap.
3. Add `--task` to `scripts/representation_trajectories.py` while preserving Interference defaults.
4. Add multiclass trajectory tests and rerun the Interference regression suite.
5. Generalize `src/temporal_head_robustness.py` to dynamic class count.
6. Add `--task` to `scripts/temporal_head_robustness.py`.
7. Add multiclass robustness tests.
8. Audit prefix semantics for each target task.
9. Run validation/test trajectory extraction for one small multiclass task (recommended: `human-activity`) as the first production smoke test.
10. Train five-seed full-information heads and run robustness for that task.
11. Review artifacts before launching RFS, jamming, and DeepMIMO beam.
12. After all task-level reports are frozen, create the cross-task summary.

This order minimizes the risk of launching a large set of GPU jobs before the multiclass and semantic contracts are proven.