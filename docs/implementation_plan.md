# MultiScaleJEPA implementation plan

This repository extends LatentWave with intermediate-representation exits and
hierarchical JEPA supervision. Work is intentionally phase-gated: each phase
has a review checkpoint, and implementation stops after that phase until the
next approval.

## Research direction and fixed decisions

The pinned LatentWave JEPA implementation is the P0 reference. The extension
uses its encoder, predictor, masking, EMA target encoder, optimizer, scheduler,
and final-layer loss as the baseline. The EMA target encoder is the downstream
foundation backbone because it sees the complete input during pretraining.

The corrected extension has two variants:

- P1 adds same-scale auxiliary JEPA losses at L2, L4, and L6 using the
  corresponding EMA target features at the original fine mask resolution.
- P2 uses mask-relative fine-to-coarse target regions. For the default `2x14`
  hidden block, L2 has 28 individual patch targets, L4 has seven contiguous
  `2x2` regions, and L6 has two contiguous `2x7` regions. Target tokens are
  pooled within each region and normalized before matching.

The original L8 LatentWave loss remains the only final-layer anchor in both
variants. The implementation must assert that every target region is fully
masked and must preserve explicit batch, mask, channel, region, and embedding
axes until prediction and target tensors are matched.

## Phase 0 — Documentation and bootstrap

Scope:

- Document this plan and the research decisions.
- Vendor the official LatentWave source at commit
  `330e9ef5f6eb69e728ba43b3755036cffdd086cd`.
- Record upstream provenance and environment requirements.
- Add repository-local dataset paths and a read-only HDF5 schema audit.
- Define the future CUDA environment and record the RTX 5070 Ti as the target
  training device.
- Do not change model behavior.

Review gate: inspect the imported source tree, environment specification,
dataset paths, and documentation.

Acceptance criteria:

- Repository structure is clear.
- All seven primary HDF5 files pass schema validation.
- The available source modules import successfully, or any upstream import
  defect is explicitly recorded before Phase 1.
- No HDF5 file is tracked or staged by Git.

## Phase 1 — Exact P0 LatentWave baseline

Scope:

- Preserve the original context encoder, EMA target encoder, predictor,
  masking, optimizer, scheduler, and loss.
- Configure the four original pretraining datasets.
- Use the EMA target encoder as the downstream foundation backbone.
- Add checkpoint metadata for source commit, configuration, seed, dataset
  manifest, environment, and hardware.
- Add a numerical regression test showing hierarchy-disabled behavior matches
  the pinned baseline.
- Run CPU micro-smoke tests and then a short RTX 5070 Ti CUDA smoke run.

Review gate: inspect baseline behavior, losses, checkpoint structure, and
regression-test results.

Acceptance criteria: finite losses; no target-encoder gradients; correct EMA
updates; numerical equivalence to P0 with hierarchy disabled; and a successful
short CUDA run.

## Phase 2 — Intermediate representations and P0 exits

Scope:

- Add optional L2/L4/L6/L8 intermediate outputs without changing the default
  forward path.
- Add `forward_to_layer(x, exit_layer)` for full-input target inference.
- Apply the frozen target encoder's final normalization consistently at every
  exit.
- Add independent exit heads with mean pooling, trainable LayerNorm, and a
  linear task layer.
- Freeze the entire target backbone, including patch embedding and
  normalization.
- Evaluate RF classification, NR positioning, beam prediction, and LoS/NLoS
  classification.
- Implement fixed validation-based selection and earliest-within-tolerance
  selection without using test data.

Review gate: inspect exit shapes, frozen-parameter checks, baseline metrics,
and selection logic.

Acceptance criteria: L2/L4/L6/L8 execute correctly; only heads update; exit
results are reproducible; and test data is excluded from selection.

## Phase 3 — P1 same-scale multi-depth JEPA

Scope:

- Add independent auxiliary predictors at L2/L4/L6.
- Use same-depth EMA target features at the original fine mask resolution.
- Keep the original L8 loss as the only final-layer anchor.
- Add hierarchical configuration with `lambda_H=1.0` and a 12-epoch warm-up.
- Include auxiliary predictors in optimizer groups.
- Log per-layer losses, valid target counts, target/predictor RMS, gradients,
  and EMA momentum.

Review gate: inspect auxiliary gradients, target gradient absence, short-run
behavior, and optimizer configuration.

Acceptance criteria: auxiliary gradients are nonzero; target gradients remain
absent; losses are finite; the L8 anchor is stable; and intermediate training
signals improve or are preserved relative to P0.

## Phase 4 — P2 mask-relative fine-to-coarse JEPA

Scope:

- Build target regions from each actual prediction mask, never from global
  `14/7/4/2` pooling assumptions.
- Use the default region schedule described above for a `2x14` hidden block.
- Pool target tokens within each region, then apply LayerNorm.
- Build one predictor query per region from the learned mask token plus averaged
  fine positional embeddings.
- Keep the L8 anchor unchanged.
- Assert full masking for every target region and preserve explicit tensor axes.

Review gate: inspect synthetic pooling, region geometry, no-leakage checks,
tensor ordering, and the first P2 smoke run.

Acceptance criteria: hand-computed pooling passes; no region contains a visible
patch; prediction and target shapes match; malformed masks fail loudly; and a
finite short CUDA smoke run completes.

## Phase 5 — Main experiments

Train P0, P1, and P2 with three pretraining seeds on the RTX 5070 Ti host.
Train independent frozen target-encoder heads for all core tasks and exits.
Report mean and standard deviation for task performance by exit, selected exit,
FLOPs, batch-1 latency, and relative compute/latency savings. Use synchronized
GPU timing with warm-up iterations and fixed hardware. Keep WavesFM contextual
or deferred unless exact source and checkpoints become available.

Review gate: inspect metrics, checkpoints, logs, plots, and failure cases
before spending GPU time on ablations.

Acceptance criteria: reproducible P0/P1/P2 tables; fair early-exit comparisons;
full-depth and early-exit performance; and claims limited to observed evidence.

## Phase 6 — Ablations and OOD extension

Only after the main-result review:

- Sweep `lambda_H` over `{0.25, 0.5, 1.0, 2.0}`.
- Test optional L8 coarse supervision.
- Test alternate region schedules if P2 is promising.
- Add jamming, interference, and human-activity data as held-out OOD tasks.
- Treat alternate WiFi CSI and 5G CSI files as separate data-expansion or
  preprocessing ablations.
- Add dynamic entropy exits only for classification after fixed exits are complete.

Every ablation must answer a stated research question before GPU allocation.

## Interfaces and experiment definitions

Planned interfaces:

- `Encoder.forward(x, masks=None, return_layers=None)`
- `Encoder.forward_to_layer(x, exit_layer)`
- Hierarchical configuration fields: `enabled`, `mode`, `layers`, `lambda`,
  and `warmup_epochs`.
- Separate P0, P1, and P2 configuration files.
- Dataset audit output containing keys, shapes, dtypes, sample counts, and
  finite-value status.
- Checkpoint metadata containing source commit, configuration, seed, dataset
  manifest, environment, and hardware.

The primary pretraining set is the four original LatentWave caches. The primary
downstream set is RF sensing, NR outdoor positioning, and DeepMIMO beam/LoS
data. Extra files remain deferred until the core comparison is stable. Main
results use three seeds and report both the full-depth reference and selected
early exits. Validation determines fixed exits; test data is used only once for
final reporting.

## Human-in-the-loop protocol

For every phase:

1. Present the exact scope and expected files.
2. Wait for approval, modification, or rejection.
3. Implement only that phase.
4. Run phase-specific tests and inspect the diff.
5. Report files, results, risks, and unresolved issues.
6. Stop for review before continuing.

## Assumptions

- Full pretraining, finetuning, and latency measurement run on a CUDA-enabled
  host with the RTX 5070 Ti exposed to the driver and PyTorch.
- This bootstrap may run on CPU; the current environment is not evidence that
  the GPU runtime is ready.
- The EMA target encoder is the primary downstream backbone.
- The four original pretraining caches and three original downstream files
  define the primary experiment.
- Large HDF5 files remain local and ignored by Git.
