# Hierarchical Multi-Scale JEPA implementation plan

This repository extends LatentWave with intermediate-representation exits and
hierarchical JEPA supervision. Work is intentionally phase-gated: each phase
has a review checkpoint, and implementation stops after that phase until the
next approval.

## Version 1 design amendment

The preferred Version 1 design is recorded in
[`hierarchical_design_amendment.md`](hierarchical_design_amendment.md). It is
authoritative over older wording in this plan and the detailed historical
specification. In particular, it fixes the physical mask across context depth,
uses mask-aligned target-side pooling instead of global-grid pooling, averages
the three L2/L4/L6 losses, retains the original L8 LatentWave objective as the
only final anchor, and uses the pretrained context encoder for frozen
downstream exits.

## Research direction and fixed decisions

The pinned LatentWave JEPA implementation is the P0 reference. The extension
uses its encoder, predictor, masking, EMA target encoder, optimizer, scheduler,
and final-layer loss as the baseline. The EMA target encoder supplies
stop-gradient pretraining targets; the trained context encoder is the retained
downstream foundation backbone.

The corrected extension has two variants:

- P1 adds same-scale auxiliary JEPA losses at L2, L4, and L6 using the
  corresponding EMA target features at the original fine mask resolution.
- P2 uses mask-aligned fine-to-coarse target regions. The same physical
  prediction-mask area remains hidden at every context depth; only the target
  scale changes. For the canonical `4x4` example, L2 has 16 fine targets, L4
  has four `2x2` pooled targets, and L6 has one whole-region target. Actual
  region geometry is derived from every sample's prediction mask.

The original L8 LatentWave loss remains the only final-layer anchor in both
variants. P1 and P2 use `L_LW + lambda_H(epoch) / 3 * (L2 + L4 + L6)`.
The implementation must assert that every target region is fully masked and
preserve explicit batch, mask, channel, region, and embedding axes until
prediction and target tensors are matched.

## Conceptual lineage

| Source | Contribution to this project |
|---|---|
| LatentWave | Wireless JEPA backbone, shared per-antenna tokenizer, context/EMA-target framework, and final latent objective. |
| LocalMIM | Multi-depth supervision and fine-to-coarse target-scale philosophy. |
| V-JEPA 2.1 | Prior-art precedent for deep/intermediate JEPA supervision. |
| Fast Wireless Foundation Models with Early-Exits | Frozen-backbone task heads and task-specific exit evaluation. |
| Proposed method | Fine-to-coarse EMA latent-region prediction for stronger intermediate wireless representations. |

The research claim is not merely multi-depth JEPA supervision. The question is
whether hierarchical fine-to-coarse latent-region supervision makes frozen
intermediate wireless representations more early-exit-ready.

## Experimental baseline hierarchy

| ID | Method | Role |
|---|---|---|
| B0 | Original LatentWave | Full-depth pretraining reference. |
| B1 | Original LatentWave + frozen early-exit heads | Required post-hoc early-exit baseline. |
| B2 | Multi-depth same-scale JEPA (P1) | Isolates deep supervision. |
| B3 | Hierarchical Multi-Scale JEPA (P2) | Tests fine-to-coarse target aggregation beyond B2. |
| B4 | WavesFM | Contextual/deferred unless comparable source, checkpoints, and protocol are available. |
| B5 | V-JEPA-style deep supervision | Optional only if materially distinct from B2. |

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
- Preserve both checkpoint namespaces; later frozen early-exit experiments use
  the trained context encoder, while the EMA target encoder remains the
  stop-gradient pretraining branch.
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
- Add `forward_to_layer(x, exit_layer)` for full-input representation reads.
- Apply the encoder's final normalization consistently at every exit.
- Add independent exit heads with mean pooling, trainable LayerNorm, and a
  linear task layer.
- Freeze the entire context backbone, including patch embedding and
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
- Use `L_LW + lambda_H(epoch) / 3 * (L2 + L4 + L6)`, with `lambda_H=1.0`
  and a 12-epoch warm-up.
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
  target-grid or adaptive-pooling assumptions.
- Keep the context mask fixed across depth. Extract that same physical area
  from each full target latent grid and pool it only in a target-side branch.
- Use the canonical schedule of fine L2, local `2x2` L4 pooling, and one
  whole-block L6 target; derive counts and geometry from each actual mask.
- Pool target tokens within each region, then apply the target normalization
  policy without inserting pooling into the target backbone.
- Build one predictor query per region from the learned mask token plus averaged
  fine positional embeddings.
- Keep the L8 anchor unchanged.
- Assert full masking for every target region and preserve explicit tensor axes.
- Make each auxiliary predictor directly produce one vector per region; do not
  pool fine predictor outputs afterward.

Review gate: inspect synthetic pooling, region geometry, no-leakage checks,
tensor ordering, and the first P2 smoke run.

Acceptance criteria: hand-computed mask-aligned pooling passes; no region
contains a visible patch; the target encoder remains a full-depth EMA branch;
prediction and target shapes/order match; malformed or unsupported masks fail
loudly; and a finite short CUDA smoke run completes.

Implementation reference: [`p2_mask_relative.md`](p2_mask_relative.md).
The Phase 4 implementation and CPU/CUDA smoke verification are complete;
main-result training remains Phase 5 work.

## Phase 5 — Main experiments

Train B0/P0, B2/P1, and B3/P2 with three pretraining seeds on the RTX 5070 Ti
host. Train independent frozen context-encoder heads for all core tasks and
exits. B1 is original LatentWave with the same frozen early-exit heads. Each
GPU command requires separate explicit approval under
[`gpu_run_protocol.md`](gpu_run_protocol.md); no seeds, evaluations, or
ablations are queued automatically.
Report mean and standard deviation for task performance by exit, selected exit,
FLOPs, batch-1 latency, and relative compute/latency savings. Use synchronized
GPU timing with warm-up iterations and fixed hardware. Keep WavesFM contextual
or deferred unless exact source and checkpoints become available.

Implementation protocol: [`phase5_main_experiments.md`](phase5_main_experiments.md).
Formal Phase 5 runs persist a shared 80/10/10 downstream split manifest and a
fixed 10% held-out pretraining-cache split. Pretraining logs both train and
latent validation loss per epoch, uses validation-loss early stopping with a
240-epoch upper bound, selects exits using downstream validation only, and does
not run the legacy full-dataset downstream probe during pretraining.

Review gate: inspect metrics, checkpoints, logs, plots, and failure cases
before spending GPU time on ablations.

Acceptance criteria: reproducible P0/P1/P2 tables; fair early-exit comparisons;
full-depth and early-exit performance; reported pretraining overhead separate
from inference savings; and claims limited to observed evidence.

## Phase 6 — Ablations and OOD extension

Only after the main-result review:

- Sweep `lambda_H` over `{0.25, 0.5, 1.0, 2.0}`.
- Test optional coarse L8 supervision only as an ablation.
- Test alternate region schedules if P2 is promising.
- Add jamming, interference, and human-activity data as held-out OOD tasks.
- Treat alternate WiFi CSI and 5G CSI files as separate data-expansion or
  preprocessing ablations.
- Add dynamic entropy exits only for classification after fixed exits are complete.

Every ablation must answer a stated research question before GPU allocation.

Implementation protocol: [`phase6_ablations_ood.md`](phase6_ablations_ood.md).
The Phase 6 framework provides lambda and opt-in L8-coarse P2 ablations,
held-out jamming/interference/human-activity task registration, and
classification-only dynamic entropy routing. GPU execution remains deferred
until the Phase 5 main-result review.

## Interfaces and experiment definitions

Planned interfaces:

- `Encoder.forward(x, masks=None, return_layers=None)`
- `Encoder.forward_to_layer(x, exit_layer)`
- Hierarchical configuration fields: `enabled`, `mode`, `layers`, `lambda`,
  `warmup_epochs`, and the mask-aligned target-scale schedule.
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
- The EMA target encoder is the full-input, stop-gradient pretraining branch;
  the context encoder is the primary downstream backbone.
- The four original pretraining caches and three original downstream files
  define the primary experiment.
- Large HDF5 files remain local and ignored by Git.
