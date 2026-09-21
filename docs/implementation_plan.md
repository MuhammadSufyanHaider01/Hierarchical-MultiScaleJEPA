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

The corrected extension has two trainable variants plus a post-hoc early-exit
track:

- P1 adds same-scale auxiliary JEPA losses at L2, L4, and L6 using the
  corresponding EMA target features at the original fine mask resolution. It
  is retained as a controlled ablation, not one of the three primary methods.
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
| B2 | Multi-depth same-scale JEPA (P1) | Controlled ablation isolating intermediate supervision. |
| B3 | Hierarchical Multi-Scale JEPA (P2) | Primary proposed method; tests fine-to-coarse target aggregation. |
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
- All ten primary HDF5 inputs pass schema validation.
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
- Evaluate the six-task suite: RF classification, jamming classification,
  EfficientFi WiFi human activity recognition, interference classification,
  NR positioning, and beam prediction.
- Implement fixed validation-based selection and earliest-within-tolerance
  selection without using test data.

Review gate: inspect exit shapes, frozen-parameter checks, baseline metrics,
and selection logic.

Acceptance criteria: L2/L4/L6/L8 execute correctly; only heads update; exit
results are reproducible; and test data is excluded from selection.

### Six-task primary downstream suite

Every new masking-regime comparison and every post-epoch-60 P2 evaluation uses
the following six labelled downstream tasks. They are used only to train and
evaluate frozen task heads; none is added to the four-cache pretraining set.

| Task family | Task | Local cache | Target | Primary metric |
|---|---|---|---|---|
| Spectral / frequency | RF signal classification | `data/downstream/rfs.h5` | 20 classes | Macro accuracy |
| Spectral / frequency | Jamming classification | `../jamming.h5` | 6 classes | Macro accuracy |
| Temporal / cyclostationary | EfficientFi WiFi human activity recognition | `../has.h5` | 6 activities | Macro accuracy |
| Temporal / cyclostationary | Interference classification | `../interference.h5` | 2 classes | Macro accuracy |
| Channel / spatial geometry | 5G NR positioning | `data/downstream/nrpos-outdoor.h5` | 3-D position | Mean distance error |
| Channel / spatial geometry | DeepMIMO beam prediction | `data/downstream/deepmimo.h5` | 16 beams | Macro accuracy |

The task pairs define a pre-registered layer-affinity analysis; they do not
assume that an earlier or later layer must win. Every task uses independent
frozen L2/L4/L6/L8 heads, the same persisted 80/10/10 split protocol, and
validation-only head/exit selection. The local interference labels `{1,2}`
are remapped in memory to `{0,1}`; the HDF5 source file is never modified.

Before the first six-task job, the primary downstream registry and schema audit
must be extended to register the local jamming and interference caches. This is
an evaluation-only implementation change: it must not alter architecture,
pretraining data, masking, or the frozen-head protocol.

Jamming and interference were formerly deferred OOD-transfer caches. They are
now primary *labelled head-training* tasks for the six-task analysis, while
remaining outside pretraining. Completed four-task pilot artifacts remain
historical and must not be presented as six-task results. DeepMIMO LoS/NLoS is
excluded from new primary runs.

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

The primary Phase 5 comparison has three evaluation tracks but only two
distinct pretraining jobs. P0 is the original LatentWave pretraining, B1 is
the post-hoc early-exit evaluation of that same P0 checkpoint, and P2 is the
separately pretrained proposed method. The first pass is sequential and uses
seed `0` with the shared stochastic `multi_block`/`span_time` masking regime.
Complete P0, its derived B1 heads and evaluations, then P2 and its evaluations
before moving to repetitions. P1 is not a primary method; it is a later
same-scale intermediate-supervision ablation.

Here, “fine-tuning” means training independent heads over a frozen backbone;
no full-backbone downstream fine-tuning occurs in the primary comparison. Each
GPU command requires separate explicit approval under
[`gpu_run_protocol.md`](gpu_run_protocol.md); no seeds, evaluations, or
ablations are queued automatically.
Report pilot point estimates for task performance by exit, selected exit,
FLOPs, batch-1 latency, and relative compute/latency savings. Use synchronized
GPU timing with warm-up iterations and fixed hardware. Do not present
single-seed pilot values as statistically conclusive. After review, repeat the
unchanged P0/B1/P2 protocol for seeds `1` and `2`, then for additional
LatentWave mask regimes, and report mean ± standard deviation. Run P1/B2 only
as a reviewed ablation. Keep WavesFM contextual or deferred unless exact source
and checkpoints become available.

Implementation protocol: [`phase5_main_experiments.md`](phase5_main_experiments.md).
Formal Phase 5 pilot pretraining uses the complete four-cache input and a fixed
60-epoch horizon for P0 and P2; it has no raw-latent-loss early stopping or
pretraining validation split. That loss is non-stationary because both the EMA
target and stochastic prediction masks evolve. The legacy target-L8
source-style P0 scikit-learn probe is kept separate from the context-encoder
early-exit protocol. Formal early-exit runs persist one shared 80/10/10
downstream split manifest, choose frozen-head epochs and exits on downstream
validation only, and do not run the legacy full-dataset downstream probe during
pretraining. For the four-regime masking matrix, P2 uses the matched RTX 5070
Ti performance-safe profile: BF16 mixed precision, fused attention, microbatch
16, accumulation 4, prefetch factor 4, epoch-only telemetry, and
`PYTORCH_ALLOC_CONF=expandable_segments:True`, retaining effective batch size
64. Future P2 runs use the HDF5-balanced loader mode—three workers per cache
with `--hdf5-worker-budget 12` across all four caches—after a one-epoch
throughput diagnostic verifies finite losses. This changes input delivery only;
it does not change samples, masks, model, objective, optimizer, or effective
batch size. The completed region and active time runs retain their recorded
legacy per-cache-worker profiles. The
conservative FP32 microbatch-4/accumulation-16 `--memory-safe` profile remains
available as a separately recorded fallback if a particular mask geometry
causes an OOM. P1 uses a separately approved profile only if its ablation is
later approved.

Review gate: inspect metrics, checkpoints, logs, plots, and failure cases
before spending GPU time on ablations.

Acceptance criteria: reproducible P0/B1/P2 tables; fair early-exit comparisons;
full-depth and early-exit performance; reported pretraining overhead separate
from inference savings; and claims limited to observed evidence.

## Phase 6 — Four-regime masking comparison

The next active study repeats the three-approach comparison under four
masking regimes: frequency (`N_m=6`, the current condition), region
(`N_m=6`), time (`N_m=6`), and random masking at a fixed 75% ratio. The
implementation and approval sequence is authoritative in
[`masking_comparison_plan.md`](masking_comparison_plan.md).

The complete matrix contains four Original LatentWave pretraining jobs and
four Proposed Hierarchical Multi-Scale JEPA pretraining jobs. LatentWave with
Post-hoc Frozen Early Exits is derived from each matching Original LatentWave
checkpoint and therefore adds no distinct pretraining job. Each condition is
completed and reviewed independently before the next condition begins.

The current frequency artifacts may be reused only after metadata confirms
that they exactly match the registered frequency profile. Region, time, and
random profiles require fresh run directories. No GPU job is launched
automatically.

Keep each masking condition's manuscript-facing outputs in its own report
directory: `reports/masking_study/frequency_n6/`,
`reports/masking_study/region_n6/`, `reports/masking_study/time_n6/`, or
`reports/masking_study/random_75/`. The corresponding checkpoints and live
logs remain under `runs/masking_study/<mask-profile>/`.

After the four requested profiles are reviewed, these studies remain
deferred and separately approved:

- Repeat the unchanged masking matrix for seeds `1` and `2`.
- Sweep structured `N_m` over `{2, 3, 4, 5, 6}`.

- Sweep `lambda_H` over `{0.25, 0.5, 1.0, 2.0}`.
- Test optional coarse L8 supervision only as an ablation.
- Test alternate region schedules if P2 is promising.
- Extend held-out transfer beyond the six primary tasks only when a new dataset
  answers a defined question. Jamming and interference are primary labelled
  head-training tasks; they are not pretraining inputs.
- Treat alternate WiFi CSI and 5G CSI files as separate data-expansion or
  preprocessing ablations.
- Add dynamic entropy exits only for classification after fixed exits are complete.

Every ablation must answer a stated research question before GPU allocation.

Implementation protocol: [`phase6_ablations_ood.md`](phase6_ablations_ood.md).
The Phase 6 framework provides lambda and opt-in L8-coarse P2 ablations,
repetition/masking-study rules, six-task downstream registration requirements,
and classification-only dynamic entropy routing. GPU
execution remains approval-gated until the Phase 5 pilot review.

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
downstream set is the six-task suite: RF sensing, jamming, EfficientFi human
activity sensing, interference classification, NR outdoor positioning, and
DeepMIMO beam prediction. Jamming and interference remain labelled head-only
datasets and are never added to pretraining. DeepMIMO LoS/NLoS is retained only
as a historical/legacy evaluation and is no longer a primary task.
Main
The pilot uses seed `0` and the shared stochastic span-time masking regime.
The finalized P0/B1/P2 protocol uses three seeds and reports the original P0
full-depth reference, B1 selected early exits, and P2 full-depth/early-exit
results. Validation determines fixed exits; test data is used only once for
final reporting. P1/B2 is reported separately as an ablation.

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
- The four original pretraining caches and six primary downstream task files
  define the primary experiment; HAS, jamming, and interference caches remain
  one directory above the repository and are not committed.
- Large HDF5 files remain local and ignored by Git.
