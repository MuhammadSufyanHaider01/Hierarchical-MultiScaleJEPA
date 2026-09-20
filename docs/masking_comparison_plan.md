# Next-run implementation plan: LatentWave masking comparison

## Objective

After the completed single-seed pilot, the next study isolates masking geometry
as the experimental variable. We will compare the same three approaches under
the four masking regimes listed below, using the same four pretraining caches,
seed `0`, fixed 60-epoch formal pretraining, downstream split, head protocol,
and RTX 5070 Ti timing procedure.

The request lists four regimes, so this is a four-regime study rather than a
three-regime study.

Each condition is evaluated on the six-task primary suite: RF signal
classification and jamming classification (spectral/frequency); EfficientFi
human activity recognition and interference classification
(temporal/cyclostationary); and 5G NR positioning plus DeepMIMO beam prediction
(channel/spatial geometry). HAS, jamming, and interference are local parent
directory caches (`../has.h5`, `../jamming.h5`, and `../interference.h5`), are
used only to train frozen heads, and are never copied into or tracked by Git.
DeepMIMO LoS/NLoS is retained only as a legacy result and is excluded from new
masking-study tables.

Execution prerequisite: before a new six-task frozen-head job is submitted,
register `jamming` and `interference` in the primary downstream suite and
extend its audit manifest. Until that evaluation-only implementation change is
made, the existing runner retains its historical four-task behavior.

The research question is:

> Does the proposed hierarchical latent supervision improve intermediate
> downstream representations consistently across LatentWave masking
> geometries, or is any gain specific to one masking inductive bias?

The original LatentWave paper studies region, frequency, time, and random
masking and varies the number of structured masks. This plan uses the paper's
four geometry names and the requested first-pass settings; exact source
comparability is recorded in each run manifest.

## The three approaches

| Manuscript name | Pretraining | Downstream evaluation |
|---|---|---|
| Original LatentWave | Original LatentWave JEPA objective, EMA target encoder, no hierarchical auxiliary loss | Source-faithful full-depth EMA-target L8 reference |
| LatentWave with Post-hoc Frozen Early Exits | Reuses the matching Original LatentWave checkpoint; no additional pretraining | Frozen context encoder heads at L2/L4/L6/L8 |
| Proposed Hierarchical Multi-Scale JEPA | Separate pretraining with mask-relative L2/L4/L6 losses plus the original L8 anchor | Frozen context encoder heads at L2/L4/L6/L8 |

The second approach is derived from the first approach's checkpoint for each
mask regime. It does not require a second pretraining run. Its downstream
heads are trained independently, and its backbone remains frozen.

## Masking matrix

| ID for files only | Manuscript name | Project configuration | Requested setting |
|---|---|---|---|
| `frequency_n6` | Frequency masking | `multi_block`, `aspect_ratio_type: span_time` | `N_m=6`; each block spans the full time axis and covers contiguous frequency bins |
| `region_n6` | Region masking | `multi_block`, `aspect_ratio_type: random` | `N_m=6`; contiguous 2D time-frequency blocks |
| `time_n6` | Time masking | `multi_block`, `aspect_ratio_type: span_freq` | `N_m=6`; each block spans the full frequency axis and covers contiguous time steps |
| `random_75` | Random masking | `type: random`, `ratios: [0.75, 0.75]` | Fixed 75% patch masking; individual patches sampled uniformly |

The current formal condition uses `span_time` with six prediction blocks. In
the local collator, the patch grid height is frequency and width is time, so
`span_time` means full-time contiguous-frequency masking. It is therefore the
frequency condition, not a generic region condition.

All non-masking settings remain unchanged from the formal pilot: seed `0`,
four original pretraining caches, 60 epochs, effective batch size 64, and the
pinned LatentWave optimizer/scheduler/EMA settings. P0 remains the
source-faithful FP32/math-attention configuration. Every P2 job in this
four-regime masking matrix uses the same successful RTX 5070 Ti
performance-safe profile: BF16 mixed precision, fused attention, microbatch
16, accumulation 4, 12 workers, prefetch factor 4, epoch-only telemetry, and
`PYTORCH_ALLOC_CONF=expandable_segments:True`. This keeps the P2
configuration matched across frequency, region, time, and random masking and
reduces the activation-memory risk observed with larger FP32 hierarchical
batches. The first pass changes only the mask geometry or fixed random ratio.

Before formal execution, audit the current context-mask scale and realized
mask coverage against the paper. The repository's current formal config uses
`enc_mask_scale: [1.0, 1.0]`, while the paper describes a context scale sampled
from `[0.85, 1.0]`. This is a protocol decision, not a silent config change:
the first-pass controlled study preserves the repository formal config, and a
separate paper-faithful profile is created only if this audit is approved.

## Job count and order

The complete four-regime matrix contains:

- 4 Original LatentWave pretraining jobs;
- 4 Proposed Hierarchical Multi-Scale JEPA pretraining jobs;
- 4 derived LatentWave early-exit evaluations from the four Original LatentWave checkpoints;
- 4 Original LatentWave source-style full-depth probes;
- 8 six-task frozen-head downstream evaluations (early-exit baseline and proposed method);
- 8 batch-1 inference benchmarks; and
- 4 per-mask comparison reports.

The existing frequency results are reused only after an artifact audit confirms
the mask geometry, `N_m=6`, seed, 60-epoch horizon, dataset manifest, and
formal configuration. If they do not match, frequency is rerun into a fresh
mask-study directory.

Run one masking condition completely before beginning the next:

1. Freeze the mask profile and verify its schema, geometry, and expected mask
   coverage without launching formal training.
2. Run the mandatory GPU preflight.
3. Train Original LatentWave for the fixed 60-epoch horizon.
4. Run the source-faithful full-depth EMA-target L8 probe.
5. Train the frozen downstream heads for the early-exit baseline at L2/L4/L6/L8 on all six tasks.
6. Benchmark the early-exit baseline at batch size 1.
7. Train the Proposed Hierarchical Multi-Scale JEPA for the fixed 60-epoch
   horizon with the matched performance-safe P2 profile defined above.
8. Train its frozen downstream heads at L2/L4/L6/L8 on all six tasks.
9. Benchmark the proposed approach at batch size 1.
10. Generate the per-mask three-approach tables and figures and review them
    before proceeding to the next mask.

The preferred order is frequency (existing reference), region, time, then
random. No second mask regime, seed, OOD job, or ablation is launched
implicitly. Each GPU command remains separately approval-gated.

## Configuration implementation before the first new job

The runner now supports the registered `--mask-profile` argument. The four
profile files live under `configs/masking_study/`; the selected profile is
validated, merged only into the mask section, and recorded with its SHA-256
checksum in `experiment.json` and checkpoint metadata. All four profiles have
passed CPU dry-run validation and the mandatory CUDA/data preflight.

The current pretraining runner selects `configs/p0.yml`, `configs/p1.yml`, or
`configs/p2.yml` internally. Before region, time, or random formal jobs, add a
mask-profile interface that makes the mask change explicit and records it in
metadata. The interface must:

- accept one of the four registered profile names;
- override only the mask section of the selected P0/P2 configuration;
- reject unknown profiles and incompatible hierarchy settings;
- write the resolved mask configuration and a profile checksum to
  `experiment.json` and checkpoint metadata;
- leave the model, optimizer, EMA, scheduler, dataset order, and seed logic
  unchanged; and
- support a dry-run/schema-audit mode that does not allocate formal GPU work.

Create four resolved profile files under `configs/masking_study/` or an
equivalent explicit configuration registry. Do not copy HDF5 files and do not
modify the original `configs/p0.yml` or `configs/p2.yml` in place.

The random profile must use `[0.75, 0.75]`, not the current interval
`[0.7, 0.8]`, because the paper's random comparison is a fixed 75% condition.

## Output layout

Use a fresh, gitignored root so results cannot overwrite the completed pilot:

```text
runs/masking_study/
  frequency_n6/
    original_latentwave/       # full-depth source probe and checkpoint
    early_exit_baseline/       # derived frozen heads and benchmark
    proposed_hierarchical/     # P2 checkpoint, frozen heads, benchmark
    report/
  region_n6/
  time_n6/
  random_75/

reports/masking_study/
  frequency_n6/               # per-mask tables, figures, and manifest
  region_n6/
  time_n6/
  random_75/
```

The exact directory names may follow the runner's existing method layout, but
the mask profile must be unambiguous in every path and metadata record.
All manuscript-facing artifacts are written to the matching
`reports/masking_study/<mask-profile>/` directory; raw checkpoints and logs
remain under `runs/masking_study/<mask-profile>/`.

## Evaluation and reporting protocol

For every mask profile, retain the same outputs used by the current pilot:

- source-style Original LatentWave full-depth L8 metrics;
- all fixed L2/L4/L6/L8 downstream metrics for the two frozen-head tracks;
- validation-selected best exit and earliest exit within 5% of validation best;
- test performance after selection using the shared persisted 80/10/10 split;
- FLOPs, mean/median/p95 batch-1 latency, and savings relative to L8; and
- pretraining loss and overhead as a separate accounting table.

The final masking-study report should contain:

1. one downstream figure containing all three approaches for each mask;
2. one inference-cost figure containing all three approaches for each mask;
3. one cross-mask figure for each downstream task;
4. one cross-mask figure for latency/FLOPs;
5. Markdown and CSV tables with descriptive manuscript names; and
6. a manifest linking every value to its checkpoint, mask profile, seed,
   split manifest, hardware, and protocol.

The Original LatentWave source-style probe remains visibly marked as a
full-depth reference because it uses the pinned EMA-target 80/20 procedure.
It must not be silently merged with the frozen-head 80/10/10 metrics.

## Acceptance gates

### Preparation gate

- Four mask profiles pass configuration validation.
- Frequency, region, and time produce the intended full-axis geometry.
- Random produces exactly 75% masked patches, subject to integer patch-grid
  rounding.
- A synthetic mask audit reports dimensions, target counts, union coverage,
  visible counts, and overlap behavior.
- No formal run starts before the resolved profile is recorded.

### Per-pretraining gate

- Mandatory GPU preflight passes on the RTX 5070 Ti.
- All losses are finite through the fixed horizon.
- The EMA target receives no gradients and updates correctly.
- P2 auxiliary gradients are nonzero and target gradients remain absent.
- Checkpoints contain source revision, resolved mask profile, configuration,
  seed, dataset manifest, environment, and hardware.

### Per-evaluation gate

- The shared split manifest is reused within the mask condition.
- Test data is not used for head or exit selection.
- All six tasks and all four exits are present for both frozen-head tracks.
- FLOPs and latency use synchronized batch-1 CUDA timing with warm-up.
- The report generator produces complete three-approach tables and figures.

## Deferred work

Do not add extra seeds, the full `N_m ∈ {2,3,4,5,6}` sweep, P1 same-scale
supervision, extra transfer datasets, dynamic exits, or alternate datasets until the
four requested `N_m=6`/75% conditions have been reviewed. If the first pass
shows a task-dependent effect, the next justified study is the structured
mask-count sweep from the LatentWave paper.

## Human-in-the-loop protocol

For every mask condition and every GPU command:

1. Present the exact profile, purpose, seed, checkpoint path, command, inputs,
   expected artifacts, and completion criteria.
2. Wait for explicit approval of that specific job.
3. Launch only that job on the RTX 5070 Ti.
4. Verify logs, checkpoint metadata, metrics, and benchmark artifacts.
5. Summarize the result and stop for review before the next job.

No formal training, evaluation, benchmark, resume, or ablation job is queued
automatically.
