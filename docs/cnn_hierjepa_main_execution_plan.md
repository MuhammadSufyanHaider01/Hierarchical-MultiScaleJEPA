# CNN-HierJEPA main-branch execution plan

## Scope and branch status

## CNN naming convention

In human-facing descriptions, **C0** denotes CNN method `cnn_p0` and **C2**
denotes CNN method `cnn_p2`. ViT P0/P2 remain distinct approaches and keep
their P0/P2 names. Executable method keys, run directories, and report paths
retain `cnn_p0`/`cnn_p2` for compatibility.

## Current approved M-block packet

The immediate CNN report compares **C0** and **C2** with the matching-mask
ViT **P0** and **P2** baselines and the derived full-depth **B0** reference. B0 is the
fixed L8 result from the same ViT P0 checkpoint, not separate pretraining.
CNN-P1 and CNN-P2-Reverse rows are not included. The original plan used a
five-epoch full-data prefix for both methods, followed by a deterministic
90/10 validation-monitored continuation. Per the latest instruction, C0's
prefix was skipped and its run starts directly from scratch with a 100-epoch
maximum and early stopping. C2 was stopped after epoch 94; its latest
validation-best checkpoint is epoch 93, while the existing downstream heads
remain paired with the older epoch-69 checkpoint. The final packet reuses those
heads and adds only C2 held-out testing and benchmarking. These schedule and
checkpoint differences must be disclosed. The operational commands, artifact
paths, and gates are in
[`cnn_mblock_p0_p2_execution.md`](cnn_mblock_p0_p2_execution.md).

The CNN-HierJEPA implementation is already present on `main`: `cnn-jepa` is
an ancestor of `main` at commit `336e678`.  No merge commit is required.

This plan covers the first formal CNN comparison only:

- M-block / span-time masking with six prediction masks;
- seed 0;
- the finalized six downstream tasks and their persisted shared split;
- C0: direct deterministic 90/10 validation-monitored training, 100-epoch
  maximum with early stopping; C2: existing staged run, stopped after epoch
  94, using the existing epoch-69 downstream heads for the report;
- C0 and C2.

The existing per-mask ViT P0 packets provide the baselines for the CNN reports;
each report derives B0 from that profile's P0 fixed-L8 result and verifies the
shared downstream split. The executable follow-on protocol for separate C0/C2
frequency, region, and time packets is maintained in
[`cnn_frequency_region_time_p0_p2_execution.md`](cnn_frequency_region_time_p0_p2_execution.md).
Random masking remains deferred.

## Implemented architecture contract

All CNN methods use the native PyTorch ConvNeXt V2 hierarchical encoder at
`src/models/convnext_v2_hier.py`:

| Stage | Logical exit | Depth | Width | Native grid | Adapter to JEPA grid |
| --- | --- | ---: | ---: | --- | --- |
| S1 | L2 | 2 | 48 | 56x56 | 4x4 average pool to 14x14 |
| S2 | L4 | 2 | 96 | 28x28 | 2x2 average pool to 14x14 |
| S3 | L6 | 6 | 192 | 14x14 | native 14x14 |
| S4 | L8 | 2 | 384 | 7x7 | nearest-neighbour upsample to 14x14 |

Every stage is projected to 256-dimensional tokens.  Physical input channels
share CNN weights and use learned residual mean-based cross-channel fusion
after each stage.  Context masking replaces hidden 16x16 input patches with
zero before the CNN; the EMA target encoder sees unmasked inputs.  S4's
upsampling duplicates 7x7 features on the canonical 14x14 grid, so this
representation is not equivalent to 196 independent native S4 tokens.

C0 is the single-scale architecture baseline: it uses the same masked CNN
context encoder, EMA CNN target encoder, and one original L8 predictor with
the original masked-token MSE loss. It has no auxiliary predictors and no
L2/L4/L6 losses. CNN P1/C2/P2-Reverse objectives use the existing independent
L2/L4/L6 auxiliary predictors and the original L8 JEPA anchor. Intermediate CNN losses
are stage-local: L2 trains the stem plus S1; L4 trains the S1-to-S2 transition
plus S2; L6 trains the S2-to-S3 transition plus S3.  The L8 anchor remains
end-to-end.  Target-encoder parameters remain frozen and are updated only by
EMA.

## Locked execution profile

Every CNN pretraining run in these comparison packets must use the same locked
execution profile:

| Setting | Value |
| --- | --- |
| Microbatch / accumulation / effective batch | 16 / 4 / 64 |
| Precision | BF16 mixed |
| Attention backend | fused scaled-dot-product attention |
| HDF5 workers | 12 total, allocated as 3 per pretraining cache |
| Loader prefetch factor | 4 |
| Activation checkpointing | enabled for CNN blocks |
| Optimizer and schedules | AdamW, existing P1/P2 YAML schedules, EMA 0.996 to 1.0 |
| Hierarchy | lambda_H=1.0, 12-epoch linear warm-up, no L8 coarse auxiliary loss |
| Validation / stopping | deterministic 90/10 split, seed 2025; minimize `val_loss`, patience 20, minimum delta `1e-4` |
| Maximum horizon | 100 epochs |
| Checkpoints | best validation checkpoint plus `last.ckpt` |

The pretraining process must record the resolved configuration, backbone type,
CNN stage configuration, precision profile, mask profile, seed, and split/data
manifest. Both C0 and C2 use matched five-epoch full-data prefixes followed by
validation-monitored continuations. The `checkpoint_for_downstream` value in
each completed summary is the only checkpoint used for downstream work.

## Current artifact audit

### C0 (`cnn_p0`) — original L8 CNN-JEPA baseline

C0 is not the paper-faithful ViT LatentWave reference. It isolates the
backbone change by retaining the original single L8 JEPA objective under the
same CNN execution profile as the hierarchical CNN variants. It must be
reported as **C0 — Original L8 JEPA objective**, never as Original
LatentWave.

| Item | Required protocol |
| --- | --- |
| Pretraining | matched five-epoch full-data prefix, then resumed 100-epoch-maximum validation-monitored continuation, seed 0 |
| Objective | one final predictor and L8 masked-token MSE only |
| Downstream | frozen context-encoder exits L2/L4/L6/L8 on the six persisted-split tasks |
| Evaluation | one held-out test pass and batch-1 FLOPs/latency benchmark |

### CNN-P1 — same-scale multi-depth JEPA (existing, excluded from current packet)

Pretraining completed successfully.

| Item | Evidence |
| --- | --- |
| Horizon | 60 completed epochs, 13,560 optimizer steps |
| Checkpoint | `runs/masking_study/multi_block_n6/cnn_p1/seed_000/checkpoints/last.ckpt` |
| Execution profile | BF16/fused, microbatch 16, accumulation 4, effective batch 64 |
| Downstream packet | six-task validation selection and one held-out test evaluation exist |
| Benchmark packet | batch-1 FLOPs and CUDA latency data exist for all task/exit pairs |

Its pretraining loss fell from 0.9500 at epoch 1 to 0.4277 at epoch 60.  The
anchor loss fell from 0.8759 to 0.1304.  The curve was still declining slowly
at the fixed horizon, rather than showing numerical divergence or an abrupt
failure.  Because pretraining validation was intentionally disabled for the
formal fixed-horizon protocol, this is evidence of optimization convergence,
not evidence of validation-optimality.

One early downstream-head attempt failed on the smaller DeepMIMO input grid.
The current encoder supports valid 32x32 inputs and the final six-task packet
contains saved heads and held-out results for DeepMIMO as well.  Before report
generation, verify the packet paths, checkpoint hash, and split manifest; do
not silently replace its reported results.

### C2 (`cnn_p2`) — mask-relative fine-to-coarse JEPA

The C2 prefix was completed as five full-data epochs under the 60-epoch
schedule. Its validation-monitored continuation was stopped after epoch 94;
the later best checkpoint is retained, but the existing downstream heads are
paired with the earlier epoch-69 checkpoint. The current report will reuse
those heads and fill only their missing test and inference artifacts.

| Item | State |
| --- | --- |
| Stopped state | continuation stopped after epoch 94; no C2 restart is planned for this report |
| Later validation-best checkpoint | epoch index 93, `val_loss=0.431457`; retained but not used by the existing heads |
| Checkpoint paired with existing heads | epoch index 69 / step 14,390, `val_loss=0.466410` |
| Frozen-head selection | completed for six tasks × L2/L4/L6/L8 from the epoch-69 checkpoint; reuse without retraining |
| Held-out test, benchmark | pending; add using the saved epoch-69 heads/checkpoint |
| Loss behaviour | logged losses were finite through the stopped continuation |

The P2 intermediate losses are L2/L4/L6; each is finite-checked on the first
continuation epoch before downstream work. The report labels the approach C2
while preserving `cnn_p2` in artifact paths and provenance.

#### C2 loss and selection protocol

C2 resumed the matched five-epoch full-data prefix into a 100-epoch-maximum
run, with a deterministic 90/10 cache split (seed 2025), rebased scheduler
horizon, and early stopping on mean pretraining `val_loss` (patience 20,
minimum improvement `1e-4`). Its L2/L4/L6 intermediate losses were finite in
the logged epochs. The continuation was stopped after epoch 94. The later
epoch-93 validation-best checkpoint is retained, while existing heads remain
paired with epoch 69; the current report adds test evaluation and benchmarking
for that existing head/checkpoint pair without retraining C2.

### CNN-P2-Reverse — coarse-to-fine JEPA (deferred)

No formal pretraining, downstream, or benchmark artifact exists.  This method
starts as a new 60-epoch run after C2 is complete.

### Non-formal attempts

`failed_attempts/cnn_p1_oom_no_checkpointing` and
`failed_attempts/cnn_p1_adaptive_pool` are diagnostic/failed attempts.  They
are excluded from all comparisons and must never be used as checkpoints or
report inputs.

## Execution sequence

### Gate 0 — reproducibility audit

Before launching a GPU job:

1. Run the CNN architecture, mask-isolation, stage-local-gradient,
   target-no-gradient, checkpoint round-trip, and frozen-head tests.
2. Check CUDA availability and verify no unrelated training process is using
   the GPU.
3. Check that the four local pretraining HDF5 caches and six downstream files
   are available.
4. Confirm that the persisted M-block split manifest is used by all CNN head
   jobs.
5. Record the current `main` revision.  The completed CNN-P1 checkpoint was
   trained at revision `6286f35`; later code changes add deterministic adapter
   pooling and support valid smaller downstream input grids.  These changes do
   not alter the P1 checkpoint weights, but this provenance must appear in the
   final report.

### Gate 1 — preserve CNN-P1 without including it

No CNN-P1 pretraining rerun is planned. Its existing
`exit_metrics.json` has all six tasks, all L2/L4/L6/L8 heads, validation-only
selection metadata, one held-out test result, and the shared split manifest.
Verify that `benchmarks.json` has batch-1 FLOPs and synchronized CUDA latency
for every exit.  If this audit fails, refresh only the missing downstream or
benchmark stage from the saved CNN-P1 checkpoint.

### Gate 2 — complete C0 and finish the existing C2 packet

1. Finish the active C0 validation-monitored training and retain its
   validation-best checkpoint.
2. Train C0 frozen heads, run held-out evaluation once, and benchmark all four
   exits.
3. Reuse the existing C2 frozen heads and matching epoch-69 checkpoint. Add
   its missing held-out test evaluation and benchmark all four exits; do not
   restart C2 pretraining or retrain its heads on epoch 93 for this packet.

### Gate 3 — CNN-P2-Reverse (deferred)

Do not launch CNN-P2-Reverse in the current packet. A later extension may
repeat the same five operations for a new CNN-P2-Reverse run. Its only
objective difference is the coarse-to-fine region mapping; all architecture,
data, seed, horizon, masks, loader profile, optimizer, EMA, and downstream
protocol settings remain identical to C2.

### Gate 4 — C0 vs C2 M-block report

Once C0 and C2 packets are complete:

1. Run the report generator with `--comparison-mode cnn_only --cnn-methods cnn_p0 cnn_p2`,
   `--cnn-run-root` pointing at the M-block run directory, and
   `--vit-p0-run-root runs/masking_head_seeds/multi_block_n6` and
   `--p2-run-root runs/masking_study/multi_block_n6`.
2. Generate downstream tables for every task and L2/L4/L6/L8 exit, as well as
   validation-selected best-exit tables.
3. Generate matching FLOP, latency, relative-compute, and relative-latency
   tables and figures.
4. Include matching M-block ViT P0/P2 baselines and derived fixed-L8 B0
   reference alongside C0/C2. Record architecture, S4 upsampling caveat, fixed
   seed, early-stopping protocol, and exact artifact/checkpoint provenance.
5. Do not make superiority claims from one seed.  Treat the M-block packet as
   a controlled architectural ablation and schedule additional seeds only
   after inspecting the first completed report.

## Completion criteria

The M-block CNN study is complete only when:

- C0 has a completed validation-monitored run and C2's packet records its
  existing epoch-69 head/checkpoint provenance plus the later epoch-93
  validation-best pretraining checkpoint that was not evaluated downstream;
- both have complete six-task frozen-head selection packets;
- each selected head has one held-out test evaluation;
- both methods have L2/L4/L6/L8 FLOPs and synchronized batch-1 latency;
- the report uses only traceable, compatible artifacts; and
- the manifest records all source checkpoints, current code revision, training
  revisions where different, masks, split manifest, and execution profile.
