# CNN-HierJEPA main-branch execution plan

## Scope and branch status

The CNN-HierJEPA implementation is already present on `main`: `cnn-jepa` is
an ancestor of `main` at commit `336e678`.  No merge commit is required.

This plan covers the first formal CNN comparison only:

- M-block / span-time masking with six prediction masks;
- seed 0;
- the finalized six downstream tasks and their persisted shared split;
- a 60-epoch maximum pretraining horizon;
- CNN-P0, CNN-P1, CNN-P2, and CNN-P2-Reverse.

The existing ViT B0/P0 and P1/P2/P2-Reverse M-block packets remain comparison
references.  Frequency, time, region, and random masking are deferred until
the M-block CNN packet has been completed and reviewed.

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

CNN-P0 is the single-scale architecture baseline: it uses the same masked CNN
context encoder, EMA CNN target encoder, and one original L8 predictor with
the original masked-token MSE loss. It has no auxiliary predictors and no
L2/L4/L6 losses. CNN P1/P2/P2-Reverse objectives use the existing independent L2/L4/L6
auxiliary predictors and the original L8 JEPA anchor.  Intermediate CNN losses
are stage-local: L2 trains the stem plus S1; L4 trains the S1-to-S2 transition
plus S2; L6 trains the S2-to-S3 transition plus S3.  The L8 anchor remains
end-to-end.  Target-encoder parameters remain frozen and are updated only by
EMA.

## Locked execution profile

Every fixed-horizon formal CNN pretraining run must use the same execution profile:

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
| Checkpoints | every five epochs plus `last.ckpt` |

The pretraining process must record the resolved configuration, backbone type,
CNN stage configuration, precision profile, mask profile, seed, and split/data
manifest.  Formal runs use the fixed horizon; latent-loss early stopping is not
used to select a formal checkpoint.

## Current artifact audit

### CNN-P0 — original L8 CNN-JEPA baseline

CNN-P0 is not the paper-faithful ViT LatentWave reference. It isolates the
backbone change by retaining the original single L8 JEPA objective under the
same CNN execution profile as the hierarchical CNN variants. It must be
reported as **CNN-P0 — Original L8 JEPA objective**, never as Original
LatentWave.

| Item | Required protocol |
| --- | --- |
| Pretraining | new 60-epoch M-block run, seed 0, locked CNN execution profile |
| Objective | one final predictor and L8 masked-token MSE only |
| Downstream | frozen context-encoder exits L2/L4/L6/L8 on the six persisted-split tasks |
| Evaluation | one held-out test pass and batch-1 FLOPs/latency benchmark |

### CNN-P1 — same-scale multi-depth JEPA

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

### CNN-P2 — mask-relative fine-to-coarse JEPA

This run is incomplete and cannot be compared or reported.

| Item | State |
| --- | --- |
| Observed terminal progress | epoch 8 completed |
| Durable checkpoint | epoch 4 / step 1,130 and `last.ckpt` |
| Downstream heads, held-out test, benchmark | not run |
| Loss behaviour through epoch 8 | finite and decreasing; not enough evidence to assess convergence |

Resume from the durable epoch-4 checkpoint, preserving the original P2
configuration and run directory.  The repeated epochs after that checkpoint
are expected: progress through epochs 5--8 was not checkpointed.

#### Approved protocol amendment: 100-epoch validation-monitored extension

The requested continuation is intentionally separated from the fixed 60-epoch
formal packet. It resumes the epoch-4 checkpoint into a new run directory,
uses a 100-epoch maximum, and selects the best checkpoint with pretraining
validation loss.

| Setting | Value |
| --- | --- |
| Resume source | `cnn_p2/seed_000/checkpoints/last.ckpt` at epoch 4 / step 1,130 |
| New run name | `cnn_p2_extended100_es` |
| Horizon | 100 epochs maximum, including the restored completed epochs |
| Validation | deterministic 90/10 cache split, seed 2025 |
| Stop monitor | mean pretraining `val_loss`, minimize |
| Patience / minimum improvement | 20 epochs / `1e-4` |
| Scheduler handling | preserve optimizer/EMA/completed step; rebase LR and weight-decay cosine horizons to the new loader's 100-epoch horizon |
| Status | diagnostic extension, not interchangeable with the 60-epoch fixed-horizon P1 comparison |

The extension retains the original CNN-P2 directory untouched. Because
validation removes 10% of each pretraining cache from resumed training and
changes the schedule horizon, it is useful for choosing a CNN-P2 candidate
checkpoint but must be labelled as an extended validation-monitored protocol.
A direct fixed-horizon comparison would require matching P1 and P2-Reverse
extensions.

### CNN-P2-Reverse — coarse-to-fine JEPA

No formal pretraining, downstream, or benchmark artifact exists.  This method
starts as a new 60-epoch run after CNN-P2 is complete.

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

### Gate 1 — preserve and audit CNN-P1

No CNN-P1 pretraining rerun is planned.  Verify that its existing
`exit_metrics.json` has all six tasks, all L2/L4/L6/L8 heads, validation-only
selection metadata, one held-out test result, and the shared split manifest.
Verify that `benchmarks.json` has batch-1 FLOPs and synchronized CUDA latency
for every exit.  If this audit fails, refresh only the missing downstream or
benchmark stage from the saved CNN-P1 checkpoint.

### Gate 2 — finish CNN-P2

1. Resume the epoch-4 CNN-P2 checkpoint through the documented 100-epoch
   validation-monitored extension using a separate output directory.
2. Require finite anchor and L2/L4/L6 auxiliary losses through a short resume
   check before continuing the full fit.
3. Train independent frozen L2/L4/L6/L8 heads on the six tasks; use only
   validation metrics to select each task's best exit.
4. Run the selected saved heads exactly once on the held-out test splits.
5. Measure FLOPs and synchronized batch-1 CUDA latency for all four exits.

### Gate 3 — CNN-P2-Reverse

Repeat the same five operations for a new CNN-P2-Reverse run.  Its only
objective difference is the coarse-to-fine region mapping; all architecture,
data, seed, horizon, masks, loader profile, optimizer, EMA, and downstream
protocol settings remain identical to CNN-P2.

### Gate 4 — unified M-block CNN report

Once all three CNN method packets are complete:

1. Run the report generator with `--cnn-run-root` pointing at the directory
   containing `cnn_p1`, `cnn_p2`, and `cnn_p2_reverse`.
2. Generate downstream tables for every task and L2/L4/L6/L8 exit, as well as
   validation-selected best-exit tables.
3. Generate matching FLOP, latency, relative-compute, and relative-latency
   tables and figures.
4. Preserve the ViT B0/P0 packet as a contextual reference, but label the
   CNN comparison clearly: architecture, S4 upsampling caveat, fixed seed,
   fixed horizon, and exact artifact/checkpoint provenance.
5. Do not make superiority claims from one seed.  Treat the M-block packet as
   a controlled architectural ablation and schedule additional seeds only
   after inspecting the first completed report.

## Completion criteria

The M-block CNN study is complete only when:

- CNN-P1, CNN-P2, and CNN-P2-Reverse each have a 60-epoch checkpoint;
- all three have complete six-task frozen-head selection packets;
- each selected head has one held-out test evaluation;
- every method has L2/L4/L6/L8 FLOPs and synchronized batch-1 latency;
- the report uses only traceable, compatible artifacts; and
- the manifest records all source checkpoints, current code revision, training
  revisions where different, masks, split manifest, and execution profile.
