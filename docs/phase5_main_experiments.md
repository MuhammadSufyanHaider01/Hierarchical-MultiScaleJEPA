# Phase 5: 60-epoch P0/B1/P2 pilot experiments

Phase 5 compares the pinned LatentWave baseline with the two hierarchical
extensions on the local NVIDIA GeForce RTX 5070 Ti. Every GPU command remains
subject to the per-run approval rule in
[`gpu_run_protocol.md`](gpu_run_protocol.md). Nothing in this document queues
or starts a job.

## Recovery status

The earlier `runs/phase5_earlystop*` P0 artifacts and the interrupted P1
artifacts are retained for diagnostics only. They are not formal baseline or
proposal results, and no primary P0/B1/P2 comparison may use them.

- The earlier P0 jobs selected very early checkpoints from a raw latent
  validation loss. Their 10% cache split was fixed, but its prediction masks
  were sampled anew and the EMA target encoder changed throughout training.
  That loss is therefore not a stationary model-selection criterion.
- `runs/phase5/p0/seed_000/checkpoints/last.ckpt` is an older interrupted P0
  diagnostic checkpoint and is not part of the 60-epoch pilot.
- The current `runs/phase5_formal/p0/seed_000/` run was stopped after 65
  completed epochs. Its `epoch=epoch=059-step=step=13560.ckpt` file is the
  exact 60-epoch checkpoint to use for the pilot; its later checkpoint is
  preserved but is not the standardized pilot checkpoint.
- The latest P1 batch-4 attempt was interrupted before completion. Preserve
  its files under `runs/phase5_earlystop_b4fp32/` as memory diagnostics; do
  not resume or report it.

## Two P0 evaluations that must remain separate

| Evaluation | Backbone and representation | Split/probe | Purpose |
|---|---|---|---|
| Source-style P0 probe | EMA `target_encoder`, full-input L8, mean pooling | Original in-source scikit-learn 80/20 probe | Check whether a P0 checkpoint follows the pinned LatentWave validation path. |
| Phase 5 early-exit evaluation | Optimized `context_encoder`, full-input L2/L4/L6/L8, frozen mean-pool/LayerNorm/linear heads | Shared persisted 80/10/10 split | Compare B1 and P2 fairly for early exits; evaluate P1/B2 separately as an ablation. |

The pinned `src/pretrain.py` and its original `JEPA._get_validation_tokens_reps`
use the target branch for the first row. The Version 1 early-exit design uses
the context branch for the second row. Neither protocol is a substitute for
the other, so their metrics are stored in different output paths and must not
be put in the same comparison table.

`scripts/p0_source_probe.py` reproduces the source-style L8 path: target
encoder, mean pooled tokens, `StandardScaler + LogisticRegression(C=5,
newton-cg)` over stratified split seeds 0/1/2 for RFS and beam prediction, and
`StandardScaler + LinearRegression` for positioning. It uses the pinned source
evaluation functions rather than the Phase 5 exit heads.

## Authoritative pilot protocol

The first comparison is an end-to-end, single-seed pilot. It is deliberately
sequential: finish and review one method's pretraining, frozen-head evaluation,
and primary benchmark before allocating GPU time to the next method. OOD
transfer and Phase 6 ablations are deferred and are not gates for this
paper-comparable pilot.

| Pilot order | Method | What changes relative to the pinned source | Representation used in the primary early-exit table |
|---:|---|---|---|
| 1 | P0: original LatentWave | Nothing in pretraining. | EMA `target_encoder` at full-depth L8 for the source-faithful reference. |
| 2 | B1: post-hoc early-exit baseline | Reuses the completed P0 checkpoint; no intermediate pretraining loss. | The same P0 `context_encoder` at L2/L4/L6/L8 with frozen task heads. |
| 3 | P2: proposed multi-scale JEPA | Separately trains intermediate mask-relative fine-to-coarse losses plus the original L8 anchor. | The P2 `context_encoder` at L2/L4/L6/L8 with frozen task heads. |

All three pilot tracks use seed `0` and the same **stochastic** `multi_block`,
`span_time` masking regime from `configs/p0.yml`, `configs/p1.yml`, and
`configs/p2.yml` (one context mask and six prediction masks, with the existing
prediction-block scale). “One mask” therefore means one common masking
*configuration*, not one fixed mask sample repeated for every batch. This is
the only meaningful way to retain the pinned masking behavior while isolating
the architectural comparison.

P1/B2, which adds same-scale intermediate JEPA losses, is retained as a
controlled ablation and is not required before the primary P0/B1/P2 pilot.
After the pilot is reviewed, repeat the unchanged P0/B1/P2 protocol across
seeds `1` and `2`, then repeat it for the other LatentWave mask regimes.
Single-seed pilot results are screening evidence only: do not attach error bars
or make final statistical claims from them.

## Formal pretraining rules

- Distinct pretraining methods and pilot seed: P0 and P2; seed `0`. B1 is
  derived from P0 after pretraining. P1/B2 is a deferred controlled ablation.
- Pilot pretraining jobs run the full four original pretraining caches for
  exactly 60 epochs. This is a fixed, matched pilot budget, not early stopping.
  The pinned LatentWave 240-epoch schedule remains an optional separate
  source-faithful reproduction and must not be conflated with this pilot.
- Formal jobs use FP32, deterministic Lightning execution, one GPU,
  `torch.set_float32_matmul_precision("high")`, the original masking,
  optimizer, LR/weight-decay schedules, EMA branch, and L8 anchor.
- P0 uses microbatch 16 and gradient accumulation 4. Its effective batch is
  64. P2 uses microbatch 4 and accumulation 16, also effective batch 64;
  this preserves optimizer-step and scheduler timing while fitting the
  hierarchical branches in the RTX 5070 Ti memory budget.
- P2 formal commands must set
  `PYTORCH_ALLOC_CONF=expandable_segments:True` before Python starts and pass
  `--memory-safe`. The runner verifies batch size 4, accumulation 16, FP32,
  CUDA, and the allocator setting and records it in `experiment.json`.
- Pilot pretraining has no held-out latent validation set and no early
  stopping. Its downstream checkpoint is `last.ckpt` after the fixed horizon.
  The `--early-stopping` switch remains available only with
  `--run-kind diagnostic`; its result directory and metadata explicitly say it
  cannot be used for the main comparison.
- The legacy all-record downstream PCA probe remains disabled during formal
  pretraining. It would touch downstream examples before Phase 5 split-based
  evaluation.

Formal terminal output has one total training-loss line per epoch. P2 lines
also include hierarchy weight, mean anchor loss, and mean L2/L4/L6 auxiliary
losses when those epoch metrics are available. A diagnostic latent-validation
run prints both train and latent validation loss per epoch, but that output is
diagnostic rather than a selection criterion for formal checkpoints. P1/B2
uses the same logging when its ablation is approved.

## Frozen-head, OOD-transfer, and benchmark protocol

The revised primary task suite is RF signal classification, 5G NR positioning,
DeepMIMO 16-beam prediction, and EfficientFi WiFi human activity recognition
from `../has.h5`. Classification reports macro accuracy; positioning reports
mean physical distance error. All four tasks use the shared version-2
stratified 80/10/10 split manifest, with test data read only after head and
exit selection.

The early-exit runner trains independent heads at L2, L4, L6, and L8 over the
full frozen context encoder. RF sensing, DeepMIMO beam, and EfficientFi human
activity splits are stratified; NR outdoor positioning is deterministically
random. Every method reuses the same persisted 80/10/10 manifest.

The head maximum is 200 epochs. Each independent head stops after 20
non-improving epochs on its task-specific validation metric (macro accuracy
for classification; physical mean distance error for position); the best
validation head is saved. This early stopping is valid because the labelled
validation split and metric are stationary. Test data is read only after all
four validation-selected heads and both fixed-exit policies are chosen.

“Fine-tuning” in this pilot means fitting these frozen linear exit heads only;
the pretrained backbone is never updated. The primary paper-comparable packet
uses RF classification, NR positioning, DeepMIMO beam prediction, and
EfficientFi human activity recognition. The EfficientFi cache is preprocessed
CSI with six activity labels and is loaded from `../has.h5`. OOD transfer uses
separately registered labelled datasets and remains a deferred supplementary
study; its partial diagnostic artifacts are excluded from all current
comparisons. LoS/NLoS results from earlier packets are historical and must not
be mixed into the revised four-task tables.

Report each method's L2/L4/L6/L8 test performance, L8 full-depth reference,
best validation-selected exit, earliest exit within 5% of the best validation
metric, FLOPs, batch-1 latency, and relative compute/latency savings. OOD
transfer, if later approved, is reported separately and never mixed into the
paper-comparable table. The pretraining overhead table remains separate from
deployment savings.

## Approved-job sequence

The next GPU job is never implicit. The intended order after this protocol
change is:

1. Use the exact 60-epoch P0 seed-0 checkpoint under
   `runs/phase5_formal/p0/seed_000/`. The current run's
   `epoch=epoch=059-step=step=13560.ckpt` is the standardized pilot checkpoint.
   The historical P0 checkpoint may receive its own source-style probe, but
   remains diagnostic and cannot replace the pilot checkpoint.
2. For the completed P0 checkpoint, run the source-style EMA-target L8 probe,
   primary context-encoder frozen exits, and the primary benchmark. Review the
   complete P0/B1 paper-comparable packet before starting P2.
3. Run P2 seed 0 for the fixed 60-epoch horizon with the memory-safe profile,
   then the same primary exits and primary benchmark.
   Review the three-track pilot comparison.
4. Only then repeat the fixed P0/B1/P2 protocol for seeds 1 and 2 and the
   remaining mask regimes. Run P1/B2 only when a specific ablation question is
   approved.

## Command templates

Run the mandatory preflight successfully before the first approved job after
a reboot, environment change, data change, or CUDA failure:

```bash
PYTHONPATH=src .venv/bin/python scripts/gpu_preflight.py
```

The first diagnostic probe, using the retained historical checkpoint, would be
the following single GPU evaluation job after explicit approval:

```bash
PYTHONPATH=src .venv/bin/python scripts/p0_source_probe.py \
  --checkpoint runs/phase5/p0/seed_000/checkpoints/last.ckpt \
  --config configs/p0.yml --device cuda \
  --output runs/diagnostics/p0_epoch040_source_probe.json
```

A new formal P0 pretraining seed uses a fresh directory and does not pass
`--early-stopping`:

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_pretrain.py \
  --method p0 --seed 0 --run-root runs/phase5_formal \
  --run-kind formal --batch-size 16 --accumulate-grad-batches 4 \
  --precision 32-true --device cuda
```

A formal P2 seed uses the required memory-safe profile:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
  .venv/bin/python scripts/phase5_pretrain.py \
  --method p2 --seed 0 --run-root runs/phase5_formal \
  --run-kind formal --batch-size 4 --accumulate-grad-batches 16 \
  --precision 32-true --device cuda --memory-safe
```

For a completed formal P0 checkpoint, the source-style probe and the early
exit evaluation are separate approved jobs. The latter has context heads and
validation-based head stopping:

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_exits.py \
  --method p0 --pretrain-seed 0 --run-root runs/phase5_formal \
  --checkpoint runs/phase5_formal/p0/seed_000/checkpoints/last.ckpt \
  --epochs 200 --head-early-stopping-patience 20 --device cuda
```

The following OOD frozen-head command is deferred and is not part of the
active paper-comparable sequence. It is retained only as a separately
approvable supplementary study:

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_exits.py \
  --dataset-suite ood --ood-root .. --run-root runs/phase5_pilot_ood \
  --method p0 --pretrain-seed 0 \
  --checkpoint runs/phase5_formal/p0/seed_000/checkpoints/last.ckpt \
  --epochs 200 --head-early-stopping-patience 20 --device cuda
```

Benchmarking and reporting are additional approved jobs, not follow-ons that
may be launched automatically:

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_benchmark.py \
  --method p0 --pretrain-seed 0 --run-root runs/phase5_formal \
  --checkpoint runs/phase5_formal/p0/seed_000/checkpoints/last.ckpt --device cuda

PYTHONPATH=src .venv/bin/python scripts/phase5_report.py \
  --run-root runs/phase5_formal --output-dir runs/phase5_formal
```

For the completed single-seed pilot, use the dedicated primary packet
generator. It writes all-approach figures using the descriptive names
`Original LatentWave`, `LatentWave with Post-hoc Frozen Early Exits`, and
`Proposed Hierarchical Multi-Scale JEPA`, while keeping the source-faithful
full-depth reference visibly distinct from the comparable frozen-exit
protocol. It also writes submission-oriented Markdown/CSV tables plus 300-DPI
PNG and vector PDF figures under `reports/phase5_primary/`:

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_primary_report.py \
  --run-root runs/phase5_formal \
  --source-root runs/phase5_pilot \
  --output-dir reports/phase5_primary
```

All HDF5 inputs and artifacts beneath `runs/` remain untracked. Formal
metadata records configuration, source revision, environment, GPU, effective
batch size, allocator configuration, and checkpoint selection rule.
