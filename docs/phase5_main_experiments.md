# Phase 5: formal P0/P1/P2 experiments

Phase 5 compares the pinned LatentWave baseline with the two hierarchical
extensions on the local NVIDIA GeForce RTX 5070 Ti. Every GPU command remains
subject to the per-run approval rule in
[`gpu_run_protocol.md`](gpu_run_protocol.md). Nothing in this document queues
or starts a job.

## Recovery status

The earlier `runs/phase5_earlystop*` P0 artifacts and the interrupted P1
artifacts are retained for diagnostics only. They are not formal baseline or
proposal results, and no P1/P2 comparison may use them.

- The earlier P0 jobs selected very early checkpoints from a raw latent
  validation loss. Their 10% cache split was fixed, but its prediction masks
  were sampled anew and the EMA target encoder changed throughout training.
  That loss is therefore not a stationary model-selection criterion.
- `runs/phase5/p0/seed_000/checkpoints/last.ckpt` is an interrupted P0
  diagnostic checkpoint saved at zero-based epoch 39 (40 completed epochs;
  optimizer step 9,040). It is useful for the source-style probe below, but it
  is not a completed 240-epoch baseline.
- The latest P1 batch-4 attempt was interrupted before completion. Preserve
  its files under `runs/phase5_earlystop_b4fp32/` as memory diagnostics; do
  not resume or report it.

## Two P0 evaluations that must remain separate

| Evaluation | Backbone and representation | Split/probe | Purpose |
|---|---|---|---|
| Source-style P0 probe | EMA `target_encoder`, full-input L8, mean pooling | Original in-source scikit-learn 80/20 probe | Check whether a P0 checkpoint follows the pinned LatentWave validation path. |
| Phase 5 early-exit evaluation | Optimized `context_encoder`, full-input L2/L4/L6/L8, frozen mean-pool/LayerNorm/linear heads | Shared persisted 80/10/10 split | Compare B1, P1/B2, and P2/B3 fairly for early exits. |

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
OOD transfer evaluation, and benchmark before allocating GPU time to the next
method. This makes an implementation or protocol failure visible before it can
propagate into the other two methods.

| Pilot order | Method | What changes relative to the pinned source | Representation used in the primary early-exit table |
|---:|---|---|---|
| 1 | P0 / B1: original LatentWave | Nothing in pretraining. | `context_encoder` at L2/L4/L6/L8; the separate source-faithful target-L8 reference is also recorded. |
| 2 | P1 / B2: intermediate same-scale JEPA | Adds L2/L4/L6 masked latent-prediction losses. | `context_encoder` at L2/L4/L6/L8. |
| 3 | P2 / B3: proposed multi-scale JEPA | Replaces P1's intermediate targets with mask-relative fine-to-coarse targets. | `context_encoder` at L2/L4/L6/L8. |

All three pilot runs use seed `0` and the same **stochastic** `multi_block`,
`span_time` masking regime from `configs/p0.yml`, `configs/p1.yml`, and
`configs/p2.yml` (one context mask and six prediction masks, with the existing
prediction-block scale). “One mask” therefore means one common masking
*configuration*, not one fixed mask sample repeated for every batch. This is
the only meaningful way to retain the pinned masking behavior while isolating
the architectural comparison.

After the pilot is reviewed, repeat the unchanged protocol across seeds `1`
and `2`, then repeat it for the other LatentWave mask regimes. Single-seed
pilot results are screening evidence only: do not attach error bars or make
final statistical claims from them.

## Formal pretraining rules

- Methods and pilot seed: P0/B1, P1/B2, and P2/B3; seed `0`.
- Formal jobs run the full four original pretraining caches for exactly 240
  epochs. This is the pinned source configuration, not an upper bound.
- Formal jobs use FP32, deterministic Lightning execution, one GPU,
  `torch.set_float32_matmul_precision("high")`, the original masking,
  optimizer, LR/weight-decay schedules, EMA branch, and L8 anchor.
- P0 uses microbatch 16 and gradient accumulation 4. Its effective batch is
  64. P1/P2 use microbatch 4 and accumulation 16, also effective batch 64;
  this preserves optimizer-step and scheduler timing while fitting the
  hierarchical branches in the RTX 5070 Ti memory budget.
- P1/P2 formal commands must set
  `PYTORCH_ALLOC_CONF=expandable_segments:True` before Python starts and pass
  `--memory-safe`. The runner verifies batch size 4, accumulation 16, FP32,
  CUDA, and the allocator setting and records it in `experiment.json`.
- Formal pretraining has no held-out latent validation set and no early
  stopping. Its downstream checkpoint is `last.ckpt` after the fixed horizon.
  The `--early-stopping` switch remains available only with
  `--run-kind diagnostic`; its result directory and metadata explicitly say it
  cannot be used for the main comparison.
- The legacy all-record downstream PCA probe remains disabled during formal
  pretraining. It would touch downstream examples before Phase 5 split-based
  evaluation.

Formal terminal output has one total training-loss line per epoch. P1/P2 lines
also include hierarchy weight, mean anchor loss, and mean L2/L4/L6 auxiliary
losses when those epoch metrics are available. A diagnostic latent-validation
run prints both train and latent validation loss per epoch, but that output is
diagnostic rather than a selection criterion for formal checkpoints.

## Frozen-head, OOD-transfer, and benchmark protocol

The early-exit runner trains independent heads at L2, L4, L6, and L8 over the
full frozen context encoder. RF sensing, DeepMIMO beam, and LoS/NLoS splits
are stratified; NR outdoor positioning is deterministically random. Every
method reuses the same persisted 80/10/10 manifest.

The head maximum is 200 epochs. Each independent head stops after 20
non-improving epochs on its task-specific validation metric (macro accuracy
for classification; physical mean distance error for position); the best
validation head is saved. This early stopping is valid because the labelled
validation split and metric are stationary. Test data is read only after all
four validation-selected heads and both fixed-exit policies are chosen.

“Fine-tuning” in this pilot means fitting these frozen linear exit heads only;
the pretrained backbone is never updated. For every completed method, first
evaluate the primary downstream suite, then fit the same frozen heads on each
OOD task's own labelled train/validation split and test it once. That is an
OOD transfer evaluation, not zero-shot inference: the OOD datasets have
different label spaces.

Report each method's L2/L4/L6/L8 test performance, L8 full-depth reference,
best validation-selected exit, earliest exit within 5% of the best validation
metric, FLOPs, batch-1 latency, and relative compute/latency savings. Record
the primary downstream and OOD-transfer results separately. The pretraining
overhead table remains separate from deployment savings.

## Approved-job sequence

The next GPU job is never implicit. The intended order after this protocol
change is:

1. Run fresh fixed-horizon P0 seed 0 under `runs/phase5_formal/`. The
   historical 40-epoch P0 checkpoint may receive its own source-style probe,
   but remains diagnostic and cannot replace this run.
2. For the completed P0 checkpoint, run the source-style EMA-target L8 probe,
   primary context-encoder frozen exits, the primary benchmark, OOD frozen
   exits, and an OOD-shape benchmark when an OOD input shape differs. Review
   the complete P0 packet before starting P1.
3. Run P1 seed 0 with the memory-safe profile, then the same primary exits,
   primary benchmark, OOD frozen exits, and any needed OOD-shape benchmark.
   Review it before starting P2.
4. Run P2 seed 0 with the memory-safe profile, then the same evaluation and
   benchmark packet. Review the three-method pilot comparison.
5. Only then repeat the fixed protocol for seeds 1 and 2 and the remaining
   mask regimes. Ablations follow only when the repeated comparison identifies
   a specific question.

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

A formal P1 or P2 seed uses the required memory-safe profile:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
  .venv/bin/python scripts/phase5_pretrain.py \
  --method p1 --seed 0 --run-root runs/phase5_formal \
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

Run the OOD frozen-head packet before advancing to the next method. It uses
the OOD task's own persisted labelled split and must be separately approved:

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

All HDF5 inputs and artifacts beneath `runs/` remain untracked. Formal
metadata records configuration, source revision, environment, GPU, effective
batch size, allocator configuration, and checkpoint selection rule.
