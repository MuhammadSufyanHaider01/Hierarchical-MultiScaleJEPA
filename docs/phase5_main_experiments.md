# Phase 5: main P0/P1/P2 experiments

Phase 5 runs the approved main comparison on the local NVIDIA GeForce RTX
5070 Ti. It does not introduce a new model behavior: P0 is the vendored
LatentWave reference, P1 is B2, and P2 is B3. The P0 checkpoint is also the
source for B1, the required post-hoc frozen early-exit baseline.

## Fixed protocol

- Pretraining methods and seeds: P0, P1, and P2; seeds `0`, `1`, and `2`.
- The four original pretraining caches and their original multi-block mask,
  optimizer, scheduler, EMA target branch, and L8 objective remain in use.
- Microbatch size is 16 with accumulation of 4 (effective batch size 64).
  Scheduler lengths and EMA schedules use optimizer steps, not microbatches;
  the target EMA updates only after a context optimizer update. With
  accumulation 1, the pinned P0 update timing is unchanged.
- Formal runs use 240 epochs as an upper bound, FP32 with the original
  entrypoint's `torch.set_float32_matmul_precision("high")` policy,
  deterministic Lightning execution, one GPU, and local CSV logs/checkpoints
  only. New runs use a fixed 10% held-out pretraining split and stop on the
  latent `val_loss` with patience 20 and `min_delta=1e-4`; 240 remains the
  maximum epoch count, not a requirement to train all 240 epochs.
- The legacy all-record downstream PCA probe is disabled during formal
  pretraining. This prevents downstream test records from being touched before
  frozen-head selection.

## Downstream protocol

The source files do not provide a compatible train/validation/test partition.
Phase 5 therefore uses one persisted split manifest per seed: a fixed 80/10/10
partition with seed 2025. RF sensing, DeepMIMO beam, and DeepMIMO LoS/NLoS are
stratified by their scalar labels; NR outdoor positioning is randomly split.
The manifest is local under `runs/phase5/` and is reused by every method and
pretraining seed.

For each checkpoint, independently train the L2, L4, L6, and L8 mean-pool,
LayerNorm, linear heads while the entire trained **context** encoder is frozen.
The best epoch for a head and the exit policy are chosen using validation only.
Only after both fixed-exit policies are chosen are saved heads evaluated on the
test split. The policies are best validation metric and earliest exit within
5% of the best validation metric. Classification uses macro accuracy; position
uses physical mean Euclidean distance after undoing label normalization.

## Per-run commands

Follow the explicit approval procedure in
[`gpu_run_protocol.md`](gpu_run_protocol.md). Run the mandatory preflight
successfully before the first formal job. Each command below is one job; do not
wrap them in a loop or queue multiple seeds without separate approval.

The interrupted fixed-budget P0 seed-0 run is preserved under
`runs/phase5/p0/seed_000/` and is not reused for the early-stopping protocol.
The replacement run starts from scratch under a separate output root:

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_pretrain.py \
  --method p0 --seed 0 --run-root runs/phase5_earlystop \
  --device cuda --early-stopping \
  --validation-fraction 0.1 --validation-seed 2025 \
  --early-stopping-patience 20 --early-stopping-min-delta 1e-4
```

Use this template only for an individually approved new pretraining job. A job
can be resumed with its matching `last.ckpt`; do not reuse a nonempty run
directory without `--resume`.

Each validation epoch prints both total losses in the terminal, for example
`[epoch 012] train_loss=0.412345 val_loss=0.398765`. The CSV logger also
records `train_loss_epoch`, `val_loss`, and per-dataset validation losses.
Frozen-exit evaluation must use the saved `best_checkpoint` from
`summary.json`, not `last.ckpt`, when early stopping is enabled.

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_pretrain.py \
  --method p2 --seed 0 --device cuda
```

For an individually approved completed checkpoint, train frozen exits or
benchmark it with one of these jobs:

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_exits.py \
  --method p2 --pretrain-seed 0 \
  --checkpoint runs/phase5/p2/seed_000/checkpoints/last.ckpt --device cuda

PYTHONPATH=src .venv/bin/python scripts/phase5_benchmark.py \
  --method p2 --pretrain-seed 0 \
  --checkpoint runs/phase5/p2/seed_000/checkpoints/last.ckpt --device cuda

PYTHONPATH=src .venv/bin/python scripts/phase5_report.py
```

`phase5_exits.py --resume` retains completed task records after an
interruption. The report writes JSON, CSV, Markdown, and (when Matplotlib is
available) an earliest-within-tolerance plot under `runs/phase5/`. Matplotlib
3.10.8 is pinned in `uv.lock` and installed in the project environment. The
report flags missing artifacts and labels a single-seed standard deviation as
unavailable; it makes no scientific claim from incomplete runs.

Every pretraining summary records wall time, optimizer steps/second, and peak
allocated GPU memory. The final report aggregates those fields in a separate
pretraining-overhead table; they are never presented as deployment savings.

## Inference measurement

`phase5_benchmark.py` measures the frozen context encoder plus the respective
head at batch size 1 for the native input shape of each downstream task. It
uses 50 warm-up and 200 measured iterations by default, synchronizes CUDA
before and after every measured forward pass, records mean/median/p95 latency,
and counts FLOPs with PyTorch's `FlopCounterMode`. The report expresses L2/L4/
L6 cost and latency savings relative to L8 for the same method, seed, and task.
Pretraining compute and memory are intentionally not represented as inference
savings.

All produced checkpoints, split manifests, logs, metrics, and plots are under
the gitignored `runs/` directory; HDF5 inputs remain local and untracked.
Checkpoint metadata records both the Git revision and a SHA-256 fingerprint of
all non-ignored source files, including any uncommitted Phase 5 files.
