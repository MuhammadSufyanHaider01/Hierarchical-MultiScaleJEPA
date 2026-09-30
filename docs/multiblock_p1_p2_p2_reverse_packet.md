# M-block P1/P2/P2-Reverse seed-0 packet

This is the exact, deliberately unchained job order for the single-seed
m-block screening packet. Each command is one GPU job and requires separate
approval under [the GPU protocol](gpu_run_protocol.md). Do not add
`--mask-profile`: the P1, P2, and P2-Reverse base configurations already pin
six prediction masks with `multi_block` / `span_time` masking.

```bash
export PYTHONPATH=src
export PYTORCH_ALLOC_CONF=expandable_segments:True
PACKET_RUN_ROOT=runs/masking_study/multi_block_n6/p1_p2_p2_reverse_seed0
PACKET_REPORT_ROOT=reports/masking_study/multi_block_n6/p1_p2_p2_reverse_seed0
```

Before the first approved formal job, confirm that `$PACKET_RUN_ROOT` is
absent or empty and that all four pretraining caches exist. Then run the
mandatory read-only preflight, which includes P2-Reverse:

```bash
.venv/bin/python scripts/gpu_preflight.py
```

## 1. Formal pretraining — one command per method

Run the following in order: P1, P2, then P2-Reverse. Each must finish 60
epochs with finite losses and write `checkpoints/last.ckpt`, `experiment.json`,
and `summary.json` before its downstream work is approved.

```bash
.venv/bin/python scripts/phase5_pretrain.py --method p1 --seed 0 --run-root "$PACKET_RUN_ROOT" --run-kind formal --max-epochs 60 --batch-size 16 --accumulate-grad-batches 4 --num-workers 3 --hdf5-worker-budget 12 --prefetch-factor 4 --epoch-metrics-only --attention-backend fused --precision bf16-mixed --device cuda --performance-safe

.venv/bin/python scripts/phase5_pretrain.py --method p2 --seed 0 --run-root "$PACKET_RUN_ROOT" --run-kind formal --max-epochs 60 --batch-size 16 --accumulate-grad-batches 4 --num-workers 3 --hdf5-worker-budget 12 --prefetch-factor 4 --epoch-metrics-only --attention-backend fused --precision bf16-mixed --device cuda --performance-safe

.venv/bin/python scripts/phase5_pretrain.py --method p2_reverse --seed 0 --run-root "$PACKET_RUN_ROOT" --run-kind formal --max-epochs 60 --batch-size 16 --accumulate-grad-batches 4 --num-workers 3 --hdf5-worker-budget 12 --prefetch-factor 4 --epoch-metrics-only --attention-backend fused --precision bf16-mixed --device cuda --performance-safe
```

## 2. Validation-only frozen-exit selection

Run one selection-only job for every completed checkpoint. These jobs share
one seed-2025 split manifest in `$PACKET_RUN_ROOT` and never read test data.
The default `primary` suite contains all six required tasks.

```bash
.venv/bin/python scripts/phase5_exits.py --checkpoint "$PACKET_RUN_ROOT/p1/seed_000/checkpoints/last.ckpt" --method p1 --pretrain-seed 0 --split-seed 2025 --run-root "$PACKET_RUN_ROOT" --epochs 200 --head-early-stopping-patience 20 --selection-only --device cuda

.venv/bin/python scripts/phase5_exits.py --checkpoint "$PACKET_RUN_ROOT/p2/seed_000/checkpoints/last.ckpt" --method p2 --pretrain-seed 0 --split-seed 2025 --run-root "$PACKET_RUN_ROOT" --epochs 200 --head-early-stopping-patience 20 --selection-only --device cuda

.venv/bin/python scripts/phase5_exits.py --checkpoint "$PACKET_RUN_ROOT/p2_reverse/seed_000/checkpoints/last.ckpt" --method p2_reverse --pretrain-seed 0 --split-seed 2025 --run-root "$PACKET_RUN_ROOT" --epochs 200 --head-early-stopping-patience 20 --selection-only --device cuda
```

## 3. Held-out test evaluation

Only after all three validation selections complete, run the corresponding
test-only jobs:

```bash
.venv/bin/python scripts/phase5_exits.py --checkpoint "$PACKET_RUN_ROOT/p1/seed_000/checkpoints/last.ckpt" --method p1 --pretrain-seed 0 --split-seed 2025 --run-root "$PACKET_RUN_ROOT" --test-only --device cuda

.venv/bin/python scripts/phase5_exits.py --checkpoint "$PACKET_RUN_ROOT/p2/seed_000/checkpoints/last.ckpt" --method p2 --pretrain-seed 0 --split-seed 2025 --run-root "$PACKET_RUN_ROOT" --test-only --device cuda

.venv/bin/python scripts/phase5_exits.py --checkpoint "$PACKET_RUN_ROOT/p2_reverse/seed_000/checkpoints/last.ckpt" --method p2_reverse --pretrain-seed 0 --split-seed 2025 --run-root "$PACKET_RUN_ROOT" --test-only --device cuda
```

## 4. Synchronized inference benchmarks

Each benchmark writes 24 records: six tasks by L2/L4/L6/L8. Use 50 warm-up
iterations and 200 synchronized measurements.

```bash
.venv/bin/python scripts/phase5_benchmark.py --checkpoint "$PACKET_RUN_ROOT/p1/seed_000/checkpoints/last.ckpt" --method p1 --pretrain-seed 0 --run-root "$PACKET_RUN_ROOT" --warmup-iterations 50 --measure-iterations 200 --device cuda

.venv/bin/python scripts/phase5_benchmark.py --checkpoint "$PACKET_RUN_ROOT/p2/seed_000/checkpoints/last.ckpt" --method p2 --pretrain-seed 0 --run-root "$PACKET_RUN_ROOT" --warmup-iterations 50 --measure-iterations 200 --device cuda

.venv/bin/python scripts/phase5_benchmark.py --checkpoint "$PACKET_RUN_ROOT/p2_reverse/seed_000/checkpoints/last.ckpt" --method p2_reverse --pretrain-seed 0 --run-root "$PACKET_RUN_ROOT" --warmup-iterations 50 --measure-iterations 200 --device cuda
```

## 5. Immutable comparison report

This command reads the historical B0/P0 packet without modifying it, refuses
a nonempty new report directory, hashes its reused sources, and records that
the historical split hash cannot be checked against the newly created
seed-2025 split.

```bash
.venv/bin/python scripts/multiblock_comparison_report.py --baseline-report reports/masking_study/multi_block_n6 --run-root "$PACKET_RUN_ROOT" --output-dir "$PACKET_REPORT_ROOT"
```

P2-GR-LP is intentionally excluded from this packet. B0 is the historical
Original LatentWave L8 reference; P0 is its historical post-hoc frozen L2/L4/L6/L8
exit packet. The result is a screening packet, not an inferential comparison.
