# P1 and Reverse-P2 hierarchy ablation on ARC

This adds two seed-0, 60-epoch pretraining runs to the existing
`multi_block_n6` comparison. It reuses the checked-in P0/P2 downstream results;
it does not retrain them or run any new inference benchmark.

## Matched methods

- P1: same-scale auxiliary losses at L2/L4/L6 using `configs/p1.yml`.
- P2-Reverse: L2 coarse, L4 local, L6 fine mask-relative losses using
  `configs/p2_reverse.yml`.
- Both use `frequency_n6` (multi-block, span-time, six prediction masks), seed
  0, 60 epochs, effective batch 64, and the performance-safe BF16/fused profile.
- On ARC, prefer one H200 in the `gpu-h100` partition; use one H100 as fallback.

## Environment and preflight

Stage the HDF5 inputs at the configured `data/pretraining/*.h5` and
`data/downstream/*.h5` paths on ARC storage outside Git. On a compute node,
create the locked environment with `uv sync --locked --group dev`, then check
CUDA, BF16, and the data schema. Run `scripts/gpu_preflight.py`; it includes
P0/P1/P2/reverse-P2 smokes and launches no formal experiments.

The commands below are templates, not authorization to submit. Follow
`gpu_run_protocol.md`: review and approve each exact GPU/Slurm command
separately, submit jobs one at a time, and inspect each result before proceeding.

## Pretraining

Run once with `--method p1`, then once with `--method p2_reverse`, keeping the
other arguments fixed:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
  .venv/bin/python scripts/phase5_pretrain.py \
  --method p1 --seed 0 --run-root runs/masking_study/multi_block_n6/hierarchy_ablation \
  --mask-profile frequency_n6 --run-kind formal \
  --batch-size 16 --accumulate-grad-batches 4 \
  --num-workers 3 --hdf5-worker-budget 12 --prefetch-factor 4 \
  --epoch-metrics-only --attention-backend fused \
  --precision bf16-mixed --device cuda --performance-safe
```

The runner records resolved config, mask profile/checksum, seed, epoch count,
batch size/accumulation, and precision settings in the run metadata. Confirm
60 completed epochs and effective batch 64 before beginning downstream work.

## Frozen exits and downstream evaluation

For each method, train fresh frozen heads and choose head checkpoints/exits
using validation only. After selection is complete, run the test-only pass on
those saved heads. Replace `METHOD` with `p1` or `p2_reverse` in each command;
use the matching method checkpoint and do these as separately approved jobs:

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_exits.py \
  --method METHOD --checkpoint runs/masking_study/multi_block_n6/hierarchy_ablation/METHOD/seed_000/checkpoints/last.ckpt \
  --pretrain-seed 0 --head-seed 2025 --split-seed 2025 \
  --run-root runs/masking_study/multi_block_n6/hierarchy_ablation \
  --selection-only --device cuda
```

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_exits.py \
  --method METHOD --checkpoint runs/masking_study/multi_block_n6/hierarchy_ablation/METHOD/seed_000/checkpoints/last.ckpt \
  --pretrain-seed 0 --head-seed 2025 --split-seed 2025 \
  --run-root runs/masking_study/multi_block_n6/hierarchy_ablation \
  --test-only --device cuda
```

The protocol regenerates the seed-2025 80/10/10 split from the current data.
The historical split manifest and checkpoints are absent from this checkout,
so exact split-hash equivalence with P0/P2 cannot be verified. Preserve that
limitation alongside any performance comparison.

## Downstream-only report update

Only after both methods have all six tasks and four exits with test metrics,
run the updater:

```bash
.venv/bin/python scripts/hierarchy_ablation_report.py \
  --p1-metrics runs/masking_study/multi_block_n6/hierarchy_ablation/p1/seed_000/downstream/exit_metrics.json \
  --p2-reverse-metrics runs/masking_study/multi_block_n6/hierarchy_ablation/p2_reverse/seed_000/downstream/exit_metrics.json
```

It updates only the downstream tables and figures in `reports/masking_study/multi_block_n6/`,
records the split caveat, and verifies via hashes that inference report files
remain unchanged.

## Acceptance

Accept when both pretraining runs complete 60 epochs with matched settings,
both downstream runs contain four exits for all six tasks and validation-only
selected test metrics, and the downstream-only report includes P1 and
P2-Reverse. Inference reports remain untouched; no inference run is part of
this ablation.
