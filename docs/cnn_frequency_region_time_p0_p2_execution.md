# CNN frequency, region, and time P0 vs P2 execution protocol

This runbook produces three separate CNN-only, single-seed comparison packets.

| Profile | Mask geometry | Run root | Report directory |
| --- | --- | --- | --- |
| `frequency_n6` | six span-time blocks: contiguous frequency bands across full time | `runs/masking_study/frequency_n6/` | `reports/cnn_backbone/frequency_n6/cnn_p0_vs_cnn_p2/` |
| `region_n6` | six random contiguous two-dimensional time-frequency blocks | `runs/masking_study/region_n6/` | `reports/cnn_backbone/region_n6/cnn_p0_vs_cnn_p2/` |
| `time_n6` | six span-frequency blocks: contiguous time bands across full frequency | `runs/masking_study/time_n6/` | `reports/cnn_backbone/time_n6/cnn_p0_vs_cnn_p2/` |

`frequency_n6` deliberately repeats the nominal `multi_block_n6` span-time
geometry in a separately trained historical-style packet. It is not
independent mask-family evidence and the generated report retains that caveat.

## Locked settings and order

Set the project interpreter from the repository root (this checkout's local
Python 3.13 environment):

```bash
PY=.venv/bin/python
```

Every pretraining command uses seed `0`, BF16 mixed precision, fused attention,
microbatch `16`, accumulation `4`, three workers per cache (`--hdf5-worker-budget 12`),
prefetch factor `4`, expandable allocator segments, and epoch-only telemetry.
The CNN variant enables activation checkpointing automatically. Submit the six
direct-from-initialization jobs (P0 and P2 for each profile) without staged
prefixes or checkpoint resumes. They may run concurrently as GPU capacity
allows; downstream work waits for all six to finish.

## Pretraining commands

For each literal `PROFILE` (`frequency_n6`, `region_n6`, `time_n6`), submit
these two commands through `scripts/run_logged_job.sh`:

### CNN-P0 direct validation-monitored pretraining

```bash
bash scripts/run_logged_job.sh runs/masking_study/PROFILE/cnn_p0/terminal_logs/pretrain.log \
  env PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" -u scripts/phase5_pretrain.py \
  --method cnn_p0 --seed 0 --mask-profile PROFILE --run-root runs/masking_study/PROFILE \
  --run-name cnn_p0 --run-kind diagnostic --max-epochs 100 \
  --early-stopping --validation-fraction 0.1 \
  --validation-seed 2025 --early-stopping-patience 20 --early-stopping-min-delta 1e-4 \
  --batch-size 16 --accumulate-grad-batches 4 --num-workers 3 --hdf5-worker-budget 12 \
  --prefetch-factor 4 --attention-backend fused --precision bf16-mixed \
  --performance-safe --epoch-metrics-only --progress-every-n-batches 25
```

### CNN-P2 direct validation-monitored pretraining

```bash
bash scripts/run_logged_job.sh runs/masking_study/PROFILE/cnn_p2/terminal_logs/pretrain.log \
  env PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" -u scripts/phase5_pretrain.py \
  --method cnn_p2 --seed 0 --mask-profile PROFILE --run-root runs/masking_study/PROFILE \
  --run-name cnn_p2 --run-kind diagnostic --max-epochs 100 \
  --early-stopping --validation-fraction 0.1 \
  --validation-seed 2025 --early-stopping-patience 20 --early-stopping-min-delta 1e-4 \
  --batch-size 16 --accumulate-grad-batches 4 --num-workers 3 --hdf5-worker-budget 12 \
  --prefetch-factor 4 --attention-backend fused --precision bf16-mixed \
  --performance-safe --epoch-metrics-only --progress-every-n-batches 25
```

Each direct run starts from initialization, has a 100-epoch maximum, and uses
validation-loss early stopping with patience 20 and minimum delta `1e-4`. Use
each run's `summary.json["checkpoint_for_downstream"]`; do not select a checkpoint
ad hoc.

Before downstream work, require every continuation `summary.json` to name a
finite `checkpoint_for_downstream` and compare each profile's P0/P2
`pretrain_split.json` files for equality.

## Downstream and benchmark commands

After all six direct pretraining runs complete, process each `PROFILE` and each `METHOD`
(`cnn_p0`, then `cnn_p2`). Resolve only the selected continuation checkpoint:

```bash
CKPT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_for_downstream"])' \
  runs/masking_study/PROFILE/METHOD/seed_000/summary.json)
```

Run selection, held-out testing, and benchmarking in that order:

```bash
bash scripts/run_logged_job.sh runs/masking_study/PROFILE/terminal_logs/METHOD_heads_selection.log \
  "$PY" -u scripts/phase5_exits.py --checkpoint "$CKPT" --method METHOD \
  --pretrain-seed 0 --run-root runs/masking_study/PROFILE --dataset-suite primary --selection-only

bash scripts/run_logged_job.sh runs/masking_study/PROFILE/terminal_logs/METHOD_heldout.log \
  "$PY" -u scripts/phase5_exits.py --checkpoint "$CKPT" --method METHOD \
  --pretrain-seed 0 --run-root runs/masking_study/PROFILE --dataset-suite primary --test-only

bash scripts/run_logged_job.sh runs/masking_study/PROFILE/terminal_logs/METHOD_benchmark.log \
  "$PY" -u scripts/phase5_benchmark.py --checkpoint "$CKPT" --method METHOD \
  --pretrain-seed 0 --run-root runs/masking_study/PROFILE --dataset-suite primary --device cuda
```

This produces six selection jobs, six held-out test jobs, and six benchmark jobs.
Do not use prefix checkpoints or select exits on the held-out split.

## Per-profile reports

Generate a report only after both method packets contain all six downstream tasks,
every L2/L4/L6/L8 head, a held-out test result, and a benchmark entry.

```bash
"$PY" -u scripts/phase5_primary_report.py \
  --comparison-mode cnn_only --cnn-run-root runs/masking_study/PROFILE \
  --cnn-methods cnn_p0 cnn_p2 --mask-profile PROFILE \
  --output-dir reports/cnn_backbone/PROFILE/cnn_p0_vs_cnn_p2 \
  --masking-label "PROFILE_LABEL" \
  --report-title "CNN PROFILE_TITLE: CNN-P0 vs CNN-P2, single-seed packet" \
  --provenance-note "Both methods start from initialization and use a 100-epoch-maximum validation-monitored run with early stopping."
```

| `PROFILE` | `PROFILE_LABEL` | `PROFILE_TITLE` |
| --- | --- | --- |
| `frequency_n6` | `Frequency masking / span-time, N_m=6` | `frequency masking` |
| `region_n6` | `Region masking / random 2D blocks, N_m=6` | `region masking` |
| `time_n6` | `Time masking / span-frequency, N_m=6` | `time masking` |

The report records the canonical profile checksum, resolved masks, common
continuation execution profile, matching pretraining-validation split hash,
shared downstream split signature, and the frequency repeated-geometry caveat.
