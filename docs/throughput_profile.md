# Hierarchical pretraining execution profiles

The conservative profile uses microbatch 4, gradient accumulation 16, FP32,
and four workers. The profiles below retain effective batch 64 and the same
optimizer-step schedule. They are explicit configurations, never silent
changes to an existing run:

| Setting | Memory-safe | FP32 throughput | BF16 performance |
|---|---:|---:|---:|
| Per-device microbatch | 4 | 8 | 16 |
| Gradient accumulation | 16 | 8 | 4 |
| Effective batch size | 64 | 64 | 64 |
| Data-loader workers | 4 | 8 | 12 |
| Prefetch factor | 2 | 2 | 4 |
| Logger interval | 1 step | 1 step | 25 steps |
| Precision | FP32 | FP32 | BF16 mixed |
| CUDA allocator | expandable segments | expandable segments | expandable segments |

The FP32 profile reduces serialized microbatches per optimizer step from 16 to
8; the BF16 profile reduces it from 16 to 4. `--throughput-safe` validates the FP32 configuration, and
`--performance-safe` validates the BF16 configuration, including loader and
logger bounds, before model allocation. They are mutually exclusive with
`--memory-safe` and currently apply to P1/P2 only.

## HDF5 worker check

The pretraining cache opens its HDF5 handle lazily inside each worker and uses
read-only SWMR access. This avoids inheriting an already-open handle across a
worker fork. Four caches with eight workers create up to 32 read-only worker
handles, so the first throughput run must be watched for file-descriptor,
storage-I/O, and GPU-utilization regressions. Reduce workers only in a
separately labelled contention diagnostic; do not silently change the
throughput profile.

The FP32 throughput command is:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
  .venv/bin/python scripts/phase5_pretrain.py \
  --method p2 --seed 0 --run-root runs/masking_study/region_n6/throughput \
  --run-name proposed_hierarchical --mask-profile region_n6 \
  --run-kind formal --max-epochs 60 --batch-size 8 \
  --accumulate-grad-batches 8 --num-workers 8 --checkpoint-every 5 \
  --precision 32-true --device cuda --throughput-safe
```

This is a new run directory and must receive its own per-run GPU approval.
No existing run is resumed with a different microbatch or precision profile.

## BF16 performance profile

The RTX 5070 Ti BF16 profile reduces activation memory and logger overhead
while retaining the same effective batch size. It is explicitly recorded in
`experiment.json`; it is not source-faithful FP32 reproduction. Before using
it in a final P0/P2 comparison, rerun every compared pretraining method under
the same precision profile.

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
  .venv/bin/python scripts/phase5_pretrain.py \
  --method p2 --seed 0 --run-root runs/masking_study/region_n6/performance \
  --run-name proposed_hierarchical --mask-profile region_n6 \
  --run-kind formal --max-epochs 60 --batch-size 16 \
  --accumulate-grad-batches 4 --num-workers 12 --prefetch-factor 4 \
  --log-every-n-steps 25 --precision bf16-mixed --device cuda \
  --performance-safe
```

`torch.compile` is intentionally not included: variable mask geometry can
trigger recompilation and raise both latency and memory use. Evaluate it only
as an isolated diagnostic after this bounded profile has a stable epoch-time
measurement.
