# Hierarchical pretraining throughput profile

The active P2 job uses the conservative memory-safe profile (microbatch 4,
gradient accumulation 16, FP32, four workers). The following profile is an
explicit speed experiment for future approved runs; it does not rewrite the
formal reproducibility profile or alter the active process:

| Setting | Memory-safe profile | Throughput profile |
|---|---:|---:|
| Per-device microbatch | 4 | 8 |
| Gradient accumulation | 16 | 8 |
| Effective batch size | 64 | 64 |
| Data-loader workers | 4 | 8 (initial test) |
| Precision | FP32 | FP32 |
| CUDA allocator | expandable segments | expandable segments |

The throughput profile therefore preserves the effective batch size and the
optimizer-step schedule while reducing the number of serialized microbatches
per optimizer step from 16 to 8. `--throughput-safe` validates these settings
before the trainer allocates the model. It is mutually exclusive with
`--memory-safe` and is currently intended for P1/P2 jobs.

## HDF5 worker check

The pretraining cache opens its HDF5 handle lazily inside each worker and uses
read-only SWMR access. This avoids inheriting an already-open handle across a
worker fork. Four caches with eight workers create up to 32 read-only worker
handles, so the first throughput run must be watched for file-descriptor,
storage-I/O, and GPU-utilization regressions. Reduce workers only in a
separately labelled contention diagnostic; do not silently change the
throughput profile.

The initial approved-command template is:

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
The existing `retry_odd_geometry` run is never modified or resumed with a
different microbatch profile.

## Separate mixed-precision speed variant

Mixed precision is not part of the matched FP32 throughput profile and must
not be used for the primary comparison without a protocol amendment. The
runner already supports a separately labelled diagnostic variant, for example:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
  .venv/bin/python scripts/phase5_pretrain.py \
  --method p2 --seed 0 --run-root runs/diagnostics/p2_mixed_precision \
  --run-name fp16_throughput --mask-profile region_n6 \
  --run-kind diagnostic --max-epochs 1 --batch-size 8 \
  --accumulate-grad-batches 8 --num-workers 8 \
  --precision 16-mixed --device cuda
```

Its checkpoint and timing are diagnostic only. It is not combined with the
FP32 formal result, and it requires a separate approval.
