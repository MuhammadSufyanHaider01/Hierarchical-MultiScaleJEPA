# Per-run GPU approval protocol

Formal GPU work is intentionally operator-driven. No script in this repository
queues, resumes, or chains pretraining, frozen-head evaluation, benchmarking,
or Phase 6 experiments. Every runner performs one explicitly named job.

Before launching a requested GPU job, record the following in the experiment
discussion or run log:

- method and purpose: P0, B1 post-hoc exits, P2, P1/B2 ablation, an OOD frozen-head transfer job,
  or a Phase 6 study;
- seed, configuration, checkpoint or resume source, and exact command;
- input datasets, expected output directory, and expected GPU work;
- completion criteria and known dependencies or risks.

Launch only after explicit approval of that exact command. Once it finishes,
inspect the resulting logs and artifacts, report the outcome, and wait for the
next approved job. A command may not be broadened into another seed, an exit
evaluation, a benchmark, or an ablation without a separate approval.

## Mandatory launch gate

Run the read-only gate before the first formal job after a reboot, driver
change, environment change, data change, or failed CUDA run:

```bash
PYTHONPATH=src .venv/bin/python scripts/gpu_preflight.py
```

The gate runs `nvidia-smi`, verifies a CUDA matrix multiplication in the
project environment, fully audits all ten primary HDF5 inputs once the
six-task registry update is implemented, and runs two
finite real-data CUDA smoke steps for P0, P1, and P2. It never launches a
formal Phase 5 or Phase 6 run and does not create checkpoints, split
manifests, metrics, or benchmarks.

The gate must succeed on the RTX 5070 Ti before a formal command with
`--device cuda` is approved. Do not replace it with CPU execution or
`--device auto` for formal results.

## Formal-job rules

- Use `--device cuda` explicitly for all approved Phase 5 and Phase 6 jobs.
- Phase 5 pilot P0/P2 pretraining uses the full four-cache input and a fixed
  60-epoch horizon. It does not use latent-loss early stopping. The pinned
  240-epoch source schedule remains a separate optional reproduction.
  `--early-stopping` requires `--run-kind diagnostic` and its artifacts cannot
  be used in the main result table.
- Formal P2 jobs in the primary pilot use microbatch 4, accumulation 16,
  FP32, and `PYTORCH_ALLOC_CONF=expandable_segments:True` with
  `--memory-safe`. The runner verifies and records this profile before
  allocating GPU memory. A separately approved `--throughput-safe` profile
  tests microbatch 8, accumulation 8, effective batch 64, and at least eight
  workers in FP32; it must not be substituted silently into an existing run.
- The throughput profile's eight-worker setting is an initial HDF5 contention
  test. The pretraining cache opens one read-only HDF5 handle lazily per
  worker, so the first run must be monitored for I/O and file-descriptor
  pressure. The BF16 profile below is a separate, explicitly recorded
  performance variant rather than a replacement for source-faithful FP32.
- `--performance-safe` is the bounded BF16 speed profile: microbatch 16,
  accumulation 4, twelve workers, prefetch factor 4, epoch-only training
  telemetry, and PyTorch fused scaled-dot-product attention. It preserves
  effective batch 64 but is not source-faithful FP32. Fused attention computes
  the same attention operation, with only small precision-level differences.
  Record it
  in every compared method's metadata; do not compare a BF16-trained method
  to an FP32-trained method as a final precision-controlled claim.
- Future pretraining commands print a durable `[progress epoch ...]` terminal
  line every 25 completed microbatches, followed by the epoch-loss line. This
  progress output is terminal-only and does not add per-step CSV metrics.
- Do not reuse a nonempty pretraining directory without the matching
  `--resume path/to/last.ckpt` argument.
- Use `phase5_exits.py --resume` only to continue the same interrupted
  frozen-head job.
- Preserve generated artifacts under gitignored `runs/`; HDF5 files remain
  local and untracked.
- The historical P0 seed-0 checkpoint under
  `runs/phase5/p0/seed_000/checkpoints/last.ckpt` is diagnostic-only and is
  not resumed into a formal result. It may be used only by an explicitly
  approved source-style target-L8 probe.
- The first official comparison is sequential P0 → derived B1 → P2 and uses
  seed `0` plus the shared stochastic `multi_block`/`span_time` masking
  configuration. Complete P0's source reference, B1's exits, and the primary
  benchmark packet before approving the next distinct pretraining job. OOD
  transfer is deferred and is not a gate for the paper-comparable pilot.
  P1/B2 is a later ablation. Seeds `1`/`2` and additional masking regimes are
  follow-up work after reviewing that pilot.

See [`phase5_main_experiments.md`](phase5_main_experiments.md) and
[`phase6_ablations_ood.md`](phase6_ablations_ood.md) for the per-job command
templates and experiment-specific acceptance criteria.
