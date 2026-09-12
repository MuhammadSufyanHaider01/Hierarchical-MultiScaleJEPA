# Per-run GPU approval protocol

Formal GPU work is intentionally operator-driven. No script in this repository
queues, resumes, or chains pretraining, frozen-head evaluation, benchmarking,
or Phase 6 experiments. Every runner performs one explicitly named job.

Before launching a requested GPU job, record the following in the experiment
discussion or run log:

- method and purpose: P0/B0-B1, P1/B2, P2/B3, or a Phase 6 study;
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
project environment, fully audits the seven primary HDF5 files, and runs two
finite real-data CUDA smoke steps for P0, P1, and P2. It never launches a
formal Phase 5 or Phase 6 run and does not create checkpoints, split
manifests, metrics, or benchmarks.

The gate must succeed on the RTX 5070 Ti before a formal command with
`--device cuda` is approved. Do not replace it with CPU execution or
`--device auto` for formal results.

## Formal-job rules

- Use `--device cuda` explicitly for all approved Phase 5 and Phase 6 jobs.
- New pretraining jobs use the fixed held-out latent validation loss and early
  stopping policy documented in `phase5_main_experiments.md`; the 240-epoch
  value is an upper bound.
- Do not reuse a nonempty pretraining directory without the matching
  `--resume path/to/last.ckpt` argument.
- Use `phase5_exits.py --resume` only to continue the same interrupted
  frozen-head job.
- Preserve generated artifacts under gitignored `runs/`; HDF5 files remain
  local and untracked.
- The interrupted P0 seed-0 run is resumable only from
  `runs/phase5/p0/seed_000/checkpoints/last.ckpt`, recorded at epoch 9 and
  global optimizer step 2,260. It is not resumed until that specific command
  is approved.

See [`phase5_main_experiments.md`](phase5_main_experiments.md) and
[`phase6_ablations_ood.md`](phase6_ablations_ood.md) for the per-job command
templates and experiment-specific acceptance criteria.
