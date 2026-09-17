# Pareto-Knee Early-Exit Ablation

This is a post-hoc, validation-only ablation for a completed frozen-exit run.
It compares the established earliest-within-5%-tolerance policy with a
compute-aware Pareto-knee policy. It does not pretrain a model, fit task
heads, load a checkpoint, or modify any source experiment artifact.

For each task, the selector receives validation metrics at L2/L4/L6/L8 and a
single cost value for each corresponding exit. It orients the metric so larger
is better, removes Pareto-dominated exits, normalizes utility and cost between
the derived frontier endpoints, and selects the positive interior point that
maximizes:

```text
normalized utility - normalized compute
```

When no geometric knee exists, it explicitly falls back to the
best-performance frontier endpoint. Test metrics are read only after the
validation-based selections are complete.

Run the primary FLOPs ablation once matching downstream and benchmark
artifacts exist:

```bash
python scripts/pareto_knee_ablation.py \
  --exit-metrics runs/phase5/p2/seed_000/downstream/exit_metrics.json \
  --benchmarks runs/phase5/p2/seed_000/benchmarks.json \
  --cost flops
```

The script verifies method, seed, dataset suite, checkpoint identity, task
set, and the exact L2/L4/L6/L8 candidate set before writing a separate JSON
artifact under `downstream/ablations/`. FLOPs are the primary cost. The
`latency-median` and `latency-mean` modes are separate sensitivity analyses.
