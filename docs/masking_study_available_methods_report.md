# Available-method reports for frequency, time, and region masks

The per-mask P2-Reverse comparison packet is downstream-only. It includes a
method only when its complete downstream results are available: six primary
tasks, all fixed exits, validation-only exit selection, and held-out test
metrics. P1 and the derived post-hoc B1 baseline are excluded. Historical P0
and P2 rows are reused only from a complete report packet for that mask; the
report builder does not infer results from a checkpoint or from a pretraining
log.

After each P2-Reverse pretraining job has completed successfully, run its
selection and test stages using the final checkpoint. Substitute
`MASK` with `frequency_n6`, `time_n6`, or `region_n6`:

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_exits.py \
  --method p2_reverse \
  --checkpoint runs/masking_study/MASK/proposed_hierarchical/seed_000/checkpoints/last.ckpt \
  --pretrain-seed 0 --head-seed 2025 --split-seed 2025 \
  --run-root runs/masking_study/MASK/p2_reverse_downstream \
  --selection-only --device cuda
```

After selection succeeds, run the test-only stage:

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_exits.py \
  --method p2_reverse \
  --checkpoint runs/masking_study/MASK/proposed_hierarchical/seed_000/checkpoints/last.ckpt \
  --pretrain-seed 0 --head-seed 2025 --split-seed 2025 \
  --run-root runs/masking_study/MASK/p2_reverse_downstream \
  --test-only --device cuda
```

Then build the per-mask packet:

```bash
.venv/bin/python scripts/masking_study_available_methods_report.py \
  --mask MASK \
  --p2-reverse-metrics runs/masking_study/MASK/p2_reverse_downstream/p2_reverse/seed_000/downstream/exit_metrics.json
```

Outputs are written under
`reports/masking_study/MASK/available_methods_p2_reverse/`. The manifest lists
the included and omitted approaches and records hashes of the metrics/tables
used, plus the shared FLOPs source and protocol. Each table reports batch-1 FLOPs per example at its task/exit. For time and region, complete historical P0/P2 downstream rows are
available in this checkout. Frequency currently has no checked-in per-mask
report table, so only P2-Reverse will be included unless complete P0/P2
downstream results become available before report generation.

Latency benchmarks are not part of this packet because no matching P2-Reverse
inference benchmark run is planned. FLOPs are counted with
`torch.utils.flop_counter.FlopCounterMode` for supported operations on the
matched frozen encoder/exit-head graph. These operation counts are
hardware-independent at fixed shape and exit, but do not measure latency or
pretraining compute. The existing historical packets are left untouched.
