# Region-masking report packet

This directory is the report home for the `region_n6` masking condition:
region masking with `N_m=6`, seed `0`, and the fixed 60-epoch pilot protocol.

Place all manuscript-facing artifacts for this condition here:

- downstream comparison figures and tables for the three approaches;
- inference/FLOPs/latency comparison figures and tables;
- per-exit and selected-exit summaries;
- CSV/Markdown tables and the report manifest; and
- a README recording the checkpoints, split manifest, hardware, and protocol.

The corresponding raw experiment artifacts remain under:

```text
runs/masking_study/region_n6/
```

The report generator should be invoked with this directory as its output:

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_primary_report.py \
  --run-root runs/masking_study/region_n6 \
  --output-dir reports/masking_study/region_n6
```

Do not place HDF5 files, checkpoints, or live training logs in this report
directory. They remain local under the dataset and `runs/` paths.
