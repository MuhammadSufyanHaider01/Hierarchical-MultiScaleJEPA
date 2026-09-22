# Input-information early-exit inference study

This is an inference-only characterization. It never trains a head, updates a
foundation model, modifies an HDF5 cache, or creates a split manifest.

The source `exit_metrics.json` is authoritative. Before starting, it must
provide the matching P0 or P2 checkpoint, its persisted split manifest, and
saved L2/L4/L6/L8 full-information heads for both `nrpos-outdoor` and
`human-activity`. The runner fails before inference if any prerequisite is
missing; it does not borrow heads from another run.

Run validation selection first. This reads only the validation subsets, uses
the existing 5% relative tolerance rule, and records FLOPs plus synchronized
batch-1 latency for the selected exit at every availability level.

With the project environment activated:

```bash
PYTHONPATH=src python scripts/input_information_exits.py select \
  --source-metrics runs/phase5_formal/p0/seed_000/downstream/exit_metrics.json \
  --output-root runs/input_information/p0_seed000 --study antenna --device cuda
```

The resulting `input_information_exit_metrics.json` contains the requested
antenna conditions `C1/C2/C3/C4` for NR positioning and, when `--study all`
is used, temporal conditions `Tp2/Tp4/Tp8/Tp14` for HAR. Antenna conditions
use ordered channel prefixes; temporal conditions use leading 16-pixel
patch-aligned windows from the existing processed cache, without resizing or
renormalizing.

Only after reviewing the selection artifact, run the held-out test stage. It
evaluates all fixed exits using the saved heads and records the selected exit's
test metric without reopening validation selection.

```bash
PYTHONPATH=src python scripts/input_information_exits.py test \
  --artifact runs/input_information/p0_seed000/input_information_exit_metrics.json \
  --device cuda
```

Render compact tables after test evaluation:

```bash
PYTHONPATH=src python scripts/input_information_report.py \
  --input runs/input_information/p0_seed000/input_information_exit_metrics.json \
  --output-dir reports/input_information/p0_seed000
```
