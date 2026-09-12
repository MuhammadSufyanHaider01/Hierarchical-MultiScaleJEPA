# Phase 6: ablations and held-out OOD evaluation

Phase 6 is implemented as a reproducible framework. It does not alter the
primary P0/P1/P2 definition or consume the active Phase 5 GPU allocation. Run
the scheduled experiments only after reviewing the complete three-seed main
result, as required by the implementation plan.

## Registered OOD tasks

The local parent directory contains three held-out classification datasets. They
are not added to pretraining and no HDF5 file is copied into or tracked by this
repository.

| Task | Source file | Sample key | Label key | Classes |
|---|---|---|---|---:|
| Jamming | `jamming.h5` | `normalized_spectrograms` | `labels` | 6 |
| Interference | `interference.h5` | `cls_spectrograms` | `cls_labels` | 2 |
| Human activity | `has.h5` | `csi` | `label` | 6 |

The audit confirms that the available tensors are finite `[N,C,H,W]` float32
inputs with integer labels. Interference source labels `{1,2}` are remapped in
memory to `{0,1}` and that mapping is preserved in every split manifest; the
original file is never changed. Each OOD task receives its own fixed,
stratified 80/10/10 train/validation/test split with seed 2025.

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_phase6_ood.py \
  --ood-root .. --scan-finite \
  --output runs/phase6/ood_schema_audit.json
```

## Lambda and L8 ablations

The lambda sweep uses P2's canonical target geometry and changes only
`lambda_H`. It runs `0.25`, `0.5`, `1.0`, and `2.0`, retaining the 12-epoch
warm-up, original L8 anchor, masks, optimizer, and data.

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_pretrain.py \
  --method p2 --run-root runs/phase6/ablations \
  --run-name p2-lambda-0.25 --seed 0 --hierarchy-lambda 0.25 --device cuda
```

Run the same command for the remaining values and three seeds. The original
P2 is the `lambda_H=1.0` reference, so an independently regenerated lambda-1
run is optional and must be labelled a reproducibility check rather than a new
method.

The optional L8 coarse ablation is explicitly separate from the original L8
LatentWave anchor. It adds a second, independent L8 region predictor that
predicts one vector for the complete actual prediction-mask block, pooled from
the full-depth EMA target. The total hierarchy term then averages L2/L4/L6 and
this L8-coarse loss, while the original fine-token L8 anchor remains unchanged.
It is enabled only by `--include-l8-coarse` on P2:

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_pretrain.py \
  --method p2 --run-root runs/phase6/ablations \
  --run-name p2-l8-coarse --seed 0 --include-l8-coarse --device cuda
```

No alternate region schedule is pre-registered: the plan makes it conditional
on a promising P2 main result, and an arbitrary schedule would not answer a
well-defined research question. A reviewed Phase 5 result must identify the
specific failure mode before one is added.

## OOD frozen-exit protocol

For a completed P0/P1/P2 checkpoint, train the same independent frozen
context-encoder heads at L2/L4/L6/L8. Validation selects the head epoch and
fixed exit; test data is read only after selection. OOD classification reports
macro accuracy. Use a Phase 6 output root so these artifacts remain separate
from the primary comparison.

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_exits.py \
  --dataset-suite ood --ood-root .. --run-root runs/phase6/ood \
  --method p2 --pretrain-seed 0 \
  --checkpoint runs/phase5/p2/seed_000/checkpoints/last.ckpt --device cuda

PYTHONPATH=src .venv/bin/python scripts/phase5_benchmark.py \
  --dataset-suite ood --ood-root .. --run-root runs/phase6/ood \
  --method p2 --pretrain-seed 0 \
  --checkpoint runs/phase5/p2/seed_000/checkpoints/last.ckpt --device cuda

PYTHONPATH=src .venv/bin/python scripts/phase5_report.py \
  --run-root runs/phase6/ood --output-dir runs/phase6/ood \
  --output-prefix phase6_ood --title "Phase 6 OOD summary" \
  --skip-pretraining-overhead
```

## Dynamic entropy exits

Dynamic exits are a classification-only secondary analysis and require the
fixed heads first. `phase6_dynamic_exits.py` checks for all four saved heads,
calibrates a normalized predictive-entropy threshold on validation only, and
chooses the least-average-depth threshold within the configured relative
macro-accuracy tolerance of L8. Test examples are then evaluated exactly once
using that frozen threshold.

```bash
PYTHONPATH=src .venv/bin/python scripts/phase6_dynamic_exits.py \
  --method p2 --pretrain-seed 0 \
  --checkpoint runs/phase5/p2/seed_000/checkpoints/last.ckpt --device cuda
```

Dynamic routing is not defined for NR positioning. It is not a replacement for
the fixed-exit primary analysis, and its selected depth distribution must be
reported alongside its held-out macro accuracy.
