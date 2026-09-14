# Phase 6: repetitions, masking studies, and ablations

This document serves two purposes: it defines the OOD frozen-head transfer
packet required for each completed single-seed Phase 5 P0/B1/P2 pilot track, and it
defines later repetition/masking/ablation work. OOD transfer does not alter the
primary P0/B1/P2 definition. P1/B2 is the same-scale control ablation. Lambda,
coarse-L8, dynamic-exit, and alternate
masking studies remain deferred until the sequential three-method pilot is
reviewed. Diagnostic early-stopped checkpoints are not eligible. Explicitly
approve that one command under the
[`gpu_run_protocol.md`](gpu_run_protocol.md). No ablation or OOD command is
queued automatically.

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
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
  .venv/bin/python scripts/phase5_pretrain.py \
  --method p2 --run-root runs/phase6/ablations \
  --run-name p2-lambda-0.25 --seed 0 --hierarchy-lambda 0.25 \
  --run-kind formal --batch-size 4 --accumulate-grad-batches 16 \
  --precision 32-true --device cuda --memory-safe
```

Run the same command for the remaining values and three seeds only after the
pilot and its seed/masking repetitions are reviewed. The original P2 is the
`lambda_H=1.0` reference, so an independently regenerated lambda-1 run is
optional and must be labelled a reproducibility check rather than a new method.

The optional L8 coarse ablation is explicitly separate from the original L8
LatentWave anchor. It adds a second, independent L8 region predictor that
predicts one vector for the complete actual prediction-mask block, pooled from
the full-depth EMA target. The total hierarchy term then averages L2/L4/L6 and
this L8-coarse loss, while the original fine-token L8 anchor remains unchanged.
It is enabled only by `--include-l8-coarse` on P2:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
  .venv/bin/python scripts/phase5_pretrain.py \
  --method p2 --run-root runs/phase6/ablations \
  --run-name p2-l8-coarse --seed 0 --include-l8-coarse \
  --run-kind formal --batch-size 4 --accumulate-grad-batches 16 \
  --precision 32-true --device cuda --memory-safe
```

No alternate region schedule is pre-registered: the plan makes it conditional
on a promising P2 main result, and an arbitrary schedule would not answer a
well-defined research question. A reviewed Phase 5 result must identify the
specific failure mode before one is added.

## OOD frozen-exit protocol

For the completed P0 checkpoint, B1 and the original P0 source reference use
the same checkpoint but remain separate evaluation tracks. For the completed
P2 checkpoint, train the same independent frozen context-encoder heads at
L2/L4/L6/L8 before closing the pilot. P1/B2 uses the same OOD packet only if
its ablation is later approved. Validation selects the head epoch and fixed
exit; test data is read only after selection. OOD classification reports macro
accuracy. This is labelled OOD transfer rather than zero-shot evaluation
because every OOD head is fit on that task's labelled training split. Use a
dedicated output root so these artifacts remain separate from the primary
in-distribution comparison.

```bash
PYTHONPATH=src .venv/bin/python scripts/phase5_exits.py \
  --dataset-suite ood --ood-root .. --run-root runs/phase5_pilot_ood \
  --method p2 --pretrain-seed 0 \
  --checkpoint runs/phase5_formal/p2/seed_000/checkpoints/last.ckpt --device cuda

PYTHONPATH=src .venv/bin/python scripts/phase5_benchmark.py \
  --dataset-suite ood --ood-root .. --run-root runs/phase5_pilot_ood \
  --method p2 --pretrain-seed 0 \
  --checkpoint runs/phase5_formal/p2/seed_000/checkpoints/last.ckpt --device cuda

PYTHONPATH=src .venv/bin/python scripts/phase5_report.py \
  --run-root runs/phase5_pilot_ood --output-dir runs/phase5_pilot_ood \
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
  --checkpoint runs/phase5_formal/p2/seed_000/checkpoints/last.ckpt --device cuda
```

Dynamic routing is not defined for NR positioning. It is not a replacement for
the fixed-exit primary analysis, and its selected depth distribution must be
reported alongside its held-out macro accuracy.
