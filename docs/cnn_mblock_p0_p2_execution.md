# CNN M-block C0 vs C2 execution protocol

This document defines the first CNN M-block comparison. It includes
**C0 (`cnn_p0`) — Original L8 JEPA Objective** and **C2 (`cnn_p2`) — Mask-relative
Fine-to-coarse JEPA**, alongside the per-mask ViT P0 baseline and its derived
full-depth B0 reference. The output packet is separate from the ViT report:
`reports/cnn_backbone/multi_block_n6/cnn_p0_vs_cnn_p2/`. This is deliberately
separate from `reports/masking_study/`, which remains reserved for the ViT
masking-study packets.

The human-facing labels C0/C2 distinguish these CNN approaches from the ViT
P0/P2 methods. Internal method keys and artifact paths keep their historical
`cnn_p0`/`cnn_p2` names.

## Locked protocol

| Setting | Value |
| --- | --- |
| Seed | 0 |
| Pretraining masks | Default M-block/span-time profile with six prediction masks |
| CNN backbone | ConvNeXt V2 stages S1/S2/S3/S4, mapped to L2/L4/L6/L8 |
| Microbatch / accumulation / effective batch | 16 / 4 / 64 |
| Precision and attention | BF16 mixed / fused scaled-dot-product attention |
| HDF5 loading | 12 total workers, 3 per cache, prefetch factor 4 |
| Pretraining | Direct from initialization; 100 epochs maximum; deterministic 90/10 pretraining split, seed 2025 |
| Early stopping | Minimize `val_loss`; patience 20; minimum delta `1e-4` |
| Downstream selection | Frozen context heads; best validation-performing exit, never test-selected |

C0 has only the original L8 predictor and masked-token MSE. C2 keeps
the L8 anchor plus L2/L4/L6 mask-relative losses with stage-local CNN routing.

## Ordered jobs

Run all long commands through `scripts/run_logged_job.sh`; its log retains the
carriage-return batch bar and the epoch loss summary. Monitor any active log
with `tail -n 0 -F PATH_TO_LOG`.

From the `MultiScaleJEPA-main` repository root, set:

```bash
PY=../MultiScaleJEPA/.venv/bin/python
```

1. **Finish C2 (`cnn_p2`) continuation already in progress.** Its output is
   `runs/masking_study/multi_block_n6/cnn_p2_extended100_es/seed_000/`.
   Use the `checkpoint_for_downstream` field in its completed `summary.json`.

2. **Run C0 (`cnn_p0`) directly from initialization with validation early stopping.**
   This is a 100-epoch-maximum run with a 90/10 training/validation split.

   ```bash
   bash scripts/run_logged_job.sh \
     runs/masking_study/multi_block_n6/cnn_p0/terminal_logs/pretrain.log \
     env PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" -u scripts/phase5_pretrain.py \
       --method cnn_p0 --seed 0 --run-root runs/masking_study/multi_block_n6 \
       --run-name cnn_p0 --run-kind diagnostic --max-epochs 100 \
       --early-stopping --validation-fraction 0.1 --validation-seed 2025 \
       --early-stopping-patience 20 --early-stopping-min-delta 1e-4 \
       --batch-size 16 --accumulate-grad-batches 4 --num-workers 3 --hdf5-worker-budget 12 \
       --prefetch-factor 4 --attention-backend fused --precision bf16-mixed \
       --performance-safe --epoch-metrics-only --progress-every-n-batches 25
   ```

3. **Run C2 (`cnn_p2`) directly from initialization with the same validation protocol.**

   ```bash
   bash scripts/run_logged_job.sh \
     runs/masking_study/multi_block_n6/cnn_p2_extended100_es/terminal_logs/pretrain.log \
     env PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" -u scripts/phase5_pretrain.py \
       --method cnn_p2 --seed 0 --run-root runs/masking_study/multi_block_n6 \
       --run-name cnn_p2_extended100_es --run-kind diagnostic --max-epochs 100 \
       --early-stopping --validation-fraction 0.1 \
       --validation-seed 2025 --early-stopping-patience 20 --early-stopping-min-delta 1e-4 \
       --batch-size 16 --accumulate-grad-batches 4 --num-workers 3 --hdf5-worker-budget 12 \
       --prefetch-factor 4 --attention-backend fused --precision bf16-mixed \
       --performance-safe --epoch-metrics-only --progress-every-n-batches 25
   ```

4. **For C0, then C2, run three downstream stages in order.** Resolve
   each validation-best pretraining checkpoint after its summary exists:

   ```bash
   CNN_P0_CKPT=$("$PY" -c 'import json; print(json.load(open("runs/masking_study/multi_block_n6/cnn_p0/seed_000/summary.json"))["checkpoint_for_downstream"])')
   CNN_P2_CKPT=$("$PY" -c 'import json; print(json.load(open("runs/masking_study/multi_block_n6/cnn_p2_extended100_es/seed_000/summary.json"))["checkpoint_for_downstream"])')
   ```

   Selection, held-out evaluation, and benchmarking use the same commands for
   each method/checkpoint pair. Substitute `cnn_p0`/`$CNN_P0_CKPT` first and
   then `cnn_p2`/`$CNN_P2_CKPT`.

   ```bash
   bash scripts/run_logged_job.sh runs/masking_study/multi_block_n6/terminal_logs/METHOD_heads_selection.log \
     "$PY" -u scripts/phase5_exits.py --checkpoint CHECKPOINT --method METHOD \
     --pretrain-seed 0 --run-root runs/masking_study/multi_block_n6 \
     --dataset-suite primary --selection-only

   bash scripts/run_logged_job.sh runs/masking_study/multi_block_n6/terminal_logs/METHOD_heldout.log \
     "$PY" -u scripts/phase5_exits.py --checkpoint CHECKPOINT --method METHOD \
     --pretrain-seed 0 --run-root runs/masking_study/multi_block_n6 \
     --dataset-suite primary --test-only

   bash scripts/run_logged_job.sh runs/masking_study/multi_block_n6/terminal_logs/METHOD_benchmark.log \
     "$PY" -u scripts/phase5_benchmark.py --checkpoint CHECKPOINT --method METHOD \
     --pretrain-seed 0 --run-root runs/masking_study/multi_block_n6 \
     --dataset-suite primary --device cuda
   ```

   The P2 downstream packet is written below `cnn_p2/seed_000/` for the report
   interface, but its saved `exit_metrics.json` identifies the actual
   `cnn_p2_extended100_es` checkpoint used to create the heads.

5. **Generate the combined ViT-baseline/CNN packet only after both CNN benchmark files exist.**

   ```bash
   "$PY" -u scripts/phase5_primary_report.py \
     --comparison-mode cnn_only --cnn-run-root runs/masking_study/multi_block_n6 \
     --vit-p0-run-root runs/masking_head_seeds/multi_block_n6 \
     --cnn-methods cnn_p0 cnn_p2 \
     --output-dir reports/cnn_backbone/multi_block_n6/cnn_p0_vs_cnn_p2 \
     --masking-label "M-block / span-time, N_m=6" \
     --report-title "M-block: ViT B0/P0 vs CNN C0/C2, single-seed packet" \
     --provenance-note "Both CNN methods start from initialization and use a 100-epoch-maximum validation-monitored run with early stopping."
   ```

## Completion checks

- Both continuation summaries identify a finite validation-best checkpoint.
- Both six-task packets share the persisted downstream split signature.
- Every task has L2/L4/L6/L8 head records, a validation-best exit, and one test result.
- Every task/exit pair has FLOPs and synchronized CUDA mean/p95 latency.
- The report includes ViT B0/P0 and CNN C0/C2, with B0 derived from the same ViT P0 checkpoint at fixed L8.
