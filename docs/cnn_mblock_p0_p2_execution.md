# CNN M-block P0 vs P2 execution protocol

This document defines the first CNN-only M-block comparison. It includes
**CNN-P0 — Original L8 JEPA Objective** and **CNN-P2 — Mask-relative
Fine-to-coarse JEPA** only. The output packet is separate from the ViT report:
`reports/cnn_backbone/multi_block_n6/cnn_p0_vs_cnn_p2/`. This is deliberately
separate from `reports/masking_study/`, which remains reserved for the ViT
masking-study packets.

## Locked protocol

| Setting | Value |
| --- | --- |
| Seed | 0 |
| Pretraining masks | Default M-block/span-time profile with six prediction masks |
| CNN backbone | ConvNeXt V2 stages S1/S2/S3/S4, mapped to L2/L4/L6/L8 |
| Microbatch / accumulation / effective batch | 16 / 4 / 64 |
| Precision and attention | BF16 mixed / fused scaled-dot-product attention |
| HDF5 loading | 12 total workers, 3 per cache, prefetch factor 4 |
| Prefix schedule | Five full-data epochs while retaining a 60-epoch schedule horizon |
| Continuation | 100 epochs maximum; deterministic 90/10 pretraining split, seed 2025 |
| Early stopping | Minimize `val_loss`; patience 20; minimum delta `1e-4` |
| Downstream selection | Frozen context heads; best validation-performing exit, never test-selected |

CNN-P0 has only the original L8 predictor and masked-token MSE. CNN-P2 keeps
the L8 anchor plus L2/L4/L6 mask-relative losses with stage-local CNN routing.

## Ordered jobs

Run all long commands through `scripts/run_logged_job.sh`; its log retains the
carriage-return batch bar and the epoch loss summary. Monitor any active log
with `tail -n 0 -F PATH_TO_LOG`.

From the `MultiScaleJEPA-main` repository root, set:

```bash
PY=../MultiScaleJEPA/.venv/bin/python
```

1. **Finish CNN-P2 continuation already in progress.** Its output is
   `runs/masking_study/multi_block_n6/cnn_p2_extended100_es/seed_000/`.
   Use the `checkpoint_for_downstream` field in its completed `summary.json`.

2. **Run the matched CNN-P0 full-data prefix.** This deliberately saves the
   epoch-4 / step-1,130-style resume checkpoint while retaining the original
   60-epoch scheduler horizon.

   ```bash
   bash scripts/run_logged_job.sh \
     runs/masking_study/multi_block_n6/cnn_p0_prefix60/terminal_logs/pretrain.log \
     env PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" -u scripts/phase5_pretrain.py \
       --method cnn_p0 --seed 0 --run-root runs/masking_study/multi_block_n6 \
       --run-name cnn_p0_prefix60 --run-kind staged_prefix --max-epochs 60 \
       --batch-size 16 --accumulate-grad-batches 4 --num-workers 3 --hdf5-worker-budget 12 \
       --prefetch-factor 4 --attention-backend fused --precision bf16-mixed \
       --performance-safe --epoch-metrics-only --progress-every-n-batches 25
   ```

3. **Resume CNN-P0 into its matched validation-monitored continuation.**

   ```bash
   bash scripts/run_logged_job.sh \
     runs/masking_study/multi_block_n6/cnn_p0/terminal_logs/pretrain.log \
     env PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" -u scripts/phase5_pretrain.py \
       --method cnn_p0 --seed 0 --run-root runs/masking_study/multi_block_n6 \
       --run-name cnn_p0 --run-kind diagnostic --max-epochs 100 \
       --resume runs/masking_study/multi_block_n6/cnn_p0_prefix60/seed_000/checkpoints/last.ckpt \
       --extend-resume-schedule-horizon --early-stopping --validation-fraction 0.1 \
       --validation-seed 2025 --early-stopping-patience 20 --early-stopping-min-delta 1e-4 \
       --batch-size 16 --accumulate-grad-batches 4 --num-workers 3 --hdf5-worker-budget 12 \
       --prefetch-factor 4 --attention-backend fused --precision bf16-mixed \
       --performance-safe --epoch-metrics-only --progress-every-n-batches 25
   ```

4. **For CNN-P0, then CNN-P2, run three downstream stages in order.** Resolve
   the two validation-best pretraining checkpoints after their summaries exist:

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

5. **Generate the CNN-only packet only after both benchmark files exist.**

   ```bash
   "$PY" -u scripts/phase5_primary_report.py \
     --comparison-mode cnn_only --cnn-run-root runs/masking_study/multi_block_n6 \
     --cnn-methods cnn_p0 cnn_p2 \
     --output-dir reports/cnn_backbone/multi_block_n6/cnn_p0_vs_cnn_p2 \
     --masking-label "M-block / span-time, N_m=6" \
     --report-title "CNN M-block: CNN-P0 vs CNN-P2, single-seed packet" \
     --provenance-note "Both methods use a matched five-epoch full-data prefix and a 100-epoch-maximum validation-monitored continuation."
   ```

## Completion checks

- Both continuation summaries identify a finite validation-best checkpoint.
- Both six-task packets share the persisted downstream split signature.
- Every task has L2/L4/L6/L8 head records, a validation-best exit, and one test result.
- Every task/exit pair has FLOPs and synchronized CUDA mean/p95 latency.
- The CNN-only report includes only CNN-P0 and CNN-P2 and leaves the ViT M-block report untouched.
