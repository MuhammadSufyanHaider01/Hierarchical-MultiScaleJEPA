# CNN M-block C0 vs C2 execution protocol

This document defines the first CNN M-block comparison. It includes
**C0 (`cnn_p0`) — Original L8 JEPA Objective** and **C2 (`cnn_p2`) — Mask-relative
Fine-to-coarse JEPA**, alongside the matching-mask ViT P0/P2 baselines and
the derived full-depth B0 reference. The output packet is separate from the ViT report:
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
| Pretraining | Five full-data prefix epochs under the original 60-epoch schedule, then resume with deterministic 90/10 split and a rebased 100-epoch maximum |
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

1. **C2 (`cnn_p2`) pretraining is paused at the user's request.** It reached
   the end of one-based epoch 70 and was interrupted during epoch 71. The
   newest validation-best checkpoint is
   `runs/masking_study/multi_block_n6/cnn_p2_extended100_es/seed_000/checkpoints/best-epoch=epoch=069-val_loss=val_loss=0.466410.ckpt`
   (zero-based checkpoint epoch 69, global step 14,390, validation loss
   0.466410). Use this best checkpoint for the current C2 downstream heads.
   `checkpoints/last.ckpt` is retained at the same epoch/step for possible
   continuation; the 260 batches from the interrupted epoch were not saved.

2. **C2 downstream head selection is running** from that validation-best
   checkpoint. Its log is
   `runs/masking_study/multi_block_n6/terminal_logs/cnn_p2_best_epoch69_heads_selection.log`.
   After selection, run held-out evaluation once, then FLOPs/latency benchmark.

3. **Run C0 (`cnn_p0`) five-epoch full-data prefix.** Preserve the original
   60-epoch scheduler horizon and do not use validation in this prefix.

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

4. **Resume C0 from the prefix with validation early stopping.** Use the same
   90/10 split seed, patience, minimum delta, and runtime profile as the C2
   continuation.

   ```bash
   bash scripts/run_logged_job.sh \
     runs/masking_study/multi_block_n6/cnn_p0/terminal_logs/pretrain.log \
     env PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" -u scripts/phase5_pretrain.py \
       --method cnn_p0 --seed 0 --run-root runs/masking_study/multi_block_n6 \
       --run-name cnn_p0 --run-kind diagnostic --max-epochs 100 \
       --resume runs/masking_study/multi_block_n6/cnn_p0_prefix60/seed_000/checkpoints/last.ckpt \
       --extend-resume-schedule-horizon \
       --early-stopping --validation-fraction 0.1 --validation-seed 2025 \
       --early-stopping-patience 20 --early-stopping-min-delta 1e-4 \
       --batch-size 16 --accumulate-grad-batches 4 --num-workers 3 --hdf5-worker-budget 12 \
       --prefetch-factor 4 --attention-backend fused --precision bf16-mixed \
       --performance-safe --epoch-metrics-only --progress-every-n-batches 25
   ```

5. **Complete C0 downstream stages after its summary.** C2 head selection is
   already using the explicit epoch-69 checkpoint above; only repeat C2
   downstream stages if C2 is later resumed and a newer best checkpoint is
   selected:

   ```bash
   CNN_P0_CKPT=$("$PY" -c 'import json; print(json.load(open("runs/masking_study/multi_block_n6/cnn_p0/seed_000/summary.json"))["checkpoint_for_downstream"])')
   CNN_P2_CKPT=runs/masking_study/multi_block_n6/cnn_p2_extended100_es/seed_000/checkpoints/best-epoch=epoch=069-val_loss=val_loss=0.466410.ckpt
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

6. **Generate the combined ViT/CNN packet only after all four learned-method benchmark files exist.**

   ```bash
   "$PY" -u scripts/phase5_primary_report.py \
     --comparison-mode cnn_only --cnn-run-root runs/masking_study/multi_block_n6 \
     --vit-p0-run-root runs/masking_head_seeds/multi_block_n6 \
     --p2-run-root runs/masking_study/multi_block_n6 \
     --vit-p2-run-root runs/masking_study/multi_block_n6 \
     --cnn-methods cnn_p0 cnn_p2 \
     --output-dir reports/cnn_backbone/multi_block_n6/cnn_p0_vs_cnn_p2 \
     --masking-label "M-block / span-time, N_m=6" \
     --report-title "M-block: ViT B0/P0/P2 vs CNN C0/C2, single-seed packet" \
     --provenance-note "Both CNN methods use a matched five-epoch full-data prefix under the original 60-epoch schedule, then resume into a 100-epoch-maximum validation-monitored continuation with early stopping."
   ```

## Completion checks

- C0 has a completed prefix/continuation with a finite validation-best
  checkpoint; C2's report identifies the paused epoch-69 validation-best
  checkpoint and keeps `last.ckpt` available for resumption.
- Both six-task packets share the persisted downstream split signature.
- Every task has L2/L4/L6/L8 head records, a validation-best exit, and one test result.
- Every task/exit pair has FLOPs and synchronized CUDA mean/p95 latency.
- The report includes ViT B0/P0/P2 and CNN C0/C2, with B0 derived from the same ViT P0 checkpoint at fixed L8.
