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
| Pretraining | C2: five full-data prefix epochs followed by 90/10 validation-monitored continuation; C0: direct deterministic 90/10 validation-monitored training, 100-epoch maximum (the C0 prefix was explicitly skipped) |
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

1. **C2 continuation was stopped on request after epoch 94.** Its best saved
   checkpoint is `runs/masking_study/multi_block_n6/cnn_p2_extended100_es/seed_000/checkpoints/best-epoch=epoch=093-val_loss=val_loss=0.431457.ckpt`.
   Existing C2 heads were trained from the epoch-69 checkpoint
   (`val_loss=0.466410`). Per the latest instruction, reuse those heads and
   their matching checkpoint; do not restart C2 pretraining or retrain its
   heads. Add only the missing held-out test evaluation and inference
   benchmark. The later epoch-93 checkpoint is retained but is not the C2
   representation in this report because no heads were trained on it.

2. **C0 (`cnn_p0`) validation-monitored pretraining is active from scratch.**
   The planned C0 five-epoch full-data prefix was explicitly skipped. Because
   no C0 checkpoint existed, this is not a resume: it uses a deterministic
   90/10 split from its first epoch, up to 100 epochs, with validation-loss
   early stopping (patience 20, min-delta `1e-4`). It retains the matched
   batch/precision/attention/HDF5 loader profile but not the prefix/schedule
   history used by C2; the final report must disclose this difference.

   The sequence controller
   [`run_cnn_mblock_final_sequence.sh`](../scripts/run_cnn_mblock_final_sequence.sh)
   waits for C0 to finish, then trains its heads, performs one held-out
   evaluation, and benchmarks C0. It next evaluates and benchmarks C2 using
   its existing epoch-69 heads/checkpoint without retraining either.

3. **Complete C0 downstream stages and fill C2's missing artifacts.** The
   controller trains C0's independent heads, evaluates each saved
   validation-selected head on test exactly once, and benchmarks every exit.
   For C2, preserve the existing epoch-69 heads and matching checkpoint:

   ```bash
   CNN_P0_CKPT=$("$PY" -c 'import json; print(json.load(open("runs/masking_study/multi_block_n6/cnn_p0/seed_000/summary.json"))["checkpoint_for_downstream"])')
   CNN_P2_CKPT=$("$PY" -c 'import json; print(json.load(open("runs/masking_study/multi_block_n6/cnn_p2/seed_000/downstream/exit_metrics.json"))["checkpoint"])')
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

   The C2 downstream packet is written below `cnn_p2/seed_000/`; its existing
   `exit_metrics.json` identifies the epoch-69 `cnn_p2_extended100_es`
   checkpoint used to create the saved heads. The test-only run adds test
   metrics without training or reselecting heads.

6. **Generate the combined ViT/CNN packet only after all four learned-method benchmark files exist.**

   ```bash
   "$PY" -u scripts/phase5_primary_report.py \
     --comparison-mode cnn_only --cnn-run-root runs/masking_study/multi_block_n6 \
     --vit-p2-run-root runs/masking_study/multi_block_n6 \
     --cnn-methods cnn_p0 cnn_p2 \
     --output-dir reports/cnn_backbone/multi_block_n6/cnn_p0_vs_cnn_p2 \
     --masking-label "M-block / span-time, N_m=6" \
     --report-title "M-block: ViT B0/P0/P2 vs CNN C0/C2, single-seed packet" \
     --provenance-note "C0 was trained directly with a deterministic 90/10 pretraining split and validation-loss early stopping (100-epoch maximum); its planned five-epoch full-data prefix was explicitly skipped. C2 reuses its already-trained frozen heads and matching epoch-69 checkpoint (pretraining validation loss 0.466410); only its missing held-out test evaluation and inference benchmark are added. A later C2 continuation reached a better epoch-93 pretraining checkpoint (validation loss 0.431457), but its heads were not retrained on that checkpoint and it is not the C2 representation evaluated in this report. Training histories differ; all results are seed-0 descriptive comparisons."
   ```

## Completion checks

- C0 has a completed direct 90/10 validation-monitored run with a finite
  validation-best checkpoint; C2's report identifies its saved epoch-93
  validation-best checkpoint and records that the 100-epoch continuation was
  interrupted after epoch 94.
- Both six-task packets share the persisted downstream split signature.
- Every task has L2/L4/L6/L8 head records, a validation-best exit, and one test result.
- Every task/exit pair has FLOPs and synchronized CUDA mean/p95 latency.
- The report includes ViT B0/P0/P2 and CNN C0/C2, with B0 derived from the same ViT P0 checkpoint at fixed L8.
