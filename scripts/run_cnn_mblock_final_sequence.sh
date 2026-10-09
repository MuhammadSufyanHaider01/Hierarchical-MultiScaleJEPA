#!/usr/bin/env bash
# Sequentially finish CNN C0/C2 downstream packets and the M-block report.
# C0 is now trained directly with a validation split (the full-data prefix was
# explicitly skipped); C2 uses its saved best-validation checkpoint.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/../MultiScaleJEPA/.venv/bin/python"
RUN_ROOT="runs/masking_study/multi_block_n6"
LOG_ROOT="${RUN_ROOT}/terminal_logs"
CTRL_LOG="${LOG_ROOT}/cnn_c0_c2_after_c0.log"
WAIT_PID="${1:?Usage: $0 C0_WRAPPER_PID [C2_BEST_CHECKPOINT]}"
C2_CKPT="${2:-}"

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$CTRL_LOG") 2>&1

say() { printf '[sequence %s] %s\n' "$(date '+%F %T %z')" "$*"; }
run_logged() { bash scripts/run_logged_job.sh "$@"; }

say "Waiting for active C0 validation-monitored pretraining wrapper PID ${WAIT_PID}."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done

C0_SUMMARY="${RUN_ROOT}/cnn_p0/seed_000/summary.json"
if [[ ! -s "$C0_SUMMARY" ]]; then
  say "ERROR: C0 training ended without summary.json; refusing downstream jobs."
  exit 2
fi
C0_CKPT="$("$PY" - "$C0_SUMMARY" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
path = Path(payload["checkpoint_for_downstream"])
if not path.is_file():
    raise SystemExit(f"missing C0 validation-best checkpoint: {path}")
print(path)
PY
)"
if [[ -z "$C2_CKPT" ]]; then
  C2_CKPT="$("$PY" - "${RUN_ROOT}/cnn_p2/seed_000/downstream/exit_metrics.json" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
path = Path(payload["checkpoint"])
if not path.is_file():
    raise SystemExit(f"missing checkpoint associated with existing C2 heads: {path}")
print(path)
PY
)"
fi
if [[ ! -s "$C2_CKPT" ]]; then
  say "ERROR: C2 best checkpoint is missing: ${C2_CKPT}"
  exit 2
fi
say "C0 validation-best checkpoint: ${C0_CKPT}"
say "C2 validation-best checkpoint: ${C2_CKPT}"

say "Training C0 frozen heads and selecting exits on validation only."
run_logged "${LOG_ROOT}/cnn_p0_heads_selection.log" "$PY" -u scripts/phase5_exits.py \
  --checkpoint "$C0_CKPT" --method cnn_p0 --pretrain-seed 0 \
  --run-root "$RUN_ROOT" --dataset-suite primary --selection-only
say "Evaluating C0 saved validation-selected heads once on held-out test."
run_logged "${LOG_ROOT}/cnn_p0_heldout.log" "$PY" -u scripts/phase5_exits.py \
  --checkpoint "$C0_CKPT" --method cnn_p0 --pretrain-seed 0 \
  --run-root "$RUN_ROOT" --dataset-suite primary --test-only
say "Benchmarking C0 batch-1 FLOPs and synchronized CUDA latency."
run_logged "${LOG_ROOT}/cnn_p0_benchmark.log" "$PY" -u scripts/phase5_benchmark.py \
  --checkpoint "$C0_CKPT" --method cnn_p0 --pretrain-seed 0 \
  --run-root "$RUN_ROOT" --dataset-suite primary --device cuda

say "Keeping the existing C2 heads; no C2 head training or pretraining will be started."
say "Evaluating those saved C2 heads on held-out test (the current packet has not tested them)."
run_logged "${LOG_ROOT}/cnn_p2_final_heldout.log" "$PY" -u scripts/phase5_exits.py \
  --checkpoint "$C2_CKPT" --method cnn_p2 --pretrain-seed 0 \
  --run-root "$RUN_ROOT" --dataset-suite primary --test-only
say "Benchmarking the existing C2 heads/checkpoint for batch-1 FLOPs and synchronized CUDA latency."
run_logged "${LOG_ROOT}/cnn_p2_final_benchmark.log" "$PY" -u scripts/phase5_benchmark.py \
  --checkpoint "$C2_CKPT" --method cnn_p2 --pretrain-seed 0 \
  --run-root "$RUN_ROOT" --dataset-suite primary --device cuda

say "Generating the unified M-block ViT/CNN comparison report."
"$PY" -u scripts/phase5_primary_report.py \
  --comparison-mode cnn_only --cnn-run-root "$RUN_ROOT" \
  --vit-p0-run-root "$RUN_ROOT" --vit-p2-run-root "$RUN_ROOT" \
  --cnn-methods cnn_p0 cnn_p2 \
  --output-dir reports/cnn_backbone/multi_block_n6/cnn_p0_vs_cnn_p2 \
  --masking-label "M-block / span-time, N_m=6" \
  --report-title "M-block: ViT B0/P0/P2 vs CNN C0/C2, single-seed packet" \
  --provenance-note "C0 was trained directly with a deterministic 90/10 pretraining split and validation-loss early stopping (100-epoch maximum); its planned five-epoch full-data prefix was explicitly skipped. C2 reuses its already-trained frozen heads and matching epoch-69 checkpoint (pretraining validation loss 0.466410); only its missing held-out test evaluation and inference benchmark are added. A later C2 continuation reached a better epoch-93 pretraining checkpoint (validation loss 0.431457), but its heads were not retrained on that checkpoint and it is not the C2 representation evaluated in this report. Training histories differ; all results are seed-0 descriptive comparisons."

say "Sequence finished; report packet is at reports/cnn_backbone/multi_block_n6/cnn_p0_vs_cnn_p2/."
