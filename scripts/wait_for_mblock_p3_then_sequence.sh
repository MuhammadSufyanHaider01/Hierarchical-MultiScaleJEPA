#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRAINER_PID="${1:?pass the PID of the active p3_performance_variant trainer}"
EXPECTED='phase5_pretrain.py --method p3 --seed 0 --run-root runs/masking_study/multi_block_n6 --run-name p3_performance_variant'
LOG="runs/masking_study/multi_block_n6/terminal_logs/p3_sequence_controller.log"
PY="${ROOT%/MultiScaleJEPA-main}/MultiScaleJEPA/.venv/bin/python"
RUN="runs/masking_study/multi_block_n6/p3_performance_variant/seed_000"

mkdir -p "$(dirname "$LOG")"
echo "[controller] waiting for P3 trainer PID $TRAINER_PID" | tee -a "$LOG"
while kill -0 "$TRAINER_PID" 2>/dev/null; do
  CMD="$(tr '\0' ' ' < "/proc/$TRAINER_PID/cmdline" 2>/dev/null || true)"
  [[ "$CMD" == *"$EXPECTED"* ]] || break
  sleep 30
done

echo "[controller] trainer exited; validating the completed 50-epoch result" | tee -a "$LOG"
"$PY" - "$RUN/experiment.json" "$RUN/summary.json" <<'PY' | tee -a "$LOG"
import json, os, sys
experiment = json.load(open(sys.argv[1]))
summary = json.load(open(sys.argv[2]))
runtime = experiment["metadata"]["phase5"]
assert runtime["method"] == "p3", runtime
assert runtime["max_epochs"] == 50, runtime
assert summary["runtime"]["max_epochs"] == 50, summary["runtime"]
assert summary["global_step"] == 11300, summary["global_step"]
checkpoint = summary["last_checkpoint"]
assert os.path.isfile(checkpoint), checkpoint
print(f"[controller] validated 50 epochs / 11,300 steps: {checkpoint}")
PY

echo "[controller] starting the remaining sequential P3 masking workflow" | tee -a "$LOG"
exec bash scripts/run_p3_masking_sequence.sh
