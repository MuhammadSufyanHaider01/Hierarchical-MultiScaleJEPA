#!/usr/bin/env bash
set -Eeuo pipefail

# Serial P3 experiment chain. Each P3 pretraining profile follows its P2
# report-source profile; all four P3 runs use the fixed 50-epoch horizon.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT%/MultiScaleJEPA-main}/MultiScaleJEPA/.venv/bin/python"
cd "$ROOT"

run_logged() {
  local log="$1"
  shift
  mkdir -p "$(dirname "$log")"
  printf '\n[chain] %s\n' "$*" | tee -a "$log"
  "$@" 2>&1 | tee -a "$log"
}

verify_pretrain() {
  local run="$1"
  "$PY" - "$run/experiment.json" "$run/summary.json" <<'PY'
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
print(f"[chain] verified {runtime['mask_profile']['name']}: 50 epochs, 11,300 steps")
PY
}

run_downstream_and_report() {
  local key="$1"
  local ckpt_override="${2:-}"
  local cond="runs/masking_study/$key"
  local report="reports/masking_study/$key"
  local logdir="$cond/terminal_logs"
  local ckpt="${ckpt_override:-$cond/p3/seed_000/checkpoints/last.ckpt}"
  local manifest="$report/manifest.json"
  [[ -s "$ckpt" ]]

  run_logged "$logdir/p3_heads_selection_terminal.log" "$PY" -u scripts/phase5_exits.py \
    --checkpoint "$ckpt" --method p3 --pretrain-seed 0 --run-root "$cond" \
    --dataset-suite primary --selection-only
  run_logged "$logdir/p3_heldout_test_terminal.log" "$PY" -u scripts/phase5_exits.py \
    --checkpoint "$ckpt" --method p3 --pretrain-seed 0 --run-root "$cond" \
    --dataset-suite primary --test-only
  run_logged "$logdir/p3_inference_benchmark_terminal.log" "$PY" -u scripts/phase5_benchmark.py \
    --checkpoint "$ckpt" --method p3 --pretrain-seed 0 --run-root "$cond" \
    --dataset-suite primary --device cuda

  local p1root p2root reverse output label note title
  p1root=$(jq -er '.p1_run_root' "$manifest")
  p2root=$(jq -er '.p2_run_root' "$manifest")
  reverse=$(jq -er '.p2_reverse_import_root' "$manifest")
  output=$(jq -er '.output_dir' "$manifest")
  label=$(jq -er '.masking' "$manifest")
  note=$(jq -r '.provenance_note // empty' "$manifest")
  title=$(jq -r '.report_title // empty' "$manifest")
  local args=(--run-root "$cond" --p2-run-root "$p2root" --p1-run-root "$p1root"
    --p3-run-root "$cond" --p2-reverse-import-root "$reverse" --output-dir "$output"
    --masking-label "$label")
  [[ -n "$note" ]] && args+=(--provenance-note "$note")
  [[ -n "$title" ]] && args+=(--report-title "$title")
  run_logged "$logdir/p3_report_generation.log" "$PY" -u scripts/phase5_primary_report.py "${args[@]}"
  jq -e '.included_inputs.p3_metrics and .included_inputs.p3_benchmark and (.methods|any(contains("P3")))' \
    "$manifest" >/dev/null
  echo "[chain] verified P3 metrics and benchmark in $manifest"
}

pretrain_condition() {
  local key="$1" profile="$2" reference="$3" execution="$4"
  local cond="runs/masking_study/$key"
  local out="$cond/p3/seed_000"
  local log="$cond/terminal_logs/p3_pretrain_terminal.log"
  [[ ! -e "$out" ]] || { echo "[chain] refusing to overwrite $out" >&2; return 1; }
  [[ -s "$reference" ]] || { echo "[chain] missing matched P2 reference: $reference" >&2; return 1; }

  local common=(--method p3 --seed 0 --run-root "$cond" --run-name p3 --mask-profile "$profile"
    --run-kind matched_horizon --max-epochs 50 --matched-reference-checkpoint "$reference"
    --epoch-metrics-only --progress-every-n-batches 25)
  case "$execution" in
    memory_safe)
      run_logged "$log" env PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" -u scripts/phase5_pretrain.py \
        "${common[@]}" --batch-size 4 --accumulate-grad-batches 16 --num-workers 4 \
        --prefetch-factor 2 --attention-backend math --precision 32-true --memory-safe
      ;;
    performance_balanced)
      run_logged "$log" env PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" -u scripts/phase5_pretrain.py \
        "${common[@]}" --batch-size 16 --accumulate-grad-batches 4 --num-workers 3 \
        --hdf5-worker-budget 12 --prefetch-factor 4 --attention-backend fused \
        --precision bf16-mixed --performance-safe
      ;;
    performance_p2_loader)
      run_logged "$log" env PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" -u scripts/phase5_pretrain.py \
        "${common[@]}" --batch-size 16 --accumulate-grad-batches 4 --num-workers 12 \
        --prefetch-factor 4 --attention-backend fused --precision bf16-mixed --performance-safe
      ;;
    *) echo "[chain] unknown execution profile: $execution" >&2; return 2 ;;
  esac
  verify_pretrain "$out"
}

# M-block P3 pretraining was already resumed as `p3_performance_variant`.
# Reuse that run's final checkpoint rather than starting a duplicate pretrain.
# Complete one masking's full packet before advancing to the next condition.
run_downstream_and_report multi_block_n6 \
  runs/masking_study/multi_block_n6/p3_performance_variant/seed_000/checkpoints/last.ckpt

pretrain_condition frequency_n6 frequency_n6 \
  runs/masking_study/frequency_n6/proposed_hierarchical_frequency/seed_000/checkpoints/epoch=epoch=054-step=step=12430.ckpt \
  performance_balanced
run_downstream_and_report frequency_n6

pretrain_condition time_n6 time_n6 \
  runs/masking_study/time_n6/proposed_hierarchical_performance/seed_000/checkpoints/epoch=epoch=054-step=step=12430.ckpt \
  performance_balanced
run_downstream_and_report time_n6

pretrain_condition region_n6 region_n6 \
  runs/masking_study/region_n6/performance_fused/proposed_hierarchical/seed_000/checkpoints/epoch=epoch=044-step=step=10170.ckpt \
  performance_p2_loader
run_downstream_and_report region_n6

echo "[chain] all four P3 masking packets completed and reports regenerated"
