#!/usr/bin/env bash
# Train and test C0/C2 frozen exit heads for head seeds 0..6.
# Each seed has an isolated run root so head weights and metrics cannot overwrite
# one another. All seeds reuse the canonical M-block downstream split manifest.
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-$repo_root/.venv-synthetic-study/bin/python}"
canonical_split="${SPLIT_MANIFEST:-$repo_root/runs/masking_study/multi_block_n6/splits_seed_2025_primary_v3.json}"
output_root="$repo_root/runs/cnn_head_multiseed/multi_block_n6_cnn_split_v1"
log_root="$output_root/terminal_logs"
c0_checkpoint="$repo_root/runs/masking_study/multi_block_n6/cnn_p0/seed_000/checkpoints/best-epoch=epoch=019-val_loss=val_loss=0.246513.ckpt"
c2_checkpoint="$repo_root/runs/masking_study/multi_block_n6/cnn_p2_extended100_es/seed_000/checkpoints/best-epoch=epoch=069-val_loss=val_loss=0.466410.ckpt"

[[ -x "$python_bin" ]] || { echo "Python executable not found: $python_bin" >&2; exit 2; }
[[ -f "$canonical_split" ]] || {
  echo "Downstream split manifest missing: $canonical_split" >&2
  echo "Set SPLIT_MANIFEST to the persisted M-block split manifest." >&2
  exit 2
}
[[ -f "$c0_checkpoint" ]] || { echo "C0 checkpoint missing: $c0_checkpoint" >&2; exit 2; }
[[ -f "$c2_checkpoint" ]] || { echo "C2 checkpoint missing: $c2_checkpoint" >&2; exit 2; }

split_seed="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("seed", -1))' "$canonical_split")"
[[ "$split_seed" == 2025 ]] || {
  echo "Expected downstream split seed 2025, got $split_seed in $canonical_split" >&2
  exit 2
}

if ! "$python_bin" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "CUDA is unavailable; refusing to launch the seven-seed head study." >&2
  exit 3
fi

mkdir -p "$log_root"

for method in cnn_p0 cnn_p2; do
  if [[ "$method" == cnn_p0 ]]; then checkpoint="$c0_checkpoint"; else checkpoint="$c2_checkpoint"; fi
  for head_seed in 0 1 2 3 4 5 6; do
    seed_name="$(printf '%03d' "$head_seed")"
    seed_root="$output_root/head_seed_$seed_name"
    mkdir -p "$seed_root"
    cp -n "$canonical_split" "$seed_root/splits_seed_2025_primary_v3.json"
    for phase in selection test; do
      log_file="$log_root/${method}_head_seed_${seed_name}_${phase}.log"
      args=(
        --checkpoint "$checkpoint"
        --method "$method"
        --pretrain-seed 0
        --head-seed "$head_seed"
        --split-seed 2025
        --run-root "$seed_root"
        --dataset-suite primary
        --device cuda
        --batch-size 32
        --num-workers 4
      )
      if [[ "$phase" == selection ]]; then args+=(--selection-only); else args+=(--test-only); fi
      bash "$repo_root/scripts/run_logged_job.sh" "$log_file" "$python_bin" -u \
        "$repo_root/scripts/phase5_exits.py" "${args[@]}"
    done
  done
done

echo "Completed C0/C2 head seeds 0..6. Raw packets are under $output_root/head_seed_*/{cnn_p0,cnn_p2}/seed_000/downstream/."
