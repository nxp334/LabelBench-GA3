#!/usr/bin/env bash
# AL runs: margin and badge x K in {10,20,50,100} x s in 1..num_seeds.
# Runs at most MAX_PARALLEL processes per GPU and starts the next when one ends, so this script stays in the foreground:
# start it inside tmux. Each run logs to $SAVE_DIR/logs/acquisition/. Exits with 1 if a run failed.
# Restarting is safe: runs whose final labeled set (B = 200*K) exists are skipped, and unfinished runs start again.
# The first line of each run log names the trainer config of that run. TRAINER_CONFIG overrides it.
# Usage: scripts/launch_acquisition.sh [--num_seeds 3 | --first_seed 1 --last_seed 3] [--skip_badge_k100]
set -euo pipefail

FIRST_SEED=1
LAST_SEED=3
SKIP_BADGE_K100=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --first_seed) FIRST_SEED="$2"; shift 2 ;;
    --num_seeds|--last_seed) LAST_SEED="$2"; shift 2 ;;
    --skip_badge_k100) SKIP_BADGE_K100=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${DATA_DIR:?}"
SAVE_DIR="${SAVE_DIR:-/work/ga3_output}"
: "${PYTHON_BIN:?}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
WANDB_PROJECT_ACQ="${WANDB_PROJECT_ACQ:-LabelBench - ClassBaseline - acquisition}"
TRAINER_CONFIG="${TRAINER_CONFIG:-cifar100/flexmatch/clip_ViTB32_linear_minsteps.json}"
LOG_DIR="$SAVE_DIR/logs/acquisition"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$REPO_DIR" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2
cd "$REPO_DIR"

# Runs start on a filled cache: the feature files are written without locking.
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" tools/precompute_features.py --data_dir "$DATA_DIR"

runs=()
for seed in $(seq "$FIRST_SEED" "$LAST_SEED"); do
  for K in 10 20 50 100; do
    for strategy in margin badge; do
      if [[ "$SKIP_BADGE_K100" -eq 1 && "$strategy" == badge && "$K" -eq 100 ]]; then
        continue
      fi
      if [[ -f "$SAVE_DIR/acq/${strategy}_K${K}_s${seed}_B$((200 * K)).npy" ]]; then
        echo "Skipping ${strategy} K=${K} s=${seed}: already finished."
        continue
      fi
      runs+=("${strategy} ${K} ${seed}")
    done
  done
done

start_run() {
  local slot="$1" strategy="$2" K="$3" seed="$4"
  local name="${strategy}_K${K}_s${seed}"
  echo "trainer_config: $TRAINER_CONFIG" > "$LOG_DIR/${name}.log"
  CUDA_VISIBLE_DEVICES=$((slot % NUM_GPUS)) timeout 24h "$PYTHON_BIN" main.py --seed "$seed" --wandb_name ATDL-labelbench \
    --wandb_project "$WANDB_PROJECT_ACQ" --dataset cifar100_subset --num_classes "$K" --class_seed "$seed" \
    --save_dir "$SAVE_DIR" --data_dir "$DATA_DIR" --metric multi_class --batch_size $((10 * K)) --num_batch 20 \
    --embed_model_config clip_ViTB32.json --classifier_model_config linear.json \
    --strategy_config "${strategy}_sampling.json" --trainer_config "$TRAINER_CONFIG" \
    --corrupter_config noiseless.json >> "$LOG_DIR/${name}.log" 2>&1 &
  slot_pids[$slot]=$!
  slot_names[$slot]="$name"
  echo "$(date '+%F %T') start ${name} in slot ${slot}"
}

trap 'kill $(jobs -p) 2>/dev/null || true; exit 143' INT TERM HUP

num_slots=$((NUM_GPUS * MAX_PARALLEL))
slot_pids=()
slot_names=()
next_run=0
running=0
num_failed=0
while [[ "$next_run" -lt "${#runs[@]}" || "$running" -gt 0 ]]; do
  for slot in $(seq 0 $((num_slots - 1))); do
    pid="${slot_pids[$slot]:-}"
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      status=0
      wait "$pid" || status=$?
      echo "$(date '+%F %T') finished ${slot_names[$slot]} with exit code ${status}"
      if [[ "$status" -ne 0 ]]; then
        num_failed=$((num_failed + 1))
      fi
      slot_pids[$slot]=""
      pid=""
      running=$((running - 1))
    fi
    if [[ -z "$pid" && "$next_run" -lt "${#runs[@]}" ]]; then
      # Word splitting is intended: a run is "strategy K seed".
      start_run "$slot" ${runs[$next_run]}
      next_run=$((next_run + 1))
      running=$((running + 1))
    fi
  done
  sleep "${POLL_SECONDS:-10}"
done

echo "All runs finished, ${num_failed} failed."
[[ "$num_failed" -eq 0 ]]
