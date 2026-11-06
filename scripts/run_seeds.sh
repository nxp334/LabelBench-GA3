#!/usr/bin/env bash
# One stage for seeds first_seed..last_seed: acquisition, baselines, queue, evaluation, results table.
# Stays in the foreground and exits with 0 only if every job of the stage finished, so stages can be chained with &&.
# Restarting is safe: finished runs and jobs are skipped, and baselines are rebuilt identically.
# The variables below have no default here, so a run cannot fall back on the old directory, config or W&B projects.
# Usage: SAVE_DIR=... TRAINER_CONFIG=... DATA_DIR=... PYTHON_BIN=... WANDB_PROJECT_ACQ=... WANDB_PROJECT_EVAL=... \
#   scripts/run_seeds.sh <first_seed> <last_seed>
set -euo pipefail

FIRST_SEED="$1"
LAST_SEED="$2"
: "${SAVE_DIR:?Set SAVE_DIR to a new directory.}"
: "${TRAINER_CONFIG:?Set TRAINER_CONFIG.}"
: "${DATA_DIR:?}"
: "${PYTHON_BIN:?}"
: "${WANDB_PROJECT_ACQ:?}"
: "${WANDB_PROJECT_EVAL:?}"
if [[ "$(basename "$SAVE_DIR")" == ga3_output ]]; then
  echo "Refusing to write to $SAVE_DIR: it holds the finished n=3 results." >&2
  exit 1
fi

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
QUEUE_FILE="$SAVE_DIR/queue_s${FIRST_SEED}-${LAST_SEED}.tsv"
SEEDS=$(seq "$FIRST_SEED" "$LAST_SEED")
# Per (K, s): one job at b = 10 and five conditions at each of the 19 later b, over four K.
JOBS_PER_SEED=$((4 * (1 + 5 * 19)))

export SAVE_DIR TRAINER_CONFIG DATA_DIR PYTHON_BIN WANDB_PROJECT_ACQ WANDB_PROJECT_EVAL QUEUE_FILE
export PYTHONPATH="$REPO_DIR" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$REPO_DIR"

scripts/launch_acquisition.sh --first_seed "$FIRST_SEED" --last_seed "$LAST_SEED"

for seed in $SEEDS; do
  other_config=$(grep -LFx "trainer_config: $TRAINER_CONFIG" "$SAVE_DIR"/logs/acquisition/*_s"${seed}".log || true)
  if [[ -n "$other_config" ]]; then
    echo "These runs did not use $TRAINER_CONFIG:" >&2
    echo "$other_config" >&2
    exit 1
  fi
done

for seed in $SEEDS; do
  for K in 10 20 50 100; do
    "$PYTHON_BIN" tools/make_baselines.py --save_dir "$SAVE_DIR" --K "$K" --s "$seed" --data_dir "$DATA_DIR"
  done
done

# Word splitting is intended: --seeds takes one argument per seed.
"$PYTHON_BIN" tools/make_queue.py --save_dir "$SAVE_DIR" --modes proxy --seeds $SEEDS --output "$QUEUE_FILE" \
  --trainer_config "$TRAINER_CONFIG"

scripts/launch_eval.sh
while tmux has-session -t ga3_eval 2>/dev/null; do
  sleep "${POLL_SECONDS:-60}"
done

num_jobs=0
num_missing=0
while IFS=$'\t' read -r mode file _; do
  num_jobs=$((num_jobs + 1))
  if [[ ! -f "$SAVE_DIR/eval/$mode/$(basename "$file" .npy).json" ]]; then
    num_missing=$((num_missing + 1))
  fi
done < "$QUEUE_FILE"
num_expected=$((JOBS_PER_SEED * (LAST_SEED - FIRST_SEED + 1)))
if [[ "$num_jobs" -ne "$num_expected" || "$num_missing" -ne 0 ]]; then
  echo "Stage incomplete: ${num_jobs} of ${num_expected} jobs queued, ${num_missing} without a result." >&2
  exit 1
fi

"$PYTHON_BIN" tools/collect_results.py --save_dir "$SAVE_DIR"
"$PYTHON_BIN" - "$SAVE_DIR/results_proxy.csv" "$TRAINER_CONFIG" <<'CHECK'
import csv
import sys

with open(sys.argv[1], newline="") as f:
    configs = {row["trainer_config"] for row in csv.DictReader(f)}
if configs != {sys.argv[2]}:
    sys.exit(f"Results use trainer configs {sorted(configs)}, expected only {sys.argv[2]}.")
CHECK
echo "Seeds ${FIRST_SEED}-${LAST_SEED} finished: ${num_jobs} jobs."
