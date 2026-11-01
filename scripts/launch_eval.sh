#!/usr/bin/env bash
# MAX_PARALLEL run_queue.py workers per GPU, one tmux window per worker. Safe to rerun: finished jobs are skipped.
# First, after the AL runs: tools/make_baselines.py for every (K, s), then tools/make_queue.py.
# Usage: scripts/launch_eval.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-/work/LabelBench/data}"
SAVE_DIR="${SAVE_DIR:-/work/ga3_output}"
PYTHON_BIN="${PYTHON_BIN:-/work/LabelBench/env/bin/python}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
WANDB_PROJECT_EVAL="${WANDB_PROJECT_EVAL:-LabelBench - ClassBaseline - eval}"
QUEUE_FILE="${QUEUE_FILE:-$SAVE_DIR/queue.tsv}"
LOG_DIR="$SAVE_DIR/logs"
SESSION=ga3_eval
mkdir -p "$LOG_DIR"

if [[ ! -f "$QUEUE_FILE" ]]; then
  echo "Queue file not found: $QUEUE_FILE" >&2
  exit 1
fi

# Remove any existing session to avoid stale windows.
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION" || true

num_workers=$((NUM_GPUS * MAX_PARALLEL))
for worker in $(seq 0 $((num_workers - 1))); do
  gpu=$((worker % NUM_GPUS))
  window_name="worker${worker}_gpu${gpu}"
  cmd="cd \"$REPO_DIR\" && export PYTHONPATH=\"$REPO_DIR\" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=${gpu} && \"$PYTHON_BIN\" tools/run_queue.py --queue \"$QUEUE_FILE\" --worker ${worker} --num_workers ${num_workers} --save_dir \"$SAVE_DIR\" --data_dir \"$DATA_DIR\" --wandb_name ATDL-labelbench --wandb_project \"$WANDB_PROJECT_EVAL\" > \"$LOG_DIR/${window_name}.out\" 2>&1"
  if [[ "$worker" -eq 0 ]]; then
    tmux new-session -d -s "$SESSION" -n "$window_name" "$cmd"
  else
    tmux new-window -t "$SESSION" -n "$window_name" "$cmd"
  fi
done

tmux list-windows -t "$SESSION"
