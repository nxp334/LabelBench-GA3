#!/usr/bin/env bash
# Constant acquisition batch experiment: 100 labels per acquisition round at every K, otherwise the floor500 run.
# Runs the stages seed 1, seeds 2-3, seed 4 with scripts/run_seeds.sh. Each stage is retried once and must finish before
# the next starts, so finished seeds are usable on their own. K order 10, 100, 20, 50 starts the longest runs early.
# Usage: SAVE_DIR=... DATA_DIR=... PYTHON_BIN=... WANDB_PROJECT_ACQ=... WANDB_PROJECT_EVAL=... scripts/run_constbatch.sh
set -euo pipefail

: "${SAVE_DIR:?Set SAVE_DIR to a new directory.}"
case "$(basename "$SAVE_DIR")" in
  ga3_output|ga3_output_floor500)
    echo "Refusing to write to $SAVE_DIR: it holds the results of another run." >&2
    exit 1 ;;
esac

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROUND_SIZE=100 KS="10 100 20 50" TRAINER_CONFIG=cifar100/flexmatch/clip_ViTB32_linear_floor500.json
cd "$REPO_DIR"

for stage in "1 1" "2 3" "4 4"; do
  # Word splitting is intended: a stage is "first_seed last_seed".
  scripts/run_seeds.sh $stage || scripts/run_seeds.sh $stage
done
