# Running GA3 on a Google Cloud VM with one L4

For a Deep Learning VM image with one NVIDIA L4, proxy evaluation only. The code must be committed and pushed before
step 1, since the VM clones it from GitHub. Do the setup and smoke test first, then the full run in the order below.

## 1. Setup

```bash
git clone -b ga3-class-composition https://github.com/nxp334/LabelBench-GA3.git
cd LabelBench-GA3
pip install -r requirements.txt
wandb login

export REPO_DIR=$PWD PYTHONPATH=$PWD TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export DATA_DIR=$HOME/data SAVE_DIR=$HOME/ga3_output PYTHON_BIN=$(command -v python)
export NUM_GPUS=1 MAX_PARALLEL=4
export WANDB_PROJECT_SMOKE="LabelBench - ClassBaseline"
export WANDB_PROJECT_ACQ="LabelBench - ClassBaseline - acquisition"
export WANDB_PROJECT_EVAL="LabelBench - ClassBaseline - eval"
```

Run every command below from the repo root, in a shell with these exports. Add them to `~/.bashrc` so that new tmux
windows have them too. `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is needed because the feature files contain numpy arrays.

W&B creates these projects in the entity `ATDL-labelbench` on first use. Check the connection before anything else:

```bash
"$PYTHON_BIN" -c "import wandb; r = wandb.init(entity='ATDL-labelbench', project='LabelBench - ClassBaseline', name='wandb-check', job_type='check'); r.log({'ok': 1}); r.finish()"
```

Expect a run named `wandb-check` in that project.

## 2. Precompute the features

```bash
"$PYTHON_BIN" tools/precompute_features.py --data_dir $DATA_DIR
```

Downloads CIFAR-100 and CLIP, and writes about 2.3 GB to `$DATA_DIR/cifar100/`. Run it in tmux, it needs one pass over
50000 images per file. Expect `All feature files are ready.` Running it again does nothing new.

## 3. Smoke test

The results go to `./smoke_output`, so they can never be picked up as real results. The numbers mean nothing, except in
3a and 3e.

**3a. Zero-shot check**

```bash
"$PYTHON_BIN" tools/zero_shot_check.py --K 10 --s 1 --data_dir $DATA_DIR
```

Expect ten class names, starting with `squirrel`, `table`, `forest`, and one accuracy line on about 500 images. Accuracy
must be far above the chance level of 0.10. Close to 0.10 means labels and class names are misaligned: stop here.

**3b. Acquisition**

```bash
"$PYTHON_BIN" main.py --seed 1 --wandb_name ATDL-labelbench --wandb_project "$WANDB_PROJECT_SMOKE" \
  --dataset cifar100_subset --num_classes 10 --class_seed 1 --save_dir ./smoke_output --data_dir $DATA_DIR \
  --metric multi_class --batch_size 100 --num_batch 3 \
  --embed_model_config clip_ViTB32.json --classifier_model_config linear.json \
  --strategy_config margin_sampling.json --trainer_config cifar100/smoke/clip_ViTB32_linear.json
```

Expect `Round 1: 100 labels, training took ...` for each round, and in `smoke_output/acq/`: `margin_K10_s1_B100.npy`,
`_B200.npy`, `_B300.npy`. Also expect `smoke_output/tasks/K10_s1.json`, with 10 `class_ids`, their class names, pool size
5000 and 100 `init_idxs`. In W&B each round logs `B`, `b`, `Test Accuracy` and the class-distribution summary. At B=100,
`class_entropy_norm` must be 1.0.

**3c. Baselines**

```bash
"$PYTHON_BIN" tools/make_baselines.py --save_dir ./smoke_output --K 10 --s 1 --data_dir $DATA_DIR
```

Expect `Wrote 6 baseline files`: `cb-margin` and `random`, each at B = 100, 200, 300. An assertion error means the AL
files and the reconstructed init disagree.

**3d. Evaluation at B=200**

```bash
"$PYTHON_BIN" tools/eval_selection.py --file smoke_output/acq/margin_K10_s1_B200.npy --mode proxy \
  --save_dir ./smoke_output --data_dir $DATA_DIR --wandb_project "$WANDB_PROJECT_SMOKE" \
  --trainer_config cifar100/smoke/clip_ViTB32_linear.json
"$PYTHON_BIN" tools/collect_results.py --save_dir ./smoke_output
```

Expect:

- `smoke_output/eval/proxy/margin_K10_s1_B200.json` with `b` = 20, `condition` = `margin`, `per_class_counts` (10 numbers
  summing to 200), `per_class_test_accuracy` (10 numbers), `steps_taken` = 1, `stopped_epoch` = 1, `best_val_loss`,
  `train_seconds`, `git_commit` and the class-distribution summary. Rerunning the command must print `Skipping`.
- `smoke_output/eval/proxy/preds_margin_K10_s1_B200.npy`: predictions in test-set order.
- `smoke_output/tasks/K10_s1_test_labels.npy`: labels of the same length. Their match with the predictions must give
  the `test_accuracy` in the JSON.
- `smoke_output/results_proxy.csv` with a row for `margin` at B=200.
- A W&B run named `margin_K10_s1_B200`, with job type `eval-proxy` and group `K10_s1`.

**3e. Is 500 steps enough?**

```bash
"$PYTHON_BIN" tools/proxy_check.py --K 10 --s 1 --B 100 --file smoke_output/acq/margin_K10_s1_B100.npy \
  --save_dir ./smoke_output --data_dir $DATA_DIR
```

This uses the real min_steps config. Expect `optimizer steps=500`, two accuracies, and the same result in
`smoke_output/checks/proxy_check_K10_B100.json`. Logistic regression is trained to
convergence on the same labeled images, so it shows roughly what a well-trained proxy reaches. If the proxy is far below
it, 500 steps is too few: raise `min_steps` in `clip_ViTB32_linear_minsteps.json` before the full run.

**Optional, for later e2e use:** the same `eval_selection.py` command with `--mode e2e` and
`--trainer_config cifar100/smoke/clip_ViTB32_finetune.json`. Expect `steps_taken` = 4.

## 4. Full run

**4a. Acquisition.** Start it in tmux: it stays in the foreground, runs at most `MAX_PARALLEL` runs at a time and starts
the next when one ends.

```bash
tmux new -s acq
scripts/launch_acquisition.sh --num_seeds 3
```

Each run logs to `$SAVE_DIR/logs/acquisition/`. If it stops, or a run failed, run the same command again: finished runs
are skipped and unfinished runs start from the beginning. Add `--skip_badge_k100` if BADGE at K=100 runs out of memory.

**4b. Baselines**

```bash
for s in 1 2 3; do for K in 10 20 50 100; do
  "$PYTHON_BIN" tools/make_baselines.py --save_dir $SAVE_DIR --K $K --s $s --data_dir $DATA_DIR
done; done
```

**4c. Queue**

```bash
"$PYTHON_BIN" tools/make_queue.py --save_dir $SAVE_DIR --modes proxy
```

Expect `Wrote 1152 jobs`. With `--skip_badge_k100`, the K=100 BADGE jobs are dropped with a warning.

**4d. Evaluation**

```bash
scripts/launch_eval.sh
```

This starts one tmux window per worker in the session `ga3_eval` and returns. Follow the progress with
`tail -f $SAVE_DIR/logs/worker0.log`. The output of the jobs is in `$SAVE_DIR/logs/worker*.out`. If a worker stops, run the
same command again: finished jobs are skipped. Results are in `$SAVE_DIR/eval/proxy/`.

**4e. Results table**

```bash
"$PYTHON_BIN" tools/collect_results.py --save_dir $SAVE_DIR
```

Writes `$SAVE_DIR/results_proxy.csv`, with one row per (K, s, B, condition). At b = 10 all conditions share the same
labeled set, so there is one job per (K, s), and the CSV repeats that result for all five conditions. The column
`evaluated_as` shows which condition was actually trained. Per-class results are in the JSON files, and the files in
`$SAVE_DIR` are the source of truth. W&B only mirrors the main metrics.
