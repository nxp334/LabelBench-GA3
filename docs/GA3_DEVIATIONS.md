# GA3 deviations from stock LabelBench

Every way the GA3 experiment differs from stock LabelBench (upstream commit `9e23393`). Five existing files are
modified; everything else is a new file.

| Modified stock file | Change |
|---|---|
| `LabelBench/dataset/datasets.py` | `get_dataset` takes keyword arguments; with them, labels bypass the label cache. |
| `LabelBench/trainer/utils.py` | New functions `effective_max_epoch`, `should_validate`, `returned_model_step`. `EarlyStopping` is unchanged. |
| `LabelBench/trainer/trainer_impl/pytorch_semi_trainer.py` | Epoch limit from `effective_max_epoch`, checks through `should_validate`, new run attributes. |
| `main.py` | New arguments, W&B metadata, balanced init, saved labeled sets, per-round logs. |
| `configs/embed_model/clip_ViTB32.json` | `num_workers` 40 -> 6 ("for 8-vCPU VM", commit `c1e2bba`). |

Without the GA3 arguments and the new config keys, the code paths of stock runs are as before, with two exceptions:
`config=vars(args)` sends the five new `main.py` arguments to W&B, and stock runs share the changed embed config. A
different embed worker count may change the RandAugment draws of newly computed strong-augmented feature files (not
verified). Existing feature files are reused unchanged.

| Deviation | Reason |
|---|---|
| `cifar100_subset` task construction: K classes from a seeded shuffle of the 100 ids, labels remapped to 0..K-1, full pool of those classes, stock validation/test split restricted to them | Gives nested K-class tasks, which the stock `n_class` option cannot, since it merges extra classes into "others". |
| Balanced init: 10 labels per class, drawn per (s, class id), instead of a uniform first batch | The starting composition is then equal across conditions, and identical across strategies and K for shared classes. |
| Batch size 10*K, for 20 batches | Keeps labels per class (b = 10, 20, ..., 200) comparable across K. |
| Labeled sets saved after every round as `orig_idx` | Every round can be evaluated later, and baselines can match AL's per-class counts at each B. |
| Label-cache bypass for `cifar100_subset` | The label cache is keyed by dataset name only, so different (K, s) would load each other's labels. |
| `"cra"` -> `"crab"` in `cifar100_subset` only | `"cra"` is a typo in `classnames.py` that would corrupt the zero-shot prompt of that class. Other datasets stay as they are. |
| Proxy `min_steps` = 500 (minsteps config, finished n=3 run): more epochs when needed, and no early stopping before 500 steps | Stock proxy training scales with B, so small K took as few as 20 steps. That would confound the quality of AL selection with K. E2e is unchanged. |
| Floor500 config for the rerun: see "Training regimes" | At `c1e2bba`, `min_steps` also ended training at small B, so early stopping could not choose a model there. |
| Feature slices kept in memory per process, and `num_workers` = 0 for the proxy (see "Training regimes") | `min_steps` means many short epochs, and stock reloads features and restarts loader workers every epoch. |
| `random` built offline (`tools/make_baselines.py`): the init set plus a fixed permutation of the rest of the pool, instead of the stock `random` strategy | Needs no acquisition run, and is nested across B. |
| ClassBaseline (`cb-margin`, `cb-badge`): a new condition with AL's per-class counts at each B, drawn at random within each class | Separates the class composition of AL's labels from which images AL picked within each class. |
| New `main.py` arguments (`--num_classes`, `--class_seed`, `--init_per_class`, `--save_dir`, `--wandb_project`) and W&B `job_type`, `group`, `tags` for subset runs | Carry K, s and the output folder, and group runs by task. |
| New trainer attributes: `num_steps_taken`, `stopped_epoch`, `best_val_loss`, `ended_by`, `best_step` | Training length and model selection can be audited per evaluation. They use no random numbers. |
| Evaluation trains a new model per saved set (`tools/eval_selection.py`) | Every condition is scored the same way. |
| Evaluation skips the full-pool pass | Only test scores are needed, and predicting on the whole pool is slow in e2e. |
| One evaluation job per (K, s) at b = 10 | All conditions share the init set there, so `collect_results.py` repeats that one result for every condition. |
| Smoke configs (`configs/trainer/cifar100/smoke/`, `max_epoch` 1), single-GPU launchers (`scripts/launch_acquisition.sh`, `scripts/launch_eval.sh`), queue tiers (`tools/make_queue.py`), `tools/precompute_features.py` | Fast smoke test; parallel runs on one GPU; a partly finished queue has complete seeds; feature files are written once before parallel runs, because `FeatureExtractor` writes without locking. |
| Embedding standardisation uses task-pool statistics (`dataset_skeleton.py:247-256`, unchanged code) | A consequence of the task pool being the train set, not a code change. Mean and std are the same for all conditions of one task, but differ across K. |
| Raw outputs saved to disk: task metadata, test predictions, test labels, per-class results | Any later analysis can run without retraining. W&B only mirrors the main metrics. |
| Seeding: evaluation seeds torch, numpy and `random` from (s, B) only, right before training; acquisition uses `--seed s` as stock | Conditions at the same (s, B) get the same training seed, so differences come from the labeled set. Runs are seeded, not bit-exact, because `cudnn.benchmark` stays on. |

## Training regimes

There are two runs with different training:

1. **Finished n=3 run** (commit `c1e2bba`, `clip_ViTB32_linear_minsteps.json`). `min_steps` raises the epoch limit
   only far enough to reach 500 steps, so below about 3500 labels this is effectively a fixed 500-step budget. Early
   stopping is inactive until the last epoch, so only one check runs and the final model is kept (derived).
2. **Floor500 rerun** (`clip_ViTB32_linear_floor500.json`), for acquisition and evaluation. "Floor" means a lower
   bound, not rounding down: training runs for at least 500 optimizer steps (`min_steps`), and after that early
   stopping can end it, at most 7500 steps (`max_steps`).

Selections are path dependent: margin and BADGE pick labels with the model trained in the previous round. The two runs
therefore have different labeled sets and cannot be compared set by set.

| Key | Stock `clip_ViTB32_linear.json` | Floor500 |
|---|---|---|
| `max_steps` | none | 7500 (new) |
| `min_steps` | none | 500 (new, floor) |
| `val_every_steps` | none | 100 (new) |
| `patience` | 3 | 10 |
| `num_workers` | 10 | 0 |
| `max_epoch` | 20 | 20, unused when `max_steps` is set (`effective_max_epoch`, `utils.py:10-23`) |

All other keys are the same. The justification below is ours. The values were not tuned, and their effect was not
measured.

- **`warmup_steps` is not a linear warm-up.** It is `T_0` of `CosineAnnealingWarmRestarts`, the length of the first
  cosine cycle (`trainer.py:101-107`):

  ```python
  if trainer_config["scheduler_name"] == "CosineLR":
      def scheduler_fn(optimizer, total_steps):
          return CosineAnnealingWarmRestarts(
              optimizer,
              T_0=trainer_config["warmup_steps"],
              T_mult=trainer_config["T_multi"] if "T_multi" in trainer_config else 2,
              eta_min=trainer_config["eta_min"] if "eta_min" in trainer_config else 0)
  ```

  The scheduler steps once per optimizer step (`pytorch_semi_trainer.py:122`). The learning rate starts at its maximum,
  anneals to `eta_min` 0 at step 500, then restarts. `T_mult` 2 gives cycles of 500, 1000, 2000 and 4000 steps, so
  restarts fall at 500, 1500, 3500 and 7500 (derived from PyTorch's definition).
- **`min_steps` 500:** one complete first cycle, so no stopping decision is made while the learning rate is still high.
- **`val_every_steps` 100:** at 100 or more steps per epoch (B >= 12800), validation runs every epoch, as in stock. At
  smaller B it runs at the first epoch end at least 100 steps after the last check. Below 256 labels one epoch is one
  step, so checks are exactly 100 steps apart (derived).
- **`patience` 10:** 10 checks x 100 steps = 1000 steps without improvement, the length of the second cycle
  (`T_0 x T_mult`). The earliest possible stop is step 1500 (the second restart) only when an epoch is one step
  (B < 256). For larger B the checks fall at epoch ends, e.g. step 1560 at K=10, B=2000 and 2184 at K=100, B=20000
  (derived).
- **`max_steps` 7500:** 500 + 1000 + 2000 + 4000, the end of the fourth cosine cycle (derived). It also acts as a
  safety cap. The limit is in whole epochs, so it can be passed by less than one epoch (7644 steps at K=100, B=20000,
  derived).
- **`num_workers` 0:** training runs on precomputed in-memory features (5 augmented views from
  `tools/precompute_features.py`, `num_transform_seeds` 5 in the embed config), and the trainer rebuilds its
  DataLoaders every epoch (`pytorch_semi_trainer.py:76-92`). Workers add overhead and do no useful work. Not expected to
  change the distribution of results; individual numbers can differ through RNG order (not verified).

**Unchanged vs stock:** the FlexMatch trainer (`use_strong = True`, `flexmatch.py:13`); frozen CLIP ViT-B/32 features
in 5 augmented views; the unlabeled pool in the FlexMatch loss (`unlabeled_idxs`, `pytorch_semi_trainer.py:86`);
validation and test features without augmentation (`feature_extractor.py:29-32`).

## Recording

Each evaluation JSON and each row of `results_{mode}.csv` records `ended_by` (`early_stopping` or `epoch_limit`),
`best_step` (step of the model training returns), `trainer_config` and `git_commit`. `best_step` is the last CSV
column. Results written before a field existed have an empty cell.

## Pipeline

`scripts/run_seeds.sh <first_seed> <last_seed>` runs one stage: acquisition, baselines, queue, evaluation, results
table. The floor500 rerun uses three stages: seed 1, seeds 2-3, seed 4. A stage exits with 0 only if all of these
hold:

- `SAVE_DIR`, `TRAINER_CONFIG`, `DATA_DIR`, `PYTHON_BIN`, `WANDB_PROJECT_ACQ` and `WANDB_PROJECT_EVAL` are set;
- `SAVE_DIR` is not named `ga3_output`;
- every acquisition log of the stage's seeds names `TRAINER_CONFIG`;
- 384 jobs per seed are queued, and each has a result;
- every row of `results_proxy.csv` has `TRAINER_CONFIG`.

Acquisition runs have a 24-hour limit and evaluation jobs a 1-hour limit.

## Limitation

Early stopping interacts with the learning-rate restarts. After a restart the learning rate is back at its maximum,
and validation loss can rise. The effect on when training stops, and on which model is kept, is not analysed.
