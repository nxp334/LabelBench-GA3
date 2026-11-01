# GA3 deviations from stock LabelBench

Every way the GA3 experiment differs from stock LabelBench. Without the GA3 arguments and configs, stock runs behave as
before.

| Deviation | Reason |
|---|---|
| `cifar100_subset` task construction: K classes from a seeded shuffle of the 100 ids, labels remapped to 0..K-1, full pool of those classes, stock validation/test split restricted to them | Gives nested K-class tasks, which the stock `n_class` option cannot, since it merges extra classes into "others". |
| Balanced init: 10 labels per class, drawn per (s, class id), instead of a uniform first batch | The starting composition is then equal across conditions, and identical across strategies and K for shared classes. |
| Batch size 10*K, for 20 batches | Keeps labels per class (b = 10, 20, ..., 200) comparable across K. |
| Labeled sets saved after every round as `orig_idx` | Every round can be evaluated later, and baselines can match AL's per-class counts at each B. |
| Label-cache bypass for `cifar100_subset` | The label cache is keyed by dataset name only, so different (K, s) would load each other's labels. |
| `"cra"` -> `"crab"` in `cifar100_subset` only | `"cra"` is a typo in `classnames.py` that would corrupt the zero-shot prompt of that class. Other datasets stay as they are. |
| Proxy `min_steps` = 500: more epochs when needed, and no early stopping before 500 steps | Stock proxy training scales with B, so small K took as few as 20 steps. That would confound the quality of AL selection with K. E2e is unchanged. |
| Feature slices kept in memory per process, and `num_workers` = 0 for the proxy | `min_steps` means many short epochs, and stock reloads features and restarts loader workers every epoch. Results are not affected. |
| Evaluation skips the full-pool pass | Only test scores are needed, and predicting on the whole pool is slow in e2e. |
| One evaluation job per (K, s) at b = 10 | All conditions share the init set there, so `collect_results.py` repeats that one result for every condition. |
| Raw outputs saved to disk: task metadata, test predictions, test labels, per-class results | Any later analysis can run without retraining. W&B only mirrors the main metrics. |
| Seeding: evaluation seeds torch, numpy and `random` from (s, B) only, right before training; acquisition uses `--seed s` as stock | Conditions at the same (s, B) get the same training seed, so differences come from the labeled set. Runs are seeded, not bit-exact, because `cudnn.benchmark` stays on. |
