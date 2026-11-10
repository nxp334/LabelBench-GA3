# Analysis

What is computed from the results CSV of the floor500 run (`ga3_results/floor500/results_proxy.csv`), and how. This
document contains no results. Floor500 = at least 500 training steps per run (see "Training regimes" in
`GA3_DEVIATIONS.md`).

## Symbols

| Symbol | Meaning |
|---|---|
| K | Number of classes of a task, K ∈ {10, 20, 50, 100}. |
| s | Seed, s ∈ {1, 2, 3, 4}. It selects the class subset and seeds acquisition and evaluation. |
| b | Labels per class. The labeled set has B = b·K labels, b ∈ {10, 20, ..., 200}. |
| A_AL(K, s, b) | Test accuracy of the labeled set chosen by the AL strategy (margin or badge). |
| A_CB(K, s, b) | Test accuracy of the ClassBaseline of that AL set (cb-margin or cb-badge): the same number of labels per class as the AL set, with images drawn at random within each class. |
| A_R(K, s, b) | Test accuracy of the random labeled set of size B. |

Each accuracy comes from one evaluation training on that labeled set (column `test_accuracy`).

## Primary analysis

Implemented in `tools/analyze_results.py`.

- total = A_AL − A_R, composition = A_CB − A_R, selection = A_AL − A_CB.
- composition + selection = total, because A_CB cancels. The code asserts this for every row.
- Pairing: margin with cb-margin, badge with cb-badge, both with random, at the same (K, s, b).
- b = 10 is excluded: all conditions share the initial labeled set there, so every gain is 0 by construction. The
  analysis uses b = 20, 30, ..., 200.
- Aggregation: per (K, strategy, s), the mean over b. Per (K, strategy), the mean, the sample standard deviation
  (n − 1) and the 95% t-interval over the 4 seed means (3 degrees of freedom, t = 3.182). The seed is the unit of
  replication.
- Metric: `test_accuracy`, gains in percentage points (pp).
- share_from_composition = mean composition / mean total, defined only when the mean total gain is at least
  0.5 pp, otherwise empty.

## Secondary analyses

- **Relative error reduction:** rel_total = total / (1 − A_R) and rel_composition = composition / (1 − A_R) per
  (K, s, b, strategy), in %, aggregated as in the primary analysis.
- **Macro-F1:** the same decomposition and aggregation with `macro_f1` in place of `test_accuracy`.
- **Equal B:** gains at the B values that occur at two or more K (for example B = 1000 at K = 10, 20 and 50). Per
  (B, K, strategy), the mean and sample standard deviation over seeds.

## Exploratory analyses

- **Class composition:** for b = 100 (main), b = 30 and b = 200 (appendix), conditions margin, badge and random, every K
  and seed. The labeled set `acq/<condition>_K<K>_s<s>_B<B>.npy` holds CIFAR-100 train indices, and each index is
  mapped to its class with the CIFAR-100 train targets. For each of the K classes of the task,
  q_k = c_k / b (= K·c_k / B), where c_k is the number of labels of class k. q = 1.0 is balanced. q is sorted in
  descending order, plotted at x = (rank − 0.5) / K, and averaged over seeds per rank. The share of classes with
  q < 0.5 is printed per (condition, K), as the mean over seeds.
- **Most and least labeled classes:** at K = 100 the class set is the same in all seeds, so label counts can be
  averaged per class. At b = 100, per strategy (margin, badge) and class: the mean, sample standard deviation,
  minimum and maximum of the label count over seeds, and the mean divided by b. The 10 classes with the largest and
  the 10 with the smallest mean are listed, with the number of seeds in which the class is among the 10 most (or
  least) labeled of that seed. Computed in `tools/make_figures.py` (`--tables_dir`).
- **Image examples:** at K = 100 and b = 100, per strategy, the 9 most and the 9 least labeled classes, ranked as in
  the previous item and shown from most to least labeled. For a condition, the eligible images are those in its
  labeled set at B = 10000 but not in the initial set (B = 1000), pooled over the 4 seeds. One image per class is
  drawn (`RandomState(0)`) for the strategy (images it picked) and one for its ClassBaseline (images drawn at random
  within each class). An image in both labeled sets is not shown twice: the ClassBaseline draw excludes the
  strategy's image.
- **Gain vs b:** total, composition and selection gain per b, mean over seeds.
- **Stopping:** per K, over rows that were trained as their own condition (`evaluated_as == condition`) with
  b ≥ 20: the share of rows with `ended_by == early_stopping`, and the medians of `steps_taken` and `best_step`.

## Notes

- Absolute accuracy is not compared across K. The tasks differ in their number of classes.
- The test set has about 50 images per class at every K, so the total test size grows with K (`test_size` in the
  task files).
- Seed-level variation mixes class-subset and algorithm randomness at K = 10, 20 and 50. At K = 100 the class set is
  identical for all seeds, so only algorithm randomness varies.
- Chance normalisation (gain / (1 − 1/K)) changes the values by at most about 0.1 pp, so it is not used.
- Top-k accuracy is not computed: evaluation saves only the predicted class (`tools/eval_selection.py`).

## Reproducing

From the repository root, with CIFAR-100 in `../ga3_data` and the raw output in `../ga3_results_raw/ga3_output_floor500`:

```bash
python tools/analyze_results.py --results ga3_results/floor500/results_proxy.csv \
  --tasks_dir ../ga3_results_raw/ga3_output_floor500/tasks --out_dir ga3_results/tables
PYTHONPATH=. python tools/make_figures.py --results ga3_results/floor500/results_proxy.csv \
  --raw_dir ../ga3_results_raw/ga3_output_floor500 --out_dir ga3_results/figures --cifar_dir ../ga3_data \
  --tables_dir ga3_results/tables
```

`--metric`, `--min_b` (default 20) and `--ci` (default 0.95) set the primary metric, the smallest b and the interval
level.

## Figures and tables

All use n = 4 seeds. Figures are in `ga3_results/figures/` (PNG), tables in `ga3_results/tables/` (CSV and LaTeX).

| File | Shows |
|---|---|
| `fig1_accuracy_vs_b` | Test accuracy vs b for AL, ClassBaseline and Random, per strategy and K (band: min to max over seeds). |
| `fig2_decomposition` | Total, composition and selection gain per K and strategy: mean, 95% t-interval and per-seed means. |
| `fig3_class_composition_b100` (`figA4`: b = 30, `figA1`: b = 200) | Labels per class divided by b, sorted by rank, AL and Random, per K. |
| `figA2_accuracy_all_K` | Test accuracy vs b, all K in one panel per strategy. |
| `figA3_gain_vs_b` | Total, composition and selection gain vs b, per K. |
| `figA5`, `figA6` (compact: `figA7`, `figA8`) | K = 100, b = 100: example images of the most and least labeled classes, strategy vs its ClassBaseline. |
| `T1_task_overlap`, `T1_overlap_compact`, `T1_class_lists` | Classes shared between seeds per K, and the class list of each task. |
| `T2_decomposition` | Gains per K and strategy: mean, 95% t-interval, and relative error reduction of the total gain. |
| `A_macro_f1` | As T2, with macro-F1. |
| `A_equal_B` | Gains at the B values reached by two or more K. |
| `A_class_counts_K100_b100` | K = 100, b = 100: the 10 most and 10 least labeled classes per strategy. |
| `A_stopping` | Per K: share of runs ended by early stopping, median `steps_taken` and `best_step`. |
