# Analysis

What is computed from the results CSV of the floor500 run (`ga3_results/floor500/results_proxy.csv`), and how. This
document contains no results.

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

These are exploratory.

- **Class composition:** for b = 100 (main) and b = 200 (appendix), conditions margin, badge and random, every K
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
- **Gain vs b:** total and composition per b, mean over seeds.
- **Stopping:** per K, over rows that were trained as their own condition (`evaluated_as == condition`) with
  b ≥ 20: the share of rows with `ended_by == early_stopping`, and the medians of `steps_taken` and `best_step`.

## Notes

- Absolute accuracy is not compared across K. The tasks differ in their number of classes.
- The test set has about 50 images per class at every K, so the total test size grows with K (`test_size` in the
  task files).
- Seed-level variation mixes class-subset and algorithm randomness at K = 10, 20 and 50. At K = 100 the class set is
  identical for all seeds, so only algorithm randomness varies.
- Chance normalisation (gain / (1 − 1/K)) changes the values by at most about 0.1 pp, so it is not used.

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

## Captions

All figures and tables use n = 4 seeds. All gains are paired by (K, s, b): margin with cb-margin, badge with
cb-badge, both with random.

**Figure 1** (`ga3_results/figures/fig1_accuracy_vs_b.png`). Test accuracy (%) vs b, rows margin and badge,
columns K = 10, 20, 50, 100, b = 10 to 200. Lines: random (grey, dotted), cb-AL (orange, dashed) and AL (blue,
solid), mean over 4 seeds. Bands: minimum to maximum over the 4 seeds. The y-axis differs per K. b = 10 (the shared
initial labeled set) is plotted but excluded from the analysis.

**Figure 2** (`fig2_decomposition.png`). Total, composition and selection gain (pp) per K, panels margin and badge.
Large markers with bars: mean over 4 seeds of the per-seed mean over b = 20 to 200, with the 95% t-interval (3
degrees of freedom). Small points: the 4 per-seed means, shifted to the right with a small horizontal jitter.
Horizontal line at 0.

**Figure 3** (`fig3_class_composition_b100.png`, exploratory). Class composition of the labeled sets at b = 100.
Per K, the labels in each of the K classes divided by the balanced share b (q = c/b above), sorted from most to least
labeled, at x = (rank − 0.5)/K (fraction of classes), mean over 4 seeds per rank. Panels margin and badge. Solid
lines: AL, one per K, colour from light (K = 10) to dark (K = 100). Grey dotted lines: random, one per K, drawn in
the same grey. Horizontal line at 1.0 (balanced). Class identity is not shown, because ranks are averaged over seeds
with different class sets (see Table A-class-counts for K = 100). cb-AL is not shown: by construction it has the same
per-class counts as AL.

**Figure A1** (`figA1_class_composition_b200.png`, exploratory). As Figure 3, at b = 200.

**Figure A2** (`figA2_accuracy_all_K.png`). Test accuracy (%) vs b = 10 to 200, panels margin and badge. Solid: AL.
Dotted: random. Colour: K. Lines are the mean over 4 seeds. No bands.

**Figure A3** (`figA3_gain_vs_b.png`, exploratory). Total gain (top row) and composition gain (bottom row) in pp vs
b = 20 to 200, columns margin and badge. One line per K, mean over 4 seeds. No bands. Horizontal line at 0.

**Table T1** (`ga3_results/tables/T1_task_overlap`). Number of classes shared by each pair of seeds, per K, with
the minimum and maximum over the 6 pairs. Appendix (`T1_class_lists`): class ids, class names, validation and test
size per (K, s).

**Table T2** (`T2_decomposition`). Total, composition and selection gain in pp per K and strategy: mean over 4 seeds
of the per-seed mean over b = 20 to 200, with the 95% t-interval [lower, upper]. `rel_total_pct`: mean relative error
reduction of the total gain in %, aggregated the same way.

**Table A-macro-F1** (`A_macro_f1`). As T2 without `rel_total_pct`, with macro-F1 in place of test accuracy.

**Table A-equal-B** (`A_equal_B`). Total, composition and selection gain in pp at each B that is reached by at least
two K, b = 20 to 200. Per (B, K, strategy): mean and sample standard deviation over 4 seeds. One b per (B, K).

**Table A-class-counts** (`A_class_counts_K100_b100`, exploratory). K = 100, b = 100, 4 seeds with the same 100
classes. Per strategy, the 10 classes with the largest and the 10 with the smallest mean label count over the 4
seeds: class id and name, mean, sample standard deviation, minimum and maximum of the count, mean divided by b
(1.0 = balanced), and the number of seeds (of 4) in which the class is among the 10 most (or least) labeled.

**Table A-stopping** (`A_stopping`). Per K, over the rows trained as their own condition with b = 20 to 200 (all 5
conditions, 4 seeds): number of rows, share with `ended_by = early_stopping`, median `steps_taken` and median
`best_step`.
