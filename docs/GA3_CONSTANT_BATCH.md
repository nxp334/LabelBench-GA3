# Constant acquisition batch experiment

A separate experimental regime on branch `ga3-constant-batch`. It changes one thing relative to the floor500 run
(branch `ga3-class-composition`, results with `git_commit` `2129289`): every acquisition round adds 100 labels,
independent of K. No results are reported here.

## What changes and what stays

| | Floor500 run | Constant batch |
|---|---|---|
| Labels per acquisition round | 10·K | 100 |
| Rounds after the init | 19 | 1.9·K (derived) |
| K, seeds | {10, 20, 50, 100}, 1–4 | same |
| Init | 10 per class, `balanced_init` | same sets (derived: seeded by (s, class id) only) |
| Evaluated grid | B = b·K, b = 10, 20, ..., 200 | same |
| Strategies, baselines | margin, badge; random, cb-margin, cb-badge | same |
| Trainer config | `clip_ViTB32_linear_floor500.json` | same, unchanged |
| Output directory | `~/ga3_output_floor500` | `~/ga3_output_constbatch` |

| K | Rounds now | Rounds at 100 | Ratio | `--num_batch` |
|---|---|---|---|---|
| 10 | 19 | 19 | 1× | 20 |
| 20 | 19 | 38 | 2× | 39 |
| 50 | 19 | 95 | 5× | 96 |
| 100 | 19 | 190 | 10× | 191 |

All values are derived. Every grid point is reached, because 100 divides 10·K for every K. At K=10 the design is the
same as the floor500 run (10·K = 100).

## Implementation

- `scripts/launch_acquisition.sh`: `ROUND_SIZE` (unset by default) sets `--batch_size ROUND_SIZE` and
  `--num_batch 1 + 190·K / ROUND_SIZE`. It must divide 10·K, otherwise the script stops before any run. The second line
  of each run log is then `round_size: ROUND_SIZE`, and a finished run without that line is refused on restart. `KS`
  (default `10 20 50 100`) sets the K order. With both unset, the 32 `main.py` commands of seeds 1–4 are identical to
  before (`test_default_acquisition_plan_is_unchanged`, and a direct comparison with the previous script).
- `scripts/run_constbatch.sh`: sets `ROUND_SIZE=100`, `KS="10 100 20 50"` and the floor500 config, and runs
  `scripts/run_seeds.sh` for the stages seed 1, seeds 2–3, seed 4, each retried once. A stage must finish before the
  next starts, so finished seeds are usable on their own. K=100 starts second because its runs are the longest. It
  refuses `SAVE_DIR` named `ga3_output` or `ga3_output_floor500`.
- `main.py`, `make_baselines.py`, the queue and evaluation are unchanged.

## Dependence on the round size

- **Evaluation:** none. The queue is built from the b grid (384 jobs per seed), and the training seed is
  `derive_seed("train", s, B)`.
- **Baselines:** the logic does not depend on it, but `make_baselines.py` builds baselines for every saved AL file,
  and `main.py` saves one per round. Off-grid baseline files are written but never queued.
- **Init:** shared with the floor500 run for the same (K, s), so the b=10 evaluation is paired. Later sets differ,
  because selection depends on the earlier rounds.

## Expected counts per seed (derived)

| Output | Count |
|---|---|
| AL files `acq/{margin,badge}_K*_s*_B*.npy` | 2 × (20 + 39 + 96 + 191) = 692 |
| CB files | 692 |
| `random` files | 346 |
| Evaluation jobs | 384 |
| `results_proxy.csv` rows | 400 |

## Run time (estimate)

Based on the floor500 run (about 1.5 h acquisition and 2–3.5 h evaluation per seed). It assumes evaluation is
unchanged and acquisition time grows with the number of rounds, at between equal cost per round and cost proportional
to K. About 9–18.5 h per seed, 36–74 h in total, central about 50 h. Not measured.

## Risks

1. **24-hour limit per acquisition run.** A K=100 run has 190 rounds. At the slowest assumed rate (about 4.7 min per
   round) it takes about 15 h (estimate), within the limit. A run that hits it fails its stage.
2. **K=10 repeats the floor500 design.** Its results are a repeat of the floor500 K=10 runs, seeded but not bit-exact.
3. **Disk:** about 1730 extra small `.npy` files per seed (derived).
