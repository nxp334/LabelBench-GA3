# LabelBench-GA3 (ATDL – GA3)

This fork of [LabelBench](https://github.com/EfficientTraining/LabelBench) asks how much of the active learning gain
of Margin and BADGE comes from class composition, meaning how many labels each class gets. The tasks are nested
CIFAR-100 subsets with K = 10, 20, 50 and 100 classes. Each active learning set is compared with a ClassBaseline: the
same number of labels per class, with the images drawn at random within each class. Setting: frozen CLIP ViT-B/32
features with a linear head (proxy), FlexMatch training, 4 seeds. The reported run is called floor500: every training
run takes at least 500 optimizer steps (a lower bound, not rounding). The original LabelBench documentation is in
[README.md](README.md).

## Where things are

| Path | Contents |
|---|---|
| [docs/GA3_DEVIATIONS.md](docs/GA3_DEVIATIONS.md) | Every change from stock LabelBench, and why. |
| [docs/ANALYSIS.md](docs/ANALYSIS.md) | Definitions, aggregation, and the captions of all figures and tables. |
| [scripts/SMOKE_TEST.md](scripts/SMOKE_TEST.md) | VM setup, smoke test and the individual pipeline steps. |
| `ga3_results/floor500/results_proxy.csv` | Results: one row per (K, seed, B, condition), 1600 rows. |
| `ga3_results/tables/`, `ga3_results/figures/` | Tables (CSV and LaTeX) and figures. |
| `tools/` | Baselines, evaluation queue, evaluation, result collection, analysis and figures. |
| `scripts/` | Launchers; `run_seeds.sh` runs the whole pipeline for a range of seeds. |
| `tests/` | Unit tests (`pytest tests/`). |

## Reproducing

- **Tables and figures:** see "Reproducing" in [docs/ANALYSIS.md](docs/ANALYSIS.md). The gain tables need only the
  committed CSV. The task tables (T1) and all figures also need the raw run output (task files and labeled sets, not
  in the repository), and `tools/make_figures.py` also reads CIFAR-100.
- **Full run:** set up as in sections 1 and 2 of [scripts/SMOKE_TEST.md](scripts/SMOKE_TEST.md), then

  ```bash
  TRAINER_CONFIG=cifar100/flexmatch/clip_ViTB32_linear_floor500.json SAVE_DIR=... scripts/run_seeds.sh 1 4
  ```

  The other required variables are listed under "Pipeline" in [docs/GA3_DEVIATIONS.md](docs/GA3_DEVIATIONS.md).
