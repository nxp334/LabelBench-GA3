import json
import os

import numpy as np
import pandas as pd
import pytest

from tools.analyze_results import CONDITIONS
from tools.make_figures import class_count_table, example_picks, make_figures, matched_indices


def test_make_figures_smoke(tmp_path, train_targets):
    rows = []
    rng = np.random.RandomState(0)
    raw_dir = tmp_path / "raw"
    os.makedirs(raw_dir / "acq")
    os.makedirs(raw_dir / "tasks")
    for K, class_ids in ((2, [0, 1]), (4, [0, 1, 2, 3])):
        for s in (1, 2):
            with open(raw_dir / "tasks" / f"K{K}_s{s}.json", "w") as f:
                json.dump({"K": K, "s": s, "class_ids": class_ids, "classnames": [f"c{c}" for c in class_ids],
                           "val_size": 10, "test_size": 10}, f)
            for b in range(10, 210, 10):
                for condition in CONDITIONS:
                    value = 0.8 + 0.001 * b + 0.01 * rng.rand()
                    rows.append({"K": K, "s": s, "B": K * b, "b": float(b), "condition": condition,
                                 "evaluated_as": condition, "test_accuracy": value, "macro_f1": value,
                                 "steps_taken": 1000, "ended_by": "early_stopping", "best_step": 500,
                                 "trainer_config": "floor500.json"})
                    if b in (10, 30, 100, 200):
                        pool = np.where(np.isin(train_targets, class_ids))[0]
                        np.save(raw_dir / "acq" / f"{condition}_K{K}_s{s}_B{K * b}.npy",
                                rng.choice(pool, K * b, replace=False))
    results = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(results, index=False)

    images = np.broadcast_to(np.zeros((1, 32, 32, 3), dtype=np.uint8), (len(train_targets), 32, 32, 3))
    paths, shares = make_figures(str(results), str(raw_dir), str(tmp_path / "figures"), train_targets,
                                 tables_dir=str(tmp_path / "tables"), train_images=images)
    assert len(paths) == 13
    for path in paths:
        assert os.path.getsize(path) > 0
    assert set(shares["condition"]) == {"margin", "badge", "random"}


def test_class_count_table(tmp_path):
    # Class 7 has 3 labels in both seeds, class 8 has 2 and 0, class 9 has 1 and 3.
    targets = np.array([7, 7, 7, 8, 8, 9, 9, 9, 9])
    sets = {1: [0, 1, 2, 3, 4, 5], 2: [0, 1, 2, 5, 6, 7]}
    tasks = pd.DataFrame([{"K": 3, "s": s, "class_ids": [7, 8, 9], "classnames": ["a", "b", "c"]} for s in sets])
    for s, idxs in sets.items():
        for strategy in ("margin", "badge"):
            np.save(tmp_path / f"{strategy}_K3_s{s}_B6.npy", np.array(idxs))
    table = class_count_table(str(tmp_path), tasks, targets, K=3, b=2, n=1)
    margin = table[table["strategy"] == "margin"].set_index("group")
    assert margin.loc["most", "classname"] == "a" and margin.loc["most", "mean_labels"] == 3
    assert margin.loc["most", "seeds_in_group"] == 2 and margin.loc["most", "mean_labels_div_b"] == 1.5
    assert margin.loc["least", "classname"] == "b" and margin.loc["least", "mean_labels"] == 1
    assert margin.loc["least", "sd_labels"] == pytest.approx(np.sqrt(2))
    assert margin.loc["least", "seeds_in_group"] == 1


def test_matched_indices(tmp_path):
    # Initial sets are excluded, so class 7 has images 2 and 6 and class 8 has 3, 4 and 8.
    targets = np.array([7, 7, 7, 8, 8, 9, 7, 8, 8, 9])
    for s, initial, labeled in ((1, [0, 1], [0, 1, 2, 3, 4, 5]), (2, [0, 9], [0, 9, 6, 8, 5])):
        np.save(tmp_path / f"margin_K2_s{s}_B20.npy", np.array(initial))
        np.save(tmp_path / f"margin_K2_s{s}_B60.npy", np.array(labeled))
    tasks = pd.DataFrame([{"K": 2, "s": s, "class_ids": [7, 8]} for s in (1, 2)])
    matched = matched_indices(str(tmp_path), tasks, "margin", targets, [8, 7, 8, 7, 8], K=2, b=30)
    assert targets[matched].tolist() == [8, 7, 8, 7, 8]
    assert sorted(matched[[1, 3]]) == [2, 6] and sorted(matched[[0, 2, 4]]) == [3, 4, 8]
    with pytest.raises(ValueError, match="Too few images of class 7"):
        matched_indices(str(tmp_path), tasks, "margin", targets, [7, 7, 7], K=2, b=30)
    assert matched_indices(str(tmp_path), tasks, "margin", targets, [7], K=2, b=30, exclude=[2]).tolist() == [6]


def test_example_picks_sorted(tmp_path):
    # Label counts per class: 0 -> 5, 1 -> 4, 2 -> 3, 3 -> 2. The initial set is empty.
    targets = np.repeat([0, 1, 2, 3], [5, 4, 3, 2])
    tasks = pd.DataFrame([{"K": 4, "s": 1, "class_ids": [0, 1, 2, 3], "classnames": ["a", "b", "c", "d"]}])
    for condition in CONDITIONS:
        np.save(tmp_path / f"{condition}_K4_s1_B8.npy", np.arange(len(targets)))
        np.save(tmp_path / f"{condition}_K4_s1_B40.npy", np.array([], dtype=int))
    n, titles, picks = example_picks(targets, str(tmp_path), tasks, "margin", K=4, b=2)
    assert n == 2 and titles[0] == "a (5)"
    for al, cb in picks.values():
        assert targets[al].tolist() == targets[cb].tolist() and not set(al) & set(cb)
    assert targets[picks["most"][0]].tolist() == [0, 1]
    assert targets[picks["least"][0]].tolist() == [2, 3]


def test_class_count_table_needs_one_class_set(tmp_path):
    tasks = pd.DataFrame([{"K": 2, "s": 1, "class_ids": [0, 1], "classnames": ["a", "b"]},
                          {"K": 2, "s": 2, "class_ids": [0, 2], "classnames": ["a", "c"]}])
    with pytest.raises(ValueError, match="one class set"):
        class_count_table(str(tmp_path), tasks, np.zeros(1, dtype=int), K=2, b=1)
