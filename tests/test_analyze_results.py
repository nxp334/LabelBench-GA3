import json

import numpy as np
import pandas as pd
import pytest

from tools.analyze_results import (CONDITIONS, check_nested, decompose, equal_b_table, load_results, load_tasks,
                                   relative_error_reduction, summarize, task_overlap)

CONFIG = "cifar100/flexmatch/clip_ViTB32_linear_floor500.json"
# Accuracy per condition: margin gains are 2 pp total (1 composition), badge gains 4 pp total (3 composition).
ACCURACY = {"random": 0.90, "margin": 0.92, "cb-margin": 0.91, "badge": 0.94, "cb-badge": 0.93}


def make_rows(Ks=(10,), seeds=(1,), bs=(10, 20, 30), accuracy=None):
    accuracy = accuracy or (lambda K, s, b, condition: ACCURACY[condition])
    rows = []
    for K in Ks:
        for s in seeds:
            for b in bs:
                for condition in CONDITIONS:
                    value = accuracy(K, s, b, condition)
                    rows.append({"K": K, "s": s, "B": K * b, "b": float(b), "condition": condition,
                                 "evaluated_as": condition if b > 10 else "random", "test_accuracy": value,
                                 "macro_f1": value, "steps_taken": 1000, "ended_by": "early_stopping",
                                 "best_step": 500, "trainer_config": CONFIG})
    return pd.DataFrame(rows)


def write_csv(tmp_path, df):
    path = tmp_path / "results.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_load_casts_b_to_int(tmp_path):
    df = load_results(write_csv(tmp_path, make_rows()))
    assert df["b"].dtype.kind == "i" and sorted(df["b"].unique()) == [10, 20, 30]


def test_pairing_and_sum():
    dec = decompose(make_rows(), min_b=20).set_index(["strategy", "b"])
    assert dec.loc[("margin", 20), ["total", "composition", "selection"]].tolist() == pytest.approx([2, 1, 1])
    assert dec.loc[("badge", 20), ["total", "composition", "selection"]].tolist() == pytest.approx([4, 3, 1])
    assert np.allclose(dec["composition"] + dec["selection"], dec["total"])


def test_b10_dropped():
    assert sorted(decompose(make_rows(), min_b=20)["b"].unique()) == [20, 30]


def test_missing_condition_raises():
    df = make_rows()
    df = df[~((df["b"] == 20) & (df["condition"] == "cb-badge"))]
    with pytest.raises(ValueError, match="Missing conditions"):
        decompose(df)


def test_duplicate_rows_raise(tmp_path):
    df = make_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        load_results(write_csv(tmp_path, pd.concat([df, df.iloc[:1]])))


def test_two_trainer_configs_raise(tmp_path):
    df = make_rows()
    df.loc[0, "trainer_config"] = "other.json"
    with pytest.raises(ValueError, match="one trainer_config"):
        load_results(write_csv(tmp_path, df))


def test_mean_over_b_first_and_sd_n_minus_1():
    # Seed s has margin total gain s pp at b = 20 and 3 * s pp at b = 30, so its mean over b is 2 * s.
    def accuracy(K, s, b, condition):
        gain = s / 100 * (1 if b == 20 else 3)
        return 0.5 + (gain if condition in ("margin", "cb-margin") else 0)
    summary = summarize(decompose(make_rows(seeds=(1, 2, 3, 4), accuracy=accuracy))).set_index("strategy")
    assert summary.loc["margin", "total_mean"] == pytest.approx(5.0)
    assert summary.loc["margin", "total_sd"] == pytest.approx(2 * np.std([1, 2, 3, 4], ddof=1))
    assert summary.loc["margin", "total_ci"] == pytest.approx(3.182446 * summary.loc["margin", "total_sd"] / 2)


def test_share_nan_when_total_small():
    def accuracy(K, s, b, condition):
        return {"random": 0.900, "margin": 0.903, "cb-margin": 0.902}.get(condition, ACCURACY.get(condition))
    summary = summarize(decompose(make_rows(seeds=(1, 2), accuracy=accuracy))).set_index("strategy")
    assert np.isnan(summary.loc["margin", "share_from_composition"])
    assert summary.loc["badge", "share_from_composition"] == pytest.approx(0.75)


def test_equal_b_keeps_shared_values():
    # B of K=10: 200, 300; of K=20: 400, 600; of K=30: 600, 900. Only B = 600 is shared.
    table = equal_b_table(decompose(make_rows(Ks=(10, 20, 30), seeds=(1, 2))))
    assert sorted(table["B"].unique()) == [600]
    assert sorted(table["K"].unique()) == [20, 30]


def test_relative_error_reduction():
    rel = relative_error_reduction(decompose(make_rows())).set_index(["strategy", "b"])
    # (0.92 - 0.90) / (1 - 0.90) = 20 %, (0.91 - 0.90) / (1 - 0.90) = 10 %.
    assert rel.loc[("margin", 20), "rel_total"] == pytest.approx(20.0)
    assert rel.loc[("margin", 20), "rel_composition"] == pytest.approx(10.0)


def write_task(tasks_dir, K, s, class_ids):
    with open(tasks_dir / f"K{K}_s{s}.json", "w") as f:
        json.dump({"K": K, "s": s, "class_ids": class_ids, "classnames": [f"c{c}" for c in class_ids],
                   "val_size": 50 * K, "test_size": 50 * K}, f)


def test_tasks_nested_and_overlap(tmp_path):
    write_task(tmp_path, 2, 1, [1, 2])
    write_task(tmp_path, 3, 1, [1, 2, 3])
    write_task(tmp_path, 2, 2, [2, 4])
    write_task(tmp_path, 3, 2, [2, 4, 3])
    tasks = load_tasks(str(tmp_path))
    check_nested(tasks)
    overlap = task_overlap(tasks).set_index("K")["shared_classes"]
    assert overlap.to_dict() == {2: 1, 3: 2}


def test_tasks_not_nested_raise(tmp_path):
    write_task(tmp_path, 2, 1, [1, 2])
    write_task(tmp_path, 3, 1, [1, 3, 4])
    with pytest.raises(ValueError, match="not a subset"):
        check_nested(load_tasks(str(tmp_path)))
