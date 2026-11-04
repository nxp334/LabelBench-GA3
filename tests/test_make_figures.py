import json
import os

import numpy as np
import pandas as pd

from tools.analyze_results import CONDITIONS
from tools.make_figures import make_figures


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
                    if b in (100, 200):
                        pool = np.where(np.isin(train_targets, class_ids))[0]
                        np.save(raw_dir / "acq" / f"{condition}_K{K}_s{s}_B{K * b}.npy",
                                rng.choice(pool, K * b, replace=False))
    results = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(results, index=False)

    paths, shares = make_figures(str(results), str(raw_dir), str(tmp_path / "figures"), train_targets)
    assert len(paths) == 6
    for path in paths:
        assert os.path.getsize(path) > 0
    assert set(shares["condition"]) == {"margin", "badge", "random"}
