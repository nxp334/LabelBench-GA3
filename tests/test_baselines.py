import os

import numpy as np
import pytest

from LabelBench.dataset.class_subset import balanced_init, save_labeled_set, select_classes
from tools.make_baselines import make_baselines

K, S = 10, 1
BS = [100 * i for i in range(1, 6)]


@pytest.fixture
def fake_runs(tmp_path, train_targets) -> str:
    """Fake AL runs that strongly favour some classes, one file per round."""
    class_ids = select_classes(K, S)
    pool = np.where(np.isin(train_targets, class_ids))[0]
    init = balanced_init(train_targets, class_ids, 10, S)
    for strategy, seed in (("margin", 5), ("badge", 6)):
        rnd = np.random.RandomState(seed)
        non_init = np.setdiff1d(pool, init)
        weights = (train_targets[non_init] == class_ids[0]) * 20.0 + 1
        acquired = rnd.choice(non_init, size=BS[-1] - len(init), replace=False, p=weights / weights.sum())
        labeled = np.concatenate((init, acquired))
        for B in BS:
            save_labeled_set(str(tmp_path), strategy, K, S, labeled[:B])
    make_baselines(str(tmp_path), K, S, train_targets)
    return str(tmp_path)


def load(save_dir: str, condition: str, B: int) -> np.ndarray:
    return np.load(os.path.join(save_dir, "acq", f"{condition}_K{K}_s{S}_B{B}.npy"))


def test_random_exists_for_every_B_and_nested(fake_runs):
    for small, large in zip(BS, BS[1:]):
        assert np.array_equal(load(fake_runs, "random", large)[:small], load(fake_runs, "random", small))
    assert len(load(fake_runs, "random", BS[-1])) == BS[-1]


@pytest.mark.parametrize("strategy", ["margin", "badge"])
def test_cb_counts_equal_al_counts(fake_runs, train_targets, strategy):
    for B in BS:
        cb_counts = np.bincount(train_targets[load(fake_runs, f"cb-{strategy}", B)], minlength=100)
        al_counts = np.bincount(train_targets[load(fake_runs, strategy, B)], minlength=100)
        assert np.array_equal(cb_counts, al_counts)


@pytest.mark.parametrize("strategy", ["margin", "badge"])
def test_cb_nested_across_B(fake_runs, strategy):
    for small, large in zip(BS, BS[1:]):
        assert set(load(fake_runs, f"cb-{strategy}", small)) <= set(load(fake_runs, f"cb-{strategy}", large))


@pytest.mark.parametrize("condition", ["random", "cb-margin", "cb-badge"])
def test_init_preserved(fake_runs, train_targets, condition):
    init = balanced_init(train_targets, select_classes(K, S), 10, S)
    for B in BS:
        assert np.array_equal(load(fake_runs, condition, B)[:len(init)], init)


@pytest.mark.parametrize("condition", ["random", "cb-margin", "cb-badge"])
def test_sets_have_distinct_images_of_the_task(fake_runs, train_targets, condition):
    labeled = load(fake_runs, condition, BS[-1])
    assert len(np.unique(labeled)) == BS[-1]
    assert np.all(np.isin(train_targets[labeled], select_classes(K, S)))


def test_all_conditions_share_the_set_at_init(fake_runs):
    sets = [load(fake_runs, condition, BS[0]) for condition in ("margin", "badge", "random", "cb-margin", "cb-badge")]
    assert all(np.array_equal(np.sort(sets[0]), np.sort(other)) for other in sets[1:])


def test_rejects_al_file_without_the_init(tmp_path, train_targets):
    task_pool = np.where(np.isin(train_targets, select_classes(K, S)))[0]
    save_labeled_set(str(tmp_path), "margin", K, S, task_pool[:100])
    with pytest.raises(AssertionError):
        make_baselines(str(tmp_path), K, S, train_targets)
