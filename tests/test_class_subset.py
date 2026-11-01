import json
from typing import Tuple

import numpy as np
import pytest

from LabelBench.dataset.class_subset import RemapOneHot, balanced_init, build_task, class_distribution_summary, \
    labeled_filename, parse_labeled_filename, pool_positions, select_classes, write_task_metadata, write_test_labels

KS = (10, 20, 50, 100)


def splits() -> Tuple[np.ndarray, np.ndarray]:
    """The stock validation and test splits of 10000 test images."""
    idxs = np.random.RandomState(42).permutation(10000)
    return idxs[:5000], idxs[5000:]


@pytest.mark.parametrize("s", [1, 2, 3])
def test_classes_nested_across_K(s):
    for small, large in zip(KS, KS[1:]):
        assert np.array_equal(select_classes(large, s)[:small], select_classes(small, s))


@pytest.mark.parametrize("K", KS)
def test_labels_align_with_class_order(K, train_targets, test_targets):
    names = np.array([f"class{c}" for c in range(100)])
    val_split, test_split = splits()
    task = build_task(train_targets, test_targets, val_split, test_split, K, 1)
    task_names = names[task.class_ids]
    assert np.array_equal(task_names[task.train_labels], names[train_targets[task.orig_idx]])
    assert np.array_equal(task_names[task.val_labels], names[test_targets[val_split[task.val_positions]]])
    assert np.array_equal(task_names[task.test_labels], names[test_targets[test_split[task.test_positions]]])


def test_remap_one_hot_gives_task_label(train_targets, test_targets):
    task = build_task(train_targets, test_targets, *splits(), 10, 1)
    transform = RemapOneHot(task.remap, 10)
    for label, class_id in enumerate(task.class_ids):
        assert np.argmax(transform(np.eye(100)[class_id])) == label
        assert transform(np.eye(100)[class_id]).shape == (10,)


@pytest.mark.parametrize("K", KS)
def test_pool_has_all_images_of_selected_classes(K, train_targets, test_targets):
    task = build_task(train_targets, test_targets, *splits(), K, 1)
    assert len(task.orig_idx) == 500 * K
    assert np.array_equal(np.bincount(task.train_labels), np.full(K, 500))
    assert np.all(np.diff(task.orig_idx) > 0)


@pytest.mark.parametrize("K", KS)
def test_pool_val_test_disjoint(K, train_targets, test_targets):
    val_split, test_split = splits()
    task = build_task(train_targets, test_targets, val_split, test_split, K, 2)
    val_idxs, test_idxs = val_split[task.val_positions], test_split[task.test_positions]
    # The pool comes from the official train set, so it can only overlap with itself. Validation and test both come
    # from the official test set, and together they cover all its images of the task.
    assert len(np.unique(task.orig_idx)) == len(task.orig_idx)
    assert len(np.intersect1d(val_idxs, test_idxs)) == 0
    assert len(val_idxs) + len(test_idxs) == 100 * K
    assert set(task.val_labels) | set(task.test_labels) == set(range(K))


@pytest.mark.parametrize("K", KS)
def test_init_has_exactly_10_per_class(K, train_targets):
    class_ids = select_classes(K, 1)
    init = balanced_init(train_targets, class_ids, 10, 1)
    assert len(np.unique(init)) == 10 * K
    assert np.array_equal(np.bincount(train_targets[init], minlength=100)[class_ids], np.full(K, 10))
    assert np.sum(np.bincount(train_targets[init], minlength=100)) == 10 * K


def test_init_identical_across_K_for_shared_classes(train_targets):
    init_10 = balanced_init(train_targets, select_classes(10, 1), 10, 1)
    init_100 = balanced_init(train_targets, select_classes(100, 1), 10, 1)
    assert np.array_equal(init_100[:100], init_10)


def test_init_identical_across_calls_and_varies_with_seed(train_targets):
    # Strategies get the same init because the function has no input that depends on them.
    class_ids = select_classes(10, 1)
    assert np.array_equal(balanced_init(train_targets, class_ids, 10, 1), balanced_init(train_targets, class_ids, 10, 1))
    assert not np.array_equal(balanced_init(train_targets, class_ids, 10, 1),
                              balanced_init(train_targets, class_ids, 10, 2))


def test_pool_positions_roundtrip_and_rejects_foreign_index(train_targets, test_targets):
    task = build_task(train_targets, test_targets, *splits(), 10, 1)
    positions = np.array([0, 17, 4999])
    assert np.array_equal(pool_positions(task.orig_idx, task.orig_idx[positions]), positions)
    outside = np.setdiff1d(np.arange(50000), task.orig_idx)[:1]
    with pytest.raises(AssertionError):
        pool_positions(task.orig_idx, outside)


def test_class_distribution_summary():
    balanced = class_distribution_summary([10, 10, 10, 10])
    assert balanced["class_entropy_norm"] == pytest.approx(1.0) and balanced["class_count_std"] == 0

    skewed = class_distribution_summary([0, 0, 10, 30])
    assert (skewed["class_count_min"], skewed["class_count_max"]) == (0, 30)
    assert skewed["class_count_std"] == pytest.approx(np.std([0, 0, 10, 30]))
    expected = -(0.25 * np.log(0.25) + 0.75 * np.log(0.75)) / np.log(4)
    assert skewed["class_entropy_norm"] == pytest.approx(expected)


def test_task_files_are_written_once(tmp_path):
    save_dir = str(tmp_path)
    write_task_metadata(save_dir, 2, 1, np.array([7, 3]), ("a", "b"), 1000, 90, 110, np.arange(4))
    write_task_metadata(save_dir, 2, 1, np.array([0, 0]), ("x", "y"), 0, 0, 0, np.arange(1))
    metadata = json.loads((tmp_path / "tasks" / "K2_s1.json").read_text())
    assert metadata["class_ids"] == [7, 3] and metadata["classnames"] == ["a", "b"]
    assert (metadata["pool_size"], metadata["val_size"], metadata["test_size"]) == (1000, 90, 110)
    assert metadata["init_idxs"] == [0, 1, 2, 3]

    write_test_labels(save_dir, 2, 1, np.array([0, 1]))
    write_test_labels(save_dir, 2, 1, np.array([1, 1]))
    assert np.array_equal(np.load(tmp_path / "tasks" / "K2_s1_test_labels.npy"), [0, 1])
    assert sorted(path.name for path in (tmp_path / "tasks").iterdir()) == ["K2_s1.json", "K2_s1_test_labels.npy"]


def test_filename_roundtrip():
    name = labeled_filename("cb-margin", 50, 3, 2000)
    assert name == "cb-margin_K50_s3_B2000.npy"
    assert parse_labeled_filename("/x/acq/" + name) == ("cb-margin", 50, 3, 2000)
