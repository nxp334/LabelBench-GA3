import json
import os

import numpy as np
import pytest

from LabelBench.dataset.class_subset import derive_seed, save_labeled_set
from tools.eval_selection import MODE_CONFIGS, compute_scores, evaluate_file


class StubTrainer:
    """Stands in for training, and records the calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, orig_idxs, K, s, train_seed, mode, data_dir, trainer_config):
        self.calls.append((K, s, train_seed, mode, trainer_config))
        return {"test_predictions": np.array([0, 1, 1, 1]), "test_labels": np.array([0, 0, 1, 1]), "steps_taken": 500,
                "stopped_epoch": 34, "best_val_loss": 1.5, "train_seconds": 1.0, "per_class_counts": [10] * K}


def load_json(save_dir, mode, name):
    with open(os.path.join(save_dir, "eval", mode, name + ".json"), "r") as f:
        return json.load(f)


def test_scores():
    accuracy, macro_f1, per_class = compute_scores(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 0]), 3)
    assert accuracy == 0.75
    assert macro_f1 == pytest.approx((0.8 + 2 / 3) / 2)
    assert per_class == pytest.approx([2 / 3, 1, 0])


def test_writes_all_fields_and_uses_default_config(tmp_path):
    save_dir, stub = str(tmp_path), StubTrainer()
    file = save_labeled_set(save_dir, "cb-margin", 10, 1, np.arange(200))
    evaluate_file(file, "proxy", save_dir, "./data", train_fn=stub)

    result = load_json(save_dir, "proxy", "cb-margin_K10_s1_B200")
    assert (result["K"], result["s"], result["B"], result["b"]) == (10, 1, 200, 20)
    assert (result["condition"], result["mode"]) == ("cb-margin", "proxy")
    assert result["test_accuracy"] == 0.75 and result["per_class_test_accuracy"][:2] == [0.5, 1.0]
    assert result["steps_taken"] == 500 and result["per_class_counts"] == [10] * 10
    assert (result["stopped_epoch"], result["best_val_loss"]) == (34, 1.5)
    assert result["train_seconds"] == 1.0 and result["runtime_seconds"] >= 0
    assert result["class_entropy_norm"] == pytest.approx(1.0) and result["class_count_std"] == 0
    assert result["train_seed"] == derive_seed("train", 1, 200) and "git_commit" in result
    assert "test_predictions" not in result and "test_labels" not in result
    assert stub.calls[0][4] == MODE_CONFIGS["proxy"]["trainer"]
    assert "minsteps" in MODE_CONFIGS["proxy"]["trainer"] and "minsteps" not in MODE_CONFIGS["e2e"]["trainer"]


def test_saves_predictions_and_test_labels_that_reproduce_the_accuracy(tmp_path):
    save_dir = str(tmp_path)
    file = save_labeled_set(save_dir, "random", 10, 1, np.arange(200))
    evaluate_file(file, "proxy", save_dir, "./data", train_fn=StubTrainer())

    predictions = np.load(os.path.join(save_dir, "eval", "proxy", "preds_random_K10_s1_B200.npy"))
    labels = np.load(os.path.join(save_dir, "tasks", "K10_s1_test_labels.npy"))
    assert predictions.dtype.kind == "i" and labels.dtype.kind == "i"
    assert np.mean(predictions == labels) == load_json(save_dir, "proxy", "random_K10_s1_B200")["test_accuracy"]


def test_skips_existing_json(tmp_path):
    save_dir, stub = str(tmp_path), StubTrainer()
    file = save_labeled_set(save_dir, "random", 10, 1, np.arange(200))
    evaluate_file(file, "e2e", save_dir, "./data", train_fn=stub)
    assert evaluate_file(file, "e2e", save_dir, "./data", train_fn=stub) is None
    assert len(stub.calls) == 1


def test_training_seed_depends_on_s_and_B_only(tmp_path):
    save_dir, stub = str(tmp_path), StubTrainer()
    evaluate_file(save_labeled_set(save_dir, "margin", 10, 1, np.arange(200)), "e2e", save_dir, ".", train_fn=stub)
    evaluate_file(save_labeled_set(save_dir, "random", 10, 1, np.arange(1, 201)), "e2e", save_dir, ".", train_fn=stub)
    evaluate_file(save_labeled_set(save_dir, "random", 20, 1, np.arange(200)), "e2e", save_dir, ".", train_fn=stub)
    seeds = {call[2] for call in stub.calls}
    assert len(stub.calls) == 3 and seeds == {derive_seed("train", 1, 200)}


def test_rejects_file_with_repeated_images(tmp_path):
    save_dir = str(tmp_path)
    file = save_labeled_set(save_dir, "margin", 10, 1, np.zeros(100, dtype=int))
    with pytest.raises(AssertionError):
        evaluate_file(file, "e2e", save_dir, "./data", train_fn=StubTrainer())
