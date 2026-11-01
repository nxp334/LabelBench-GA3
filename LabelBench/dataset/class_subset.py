"""
Numpy-only logic for the GA3 class-composition experiment.

Notation: K = number of classes, s = experiment seed, B = total labels so far, b = B / K.
Indices called orig_idx are CIFAR-100 train indices; pool positions are indices into a task's pool.

This module must not import torch or other LabelBench modules, so tests can run without a GPU stack.
"""
import hashlib
import json
import os
import re
from typing import Dict, NamedTuple, Sequence, Tuple, Union

import numpy as np

NUM_CIFAR100_CLASSES = 100

WANDB_PROJECT_EVAL = os.environ.get("WANDB_PROJECT_EVAL", "LabelBench - ClassBaseline - eval")

FILENAME_PATTERN = re.compile(r"^(?P<condition>[a-z-]+)_K(?P<K>\d+)_s(?P<s>\d+)_B(?P<B>\d+)\.npy$")


class ClassSubsetTask(NamedTuple):
    """All index arrays of one (K, s) task. Labels are in 0..K-1, in the order of class_ids."""
    class_ids: np.ndarray  # Original CIFAR-100 class ids.
    remap: np.ndarray  # Task label of each original class id, -1 if the class is not in the task.
    orig_idx: np.ndarray  # Ascending CIFAR-100 train index of every pool image.
    train_labels: np.ndarray
    val_positions: np.ndarray  # Positions of the task's images inside the full validation split.
    val_labels: np.ndarray
    test_positions: np.ndarray
    test_labels: np.ndarray


class RemapOneHot:
    """Target transform from a 100-dim one-hot label to the K-dim one-hot label of a task."""

    def __init__(self, remap: np.ndarray, n_class: int):
        self.remap = remap
        self.one_hot = np.eye(n_class, dtype=np.int64)

    def __call__(self, y) -> np.ndarray:
        return self.one_hot[self.remap[int(np.argmax(y))]]


def derive_seed(*parts: Union[int, str]) -> int:
    """Deterministic 32-bit seed from the parts. Python's hash() differs between processes, so it is not used."""
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def select_classes(K: int, s: int) -> np.ndarray:
    """First K classes of a seeded shuffle, so tasks with the same s are nested across K."""
    return np.random.RandomState(s).permutation(NUM_CIFAR100_CLASSES)[:K]


def build_task(train_targets: np.ndarray, test_targets: np.ndarray, val_split: np.ndarray, test_split: np.ndarray,
               K: int, s: int) -> ClassSubsetTask:
    """
    Build the pool and restrict the validation/test splits of one task.

    :param numpy.ndarray train_targets: Original class id of every CIFAR-100 train image.
    :param numpy.ndarray test_targets: Original class id of every CIFAR-100 test image.
    :param numpy.ndarray val_split: Test-set indices of the full validation split.
    :param numpy.ndarray test_split: Test-set indices of the full test split.
    :param int K: Number of classes.
    :param int s: Experiment seed.
    """
    train_targets, test_targets = np.asarray(train_targets), np.asarray(test_targets)
    class_ids = select_classes(K, s)
    remap = np.full(NUM_CIFAR100_CLASSES, -1, dtype=int)
    remap[class_ids] = np.arange(K)

    orig_idx = np.where(np.isin(train_targets, class_ids))[0]
    val_positions = np.where(np.isin(test_targets[val_split], class_ids))[0]
    test_positions = np.where(np.isin(test_targets[test_split], class_ids))[0]
    return ClassSubsetTask(class_ids, remap, orig_idx, remap[train_targets[orig_idx]],
                           val_positions, remap[test_targets[val_split[val_positions]]],
                           test_positions, remap[test_targets[test_split[test_positions]]])


def balanced_init(train_targets: np.ndarray, class_ids: np.ndarray, init_per_class: int,
                  class_seed: int) -> np.ndarray:
    """
    Initial labeled set as orig_idx. Each class only depends on (class_seed, class id), never on K or the strategy.
    """
    train_targets = np.asarray(train_targets)
    chosen = []
    for c in class_ids:
        class_images = np.where(train_targets == c)[0]
        permutation = np.random.RandomState(derive_seed("init", class_seed, int(c))).permutation(len(class_images))
        chosen.append(class_images[permutation[:init_per_class]])
    return np.concatenate(chosen)


def pool_positions(orig_idx: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Map CIFAR-100 train indices to positions in a task's pool. orig_idx must be ascending."""
    query = np.asarray(query, dtype=int)
    positions = np.searchsorted(orig_idx, query)
    assert np.all(positions < len(orig_idx)) and np.all(orig_idx[positions] == query), \
        "Some indices are not in the pool of this task."
    return positions


def class_distribution_summary(counts: np.ndarray) -> Dict[str, float]:
    """Summary of per-class labeled counts. Normalized entropy is 1.0 for a balanced set."""
    counts = np.asarray(counts, dtype=float)
    probabilities = counts[counts > 0] / np.sum(counts)
    entropy = -np.sum(probabilities * np.log(probabilities))
    return {"class_count_min": float(np.min(counts)), "class_count_max": float(np.max(counts)),
            "class_count_std": float(np.std(counts)), "class_entropy_norm": float(entropy / np.log(len(counts)))}


def labeled_filename(condition: str, K: int, s: int, B: int) -> str:
    return f"{condition}_K{K}_s{s}_B{B}.npy"


def parse_labeled_filename(path: str) -> Tuple[str, int, int, int]:
    """Return (condition, K, s, B) of a labeled-set file."""
    match = FILENAME_PATTERN.match(os.path.basename(path))
    assert match is not None, f"Not a labeled-set file: {path}"
    return match["condition"], int(match["K"]), int(match["s"]), int(match["B"])


def save_labeled_set(save_dir: str, condition: str, K: int, s: int, orig_idxs: np.ndarray) -> str:
    """Save a labeled set as {save_dir}/acq/{condition}_K{K}_s{s}_B{B}.npy, keeping the given order."""
    acq_dir = os.path.join(save_dir, "acq")
    os.makedirs(acq_dir, exist_ok=True)
    path = os.path.join(acq_dir, labeled_filename(condition, K, s, len(orig_idxs)))
    np.save(path, np.asarray(orig_idxs, dtype=int))
    return path


def write_json_atomic(path: str, result: Dict) -> None:
    """Write through a temporary file, so a crash cannot leave a partial file that counts as finished."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, path)


def task_file_prefix(save_dir: str, K: int, s: int) -> str:
    """Prefix of the metadata files of a task."""
    return os.path.join(save_dir, "tasks", f"K{K}_s{s}")


def write_task_metadata(save_dir: str, K: int, s: int, class_ids: np.ndarray, classnames: Sequence[str],
                        pool_size: int, val_size: int, test_size: int, init_idxs: np.ndarray) -> None:
    """Write {save_dir}/tasks/K{K}_s{s}.json once. Runs of the same task write identical content."""
    path = task_file_prefix(save_dir, K, s) + ".json"
    if not os.path.exists(path):
        write_json_atomic(path, {"K": K, "s": s, "class_ids": [int(c) for c in class_ids],
                                 "classnames": list(classnames), "pool_size": pool_size, "val_size": val_size,
                                 "test_size": test_size, "init_idxs": [int(i) for i in init_idxs]})


def write_test_labels(save_dir: str, K: int, s: int, labels: np.ndarray) -> None:
    """Write {save_dir}/tasks/K{K}_s{s}_test_labels.npy once. Parallel workers can reach this for the same task."""
    path = task_file_prefix(save_dir, K, s) + "_test_labels.npy"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.{os.getpid()}.tmp.npy"
        np.save(tmp_path, labels)
        os.replace(tmp_path, path)


def eval_json_path(save_dir: str, mode: str, labeled_file: str) -> str:
    """Result file of evaluating a labeled-set file in the given mode."""
    name = os.path.splitext(os.path.basename(labeled_file))[0]
    return os.path.join(save_dir, "eval", mode, name + ".json")
