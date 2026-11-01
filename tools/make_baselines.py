"""
Build the offline baselines of one (K, s) from its saved AL runs:
- random: init + a prefix of one fixed permutation of the non-init pool.
- cb-{strategy}: same per-class counts as the AL set, chosen uniformly at random within each class.
"""
import argparse
import glob
import os
from typing import Dict, List, Tuple

import numpy as np

from LabelBench.dataset.class_subset import balanced_init, derive_seed, parse_labeled_filename, save_labeled_set, \
    select_classes

AL_STRATEGIES = ("margin", "badge")


def random_baseline(train_targets: np.ndarray, class_ids: np.ndarray, init: np.ndarray, K: int, s: int,
                    B: int) -> np.ndarray:
    """Random set of B labels as orig_idx. Sets of one (K, s) are nested across B."""
    pool = np.where(np.isin(train_targets, class_ids))[0]
    non_init = np.setdiff1d(pool, init)
    permuted = non_init[np.random.RandomState(derive_seed("random", K, s)).permutation(len(non_init))]
    return np.concatenate((init, permuted[:B - len(init)]))


def class_baseline(train_targets: np.ndarray, class_ids: np.ndarray, init: np.ndarray, al_idxs: np.ndarray, s: int,
                   strategy: str) -> np.ndarray:
    """ClassBaseline of an AL set as orig_idx. Sets of one (s, strategy) are nested across B."""
    chosen = [init]
    for c in class_ids:
        n_c = int(np.sum(train_targets[al_idxs] == c))
        n_init = int(np.sum(train_targets[init] == c))
        non_init = np.setdiff1d(np.where(train_targets == c)[0], init)
        permutation = np.random.RandomState(derive_seed("cb", s, int(c), strategy)).permutation(len(non_init))
        chosen.append(non_init[permutation[:n_c - n_init]])
    baseline = np.concatenate(chosen)

    assert np.array_equal(np.bincount(train_targets[baseline], minlength=100),
                          np.bincount(train_targets[al_idxs], minlength=100)), "Per-class counts differ from AL set."
    return baseline


def find_al_files(save_dir: str, K: int, s: int) -> Dict[Tuple[str, int], str]:
    """Map (strategy, B) to the saved AL file."""
    files = {}
    for strategy in AL_STRATEGIES:
        for path in glob.glob(os.path.join(save_dir, "acq", f"{strategy}_K{K}_s{s}_B*.npy")):
            files[(strategy, parse_labeled_filename(path)[3])] = path
    return files


def make_baselines(save_dir: str, K: int, s: int, train_targets: np.ndarray, init_per_class: int = 10) -> List[str]:
    """Write all baseline files of (K, s) next to the AL files and return their paths."""
    train_targets = np.asarray(train_targets)
    al_files = find_al_files(save_dir, K, s)
    assert al_files, f"No AL files for K={K}, s={s} in {save_dir}/acq."

    class_ids = select_classes(K, s)
    init = balanced_init(train_targets, class_ids, init_per_class, s)

    written = []
    for (strategy, B), path in sorted(al_files.items()):
        al_idxs = np.load(path)
        assert len(al_idxs) == B, f"{path} does not contain {B} labels."
        assert np.array_equal(al_idxs[:len(init)], init), f"{path} does not start with the balanced init."
        written.append(save_labeled_set(save_dir, f"cb-{strategy}", K, s,
                                        class_baseline(train_targets, class_ids, init, al_idxs, s, strategy)))
    for B in sorted({B for _, B in al_files}):
        written.append(save_labeled_set(save_dir, "random", K, s,
                                        random_baseline(train_targets, class_ids, init, K, s, B)))
    return written


if __name__ == "__main__":
    from torchvision.datasets import CIFAR100

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="Directory with the acq folder.")
    parser.add_argument("--K", type=int, required=True, help="Number of classes.")
    parser.add_argument("--s", type=int, required=True, help="Experiment seed.")
    parser.add_argument("--data_dir", type=str, help="Directory to store dataset.", default="./data")
    parser.add_argument("--init_per_class", type=int, help="Initial labels per class.", default=10)
    args = parser.parse_args()

    targets = np.array(CIFAR100(args.data_dir, train=True, download=True).targets)
    paths = make_baselines(args.save_dir, args.K, args.s, targets, args.init_per_class)
    print(f"Wrote {len(paths)} baseline files to {os.path.join(args.save_dir, 'acq')}.")
