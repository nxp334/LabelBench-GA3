import os
from typing import Dict

import numpy as np
from torch.utils.data import Dataset, Subset

from LabelBench.dataset.class_subset import RemapOneHot, build_task
from LabelBench.dataset.dataset_impl.cifar100_dataset import get_cifar100_dataset
from LabelBench.dataset.dataset_impl.label_name.classnames import get_classnames
from LabelBench.dataset.feature_extractor import FeatureExtractor
from LabelBench.skeleton.dataset_skeleton import LabelType, register_dataset, TransformDataset


class SubsetFeatureExtractor(FeatureExtractor):
    """
    Keeps embeddings of the full CIFAR-100 dataset on disk, so that all (K, s) tasks share one set of feature files.
    Each file is read at most once per process: the slice of the current task then stays in memory.
    """

    def __init__(self, model_fn, file_name: str, embed_model_config: Dict, full_datasets: Dict[str, Dataset],
                 feature_idxs: Dict[str, np.ndarray]):
        """
        :param Dict full_datasets: Full CIFAR-100 dataset of each of "train", "val" and "test".
        :param Dict feature_idxs: Positions of the task's images in each of those datasets.
        """
        super().__init__(model_fn, file_name, embed_model_config)
        self.full_datasets = full_datasets
        self.feature_idxs = feature_idxs
        self.task_features = {}

    def get_feature(self, dataset, dataset_split, epoch, use_strong):
        # Same rule as the parent: only the train split has strong features and several transform seeds.
        is_train = dataset_split == "train"
        key = (dataset_split, epoch % self.num_transform_seeds if is_train else 0, use_strong and is_train)
        if key not in self.task_features:
            features = super().get_feature(self.full_datasets[dataset_split], dataset_split, epoch, use_strong)
            idxs = self.feature_idxs[dataset_split]
            self.task_features[key] = (features[0][idxs], features[1][idxs]) if isinstance(features, tuple) \
                else features[idxs]
        return self.task_features[key]


def feature_file_prefix(data_dir: str, model_name: str) -> str:
    """Prefix of the feature files of full CIFAR-100. Same files as dataset `cifar100`."""
    folder_name = os.path.join(data_dir, "cifar100")
    os.makedirs(folder_name, exist_ok=True)
    return "{}/cifar100_{}".format(folder_name, model_name)


def get_subset_feature_extractor(model_fn, embed_model_config: Dict, data_dir: str, dataset) -> SubsetFeatureExtractor:
    """Build the feature extractor of an ALDataset created from `cifar100_subset`."""
    feature_idxs = {"train": dataset.train_dataset.orig_idx, "val": dataset.val_dataset.feature_idx,
                    "test": dataset.test_dataset.feature_idx}
    return SubsetFeatureExtractor(model_fn, feature_file_prefix(data_dir, embed_model_config["model_name"]),
                                  embed_model_config, dataset.train_dataset.full_datasets, feature_idxs)


@register_dataset("cifar100_subset", LabelType.MULTI_CLASS)
def get_cifar100_subset_dataset(data_dir, num_classes, class_seed, *args):
    full_train, full_val, full_test, *_ = get_cifar100_dataset(data_dir)
    val_split, test_split = full_val.dataset, full_test.dataset
    train_targets = np.array(full_train.dataset.targets)
    task = build_task(train_targets, np.array(val_split.dataset.targets), np.array(val_split.indices),
                      np.array(test_split.indices), num_classes, class_seed)

    target_transform = RemapOneHot(task.remap, num_classes)
    pool = TransformDataset(Subset(full_train.dataset, task.orig_idx), transform=full_train.get_transform(),
                            target_transform=target_transform)
    val = TransformDataset(Subset(val_split, task.val_positions), transform=full_val.get_transform(),
                           target_transform=target_transform)
    test = TransformDataset(Subset(test_split, task.test_positions), transform=full_test.get_transform(),
                            target_transform=target_transform)

    # ALDataset has no field for these, so they stay on the datasets.
    pool.orig_idx = task.orig_idx
    pool.class_ids = task.class_ids
    pool.all_targets = train_targets
    pool.full_datasets = {"train": full_train, "val": full_val, "test": full_test}
    val.feature_idx = task.val_positions
    test.feature_idx = task.test_positions

    # "cra" is a typo in classnames.py. Only corrected here, so other datasets stay as they are.
    all_names = ["crab" if name == "cra" else name for name in get_classnames("cifar100")]
    classnames = tuple(all_names[c] for c in task.class_ids)

    one_hot = np.eye(num_classes, dtype=np.int64)
    return pool, val, test, one_hot[task.train_labels], one_hot[task.val_labels], one_hot[task.test_labels], \
           num_classes, classnames
