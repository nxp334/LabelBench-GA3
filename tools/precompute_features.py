"""
Compute the CLIP embedding files of full CIFAR-100 once, before parallel runs start. FeatureExtractor writes its files
without locking, so processes that start on an empty cache would overwrite and read half-written files. Files that
exist are not computed again. Run from the repo root.
"""
import argparse
import json

from LabelBench.dataset.dataset_impl.cifar100_dataset import get_cifar100_dataset
from LabelBench.dataset.dataset_impl.cifar100_subset_dataset import feature_file_prefix
from LabelBench.dataset.feature_extractor import FeatureExtractor
from LabelBench.model.model import get_model_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Directory to store dataset.", default="./data")
    parser.add_argument("--embed_model_config", type=str, help="Encoder configuration file.",
                        default="clip_ViTB32.json")
    args = parser.parse_args()

    with open(f"./configs/embed_model/{args.embed_model_config}", "r") as f:
        embed_model_config = json.load(f)
        embed_model_config["num_output"] = 0

    embed_model_fn = get_model_fn(embed_model_config["model_name"])
    file_name = feature_file_prefix(args.data_dir, embed_model_config["model_name"])
    extractor = FeatureExtractor(embed_model_fn, file_name, embed_model_config)
    train_dataset, val_dataset, test_dataset, *_ = get_cifar100_dataset(args.data_dir)

    # FlexMatch reads strong features of every transform seed, and evaluation reads seed 0 without strong transform.
    for seed in range(extractor.num_transform_seeds):
        extractor.get_feature(train_dataset, "train", seed, True)
    extractor.get_feature(train_dataset, "train", 0, False)
    extractor.get_feature(val_dataset, "val", 0, False)
    extractor.get_feature(test_dataset, "test", 0, False)
    print("All feature files are ready.")
