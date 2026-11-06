"""
Test accuracy of the e2e model's zero-shot head with no labels, for a task (K, s). Sanity check that labels and class
names align: if they do not, accuracy falls to about 1 / K. Run from the repo root.
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from LabelBench.dataset.datasets import get_dataset
from LabelBench.model.model import get_model_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, required=True, help="Number of classes.")
    parser.add_argument("--s", type=int, required=True, help="Experiment seed.")
    parser.add_argument("--data_dir", type=str, help="Directory to store dataset.", default="./data")
    parser.add_argument("--batch_size", type=int, help="Test batch size.", default=1000)
    args = parser.parse_args()

    dataset = get_dataset("cifar100_subset", args.data_dir, num_classes=args.K, class_seed=args.s)
    with open("./configs/model/clip_ViTB32_pretrained.json", "r") as f:
        model_config = json.load(f)
    model_config["num_output"] = args.K

    model = get_model_fn(model_config["model_name"])(model_config)
    model.init_head_withzeroshot(classnames=dataset.get_classnames(), template=model_config["template"])
    model = model.cuda().eval()

    _, _, test_dataset = dataset.get_input_datasets()
    test_dataset.set_transform(model.get_preprocess(split="test"))
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    correct = []
    for img, target in loader:
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = model(img.float().cuda())
        correct.append((logits.argmax(dim=-1).cpu() == target.argmax(dim=-1)).numpy())
    correct = np.concatenate(correct)

    print(f"Classes: {dataset.get_classnames()}")
    print(f"Zero-shot test accuracy K={args.K}, s={args.s}: {np.mean(correct):.4f} "
          f"on {len(correct)} images (chance {1 / args.K:.4f})")
