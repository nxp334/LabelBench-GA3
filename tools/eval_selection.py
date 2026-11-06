"""
Evaluate one saved labeled set with a fresh model, in mode e2e (FlexMatch fine-tune of CLIP) or proxy (FlexMatch linear
on frozen CLIP embeddings). Run from the repo root.
"""
import argparse
import json
import os
import random
import subprocess
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score

from LabelBench.dataset.class_subset import WANDB_PROJECT_EVAL, class_distribution_summary, derive_seed, \
    eval_json_path, parse_labeled_filename, pool_positions, write_json_atomic, write_test_labels

MODE_CONFIGS = {
    "e2e": {"embed": "none.json", "model": "clip_ViTB32_pretrained.json",
            "trainer": "cifar100/flexmatch/clip_ViTB32_finetune.json"},
    "proxy": {"embed": "clip_ViTB32.json", "model": "linear.json",
              "trainer": "cifar100/flexmatch/clip_ViTB32_linear_minsteps.json"},
}


def compute_scores(y_pred: np.ndarray, y_true: np.ndarray, K: int) -> Tuple[float, float, List[float]]:
    """Accuracy, macro-F1 and per-class accuracy from predicted and true class labels."""
    class_counts = np.bincount(y_true, minlength=K)
    class_correct = np.bincount(y_true[y_pred == y_true], minlength=K)
    per_class_accuracy = class_correct / np.maximum(class_counts, 1)
    return float(np.mean(y_pred == y_true)), float(f1_score(y_true, y_pred, average="macro")), \
           [float(accuracy) for accuracy in per_class_accuracy]


def git_commit() -> str:
    """Hash of the code that produced a result."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_trainer(K: int, s: int, mode: str, data_dir: str, trainer_config_name: str):
    """Build the dataset and a trainer of the given mode for task (K, s). Returns (dataset, trainer)."""
    from LabelBench.corrupter.corrupter import get_corrupter_fn
    from LabelBench.dataset.datasets import get_dataset
    from LabelBench.dataset.dataset_impl.cifar100_subset_dataset import get_subset_feature_extractor
    from LabelBench.metric.metrics import get_metric
    from LabelBench.model.model import get_model_fn
    from LabelBench.trainer.trainer import get_fns, get_optimizer_fn, get_scheduler_fn, get_trainer

    configs = MODE_CONFIGS[mode]
    with open(os.path.join("./configs/model", configs["model"]), "r") as f:
        classifier_model_config = json.load(f)
    with open(os.path.join("./configs/embed_model", configs["embed"]), "r") as f:
        embed_model_config = json.load(f)
        embed_model_config["num_output"] = 0
    with open(os.path.join("./configs/trainer", trainer_config_name), "r") as f:
        trainer_config = json.load(f)
    with open("./configs/corrupter/noiseless.json", "r") as f:
        corrupter_config = json.load(f)

    dataset = get_dataset("cifar100_subset", data_dir, num_classes=K, class_seed=s)
    classifier_model_config["num_output"] = K
    dataset.set_corrupter(get_corrupter_fn(corrupter_config["name"], corrupter_config))

    feature_extractor = None
    if embed_model_config["model_name"] != "none":
        embed_model_fn = get_model_fn(embed_model_config["model_name"])
        classifier_model_config["input_dim"] = embed_model_fn(embed_model_config).get_embedding_dim()
        feature_extractor = get_subset_feature_extractor(embed_model_fn, embed_model_config, data_dir, dataset)

    model_fn = get_model_fn(classifier_model_config["model_name"])
    trainer_config = get_fns(trainer_config)
    trainer_config = get_optimizer_fn(trainer_config)
    trainer_config = get_scheduler_fn(trainer_config)
    trainer = get_trainer(trainer_config["trainer_name"], trainer_config, dataset, model_fn, classifier_model_config,
                          get_metric("multi_class"), feature_extractor)
    return dataset, trainer


def seed_training(seed: int) -> None:
    """Seed every generator used in training. RandAugment uses both `random` and numpy."""
    import torch

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_and_test(orig_idxs: np.ndarray, K: int, s: int, train_seed: int, mode: str, data_dir: str,
                   trainer_config_name: str) -> Dict:
    """
    Train a fresh model on the labeled set and predict the test set. Skips the full-pool pass of main.py, since only
    test results are needed.
    """
    dataset, trainer = build_trainer(K, s, mode, data_dir, trainer_config_name)
    dataset.update_labeled_idxs(pool_positions(dataset.train_dataset.orig_idx, orig_idxs))

    seed_training(train_seed)
    train_start = time.time()
    model = trainer.train()
    train_seconds = time.time() - train_start
    preds, labels, _, _ = trainer._test("test", model)

    return {"test_predictions": np.argmax(preds, axis=-1), "test_labels": np.argmax(labels, axis=-1),
            "steps_taken": int(trainer.num_steps_taken), "stopped_epoch": trainer.stopped_epoch,
            "ended_by": trainer.ended_by, "best_step": trainer.best_step, "best_val_loss": trainer.best_val_loss,
            "train_seconds": train_seconds,
            "per_class_counts": [int(n) for n in np.sum(dataset.get_train_labels(), axis=0)]}


def log_wandb(result: Dict, wandb_name: str, wandb_project: str) -> None:
    """Log one result, with the experiment fields as config."""
    import wandb

    config_keys = ("K", "s", "B", "b", "condition", "mode", "train_seed", "trainer_config", "git_commit", "ended_by")
    K, s, B, condition, mode = (result[key] for key in ("K", "s", "B", "condition", "mode"))
    wandb.init(project=wandb_project, entity=wandb_name, config={key: result[key] for key in config_keys},
               name=f"{condition}_K{K}_s{s}_B{B}", job_type=f"eval-{mode}", group=f"K{K}_s{s}",
               tags=[condition, mode, f"K{K}"])
    wandb.log({key: result[key] for key in ("test_accuracy", "macro_f1", "class_count_min", "class_count_max",
                                            "class_count_std", "class_entropy_norm", "runtime_seconds",
                                            "train_seconds", "steps_taken")})
    wandb.run.summary["per_class_counts"] = result["per_class_counts"]
    wandb.finish()


def evaluate_file(file: str, mode: str, save_dir: str, data_dir: str, trainer_config: Optional[str] = None,
                  train_fn: Callable[..., Dict] = train_and_test,
                  log_fn: Optional[Callable[[Dict], None]] = None) -> Optional[Dict]:
    """
    Evaluate one labeled-set file unless its result exists, and return the new result.

    :param Optional[str] trainer_config: Overrides the trainer config of the mode.
    :param train_fn: Function that trains and tests a fresh model.
    :param log_fn: Function that logs a new result.
    """
    json_path = eval_json_path(save_dir, mode, file)
    if os.path.exists(json_path):
        print(f"Skipping {file}: {json_path} exists.")
        return None

    condition, K, s, B = parse_labeled_filename(file)
    orig_idxs = np.load(file)
    assert len(orig_idxs) == B and len(np.unique(orig_idxs)) == B, f"{file} does not contain {B} distinct labels."
    if trainer_config is None:
        trainer_config = MODE_CONFIGS[mode]["trainer"]

    train_seed = derive_seed("train", s, B)
    start_time = time.time()
    outcome = train_fn(orig_idxs, K, s, train_seed, mode, data_dir, trainer_config)
    predictions, test_labels = outcome.pop("test_predictions"), outcome.pop("test_labels")
    test_accuracy, macro_f1, per_class_accuracy = compute_scores(predictions, test_labels, K)
    result = dict(outcome, test_accuracy=test_accuracy, macro_f1=macro_f1, per_class_test_accuracy=per_class_accuracy,
                  runtime_seconds=time.time() - start_time, K=K, s=s, B=B, b=B / K, condition=condition, mode=mode,
                  train_seed=train_seed, trainer_config=trainer_config, file=os.path.basename(file),
                  git_commit=git_commit(), **class_distribution_summary(outcome["per_class_counts"]))

    # The JSON is written last: its existence marks the job as finished.
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    np.save(os.path.join(os.path.dirname(json_path), "preds_" + os.path.basename(file)), predictions.astype(int))
    write_test_labels(save_dir, K, s, test_labels.astype(int))
    write_json_atomic(json_path, result)
    if log_fn is not None:
        log_fn(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Labeled-set .npy file.")
    parser.add_argument("--mode", type=str, required=True, choices=sorted(MODE_CONFIGS), help="Evaluation mode.")
    parser.add_argument("--save_dir", type=str, required=True, help="Results go to {save_dir}/eval/{mode}.")
    parser.add_argument("--data_dir", type=str, help="Directory to store dataset.", default="./data")
    parser.add_argument("--trainer_config", type=str, help="Overrides the trainer config of the mode.")
    parser.add_argument("--wandb_name", type=str, help="Wandb entity.", default="ATDL-labelbench")
    parser.add_argument("--wandb_project", type=str, help="Wandb project.", default=WANDB_PROJECT_EVAL)
    args = parser.parse_args()

    evaluate_file(args.file, args.mode, args.save_dir, args.data_dir, args.trainer_config,
                  log_fn=lambda result: log_wandb(result, args.wandb_name, args.wandb_project))
