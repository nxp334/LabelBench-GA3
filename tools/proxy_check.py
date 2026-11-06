"""
Check whether the proxy trains enough: compare it with logistic regression, trained to convergence on the same
standardised embeddings and labeled set. Logistic regression is purely supervised, while the proxy also uses FlexMatch
on the unlabeled pool, so the proxy can legitimately score higher. Run from the repo root.
"""
import argparse
import os
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from LabelBench.dataset.class_subset import derive_seed, parse_labeled_filename, pool_positions, write_json_atomic
from tools.eval_selection import MODE_CONFIGS, build_trainer, compute_scores, seed_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, required=True, help="Number of classes.")
    parser.add_argument("--s", type=int, required=True, help="Experiment seed.")
    parser.add_argument("--B", type=int, required=True, help="Number of labels.")
    parser.add_argument("--file", type=str, required=True, help="Labeled-set .npy file.")
    parser.add_argument("--save_dir", type=str, required=True, help="Results go to {save_dir}/checks.")
    parser.add_argument("--data_dir", type=str, help="Directory to store dataset.", default="./data")
    parser.add_argument("--trainer_config", type=str, help="Proxy trainer config.",
                        default=MODE_CONFIGS["proxy"]["trainer"])
    parser.add_argument("--max_iter", type=int, help="Iteration limit of logistic regression.", default=10000)
    args = parser.parse_args()

    _, file_K, file_s, file_B = parse_labeled_filename(args.file)
    assert (file_K, file_s, file_B) == (args.K, args.s, args.B), "File does not match (K, s, B)."

    dataset, trainer = build_trainer(args.K, args.s, "proxy", args.data_dir, args.trainer_config)
    dataset.update_labeled_idxs(pool_positions(dataset.train_dataset.orig_idx, np.load(args.file)))
    assert dataset.num_labeled() == args.B, "File does not contain B labels."

    seed_training(derive_seed("train", args.s, args.B))
    model = trainer.train()
    preds, labels, _, _ = trainer._test("test", model)
    proxy_accuracy = compute_scores(np.argmax(preds, axis=-1), np.argmax(labels, axis=-1), args.K)[0]

    # After training the dataset holds the unaugmented embeddings of transform seed 0.
    train_embs, _, test_embs = dataset.get_embedding_datasets()
    labeled = dataset.labeled_idxs()
    classifier = LogisticRegression(max_iter=args.max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(train_embs.get_inputs()[labeled], np.argmax(train_embs.get_labels()[labeled], axis=-1))
    logistic_accuracy = float(np.mean(classifier.predict(test_embs.get_inputs()) ==
                                      np.argmax(test_embs.get_labels(), axis=-1)))

    print(f"K={args.K}, s={args.s}, B={args.B}, optimizer steps={trainer.num_steps_taken}")
    print(f"Proxy test accuracy:               {proxy_accuracy:.4f}")
    print(f"Logistic regression test accuracy: {logistic_accuracy:.4f}")
    write_json_atomic(os.path.join(args.save_dir, "checks", f"proxy_check_K{args.K}_B{args.B}.json"),
                      {"proxy_accuracy": proxy_accuracy, "sklearn_accuracy": logistic_accuracy, "K": args.K,
                       "s": args.s, "B": args.B, "steps": trainer.num_steps_taken})
