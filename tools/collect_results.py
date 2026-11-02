"""
Collect the evaluation JSONs of each mode into one tidy CSV, {save_dir}/results_{mode}.csv, with one row per
(K, s, B, condition). All conditions share the init set, so the b = 10 result, evaluated once as "random", is repeated
for all conditions. The column evaluated_as shows which condition was actually trained.
"""
import argparse
import csv
import glob
import json
import os
from typing import Dict, List

from tools.make_queue import ALL_CONDITIONS

COLUMNS = ("K", "s", "B", "b", "condition", "evaluated_as", "mode", "train_seed", "test_accuracy", "macro_f1",
           "class_count_min", "class_count_max", "class_count_std", "class_entropy_norm", "steps_taken",
           "stopped_epoch", "ended_by", "best_val_loss", "train_seconds", "trainer_config", "git_commit", "best_step")


def tidy_rows(results: List[Dict]) -> List[Dict]:
    """One row per (K, s, B, condition), sorted. A condition evaluated at b = 10 keeps its own result."""
    rows = {}
    for result in results:
        row = dict(result, evaluated_as=result["condition"])
        rows[(row["K"], row["s"], row["B"], row["condition"])] = row
    for result in results:
        if result["b"] == 10 and result["condition"] == "random":
            for condition in ALL_CONDITIONS:
                key = (result["K"], result["s"], result["B"], condition)
                if key not in rows:
                    rows[key] = dict(result, condition=condition, evaluated_as="random")
    return [rows[key] for key in sorted(rows)]


def collect_mode(save_dir: str, mode: str) -> int:
    """Write the CSV of one mode and return its number of rows."""
    results = []
    for path in glob.glob(os.path.join(save_dir, "eval", mode, "*.json")):
        with open(path, "r") as f:
            results.append(json.load(f))
    rows = tidy_rows(results)
    with open(os.path.join(save_dir, f"results_{mode}.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="Directory with the eval folder.")
    args = parser.parse_args()

    modes = sorted(os.path.basename(path) for path in glob.glob(os.path.join(args.save_dir, "eval", "*")))
    for evaluation_mode in modes:
        print(f"{evaluation_mode}: {collect_mode(args.save_dir, evaluation_mode)} rows.")
