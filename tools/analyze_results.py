"""
Gain decomposition of the results CSV, see docs/ANALYSIS.md. Per (K, s, b) and strategy, with A_AL the AL set, A_CB its
ClassBaseline and A_R the random set: total = A_AL - A_R, composition = A_CB - A_R, selection = A_AL - A_CB, in
percentage points. Gains are averaged over b within each seed first; the seed is the unit of replication.
"""
import argparse
import glob
import itertools
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

STRATEGIES = ("margin", "badge")
CONDITIONS = ("random",) + STRATEGIES + tuple(f"cb-{strategy}" for strategy in STRATEGIES)
GAINS = ("total", "composition", "selection")
REQUIRED_COLUMNS = ("K", "s", "B", "b", "condition", "evaluated_as", "test_accuracy", "macro_f1", "steps_taken",
                    "ended_by", "best_step", "trainer_config")
# Mean total gain in pp below which share_from_composition is not defined.
MIN_TOTAL_FOR_SHARE = 0.5


def load_results(path: str) -> pd.DataFrame:
    """Results CSV with K, s, b, B as int. Raises on missing columns, duplicate rows or several trainer configs."""
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for column in ("K", "s", "b", "B"):
        values = df[column].astype(float)
        if not np.all(values == np.round(values)):
            raise ValueError(f"Column {column} has non-integer values.")
        df[column] = values.astype(int)
    duplicated = df.duplicated(["K", "s", "b", "condition"], keep=False)
    if duplicated.any():
        raise ValueError(f"Duplicate (K, s, b, condition) rows:\n{df.loc[duplicated, ['K', 's', 'b', 'condition']]}")
    configs = df["trainer_config"].unique()
    if len(configs) != 1:
        raise ValueError(f"Expected one trainer_config, found {list(configs)}")
    return df


def decompose(df: pd.DataFrame, metric: str = "test_accuracy", min_b: int = 20) -> pd.DataFrame:
    """One row per (K, s, b, strategy) with the three gains in pp. AL is paired with cb-AL, both with random."""
    wide = df.pivot(index=["K", "s", "b", "B"], columns="condition", values=metric)
    missing = [condition for condition in CONDITIONS if condition not in wide.columns]
    if missing:
        raise ValueError(f"Missing conditions: {missing}")
    incomplete = wide[list(CONDITIONS)].isna().any(axis=1)
    if incomplete.any():
        raise ValueError(f"Missing conditions at (K, s, b, B): {list(wide.index[incomplete])}")
    wide = wide[wide.index.get_level_values("b") >= min_b]
    parts = []
    for strategy in STRATEGIES:
        part = pd.DataFrame({"strategy": strategy, "a_al": wide[strategy], "a_cb": wide[f"cb-{strategy}"],
                             "a_random": wide["random"]})
        part["total"] = 100 * (part["a_al"] - part["a_random"])
        part["composition"] = 100 * (part["a_cb"] - part["a_random"])
        part["selection"] = 100 * (part["a_al"] - part["a_cb"])
        parts.append(part.reset_index())
    dec = pd.concat(parts, ignore_index=True).sort_values(["K", "strategy", "s", "b"], ignore_index=True)
    assert np.allclose(dec["composition"] + dec["selection"], dec["total"]), "composition + selection != total"
    return dec


def per_seed(dec: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Mean over b within each (K, strategy, s)."""
    return dec.groupby(["K", "strategy", "s"], as_index=False)[list(columns)].mean()


def summarize(dec: pd.DataFrame, columns: Sequence[str] = GAINS, ci: float = 0.95) -> pd.DataFrame:
    """Per (K, strategy): mean, sample sd (n - 1) and half-width of the t-interval over the seed means."""
    seeds = per_seed(dec, columns)
    rows = []
    for (K, strategy), group in seeds.groupby(["K", "strategy"]):
        n = len(group)
        row = {"K": K, "strategy": strategy, "n": n}
        for column in columns:
            sd = group[column].std(ddof=1)
            row[f"{column}_mean"] = group[column].mean()
            row[f"{column}_sd"] = sd
            row[f"{column}_ci"] = stats.t.ppf(0.5 + ci / 2, n - 1) * sd / np.sqrt(n)
        if "total" in columns and "composition" in columns:
            row["share_from_composition"] = (row["composition_mean"] / row["total_mean"]
                                             if row["total_mean"] >= MIN_TOTAL_FOR_SHARE else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def relative_error_reduction(dec: pd.DataFrame) -> pd.DataFrame:
    """Gains divided by the error of the random set, 1 - A_R, in %. total is in pp, so pp / (1 - A_R) is %."""
    rel = dec.copy()
    rel["rel_total"] = dec["total"] / (1 - dec["a_random"])
    rel["rel_composition"] = dec["composition"] / (1 - dec["a_random"])
    return rel


def equal_b_table(dec: pd.DataFrame) -> pd.DataFrame:
    """Gains at B values reached by at least two K: per (B, K, strategy), mean and sd over seeds."""
    k_per_b = dec.groupby("B")["K"].nunique()
    shared = dec[dec["B"].isin(k_per_b.index[k_per_b >= 2])]
    table = shared.groupby(["B", "K", "b", "strategy"])[list(GAINS)].agg(["mean", "std"])
    table.columns = [f"{gain}_{stat.replace('std', 'sd')}" for gain, stat in table.columns]
    return table.reset_index()


def stopping_table(df: pd.DataFrame, min_b: int = 20) -> pd.DataFrame:
    """Per K, over trained rows (evaluated_as == condition) with b >= min_b: how training ended."""
    trained = df[(df["evaluated_as"] == df["condition"]) & (df["b"] >= min_b)]
    return trained.groupby("K").agg(
        n_rows=("ended_by", "size"),
        share_early_stopping=("ended_by", lambda ended_by: float(np.mean(ended_by == "early_stopping"))),
        median_steps_taken=("steps_taken", "median"),
        median_best_step=("best_step", "median")).reset_index()


def load_tasks(tasks_dir: str) -> pd.DataFrame:
    """One row per (K, s) from tasks/K<K>_s<s>.json."""
    rows = []
    for path in glob.glob(os.path.join(tasks_dir, "K*_s*.json")):
        match = re.fullmatch(r"K(\d+)_s(\d+)\.json", os.path.basename(path))
        if match is None:
            continue
        with open(path, "r") as f:
            task = json.load(f)
        rows.append({"K": int(match.group(1)), "s": int(match.group(2)), "class_ids": list(task["class_ids"]),
                     "classnames": list(task["classnames"]), "val_size": task["val_size"],
                     "test_size": task["test_size"]})
    if not rows:
        raise ValueError(f"No task files in {tasks_dir}")
    return pd.DataFrame(rows).sort_values(["K", "s"], ignore_index=True)


def check_nested(tasks: pd.DataFrame) -> None:
    """Within each seed, the classes of a smaller K are a subset of the classes of every larger K."""
    for s, group in tasks.groupby("s"):
        group = group.sort_values("K")
        for (K_small, small), (K_large, large) in itertools.combinations(zip(group["K"], group["class_ids"]), 2):
            if not set(small) <= set(large):
                raise ValueError(f"Seed {s}: classes of K={K_small} are not a subset of K={K_large}.")


def task_overlap(tasks: pd.DataFrame) -> pd.DataFrame:
    """Per K and pair of seeds, the number of shared classes."""
    rows = []
    for K, group in tasks.groupby("K"):
        for (s_a, a), (s_b, b) in itertools.combinations(zip(group["s"], group["class_ids"]), 2):
            rows.append({"K": K, "s_a": s_a, "s_b": s_b, "shared_classes": len(set(a) & set(b))})
    return pd.DataFrame(rows)


def overlap_summary(overlap: pd.DataFrame) -> pd.DataFrame:
    """Per K: shared classes of every seed pair as columns, plus min and max."""
    table = overlap.assign(pair=[f"s{a}-s{b}" for a, b in zip(overlap["s_a"], overlap["s_b"])])
    table = table.pivot(index="K", columns="pair", values="shared_classes")
    table["min"] = table.min(axis=1)
    table["max"] = table.max(axis=1)
    return table.reset_index()


def interval_table(summary: pd.DataFrame, columns: Sequence[str], decimals: int = 2) -> pd.DataFrame:
    """mean [lower, upper] strings of the t-interval."""
    table = summary[["K", "strategy"]].copy()
    for column in columns:
        mean, half = summary[f"{column}_mean"], summary[f"{column}_ci"]
        table[column] = [f"{m:.{decimals}f} [{m - h:.{decimals}f}, {m + h:.{decimals}f}]" for m, h in zip(mean, half)]
    return table


def write_table(table: pd.DataFrame, out_dir: str, name: str) -> List[str]:
    """Write a table as CSV and LaTeX, return both paths."""
    paths = [os.path.join(out_dir, f"{name}.csv"), os.path.join(out_dir, f"{name}.tex")]
    table.to_csv(paths[0], index=False)
    table.to_latex(paths[1], index=False, float_format="%.2f")
    return paths


def write_tables(out_dir: str, summary: pd.DataFrame, rel_summary: pd.DataFrame, f1_summary: pd.DataFrame,
                 equal_b: pd.DataFrame, stopping: pd.DataFrame, tasks: Optional[pd.DataFrame]) -> List[str]:
    """All tables of docs/ANALYSIS.md as CSV and LaTeX."""
    os.makedirs(out_dir, exist_ok=True)
    t2 = interval_table(summary, GAINS)
    t2["rel_total_pct"] = [f"{value:.1f}" for value in rel_summary["rel_total_mean"]]
    paths = write_table(t2, out_dir, "T2_decomposition")
    paths += write_table(interval_table(f1_summary, GAINS), out_dir, "A_macro_f1")
    paths += write_table(equal_b, out_dir, "A_equal_B")
    paths += write_table(stopping, out_dir, "A_stopping")
    if tasks is not None:
        paths += write_table(overlap_summary(task_overlap(tasks)), out_dir, "T1_task_overlap")
        class_lists = tasks.assign(class_ids=[" ".join(map(str, ids)) for ids in tasks["class_ids"]],
                                   classnames=[", ".join(names) for names in tasks["classnames"]])
        paths += write_table(class_lists, out_dir, "T1_class_lists")
    return paths


def analyze(results: str, metric: str, min_b: int, ci: float,
            tasks_dir: Optional[str]) -> Tuple[Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    """All summaries of one results CSV, and the task table if tasks_dir is given."""
    df = load_results(results)
    dec = decompose(df, metric, min_b)
    tables = {"summary": summarize(dec, GAINS, ci),
              "relative": summarize(relative_error_reduction(dec), ("rel_total", "rel_composition"), ci),
              "macro_f1": summarize(decompose(df, "macro_f1", min_b), GAINS, ci),
              "equal_b": equal_b_table(dec),
              "stopping": stopping_table(df, min_b)}
    tasks = None
    if tasks_dir is not None:
        tasks = load_tasks(tasks_dir)
        check_nested(tasks)
    return tables, tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Results CSV of tools/collect_results.py.")
    parser.add_argument("--out_dir", type=str, help="Directory for the CSV and LaTeX tables. Without it, only print.")
    parser.add_argument("--metric", type=str, default="test_accuracy", help="Metric of the primary decomposition.")
    parser.add_argument("--min_b", type=int, default=20, help="Smallest b (labels per class) that is used.")
    parser.add_argument("--ci", type=float, default=0.95, help="Level of the t-interval over seeds.")
    parser.add_argument("--tasks_dir", type=str, help="Directory with K<K>_s<s>.json. Default: tasks next to results.")
    args = parser.parse_args()

    tasks_dir = args.tasks_dir
    if tasks_dir is None and os.path.isdir(os.path.join(os.path.dirname(args.results), "tasks")):
        tasks_dir = os.path.join(os.path.dirname(args.results), "tasks")
    tables, task_frame = analyze(args.results, args.metric, args.min_b, args.ci, tasks_dir)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
        for name, frame in tables.items():
            print(f"\n{name}\n{frame.to_string(index=False)}")
        if task_frame is not None:
            print(f"\ntask overlap\n{overlap_summary(task_overlap(task_frame)).to_string(index=False)}")
            print(f"\ntask sizes\n{task_frame[['K', 's', 'val_size', 'test_size']].to_string(index=False)}")
    if args.out_dir is not None:
        for path in write_tables(args.out_dir, tables["summary"], tables["relative"], tables["macro_f1"],
                                 tables["equal_b"], tables["stopping"], task_frame):
            print(f"Wrote {path}")
