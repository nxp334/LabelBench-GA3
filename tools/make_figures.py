"""
Figures of docs/ANALYSIS.md as PNG. Accuracy and gains come from the results CSV; the class composition figures read
the labeled sets acq/<condition>_K<K>_s<s>_B<B>.npy and the task files of the raw output directory, plus the CIFAR-100
train targets.
"""
import argparse
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from LabelBench.dataset.class_subset import labeled_filename  # noqa: E402
from tools.analyze_results import GAINS, STRATEGIES, decompose, load_results, load_tasks, summarize  # noqa: E402

DPI = 200
CONDITION_STYLE = {"random": ("#7f7f7f", ":"), "cb": ("#ff7f0e", "--"), "al": ("#1f77b4", "-")}
GAIN_STYLE = {"total": ("#333333", "o"), "composition": ("#ff7f0e", "s"), "selection": ("#1f77b4", "^")}


def k_colors(Ks) -> Dict[int, tuple]:
    """One blue ramp, light for the smallest K, dark for the largest."""
    return dict(zip(sorted(Ks), plt.cm.Blues(np.linspace(0.4, 1.0, len(Ks)))))


def save(fig, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def seed_stats(df: pd.DataFrame, condition: str, K: int, metric: str = "test_accuracy") -> pd.DataFrame:
    """Per b: mean, min and max over seeds, in %."""
    rows = df[(df["condition"] == condition) & (df["K"] == K)]
    return (100 * rows.groupby("b")[metric].agg(["mean", "min", "max"])).reset_index()


def fig_accuracy_vs_b(df: pd.DataFrame, out_dir: str) -> str:
    """Rows strategy, columns K; random, cb-AL and AL as seed mean with a min-max band."""
    Ks = sorted(df["K"].unique())
    fig, axes = plt.subplots(len(STRATEGIES), len(Ks), figsize=(3.4 * len(Ks), 3.0 * len(STRATEGIES)),
                             sharex=True, squeeze=False)
    for row, strategy in enumerate(STRATEGIES):
        for col, K in enumerate(Ks):
            ax = axes[row, col]
            for key, condition, label in (("random", "random", "random"), ("cb", f"cb-{strategy}", "cb-AL"),
                                          ("al", strategy, "AL")):
                color, style = CONDITION_STYLE[key]
                curve = seed_stats(df, condition, K)
                ax.plot(curve["b"], curve["mean"], color=color, linestyle=style, label=label)
                ax.fill_between(curve["b"], curve["min"], curve["max"], color=color, alpha=0.15, linewidth=0)
            ax.set_title(f"{strategy}, K={K}", fontsize=10)
            ax.grid(alpha=0.3)
            if row == len(STRATEGIES) - 1:
                ax.set_xlabel("b (labels per class)")
            if col == 0:
                ax.set_ylabel("test accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.tight_layout()
    return save(fig, out_dir, "fig1_accuracy_vs_b.png")


def fig_decomposition(dec: pd.DataFrame, summary: pd.DataFrame, out_dir: str) -> str:
    """Per strategy: the three gains per K, mean with t-interval and the per-seed means as jittered points."""
    Ks = sorted(dec["K"].unique())
    seeds = dec.groupby(["K", "strategy", "s"], as_index=False)[list(GAINS)].mean()
    rng = np.random.RandomState(0)
    offsets = dict(zip(GAINS, (-0.25, 0.0, 0.25)))
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(5.5 * len(STRATEGIES), 3.8), sharey=True, squeeze=False)
    for ax, strategy in zip(axes[0], STRATEGIES):
        for gain in GAINS:
            color, marker = GAIN_STYLE[gain]
            rows = summary[summary["strategy"] == strategy].set_index("K").loc[Ks]
            x = np.arange(len(Ks)) + offsets[gain]
            ax.errorbar(x, rows[f"{gain}_mean"], yerr=rows[f"{gain}_ci"], fmt=marker, color=color, capsize=3,
                        markersize=6, label=f"{gain} (mean, 95% CI)")
            for i, K in enumerate(Ks):
                points = seeds[(seeds["strategy"] == strategy) & (seeds["K"] == K)][gain]
                # Points to the right of the interval, so they do not hide behind it.
                jitter = 0.09 + rng.uniform(-0.03, 0.03, len(points))
                ax.scatter(x[i] + jitter, points, s=10, color=color, alpha=0.5, marker=marker, linewidths=0)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(Ks)))
        ax.set_xticklabels([f"K={K}" for K in Ks])
        ax.set_title(strategy, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].set_ylabel("gain (pp)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout()
    return save(fig, out_dir, "fig2_decomposition.png")


def class_composition(acq_dir: str, tasks: pd.DataFrame, targets: np.ndarray, condition: str, K: int,
                      b: int) -> Dict[str, np.ndarray]:
    """q_k = c_k / b of every seed, sorted descending. Returns the seed mean per rank and the share with q < 0.5."""
    sorted_q, shares = [], []
    for _, task in tasks[tasks["K"] == K].iterrows():
        idxs = np.load(os.path.join(acq_dir, labeled_filename(condition, K, task["s"], K * b)))
        classes = np.asarray(targets)[idxs]
        q = np.array([np.sum(classes == c) for c in task["class_ids"]]) / b
        sorted_q.append(np.sort(q)[::-1])
        shares.append(np.mean(q < 0.5))
    return {"q": np.mean(sorted_q, axis=0), "share_below_half": float(np.mean(shares))}


def fig_class_composition(acq_dir: str, tasks: pd.DataFrame, targets: np.ndarray, b: int, out_dir: str,
                          name: str) -> (str, pd.DataFrame):
    """Panels margin and badge: sorted q per K (seed mean), random dotted, balanced at 1.0."""
    Ks = sorted(tasks["K"].unique())
    colors = k_colors(Ks)
    shares = []
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(5.5 * len(STRATEGIES), 3.8), sharey=True, squeeze=False)
    for ax, strategy in zip(axes[0], STRATEGIES):
        for K in Ks:
            x = (np.arange(1, K + 1) - 0.5) / K
            for condition, style, label in ((strategy, "-", f"{strategy} K={K}"), ("random", ":", f"random K={K}")):
                result = class_composition(acq_dir, tasks, targets, condition, K, b)
                ax.plot(x, result["q"], color=colors[K], linestyle=style, label=label)
                if ax is axes[0, 0] or condition != "random":
                    shares.append({"condition": condition, "K": K, "b": b,
                                   "share_q_below_0.5": result["share_below_half"]})
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_title(f"{strategy}, b={b}", fontsize=10)
        ax.set_xlabel("class rank (rank - 0.5) / K")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="upper right", frameon=False)
    axes[0, 0].set_ylabel("q = c / b (1.0 = balanced)")
    fig.tight_layout()
    return save(fig, out_dir, name), pd.DataFrame(shares)


def fig_accuracy_all_k(df: pd.DataFrame, out_dir: str) -> str:
    """One panel per strategy: AL solid and random dotted per K, seed mean."""
    Ks = sorted(df["K"].unique())
    colors = k_colors(Ks)
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(5.5 * len(STRATEGIES), 3.8), sharey=True, squeeze=False)
    for ax, strategy in zip(axes[0], STRATEGIES):
        for K in Ks:
            for condition, style, label in ((strategy, "-", f"{strategy} K={K}"), ("random", ":", f"random K={K}")):
                curve = seed_stats(df, condition, K)
                ax.plot(curve["b"], curve["mean"], color=colors[K], linestyle=style, label=label)
        ax.set_title(strategy, fontsize=10)
        ax.set_xlabel("b (labels per class)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="lower right", frameon=False)
    axes[0, 0].set_ylabel("test accuracy (%)")
    fig.tight_layout()
    return save(fig, out_dir, "figA2_accuracy_all_K.png")


def fig_gain_vs_b(dec: pd.DataFrame, out_dir: str) -> str:
    """Rows total and composition, columns strategy: seed mean of the gain per b, one line per K."""
    Ks = sorted(dec["K"].unique())
    colors = k_colors(Ks)
    styles = dict(zip(Ks, ("-", "--", "-.", ":")))
    gains = ("total", "composition")
    fig, axes = plt.subplots(len(gains), len(STRATEGIES), figsize=(5.5 * len(STRATEGIES), 3.2 * len(gains)),
                             sharex=True, sharey="row", squeeze=False)
    for row, gain in enumerate(gains):
        for col, strategy in enumerate(STRATEGIES):
            ax = axes[row, col]
            for K in Ks:
                curve = dec[(dec["K"] == K) & (dec["strategy"] == strategy)].groupby("b")[gain].mean()
                ax.plot(curve.index, curve.values, color=colors[K], linestyle=styles.get(K, "-"), label=f"K={K}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"{gain}, {strategy}", fontsize=10)
            ax.grid(alpha=0.3)
            if row == len(gains) - 1:
                ax.set_xlabel("b (labels per class)")
            if col == 0:
                ax.set_ylabel("gain (pp)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(Ks), bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.tight_layout()
    return save(fig, out_dir, "figA3_gain_vs_b.png")


def make_figures(results: str, raw_dir: str, out_dir: str, targets: Optional[np.ndarray],
                 min_b: int = 20) -> (List[str], Optional[pd.DataFrame]):
    """All figures. Without targets, the class composition figures are skipped."""
    os.makedirs(out_dir, exist_ok=True)
    df = load_results(results)
    dec = decompose(df, "test_accuracy", min_b)
    paths = [fig_accuracy_vs_b(df, out_dir), fig_decomposition(dec, summarize(dec), out_dir),
             fig_accuracy_all_k(df, out_dir), fig_gain_vs_b(dec, out_dir)]
    shares = None
    if targets is not None:
        tasks = load_tasks(os.path.join(raw_dir, "tasks"))
        acq_dir = os.path.join(raw_dir, "acq")
        parts = []
        for b, name in ((100, "fig3_class_composition_b100.png"), (200, "figA1_class_composition_b200.png")):
            path, part = fig_class_composition(acq_dir, tasks, targets, b, out_dir, name)
            paths.append(path)
            parts.append(part)
        shares = pd.concat(parts, ignore_index=True)
    return paths, shares


def load_cifar_targets(cifar_dir: str) -> np.ndarray:
    from torchvision.datasets import CIFAR100
    return np.array(CIFAR100(cifar_dir, train=True, download=False).targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Results CSV of tools/collect_results.py.")
    parser.add_argument("--raw_dir", type=str, required=True, help="Output directory with acq/ and tasks/.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory for the PNG files.")
    parser.add_argument("--cifar_dir", type=str, default="../ga3_data", help="Directory with cifar-100-python.")
    args = parser.parse_args()

    try:
        train_targets = load_cifar_targets(args.cifar_dir)
    except Exception as error:  # Only the class composition figures need the targets.
        print(f"CIFAR-100 train targets not loaded ({error}); skipping the class composition figures.")
        train_targets = None
    figure_paths, share_table = make_figures(args.results, args.raw_dir, args.out_dir, train_targets)
    for figure_path in figure_paths:
        print(f"Wrote {figure_path}")
    if share_table is not None:
        print(f"\nMean share of classes with q < 0.5 (over seeds)\n{share_table.to_string(index=False)}")
