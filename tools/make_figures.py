"""
Figures of docs/ANALYSIS.md as PNG, sized for a full-width figure (7 in) in a two-column LaTeX document. Accuracy and
gains come from the results CSV; the class composition and image example figures read the labeled sets
acq/<condition>_K<K>_s<s>_B<B>.npy and the task files of the raw output directory, plus the CIFAR-100 train set.
"""
import argparse
import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import pandas as pd  # noqa: E402

from LabelBench.dataset.class_subset import labeled_filename  # noqa: E402
from tools.analyze_results import (GAINS, STRATEGIES, decompose, load_results, load_tasks, summarize,  # noqa: E402
                                   write_table)

DPI = 200
FIG_WIDTH = 7.0
COLUMN_WIDTH = 3.4
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "legend.title_fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
NAME = {"margin": "Margin", "badge": "BADGE", "total": "Total", "composition": "Composition",
        "selection": "Selection", "random": "Random", "cb": "ClassBaseline", "al": "AL"}
CONDITION_STYLE = {"random": ("#7f7f7f", ":"), "cb": ("#ff7f0e", "--"), "al": ("#1f77b4", "-")}
SEED_MARKERS = ("o", "s", "^", "D")
GAIN_STYLE = {"total": ("#333333", "o"), "composition": ("#ff7f0e", "s"), "selection": ("#1f77b4", "^")}
INIT_PER_CLASS = 10


def k_colors(Ks) -> Dict[int, tuple]:
    """One blue ramp, light for the smallest K, dark for the largest."""
    return dict(zip(sorted(Ks), plt.cm.Blues(np.linspace(0.5, 1.0, len(Ks)))))


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
    fig, axes = plt.subplots(len(STRATEGIES), len(Ks), figsize=(FIG_WIDTH, 1.9 * len(STRATEGIES)), sharex=True,
                             squeeze=False)
    for row, strategy in enumerate(STRATEGIES):
        for col, K in enumerate(Ks):
            ax = axes[row, col]
            for key, condition in (("random", "random"), ("cb", f"cb-{strategy}"), ("al", strategy)):
                color, style = CONDITION_STYLE[key]
                curve = seed_stats(df, condition, K)
                ax.plot(curve["b"], curve["mean"], color=color, linestyle=style, label=NAME[key])
                ax.fill_between(curve["b"], curve["min"], curve["max"], color=color, alpha=0.15, linewidth=0)
            ax.set_title(f"{NAME[strategy]}, K={K}")
            ax.set_xticks([50, 100, 150, 200])
            ax.grid(alpha=0.3)
            if row == len(STRATEGIES) - 1:
                ax.set_xlabel("b (labels per class)")
            if col == 0:
                ax.set_ylabel("Test accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.tight_layout()
    return save(fig, out_dir, "fig1_accuracy_vs_b.png")


def fig_decomposition(dec: pd.DataFrame, summary: pd.DataFrame, out_dir: str) -> str:
    """
    Per strategy: the three gains per K, mean (horizontal bar) with t-interval, and the per-seed means to its right,
    one marker shape per seed, the same shape in every gain.
    """
    Ks = sorted(dec["K"].unique())
    seeds = dec.groupby(["K", "strategy", "s"], as_index=False)[list(GAINS)].mean()
    seed_ids = sorted(seeds["s"].unique())
    seed_markers = dict(zip(seed_ids, SEED_MARKERS))
    # Points to the right of the interval, in seed order, so they do not hide behind it.
    seed_offsets = dict(zip(seed_ids, 0.07 + 0.035 * np.arange(len(seed_ids))))
    offsets = dict(zip(GAINS, (-0.25, 0.0, 0.25)))
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(FIG_WIDTH, 2.6), sharey=True, squeeze=False)
    for ax, strategy in zip(axes[0], STRATEGIES):
        for gain in GAINS:
            color, _ = GAIN_STYLE[gain]
            rows = summary[summary["strategy"] == strategy].set_index("K").loc[Ks]
            x = np.arange(len(Ks)) + offsets[gain]
            ax.errorbar(x, rows[f"{gain}_mean"], yerr=rows[f"{gain}_ci"], fmt="_", color=color, capsize=0,
                        elinewidth=1.2, markersize=8, markeredgewidth=2, label=f"{NAME[gain]} (mean, 95% CI)")
            for i, K in enumerate(Ks):
                points = seeds[(seeds["strategy"] == strategy) & (seeds["K"] == K)]
                for s, value in zip(points["s"], points[gain]):
                    ax.scatter(x[i] + seed_offsets[s], value, s=9, color=color, alpha=0.75,
                               marker=seed_markers[s], linewidths=0)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(Ks)))
        ax.set_xticklabels([f"K={K}" for K in Ks])
        ax.set_title(NAME[strategy])
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].set_ylabel("Gain (pp)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for s in seed_ids:
        handles.append(Line2D([], [], marker=seed_markers[s], linestyle="", color="#7f7f7f", markersize=4))
        labels.append(f"Seed {s}")
    fig.legend(handles, labels, loc="upper center", ncol=len(GAINS) + len(seed_ids), bbox_to_anchor=(0.5, 1.07),
               frameon=False, columnspacing=1.2, handletextpad=0.3)
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
    """Panels margin and badge: sorted q per K (seed mean), random grey dotted, balanced at 1.0."""
    Ks = sorted(tasks["K"].unique())
    colors = k_colors(Ks)
    random_color, random_style = CONDITION_STYLE["random"]
    random_label = f"{NAME['random']} (all K)"
    shares = []
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(FIG_WIDTH, 2.6), sharey=True, squeeze=False)
    for ax, strategy in zip(axes[0], STRATEGIES):
        for K in Ks:
            x = (np.arange(1, K + 1) - 0.5) / K
            for condition in (strategy, "random"):
                result = class_composition(acq_dir, tasks, targets, condition, K, b)
                if condition == "random":
                    ax.plot(x, result["q"], color=random_color, linestyle=random_style, linewidth=1.0,
                            label=random_label if K == Ks[0] else None)
                else:
                    ax.plot(x, result["q"], color=colors[K], linestyle="-", label=f"K={K}")
                if ax is axes[0, 0] or condition != "random":
                    shares.append({"condition": condition, "K": K, "b": b,
                                   "share_q_below_0.5": result["share_below_half"]})
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_title(f"{NAME[strategy]}, b = {b} labels per class")
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        # AL curves first, random last.
        order = [i for i, label in enumerate(labels) if label != random_label] + [labels.index(random_label)]
        ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right", frameon=False,
                  title=f"{NAME[strategy]}, by K")
    axes[0, 0].set_ylabel("Labels in class ÷ b\n(1.0 = balanced, b per class)")
    fig.supxlabel("Classes sorted from most to least labeled (fraction of classes)", fontsize=8)
    fig.tight_layout()
    return save(fig, out_dir, name), pd.DataFrame(shares)


def class_count_table(acq_dir: str, tasks: pd.DataFrame, targets: np.ndarray, K: int, b: int,
                      n: int = 10) -> pd.DataFrame:
    """
    The n most and n least labeled classes per strategy at (K, b), by mean label count over seeds. Needs the same class
    set in every seed. seeds_in_group counts the seeds in which the class is among the n most (or least) labeled.
    """
    tasks = tasks[tasks["K"] == K]
    if len({tuple(sorted(ids)) for ids in tasks["class_ids"]}) != 1:
        raise ValueError(f"The seeds of K={K} do not share one class set.")
    names = dict(zip(tasks.iloc[0]["class_ids"], tasks.iloc[0]["classnames"]))
    rows = []
    for strategy in STRATEGIES:
        counts = pd.DataFrame({task["s"]: pd.Series(np.asarray(targets)[np.load(os.path.join(
            acq_dir, labeled_filename(strategy, K, task["s"], K * b)))]).value_counts().reindex(names, fill_value=0)
            for _, task in tasks.iterrows()})
        ranks = counts.rank(ascending=False, method="first")
        stats = pd.DataFrame({"mean_labels": counts.mean(axis=1), "sd_labels": counts.std(axis=1, ddof=1),
                              "min_labels": counts.min(axis=1), "max_labels": counts.max(axis=1)})
        stats = stats.sort_values("mean_labels", ascending=False, kind="mergesort")
        for group, chosen, in_group in (("most", stats.index[:n], (ranks <= n).sum(axis=1)),
                                        ("least", stats.index[::-1][:n], (ranks > K - n).sum(axis=1))):
            for rank, class_id in enumerate(chosen, start=1):
                rows.append({"strategy": strategy, "group": group, "rank": rank, "class_id": int(class_id),
                             "classname": names[class_id], **stats.loc[class_id].to_dict(),
                             "mean_labels_div_b": stats.loc[class_id, "mean_labels"] / b,
                             "seeds_in_group": int(in_group[class_id])})
    return pd.DataFrame(rows)


def eligible_pool(acq_dir: str, tasks: pd.DataFrame, condition: str, targets: np.ndarray,
                  class_ids: Sequence[int], K: int, b: int) -> np.ndarray:
    """
    Train indices of class_ids that the condition labeled after the balanced init, pooled over the seeds of K: the
    labeled set at B = K * b minus the initial set at B = 10 * K.
    """
    pool = []
    for s in tasks[tasks["K"] == K]["s"]:
        labeled = np.load(os.path.join(acq_dir, labeled_filename(condition, K, s, K * b)))
        initial = np.load(os.path.join(acq_dir, labeled_filename(condition, K, s, K * INIT_PER_CLASS)))
        pool.append(np.setdiff1d(labeled, initial))
    pool = np.unique(np.concatenate(pool))
    return pool[np.isin(np.asarray(targets)[pool], class_ids)]


def matched_indices(acq_dir: str, tasks: pd.DataFrame, condition: str, targets: np.ndarray, classes: Sequence[int],
                    K: int, b: int, seed: int = 0, exclude: Sequence[int] = ()) -> np.ndarray:
    """
    One index per entry of classes, of that class, drawn without replacement from eligible_pool and never one of
    exclude.
    """
    targets = np.asarray(targets)
    pool = eligible_pool(acq_dir, tasks, condition, targets, sorted(set(classes)), K, b)
    rng = np.random.RandomState(seed)
    chosen = []
    for c in classes:
        candidates = np.setdiff1d(pool[targets[pool] == c], list(chosen) + list(exclude))
        if len(candidates) == 0:
            raise ValueError(f"Too few images of class {c} for {condition} at K={K}, b={b}.")
        chosen.append(rng.choice(candidates))
    return np.array(chosen)


def example_picks(targets: np.ndarray, acq_dir: str, tasks: pd.DataFrame, strategy: str, K: int, b: int,
                  n: int = 9) -> (int, Dict[int, str], Dict[str, tuple]):
    """
    For the n most and the n least labeled classes of the strategy at (K, b), each group sorted from most to least
    labeled: one image per class labeled by the strategy and one labeled by its ClassBaseline. Returns the number of
    classes per group, the title "name (mean label count)" per class and {group: (strategy indices, ClassBaseline
    indices)}.
    """
    n = min(n, K // 2)
    table = class_count_table(acq_dir, tasks, targets, K, b, n=n)
    table = table[table["strategy"] == strategy]
    titles = {row.class_id: f"{row.classname} ({row.mean_labels:.0f})" for row in table.itertuples()}
    picks = {}
    for group in ("most", "least"):
        # The table lists "least" from the least labeled up; reversed, both groups follow the table's ranking.
        classes = table[table["group"] == group]["class_id"].tolist()
        classes = classes if group == "most" else classes[::-1]
        al_idxs = matched_indices(acq_dir, tasks, strategy, targets, classes, K, b)
        # Both conditions can hold the same image; the ClassBaseline row never repeats one of the strategy row.
        picks[group] = (al_idxs, matched_indices(acq_dir, tasks, f"cb-{strategy}", targets, classes, K, b,
                                                 exclude=al_idxs))
    return n, titles, picks


def show_image(ax, image: np.ndarray, title: str):
    ax.imshow(image)
    ax.set_title(title, fontsize=6, pad=2)
    ax.axis("off")


def fig_examples(images: np.ndarray, targets: np.ndarray, acq_dir: str, tasks: pd.DataFrame, strategy: str, K: int,
                 b: int, out_dir: str, name: str) -> str:
    """
    Rows: the 9 most and the 9 least labeled classes of the strategy at (K, b). Columns: images labeled by the strategy
    and by its ClassBaseline, 3 x 3 each, one per class, sorted from most to least labeled. The ClassBaseline image at
    each position is of the same class as the strategy's image at that position.
    """
    n_classes, titles, picks = example_picks(targets, acq_dir, tasks, strategy, K, b)
    fig = plt.figure(figsize=(FIG_WIDTH, 7.6))
    subfigs = fig.subfigures(2, 2, wspace=0.04, hspace=0.04)
    groups = (("most", f"Most labeled {n_classes} classes"), ("least", f"Least labeled {n_classes} classes"))
    columns = (f"chosen by {NAME[strategy]}", f"{NAME['cb']} (random within class)")
    for row, (group, group_title) in enumerate(groups):
        for col, (column_title, idxs) in enumerate(zip(columns, picks[group])):
            subfig = subfigs[row, col]
            subfig.suptitle(f"{group_title}, {column_title}", fontsize=8)
            for ax, idx in zip(subfig.subplots(3, 3, gridspec_kw={"wspace": 0.05, "hspace": 0.3}).flat, idxs):
                show_image(ax, images[idx], titles[int(targets[idx])])
    fig.suptitle(f"{NAME[strategy]}, K = {K}, b = {b} labels per class", y=1.02)
    return save(fig, out_dir, name)


def fig_examples_compact(images: np.ndarray, targets: np.ndarray, acq_dir: str, tasks: pd.DataFrame, strategy: str,
                         K: int, b: int, out_dir: str, name: str, n: int = 3) -> str:
    """
    The n most and the n least labeled classes of fig_examples (its first and last row, same images), one column
    wide. Top block: most labeled, bottom block: least labeled. In each block, row 1 is the strategy and row 2 its
    ClassBaseline, with the same class in each column.
    """
    _, titles, picks = example_picks(targets, acq_dir, tasks, strategy, K, b)
    fig = plt.figure(figsize=(COLUMN_WIDTH, 4.5))
    subfigs = fig.subfigures(2, 1, hspace=0.04)
    rows = (NAME[strategy], NAME["cb"])
    for subfig, (group, group_title, part) in zip(subfigs, (("most", f"Most labeled {n} classes", slice(None, n)),
                                                            ("least", f"Least labeled {n} classes", slice(-n, None)))):
        subfig.suptitle(group_title, fontsize=8)
        axes = subfig.subplots(2, n, gridspec_kw={"wspace": 0.05, "hspace": 0.15})
        for row, idxs in enumerate(picks[group]):
            for col, idx in enumerate(idxs[part]):
                show_image(axes[row, col], images[idx], titles[int(targets[idx])])
            axes[row, 0].text(-0.08, 0.5, rows[row], transform=axes[row, 0].transAxes, rotation=90, ha="right",
                              va="center", fontsize=7)
    fig.suptitle(f"{NAME[strategy]}, K = {K}, b = {b} labels per class", y=1.03)
    return save(fig, out_dir, name)


def fig_accuracy_all_k(df: pd.DataFrame, out_dir: str) -> str:
    """One panel per strategy: AL solid and random dotted per K, seed mean."""
    Ks = sorted(df["K"].unique())
    colors = k_colors(Ks)
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(FIG_WIDTH, 2.8), sharey=True, squeeze=False)
    for ax, strategy in zip(axes[0], STRATEGIES):
        for K in Ks:
            for condition, style, label in ((strategy, "-", f"{NAME[strategy]} K={K}"),
                                            ("random", ":", f"{NAME['random']} K={K}")):
                curve = seed_stats(df, condition, K)
                ax.plot(curve["b"], curve["mean"], color=colors[K], linestyle=style, label=label)
        ax.set_title(NAME[strategy])
        ax.set_xlabel("b (labels per class)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2, loc="lower right", frameon=False)
    axes[0, 0].set_ylabel("Test accuracy (%)")
    fig.tight_layout()
    return save(fig, out_dir, "figA2_accuracy_all_K.png")


def fig_gain_vs_b(dec: pd.DataFrame, out_dir: str) -> str:
    """Rows total, composition and selection, columns strategy: seed mean of the gain per b, one line per K."""
    Ks = sorted(dec["K"].unique())
    colors = k_colors(Ks)
    styles = dict(zip(Ks, ("-", "--", "-.", ":")))
    fig, axes = plt.subplots(len(GAINS), len(STRATEGIES), figsize=(FIG_WIDTH, 1.8 * len(GAINS)), sharex=True,
                             sharey="row", squeeze=False)
    for row, gain in enumerate(GAINS):
        for col, strategy in enumerate(STRATEGIES):
            ax = axes[row, col]
            for K in Ks:
                curve = dec[(dec["K"] == K) & (dec["strategy"] == strategy)].groupby("b")[gain].mean()
                ax.plot(curve.index, curve.values, color=colors[K], linestyle=styles.get(K, "-"), label=f"K={K}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"{NAME[gain]}, {NAME[strategy]}")
            ax.grid(alpha=0.3)
            if row == len(GAINS) - 1:
                ax.set_xlabel("b (labels per class)")
            if col == 0:
                ax.set_ylabel("Gain (pp)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(Ks), bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.tight_layout()
    return save(fig, out_dir, "figA3_gain_vs_b.png")


def make_figures(results: str, raw_dir: str, out_dir: str, targets: Optional[np.ndarray], min_b: int = 20,
                 tables_dir: Optional[str] = None,
                 train_images: Optional[np.ndarray] = None) -> (List[str], Optional[pd.DataFrame]):
    """
    All figures. Without targets, the class composition and image example figures are skipped; without train_images,
    only the image example figures. With tables_dir, also the class count table at the largest K and b = 100.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = load_results(results)
    dec = decompose(df, "test_accuracy", min_b)
    paths = [fig_accuracy_vs_b(df, out_dir), fig_decomposition(dec, summarize(dec), out_dir),
             fig_accuracy_all_k(df, out_dir), fig_gain_vs_b(dec, out_dir)]
    shares = None
    if targets is not None:
        tasks = load_tasks(os.path.join(raw_dir, "tasks"))
        acq_dir = os.path.join(raw_dir, "acq")
        K = int(tasks["K"].max())
        parts = []
        for b, name in ((100, "fig3_class_composition_b100.png"), (200, "figA1_class_composition_b200.png"),
                        (30, "figA4_class_composition_b30.png")):
            path, part = fig_class_composition(acq_dir, tasks, targets, b, out_dir, name)
            paths.append(path)
            parts.append(part)
        shares = pd.concat(parts, ignore_index=True)
        if train_images is not None:
            for strategy, full, compact in zip(STRATEGIES, ("figA5", "figA6"), ("figA7", "figA8")):
                paths.append(fig_examples(train_images, targets, acq_dir, tasks, strategy, K, 100, out_dir,
                                          f"{full}_examples_{strategy}_b100.png"))
                paths.append(fig_examples_compact(train_images, targets, acq_dir, tasks, strategy, K, 100, out_dir,
                                                  f"{compact}_examples_compact_{strategy}_b100.png"))
        if tables_dir is not None:
            os.makedirs(tables_dir, exist_ok=True)
            b = 100
            table = class_count_table(acq_dir, tasks, targets, K, b, n=min(10, K // 2))
            paths += write_table(table, tables_dir, f"A_class_counts_K{K}_b{b}")
    return paths, shares


def load_cifar_train(cifar_dir: str) -> (np.ndarray, np.ndarray):
    """CIFAR-100 train images (N, 32, 32, 3, uint8) and targets."""
    from torchvision.datasets import CIFAR100
    dataset = CIFAR100(cifar_dir, train=True, download=False)
    return dataset.data, np.array(dataset.targets)


def load_cifar_targets(cifar_dir: str) -> np.ndarray:
    return load_cifar_train(cifar_dir)[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Results CSV of tools/collect_results.py.")
    parser.add_argument("--raw_dir", type=str, required=True, help="Output directory with acq/ and tasks/.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory for the PNG files.")
    parser.add_argument("--cifar_dir", type=str, default="../ga3_data", help="Directory with cifar-100-python.")
    parser.add_argument("--tables_dir", type=str, help="Directory for the class count table. Without it, no table.")
    args = parser.parse_args()

    try:
        images, train_targets = load_cifar_train(args.cifar_dir)
    except Exception as error:  # Only the class composition and image example figures need the train set.
        print(f"CIFAR-100 train set not loaded ({error}); skipping the class composition and image example figures.")
        images, train_targets = None, None
    figure_paths, share_table = make_figures(args.results, args.raw_dir, args.out_dir, train_targets,
                                             tables_dir=args.tables_dir, train_images=images)
    for figure_path in figure_paths:
        print(f"Wrote {figure_path}")
    if share_table is not None:
        print(f"\nMean share of classes with q < 0.5 (over seeds)\n{share_table.to_string(index=False)}")
