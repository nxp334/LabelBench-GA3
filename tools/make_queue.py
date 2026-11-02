"""
Write the priority-ordered evaluation job list. Each line is "mode<TAB>file", plus "<TAB>trainer_config" if a trainer
config overrides the one of the mode, so every job of a queue uses it, also after a restart. Within a tier jobs are
ordered by s, K, b, condition, so a partly finished queue has complete seeds. All conditions share the init set, so
b = 10 has one job per (K, s), which analysis uses for every condition.
"""
import argparse
import os
from typing import Iterable, List, Optional, Sequence, Tuple

from LabelBench.dataset.class_subset import labeled_filename

MARGIN_CONDITIONS = ("random", "margin", "cb-margin")
BADGE_CONDITIONS = ("badge", "cb-badge")
ALL_CONDITIONS = MARGIN_CONDITIONS + BADGE_CONDITIONS
LATER_BS = tuple(range(20, 201, 10))

# (mode, condition, K, s, B)
Job = Tuple[str, str, int, int, int]


def _tier(mode: str, Ks: Iterable[int], seeds: Iterable[int], bs: Iterable[int],
          conditions: Sequence[str]) -> List[Job]:
    return [(mode, condition, K, s, b * K) for s in seeds for K in Ks for b in bs for condition in conditions]


def build_queue(Ks: Sequence[int], seeds: Sequence[int], modes: Sequence[str] = ("proxy",)) -> List[Job]:
    """Jobs in priority order. Tier 4 also has K = 50 at B = 2000, the size of the other K at b = 200, 100, 20."""
    queue = []
    if "e2e" in modes:
        queue += _tier("e2e", Ks, seeds, (20, 50), MARGIN_CONDITIONS)
        queue += _tier("e2e", Ks, seeds, (20, 50), BADGE_CONDITIONS)
        queue += _tier("e2e", Ks, seeds, (100,), ALL_CONDITIONS)
        queue += _tier("e2e", Ks, seeds, (200,), ALL_CONDITIONS)
        if 50 in Ks:
            queue += _tier("e2e", (50,), seeds, (40,), ALL_CONDITIONS)
        queue += _tier("e2e", Ks, seeds, (10,), ("random",))
    if "proxy" in modes:
        queue += _tier("proxy", Ks, seeds, (10,), ("random",))
        queue += _tier("proxy", Ks, seeds, LATER_BS, ALL_CONDITIONS)
    return queue


def write_queue(path: str, queue: Sequence[Job], save_dir: str,
                trainer_config: Optional[str] = None) -> Tuple[int, int]:
    """Write the jobs whose file exists. Returns the number of jobs written and dropped."""
    lines = []
    override = "" if trainer_config is None else f"\t{trainer_config}"
    for mode, condition, K, s, B in queue:
        file = os.path.join(save_dir, "acq", labeled_filename(condition, K, s, B))
        if os.path.exists(file):
            lines.append(f"{mode}\t{file}{override}\n")
        else:
            print(f"Warning: dropping job, {file} does not exist.")
    with open(path, "w") as f:
        f.writelines(lines)
    return len(lines), len(queue) - len(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="Directory with the acq folder.")
    parser.add_argument("--Ks", type=int, nargs="+", default=[10, 20, 50, 100], help="Numbers of classes.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="Experiment seeds.")
    parser.add_argument("--modes", type=str, choices=["proxy", "e2e", "e2e,proxy"], default="proxy",
                        help="Evaluation modes to queue.")
    parser.add_argument("--output", type=str, help="Queue file. Default: {save_dir}/queue.tsv.")
    parser.add_argument("--trainer_config", type=str, help="Overrides the trainer config of the mode.")
    args = parser.parse_args()
    if args.trainer_config is not None and "," in args.modes:
        parser.error("--trainer_config applies to one mode, the modes train different models.")

    output = args.output if args.output is not None else os.path.join(args.save_dir, "queue.tsv")
    written, dropped = write_queue(output, build_queue(args.Ks, args.seeds, args.modes.split(",")), args.save_dir,
                                   args.trainer_config)
    print(f"Wrote {written} jobs to {output}, dropped {dropped}.")
