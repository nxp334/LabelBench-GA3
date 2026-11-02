"""
Run one worker's share of a queue, sequentially on one GPU. Resumable: finished jobs are skipped. Each job is a
subprocess, so GPU memory is released between jobs. A failed job is logged and the worker moves on.
"""
import argparse
import os
import subprocess
import sys
import time
from typing import List, Tuple

from LabelBench.dataset.class_subset import WANDB_PROJECT_EVAL, eval_json_path

# (mode, file) or (mode, file, trainer_config)
Job = Tuple[str, ...]


def read_queue(path: str) -> List[Job]:
    """Read the jobs of a queue."""
    with open(path, "r") as f:
        return [tuple(line.rstrip("\n").split("\t")) for line in f if line.strip()]


def select_jobs(jobs: List[Job], worker: int, num_workers: int) -> List[Job]:
    """Worker i of n takes every n-th job."""
    return jobs[worker::num_workers]


def eval_command(job: Job, eval_script: str, save_dir: str, data_dir: str, wandb_name: str,
                 wandb_project: str) -> List[str]:
    """Command that evaluates one job."""
    command = [sys.executable, eval_script, "--file", job[1], "--mode", job[0], "--save_dir", save_dir,
               "--data_dir", data_dir, "--wandb_name", wandb_name, "--wandb_project", wandb_project]
    return command + ["--trainer_config", job[2]] if len(job) > 2 else command


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=str, required=True, help="Queue file from make_queue.py.")
    parser.add_argument("--worker", type=int, required=True, help="Index of this worker.")
    parser.add_argument("--num_workers", type=int, required=True, help="Total number of workers.")
    parser.add_argument("--save_dir", type=str, required=True, help="Results go to {save_dir}/eval/{mode}.")
    parser.add_argument("--data_dir", type=str, help="Directory to store dataset.", default="./data")
    parser.add_argument("--wandb_name", type=str, help="Wandb entity.", default="ATDL-labelbench")
    parser.add_argument("--wandb_project", type=str, help="Wandb project.", default=WANDB_PROJECT_EVAL)
    args = parser.parse_args()

    log_dir = os.path.join(args.save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_selection.py")
    jobs = select_jobs(read_queue(args.queue), args.worker, args.num_workers)
    num_failed = 0

    with open(os.path.join(log_dir, f"worker{args.worker}.log"), "a") as progress:
        def log(message: str) -> None:
            progress.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
            progress.flush()

        for i, job in enumerate(jobs, start=1):
            mode, file = job[:2]
            if os.path.exists(eval_json_path(args.save_dir, mode, file)):
                log(f"[{i}/{len(jobs)}] skip {mode} {file}")
                continue
            log(f"[{i}/{len(jobs)}] start {mode} {file}")
            start_time = time.time()
            returncode = subprocess.run(
                eval_command(job, eval_script, args.save_dir, args.data_dir, args.wandb_name, args.wandb_project)
            ).returncode
            log(f"[{i}/{len(jobs)}] {'done' if returncode == 0 else f'FAILED ({returncode})'} {mode} {file} "
                f"in {time.time() - start_time:.0f}s")
            num_failed += returncode != 0
        log(f"finished, {num_failed} failed")

    sys.exit(1 if num_failed else 0)
