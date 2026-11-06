import numpy as np

from tools.make_queue import ALL_CONDITIONS, build_queue, write_queue
from tools.run_queue import eval_command, read_queue, select_jobs

KS = (10, 20, 50, 100)
SEEDS = (1, 2, 3)
ALL_MODES = ("e2e", "proxy")


def bs(jobs) -> set:
    return {B / K for _, _, K, _, B in jobs}


def conditions(jobs) -> set:
    return {job[1] for job in jobs}


def test_tiers_in_priority_order():
    queue = build_queue(KS, SEEDS, ALL_MODES)
    tiers = [queue[:72], queue[72:120], queue[120:180], queue[180:255], queue[255:267], queue[267:]]

    assert bs(tiers[0]) == {20, 50} and conditions(tiers[0]) == {"random", "margin", "cb-margin"}
    assert bs(tiers[1]) == {20, 50} and conditions(tiers[1]) == {"badge", "cb-badge"}
    assert bs(tiers[2]) == {100} and conditions(tiers[2]) == set(ALL_CONDITIONS)
    assert bs(tiers[3]) == {200, 40} and conditions(tiers[3]) == set(ALL_CONDITIONS)
    assert bs(tiers[4]) == {10}
    assert bs(tiers[5]) == set(float(b) for b in range(10, 201, 10))
    assert all(job[0] == "e2e" for tier in tiers[:5] for job in tier)
    assert all(job[0] == "proxy" for job in tiers[5])


def test_job_counts_and_no_duplicates():
    queue = build_queue(KS, SEEDS, ALL_MODES)
    assert sum(job[0] == "e2e" for job in queue) == 267
    assert sum(job[0] == "proxy" for job in queue) == 1152
    assert len(set(queue)) == len(queue)


def test_default_queue_is_proxy_only():
    queue = build_queue(KS, SEEDS)
    assert len(queue) == 1152 and {job[0] for job in queue} == {"proxy"}
    assert {job[0] for job in build_queue(KS, SEEDS, ["e2e"])} == {"e2e"}


def test_K50_at_B2000_is_in_tier_4_for_all_conditions():
    extra = [job for job in build_queue(KS, SEEDS, ALL_MODES)[180:255] if job[2] == 50 and job[4] == 2000]
    assert len(extra) == 15 and conditions(extra) == set(ALL_CONDITIONS)


def test_init_has_one_job_per_task_in_both_modes():
    queue = build_queue(KS, SEEDS, ALL_MODES)
    for mode in ALL_MODES:
        init_jobs = [job for job in queue if job[0] == mode and job[4] == 10 * job[2]]
        assert len(init_jobs) == 12 and conditions(init_jobs) == {"random"}


def test_within_tier_order_completes_seed_first():
    seeds = [job[3] for job in build_queue(KS, SEEDS, ALL_MODES)[:72]]
    assert seeds == sorted(seeds)


def test_workers_split_jobs_exactly():
    jobs = [("e2e", f"file{i}") for i in range(10)]
    shares = [select_jobs(jobs, i, 4) for i in range(4)]
    assert sorted(job for share in shares for job in share) == sorted(jobs)
    assert shares[1] == [jobs[1], jobs[5], jobs[9]]


def test_write_queue_drops_missing_files_and_roundtrips(tmp_path):
    acq = tmp_path / "acq"
    acq.mkdir()
    np.save(acq / "margin_K10_s1_B200.npy", np.arange(200))
    path = tmp_path / "queue.tsv"
    queue = [("e2e", "margin", 10, 1, 200), ("e2e", "random", 10, 1, 200)]

    assert write_queue(str(path), queue, str(tmp_path)) == (1, 1)
    assert read_queue(str(path)) == [("e2e", str(acq / "margin_K10_s1_B200.npy"))]


def test_queue_without_trainer_config_keeps_two_columns(tmp_path):
    (tmp_path / "acq").mkdir()
    np.save(tmp_path / "acq" / "margin_K10_s1_B200.npy", np.arange(200))
    path = tmp_path / "queue.tsv"
    write_queue(str(path), [("proxy", "margin", 10, 1, 200)], str(tmp_path))
    assert path.read_text() == f"proxy\t{tmp_path / 'acq' / 'margin_K10_s1_B200.npy'}\n"


def test_trainer_config_goes_from_queue_to_the_eval_command(tmp_path):
    (tmp_path / "acq").mkdir()
    np.save(tmp_path / "acq" / "margin_K10_s1_B200.npy", np.arange(200))
    path = tmp_path / "queue.tsv"
    write_queue(str(path), [("proxy", "margin", 10, 1, 200)], str(tmp_path), "floor.json")

    job = read_queue(str(path))[0]
    command = eval_command(job, "eval.py", "save", "data", "entity", "project")
    assert command[command.index("--trainer_config") + 1] == "floor.json"
    assert command[command.index("--mode") + 1] == "proxy" and command[command.index("--file") + 1] == job[1]


def test_eval_command_without_trainer_config_uses_the_config_of_the_mode():
    command = eval_command(("proxy", "file.npy"), "eval.py", "save", "data", "entity", "project")
    assert "--trainer_config" not in command


def test_eval_command_has_a_time_limit():
    command = eval_command(("proxy", "file.npy"), "eval.py", "save", "data", "entity", "project")
    assert command[:2] == ["timeout", "1h"]
