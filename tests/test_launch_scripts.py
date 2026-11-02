import os
import subprocess

import pytest

FLOOR = "cifar100/flexmatch/clip_ViTB32_linear_floor500.json"
MINSTEPS = "cifar100/flexmatch/clip_ViTB32_linear_minsteps.json"

# Stands in for python: records its arguments, and make_queue.py writes a queue with one result per job, but one.
STUB_PYTHON = """#!/bin/sh
echo "$@" >> "$STUB_LOG"
if [ "$1" = tools/make_queue.py ]; then
  mkdir -p "$SAVE_DIR/eval/proxy"
  : > "$QUEUE_FILE"
  i=0
  while [ "$i" -lt "$STUB_JOBS" ]; do
    printf 'proxy\\t%s/acq/job%s.npy\\tconfig\\n' "$SAVE_DIR" "$i" >> "$QUEUE_FILE"
    [ "$i" -eq "${STUB_NO_RESULT:--1}" ] || touch "$SAVE_DIR/eval/proxy/job$i.json"
    i=$((i + 1))
  done
fi
"""


def write_executable(path, text):
    path.write_text(text)
    path.chmod(0o755)
    return path


@pytest.fixture
def env(tmp_path):
    """Environment that runs the scripts with stubs instead of python and tmux, inside tmp_path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # tmux reports no session, so the wait for the workers ends at once.
    write_executable(bin_dir / "tmux", '#!/bin/sh\n[ "$1" != has-session ] || exit 1\n')
    return dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", DATA_DIR=str(tmp_path / "data"),
                SAVE_DIR=str(tmp_path / "save"), PYTHON_BIN=str(write_executable(bin_dir / "python", STUB_PYTHON)),
                STUB_LOG=str(tmp_path / "calls.log"), STUB_JOBS="384", MAX_PARALLEL="2", POLL_SECONDS="0")


def calls(env, script):
    with open(env["STUB_LOG"], "r") as f:
        return [line.split() for line in f if line.startswith(script)]


def value(call, flag):
    return call[call.index(flag) + 1]


def run(env, *command):
    return subprocess.run(command, env=env, capture_output=True, text=True)


def test_acquisition_default_is_seeds_1_to_3_with_the_old_config(env):
    assert run(env, "scripts/launch_acquisition.sh").returncode == 0
    runs = calls(env, "main.py")
    assert len(runs) == 24 and {value(call, "--seed") for call in runs} == {"1", "2", "3"}
    assert {value(call, "--trainer_config") for call in runs} == {MINSTEPS}


def test_num_seeds_keeps_its_meaning(env):
    assert run(env, "scripts/launch_acquisition.sh", "--num_seeds", "2").returncode == 0
    assert {value(call, "--seed") for call in calls(env, "main.py")} == {"1", "2"}


def test_acquisition_seed_range_and_trainer_config(env):
    env["TRAINER_CONFIG"] = FLOOR
    assert run(env, "scripts/launch_acquisition.sh", "--first_seed", "4", "--last_seed", "4").returncode == 0
    runs = calls(env, "main.py")
    assert len(runs) == 8 and {value(call, "--seed") for call in runs} == {"4"}
    assert {value(call, "--class_seed") for call in runs} == {"4"}
    assert {value(call, "--trainer_config") for call in runs} == {FLOOR}
    with open(os.path.join(env["SAVE_DIR"], "logs", "acquisition", "margin_K10_s4.log"), "r") as f:
        assert f.readline() == f"trainer_config: {FLOOR}\n"


def test_acquisition_skips_finished_runs(env, tmp_path):
    (tmp_path / "save" / "acq").mkdir(parents=True)
    (tmp_path / "save" / "acq" / "margin_K10_s4_B2000.npy").touch()
    assert run(env, "scripts/launch_acquisition.sh", "--first_seed", "4", "--last_seed", "4").returncode == 0
    assert len(calls(env, "main.py")) == 7


def test_stage_runs_all_steps_with_the_given_config(env):
    env["TRAINER_CONFIG"] = FLOOR
    result = run(env, "scripts/run_seeds.sh", "4", "4")
    assert result.returncode == 0, result.stderr
    assert {value(call, "--trainer_config") for call in calls(env, "main.py")} == {FLOOR}
    assert len(calls(env, "tools/make_baselines.py")) == 4
    queue_call = calls(env, "tools/make_queue.py")[0]
    assert value(queue_call, "--trainer_config") == FLOOR and value(queue_call, "--seeds") == "4"
    assert len(calls(env, "tools/collect_results.py")) == 1


def test_stage_fails_if_a_job_has_no_result(env):
    env.update(TRAINER_CONFIG=FLOOR, STUB_NO_RESULT="7")
    result = run(env, "scripts/run_seeds.sh", "4", "4")
    assert result.returncode != 0 and "1 without a result" in result.stderr
    assert not calls(env, "tools/collect_results.py")


def test_stage_fails_if_the_queue_is_short(env):
    env.update(TRAINER_CONFIG=FLOOR, STUB_JOBS="383")
    result = run(env, "scripts/run_seeds.sh", "4", "4")
    assert result.returncode != 0 and "383 of 384 jobs queued" in result.stderr


def test_failed_stage_stops_the_next_one_in_a_chain(env):
    env.update(TRAINER_CONFIG=FLOOR, STUB_NO_RESULT="7", STUB_JOBS="1152")
    command = "scripts/run_seeds.sh 1 3 && scripts/run_seeds.sh 4 4"
    assert subprocess.run(["bash", "-c", command], env=env, capture_output=True).returncode != 0
    assert {value(call, "--seed") for call in calls(env, "main.py")} == {"1", "2", "3"}


def test_stage_stops_before_evaluation_if_a_finished_run_used_another_config(env, tmp_path):
    env["TRAINER_CONFIG"] = FLOOR
    (tmp_path / "save" / "acq").mkdir(parents=True)
    (tmp_path / "save" / "logs" / "acquisition").mkdir(parents=True)
    (tmp_path / "save" / "acq" / "margin_K10_s4_B2000.npy").touch()
    (tmp_path / "save" / "logs" / "acquisition" / "margin_K10_s4.log").write_text(f"trainer_config: {MINSTEPS}\n")

    result = run(env, "scripts/run_seeds.sh", "4", "4")
    assert result.returncode != 0 and "margin_K10_s4.log" in result.stderr
    assert not calls(env, "tools/make_queue.py")


def test_stage_refuses_the_old_directory_and_a_missing_config(env, tmp_path):
    env.update(TRAINER_CONFIG=FLOOR, SAVE_DIR=str(tmp_path / "ga3_output"))
    assert run(env, "scripts/run_seeds.sh", "1", "3").returncode != 0
    assert not (tmp_path / "ga3_output").exists()

    env.update(SAVE_DIR=str(tmp_path / "save"))
    del env["TRAINER_CONFIG"]
    assert run(env, "scripts/run_seeds.sh", "1", "3").returncode != 0
    assert not (tmp_path / "save").exists()
