import os
import subprocess
import sys
import time

import pytest

FLOOR = "cifar100/flexmatch/clip_ViTB32_linear_floor500.json"
MINSTEPS = "cifar100/flexmatch/clip_ViTB32_linear_minsteps.json"

# Stands in for python: records its arguments. make_queue.py writes a queue with one result per job, but one, and
# collect_results.py a CSV. The inline check of run_seeds.sh runs in real python.
STUB_PYTHON = """#!/bin/sh
echo "$@" >> "$STUB_LOG"
case "$1" in
  main.py) sleep "${STUB_SLEEP:-0}" ;;
  -) shift; exec "$REAL_PYTHON" - "$@" ;;
  tools/collect_results.py)
    printf 'K,trainer_config\\n10,%s\\n10,%s\\n' "$TRAINER_CONFIG" "${STUB_RESULT_CONFIG:-$TRAINER_CONFIG}" \\
      > "$SAVE_DIR/results_proxy.csv" ;;
  tools/make_queue.py)
    mkdir -p "$SAVE_DIR/eval/proxy"
    : > "$QUEUE_FILE"
    i=0
    while [ "$i" -lt "$STUB_JOBS" ]; do
      printf 'proxy\\t%s/acq/job%s.npy\\tconfig\\n' "$SAVE_DIR" "$i" >> "$QUEUE_FILE"
      [ "$i" -eq "${STUB_NO_RESULT:--1}" ] || touch "$SAVE_DIR/eval/proxy/job$i.json"
      i=$((i + 1))
    done ;;
esac
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
    write_executable(bin_dir / "tmux", '#!/bin/sh\necho "$@" >> "$STUB_LOG"\n[ "$1" != has-session ] || exit 1\n')
    write_executable(bin_dir / "timeout", '#!/bin/sh\necho timeout "$1" >> "$STUB_LOG"\nshift\nexec "$@"\n')
    return dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", DATA_DIR=str(tmp_path / "data"),
                SAVE_DIR=str(tmp_path / "save"), PYTHON_BIN=str(write_executable(bin_dir / "python", STUB_PYTHON)),
                WANDB_PROJECT_ACQ="acq project", WANDB_PROJECT_EVAL="eval project", REAL_PYTHON=sys.executable,
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


def test_acquisition_runs_have_a_time_limit(env):
    assert run(env, "scripts/launch_acquisition.sh").returncode == 0
    limits = calls(env, "timeout")
    assert len(limits) == 24 and {call[1] for call in limits} == {"24h"}


@pytest.mark.parametrize("name", ["DATA_DIR", "PYTHON_BIN"])
def test_acquisition_needs_data_dir_and_python(env, name):
    del env[name]
    result = run(env, "scripts/launch_acquisition.sh")
    assert result.returncode != 0 and name in result.stderr
    assert not os.path.exists(env["STUB_LOG"])


def test_stopped_acquisition_starts_no_new_runs(env):
    env.update(MAX_PARALLEL="1", STUB_SLEEP="2", POLL_SECONDS="0.1")
    process = subprocess.Popen(["scripts/launch_acquisition.sh"], env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    deadline = time.time() + 10
    while time.time() < deadline and not (os.path.exists(env["STUB_LOG"]) and calls(env, "main.py")):
        time.sleep(0.05)
    process.terminate()
    assert process.wait(timeout=10) == 143
    assert len(calls(env, "main.py")) == 1


def test_eval_workers_append_to_their_output(env, tmp_path):
    (tmp_path / "save").mkdir()
    (tmp_path / "save" / "queue.tsv").write_text("proxy\tfile.npy\n")
    assert run(env, "scripts/launch_eval.sh").returncode == 0
    with open(env["STUB_LOG"], "r") as f:
        started = [line for line in f if line.startswith("new-session")]
    assert len(started) == 1 and '>> "' in started[0] and '.out" 2>&1' in started[0]


def test_timed_out_eval_job_is_logged_as_failed(env, tmp_path):
    write_executable(tmp_path / "bin" / "timeout", "#!/bin/sh\nexit 124\n")
    (tmp_path / "queue.tsv").write_text(f"proxy\t{tmp_path}/margin_K10_s1_B200.npy\n")
    env["PYTHONPATH"] = os.getcwd()
    result = run(env, sys.executable, "tools/run_queue.py", "--queue", str(tmp_path / "queue.tsv"), "--worker", "0",
                 "--num_workers", "1", "--save_dir", env["SAVE_DIR"], "--data_dir", env["DATA_DIR"])
    assert result.returncode == 1
    with open(os.path.join(env["SAVE_DIR"], "logs", "worker0.log"), "r") as f:
        assert "FAILED (124)" in f.read()


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


def test_stage_fails_if_results_use_another_config(env):
    env.update(TRAINER_CONFIG=FLOOR, STUB_RESULT_CONFIG=MINSTEPS)
    result = run(env, "scripts/run_seeds.sh", "4", "4")
    assert result.returncode != 0 and "expected only" in result.stderr and "Seeds 4-4" not in result.stdout


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


@pytest.mark.parametrize("name", ["DATA_DIR", "PYTHON_BIN", "WANDB_PROJECT_ACQ", "WANDB_PROJECT_EVAL"])
def test_stage_needs_all_settings(env, name):
    env["TRAINER_CONFIG"] = FLOOR
    del env[name]
    result = run(env, "scripts/run_seeds.sh", "1", "3")
    assert result.returncode != 0 and name in result.stderr
    assert not os.path.exists(env["STUB_LOG"])
