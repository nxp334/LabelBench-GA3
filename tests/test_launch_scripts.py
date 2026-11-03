import os
import subprocess
import sys
import time

import pytest

FLOOR = "cifar100/flexmatch/clip_ViTB32_linear_floor500.json"
MINSTEPS = "cifar100/flexmatch/clip_ViTB32_linear_minsteps.json"

# Stands in for python: records its arguments. make_queue.py writes a queue with one result per job, but one, and
# collect_results.py a CSV. The queue has STUB_JOBS jobs, by default 384 per seed. The inline check of run_seeds.sh runs
# in real python.
STUB_PYTHON = """#!/bin/sh
echo "$@" >> "$STUB_LOG"
case "$1" in
  main.py) sleep "${STUB_SLEEP:-0}" ;;
  -) shift; exec "$REAL_PYTHON" - "$@" ;;
  tools/collect_results.py)
    printf 'K,trainer_config\\n10,%s\\n10,%s\\n' "$TRAINER_CONFIG" "${STUB_RESULT_CONFIG:-$TRAINER_CONFIG}" \\
      > "$SAVE_DIR/results_proxy.csv" ;;
  tools/make_queue.py)
    jobs="${STUB_JOBS:-}"
    if [ -z "$jobs" ]; then
      jobs=0
      in_seeds=0
      for arg in "$@"; do
        case "$arg" in
          --seeds) in_seeds=1 ;;
          --*) in_seeds=0 ;;
          *) if [ "$in_seeds" -eq 1 ]; then jobs=$((jobs + 384)); fi ;;
        esac
      done
    fi
    mkdir -p "$SAVE_DIR/eval/proxy"
    : > "$QUEUE_FILE"
    i=0
    while [ "$i" -lt "$jobs" ]; do
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
                STUB_LOG=str(tmp_path / "calls.log"), MAX_PARALLEL="2", POLL_SECONDS="0")


def calls(env, script):
    with open(env["STUB_LOG"], "r") as f:
        return [line.split() for line in f if line.startswith(script)]


def value(call, flag):
    return call[call.index(flag) + 1]


def values(call, flag):
    """All values after flag, up to the next flag."""
    rest = call[call.index(flag) + 1:]
    return rest[:next((i for i, arg in enumerate(rest) if arg.startswith("--")), len(rest))]


def started(stdout):
    """Names of the acquisition runs in the order they were started."""
    return [line.split()[3] for line in stdout.splitlines() if " start " in line]


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


def test_default_acquisition_plan_is_unchanged(env):
    # The plan before ROUND_SIZE and KS: K in 10, 20, 50, 100, batch 10*K for 20 batches, one log line.
    result = run(env, "scripts/launch_acquisition.sh")
    assert result.returncode == 0
    plan = [f"{strategy}_K{K}_s{seed}" for seed in (1, 2, 3) for K in (10, 20, 50, 100) for strategy in ("margin", "badge")]
    assert started(result.stdout) == plan
    assert sorted((value(call, "--num_classes"), value(call, "--batch_size"), value(call, "--num_batch"))
                  for call in calls(env, "main.py")) == sorted((str(K), str(10 * K), "20") for K in (10, 20, 50, 100)
                                                               for _ in range(6))
    with open(os.path.join(env["SAVE_DIR"], "logs", "acquisition", "badge_K100_s3.log"), "r") as f:
        assert f.read() == f"trainer_config: {MINSTEPS}\n"


@pytest.mark.parametrize("K, num_batch", [(10, "20"), (20, "39"), (50, "96"), (100, "191")])
def test_round_size_gives_constant_rounds_up_to_the_same_B(env, K, num_batch):
    env.update(ROUND_SIZE="100", KS=str(K))
    assert run(env, "scripts/launch_acquisition.sh", "--first_seed", "1", "--last_seed", "1").returncode == 0
    runs = calls(env, "main.py")
    assert len(runs) == 2 and {value(call, "--num_classes") for call in runs} == {str(K)}
    assert {(value(call, "--batch_size"), value(call, "--num_batch")) for call in runs} == {("100", num_batch)}
    with open(os.path.join(env["SAVE_DIR"], "logs", "acquisition", f"margin_K{K}_s1.log"), "r") as f:
        assert f.readlines()[:2] == [f"trainer_config: {MINSTEPS}\n", "round_size: 100\n"]


def test_round_size_must_reach_the_b_grid(env):
    env["ROUND_SIZE"] = "300"
    result = run(env, "scripts/launch_acquisition.sh")
    assert result.returncode != 0 and "does not divide 10*K=100" in result.stderr
    assert not os.path.exists(env["STUB_LOG"])


def test_finished_run_with_another_round_size_is_refused(env, tmp_path):
    (tmp_path / "save" / "acq").mkdir(parents=True)
    (tmp_path / "save" / "logs" / "acquisition").mkdir(parents=True)
    (tmp_path / "save" / "acq" / "margin_K10_s1_B2000.npy").touch()
    (tmp_path / "save" / "logs" / "acquisition" / "margin_K10_s1.log").write_text(f"trainer_config: {FLOOR}\n")
    env["ROUND_SIZE"] = "100"
    result = run(env, "scripts/launch_acquisition.sh", "--first_seed", "1", "--last_seed", "1")
    assert result.returncode != 0 and "without round_size 100" in result.stderr
    assert not calls(env, "main.py")


def test_constbatch_runs_complete_seeds_in_stages(env):
    result = run(env, "scripts/run_constbatch.sh")
    assert result.returncode == 0, result.stderr
    order = [f"{strategy}_K{K}_s{seed}" for K in (10, 100, 20, 50) for strategy in ("margin", "badge")
             for seed in (1,)]
    assert started(result.stdout)[:8] == order
    assert [name.split("_s")[1] for name in started(result.stdout)] == ["1"] * 8 + ["2"] * 8 + ["3"] * 8 + ["4"] * 8
    assert [values(call, "--seeds") for call in calls(env, "tools/make_queue.py")] == [["1"], ["2", "3"], ["4"]]
    runs = calls(env, "main.py")
    assert {value(call, "--batch_size") for call in runs} == {"100"}
    assert {value(call, "--trainer_config") for call in runs} == {FLOOR}
    assert result.stdout.count("finished: ") == 3


@pytest.mark.parametrize("name", ["ga3_output", "ga3_output_floor500"])
def test_constbatch_refuses_the_directories_of_other_runs(env, tmp_path, name):
    env["SAVE_DIR"] = str(tmp_path / name)
    result = run(env, "scripts/run_constbatch.sh")
    assert result.returncode != 0 and "Refusing" in result.stderr
    assert not (tmp_path / name).exists()


@pytest.mark.parametrize("name", ["DATA_DIR", "PYTHON_BIN", "WANDB_PROJECT_ACQ", "WANDB_PROJECT_EVAL"])
def test_stage_needs_all_settings(env, name):
    env["TRAINER_CONFIG"] = FLOOR
    del env[name]
    result = run(env, "scripts/run_seeds.sh", "1", "3")
    assert result.returncode != 0 and name in result.stderr
    assert not os.path.exists(env["STUB_LOG"])
