import csv
import json

from tools.collect_results import COLUMNS, collect_mode
from tools.make_queue import ALL_CONDITIONS


def write_result(save_dir, condition, K, s, B, accuracy, **fields):
    folder = save_dir / "eval" / "proxy"
    folder.mkdir(parents=True, exist_ok=True)
    result = {"K": K, "s": s, "B": B, "b": B / K, "condition": condition, "mode": "proxy", "test_accuracy": accuracy,
              "per_class_test_accuracy": [accuracy] * K, **fields}
    (folder / f"{condition}_K{K}_s{s}_B{B}.json").write_text(json.dumps(result))


def read_rows(save_dir):
    with open(save_dir / "results_proxy.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_one_row_per_result_and_shared_init(tmp_path):
    write_result(tmp_path, "random", 10, 1, 100, 0.5)
    write_result(tmp_path, "margin", 10, 1, 200, 0.7)
    write_result(tmp_path, "random", 10, 1, 200, 0.6)

    assert collect_mode(str(tmp_path), "proxy") == 7
    rows = read_rows(tmp_path)
    init_rows = [row for row in rows if row["B"] == "100"]
    assert sorted(row["condition"] for row in init_rows) == sorted(ALL_CONDITIONS)
    assert {row["test_accuracy"] for row in init_rows} == {"0.5"}
    assert {row["evaluated_as"] for row in init_rows} == {"random"}
    assert len({(row["K"], row["s"], row["B"], row["condition"]) for row in rows}) == 7


def test_later_rounds_are_not_repeated(tmp_path):
    write_result(tmp_path, "random", 10, 1, 200, 0.6)
    collect_mode(str(tmp_path), "proxy")
    assert [row["condition"] for row in read_rows(tmp_path)] == ["random"]


def test_tasks_stay_separate(tmp_path):
    write_result(tmp_path, "random", 10, 1, 100, 0.5)
    write_result(tmp_path, "random", 10, 2, 100, 0.4)
    write_result(tmp_path, "random", 20, 1, 200, 0.3)
    assert collect_mode(str(tmp_path), "proxy") == 15


def test_condition_evaluated_at_init_keeps_its_own_result(tmp_path):
    write_result(tmp_path, "random", 10, 1, 100, 0.5)
    write_result(tmp_path, "margin", 10, 1, 100, 0.9)
    collect_mode(str(tmp_path), "proxy")
    margin = next(row for row in read_rows(tmp_path) if row["condition"] == "margin")
    assert margin["test_accuracy"] == "0.9" and margin["evaluated_as"] == "margin"


def test_results_from_before_ended_by_get_an_empty_cell(tmp_path):
    write_result(tmp_path, "random", 10, 1, 200, 0.6)
    collect_mode(str(tmp_path), "proxy")
    assert read_rows(tmp_path)[0]["ended_by"] == ""


def test_every_row_records_trainer_config_and_ended_by(tmp_path):
    write_result(tmp_path, "random", 10, 1, 100, 0.5, trainer_config="floor.json", ended_by="early_stopping")
    write_result(tmp_path, "margin", 10, 1, 200, 0.7, trainer_config="floor.json", ended_by="epoch_limit")
    collect_mode(str(tmp_path), "proxy")

    rows = read_rows(tmp_path)
    assert len(rows) == 6 and {row["trainer_config"] for row in rows} == {"floor.json"}
    assert {row["ended_by"] for row in rows if row["B"] == "100"} == {"early_stopping"}
    assert [row["ended_by"] for row in rows if row["B"] == "200"] == ["epoch_limit"]


def test_best_step_is_the_last_column_so_other_columns_keep_their_number():
    assert COLUMNS[-1] == "best_step"
    assert COLUMNS.index("ended_by") == 16 and COLUMNS.index("trainer_config") == 19 and len(COLUMNS) == 22


def test_best_step_reaches_the_csv(tmp_path):
    write_result(tmp_path, "margin", 10, 1, 200, 0.7, best_step=600)
    collect_mode(str(tmp_path), "proxy")
    assert read_rows(tmp_path)[0]["best_step"] == "600"


def test_results_from_before_best_step_get_an_empty_cell(tmp_path):
    write_result(tmp_path, "margin", 10, 1, 200, 0.7)
    collect_mode(str(tmp_path), "proxy")
    assert read_rows(tmp_path)[0]["best_step"] == ""
