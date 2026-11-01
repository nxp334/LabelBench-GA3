import json

from LabelBench.trainer.utils import effective_max_epoch


def test_no_min_steps_keeps_max_epoch():
    assert effective_max_epoch(20, 100, 128) == 20
    assert effective_max_epoch(20, 100, 128, None) == 20


def test_fewer_labels_than_batch_size_gives_one_step_per_epoch():
    assert effective_max_epoch(20, 100, 128, 500) == 500


def test_last_partial_batch_is_dropped():
    # 255 labels give one full batch, 2000 labels give 15.
    assert effective_max_epoch(1, 255, 128, 500) == 500
    assert effective_max_epoch(1, 2000, 128, 150) == 10


def test_extends_epochs_to_reach_min_steps():
    # K = 10, b = 200: 15 steps per epoch, so 33 epochs fall short and 34 are enough.
    assert effective_max_epoch(20, 2000, 128, 500) == 34


def test_already_sufficient_keeps_max_epoch():
    # K = 100, b = 200: 156 steps per epoch over 20 epochs is far above 500.
    assert effective_max_epoch(20, 20000, 128, 500) == 20


def test_exact_multiple_needs_no_extra_epoch():
    assert effective_max_epoch(1, 1280, 128, 500) == 50


def test_minsteps_config_only_adds_min_steps_and_serial_loading():
    base_dir = "configs/trainer/cifar100/flexmatch/"
    with open(base_dir + "clip_ViTB32_linear.json", "r") as f:
        linear = json.load(f)
    with open(base_dir + "clip_ViTB32_linear_minsteps.json", "r") as f:
        minsteps = json.load(f)
    assert minsteps == dict(linear, min_steps=500, num_workers=0)
