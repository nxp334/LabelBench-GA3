import json

import pytest

from LabelBench.trainer.utils import effective_max_epoch, should_validate


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


@pytest.mark.parametrize("num_labeled, epochs", [
    (100, 7500),  # Fewer labels than the batch size: one partial batch per epoch.
    (127, 7500),
    (128, 7500),  # Exactly one full batch.
    (255, 7500),  # One full batch, the partial batch is dropped.
    (2000, 500),  # 15 steps per epoch, an exact multiple of the cap.
    (20000, 49),  # 156 steps per epoch: 48 epochs fall short of the cap, 49 give 7644 steps.
])
def test_max_steps_sets_the_epoch_limit(num_labeled, epochs):
    assert effective_max_epoch(20, num_labeled, 128, 500, 7500) == epochs


def test_max_steps_replaces_max_epoch_in_both_directions():
    # 20 epochs would give 3120 steps, so a lower cap also shortens training.
    assert effective_max_epoch(20, 20000, 128, None, 1560) == 10
    assert effective_max_epoch(20, 100, 128, None, 1560) == 1560


def test_max_steps_below_min_steps_is_rejected():
    with pytest.raises(AssertionError):
        effective_max_epoch(20, 100, 128, 500, 499)


def test_max_steps_equal_to_min_steps_is_allowed():
    assert effective_max_epoch(20, 100, 128, 500, 500) == 500


def test_no_validation_before_min_steps():
    assert not should_validate(499, None, 500)
    assert not should_validate(499, None, 500, 100, is_last_epoch=True)


def test_default_validates_every_epoch_as_before():
    assert should_validate(1, None)
    assert should_validate(1, 0)
    assert should_validate(500, None, 500)
    assert should_validate(501, 500, 500)


def test_val_every_steps_spaces_the_checks():
    assert should_validate(500, None, 500, 100)
    assert not should_validate(599, 500, 500, 100)
    assert should_validate(600, 500, 500, 100)


def test_large_epochs_are_checked_every_epoch():
    # K = 100, b = 200: 156 steps per epoch is already more than the spacing.
    assert should_validate(780, 624, 500, 100)


def test_last_epoch_is_always_checked_so_a_best_model_exists():
    assert should_validate(550, 500, 500, 100, is_last_epoch=True)


def test_floor_config_only_adds_the_cap_and_the_check_spacing():
    base_dir = "configs/trainer/cifar100/flexmatch/"
    with open(base_dir + "clip_ViTB32_linear_minsteps.json", "r") as f:
        minsteps = json.load(f)
    with open(base_dir + "clip_ViTB32_linear_floor500.json", "r") as f:
        floor = json.load(f)
    assert floor == dict(minsteps, max_steps=7500, val_every_steps=100, patience=10)
