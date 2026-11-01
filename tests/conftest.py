import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def train_targets() -> np.ndarray:
    """Synthetic CIFAR-100 train labels: 500 images per class, in random order."""
    return np.random.RandomState(0).permutation(np.repeat(np.arange(100), 500))


@pytest.fixture
def test_targets() -> np.ndarray:
    """Synthetic CIFAR-100 test labels: 100 images per class, in random order."""
    return np.random.RandomState(1).permutation(np.repeat(np.arange(100), 100))
