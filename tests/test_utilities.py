"""Tests for src/utilities.py: the cat/dog loader."""

import numpy as np
import pytest

from src.utilities import load_data


def test_committed_dataset_shapes_and_balance():
    X_train, y_train, X_test, y_test = load_data()
    assert X_train.shape == (1000, 64, 64)
    assert X_test.shape == (200, 64, 64)
    assert y_train.shape == (1000, 1)
    assert y_test.shape == (200, 1)
    # Balanced classes: 500/500 for training, 100/100 for testing.
    assert np.unique(y_train, return_counts=True)[1].tolist() == [500, 500]
    assert np.unique(y_test, return_counts=True)[1].tolist() == [100, 100]


def test_pixels_are_8_bit_grayscale():
    X_train, _, X_test, _ = load_data()
    for X in (X_train, X_test):
        assert X.dtype == np.uint8
        assert X.max() > 200  # real images use the full range, not a [0, 1] rescale


def test_missing_files_raise_instead_of_returning_placeholder_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="trainset.hdf5"):
        load_data(tmp_path)
