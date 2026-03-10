"""Tests for src/utilities.py — load_data, shapes, fallback."""

import numpy as np

from src.utilities import load_data


class TestLoadData:
    """load_data returns correct shapes and dtypes (HDF5 or synthetic fallback)."""

    def test_returns_four_arrays(self):
        X_train, y_train, X_test, y_test = load_data()
        assert X_train is not None
        assert y_train is not None
        assert X_test is not None
        assert y_test is not None

    def test_shapes(self):
        """Expected: 1000 train, 200 test, images 64x64."""
        X_train, y_train, X_test, y_test = load_data()
        assert X_train.shape == (1000, 64, 64)
        assert y_train.shape in ((1000, 1), (1000,))
        assert X_test.shape == (200, 64, 64)
        assert y_test.shape in ((200, 1), (200,))

    def test_dtypes(self):
        X_train, y_train, X_test, y_test = load_data()
        assert X_train.dtype in (np.uint8, np.float32, np.float64)
        assert np.issubdtype(y_train.dtype, np.floating) or np.issubdtype(
            y_train.dtype, np.integer
        )
        assert X_test.dtype in (np.uint8, np.float32, np.float64)
        assert np.issubdtype(y_test.dtype, np.floating) or np.issubdtype(
            y_test.dtype, np.integer
        )

    def test_x_in_valid_range(self):
        """X values in [0, 255] for uint8 or normalized."""
        X_train, _, X_test, _ = load_data()
        assert X_train.min() >= 0
        assert X_train.max() <= 256  # uint8 max 255, float can be 1.0
        assert X_test.min() >= 0
        assert X_test.max() <= 256

    def test_labels_binary(self):
        _, y_train, _, y_test = load_data()
        yt = np.ravel(y_train)
        ye = np.ravel(y_test)
        assert set(np.unique(yt)).issubset({0, 1, 0.0, 1.0})
        assert set(np.unique(ye)).issubset({0, 1, 0.0, 1.0})

    def test_deterministic_when_synthetic(self):
        """Same seed → same data when using fallback (no HDF5)."""
        X1, y1, _, _ = load_data()
        X2, y2, _, _ = load_data()
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
