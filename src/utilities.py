"""Loader for the 64x64 grayscale cat/dog images of Episodes IV-VII (see data/README.md)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_data(
    data_dir: Path = DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``X_train, y_train, X_test, y_test`` from ``trainset.hdf5`` and ``testset.hdf5``.

    Images are ``uint8`` arrays of shape ``(n, 64, 64)``; labels have shape ``(n, 1)``.
    Raises ``FileNotFoundError`` when a file is missing, so no result is ever computed on
    placeholder data.
    """
    train_path, test_path = data_dir / "trainset.hdf5", data_dir / "testset.hdf5"
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found: the cat/dog HDF5 files ship in data/")
    with h5py.File(train_path, "r") as train:
        X_train, y_train = np.array(train["X_train"]), np.array(train["Y_train"])
    with h5py.File(test_path, "r") as test:
        X_test, y_test = np.array(test["X_test"]), np.array(test["Y_test"])
    return X_train, y_train, X_test, y_test
