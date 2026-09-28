"""Notebook utilities.

This module provides a small helper to load local HDF5 datasets used in practice notebooks.
Prefer using the shared widl.data loaders for standard datasets (e.g., MNIST).
"""

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np


def load_local_hdf5_dataset(root: Path | str = Path("datasets")) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a local HDF5 train/test dataset from the notebooks/datasets folder.

    Expects the following files and keys:
    - datasets/trainset.hdf5 with keys X_train, Y_train
    - datasets/testset.hdf5 with keys X_test, Y_test

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    root = Path(root)
    train_path = root / "trainset.hdf5"
    test_path = root / "testset.hdf5"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected dataset files at {train_path} and {test_path}. Please provide them or adjust the path."
        )

    with h5py.File(train_path, "r") as train_dataset:
        X_train = np.array(train_dataset["X_train"][:])
        y_train = np.array(train_dataset["Y_train"][:])

    with h5py.File(test_path, "r") as test_dataset:
        X_test = np.array(test_dataset["X_test"][:])
        y_test = np.array(test_dataset["Y_test"][:])

    return X_train, y_train, X_test, y_test


# Backwards compatibility: previous notebooks imported `load_data()`
def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return load_local_hdf5_dataset()