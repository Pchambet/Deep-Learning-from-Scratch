# src/utilities.py
import os

import h5py
import numpy as np


def _synthetic_data(seed: int = 42):
    """
    Generate synthetic train/test data (same format as cats vs dogs HDF5).
    Used when HDF5 files are absent (e.g. fresh clone, CI).
    """
    rng = np.random.default_rng(seed)
    # trainset: 1000 images 64×64, labels 0/1
    X_train = rng.integers(0, 256, (1000, 64, 64), dtype=np.uint8)
    y_train = (rng.random((1000, 1)) > 0.5).astype(np.float64)
    # testset: 200 images 64×64
    X_test = rng.integers(0, 256, (200, 64, 64), dtype=np.uint8)
    y_test = (rng.random((200, 1)) > 0.5).astype(np.float64)
    return X_train, y_train, X_test, y_test


def load_data():
    """
    Load training and test datasets stored in HDF5 files.
    Falls back to synthetic data if files are absent (CI, fresh clone).
    Returns:
        X_train, y_train, X_test, y_test : np.ndarray
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    train_path = os.path.join(data_dir, "trainset.hdf5")
    test_path = os.path.join(data_dir, "testset.hdf5")

    if os.path.isfile(train_path) and os.path.isfile(test_path):
        with h5py.File(train_path, "r") as train_dataset:
            X_train = np.array(train_dataset["X_train"][:])
            y_train = np.array(train_dataset["Y_train"][:])
        with h5py.File(test_path, "r") as test_dataset:
            X_test = np.array(test_dataset["X_test"][:])
            y_test = np.array(test_dataset["Y_test"][:])
        return X_train, y_train, X_test, y_test

    return _synthetic_data()
