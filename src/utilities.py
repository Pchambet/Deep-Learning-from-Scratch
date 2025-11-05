# src/utilities.py
import h5py
import numpy as np
import os

def load_data():
    """
    Load training and test datasets stored in HDF5 files.
    Returns:
        X_train, y_train, X_test, y_test : np.ndarray
    """

    # Build absolute paths so notebooks can run from anywhere
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")

    train_path = os.path.join(data_dir, "trainset.hdf5")
    test_path = os.path.join(data_dir, "testset.hdf5")

    with h5py.File(train_path, "r") as train_dataset:
        X_train = np.array(train_dataset["X_train"][:])
        y_train = np.array(train_dataset["Y_train"][:])

    with h5py.File(test_path, "r") as test_dataset:
        X_test = np.array(test_dataset["X_test"][:])
        y_test = np.array(test_dataset["Y_test"][:])

    return X_train, y_train, X_test, y_test