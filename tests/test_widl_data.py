import importlib
import numpy as np
import pytest


def test_set_seed_idempotent():
    wd = importlib.import_module("widl.data")
    wd.set_seed(123)
    a = np.random.rand(3)
    wd.set_seed(123)
    b = np.random.rand(3)
    assert np.allclose(a, b)


@pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="TensorFlow not installed")
def test_mnist_shapes():
    wd = importlib.import_module("widl.data")
    (x_train, y_train), (x_test, y_test) = wd.load_mnist_mlp()
    assert x_train.shape[1] == 784
    assert x_test.shape[1] == 784
    assert x_train.shape[0] > 10000 and x_test.shape[0] > 1000
