import random
from typing import Tuple

import numpy as np

try:
    from tensorflow import keras as tf_keras  # type: ignore
except Exception:  # pragma: no cover
    tf_keras = None


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
    except Exception:
        pass


def _load_mnist_np() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if tf_keras is None:
        raise ImportError(
            "TensorFlow is required to use widl.data MNIST loaders. Install tensorflow-macos/tensorflow-metal on Apple Silicon."
        )
    (x_train, y_train), (x_test, y_test) = tf_keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return x_train, y_train, x_test, y_test


def load_mnist_mlp():
    x_train, y_train, x_test, y_test = _load_mnist_np()
    x_train = x_train.reshape((-1, 28 * 28))
    x_test = x_test.reshape((-1, 28 * 28))
    return (x_train, y_train), (x_test, y_test)


def load_mnist_cnn():
    x_train, y_train, x_test, y_test = _load_mnist_np()
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)
