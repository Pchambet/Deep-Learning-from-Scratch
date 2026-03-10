import os
import sys

# Ensure repo root is on path (lab/cnn/ runs from Makefile)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf  # noqa: E402, F401
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

from lab.common import save_training_curves, set_seed  # noqa: E402


def load_mnist_cnn():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    # Add channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)


def build_cnn(input_shape=(28, 28, 1), num_classes=10) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def main():
    set_seed(42)
    (x_train, y_train), (x_test, y_test) = load_mnist_cnn()

    model = build_cnn()

    os.makedirs("outputs", exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="outputs/cnn_mnist.keras", save_best_only=True, monitor="val_accuracy"
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=128,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print({"test_accuracy": float(test_acc), "test_loss": float(test_loss)})

    save_training_curves(history, "outputs/cnn_training_curves.png")


if __name__ == "__main__":
    main()
