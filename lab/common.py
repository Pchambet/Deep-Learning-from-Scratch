"""Shared utilities for lab scripts (MNIST, CNN)."""

import random

import numpy as np


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility (Python, NumPy, TensorFlow)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def save_training_curves(history, output_path: str, dpi: int = 150) -> bool:
    """
    Save loss and accuracy curves from a Keras History object.
    Returns True on success, False on failure (logs warning).
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="val")
        plt.title("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="train")
        plt.plot(history.history["val_accuracy"], label="val")
        plt.title("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi)
        plt.close()
        return True
    except Exception as e:
        print(f"Warning: could not save curves: {e}")
        return False
