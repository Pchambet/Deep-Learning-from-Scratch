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


def save_training_curves(history, output_path: str, dpi: int = 150) -> None:
    """Save train/validation loss and accuracy curves from a Keras History object."""
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in ((ax_loss, "loss"), (ax_acc, "accuracy")):
        ax.plot(history.history[metric], label="train")
        ax.plot(history.history[f"val_{metric}"], label="validation")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
