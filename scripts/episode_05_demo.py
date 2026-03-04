#!/usr/bin/env python3
"""Episode 5 demo: first two-layer neural network on make_circles."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.two_layer_network import fit_two_layer_network, predict  # noqa: E402


def plot_training_curves(loss_values, accuracy_values, output_path):
    fig, axis = plt.subplots(1, 2, figsize=(10, 4))

    axis[0].plot(loss_values)
    axis[0].set_title("Episode 5 - Loss")
    axis[0].set_xlabel("Epoch")

    axis[1].plot(accuracy_values)
    axis[1].set_title("Episode 5 - Accuracy")
    axis[1].set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_decision_boundary(x, y, parameters, output_path, resolution=0.01):
    x_min, x_max = x[0, :].min() - 1, x[0, :].max() + 1
    y_min, y_max = x[1, :].min() - 1, x[1, :].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution),
    )

    grid = np.c_[xx.ravel(), yy.ravel()].T
    boundary = predict(grid, parameters).reshape(xx.shape)

    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, boundary, alpha=0.4, cmap=plt.cm.Spectral)
    plt.scatter(x[0, :], x[1, :], c=y.flatten(), edgecolors="k", cmap=plt.cm.Spectral)
    plt.title("Episode 5 - Decision boundary")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    features, labels = make_circles(n_samples=400, noise=0.08, factor=0.35, random_state=42)
    x = features.T
    y = labels.reshape(1, -1)

    run = fit_two_layer_network(
        inputs=x,
        labels=y,
        n1=32,
        learning_rate=0.2,
        epochs=3000,
        seed=42,
    )

    output_dir = os.path.join(ROOT, "assets", "figures", "episode_05")
    os.makedirs(output_dir, exist_ok=True)

    plot_training_curves(run["loss"], run["accuracy"], os.path.join(output_dir, "training_curves.png"))
    plot_decision_boundary(x, y, run["parameters"], os.path.join(output_dir, "decision_boundary.png"))

    final_accuracy = run["accuracy"][-1]
    final_loss = run["loss"][-1]
    print({"final_accuracy": float(final_accuracy), "final_loss": float(final_loss)})


if __name__ == "__main__":
    main()
