#!/usr/bin/env python3
"""
Generate all figures for Episode VII — Horizon of Depth.

Produces high-resolution PNGs for embedding in LaTeX:
  images/circles_boundary.png     — Circles: training curves + decision boundary
  images/moons_boundary.png       — Moons:  training curves + decision boundary
  images/spirals_data.png         — Spiral dataset scatter plot
  images/spirals_boundary.png     — Spirals 3-layer: training + decision boundary
  images/spirals_deep_boundary.png— Spirals 4-layer: training + decision boundary
  images/catsdogs_overfitting.png — Cats vs dogs: train vs test overfitting curves

Usage:
    cd latex/episode_07
    python generate_figures.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

# ── Reproducibility ──────────────────────────────────────────
np.random.seed(42)
OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)
DPI = 300

# ── Core functions (same as notebook) ────────────────────────


def initialisation(dimensions):
    parameters = {}
    for l in range(1, len(dimensions)):
        parameters[f"W{l}"] = np.random.randn(dimensions[l], dimensions[l - 1]) * 0.5
        parameters[f"b{l}"] = np.zeros((dimensions[l], 1))
    return parameters


def forward_propagation(X, parameters):
    activations = {"A0": X}
    L = len(parameters) // 2
    for l in range(1, L + 1):
        Z = parameters[f"W{l}"] @ activations[f"A{l - 1}"] + parameters[f"b{l}"]
        activations[f"A{l}"] = 1 / (1 + np.exp(-Z))
    return activations


def back_propagation(y, parameters, activations):
    m = y.shape[1]
    L = len(parameters) // 2
    gradients = {}
    dA = activations[f"A{L}"] - y
    for l in range(L, 0, -1):
        gradients[f"dW{l}"] = (1 / m) * dA @ activations[f"A{l - 1}"].T
        gradients[f"db{l}"] = (1 / m) * np.sum(dA, axis=1, keepdims=True)
        if l > 1:
            dA = (
                parameters[f"W{l}"].T
                @ dA
                * activations[f"A{l - 1}"]
                * (1 - activations[f"A{l - 1}"])
            )
    return gradients


def update(gradients, parameters, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f"W{l}"] -= learning_rate * gradients[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * gradients[f"db{l}"]
    return parameters


def predict(X, parameters):
    activations = forward_propagation(X, parameters)
    L = len(parameters) // 2
    return (activations[f"A{L}"] >= 0.5).astype(int)


# ── Training (returns history) ───────────────────────────────


def train(X, y, dimensions, lr, epochs, verbose=True):
    parameters = initialisation(dimensions)
    losses, accs = [], []
    rng = tqdm(range(epochs)) if verbose else range(epochs)
    for i in rng:
        activations = forward_propagation(X, parameters)
        L = len(parameters) // 2
        if i % 10 == 0:
            losses.append(log_loss(y.flatten(), activations[f"A{L}"].flatten()))
            accs.append(
                accuracy_score(y.flatten(), (activations[f"A{L}"] >= 0.5).astype(int).flatten())
            )
        gradients = back_propagation(y, parameters, activations)
        parameters = update(gradients, parameters, lr)
    return parameters, losses, accs


def train_with_test(X_train, y_train, X_test, y_test, dimensions, lr, epochs, verbose=True):
    parameters = initialisation(dimensions)
    train_loss, train_acc = [], []
    test_loss, test_acc = [], []
    rng = tqdm(range(epochs)) if verbose else range(epochs)
    for i in rng:
        L = len(parameters) // 2
        activations = forward_propagation(X_train, parameters)
        if i % 100 == 0:
            train_loss.append(log_loss(y_train.flatten(), activations[f"A{L}"].flatten()))
            train_acc.append(
                accuracy_score(y_train.flatten(), predict(X_train, parameters).flatten())
            )
            act_test = forward_propagation(X_test, parameters)
            test_loss.append(log_loss(y_test.flatten(), act_test[f"A{L}"].flatten()))
            test_acc.append(accuracy_score(y_test.flatten(), predict(X_test, parameters).flatten()))
        gradients = back_propagation(y_train, parameters, activations)
        parameters = update(gradients, parameters, lr)
    return parameters, train_loss, train_acc, test_loss, test_acc


# ── Decision boundary helper ─────────────────────────────────


def compute_boundary(X, parameters, resolution=0.02):
    X_plot = X.T if X.shape[0] == 2 else X
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(grid.T, parameters).reshape(xx.shape)
    return xx, yy, Z, X_plot


# ── Spiral generator ─────────────────────────────────────────


def make_spirals(n_samples=1000, noise=0.2, normalize=True):
    theta = np.sqrt(np.random.rand(n_samples, 1)) * 780 * (2 * np.pi / 360)
    x1 = -np.cos(theta) * theta + np.random.rand(n_samples, 1) * noise
    y1 = np.sin(theta) * theta + np.random.rand(n_samples, 1) * noise
    x2 = np.cos(theta) * theta + np.random.rand(n_samples, 1) * noise
    y2 = -np.sin(theta) * theta + np.random.rand(n_samples, 1) * noise
    X = np.vstack((np.hstack((x1, y1)), np.hstack((x2, y2)))).T
    y = np.hstack((np.zeros(n_samples), np.ones(n_samples))).reshape(1, -1)
    if normalize:
        X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)
    return X, y


# ══════════════════════════════════════════════════════════════
#  FIGURE 1 — Circles: training curves + decision boundary
# ══════════════════════════════════════════════════════════════
print("▸ Circles ...")
X_c, y_c = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
X_c = X_c.T
y_c = y_c.reshape(1, -1)
params_c, loss_c, acc_c = train(X_c, y_c, [2, 32, 32, 1], lr=0.1, epochs=500)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].plot(np.arange(len(loss_c)) * 10, loss_c, color="#2962A3")
axes[0].set_title("Learning Curve", fontsize=13)
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[1].plot(np.arange(len(acc_c)) * 10, acc_c, color="#2962A3")
axes[1].set_title("Accuracy", fontsize=13)
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Accuracy")
xx, yy, Z, Xp = compute_boundary(X_c, params_c)
axes[2].contourf(xx, yy, Z, alpha=0.8, cmap="RdYlBu")
axes[2].scatter(Xp[:, 0], Xp[:, 1], c=y_c.flatten(), edgecolors="k", cmap="RdYlBu", s=30)
axes[2].set_title("Decision Boundary", fontsize=13)
axes[2].set_xlabel("Feature 1")
axes[2].set_ylabel("Feature 2")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "circles_boundary.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  ✓ circles_boundary.png")

# ══════════════════════════════════════════════════════════════
#  FIGURE 2 — Moons: training curves + decision boundary
# ══════════════════════════════════════════════════════════════
print("▸ Moons ...")
X_m, y_m = make_moons(n_samples=1000, noise=0.15, random_state=0)
X_m = X_m.T
y_m = y_m.reshape(1, -1)
params_m, loss_m, acc_m = train(X_m, y_m, [2, 64, 64, 1], lr=0.1, epochs=5000)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].plot(np.arange(len(loss_m)) * 10, loss_m, color="#2962A3")
axes[0].set_title("Learning Curve", fontsize=13)
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[1].plot(np.arange(len(acc_m)) * 10, acc_m, color="#2962A3")
axes[1].set_title("Accuracy", fontsize=13)
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Accuracy")
xx, yy, Z, Xp = compute_boundary(X_m, params_m)
axes[2].contourf(xx, yy, Z, alpha=0.8, cmap="RdYlBu")
axes[2].scatter(Xp[:, 0], Xp[:, 1], c=y_m.flatten(), edgecolors="k", cmap="RdYlBu", s=10)
axes[2].set_title("Decision Boundary", fontsize=13)
axes[2].set_xlabel("Feature 1")
axes[2].set_ylabel("Feature 2")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "moons_boundary.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  ✓ moons_boundary.png")

# ══════════════════════════════════════════════════════════════
#  FIGURE 3 — Spirals: scatter plot
# ══════════════════════════════════════════════════════════════
print("▸ Spirals scatter ...")
X_s, y_s = make_spirals(n_samples=1000, noise=0.2)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(X_s[0], X_s[1], c=y_s.flatten(), cmap="summer", s=8)
ax.set_title("Spiral Dataset", fontsize=14)
ax.set_aspect("equal")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "spirals_data.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  ✓ spirals_data.png")

# ══════════════════════════════════════════════════════════════
#  FIGURE 4 — Spirals 3-layer: training + boundary
# ══════════════════════════════════════════════════════════════
print("▸ Spirals 3-layer ...")
params_s3, loss_s3, acc_s3 = train(X_s, y_s, [2, 64, 64, 64, 1], lr=0.1, epochs=5000)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].plot(np.arange(len(loss_s3)) * 10, loss_s3, color="#2962A3")
axes[0].set_title("Learning Curve", fontsize=13)
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[1].plot(np.arange(len(acc_s3)) * 10, acc_s3, color="#2962A3")
axes[1].set_title("Accuracy", fontsize=13)
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Accuracy")
xx, yy, Z, Xp = compute_boundary(X_s, params_s3)
axes[2].contourf(xx, yy, Z, alpha=0.8, cmap="RdYlBu")
axes[2].scatter(Xp[:, 0], Xp[:, 1], c=y_s.flatten(), edgecolors="k", cmap="RdYlBu", s=8)
axes[2].set_title("Decision Boundary — [2,64,64,64,1]", fontsize=12)
axes[2].set_xlabel("Feature 1")
axes[2].set_ylabel("Feature 2")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "spirals_boundary.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  ✓ spirals_boundary.png")

# ══════════════════════════════════════════════════════════════
#  FIGURE 5 — Spirals 4-layer deeper: training + boundary
# ══════════════════════════════════════════════════════════════
print("▸ Spirals 4-layer ...")
params_s4, loss_s4, acc_s4 = train(X_s, y_s, [2, 128, 128, 128, 128, 1], lr=0.1, epochs=5000)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].plot(np.arange(len(loss_s4)) * 10, loss_s4, color="#2962A3")
axes[0].set_title("Learning Curve", fontsize=13)
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[1].plot(np.arange(len(acc_s4)) * 10, acc_s4, color="#2962A3")
axes[1].set_title("Accuracy", fontsize=13)
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Accuracy")
xx, yy, Z, Xp = compute_boundary(X_s, params_s4)
axes[2].contourf(xx, yy, Z, alpha=0.8, cmap="RdYlBu")
axes[2].scatter(Xp[:, 0], Xp[:, 1], c=y_s.flatten(), edgecolors="k", cmap="RdYlBu", s=8)
axes[2].set_title("Decision Boundary — [2,128,128,128,128,1]", fontsize=12)
axes[2].set_xlabel("Feature 1")
axes[2].set_ylabel("Feature 2")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "spirals_deep_boundary.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  ✓ spirals_deep_boundary.png")

# ══════════════════════════════════════════════════════════════
#  FIGURE 6 — Cats vs Dogs overfitting curves
# ══════════════════════════════════════════════════════════════
print("▸ Cats vs Dogs ...")
data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
with h5py.File(os.path.join(data_dir, "trainset.hdf5"), "r") as f:
    X_train = np.array(f["X_train"], dtype=np.float64) / 255.0
    y_train = np.array(f["Y_train"], dtype=np.float64)
with h5py.File(os.path.join(data_dir, "testset.hdf5"), "r") as f:
    X_test = np.array(f["X_test"], dtype=np.float64) / 255.0
    y_test = np.array(f["Y_test"], dtype=np.float64)

X_train = X_train.reshape(X_train.shape[0], -1).T
X_test = X_test.reshape(X_test.shape[0], -1).T
y_train = y_train.reshape(1, -1)
y_test = y_test.reshape(1, -1)

params_cd, tr_l, tr_a, te_l, te_a = train_with_test(
    X_train, y_train, X_test, y_test, [X_train.shape[0], 16, 16, 1], lr=0.01, epochs=5000
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ep = np.arange(len(tr_l)) * 100
axes[0].plot(ep, tr_l, label="Train Loss", color="#2962A3")
axes[0].plot(ep, te_l, label="Test Loss", color="#E07020")
axes[0].legend()
axes[0].set_title("Loss", fontsize=13)
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[1].plot(ep, tr_a, label="Train Accuracy", color="#2962A3")
axes[1].plot(ep, te_a, label="Test Accuracy", color="#E07020")
axes[1].legend()
axes[1].set_title("Accuracy", fontsize=13)
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Accuracy")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "catsdogs_overfitting.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  ✓ catsdogs_overfitting.png")

print("\n✅  All 6 figures generated in images/")
