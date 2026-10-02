"""Generate all figures for Episode VI PDF.

Outputs saved to latex/episode_06/images/:
  - dataset.png         : make_circles scatter (Toxic vs Non-Toxic Plants)
  - training_curves.png : loss + accuracy (toy data, 32 neurons)
  - decision_boundary.png : single decision boundary
  - neuron_comparison.png  : 6 subplots (1,2,4,8,16,32 neurons)
  - sample_images.png     : 8 cat/dog samples
  - overfitting_curves.png: train vs test (cats vs dogs)
  - outputs.json          : numeric values for LaTeX
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

np.random.seed(0)
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

# ── Dataset ─────────────────────────────────────────────────
X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
X = X.T
y = y.reshape((1, y.shape[0]))


# ── Network functions ───────────────────────────────────────
def initialization(n0, n1, n2):
    """Match notebook 06_alive exactly (sigmoid both layers, randn for all params)."""
    np.random.seed(0)
    W1 = np.random.randn(n1, n0)
    b1 = np.random.randn(n1, 1)
    W2 = np.random.randn(n2, n1)
    b2 = np.random.randn(n2, 1)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward_propagation(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    Z1 = W1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))
    return {"A1": A1, "A2": A2}


def log_loss(A, y):
    m = y.shape[1]
    eps = 1e-15
    return -1 / m * np.sum(y * np.log(A + eps) + (1 - y) * np.log(1 - A + eps))


def back_propagation(X, y, parameters, activations):
    A1, A2 = activations["A1"], activations["A2"]
    W2 = parameters["W2"]
    m = y.shape[1]
    dZ2 = A2 - y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update(gradients, parameters, learning_rate):
    return {
        "W1": parameters["W1"] - learning_rate * gradients["dW1"],
        "b1": parameters["b1"] - learning_rate * gradients["db1"],
        "W2": parameters["W2"] - learning_rate * gradients["dW2"],
        "b2": parameters["b2"] - learning_rate * gradients["db2"],
    }


def predict(X, parameters):
    A2 = forward_propagation(X, parameters)["A2"]
    return A2 >= 0.5


def neural_network(X, y, n1=32, learning_rate=0.1, n_epochs=1000):
    n0, n2 = X.shape[0], y.shape[0]
    np.random.seed(0)
    parameters = initialization(n0, n1, n2)
    train_loss, train_acc = [], []
    for _ in range(n_epochs):
        activations = forward_propagation(X, parameters)
        A2 = activations["A2"]
        train_loss.append(log_loss(A2, y))
        train_acc.append(accuracy_score(y.flatten(), (A2 >= 0.5).astype(float).flatten()))
        gradients = back_propagation(X, y, parameters, activations)
        parameters = update(gradients, parameters, learning_rate)
    return parameters, train_loss, train_acc


# ── FIGURE 0: Dataset scatter ─────────────────────────────────
print("Figure 0: dataset.png")
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(X[0, :], X[1, :], c=y.flatten(), cmap="RdBu", edgecolors="k", s=40)
ax.set_xlabel("$x_1$ (leaf length)")
ax.set_ylabel("$x_2$ (leaf width)")
ax.set_title("Toxic vs Non-Toxic Plants", fontweight="bold")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "dataset.png"))
plt.close()

# ── Train on toy data ───────────────────────────────────────
print("Training on make_circles...")
parameters, train_loss, train_acc = neural_network(X, y, n1=32, learning_rate=0.1, n_epochs=1000)

# Collect outputs for JSON
outputs = {
    "toy_circles": {
        "final_loss": float(train_loss[-1]),
        "final_accuracy": float(train_acc[-1]),
        "epochs": 1000,
        "n1": 32,
    }
}

# ── FIGURE 1: Training curves (toy data) ───────────────────
print("Figure 1: training_curves.png")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
ax1.plot(train_loss, color="#29629E", linewidth=1.5)
ax1.set_title("Training Loss", fontweight="bold")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Log-Loss")
ax1.grid(True, alpha=0.3)
ax2.plot(train_acc, color="#2E8B57", linewidth=1.5)
ax2.set_title("Training Accuracy", fontweight="bold")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "training_curves.png"))
plt.close()

# ── FIGURE 2: Decision boundary (32 neurons) ───────────────
print("Figure 2: decision_boundary.png")
x_min, x_max = X[0].min() - 1, X[0].max() + 1
y_min, y_max = X[1].min() - 1, X[1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
grid = np.c_[xx.ravel(), yy.ravel()].T
Z = predict(grid, parameters).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(5, 5))
ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
ax.scatter(X[0], X[1], c=y.flatten(), cmap="RdBu", edgecolors="k", s=40)
ax.set_title("32 neurons \u2014 1000 epochs", fontweight="bold")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "decision_boundary.png"))
plt.close()

# ── FIGURE 3: Neuron count comparison ───────────────────────
print("Figure 3: neuron_comparison.png")
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
neurons_list = [1, 2, 4, 8, 16, 32]
for idx, n1 in enumerate(neurons_list):
    ax = axes[idx // 3, idx % 3]
    params_n, _, _ = neural_network(X, y, n1=n1, learning_rate=0.1, n_epochs=1000)
    Z = predict(grid, params_n).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    ax.scatter(X[0], X[1], c=y.flatten(), cmap="RdBu", edgecolors="k", s=20)
    y_pred = predict(X, params_n)
    acc = accuracy_score(y.flatten(), y_pred.flatten())
    outputs[f"neurons_{n1}"] = {"accuracy": float(acc)}
    ax.set_title(
        f"{n1} neuron{'s' if n1 > 1 else ''} \u2014 acc: {acc:.0%}",
        fontweight="bold",
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
plt.suptitle("Decision Boundaries by Neuron Count", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "neuron_comparison.png"))
plt.close()

# ── FIGURE 4: Sample cat/dog images ────────────────────────
print("Figure 4: sample_images.png")
from src.utilities import load_data

X_train_img, y_train_img, X_test_img, y_test_img = load_data()

fig, axes = plt.subplots(1, 8, figsize=(14, 2.2))
for i in range(8):
    axes[i].imshow(X_train_img[i], cmap="gray")
    axes[i].set_title(
        "Cat" if np.ravel(y_train_img)[i] == 0 else "Dog", fontsize=10, fontweight="bold"
    )
    axes[i].axis("off")
plt.suptitle("Training Samples", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "sample_images.png"))
plt.close()

# ── FIGURE 5: Overfitting curves (train vs test) ───────────
print("Figure 5: overfitting_curves.png (this takes ~90s)...")
X_train_flat = X_train_img.reshape(X_train_img.shape[0], -1).T / 255.0
X_test_flat = X_test_img.reshape(X_test_img.shape[0], -1).T / 255.0
y_train_r = y_train_img.reshape(1, -1)
y_test_r = y_test_img.reshape(1, -1)

n0, n2 = X_train_flat.shape[0], y_train_r.shape[0]
n1_img = 64
np.random.seed(0)
params_img = initialization(n0, n1_img, n2)
train_loss_h, test_loss_h = [], []
train_acc_h, test_acc_h = [], []

for i in range(10000):
    act_train = forward_propagation(X_train_flat, params_img)
    train_loss_h.append(log_loss(act_train["A2"], y_train_r))
    act_test = forward_propagation(X_test_flat, params_img)
    test_loss_h.append(log_loss(act_test["A2"], y_test_r))
    train_acc_h.append(
        accuracy_score(y_train_r.flatten(), (act_train["A2"] >= 0.5).astype(float).flatten())
    )
    test_acc_h.append(
        accuracy_score(y_test_r.flatten(), (act_test["A2"] >= 0.5).astype(float).flatten())
    )
    gradients = back_propagation(X_train_flat, y_train_r, params_img, act_train)
    params_img = update(gradients, params_img, 0.1)
    if (i + 1) % 2000 == 0:
        print(
            f"  epoch {i + 1}/10000 — train acc: {train_acc_h[-1]:.1%}, test acc: {test_acc_h[-1]:.1%}"
        )

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
ax1.plot(train_loss_h, label="Train", color="#29629E", linewidth=1.5)
ax1.plot(test_loss_h, label="Test", color="#CC7722", linewidth=1.5, linestyle="--")
ax1.set_title("Loss", fontweight="bold")
ax1.set_xlabel("Epochs")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.plot(train_acc_h, label="Train", color="#29629E", linewidth=1.5)
ax2.plot(test_acc_h, label="Test", color="#CC7722", linewidth=1.5, linestyle="--")
ax2.set_title("Accuracy", fontweight="bold")
ax2.set_xlabel("Epochs")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.suptitle("Cats vs Dogs \u2014 64 neurons, lr=0.1", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "overfitting_curves.png"))
plt.close()

outputs["cats_dogs"] = {
    "final_train_accuracy": float(train_acc_h[-1]),
    "final_test_accuracy": float(test_acc_h[-1]),
    "final_train_loss": float(train_loss_h[-1]),
    "final_test_loss": float(test_loss_h[-1]),
    "epochs": 10000,
    "n1": 64,
}

with open(os.path.join(OUT, "outputs.json"), "w") as f:
    json.dump(outputs, f, indent=2)

print(f"\nDone! All figures saved to {OUT}/")
for f in sorted(os.listdir(OUT)):
    print(f"  {f} — {os.path.getsize(os.path.join(OUT, f)) // 1024} KB")
