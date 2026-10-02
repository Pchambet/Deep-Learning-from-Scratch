#!/usr/bin/env python3
"""Regenerate the README figures and docs/results.json from the from-scratch code.

Three experiments, all trained with src/deep_network.py (full-batch gradient descent,
tanh hidden units, sigmoid output, seeded):

1. Two interleaved spirals, the textbook problem no linear model can solve: decision
   boundaries of a single neuron, one hidden layer and three hidden layers.
2. A capacity sweep on the same spirals, five seeds per architecture: does accuracy
   come from depth or simply from more parameters? Single layers go up to 256 units
   (1,025 parameters) so that both families cover the same parameter range, and the
   widest ones are also trained four times longer to separate capacity from training time.
3. The 64x64 cat/dog photos of Episodes IV-VII: train vs test accuracy of a deep
   fully connected network, i.e. what the series' networks can and cannot do on images.

Plus a finite-difference gradient check of back-propagation at several depths.
Runtime: about 5 minutes on a laptop CPU.
"""

from __future__ import annotations

import json
import sys
import time
from functools import cache
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.deep_network import (
    accuracy,
    backward_propagation,
    fit,
    forward_propagation,
    initialize_parameters,
    log_loss,
    predict,
    predict_proba,
)
from src.gradient_check import numerical_gradient, relative_error
from src.utilities import load_data

FIGURES = ROOT / "docs" / "figures"
RESULTS = ROOT / "docs" / "results.json"

INK, TEAL, AMBER, SLATE, GRID = "#0f172a", "#0d9488", "#d97706", "#64748b", "#e2e8f0"
SEEDS = (0, 1, 2, 3, 4)
SPIRAL_EPOCHS, SPIRAL_LR = 5000, 0.5
HERO_MODELS = {
    "Single neuron": [2, 1],
    "1 hidden layer (16)": [2, 16, 1],
    "3 hidden layers (16-16-16)": [2, 16, 16, 16, 1],
}
WIDTHS = (4, 8, 16, 32, 64, 128, 256)  # one hidden layer of this many units
DEPTHS = (1, 2, 3, 4)  # this many hidden layers of 16 units
LONG_WIDTHS, LONG_EPOCHS = (128, 256), 20_000  # wide single layers, trained 4x longer (seed 0)
IMAGE_DIMS, IMAGE_LR, IMAGE_EPOCHS = [4096, 32, 32, 1], 0.02, 3000

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": SLATE,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def make_spirals(n_per_class: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaved spirals (the generator of Episode VII), standardised."""
    rng = np.random.default_rng(seed)
    theta = np.sqrt(rng.random(n_per_class)) * np.deg2rad(780)
    arm = np.c_[-np.cos(theta) * theta, np.sin(theta) * theta]
    X = np.vstack([arm + rng.random(arm.shape) * noise, -arm + rng.random(arm.shape) * noise])
    y = np.r_[np.zeros(n_per_class), np.ones(n_per_class)]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y


def spiral_split(seed: int = 0):
    """Fixed 70/30 train/test split, samples as columns."""
    X, y = make_spirals(500, noise=0.5, seed=seed)
    order = np.random.default_rng(seed).permutation(len(y))
    train, test = order[:700], order[700:]
    return X[train].T, y[train][None, :], X[test].T, y[test][None, :]


def n_parameters(dimensions: list[int]) -> int:
    return sum((a + 1) * b for a, b in pairwise(dimensions))


def binomial_ci95(p: float, n: int) -> float:
    """Half-width of the normal-approximation 95% interval of an accuracy measured on n items."""
    return 1.96 * float(np.sqrt(p * (1 - p) / n))


def train_spirals(dimensions: list[int], seed: int, data, epochs: int = SPIRAL_EPOCHS):
    X_train, y_train, X_test, y_test = data
    run = fit(X_train, y_train, dimensions, learning_rate=SPIRAL_LR, epochs=epochs, seed=seed)
    return run.parameters, {
        "train_accuracy": accuracy(y_train, predict(X_train, run.parameters)),
        "test_accuracy": accuracy(y_test, predict(X_test, run.parameters)),
    }


def hero_figure(data, results: dict) -> None:
    _, _, X_test, y_test = data
    xx, yy = np.meshgrid(np.linspace(-2.4, 2.4, 400), np.linspace(-2.4, 2.4, 400))
    grid = np.c_[xx.ravel(), yy.ravel()].T
    regions = ListedColormap(["#fde7c8", "#cdeeea"])  # light amber / light teal

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    hero = {}
    for ax, (name, dims) in zip(axes, HERO_MODELS.items(), strict=True):
        parameters, scores = train_spirals(dims, seed=0, data=data)
        hero[name] = {"dimensions": dims, "parameters": n_parameters(dims), **scores}
        zz = predict(grid, parameters).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], cmap=regions)
        ax.contour(xx, yy, zz, levels=[0.5], colors=SLATE, linewidths=0.8)
        for label, color in ((0, AMBER), (1, TEAL)):
            mask = y_test[0] == label
            ax.scatter(*X_test[:, mask], s=10, color=color, edgecolors="white", linewidths=0.3)
        ax.set_title(
            f"{name}\n{scores['test_accuracy']:.1%} held-out accuracy",
            fontsize=11,
            loc="left",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
    acc = [hero[name]["test_accuracy"] for name in HERO_MODELS]
    fig.suptitle(
        f"Same data, same gradient descent: stacking layers takes held-out accuracy "
        f"from {acc[0]:.1%} to {acc[-1]:.1%}",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.01,
        "Two interleaved spirals, 700 training / 300 test points (shown). One run per panel "
        f"(seed 0; 5-seed means in the README table), {SPIRAL_EPOCHS:,} full-batch epochs, "
        f"learning rate {SPIRAL_LR}.",
        fontsize=8.5,
        color=SLATE,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(FIGURES / "hero_spirals.png", dpi=200)
    plt.close(fig)
    results["spirals_hero"] = hero


def capacity_sweep(data, results: dict) -> None:
    @cache
    def seed_scores(dims: tuple[int, ...]) -> tuple[float, ...]:
        # [2, 16, 1] belongs to both families: train it once.
        return tuple(train_spirals(list(dims), seed, data)[1]["test_accuracy"] for seed in SEEDS)

    wide_label = f"wider: 1 hidden layer of {WIDTHS[0]}-{WIDTHS[-1]} units"
    deep_label = f"deeper: {DEPTHS[0]}-{DEPTHS[-1]} hidden layers of 16 units"
    families = {
        wide_label: [[2, w, 1] for w in WIDTHS],
        deep_label: [[2, *([16] * d), 1] for d in DEPTHS],
    }
    sweep = {}
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for (family, architectures), color in zip(families.items(), (AMBER, TEAL), strict=True):
        rows = []
        for dims in architectures:
            scores = seed_scores(tuple(dims))
            rows.append(
                {
                    "dimensions": dims,
                    "parameters": n_parameters(dims),
                    "test_accuracy_mean": float(np.mean(scores)),
                    "test_accuracy_min": float(np.min(scores)),
                    "test_accuracy_max": float(np.max(scores)),
                }
            )
            print(f"  {dims}: test accuracy {np.mean(scores):.3f} (min {np.min(scores):.3f})")
        sweep[family] = rows
        p = np.array([r["parameters"] for r in rows])
        mean = np.array([r["test_accuracy_mean"] for r in rows])
        low = mean - np.array([r["test_accuracy_min"] for r in rows])
        high = np.array([r["test_accuracy_max"] for r in rows]) - mean
        ax.errorbar(
            p,
            mean,
            yerr=[low, high],
            color=color,
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=3,
            elinewidth=1,
        )
        # The deep family ends among the open 20,000-epoch markers, so its label sits
        # under its second point, where it cannot be read as theirs; the wide family is
        # labelled under its first point, in the empty lower left.
        if color == TEAL:
            anchor, offset, text = 1, (8, -10), family.replace(" of 16", "\nof 16")
        else:
            anchor, offset, text = 0, (-4, -14), family
        ax.annotate(
            text,
            (p[anchor], mean[anchor]),
            xytext=offset,
            textcoords="offset points",
            va="top",
            color=INK,
            fontsize=9,
        )
    long_runs = []
    for width in LONG_WIDTHS:
        dims = [2, width, 1]
        scores = train_spirals(dims, seed=0, data=data, epochs=LONG_EPOCHS)[1]
        long_runs.append({"dimensions": dims, "parameters": n_parameters(dims), **scores})
        print(f"  {dims}, {LONG_EPOCHS:,} epochs: test accuracy {scores['test_accuracy']:.3f}")
    ax.scatter(
        [r["parameters"] for r in long_runs],
        [r["test_accuracy"] for r in long_runs],
        s=46,
        facecolors="white",
        edgecolors=AMBER,
        linewidths=1.8,
        zorder=3,
    )
    ax.annotate(
        f"single layers, {LONG_EPOCHS:,} epochs (seed 0)",
        (long_runs[0]["parameters"], long_runs[0]["test_accuracy"]),
        xytext=(-8, 6),
        textcoords="offset points",
        ha="right",
        color=INK,
        fontsize=9,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Held-out accuracy (mean, min-max over 5 seeds)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(right=max(r["parameters"] for r in long_runs) * 1.3)
    ax.axhline(0.5, color=SLATE, linewidth=1, linestyle=":")
    ax.text(ax.get_xlim()[0] * 1.1, 0.51, "chance", color=SLATE, fontsize=8.5)
    best_wide = max(sweep[wide_label], key=lambda r: r["test_accuracy_mean"])
    two_layers = sweep[deep_label][1]
    best_long = max(long_runs, key=lambda r: r["test_accuracy"])
    ax.set_title(
        f"{SPIRAL_EPOCHS:,} epochs: two 16-unit layers reach "
        f"{two_layers['test_accuracy_mean']:.1%}, the best single layer "
        f"{best_wide['test_accuracy_mean']:.1%}.\n{LONG_EPOCHS:,} epochs: one layer of "
        f"{best_long['dimensions'][1]} units reaches {best_long['test_accuracy']:.1%}. "
        "Depth buys speed here, not capacity",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        f"Filled: mean and min-max over {len(SEEDS)} seeds. Open: one seed. Full-batch "
        f"gradient descent, learning rate {SPIRAL_LR} for every model (not tuned).",
        fontsize=8.5,
        color=SLATE,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "capacity_sweep.png", dpi=200)
    plt.close(fig)
    results["spirals_capacity_sweep"] = sweep
    results["spirals_wide_long_training"] = {"epochs": LONG_EPOCHS, "seed": 0, "runs": long_runs}


def cats_dogs(results: dict) -> None:
    X_train, y_train, X_test, y_test = load_data()
    X_train = X_train.reshape(len(X_train), -1).T / 255.0
    X_test = X_test.reshape(len(X_test), -1).T / 255.0
    y_train, y_test = y_train.T, y_test.T
    record_every = 25
    run = fit(
        X_train,
        y_train,
        IMAGE_DIMS,
        learning_rate=IMAGE_LR,
        epochs=IMAGE_EPOCHS,
        seed=0,
        X_val=X_test,
        y_val=y_test,
        record_every=record_every,
    )
    epochs = [min(i * record_every, IMAGE_EPOCHS - 1) for i in range(len(run.train_accuracy))]
    final_train = accuracy(y_train, predict(X_train, run.parameters))
    final_test = accuracy(y_test, predict(X_test, run.parameters))
    n_test = y_test.shape[1]

    final_train_loss = log_loss(y_train, predict_proba(X_train, run.parameters))
    final_test_loss = log_loss(y_test, predict_proba(X_test, run.parameters))

    fig, (ax, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax.plot(epochs, run.train_accuracy, color=AMBER, linewidth=2)
    ax.plot(epochs, run.val_accuracy, color=TEAL, linewidth=2)
    ax.axhline(0.5, color=SLATE, linewidth=1, linestyle=":")
    ax.text(epochs[-1], final_train, f"  train {final_train:.0%}", color=INK, va="center")
    ax.text(epochs[-1], final_test, f"  test {final_test:.0%}", color=INK, va="center")
    ax.text(epochs[-1] / 2, 0.49, "chance (balanced classes)", color=SLATE, fontsize=8.5, va="top")
    ax.set_ylim(0.4, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Epoch (full-batch gradient descent)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy", loc="left", fontsize=10.5)

    ax_loss.plot(epochs, run.train_loss, color=AMBER, linewidth=2)
    ax_loss.plot(epochs, run.val_loss, color=TEAL, linewidth=2)
    ax_loss.text(epochs[-1], final_train_loss, f"  train {final_train_loss:.2f}", color=INK)
    ax_loss.text(epochs[-1], final_test_loss, f"  test {final_test_loss:.2f}", color=INK)
    ax_loss.set_ylim(bottom=0)
    # Same epoch range on both panels; the room past the last epoch holds the end labels.
    for a in (ax, ax_loss):
        a.set_xlim(0, IMAGE_EPOCHS * 1.18)
        a.set_xticks(range(0, IMAGE_EPOCHS + 1, 500))
        a.spines["bottom"].set_bounds(0, IMAGE_EPOCHS)
    ax_loss.set_xlabel("Epoch (full-batch gradient descent)")
    ax_loss.set_ylabel("Log-loss")
    ax_loss.set_title("Log-loss", loc="left", fontsize=10.5)

    fig.suptitle(
        f"Cat/dog photos: {final_train:.0%} right on training images, {final_test:.0%} on "
        f"unseen ones; test loss ends {final_test_loss / final_train_loss:.0f}x the training loss",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.01,
        f"Fully connected {'-'.join(map(str, IMAGE_DIMS))} network, 1,000 training / "
        f"{n_test} test images. 95% interval on a {n_test}-image accuracy: about "
        f"±{binomial_ci95(final_test, n_test):.0%}.",
        fontsize=8.5,
        color=SLATE,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(FIGURES / "cats_dogs_overfitting.png", dpi=200)
    plt.close(fig)
    results["cats_dogs"] = {
        "dimensions": IMAGE_DIMS,
        "parameters": n_parameters(IMAGE_DIMS),
        "epochs": IMAGE_EPOCHS,
        "learning_rate": IMAGE_LR,
        "final_train_accuracy": final_train,
        "final_test_accuracy": final_test,
        "best_test_accuracy_during_training": max(run.val_accuracy),
        "final_train_loss": final_train_loss,
        "final_test_loss": final_test_loss,
        "test_accuracy_ci95_half_width": binomial_ci95(final_test, n_test),
    }


def gradient_checks(results: dict) -> None:
    X_train, y_train, _, _ = spiral_split()
    X, y = X_train[:, :50], y_train[:, :50]
    checks = {}
    for dims in ([2, 1], [2, 8, 1], [2, 8, 8, 1], [2, 8, 8, 8, 1]):
        for activation in ("tanh", "sigmoid"):
            parameters = initialize_parameters(dims, seed=0)
            grads = backward_propagation(
                y, parameters, forward_propagation(X, parameters, activation), activation
            )

            def loss(parameters=parameters, activation=activation) -> float:
                return log_loss(y, predict_proba(X, parameters, activation))

            worst = max(
                relative_error(grads[f"d{name}"], numerical_gradient(loss, value))
                for name, value in parameters.items()
            )
            checks[f"{'-'.join(map(str, dims))} {activation}"] = worst
    results["gradient_check_max_relative_error"] = checks


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    start = time.perf_counter()
    data = spiral_split()
    steps = (
        ("gradient checks", lambda: gradient_checks(results)),
        ("hero figure", lambda: hero_figure(data, results)),
        ("capacity sweep", lambda: capacity_sweep(data, results)),
        ("cats vs dogs", lambda: cats_dogs(results)),
    )
    for label, step in steps:
        t0 = time.perf_counter()
        step()
        print(f"{label}: {time.perf_counter() - t0:.0f} s")
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")
    print(f"total {time.perf_counter() - start:.0f} s; wrote {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
