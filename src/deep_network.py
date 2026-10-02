"""L-layer fully connected network for binary classification, in plain NumPy.

This is the code of Episode VII (*Horizon of Depth*) as a tested module: the two-layer
network of Episode VI with the hard-coded layer indices replaced by loops.

Conventions follow the PDFs: inputs ``X`` have shape ``(n0, m)`` (one column per
sample), labels ``y`` have shape ``(1, m)``, and the parameters are stored as
``{"W1", "b1", ..., "WL", "bL"}`` where ``dimensions = [n0, n1, ..., nL]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Parameters = dict[str, np.ndarray]

HIDDEN_ACTIVATIONS = ("tanh", "sigmoid")


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Clipping keeps np.exp finite for very negative inputs without changing the
    # result at float64 precision (sigmoid(-500) is ~7e-218).
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def log_loss(y: np.ndarray, probabilities: np.ndarray, epsilon: float = 1e-12) -> float:
    """Mean binary cross-entropy; probabilities are clipped so log(0) never happens."""
    p = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def initialize_parameters(dimensions: list[int], seed: int = 0) -> Parameters:
    """Random weights scaled by 1/sqrt(fan-in), zero biases.

    The scaling keeps every pre-activation of order one at initialisation, so tanh
    and sigmoid units start in their sensitive range instead of saturating, which is
    what stalls deep sigmoid networks initialised with unit-variance weights.
    """
    if len(dimensions) < 2:
        raise ValueError("dimensions needs at least an input and an output size")
    rng = np.random.default_rng(seed)
    parameters: Parameters = {}
    for layer in range(1, len(dimensions)):
        fan_in = dimensions[layer - 1]
        parameters[f"W{layer}"] = rng.standard_normal((dimensions[layer], fan_in)) / np.sqrt(fan_in)
        parameters[f"b{layer}"] = np.zeros((dimensions[layer], 1))
    return parameters


def n_layers(parameters: Parameters) -> int:
    return len(parameters) // 2


def forward_propagation(
    X: np.ndarray, parameters: Parameters, hidden_activation: str = "tanh"
) -> dict[str, np.ndarray]:
    """Return every activation ``A0 = X, A1, ..., AL``; ``AL`` is the sigmoid output."""
    if hidden_activation not in HIDDEN_ACTIVATIONS:
        raise ValueError(f"hidden_activation must be one of {HIDDEN_ACTIVATIONS}")
    hidden = np.tanh if hidden_activation == "tanh" else sigmoid
    depth = n_layers(parameters)
    activations = {"A0": X}
    A = X
    for layer in range(1, depth + 1):
        Z = parameters[f"W{layer}"] @ A + parameters[f"b{layer}"]
        A = sigmoid(Z) if layer == depth else hidden(Z)
        activations[f"A{layer}"] = A
    return activations


def backward_propagation(
    y: np.ndarray,
    parameters: Parameters,
    activations: dict[str, np.ndarray],
    hidden_activation: str = "tanh",
) -> dict[str, np.ndarray]:
    """Gradients of the mean log-loss with respect to every ``W`` and ``b``.

    With a sigmoid output and the log-loss, the output error simplifies to
    ``dZ_L = A_L - y``; each hidden layer then multiplies the back-propagated error by
    the activation derivative, written in terms of the stored activation
    (tanh' = 1 - A^2, sigmoid' = A (1 - A)) so no pre-activation needs to be cached.
    """
    m = y.shape[1]
    depth = n_layers(parameters)
    gradients: dict[str, np.ndarray] = {}
    dZ = activations[f"A{depth}"] - y
    for layer in range(depth, 0, -1):
        A_prev = activations[f"A{layer - 1}"]
        gradients[f"dW{layer}"] = dZ @ A_prev.T / m
        gradients[f"db{layer}"] = np.sum(dZ, axis=1, keepdims=True) / m
        if layer > 1:
            dA_prev = parameters[f"W{layer}"].T @ dZ
            if hidden_activation == "tanh":
                dZ = dA_prev * (1.0 - A_prev**2)
            else:
                dZ = dA_prev * A_prev * (1.0 - A_prev)
    return gradients


def update_parameters(
    parameters: Parameters, gradients: dict[str, np.ndarray], learning_rate: float
) -> Parameters:
    """One step of full-batch gradient descent (returns a new dictionary)."""
    return {
        name: value - learning_rate * gradients[f"d{name}"] for name, value in parameters.items()
    }


def predict_proba(
    X: np.ndarray, parameters: Parameters, hidden_activation: str = "tanh"
) -> np.ndarray:
    activations = forward_propagation(X, parameters, hidden_activation)
    return activations[f"A{n_layers(parameters)}"]


def predict(
    X: np.ndarray, parameters: Parameters, hidden_activation: str = "tanh", threshold: float = 0.5
) -> np.ndarray:
    return (predict_proba(X, parameters, hidden_activation) >= threshold).astype(float)


def accuracy(y: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(y == predictions))


@dataclass
class TrainingRun:
    """Fitted parameters plus the per-epoch learning curves."""

    parameters: Parameters
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)


def fit(
    X: np.ndarray,
    y: np.ndarray,
    dimensions: list[int],
    learning_rate: float = 0.1,
    epochs: int = 1000,
    seed: int = 0,
    hidden_activation: str = "tanh",
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    record_every: int = 1,
) -> TrainingRun:
    """Train with full-batch gradient descent, recording curves every ``record_every`` epochs.

    Metrics are computed from the forward pass that also feeds back-propagation, so
    the recorded training loss is the loss of the parameters *before* each update.
    """
    if dimensions[0] != X.shape[0]:
        raise ValueError(f"dimensions[0]={dimensions[0]} but X has {X.shape[0]} features")
    if dimensions[-1] != 1:
        raise ValueError("this network is a binary classifier: the last dimension must be 1")
    parameters = initialize_parameters(dimensions, seed=seed)
    run = TrainingRun(parameters=parameters)
    depth = len(dimensions) - 1
    for epoch in range(epochs):
        activations = forward_propagation(X, parameters, hidden_activation)
        if epoch % record_every == 0 or epoch == epochs - 1:
            output = activations[f"A{depth}"]
            run.train_loss.append(log_loss(y, output))
            run.train_accuracy.append(accuracy(y, (output >= 0.5).astype(float)))
            if X_val is not None and y_val is not None:
                val_output = predict_proba(X_val, parameters, hidden_activation)
                run.val_loss.append(log_loss(y_val, val_output))
                run.val_accuracy.append(accuracy(y_val, (val_output >= 0.5).astype(float)))
        gradients = backward_propagation(y, parameters, activations, hidden_activation)
        parameters = update_parameters(parameters, gradients, learning_rate)
    run.parameters = parameters
    return run
