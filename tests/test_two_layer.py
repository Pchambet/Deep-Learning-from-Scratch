"""Tests for src/two_layer_network.py — forward, backward, fit."""

import numpy as np
from sklearn.datasets import make_circles

from src.two_layer_network import (
    backward_propagation,
    binary_cross_entropy,
    fit_two_layer_network,
    forward_propagation,
    initialize_parameters,
    predict,
    sigmoid,
    tanh,
)


class TestTwoLayerNetwork:
    """Two-layer network: shapes, forward, backward, fit."""

    def test_initialize_parameters_shapes(self):
        params = initialize_parameters(n0=4, n1=8, n2=2, seed=0)
        assert params["W1"].shape == (8, 4)
        assert params["b1"].shape == (8, 1)
        assert params["W2"].shape == (2, 8)
        assert params["b2"].shape == (2, 1)

    def test_forward_propagation_shapes(self):
        """Inputs (n0, m), outputs A2 (n2, m)."""
        params = initialize_parameters(n0=4, n1=6, n2=2, seed=0)
        inputs = np.random.randn(4, 10)
        activations = forward_propagation(inputs, params)
        assert activations["Z1"].shape == (6, 10)
        assert activations["A1"].shape == (6, 10)
        assert activations["Z2"].shape == (2, 10)
        assert activations["A2"].shape == (2, 10)

    def test_backward_propagation_shapes(self):
        params = initialize_parameters(n0=4, n1=6, n2=2, seed=0)
        inputs = np.random.randn(4, 10)
        labels = np.random.rand(2, 10)
        activations = forward_propagation(inputs, params)
        grads = backward_propagation(inputs, labels, params, activations)
        assert grads["dW1"].shape == (6, 4)
        assert grads["db1"].shape == (6, 1)
        assert grads["dW2"].shape == (2, 6)
        assert grads["db2"].shape == (2, 1)

    def test_predict_shapes(self):
        params = initialize_parameters(n0=4, n1=6, n2=2, seed=0)
        inputs = np.random.randn(4, 10)
        pred = predict(inputs, params)
        assert pred.shape == (2, 10)
        assert set(np.unique(pred)).issubset({0.0, 1.0})

    def test_sigmoid_tanh_bounds(self):
        z = np.array([[-10, 0, 10]])
        s = sigmoid(z)
        t = tanh(z)
        assert np.all(s >= 0) and np.all(s <= 1)
        assert np.all(t >= -1) and np.all(t <= 1)

    def test_binary_cross_entropy_non_negative(self):
        labels = np.array([[1, 0, 1]])
        probs = np.array([[0.9, 0.1, 0.8]])
        loss = binary_cross_entropy(labels, probs)
        assert loss >= 0
        assert not np.isnan(loss)

    def test_fit_loss_decreases(self):
        """Loss should decrease over epochs on separable toy data."""
        X, y = make_circles(n_samples=100, noise=0.05, random_state=42)
        # API: inputs (n_features, n_samples), labels (n_classes, n_samples)
        inputs = X.T.astype(np.float64)
        labels = np.vstack([y, 1 - y]).astype(np.float64)  # one-hot style (2, n)
        result = fit_two_layer_network(
            inputs, labels, n1=8, learning_rate=0.5, epochs=200, seed=42
        )
        loss = result["loss"]
        # Loss should generally decrease (allow some fluctuation)
        initial_avg = np.mean(loss[:20])
        final_avg = np.mean(loss[-20:])
        assert final_avg < initial_avg, f"Loss did not decrease: {initial_avg} -> {final_avg}"

    def test_fit_accuracy_improves(self):
        """Accuracy should improve on toy data."""
        X, y = make_circles(n_samples=100, noise=0.05, random_state=42)
        inputs = X.T.astype(np.float64)
        labels = np.vstack([y, 1 - y]).astype(np.float64)
        result = fit_two_layer_network(
            inputs, labels, n1=8, learning_rate=0.5, epochs=300, seed=42
        )
        acc = result["accuracy"]
        assert acc[-1] >= 0.5  # Better than random
        assert acc[-1] >= acc[0]  # Improved over training
