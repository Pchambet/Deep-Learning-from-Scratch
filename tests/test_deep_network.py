"""Tests for src/deep_network.py: shapes, exact gradients, agreement with the
earlier episodes, and recovery of a known non-linear boundary."""

import numpy as np
import pytest
from birth_of_a_neuron import gradients as neuron_gradients
from birth_of_a_neuron import model as neuron_model
from sklearn.datasets import make_circles

from src import two_layer_network
from src.deep_network import (
    accuracy,
    backward_propagation,
    fit,
    forward_propagation,
    initialize_parameters,
    log_loss,
    predict,
    predict_proba,
    sigmoid,
    update_parameters,
)
from src.gradient_check import numerical_gradient, relative_error


@pytest.fixture
def toy_batch():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 12))
    y = (rng.random((1, 12)) > 0.5).astype(float)
    return X, y


def test_parameter_shapes_follow_dimensions():
    parameters = initialize_parameters([4, 5, 3, 1], seed=0)
    assert {k: v.shape for k, v in parameters.items()} == {
        "W1": (5, 4),
        "b1": (5, 1),
        "W2": (3, 5),
        "b2": (3, 1),
        "W3": (1, 3),
        "b3": (1, 1),
    }


def test_initialisation_is_seeded_and_scaled():
    a = initialize_parameters([400, 50, 1], seed=1)
    b = initialize_parameters([400, 50, 1], seed=1)
    np.testing.assert_array_equal(a["W1"], b["W1"])
    # 1/sqrt(fan_in) scaling: the weight standard deviation is ~ 1/sqrt(400) = 0.05.
    assert a["W1"].std() == pytest.approx(0.05, rel=0.05)


def test_forward_returns_every_layer(toy_batch):
    X, _ = toy_batch
    activations = forward_propagation(X, initialize_parameters([3, 4, 4, 1]))
    assert [activations[f"A{i}"].shape for i in range(4)] == [(3, 12), (4, 12), (4, 12), (1, 12)]
    output = activations["A3"]
    assert np.all((output > 0) & (output < 1))


def test_sigmoid_is_stable_for_extreme_inputs():
    with np.errstate(over="raise"):
        values = sigmoid(np.array([-1e4, 0.0, 1e4]))
    np.testing.assert_allclose(values, [0.0, 0.5, 1.0], atol=1e-200)


def test_log_loss_hand_checked():
    # -(log 0.8 + log 0.6) / 2 for labels [1, 0] and probabilities [0.8, 0.4].
    expected = -(np.log(0.8) + np.log(0.6)) / 2
    assert log_loss(np.array([[1.0, 0.0]]), np.array([[0.8, 0.4]])) == pytest.approx(expected)


@pytest.mark.parametrize("hidden_activation", ["tanh", "sigmoid"])
@pytest.mark.parametrize("dimensions", [[3, 1], [3, 4, 1], [3, 5, 4, 3, 1]])
def test_backprop_matches_finite_differences(toy_batch, dimensions, hidden_activation):
    X, y = toy_batch
    parameters = initialize_parameters(dimensions, seed=3)
    activations = forward_propagation(X, parameters, hidden_activation)
    analytic = backward_propagation(y, parameters, activations, hidden_activation)

    def loss() -> float:
        return log_loss(y, predict_proba(X, parameters, hidden_activation))

    for name, value in parameters.items():
        numeric = numerical_gradient(loss, value)
        assert relative_error(analytic[f"d{name}"], numeric) < 1e-7, name


def test_one_layer_network_is_the_episode_iii_neuron(toy_batch):
    """With dimensions [n0, 1] the network is logistic regression: same output and
    gradients as birth_of_a_neuron.py, which stores samples as rows instead of columns."""
    X, y = toy_batch
    parameters = initialize_parameters([3, 1], seed=5)
    parameters["b1"] = np.array([[0.3]])
    activations = forward_propagation(X, parameters)
    grads = backward_propagation(y, parameters, activations)

    W, b = parameters["W1"].T, parameters["b1"].ravel()
    A = neuron_model(X.T, W, b)
    dW, db = neuron_gradients(A, X.T, y.T)
    np.testing.assert_allclose(activations["A1"], A.T)
    np.testing.assert_allclose(grads["dW1"], dW.T)
    np.testing.assert_allclose(grads["db1"].item(), db)


def test_two_layer_case_matches_episode_vi_module(toy_batch):
    X, y = toy_batch
    parameters = initialize_parameters([3, 6, 1], seed=2)
    activations = forward_propagation(X, parameters, "tanh")
    grads = backward_propagation(y, parameters, activations, "tanh")

    reference_activations = two_layer_network.forward_propagation(X, parameters)
    reference = two_layer_network.backward_propagation(X, y, parameters, reference_activations)
    np.testing.assert_allclose(activations["A2"], reference_activations["A2"])
    for name in ("dW1", "db1", "dW2", "db2"):
        np.testing.assert_allclose(grads[name], reference[name])


def test_update_is_one_gradient_step():
    parameters = {"W1": np.array([[1.0, 2.0]]), "b1": np.array([[0.5]])}
    gradients = {"dW1": np.array([[0.5, -1.0]]), "db1": np.array([[1.0]])}
    updated = update_parameters(parameters, gradients, learning_rate=0.1)
    np.testing.assert_allclose(updated["W1"], [[0.95, 2.1]])
    np.testing.assert_allclose(updated["b1"], [[0.4]])
    np.testing.assert_allclose(parameters["W1"], [[1.0, 2.0]])  # input left untouched


def test_recovers_circular_boundary_on_held_out_points():
    """Ground truth: concentric circles are separable by a circle, which no linear model
    can represent. A two-hidden-layer network must find it; a single neuron cannot."""
    X, y = make_circles(n_samples=600, noise=0.05, factor=0.4, random_state=0)
    X_train, y_train = X[:400].T, y[:400].reshape(1, -1)
    X_test, y_test = X[400:].T, y[400:].reshape(1, -1)

    deep = fit(X_train, y_train, [2, 16, 16, 1], learning_rate=0.5, epochs=2000, seed=0)
    linear = fit(X_train, y_train, [2, 1], learning_rate=0.5, epochs=2000, seed=0)

    assert accuracy(y_test, predict(X_test, deep.parameters)) >= 0.97
    assert accuracy(y_test, predict(X_test, linear.parameters)) <= 0.70
    assert deep.train_loss[-1] < 0.25 * deep.train_loss[0]


def test_fit_records_validation_curves():
    X, y = make_circles(n_samples=100, noise=0.05, random_state=1)
    run = fit(
        X.T,
        y.reshape(1, -1),
        [2, 4, 1],
        epochs=50,
        X_val=X.T,
        y_val=y.reshape(1, -1),
        record_every=10,
    )
    # Epochs 0, 10, 20, 30, 40 and the last one (49).
    assert len(run.train_loss) == len(run.val_accuracy) == 6


def test_fit_rejects_inconsistent_dimensions():
    with pytest.raises(ValueError, match="features"):
        fit(np.zeros((3, 5)), np.zeros((1, 5)), [2, 1])
    with pytest.raises(ValueError, match="binary"):
        fit(np.zeros((2, 5)), np.zeros((1, 5)), [2, 2])
