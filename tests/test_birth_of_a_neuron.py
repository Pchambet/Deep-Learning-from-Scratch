"""Tests for notebooks/birth_of_a_neuron.py — log_loss, gradients, predict."""

import numpy as np
from birth_of_a_neuron import (
    gradients,
    initialisation,
    log_loss,
    model,
    predict,
    update,
)


class TestBirthOfANeuron:
    """Single neuron functions: numerical stability, shapes, gradients."""

    def test_initialisation_shapes(self):
        X = np.random.randn(50, 4)
        W, b = initialisation(X)
        assert W.shape == (4, 1)
        assert b.shape == (1,) or b.shape == (1, 1)

    def test_model_shapes(self):
        X = np.random.randn(50, 4)
        W, b = initialisation(X)
        A = model(X, W, b)
        assert A.shape == (50, 1) or A.shape == (50,)

    def test_log_loss_with_epsilon_avoids_nan(self):
        """log_loss uses epsilon — no NaN/Inf on extreme probs."""
        y = np.array([1, 0, 1, 0])
        A_bad = np.array([0.0, 1.0, 1e-20, 1 - 1e-20])
        loss = log_loss(A_bad.reshape(-1, 1), y)
        assert not np.isnan(loss)
        assert not np.isinf(loss)
        assert loss >= 0

    def test_log_loss_perfect_prediction(self):
        y = np.array([[1], [0], [1], [0]])
        A = np.array([[1.0], [0.0], [1.0], [0.0]])
        loss = log_loss(A, y)
        assert loss < 0.1  # Near zero for perfect prediction

    def test_gradients_shapes(self):
        n = 20
        X = np.random.randn(n, 4)
        W, b = initialisation(X)
        A = model(X, W, b)
        y = np.random.rand(n).reshape(-1, 1)
        dW, db = gradients(A, X, y)
        assert dW.shape == (4, 1)
        assert np.isscalar(db) or db.shape in ((1,), (1, 1))

    def test_update_decreases_weights(self):
        """Update with positive gradient should decrease W."""
        X = np.random.randn(10, 3)
        W, b = initialisation(X)
        A = model(X, W, b)
        y = np.zeros((10, 1))
        dW, db = gradients(A, X, y)
        W_new, b_new = update(dW, db, W, b, learning_rate=0.1)
        # Gradient direction: weights are updated
        assert not np.allclose(W, W_new)
        assert not np.allclose(b, b_new)

    def test_predict_binary(self):
        X = np.random.randn(10, 3)
        W, b = initialisation(X)
        pred = predict(X, W, b)
        assert set(np.unique(pred.flatten())).issubset({0.0, 1.0})

    def test_gradients_numerical_check(self):
        """Compare analytical gradients to numerical approximation (finite diff)."""
        np.random.seed(42)
        n, d = 20, 4
        X = np.random.randn(n, d) * 0.5
        W = np.random.randn(d, 1) * 0.1
        b = np.random.randn(1) * 0.1
        y = (np.random.rand(n) > 0.5).astype(float).reshape(-1, 1)
        eps = 1e-5

        A = model(X, W, b)
        dW_ana, db_ana = gradients(A, X, y)

        # Numerical dW
        dW_num = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_plus = W.copy()
                W_plus[i, j] += eps
                loss_plus = log_loss(model(X, W_plus, b), y)
                W_minus = W.copy()
                W_minus[i, j] -= eps
                loss_minus = log_loss(model(X, W_minus, b), y)
                dW_num[i, j] = (loss_plus - loss_minus) / (2 * eps)

        # Numerical db
        b_plus = b + eps
        b_minus = b - eps
        db_num = (log_loss(model(X, W, b_plus), y) - log_loss(model(X, W, b_minus), y)) / (
            2 * eps
        )

        np.testing.assert_allclose(dW_ana, dW_num, rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(db_ana, db_num, rtol=1e-3, atol=1e-2)
