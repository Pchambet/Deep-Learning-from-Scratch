"""
Neuron functions for Ep III (Birth of a Neuron) — used by 04_training_loop_from_scratch.
Import directly: no nbimporter needed. Works on Colab, local, everywhere.
"""

import numpy as np


def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return W, b


def model(X, W, b):
    Z = np.dot(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A


def log_loss(A, y):
    """Log-loss with epsilon for numerical stability (avoids log(0))."""
    m = len(y)
    epsilon = 1e-15
    loss = -1 / m * np.sum(y * np.log(A + epsilon) + (1 - y) * np.log(1 - A + epsilon))
    return loss


def gradients(A, X, y):
    m = len(y)
    dZ = A - y
    dW = 1 / m * np.dot(X.T, dZ)
    db = 1 / m * np.sum(dZ)
    return dW, db


def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b


def predict(X, W, b):
    A = model(X, W, b)
    return (A >= 0.5).astype(float)


def artificial_neuron(X, y, learning_rate=0.1, num_iter=100):
    """Train a single neuron. Returns (W, b). Plots loss."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score

    W, b = initialisation(X)
    loss = []

    for i in range(num_iter):
        A = model(X, W, b)
        loss.append(log_loss(A, y))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    y_pred = predict(X, W, b)
    print("accuracy score :", accuracy_score(y, y_pred))

    plt.plot(loss)
    plt.xlabel("Iterations")
    plt.ylabel("Log-loss")
    plt.title("Training loss")
    plt.grid(True)
    plt.show()

    return W, b
