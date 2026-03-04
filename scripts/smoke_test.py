#!/usr/bin/env python3
"""Minimal smoke test for the custom training loop stack."""

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.utilities import load_data  # noqa: E402


def initialisation(x):
    return np.random.randn(x.shape[1], 1), np.random.randn(1)


def model(x, weights, bias):
    return 1 / (1 + np.exp(-(np.dot(x, weights) + bias)))


def gradients(activations, x, labels):
    sample_count = len(labels)
    delta = activations - labels
    return np.dot(x.T, delta) / sample_count, np.sum(delta) / sample_count


def update(d_weights, d_bias, weights, bias, learning_rate):
    return weights - learning_rate * d_weights, bias - learning_rate * d_bias


def predict(x, weights, bias):
    return (model(x, weights, bias) >= 0.5).astype(float)


def run_smoke(iterations=20, learning_rate=0.01):
    x_train, y_train, x_test, y_test = load_data()
    x_train = x_train.reshape(x_train.shape[0], -1) / x_train.max()
    x_test = x_test.reshape(x_test.shape[0], -1) / x_test.max()

    weights, bias = initialisation(x_train)
    for _ in tqdm(range(iterations), desc="SmokeTrain"):
        activations = model(x_train, weights, bias)
        d_weights, d_bias = gradients(activations, x_train, y_train)
        weights, bias = update(d_weights, d_bias, weights, bias, learning_rate)

    y_pred = predict(x_test, weights, bias)
    test_accuracy = accuracy_score(y_test, y_pred)
    print({"test_accuracy": float(test_accuracy)})


if __name__ == "__main__":
    run_smoke()
