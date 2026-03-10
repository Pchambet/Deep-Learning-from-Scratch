#!/usr/bin/env python3
"""Minimal smoke test for the custom training loop stack."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "notebooks"))

import numpy as np
from birth_of_a_neuron import gradients, initialisation, model, predict, update
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.utilities import load_data  # noqa: E402


def run_smoke(iterations=20, learning_rate=0.01):
    x_train, y_train, x_test, y_test = load_data()
    x_train = x_train.reshape(x_train.shape[0], -1) / x_train.max()
    x_test = x_test.reshape(x_test.shape[0], -1) / x_test.max()

    # Keep y_train as (n, 1) for gradients (A - y must match shapes, no broadcast)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    weights, bias = initialisation(x_train)
    for _ in tqdm(range(iterations), desc="SmokeTrain"):
        activations = model(x_train, weights, bias)
        d_weights, d_bias = gradients(activations, x_train, y_train)
        weights, bias = update(d_weights, d_bias, weights, bias, learning_rate)

    y_pred = predict(x_test, weights, bias)
    y_pred = np.asarray(y_pred).flatten()
    y_test_flat = np.asarray(y_test).flatten()
    test_accuracy = accuracy_score(y_test_flat, y_pred)
    print({"test_accuracy": float(test_accuracy)})


if __name__ == "__main__":
    run_smoke()
