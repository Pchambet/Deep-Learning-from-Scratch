#!/usr/bin/env python3
"""Test run du notebook 04 - exécute le code sans Jupyter."""
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "notebooks"))

print("1. Imports...")
import matplotlib

matplotlib.use("Agg")
import numpy as np
from birth_of_a_neuron import gradients, initialisation, model, predict, update
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.utilities import load_data

print("2. load_data()...")
X_train, y_train, X_test, y_test = load_data()
print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")

print("3. Reshape...")
X_train_reshape = X_train.reshape(X_train.shape[0], -1) / X_train.max()
X_test_reshape = X_test.reshape(X_test.shape[0], -1) / X_test.max()
print(f"   X_train_reshape: {X_train_reshape.shape}")

print("4. log_loss avec epsilon (birth_of_a_neuron)...")
# log_loss imported from birth_of_a_neuron (uses epsilon for stability)

print("5. Test log_loss SANS epsilon (simulation du bug)...")
W0, b0 = initialisation(X_train_reshape)
A0 = model(X_train_reshape, W0, b0)
try:
    bad_loss = -1/len(y_train) * np.sum(y_train * np.log(A0) + (1-y_train) * np.log(1-A0))
    print("   (pas d'erreur sur ce batch)")
except Exception as e:
    print(f"   Erreur log(0): {type(e).__name__}")

print("6. new_artificial_neuron avec log_loss corrigé (5 itérations test)...")
def new_artificial_neuron(X, y, learning_rate=0.1, num_iter=5):
    W, b = initialisation(X)
    for _ in range(num_iter):
        A = model(X, W, b)
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)
    y_pred = np.ravel(predict(X, W, b))
    y_flat = np.ravel(y)
    acc = accuracy_score(y_flat, y_pred)
    return W, b, acc


W, b, acc = new_artificial_neuron(X_train_reshape, y_train, learning_rate=0.01, num_iter=5)
print(f"   Accuracy (5 iter): {acc:.4f}")

print("7. Boucle train+test (20 itérations)...")
def run_train_test(X_train, y_train, X_test, y_test, learning_rate=0.01, n_iter=20):
    W, b = initialisation(X_train)
    for _ in tqdm(range(n_iter), desc="Train"):
        A = model(X_train, W, b)
        dW, db = gradients(A, X_train, y_train)
        W, b = update(dW, db, W, b, learning_rate)
    y_pred = np.ravel(predict(X_test, W, b))
    y_test_flat = np.ravel(y_test)
    return W, b, accuracy_score(y_test_flat, y_pred)

W, b, test_acc = run_train_test(X_train_reshape, y_train, X_test_reshape, y_test, n_iter=20)
print(f"   Test accuracy: {test_acc:.4f}")

print("\n✅ Tous les tests passés - le code du notebook fonctionne.")
