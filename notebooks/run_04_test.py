#!/usr/bin/env python3
"""Test run du notebook 04 - exécute le code sans Jupyter."""
import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)

print("1. Imports...")
import matplotlib
matplotlib.use('Agg')
import numpy as np
from src.utilities import load_data
from sklearn.metrics import accuracy_score

# Fonctions du neurone (même logique que birth_of_a_neuron)
def initialisation(X):
    return np.random.randn(X.shape[1], 1), np.random.randn(1)
def model(X, W, b):
    return 1 / (1 + np.exp(-(np.dot(X, W) + b)))
def gradients(A, X, y):
    m = len(y)
    dZ = A - y
    return np.dot(X.T, dZ) / m, np.sum(dZ) / m
def update(dW, db, W, b, lr):
    return W - lr * dW, b - lr * db
def predict(X, W, b):
    return (model(X, W, b) >= 0.5).astype(float)

print("2. load_data()...")
X_train, y_train, X_test, y_test = load_data()
print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")

print("3. Reshape...")
X_train_reshape = X_train.reshape(X_train.shape[0], -1) / X_train.max()
X_test_reshape = X_test.reshape(X_test.shape[0], -1) / X_test.max()
print(f"   X_train_reshape: {X_train_reshape.shape}")

print("4. log_loss avec epsilon...")
def log_loss(A, y):
    m = len(y)
    epsilon = 1e-15
    loss = -1/m * np.sum(y * np.log(A + epsilon) + (1 - y) * np.log(1 - A + epsilon))
    return loss

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
    loss = []
    for i in range(num_iter):
        A = model(X, W, b)
        loss.append(log_loss(A, y))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)
    y_pred = predict(X, W, b)
    acc = accuracy_score(y, y_pred)
    return W, b, acc

W, b, acc = new_artificial_neuron(X_train_reshape, y_train, learning_rate=0.01, num_iter=5)
print(f"   Accuracy (5 iter): {acc:.4f}")

print("7. Boucle train+test (20 itérations)...")
from tqdm import tqdm
def run_train_test(X_train, y_train, X_test, y_test, learning_rate=0.01, n_iter=20):
    W, b = initialisation(X_train)
    for i in tqdm(range(n_iter), desc="Train"):
        A = model(X_train, W, b)
        dW, db = gradients(A, X_train, y_train)
        W, b = update(dW, db, W, b, learning_rate)
    y_pred = predict(X_test, W, b)
    return W, b, accuracy_score(y_test, y_pred)

W, b, test_acc = run_train_test(X_train_reshape, y_train, X_test_reshape, y_test, n_iter=20)
print(f"   Test accuracy: {test_acc:.4f}")

print("\n✅ Tous les tests passés - le code du notebook fonctionne.")
