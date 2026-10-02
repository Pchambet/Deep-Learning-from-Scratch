"""Finite-difference gradient checking.

Back-propagation bugs rarely crash: a wrong sign or a missing transpose still trains,
just badly. Comparing the analytic gradients with central differences
(f(x + eps) - f(x - eps)) / 2 eps, whose error is O(eps^2), is the standard way to
prove the derivation and the code agree.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def numerical_gradient(f: Callable[[], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of ``f`` with respect to ``x``.

    ``f`` takes no argument and reads ``x``, which is perturbed in place one entry at a
    time and restored afterwards.
    """
    grad = np.zeros_like(x, dtype=float)
    for index in np.ndindex(x.shape):
        original = x[index]
        x[index] = original + eps
        loss_plus = f()
        x[index] = original - eps
        loss_minus = f()
        x[index] = original
        grad[index] = (loss_plus - loss_minus) / (2.0 * eps)
    return grad


def relative_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    """``||a - n|| / (||a|| + ||n||)``: scale-free, 0 when equal, ~1e-7 or below when correct."""
    denominator = np.linalg.norm(analytic) + np.linalg.norm(numeric)
    if denominator == 0.0:
        return 0.0
    return float(np.linalg.norm(analytic - numeric) / denominator)
