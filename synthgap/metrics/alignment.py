"""Label-alignment scorers A_real and A_text (Sec. 3.2).

A_real: softmax probability of the prompted class under a nearest-class-mean
        classifier built from the k-shot alignment split A, in CLIP space.
A_text: CLIP zero-shot probability over the K class names. Uses no real images.

Both are computed in CLIP space, never in the DINOv2 space the learner trains on.
On FGVC-Aircraft the two scorers are expected to disagree, because CLIP zero-shot
is weak on aircraft variants; that disagreement is reported rather than hidden.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def class_means(X: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    means = np.zeros((n_classes, X.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        mask = np.asarray(y) == c
        if mask.any():
            means[c] = X[mask].mean(axis=0)
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    return means / np.maximum(norms, 1e-8)


def ncm_probabilities(X: np.ndarray, means: np.ndarray, temperature: float) -> np.ndarray:
    logits = (np.asarray(X, dtype=np.float32) @ means.T) / max(float(temperature), 1e-6)
    return _softmax(logits)


def calibrate_temperature(
    X: np.ndarray, y: np.ndarray, means: np.ndarray, grid: Sequence[float]
) -> float:
    """Pick the softmax temperature that maximises NCM accuracy on the alignment split.

    The split is tiny (k=5/class), so we prefer a stable choice: the best accuracy,
    ties broken toward the larger temperature (softer, less overconfident scores).
    """
    best_t, best_acc = float(grid[-1]), -1.0
    for temperature in grid:
        probs = ncm_probabilities(X, means, temperature)
        accuracy = float(np.mean(probs.argmax(axis=1) == np.asarray(y)))
        if accuracy > best_acc + 1e-9 or (abs(accuracy - best_acc) < 1e-9 and temperature > best_t):
            best_t, best_acc = float(temperature), accuracy
    return best_t


def a_real(
    X: np.ndarray,
    y: np.ndarray,
    X_align: np.ndarray,
    y_align: np.ndarray,
    n_classes: int,
    temperature_grid: Sequence[float] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
) -> tuple[np.ndarray, float]:
    """Per-image probability of its prompted class under the few-shot NCM."""
    means = class_means(X_align, y_align, n_classes)
    temperature = calibrate_temperature(X_align, y_align, means, temperature_grid)
    probs = ncm_probabilities(X, means, temperature)
    y = np.asarray(y)
    return probs[np.arange(len(y)), y], temperature


def a_text(
    X: np.ndarray, y: np.ndarray, text_embeddings: np.ndarray, logit_scale: float = 100.0
) -> np.ndarray:
    """Per-image CLIP zero-shot probability of its prompted class."""
    probs = _softmax(float(logit_scale) * (np.asarray(X, dtype=np.float32) @ text_embeddings.T))
    y = np.asarray(y)
    return probs[np.arange(len(y)), y]


def per_class_mean(scores: np.ndarray, y: np.ndarray, classes: Sequence[int]) -> dict[int, float]:
    out = {}
    for c in classes:
        mask = np.asarray(y) == c
        if mask.any():
            out[int(c)] = float(np.mean(np.asarray(scores)[mask]))
    return out
