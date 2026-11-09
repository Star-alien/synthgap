"""Reference-free diversity: the Vendi score (Friedman & Dieng, 2023).

Vendi(S) = exp(H(lambda)) where lambda are the eigenvalues of K/n and K is the
cosine-similarity kernel of the n embeddings. It is the effective number of
distinct samples and never looks at real data.
"""
from __future__ import annotations

import numpy as np

from ..features.extract import l2_normalize


def vendi_score(X: np.ndarray, eps: float = 1e-12) -> float:
    X = l2_normalize(np.asarray(X, dtype=np.float32))
    n = X.shape[0]
    if n < 2:
        return 1.0
    K = (X @ X.T).astype(np.float64) / n
    K = (K + K.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = eigenvalues.sum()
    if total <= eps:
        return 1.0
    p = eigenvalues[eigenvalues > eps] / total
    entropy = -(p * np.log(p)).sum()
    return float(np.exp(entropy))


def vendi_per_class(X: np.ndarray, y: np.ndarray) -> float:
    """Mean Vendi over classes (the paper averages all metrics per class)."""
    scores = []
    for c in np.unique(y):
        mask = np.asarray(y) == c
        if mask.sum() >= 2:
            scores.append(vendi_score(X[mask]))
    return float(np.mean(scores)) if scores else float("nan")
