"""Reference-based set metrics: coverage/recall and precision/density.

* ``coverage_knn``   the paper's coverage: fraction of real images whose k-NN ball
  (k=5, radius = distance to the k-th real neighbour) contains >=1 synthetic image.
* ``improved_recall`` / ``improved_precision``  Kynkaanniemi et al. (2019).
* ``density_and_coverage``  Naeem et al. (2020), which separates fidelity from
  diversity instead of mixing them the way FID does.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

_CHUNK = 2048


def _as(X) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(X, dtype=np.float32))


def _kth_radius(X: np.ndarray, k: int) -> np.ndarray:
    """Distance from each sample to its k-th neighbour *within* X (self excluded)."""
    X = _as(X)
    n = X.shape[0]
    if n < 2:
        return np.full(n, np.inf, dtype=np.float32)
    kk = max(1, min(int(k), n - 1))
    neighbours = NearestNeighbors(n_neighbors=kk + 1).fit(X)
    distances, _ = neighbours.kneighbors(X)
    return distances[:, -1].astype(np.float32)


def coverage_knn(real: np.ndarray, fake: np.ndarray, k: int = 5) -> float:
    real, fake = _as(real), _as(fake)
    if real.shape[0] < 2 or fake.shape[0] == 0:
        return float("nan")
    radius = _kth_radius(real, k)
    nearest = NearestNeighbors(n_neighbors=1).fit(fake)
    distances, _ = nearest.kneighbors(real)
    return float(np.mean(distances[:, 0] <= radius))


def improved_precision(real: np.ndarray, fake: np.ndarray, k: int = 3) -> float:
    """Fraction of synthetic samples falling inside the real manifold."""
    real, fake = _as(real), _as(fake)
    if real.shape[0] < 2 or fake.shape[0] == 0:
        return float("nan")
    radius = _kth_radius(real, k)
    nearest = NearestNeighbors(n_neighbors=1).fit(real)
    inside = 0
    for start in range(0, fake.shape[0], _CHUNK):
        block = fake[start : start + _CHUNK]
        distances, indices = nearest.kneighbors(block)
        inside += int(np.sum(distances[:, 0] <= radius[indices[:, 0]]))
    return float(inside / fake.shape[0])


def improved_recall(real: np.ndarray, fake: np.ndarray, k: int = 3) -> float:
    """Fraction of real samples falling inside the synthetic manifold."""
    real, fake = _as(real), _as(fake)
    if real.shape[0] == 0 or fake.shape[0] < 2:
        return float("nan")
    radius = _kth_radius(fake, k)
    nearest = NearestNeighbors(n_neighbors=1).fit(fake)
    distances, indices = nearest.kneighbors(real)
    return float(np.mean(distances[:, 0] <= radius[indices[:, 0]]))


def density_and_coverage(real: np.ndarray, fake: np.ndarray, k: int = 5) -> tuple[float, float]:
    """Naeem et al. density and coverage (k-NN balls around the real set)."""
    real, fake = _as(real), _as(fake)
    if real.shape[0] < 2 or fake.shape[0] == 0:
        return float("nan"), float("nan")
    radius = _kth_radius(real, k)
    covered = np.zeros(real.shape[0], dtype=bool)
    density_sum = 0.0
    for start in range(0, fake.shape[0], _CHUNK):
        block = fake[start : start + _CHUNK]
        distances = np.linalg.norm(block[:, None, :] - real[None, :, :], axis=2)
        membership = distances <= radius[None, :]
        density_sum += float(membership.sum() / max(1, min(k, real.shape[0])))
        covered |= membership.any(axis=0)
    density = density_sum / fake.shape[0]
    coverage = float(covered.mean())
    return float(density), coverage
