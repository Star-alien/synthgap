"""Unit tests for the set metrics (no data or GPU needed)."""
from __future__ import annotations

import numpy as np

from synthgap.features.extract import l2_normalize
from synthgap.metrics.coverage import coverage_knn, density_and_coverage, improved_precision, improved_recall
from synthgap.metrics.diversity import vendi_per_class, vendi_score
from synthgap.metrics.fidelity import frechet_distance


def _orthogonal(n: int, d: int = 32) -> np.ndarray:
    return l2_normalize(np.eye(n, d, dtype=np.float32))


def test_vendi_of_duplicates_is_one():
    X = np.tile(np.eye(1, 16, dtype=np.float32), (12, 1))
    assert abs(vendi_score(X) - 1.0) < 1e-6


def test_vendi_of_orthogonal_is_n():
    X = _orthogonal(10)
    assert abs(vendi_score(X) - 10.0) < 1e-6


def test_vendi_per_class_averages():
    y = np.array([0] * 4 + [1] * 4)
    X = np.vstack([_orthogonal(4), _orthogonal(4)])
    value = vendi_per_class(X, y)
    assert 0.9 <= value <= 4.1


def test_coverage_and_precision_are_one_when_sets_match():
    real = _orthogonal(12)
    fake = real.copy()
    assert coverage_knn(real, fake, k=3) > 0.99
    assert improved_precision(real, fake, k=2) > 0.99
    assert improved_recall(real, fake, k=2) > 0.99
    density, coverage = density_and_coverage(real, fake, k=3)
    assert density > 0.9 and coverage > 0.99


def test_coverage_is_zero_for_a_disjoint_set():
    # two small arcs on opposite sides of the unit circle: unambiguous geometry
    angles = np.linspace(0.0, 0.3, 40)
    real = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    shifted = angles + np.pi
    fake = np.stack([np.cos(shifted), np.sin(shifted)], axis=1).astype(np.float32)
    assert coverage_knn(real, fake, k=3) == 0.0
    assert improved_precision(real, fake, k=2) == 0.0
    assert improved_recall(real, fake, k=2) == 0.0


def test_frechet_distance_vanishes_for_identical_gaussians():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 8))
    assert frechet_distance(X, X) < 1e-4
    Y = rng.normal(size=(300, 8)) + 2.0
    assert frechet_distance(X, Y) > 1.0
