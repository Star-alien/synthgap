"""Unit tests for selection rules, the factorial and the variance decomposition."""
from __future__ import annotations

import numpy as np
import pandas as pd

from synthgap.data.datasets import _carve
from synthgap.design.factorial import main_effects, summarize
from synthgap.select.strategies import build_selection, kcenter_subset
from synthgap.stats.regression import shapley_r2
from synthgap.utils import fraction_of_gap_closed


def _toy_pool(n_classes=3, n_per_class=12, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_classes), n_per_class)
    centres = rng.normal(size=(n_classes, dim)) * 3.0
    X = centres[y] + rng.normal(size=(len(y), dim))
    return X.astype(np.float32), y


def test_kcenter_prefers_spread_points():
    X = np.array([[0.0, 0.0], [0.01, 0.0], [5.0, 0.0], [5.01, 0.0]], dtype=np.float32)
    candidates = np.arange(4)
    picked = kcenter_subset(X, candidates, 2)
    assert len(picked) == 2
    spread = np.linalg.norm(X[picked[0]] - X[picked[1]])
    assert spread > 4.0


def test_every_selection_returns_n_per_class():
    X, y = _toy_pool()
    scores = np.linspace(0, 1, len(y))
    for kind in ("random", "kcenter", "clustered", "filter", "das", "align_low", "align_high"):
        indices = build_selection(kind, X, y, 5, scores=scores, seed=0)
        counts = np.bincount(y[indices], minlength=3)
        assert counts.min() == 5 and counts.max() == 5, kind


def test_das_keeps_high_alignment_images():
    X, y = _toy_pool(n_classes=2, n_per_class=20)
    scores = np.linspace(0, 1, len(y))
    indices = build_selection("das", X, y, 4, scores=scores, quantile=0.6, seed=0)
    for c in np.unique(y):
        class_indices = np.flatnonzero(y == c)
        threshold = np.quantile(scores[class_indices], 0.6)
        chosen = indices[y[indices] == c]
        assert len(chosen) == 4
        assert scores[chosen].min() >= threshold - 1e-9


def test_factorial_effects_recover_a_known_pattern():
    means = {"low/low": 0.50, "low/high": 0.55, "high/low": 0.60, "high/high": 0.70}
    effects = main_effects(means)
    assert abs(effects["diversity"] - 0.125) < 1e-9
    assert abs(effects["alignment"] - 0.075) < 1e-9
    assert abs(effects["interaction"] - 0.05) < 1e-9


def test_factorial_summary_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    cells = {key: list(0.5 + rng.normal(0, 0.01, 6)) for key in ("low/low", "low/high", "high/low", "high/high")}
    summary = summarize(cells, n_boot=200, seed=0)
    for stats in summary["cells"].values():
        assert stats["lo"] <= stats["mean"] <= stats["hi"]


def test_shapley_shares_sum_to_r2():
    rng = np.random.default_rng(2)
    n = 300
    D = rng.normal(size=n)
    C = 0.6 * D + rng.normal(size=n) * 0.8
    A = rng.normal(size=n)
    F = rng.normal(size=n)
    accuracy = 2.0 * D + 1.0 * C - 0.5 * A + rng.normal(size=n) * 0.5
    frame = pd.DataFrame({"D": D, "C": C, "A": A, "F": F, "accuracy": accuracy, "cell": "x"})
    result = shapley_r2(frame, ["D", "C", "A", "F"], "accuracy", fe=None)
    assert abs(sum(result["shares"].values()) - result["r2"]) < 1e-6
    assert max(result["shares"], key=result["shares"].get) == "D"


def test_fraction_of_gap_closed_handles_zero_denominator():
    assert abs(fraction_of_gap_closed(0.6, 0.5, 0.7) - 0.5) < 1e-9
    assert np.isnan(fraction_of_gap_closed(0.6, 0.5, 0.5))


def test_alignment_carve_is_disjoint_and_deterministic():
    per_class = [[f"c{c}_{i}" for i in range(10)] for c in range(4)]
    picked, rest = _carve(per_class, 3, np.random.default_rng(0))
    picked_again, _ = _carve(per_class, 3, np.random.default_rng(0))
    assert picked == picked_again
    for c in range(4):
        assert len(picked[c]) == 3
        assert not set(picked[c]) & set(rest[c])
        assert len(rest[c]) == 7
