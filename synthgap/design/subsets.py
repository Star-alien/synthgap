"""Balanced random-subset design for the variance decomposition (Sec. 4).

Each subset is a mixture of draws -- random, k-center, clustered and
quantile-restricted -- in random proportions. That produces broad, only weakly
correlated variation in diversity D, coverage C, alignment A and fidelity F,
which is what Eq. (1) needs.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..select.strategies import class_groups, clustered_subset, kcenter_subset

STRATEGIES = ("random", "kcenter", "clustered", "band_low", "band_mid", "band_high")


def _candidates_for(strategy: str, idx: np.ndarray, scores: np.ndarray, bands: dict) -> np.ndarray:
    if strategy == "random" or scores is None:
        return idx
    if strategy == "band_low":
        band = bands["low"]
    elif strategy == "band_mid":
        band = bands["mid"]
    elif strategy == "band_high":
        band = bands["high"]
    else:
        return idx
    values = scores[idx]
    low, high = np.quantile(values, band[0]), np.quantile(values, band[1])
    keep = idx[(values >= low) & (values <= high)]
    return keep if len(keep) else idx


def mixed_subset(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    rng: np.random.Generator,
    scores: np.ndarray | None = None,
    bands: dict | None = None,
) -> np.ndarray:
    """One subset with n images per class, mixed over strategies."""
    bands = bands or {"low": [0.0, 1 / 3], "mid": [1 / 3, 2 / 3], "high": [2 / 3, 1.0]}
    chosen_all = []
    for _, idx in class_groups(y).items():
        weights = rng.dirichlet(np.ones(len(STRATEGIES)))
        counts = np.floor(weights * n).astype(int)
        remainder = n - counts.sum()
        if remainder > 0:
            order = np.argsort(-(weights * n - counts))
            counts[order[:remainder]] += 1

        picked: list[int] = []
        for strategy, count in zip(STRATEGIES, counts):
            if count <= 0:
                continue
            pool = _candidates_for(strategy, idx, scores, bands)
            pool = np.array([i for i in pool if i not in picked])
            if len(pool) == 0:
                continue
            if strategy == "kcenter" and len(pool) > count:
                take = kcenter_subset(X, pool, min(count, len(pool)))
            elif strategy == "clustered" and len(pool) > count:
                take = clustered_subset(X, pool, min(count, len(pool)), rng)
            else:
                take = rng.choice(pool, size=min(count, len(pool)), replace=False)
            picked.extend(int(i) for i in np.atleast_1d(take))

        if len(picked) < n:  # top up from the whole class
            leftovers = np.array([i for i in idx if i not in picked])
            if len(leftovers):
                picked.extend(int(i) for i in rng.choice(leftovers, size=min(n - len(picked), len(leftovers)), replace=False))
        chosen_all.append(np.asarray(picked[:n]))
    return np.sort(np.concatenate(chosen_all))


def draw_subsets(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    n_subsets: int,
    seed: int = 0,
    scores: np.ndarray | None = None,
    bands: dict | None = None,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [mixed_subset(X, y, n, rng, scores=scores, bands=bands) for _ in range(int(n_subsets))]
