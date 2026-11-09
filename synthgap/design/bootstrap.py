"""Cluster bootstrap over subsets and seeds (Sec. 4).

Subsets drawn from the same pool are not independent: they share images and a
pool. Resampling whole clusters (pool x subset, or pool x selection seed) is the
conservative choice and is what the paper reports.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd


def cluster_bootstrap(
    frame: pd.DataFrame,
    cluster_columns: Sequence[str],
    statistic: Callable[[pd.DataFrame], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile cluster bootstrap CI. Returns (lo, hi, point)."""
    cluster_columns = list(cluster_columns)
    groups = [g for _, g in frame.groupby(cluster_columns, sort=False)]
    point = float(statistic(frame))
    if len(groups) < 2:
        return point, point, point
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(int(n_boot)):
        picks = rng.integers(0, len(groups), size=len(groups))
        sample = pd.concat([groups[i] for i in picks], ignore_index=True)
        value = statistic(sample)
        if np.isfinite(value):
            estimates.append(float(value))
    if not estimates:
        return point, point, point
    lo, hi = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), point


def paired_bootstrap_difference(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """CI of mean(a - b) with replicates paired by position."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    diff = a - b
    point = float(diff.mean())
    if diff.size == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), point
