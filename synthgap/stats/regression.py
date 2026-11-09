"""Eq. (1): within-pool fixed-effects regression and the Shapley (LMG) decomposition.

Acc(S) = b_D D~ + b_C C~ + b_A A~ + b_F F~ + gamma_{G,pi,d} + eps

Regressors are standardised; gamma is a fixed effect for each generator-prompt-
dataset cell, so variance is explained only within pools.  The Shapley share of R^2
(Gromping 2006; Lipovetsky & Conklin 2001) is the average marginal R^2 contribution
of a factor over all orderings, which stays interpretable when regressors are
correlated.  With four factors that is exactly 16 subset fits, so we compute it
exactly rather than by sampling.
"""
from __future__ import annotations

from math import factorial
from typing import Sequence

import numpy as np


def _prepare(frame, factors: Sequence[str], target: str, fe: str | None, standardize: bool):
    frame = frame.dropna(subset=list(factors) + [target]).copy()
    if frame.empty:
        raise ValueError("no complete rows for the regression")
    X = frame[list(factors)].to_numpy(dtype=float).copy()
    if standardize:
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        X = (X - X.mean(axis=0)) / sd
    y = frame[target].to_numpy(dtype=float).copy()
    if fe is not None and fe in frame:
        groups = frame[fe].astype(str).to_numpy()
        for group in np.unique(groups):
            mask = groups == group
            if mask.sum() > 1:
                y[mask] = y[mask] - y[mask].mean()
                X[mask] = X[mask] - X[mask].mean(axis=0)
    return y, X


def r2_of_subset(y: np.ndarray, X: np.ndarray) -> float:
    if X.shape[1] == 0:
        return 0.0
    design = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    total = float(((y - y.mean()) ** 2).sum())
    if total <= 1e-12:
        return float("nan")
    return float(1.0 - ((residual**2).sum()) / total)


def shapley_r2(
    frame,
    factors: Sequence[str],
    target: str,
    fe: str | None = None,
    standardize: bool = True,
) -> dict:
    """Exact Shapley/LMG shares of R^2 over `factors` (p <= 8)."""
    factors = list(factors)
    y, X = _prepare(frame, factors, target, fe, standardize)
    p = len(factors)
    cache: dict[int, float] = {}
    for mask in range(1 << p):
        cols = [i for i in range(p) if (mask >> i) & 1]
        cache[mask] = r2_of_subset(y, X[:, cols]) if cols else 0.0

    shares = {}
    for i, factor in enumerate(factors):
        total = 0.0
        for mask in range(1 << p):
            if (mask >> i) & 1:
                continue
            size = bin(mask).count("1")
            weight = factorial(size) * factorial(p - size - 1) / factorial(p)
            total += weight * (cache[mask | (1 << i)] - cache[mask])
        shares[factor] = float(total)
    full_r2 = cache[(1 << p) - 1]
    return {"shares": shares, "r2": float(full_r2), "n": int(len(y)), "factors": factors}


def fit_ols(
    frame,
    factors: Sequence[str],
    target: str,
    fe: str | None = None,
    standardize: bool = True,
) -> dict:
    factors = list(factors)
    y, X = _prepare(frame, factors, target, fe, standardize)
    design = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {
        "intercept": float(coef[0]),
        "coefs": {f: float(c) for f, c in zip(factors, coef[1:])},
        "r2": r2_of_subset(y, X),
        "n": int(len(y)),
        "fixed_effects": fe,
    }
