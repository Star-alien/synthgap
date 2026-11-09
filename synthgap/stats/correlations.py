"""Rank correlations and collinearity diagnostics."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def spearman(a, b) -> tuple[float, float]:
    rho, p = stats.spearmanr(np.asarray(a, dtype=float), np.asarray(b, dtype=float), nan_policy="omit")
    return float(rho), float(p)


def kendall(a, b) -> tuple[float, float]:
    tau, p = stats.kendalltau(np.asarray(a, dtype=float), np.asarray(b, dtype=float), nan_policy="omit")
    return float(tau), float(p)


def correlation_matrix(frame: pd.DataFrame, columns: Sequence[str], method: str = "spearman") -> pd.DataFrame:
    data = frame[list(columns)].astype(float)
    if method == "spearman":
        return data.corr(method="spearman")
    if method == "kendall":
        return data.corr(method="kendall")
    return data.corr(method="pearson")


def dosing_order_agreement(values: np.ndarray, order: Sequence[int]) -> tuple[float, float]:
    """Kendall tau between a metric and the release order of the generators.

    `order` is the index of each point in release order (0 = oldest generator).
    """
    return kendall(np.asarray(values, dtype=float), np.asarray(order, dtype=float))


def collinearity_report(frame: pd.DataFrame, columns: Sequence[str]) -> dict:
    """Pairwise |rho| and the variance inflation factor of each regressor."""
    columns = list(columns)
    corr = correlation_matrix(frame, columns).abs()
    pairs = {}
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            pairs[f"{a}-{b}"] = float(corr.loc[a, b])
    vif = {}
    data = frame[columns].astype(float).dropna()
    for column in columns:
        others = [c for c in columns if c != column]
        if not others or len(data) < 3:
            vif[column] = float("nan")
            continue
        X = np.column_stack([np.ones(len(data))] + [data[c].to_numpy() for c in others])
        y = data[column].to_numpy()
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        residual = y - X @ coef
        total = ((y - y.mean()) ** 2).sum()
        r2 = 1.0 - (residual**2).sum() / max(total, 1e-12)
        vif[column] = float(1.0 / max(1e-6, 1.0 - r2))
    return {"pairs": pairs, "vif": vif, "max_abs_pair": max(pairs.values()) if pairs else float("nan")}
