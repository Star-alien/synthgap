"""Distributional fidelity: Frechet distance and KID.

We treat these as *proxies to be tested*, not as one of the four factors: both
mix fidelity and diversity, and FID's Inception features are biased toward
ImageNet classes (Kynkaanniemi et al., 2023). FD in DINOv2 space is reported as
FD_DINO, Inception FID/KID only for comparability.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg


def _gaussian_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    if X.shape[0] < 2:
        return mu, np.zeros((X.shape[1], X.shape[1]))
    sigma = np.cov(X, rowvar=False)
    return mu, np.atleast_2d(sigma)


def frechet_distance(X1: np.ndarray, X2: np.ndarray) -> float:
    mu1, sigma1 = _gaussian_stats(X1)
    mu2, sigma2 = _gaussian_stats(X2)
    if mu1.shape != mu2.shape:
        raise ValueError("feature dimensions differ")
    diff = mu1 - mu2
    product = sigma1 @ sigma2
    covmean = None
    try:
        covmean = linalg.sqrtm(product)
    except Exception:  # pragma: no cover - numerical fallback
        covmean = None
    if covmean is None or not np.isfinite(covmean).all():
        # symmetrise and retry with a small ridge, as in the reference FID code
        eps = 1e-6 * np.eye(product.shape[0])
        covmean = linalg.sqrtm((product + eps).astype(np.float64))
        if not np.isfinite(covmean).all():
            covmean = np.zeros_like(product)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def kid_score(X1: np.ndarray, X2: np.ndarray, degree: int = 3, coef: float = 1.0) -> float:
    """Unbiased MMD^2 with the polynomial kernel used by KID."""
    X1 = np.asarray(X1, dtype=np.float64)
    X2 = np.asarray(X2, dtype=np.float64)
    m, n = X1.shape[0], X2.shape[0]
    if m < 2 or n < 2:
        return float("nan")
    dim = X1.shape[1]

    def kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return (A @ B.T / dim + coef) ** degree

    k_xx = kernel(X1, X1)
    k_yy = kernel(X2, X2)
    k_xy = kernel(X1, X2)
    mmd = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
    mmd += (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
    mmd -= 2.0 * k_xy.mean()
    return float(mmd)
