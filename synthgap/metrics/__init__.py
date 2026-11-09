"""Set-level measurements (Sec. 3.2).

Every metric is computed per class and averaged, at equal N, because several of
them (notably Vendi) depend on sample size.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .alignment import a_real, a_text, class_means, per_class_mean
from .coverage import coverage_knn, density_and_coverage, improved_precision, improved_recall
from .diversity import vendi_per_class, vendi_score
from .fidelity import frechet_distance, kid_score

METRIC_KEYS = ("vendi", "coverage", "recall", "precision", "density", "naeem_coverage", "fd")


def per_class_metric(
    fn: Callable[..., float],
    X: np.ndarray,
    y: np.ndarray,
    X_real: np.ndarray,
    y_real: np.ndarray,
    classes: Sequence[int],
    **kwargs,
) -> float:
    values = []
    for c in classes:
        fake_mask = np.asarray(y) == c
        real_mask = np.asarray(y_real) == c
        if fake_mask.sum() < 2 or real_mask.sum() < 2:
            continue
        value = fn(X[fake_mask], X_real[real_mask], **kwargs)
        if np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def _density_coverage_pair(X_fake: np.ndarray, X_real: np.ndarray, k: int) -> tuple[float, float]:
    return density_and_coverage(X_real, X_fake, k=k)


def set_metrics(
    X_fake: np.ndarray,
    y_fake: np.ndarray,
    X_real: np.ndarray,
    y_real: np.ndarray,
    cfg,
    classes: Sequence[int] | None = None,
) -> dict[str, float]:
    """All reference-based and reference-free set metrics for one training set."""
    classes = list(range(int(max(int(np.max(y_fake)), int(np.max(y_real)))) + 1)) if classes is None else list(classes)
    k = int(cfg.metrics.knn_k)
    improved_k = int(cfg.metrics.improved_k)

    out: dict[str, float] = {}
    out["vendi"] = vendi_per_class(X_fake, y_fake)
    out["coverage"] = per_class_metric(coverage_knn, X_fake, y_fake, X_real, y_real, classes, k=k)
    out["recall"] = per_class_metric(improved_recall, X_fake, y_fake, X_real, y_real, classes, k=improved_k)
    out["precision"] = per_class_metric(
        improved_precision, X_fake, y_fake, X_real, y_real, classes, k=improved_k
    )
    out["fd"] = per_class_metric(frechet_distance, X_fake, y_fake, X_real, y_real, classes)

    densities, coverages = [], []
    for c in classes:
        fake_mask = np.asarray(y_fake) == c
        real_mask = np.asarray(y_real) == c
        if fake_mask.sum() < 2 or real_mask.sum() < 2:
            continue
        density, coverage = _density_coverage_pair(X_fake[fake_mask], X_real[real_mask], k)
        if np.isfinite(density):
            densities.append(density)
        if np.isfinite(coverage):
            coverages.append(coverage)
    out["density"] = float(np.mean(densities)) if densities else float("nan")
    out["naeem_coverage"] = float(np.mean(coverages)) if coverages else float("nan")
    return out


def alignment_summary(
    scores: np.ndarray, y: np.ndarray, classes: Sequence[int]
) -> tuple[float, float]:
    """Mean per-class alignment and the fraction of images below chance level."""
    means = per_class_mean(scores, y, classes)
    if not means:
        return float("nan"), float("nan")
    chance = 1.0 / max(1, len(classes))
    low = float(np.mean(np.asarray(scores) < chance))
    return float(np.mean(list(means.values()))), low
