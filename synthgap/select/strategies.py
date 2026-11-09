"""Selection rules (Sec. 4 and Sec. 5).

Every primitive takes the *global* pool indices of a class and returns exactly
`n` global indices. Selection embeddings are always CLIP; measurement embeddings
are always DINOv2, so no selector can optimise the space it is scored in.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..metrics.diversity import vendi_score


def class_groups(y: np.ndarray) -> dict[int, np.ndarray]:
    y = np.asarray(y)
    return {int(c): np.flatnonzero(y == c) for c in np.unique(y)}


def _sample(candidates: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    candidates = np.asarray(candidates)
    if len(candidates) <= n:
        return candidates
    return rng.choice(candidates, size=n, replace=False)


# --------------------------------------------------------------------------------------
# within-class diversity rules
# --------------------------------------------------------------------------------------
def clustered_subset(X: np.ndarray, candidates: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """The n nearest neighbours of a random seed image: low diversity."""
    candidates = np.asarray(candidates)
    if len(candidates) <= n:
        return candidates
    points = np.asarray(X)[candidates]
    seed = int(rng.integers(len(candidates)))
    distances = np.linalg.norm(points - points[seed], axis=1)
    return candidates[np.argsort(distances)[:n]]


def kcenter_subset(X: np.ndarray, candidates: np.ndarray, n: int, first_local: int | None = None) -> np.ndarray:
    """Greedy farthest-point (k-center) selection: high diversity."""
    candidates = np.asarray(candidates)
    if len(candidates) <= n:
        return candidates
    points = np.asarray(X)[candidates]
    chosen = np.empty(n, dtype=int)
    if first_local is None:
        centroid = points.mean(axis=0)
        first_local = int(np.argmin(np.linalg.norm(points - centroid, axis=1)))
    chosen[0] = int(first_local)
    min_dist = np.linalg.norm(points - points[chosen[0]], axis=1)
    for i in range(1, n):
        nxt = int(np.argmax(min_dist))
        chosen[i] = nxt
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[nxt], axis=1))
    return candidates[chosen]


# --------------------------------------------------------------------------------------
# whole-set rules
# --------------------------------------------------------------------------------------
def select_random(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(
        np.concatenate([_sample(idx, n, rng) for idx in class_groups(y).values()])
    )


def select_all(y: np.ndarray) -> np.ndarray:
    return np.arange(len(y))


def select_kcenter(X: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    return np.sort(np.concatenate([kcenter_subset(X, idx, n) for idx in class_groups(y).values()]))


def select_clustered(X: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(
        np.concatenate([clustered_subset(X, idx, n, rng) for idx in class_groups(y).values()])
    )


def select_filtered(
    scores: np.ndarray,
    y: np.ndarray,
    n: int,
    quantile: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Discard images below the pool quantile, keep top-N by alignment if short."""
    chosen = []
    for _, idx in class_groups(y).items():
        threshold = float(np.quantile(scores[idx], quantile))
        keep = idx[scores[idx] >= threshold]
        if len(keep) < n:
            keep = idx[np.argsort(-scores[idx])[:n]]
        chosen.append(_sample(keep, n, rng))
    return np.sort(np.concatenate(chosen))


def select_band(
    scores: np.ndarray, y: np.ndarray, n: int, band: Sequence[float], rng: np.random.Generator
) -> np.ndarray:
    """Random draws restricted to a score quantile band [lo, hi)."""
    lo, hi = float(band[0]), float(band[1])
    chosen = []
    for _, idx in class_groups(y).items():
        values = scores[idx]
        low, high = np.quantile(values, lo), np.quantile(values, hi)
        keep = idx[(values >= low) & (values <= high)]
        if len(keep) < n:
            keep = idx[np.argsort(-values)[:n]]
        chosen.append(_sample(keep, n, rng))
    return np.sort(np.concatenate(chosen))


def pool_median_vendi(
    X: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator, draws: int = 100
) -> dict[int, float]:
    medians = {}
    for c, idx in class_groups(y).items():
        values = [vendi_score(np.asarray(X)[_sample(idx, n, rng)]) for _ in range(draws)]
        medians[c] = float(np.median(values))
    return medians


def match_vendi(
    X: np.ndarray,
    candidates: np.ndarray,
    n: int,
    target: float,
    n_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw random subsets and keep the one whose Vendi is closest to `target`."""
    best, best_error = None, np.inf
    for _ in range(int(n_candidates)):
        subset = _sample(candidates, n, rng)
        error = abs(vendi_score(np.asarray(X)[subset]) - target)
        if error < best_error:
            best, best_error = subset, error
    return np.asarray(best)


# --------------------------------------------------------------------------------------
# the arms of Sec. 4 and Sec. 5
# --------------------------------------------------------------------------------------
def select_diversity_arm(
    X: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    n: int,
    level: str,
    quantile: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fix alignment (keep the pool's top (1-quantile) fraction), vary diversity."""
    chosen = []
    for c, idx in class_groups(y).items():
        threshold = float(np.quantile(scores[idx], quantile))
        keep = idx[scores[idx] >= threshold]
        if len(keep) < n:
            keep = idx[np.argsort(-scores[idx])[:n]]
        if level == "high":
            chosen.append(kcenter_subset(X, keep, n))
        elif level == "low":
            chosen.append(clustered_subset(X, keep, n, rng))
        else:
            chosen.append(_sample(keep, n, rng))
    return np.sort(np.concatenate(chosen))


def select_alignment_arm(
    X: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    n: int,
    level: str,
    bands: dict[str, Sequence[float]],
    rng: np.random.Generator,
    targets: dict[int, float] | None = None,
    n_candidates: int = 200,
) -> np.ndarray:
    """Vary the alignment tercile while matching Vendi to the pool median."""
    if targets is None:
        targets = pool_median_vendi(X, y, n, rng)
    band = bands[level]
    chosen = []
    for c, idx in class_groups(y).items():
        values = scores[idx]
        low, high = np.quantile(values, band[0]), np.quantile(values, band[1])
        keep = idx[(values >= low) & (values <= high)]
        if len(keep) < n:
            keep = idx[np.argsort(-values)[:n]]
        chosen.append(match_vendi(X, keep, n, targets.get(c, float("nan")), n_candidates, rng))
    return np.sort(np.concatenate(chosen))


def select_das(
    X: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    n: int,
    quantile: float = 0.4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """DAS: filter by alignment, then greedy k-center anchored at the best-aligned image."""
    chosen = []
    for _, idx in class_groups(y).items():
        values = scores[idx]
        threshold = float(np.quantile(values, quantile))
        keep = idx[values >= threshold]
        if len(keep) < n:
            keep = idx[np.argsort(-values)[:n]]
        anchor = int(np.argmax(scores[keep]))
        chosen.append(kcenter_subset(X, keep, n, first_local=anchor))
    return np.sort(np.concatenate(chosen))


def build_selection(
    kind: str,
    X_sel: np.ndarray,
    y: np.ndarray,
    n: int,
    *,
    scores: np.ndarray | None = None,
    seed: int = 0,
    quantile: float = 0.4,
    bands: dict | None = None,
    targets: dict[int, float] | None = None,
    n_candidates: int = 200,
) -> np.ndarray:
    """Named selection rule used by the experiment stages."""
    rng = np.random.default_rng(seed)
    bands = bands or {"low": [0.0, 0.3333], "mid": [0.3333, 0.6667], "high": [0.6667, 1.0]}
    if kind == "random":
        return select_random(y, n, rng)
    if kind == "all":
        return select_all(y)
    if kind == "kcenter":
        return select_kcenter(X_sel, y, n)
    if kind == "clustered":
        return select_clustered(X_sel, y, n, rng)
    if kind == "filter":
        if scores is None:
            raise ValueError("'filter' needs alignment scores")
        return select_filtered(scores, y, n, quantile, rng)
    if kind in {"align_low", "align_mid", "align_high"}:
        if scores is None:
            raise ValueError(f"{kind} needs alignment scores")
        level = {"align_low": "low", "align_mid": "mid", "align_high": "high"}[kind]
        return select_alignment_arm(X_sel, y, scores, n, level, bands, rng, targets, n_candidates)
    if kind in {"div_low", "div_high", "div_mid"}:
        if scores is None:
            raise ValueError(f"{kind} needs alignment scores")
        return select_diversity_arm(X_sel, y, scores, n, kind.split("_")[1], quantile, rng)
    if kind == "das":
        if scores is None:
            raise ValueError("'das' needs alignment scores")
        return select_das(X_sel, y, scores, n, quantile, rng)
    raise ValueError(f"unknown selection kind {kind!r}")
