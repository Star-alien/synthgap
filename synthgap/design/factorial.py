"""The 2x2 factorial over diversity and label alignment (Sec. 4).

Four cells per (generator, prompt regime, dataset):
    low diversity  x low alignment   (clustered, bottom tercile)
    low diversity  x high alignment  (clustered, top tercile)
    high diversity x low alignment   (k-center, bottom tercile)
    high diversity x high alignment  (k-center, top tercile)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


CELLS = ("low/low", "low/high", "high/low", "high/high")


def cell_name(diversity: str, alignment: str) -> str:
    return f"{diversity}/{alignment}"


def main_effects(means: dict[str, float]) -> dict[str, float]:
    """Main effects and interaction from the four cell means."""
    required = set(CELLS)
    missing = required - set(means)
    if missing:
        raise KeyError(f"missing factorial cells: {sorted(missing)}")
    diversity = 0.5 * (
        (means["high/low"] + means["high/high"]) - (means["low/low"] + means["low/high"])
    )
    alignment = 0.5 * (
        (means["low/high"] + means["high/high"]) - (means["low/low"] + means["high/low"])
    )
    interaction = (means["high/high"] - means["high/low"]) - (means["low/high"] - means["low/low"])
    return {"diversity": float(diversity), "alignment": float(alignment), "interaction": float(interaction)}


def summarize(
    accuracies: dict[str, Sequence[float]],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Cell means plus bootstrap CIs for both main effects and the interaction.

    Replicates are paired by position (same selection seed / training seed across
    cells), so resampling indices rather than values keeps the pairing.
    """
    lengths = {k: len(np.asarray(v)) for k, v in accuracies.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"cells have different replicate counts: {lengths}")
    n = next(iter(lengths.values()))
    if n == 0:
        raise ValueError("no replicates")

    arrays = {k: np.asarray(v, dtype=float) for k, v in accuracies.items()}
    means = {k: float(np.nanmean(v)) for k, v in arrays.items()}
    effects = main_effects(means)

    rng = np.random.default_rng(seed)
    boot = {key: [] for key in effects}
    boot_cells = {key: [] for key in arrays}
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        sample = {k: float(np.nanmean(v[idx])) for k, v in arrays.items()}
        for key, value in sample.items():
            boot_cells[key].append(value)
        for key, value in main_effects(sample).items():
            boot[key].append(value)

    out = {"means": means, "cells": {}}
    for key in arrays:
        lo, hi = np.percentile(boot_cells[key], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out["cells"][key] = {"mean": means[key], "lo": float(lo), "hi": float(hi)}
    for key in effects:
        lo, hi = np.percentile(boot[key], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[key] = {"mean": effects[key], "lo": float(lo), "hi": float(hi)}
    out["n_replicates"] = n
    return out
