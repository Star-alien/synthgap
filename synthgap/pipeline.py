"""Experiment stages: Tables 1-2 and Figures 1-4 (Sec. 6).

Every stage writes tidy CSV/JSON under ``results/runs/<stage>/`` so figures and
tables can be rebuilt without re-running any model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import Cfg, resolve_dir
from .data.datasets import build_index
from .design.factorial import cell_name, summarize as summarize_factorial
from .design.subsets import draw_subsets
from .features.extract import compute_dataset_features, compute_pool_features, compute_text_features
from .gen.pool import Pool, build_pool, build_vae_ceiling_pool, load_pool
from .learn.probe import LinearProbe
from .metrics import METRIC_KEYS, alignment_summary, set_metrics
from .metrics.alignment import a_real, a_text
from .select.strategies import build_selection, class_groups, clustered_subset, kcenter_subset
from .utils import Logger, ensure_dir, fraction_of_gap_closed, json_dump, stable_hash

COLUMNS = [
    "dataset",
    "family",
    "regime",
    "seed",
    "selection",
    "accuracy",
    "vendi",
    "coverage",
    "precision",
    "recall",
    "density",
    "naeem_coverage",
    "fd",
    "align_real",
    "align_text",
    "n_images",
    "weight_decay",
    "mock",
]


def _save(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)
    return frame


class Study:
    """Loads the dataset, embeddings and pools once and shares them across stages."""

    def __init__(self, cfg: Cfg, log: Logger | None = None):
        self.cfg = cfg
        self.log = log or Logger("study", resolve_dir(cfg, "results") / "run.log")
        self.mock = bool(cfg.runtime.mock)
        self.index = build_index(cfg)
        self.log.info(self.index.summary())
        if self.mock:
            self.primary = self.secondary = self.selection_backend = "cheap"
            self.log.warn("MOCK MODE: procedural generators + cheap features. Outputs are NOT paper results.")
        else:
            self.primary = str(cfg.features.primary)
            self.secondary = str(cfg.features.secondary)
            self.selection_backend = str(cfg.features.selection)
        self.log.info(f"embeddings: primary={self.primary} secondary={self.secondary} selection={self.selection_backend}")
        self.real = compute_dataset_features(cfg, self.index, self.primary, self.log)
        self.real_sel = (
            self.real
            if self.selection_backend == self.primary
            else compute_dataset_features(cfg, self.index, self.selection_backend, self.log)
        )
        self.classes = list(range(self.index.num_classes))
        self.n = self.index.per_class_train
        self._pools: dict[tuple, Pool] = {}
        self._align: dict[tuple, dict] = {}

    # -- configuration -------------------------------------------------------------
    def families(self) -> list[str]:
        return [str(f) for f in self.cfg.generation.families]

    def regimes(self) -> list[str]:
        return [str(r) for r in self.cfg.prompts.regimes]

    def pool_seeds(self) -> list[int]:
        return [int(s) for s in self.cfg.generation.seeds]

    def selection_seeds(self) -> list[int]:
        return [int(s) for s in self.cfg.selection.seeds]

    def training_seeds(self) -> list[int]:
        return [int(s) for s in self.cfg.design.factorial_training_seeds]

    def run_dir(self, stage: str) -> Path:
        return resolve_dir(self.cfg, "results") / "runs" / stage

    def bands(self) -> dict:
        bands = self.cfg.selection.quantile_bands
        return {str(k): [float(v[0]), float(v[1])] for k, v in bands.items()}

    # -- pools ---------------------------------------------------------------------
    def pool(self, family: str, regime: str, seed: int) -> Pool:
        key = (family, regime, int(seed))
        if key in self._pools:
            return self._pools[key]
        pool = load_pool(self.cfg, self.index.name, family, regime, seed)
        if pool is None:
            pool = build_pool(self.cfg, self.index, family, regime, seed, self.log)
        self._pools[key] = pool
        return pool

    def vae_pool(self, family: str) -> Pool:
        key = (family, "VAE", 0)
        if key in self._pools:
            return self._pools[key]
        pool = build_vae_ceiling_pool(self.cfg, self.index, family, seed=0, log=self.log)
        self._pools[key] = pool
        return pool

    # -- alignment scores ----------------------------------------------------------
    def alignment(self, pool: Pool) -> dict:
        key = (pool.family, pool.regime, pool.seed)
        if key in self._align:
            return self._align[key]
        path = resolve_dir(self.cfg, "features") / (
            f"align__{pool.dataset}__{stable_hash(pool.family, pool.regime, pool.seed, pool.n)}"
            f"__{self.selection_backend}.npz"
        )
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                out = {"real": data["real"], "text": data["text"], "ncm_temperature": float(data["temperature"])}
        else:
            X, y = compute_pool_features(self.cfg, pool, self.selection_backend, self.log)
            grid = [float(x) for x in self.cfg.metrics.alignment_temperature_grid]
            real_scores, temperature = a_real(
                X, y, self.real_sel["X_alignment"], self.real_sel["y_alignment"], self.index.num_classes, grid
            )
            text_embeddings = compute_text_features(self.cfg, self.index, self.selection_backend, self.log)
            text_scores = a_text(X, y, text_embeddings)
            out = {
                "real": real_scores.astype(np.float32),
                "text": text_scores.astype(np.float32),
                "ncm_temperature": float(temperature),
            }
            np.savez(path, real=out["real"], text=out["text"], temperature=out["ncm_temperature"])
        self._align[key] = out
        return out

    # -- learners ------------------------------------------------------------------
    def real_subset(self, k: int | None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        y = self.real["y_train"]
        rng = np.random.default_rng(seed)
        picks = []
        for c in self.classes:
            candidates = np.flatnonzero(y == c)
            if k is None or k >= len(candidates):
                picks.append(candidates)
            else:
                picks.append(rng.choice(candidates, size=int(k), replace=False))
        indices = np.concatenate(picks) if picks else np.array([], dtype=int)
        return self.real["X_train"][indices], y[indices]

    def _fit_probe(self, X: np.ndarray, y: np.ndarray, train_seed: int) -> LinearProbe:
        probe = LinearProbe.from_config(self.cfg, self.index.num_classes)
        if str(self.cfg.learner.protocol) == "real_val_upper_bound" and len(self.real["y_val"]) > 0:
            probe.fit(X, y, self.real["X_val"], self.real["y_val"], seed=train_seed)
        else:
            probe.fit(X, y, seed=train_seed)
        return probe

    def evaluate(
        self,
        pool: Pool | None,
        indices: np.ndarray | None,
        *,
        train_seed: int = 0,
        real_k: int = 0,
        real_indices: np.ndarray | None = None,
        n_images: int | None = None,
    ) -> dict:
        if indices is not None:
            X, y = compute_pool_features(self.cfg, pool, self.primary, self.log)
            X_sel, y_sel = X[indices], y[indices]
        else:
            X_sel, y_sel = np.zeros((0, 1)), np.zeros(0, dtype=int)
        if real_indices is not None:
            X_sel = np.concatenate([X_sel, self.real["X_train"][real_indices]])
            y_sel = np.concatenate([y_sel, self.real["y_train"][real_indices]])
        elif real_k:
            Xr, yr = self.real_subset(real_k, seed=0)
            X_sel = np.concatenate([X_sel, Xr])
            y_sel = np.concatenate([y_sel, yr])
        if len(y_sel) == 0:
            raise ValueError("empty training set")
        metrics = set_metrics(X_sel, y_sel, self.real["X_train"], self.real["y_train"], self.cfg, self.classes)
        probe = self._fit_probe(X_sel, y_sel, train_seed)
        accuracy = probe.accuracy(self.real["X_test"], self.real["y_test"])
        row = {"accuracy": float(accuracy), "n_images": int(n_images if n_images is not None else len(y_sel))}
        row.update({key: float(metrics.get(key, np.nan)) for key in METRIC_KEYS})
        row["weight_decay"] = float(probe.weight_decay or 0.0)
        return row

    def real_baseline(self, k: int | None, train_seed: int = 0) -> dict:
        X, y = self.real_subset(k, seed=0)
        probe = self._fit_probe(X, y, train_seed)
        return {"accuracy": probe.accuracy(self.real["X_test"], self.real["y_test"]), "n_images": int(len(y))}

    def real_alignment_baseline(self, train_seed: int = 0) -> dict:
        X = self.real["X_alignment"]
        y = self.real["y_alignment"]
        probe = self._fit_probe(X, y, train_seed)
        return {"accuracy": probe.accuracy(self.real["X_test"], self.real["y_test"]), "n_images": int(len(y))}

    # -- factorial cells ------------------------------------------------------------
    def cell_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scores: np.ndarray,
        n: int,
        side: str,
        threshold_quantile: float,
        method: str,
        seed: int,
    ) -> np.ndarray:
        """One factorial cell: an alignment side x a within-side diversity rule.

        The alignment factor keeps images above (high) or below (low) the pool's
        threshold quantile; because that side holds m*N*(1-q) ~> N images, the
        diversity rule has room to act. Using a tercile instead would hand the rule
        exactly N images and make the diversity manipulation a no-op.
        """
        rng = np.random.default_rng(seed)
        chosen = []
        for _, idx in class_groups(y).items():
            values = scores[idx]
            threshold = float(np.quantile(values, threshold_quantile))
            keep = idx[values >= threshold] if side == "high" else idx[values <= threshold]
            if len(keep) < n:
                keep = idx[np.argsort(-values)[:n]]
            if method == "clustered":
                chosen.append(clustered_subset(X, keep, n, rng))
            else:
                chosen.append(kcenter_subset(X, keep, n))
        return np.sort(np.concatenate(chosen))


# --------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------
def stage_measure(study: Study) -> pd.DataFrame:
    """Table 1: cross-generator picture, one fixed selection rule (random-N)."""
    cfg, log = study.cfg, study.log
    log.section("stage: measure (Table 1)")
    common = {"dataset": study.index.name, "mock": study.mock}
    rows = []
    for k in (5, 10, None):
        result = study.real_baseline(k)
        rows.append(
            {**common, "family": "real", "regime": "-", "seed": -1, "selection": f"Real-{k or 'N'}",
             "accuracy": result["accuracy"], "n_images": result["n_images"], "weight_decay": 0.0}
        )
    for family in study.families():
        for regime in study.regimes():
            for seed in study.pool_seeds():
                pool = study.pool(family, regime, seed)
                scores = study.alignment(pool)
                Xsel, ysel = compute_pool_features(cfg, pool, study.selection_backend, log)
                indices = build_selection("random", Xsel, ysel, study.n, seed=seed)
                result = study.evaluate(pool, indices, train_seed=0)
                result["align_real"], _ = alignment_summary(scores["real"], ysel, study.classes)
                result["align_text"], _ = alignment_summary(scores["text"], ysel, study.classes)
                rows.append({**common, "family": family, "regime": regime, "seed": seed, "selection": "Random-N", **result})
                log.info(f"  {family}/{regime}/s{seed}: acc={result['accuracy']*100:.1f}")
    # autoencoder ceiling
    for family in study.families():
        if family in [str(f) for f in (study.cfg.get("generation.skip_vae") or [])]:
            log.warn(f"skipping VAE ceiling for {family} (generation.skip_vae)")
            continue
        pool = study.vae_pool(family)
        result = study.evaluate(pool, np.arange(pool.n), train_seed=0)
        rows.append({**common, "family": f"vae.{family}", "regime": "VAE", "seed": 0, "selection": "VAE-ceiling", **result})

    frame = pd.DataFrame(rows).reindex(columns=COLUMNS)
    return _save(frame.sort_values(["family", "regime", "seed"]), study.run_dir("measure") / "table1.csv")


def stage_factorial(study: Study) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sec. 6.2 / Fig. 2a: 2x2 factorial over diversity and alignment (P2)."""
    cfg, log = study.cfg, study.log
    log.section("stage: factorial (Fig. 2a)")
    threshold = float(cfg.selection.diversity_quantile)
    rows = []
    regime = "P2" if "P2" in study.regimes() else study.regimes()[0]
    for family in study.families():
        for seed in study.pool_seeds():
            pool = study.pool(family, regime, seed)
            scores = study.alignment(pool)["real"]
            Xsel, ysel = compute_pool_features(cfg, pool, study.selection_backend, log)
            for diversity, method in (("low", "clustered"), ("high", "kcenter")):
                for alignment in ("low", "high"):
                    indices = study.cell_selection(
                        Xsel, ysel, scores, study.n, alignment, threshold, method, seed=seed
                    )
                    for train_seed in study.training_seeds():
                        result = study.evaluate(pool, indices, train_seed=train_seed)
                        rows.append(
                            {
                                "dataset": study.index.name,
                                "family": family,
                                "cell": cell_name(diversity, alignment),
                                "selection_seed": seed,
                                "train_seed": train_seed,
                                "accuracy": result["accuracy"],
                                "vendi": result["vendi"],
                                "coverage": result["coverage"],
                                "precision": result["precision"],
                                "align_real": result["align_real"] if "align_real" in result else np.nan,
                                "mock": study.mock,
                            }
                        )
    frame = pd.DataFrame(rows)
    _save(frame, study.run_dir("factorial") / "cells.csv")

    effect_rows = []
    for (dataset, family), group in frame.groupby(["dataset", "family"]):
        per_cell = {cell: group[group.cell == cell].sort_values(["selection_seed", "train_seed"]).accuracy.to_list() for cell in group.cell.unique()}
        summary = summarize_factorial(per_cell, seed=0)
        for quantity in ("diversity", "alignment", "interaction"):
            effect_rows.append({"dataset": dataset, "family": family, "quantity": quantity, **summary[quantity]})
        for cell, stats in summary["cells"].items():
            effect_rows.append({"dataset": dataset, "family": family, "quantity": f"cell:{cell}", **stats})
    effects = pd.DataFrame(effect_rows)
    return frame, _save(effects, study.run_dir("factorial") / "effects.csv")


def stage_subsets(study: Study) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sec. 6.2 / Fig. 2b: balanced random-subset design + Shapley decomposition."""
    cfg, log = study.cfg, study.log
    log.section("stage: subsets (Fig. 2b)")
    regime = "P2" if "P2" in study.regimes() else study.regimes()[0]
    n_subsets = int(cfg.design.n_subsets)
    cache_path = study.run_dir("subsets") / "subsets.csv"
    expected = n_subsets * len(study.families())
    reuse = cache_path.exists() and len(pd.read_csv(cache_path)) >= expected
    rows = []
    if reuse:
        log.info(f"reusing {cache_path} ({expected} rows); delete it to recompute")
    for family in ([] if reuse else study.families()):
        pool = study.pool(family, regime, study.pool_seeds()[0])
        scores = study.alignment(pool)
        Xsel, ysel = compute_pool_features(cfg, pool, study.selection_backend, log)
        subsets = draw_subsets(Xsel, ysel, study.n, n_subsets, seed=0, scores=scores["real"], bands=study.bands())
        for i, indices in enumerate(subsets):
            result = study.evaluate(pool, indices, train_seed=0)
            align = float(np.mean(scores["real"][indices]))
            rows.append(
                {
                    "dataset": study.index.name,
                    "family": family,
                    "subset": i,
                    "accuracy": result["accuracy"],
                    "vendi": result["vendi"],
                    "coverage": result["coverage"],
                    "alignment": align,
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "density": result["density"],
                    "mock": study.mock,
                }
            )
            if (i + 1) % 50 == 0:
                log.info(f"  {family}: {i + 1}/{n_subsets} subsets")
    frame = pd.read_csv(cache_path) if reuse else pd.DataFrame(rows)
    _save(frame, cache_path)

    from .stats.regression import fit_ols, shapley_r2

    # factor label -> measured column (D, C, A, F as named in the paper)
    factors = {"D": "vendi", "C": "coverage", "A": "alignment", "F": "precision"}
    columns = list(factors.values())
    shapley_rows, regression_rows = [], []
    frame["cell"] = frame["family"]
    within = shapley_r2(frame, columns, "accuracy", fe="cell", standardize=True)
    for label, column in factors.items():
        shapley_rows.append({"dataset": study.index.name, "model": "within-pool", "factor": label,
                             "share": float(within["shares"][column]), "r2": within["r2"], "n": within["n"]})
    regression_rows.append({"dataset": study.index.name, "model": "within-pool",
                            **fit_ols(frame, columns, "accuracy", fe="cell")})

    between_frame = frame.groupby("family", as_index=False)[columns + ["accuracy"]].mean()
    if len(between_frame) >= 3:
        between = shapley_r2(between_frame, columns, "accuracy", fe=None, standardize=True)
        for label, column in factors.items():
            shapley_rows.append({"dataset": study.index.name, "model": "between-generator", "factor": label,
                                 "share": float(between["shares"][column]), "r2": between["r2"], "n": between["n"]})
        regression_rows.append({"dataset": study.index.name, "model": "between-generator",
                                **fit_ols(between_frame, columns, "accuracy")})

    shapley = pd.DataFrame(shapley_rows)
    _save(shapley, study.run_dir("subsets") / "shapley.csv")
    regression = pd.DataFrame(regression_rows)
    _save(regression, study.run_dir("subsets") / "regression.csv")

    from .stats.correlations import collinearity_report

    report = collinearity_report(frame, columns)
    json_dump(report, study.run_dir("subsets") / "collinearity.json")
    return frame, shapley, regression


def stage_das(study: Study) -> pd.DataFrame:
    """Sec. 6.4 / Table 2: DAS and baselines."""
    cfg, log = study.cfg, study.log
    log.section("stage: DAS (Table 2)")
    regime = "P2" if "P2" in study.regimes() else study.regimes()[0]
    tau = float(cfg.selection.das.quantile)
    rows = []
    for family in study.families():
        pool = study.pool(family, regime, study.pool_seeds()[0])
        scores = study.alignment(pool)
        Xsel, ysel = compute_pool_features(cfg, pool, study.selection_backend, log)
        methods = {
            "Random-N": build_selection("random", Xsel, ysel, study.n, seed=0),
            "All-mN": build_selection("all", Xsel, ysel, study.n),
            "k-center only": build_selection("kcenter", Xsel, ysel, study.n),
            "Filter only (A_text)": build_selection("filter", Xsel, ysel, study.n, scores=scores["text"], quantile=tau, seed=0),
            "DAS-text": build_selection("das", Xsel, ysel, study.n, scores=scores["text"], quantile=tau, seed=0),
            "Filter only (A_real)": build_selection("filter", Xsel, ysel, study.n, scores=scores["real"], quantile=tau, seed=0),
            "DAS-real": build_selection("das", Xsel, ysel, study.n, scores=scores["real"], quantile=tau, seed=0),
        }
        for name, indices in methods.items():
            result = study.evaluate(pool, indices, train_seed=0)
            rows.append(
                {"dataset": study.index.name, "family": family, "method": name, "accuracy": result["accuracy"],
                 "n_images": result["n_images"], "real_images": 5 if "real" in name and "A_real" not in name else 0,
                 "mock": study.mock}
            )
        das_real = methods["DAS-real"]
        result = study.evaluate(pool, das_real, train_seed=0, real_k=5)
        rows.append({"dataset": study.index.name, "family": family, "method": "Real 5-shot + DAS-real",
                     "accuracy": result["accuracy"], "n_images": result["n_images"], "real_images": 5, "mock": study.mock})

        # oracle P3 + random, if captions are available
        if "P3" in study.regimes():
            try:
                pool3 = study.pool(family, "P3", study.pool_seeds()[0])
                X3, y3 = compute_pool_features(cfg, pool3, study.selection_backend, log)
                indices3 = build_selection("random", X3, y3, study.n, seed=0)
                result = study.evaluate(pool3, indices3, train_seed=0)
                rows.append({"dataset": study.index.name, "family": family, "method": "P3 + Random-N",
                             "accuracy": result["accuracy"], "n_images": result["n_images"], "real_images": 0,
                             "mock": study.mock})
            except Exception as error:  # missing captions should not kill the stage
                log.warn(f"  skipping P3 for {family}: {error}")
    # matched real baselines
    rows.append({"dataset": study.index.name, "family": "real", "method": "Real 5-shot",
                 "accuracy": study.real_alignment_baseline()["accuracy"], "n_images": 5 * study.index.num_classes,
                 "real_images": 5, "mock": study.mock})
    rows.append({"dataset": study.index.name, "family": "real", "method": "Real-N",
                 "accuracy": study.real_baseline(None)["accuracy"], "n_images": study.n * study.index.num_classes,
                 "real_images": study.n, "mock": study.mock})

    frame = pd.DataFrame(rows)
    frame = _add_gap_closed(frame)
    return _save(frame, study.run_dir("das") / "table2.csv")


def _add_gap_closed(frame: pd.DataFrame) -> pd.DataFrame:
    """Eq. (2), computed per (dataset, family) against Random-N and Real-N."""
    frame = frame.copy()
    frame["delta"] = np.nan
    for (dataset,), group in frame.groupby(["dataset"]):
        real_n = group.loc[group.method == "Real-N", "accuracy"]
        real_n_value = float(real_n.iloc[0]) if len(real_n) else np.nan
        for family, sub in group.groupby("family"):
            random_n = sub.loc[sub.method == "Random-N", "accuracy"]
            if not len(random_n):
                continue
            value = fraction_of_gap_closed(sub.accuracy.to_numpy(), float(random_n.iloc[0]), real_n_value)
            frame.loc[sub.index, "delta"] = value
    return frame


def stage_mixing(study: Study) -> pd.DataFrame:
    """Sec. 6.4 / Fig. 3: does synthetic data still help once real data exists?"""
    cfg, log = study.cfg, study.log
    log.section("stage: mixing (Fig. 3)")
    regime = "P2" if "P2" in study.regimes() else study.regimes()[0]
    tau = float(cfg.selection.das.quantile)
    rows = []
    for family in study.families():
        pool = study.pool(family, regime, study.pool_seeds()[0])
        scores = study.alignment(pool)
        Xsel, ysel = compute_pool_features(cfg, pool, study.selection_backend, log)
        random_idx = build_selection("random", Xsel, ysel, study.n, seed=0)
        das_idx = build_selection("das", Xsel, ysel, study.n, scores=scores["real"], quantile=tau, seed=0)
        for k in (0, 5, 10, None):
            label = "all" if k is None else str(k)
            if k:  # a "real only" row is undefined at zero real images
                real_result = study.real_baseline(k)
                rows.append({"dataset": study.index.name, "family": family, "real_per_class": label, "synthetic": "none",
                             "accuracy": real_result["accuracy"], "mock": study.mock})
            for name, indices in (("Random-N", random_idx), ("DAS-real", das_idx)):
                result = study.evaluate(pool, indices, train_seed=0, real_k=k or 0)
                rows.append({"dataset": study.index.name, "family": family, "real_per_class": label, "synthetic": name,
                             "accuracy": result["accuracy"], "mock": study.mock})
    frame = pd.DataFrame(rows)
    return _save(frame, study.run_dir("mixing") / "mixing.csv")


def stage_ablation(study: Study) -> pd.DataFrame:
    """Supplement ablations: threshold quantile, first centre, selection embedding."""
    cfg, log = study.cfg, study.log
    log.section("stage: ablation")
    regime = "P2" if "P2" in study.regimes() else study.regimes()[0]
    rows = []
    for family in study.families():
        pool = study.pool(family, regime, study.pool_seeds()[0])
        scores = study.alignment(pool)
        Xsel, ysel = compute_pool_features(cfg, pool, study.selection_backend, log)
        for value in (0.2, 0.4, 0.6):
            indices = build_selection("das", Xsel, ysel, study.n, scores=scores["real"], quantile=value, seed=0)
            result = study.evaluate(pool, indices, train_seed=0)
            rows.append({"dataset": study.index.name, "family": family, "knob": "threshold_quantile",
                         "value": str(value), "accuracy": result["accuracy"], "mock": study.mock})
        # first-centre rule: centroid anchor instead of best-aligned anchor
        centroid_indices = []
        for c, idx in class_groups(ysel).items():
            values = scores["real"][idx]
            keep = idx[values >= np.quantile(values, float(cfg.selection.das.quantile))]
            centroid_indices.append(kcenter_subset(Xsel, keep, min(study.n, len(keep))))
        centroid_indices = np.sort(np.concatenate(centroid_indices))
        result = study.evaluate(pool, centroid_indices, train_seed=0)
        rows.append({"dataset": study.index.name, "family": family, "knob": "first_centre", "value": "centroid",
                     "accuracy": result["accuracy"], "mock": study.mock})
        if not study.mock and study.secondary != study.selection_backend:
            X2, y2 = compute_pool_features(cfg, pool, study.secondary, log)
            indices = build_selection("das", X2, y2, study.n, scores=scores["real"],
                                      quantile=float(cfg.selection.das.quantile), seed=0)
            result = study.evaluate(pool, indices, train_seed=0)
            rows.append({"dataset": study.index.name, "family": family, "knob": "selection_embedding",
                         "value": study.secondary, "accuracy": result["accuracy"], "mock": study.mock})
    frame = pd.DataFrame(rows)
    return _save(frame, study.run_dir("ablation") / "ablation.csv")


def stage_scratch(study: Study) -> pd.DataFrame:
    """Sec. 6.3: does the ranking survive a from-scratch ResNet?"""
    cfg, log = study.cfg, study.log
    log.section("stage: from-scratch control")
    from .learn.scratch import torch_available, train_from_scratch

    if not torch_available():
        log.warn("torch not installed: skipping the from-scratch control")
        return pd.DataFrame()
    regime = "P2" if "P2" in study.regimes() else study.regimes()[0]
    tau = float(cfg.selection.das.quantile)
    scratch_cfg = cfg.learner.scratch
    test_items = []
    for c in range(study.index.num_classes):
        for path in study.index.class_paths("test", c):
            test_items.append((str(path), c))
    rows = []
    for family in study.families()[: int(cfg.get("design.scratch_families", 2))]:
        pool = study.pool(family, regime, study.pool_seeds()[0])
        scores = study.alignment(pool)
        Xsel, ysel = compute_pool_features(cfg, pool, study.selection_backend, log)
        methods = {
            "Random-N": build_selection("random", Xsel, ysel, study.n, seed=0),
            "DAS-real": build_selection("das", Xsel, ysel, study.n, scores=scores["real"], quantile=tau, seed=0),
            "All-mN": build_selection("all", Xsel, ysel, study.n),
        }
        for name, indices in methods.items():
            train_items = [(pool.paths[i], int(pool.classes[i])) for i in indices]
            result = train_from_scratch(
                train_items, test_items, study.index.num_classes,
                arch=str(scratch_cfg.arch), epochs=int(scratch_cfg.epochs), lr=float(scratch_cfg.lr),
                batch_size=int(scratch_cfg.batch_size), resolution=int(scratch_cfg.resolution),
                weight_decay=float(scratch_cfg.weight_decay), seed=0, device=str(cfg.runtime.device), log=log,
            )
            probe = study.evaluate(pool, indices, train_seed=0)
            rows.append({"dataset": study.index.name, "family": family, "method": name,
                         "probe_accuracy": probe["accuracy"], "scratch_accuracy": result["accuracy"],
                         "epochs": result["epochs"], "mock": study.mock})
    frame = pd.DataFrame(rows)
    return _save(frame, study.run_dir("scratch") / "scratch.csv")


def stage_h3(study: Study) -> pd.DataFrame:
    """H3: is the gap larger for classes the generator knows poorly?

    Per class we regress the gap Acc_real(c) - Acc_synth(c) on class familiarity,
    the mean A_text of that class under P1. This is the within-dataset test the
    paper prefers over comparing to ImageNet numbers reported elsewhere.
    """
    cfg, log = study.cfg, study.log
    log.section("stage: H3 (class familiarity)")
    from scipy import stats as scipy_stats

    regime = "P1" if "P1" in study.regimes() else study.regimes()[0]
    real_X, real_y = study.real_subset(None)
    real_probe = study._fit_probe(real_X, real_y, train_seed=0)
    real_predictions = real_probe.predict(study.real["X_test"])
    test_y = study.real["y_test"]

    rows = []
    for family in study.families():
        pool = study.pool(family, regime, study.pool_seeds()[0])
        scores = study.alignment(pool)["text"]
        X, y = compute_pool_features(cfg, pool, study.primary, log)
        Xsel, ysel = compute_pool_features(cfg, pool, study.selection_backend, log)
        indices = build_selection("random", Xsel, ysel, study.n, seed=0)
        probe = study._fit_probe(X[indices], y[indices], train_seed=0)
        predictions = probe.predict(study.real["X_test"])
        for c in study.classes:
            mask = test_y == c
            if mask.sum() == 0:
                continue
            real_acc = float(np.mean(real_predictions[mask] == c))
            synth_acc = float(np.mean(predictions[mask] == c))
            rows.append(
                {
                    "dataset": study.index.name,
                    "family": family,
                    "class_index": c,
                    "class_name": study.index.classes[c],
                    "gap": real_acc - synth_acc,
                    "real_acc": real_acc,
                    "synthetic_acc": synth_acc,
                    "familiarity": float(np.mean(scores[y == c])),
                    "mock": study.mock,
                }
            )
    frame = pd.DataFrame(rows)
    _save(frame, study.run_dir("h3") / "per_class.csv")

    out = []
    for (dataset, family), group in frame.groupby(["dataset", "family"]):
        x = group.familiarity.to_numpy(dtype=float)
        gap = group.gap.to_numpy(dtype=float)
        if len(x) < 3 or len(np.unique(x)) < 2:
            continue
        fit = scipy_stats.linregress(x, gap)
        rng = np.random.default_rng(0)
        slopes = []
        for _ in range(1000):
            idx = rng.integers(0, len(x), len(x))
            if len(np.unique(x[idx])) < 2:
                continue
            slopes.append(scipy_stats.linregress(x[idx], gap[idx]).slope)
        lo, hi = np.percentile(slopes, [2.5, 97.5]) if slopes else (np.nan, np.nan)
        out.append(
            {
                "dataset": dataset,
                "family": family,
                "slope": float(fit.slope),
                "slope_lo": float(lo),
                "slope_hi": float(hi),
                "intercept": float(fit.intercept),
                "r2": float(fit.rvalue**2),
                "p": float(fit.pvalue),
                "n_classes": int(len(x)),
                "mock": study.mock,
            }
        )
    return _save(pd.DataFrame(out), study.run_dir("h3") / "regression.csv")
