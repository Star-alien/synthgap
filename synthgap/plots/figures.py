"""Figures 1-4. Reads the CSVs written by synthgap.pipeline; never calls a model."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import Cfg
from ..utils import Logger, ensure_dir

ORDER = ["sd15", "sdxl", "sd35", "flux"]
NICE = {"sd15": "SD1.5", "sdxl": "SDXL", "sd35": "SD3.5", "flux": "FLUX.1-dev", "real": "Real"}
FACTOR_COLORS = {"D": "#4C72B0", "C": "#DD8452", "A": "#55A868", "F": "#C44E52"}
FACTOR_LABELS = {"D": "Diversity (Vendi)", "C": "Coverage", "A": "Alignment", "F": "Precision"}


def _runs(cfg: Cfg) -> Path:
    return Path(cfg.paths.results) / "runs"


def _figures(cfg: Cfg) -> Path:
    return ensure_dir(Path(cfg.paths.results) / "figures")


def _ordered(families) -> list[str]:
    known = [f for f in ORDER if f in set(families)]
    return known + [f for f in dict.fromkeys(families) if f not in known]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def figure1(cfg: Cfg, log: Logger) -> Path | None:
    """Fidelity vs real-test accuracy across generator generations (P1)."""
    path = _runs(cfg) / "measure" / "table1.csv"
    if not path.exists():
        log.warn("figure 1 skipped: run `measure` first")
        return None
    frame = pd.read_csv(path)
    synthetic = frame[(frame.selection == "Random-N") & (frame.family.str.contains("vae") == False)]
    if synthetic.empty:
        log.warn("figure 1 skipped: no synthetic rows")
        return None
    regimes = [r for r in dict.fromkeys(synthetic.regime) if r != "-"]
    regime = "P1" if "P1" in regimes else regimes[0]
    synthetic = synthetic[synthetic.regime == regime]
    families = _ordered(synthetic.family.unique())
    vae = frame[frame.selection == "VAE-ceiling"]
    real_n = frame[(frame.family == "real") & (frame.selection.isin(["Real-N", "Real-all"]))]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    x = np.arange(len(families))
    for ax, metric, label, invert in (
        (axes[0], "precision", "Precision ($\\uparrow$)", False),
        (axes[1], "fd", "$FD_{DINO}$ (inverted)", True),
    ):
        ax2 = ax.twinx()
        acc = [synthetic[synthetic.family == f].accuracy.mean() * 100 for f in families]
        vals = [synthetic[synthetic.family == f][metric].mean() for f in families]
        ax.plot(x, acc, marker="o", color="#333333", label="Accuracy")
        ax.set_ylabel("Real-test accuracy (%)")
        ax2.plot(x, vals, marker="s", color="#C44E52", linestyle="--", label=label)
        ax2.set_ylabel(label)
        for f_index, family in enumerate(families):
            ceiling = vae[vae.family == f"vae.{family}"]
            if len(ceiling):
                ax.hlines(ceiling.accuracy.iloc[0] * 100, f_index - 0.35, f_index + 0.35, color="#888888", linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels([NICE.get(f, f) for f in families], rotation=15)
        ax.set_title(f"{cfg.dataset.name.upper()} - {regime}", fontsize=9)
    if len(real_n):
        axes[0].axhline(float(real_n.accuracy.mean()) * 100, color="#55A868", linestyle="-.", linewidth=1.1)
        axes[0].text(0.02, 0.04, "Real-N", transform=axes[0].transAxes, color="#55A868", fontsize=7)
    fig.suptitle("Looking better, teaching worse? (dotted = VAE ceiling)", fontsize=10)
    fig.tight_layout()
    out = _figures(cfg) / f"fig1_{cfg.dataset.name}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def figure2a(cfg: Cfg, log: Logger) -> Path | None:
    path = _runs(cfg) / "factorial" / "cells.csv"
    if not path.exists():
        log.warn("figure 2a skipped: run `factorial` first")
        return None
    frame = pd.read_csv(path)
    families = _ordered(frame.family.unique())
    cells = ["low/low", "low/high", "high/low", "high/high"]
    means = frame.groupby(["family", "cell"]).accuracy.mean()
    errors = frame.groupby(["family", "cell"]).accuracy.std() / np.sqrt(max(1, len(frame.selection_seed.unique())))

    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    width = 0.2
    for i, cell in enumerate(cells):
        values = [means.get((f, cell), np.nan) * 100 for f in families]
        errs = [errors.get((f, cell), 0) * 100 for f in families]
        ax.bar(np.arange(len(families)) + (i - 1.5) * width, values, width, yerr=errs, capsize=2, label=cell)
    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels([NICE.get(f, f) for f in families])
    ax.set_ylabel("Real-test accuracy (%)")
    ax.set_title("2x2 factorial: diversity x alignment (P2)", fontsize=9)
    ax.legend(fontsize=7, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    out = _figures(cfg) / f"fig2a_{cfg.dataset.name}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def figure2b(cfg: Cfg, log: Logger) -> Path | None:
    path = _runs(cfg) / "subsets" / "shapley.csv"
    if not path.exists():
        log.warn("figure 2b skipped: run `subsets` first")
        return None
    frame = pd.read_csv(path)
    models = ["within-pool", "between-generator"]
    models = [m for m in models if m in set(frame.model)]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    bottom = np.zeros(len(models))
    factors = [f for f in ["D", "C", "A", "F"] if f in set(frame.factor)]
    for factor in factors:
        values = np.array([float(frame[(frame.model == m) & (frame.factor == factor)].share.sum()) for m in models])
        ax.bar(models, values, bottom=bottom, color=FACTOR_COLORS[factor], label=FACTOR_LABELS[factor], width=0.55)
        bottom += values
    for i, model in enumerate(models):
        r2 = frame[frame.model == model].r2
        if len(r2):
            ax.text(i, bottom[i] + 0.01, f"$R^2$={float(r2.iloc[0]):.2f}", ha="center", fontsize=7)
    ax.set_ylabel("Shapley share of $R^2$")
    ax.set_title(f"{cfg.dataset.name.upper()}: what accounts for accuracy?", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    fig.tight_layout()
    out = _figures(cfg) / f"fig2b_{cfg.dataset.name}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def figure3(cfg: Cfg, log: Logger) -> Path | None:
    path = _runs(cfg) / "mixing" / "mixing.csv"
    if not path.exists():
        log.warn("figure 3 skipped: run `mixing` first")
        return None
    frame = pd.read_csv(path)
    family = _ordered(frame.family.unique())[0]
    frame = frame[frame.family == family]
    order = {"0": 0, "5": 5, "10": 10, "all": 30}
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for synthetic, style in (("none", "-"), ("Random-N", "--"), ("DAS-real", ":")):
        subset = frame[frame.synthetic == synthetic].copy()
        if subset.empty:
            continue
        subset["x"] = subset.real_per_class.astype(str).map(order)
        subset = subset.sort_values("x")
        ax.plot(subset.x, subset.accuracy * 100, marker="o", linestyle=style, label=synthetic)
    ax.set_xlabel("Real images per class")
    ax.set_ylabel("Real-test accuracy (%)")
    ax.set_title(f"Mixing curves ({NICE.get(family, family)}, P2)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = _figures(cfg) / f"fig3_{cfg.dataset.name}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def figure4(cfg: Cfg, log: Logger) -> Path | None:
    """Qualitative failure modes: lowest-alignment samples per class (rule, not hand-picked)."""
    from ..features.extract import npz_load_cached
    from ..gen.pool import load_pool

    regime = "P2" if "P2" in [str(r) for r in cfg.prompts.regimes] else str(cfg.prompts.regimes[0])
    family = [str(f) for f in cfg.generation.families][0]
    seed = int(cfg.generation.seeds[0])
    pool = load_pool(cfg, cfg.dataset.name, family, regime, seed)
    if pool is None:
        log.warn("figure 4 skipped: pool not found")
        return None
    candidates = list((Path(cfg.paths.features)).glob("align__*.npz"))
    if not candidates:
        log.warn("figure 4 skipped: alignment scores not found")
        return None
    data = npz_load_cached(sorted(candidates, key=lambda p: p.stat().st_mtime)[-1])
    scores = data["real"]
    rng = np.random.default_rng(0)
    classes = rng.choice(np.unique(np.asarray(pool.classes)), size=min(3, pool.n_classes), replace=False)
    fig, axes = plt.subplots(len(classes), 6, figsize=(7.4, 1.5 * len(classes)))
    axes = np.atleast_2d(axes)
    from PIL import Image

    for row, c in enumerate(classes):
        idx = np.flatnonzero(np.asarray(pool.classes) == c)
        lowest = idx[np.argsort(scores[idx])[:6]]
        for col, i in enumerate(lowest):
            ax = axes[row, col]
            ax.imshow(Image.open(pool.paths[i]).convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"A={scores[i]:.2f}", fontsize=6)
            if col == 0:
                ax.set_ylabel(str(cfg.dataset.name), fontsize=6)
    fig.suptitle(f"Lowest-alignment samples ({family}, {regime})", fontsize=9)
    fig.tight_layout()
    out = _figures(cfg) / f"fig4_{cfg.dataset.name}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def make_all_figures(cfg: Cfg, log: Logger | None = None) -> list[Path]:
    log = log or Logger("figures")
    _style()
    produced = []
    for fn in (figure1, figure2a, figure2b, figure3, figure4):
        try:
            out = fn(cfg, log)
            if out:
                produced.append(out)
                log.info(f"wrote {out}")
        except Exception as error:  # a missing stage should not break the others
            log.warn(f"{fn.__name__} failed: {error}")
    return produced
