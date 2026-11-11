"""Regenerate paper Figures 1-6 from the stored results. No numbers are typed in.

Reads results_pilot/runs/*.csv|json (Figs 1-5) and, for Fig 6, the generated SD1.5/P2 pool images
(data/pools, not in git) together with the cached alignment scores (data/features).
Writes results_pilot/figures/fig{1..6}.{pdf,png} and copies them to paper_cvpr/figs/.

    python scripts/make_figures.py            # all six
    python scripts/make_figures.py --skip6    # without the pool images
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results_pilot" / "runs"
OUT = ROOT / "results_pilot" / "figures"
PAPER = ROOT / "paper_cvpr" / "figs"
N_TEST = 585
FAMS = ["sd15", "sdturbo", "sdxl"]
NICE = {"sd15": "SD1.5", "sdturbo": "SD-Turbo", "sdxl": "SDXL"}
COL = {"sd15": "#4C72B0", "sdturbo": "#DD8452", "sdxl": "#55A868"}
MARK = {"sd15": "o", "sdturbo": "s", "sdxl": "D"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 300, "pdf.fonttype": 42,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


def wilson(p, n=N_TEST, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


# ---------------------------------------------------------------- Figure 1 (teaser)
def box(ax, xy, w, h, text, fc, bold=False, fs=8):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec="#444", lw=0.8))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333"))


def fig1():
    eff = pd.read_csv(RUNS / "factorial" / "effects.csv")
    fig = plt.figure(figsize=(7.2, 2.55))
    ax = fig.add_axes([0.0, 0.0, 0.52, 1.0]); ax.set_axis_off(); ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1)
    ax.text(0.0, 0.97, "Correlational (prior work)", fontweight="bold", fontsize=8.5, va="top")
    box(ax, (0.0, 0.70), 0.13, 0.14, "SD1.5", "#dfe3f5"); box(ax, (0.17, 0.70), 0.13, 0.14, "SDXL", "#d9efd9")
    box(ax, (0.34, 0.70), 0.15, 0.14, "newer $G$", "#fbe3cf")
    arrow(ax, (0.50, 0.77), (0.55, 0.77))
    box(ax, (0.55, 0.64), 0.44, 0.26, "architecture, VAE, training data,\nresolution, sampler: all differ", "#f2f2f2", fs=7.5)
    ax.text(0.0, 0.60, r"compare metric($G$) with Acc($G$) across $G\Rightarrow$ confounded", fontsize=7, color="#444", style="italic")
    ax.plot([0, 1], [0.54, 0.54], color="#999", lw=0.6)
    ax.text(0.0, 0.50, "Interventional (ours)", fontweight="bold", fontsize=8.5, va="top")
    box(ax, (0.0, 0.14), 0.19, 0.28, "fixed\n$(G,\\pi)$\npool $P$\n$3N$/class", "#dfe3f5")
    for y, t in [(0.33, "low/high $D$"), (0.22, "low/high $A$"), (0.11, "random\nmixtures")]:
        box(ax, (0.31, y - 0.03), 0.22, 0.095, t, "white", fs=7.5)
        arrow(ax, (0.19, 0.28), (0.31, y + 0.015)); arrow(ax, (0.53, y + 0.015), (0.60, 0.25))
    box(ax, (0.60, 0.15), 0.14, 0.22, "sets $S$\n($N$/class)\n$\\to$ probe", "#eeeeee", fs=7.5)
    arrow(ax, (0.74, 0.26), (0.78, 0.26))
    box(ax, (0.78, 0.10), 0.21, 0.32, "real-test\nacc.\nvs.\nmeasured\n$D,C,A,F$", "#d9efd9", fs=7.5)
    ax.text(0.0, 0.03, r"only the $\it{selection}$ differs $\Rightarrow$ effects attributable to set composition", fontsize=7, color="#444")
    for i, fam in enumerate(FAMS):
        a = fig.add_axes([0.625 + i * 0.130, 0.20, 0.080, 0.62])
        e = eff[eff.family == fam].set_index("quantity")["mean"]
        for dv, ls, mk in [("low", "-", "o"), ("high", "--", "^")]:
            a.plot([0, 1], [100 * e[f"cell:{dv}/low"], 100 * e[f"cell:{dv}/high"]], ls=ls, marker=mk, color=COL[fam], ms=4, lw=1.3)
        a.set_xticks([0, 1]); a.set_xticklabels(["low", "high"]); a.set_xlim(-0.2, 1.2)
        a.set_xlabel("label\nalignment", fontsize=7); a.tick_params(labelsize=7); a.set_title(NICE[fam], color=COL[fam], fontsize=8.5)
        a.text(0.04, 0.97, f"align.\neffect\n{100 * e['alignment']:+.1f} pp", transform=a.transAxes, va="top", fontsize=6)
        if i == 0:
            a.set_ylabel("Real-test acc. (%)")
    h = [plt.Line2D([], [], color="#333", marker="o", ls="-", ms=4), plt.Line2D([], [], color="#333", marker="^", ls="--", ms=4)]
    fig.legend(h, ["low diversity", "high diversity"], loc="upper center", bbox_to_anchor=(0.80, 1.0), ncol=2, frameon=False, fontsize=7)
    save(fig, "fig1")


# ---------------------------------------------------------------- Figure 2 (generator metrics)
def fig2():
    t = pd.read_csv(RUNS / "measure" / "table1.csv")
    real = t[t.selection == "Real-N"].accuracy.iloc[0]
    lo, hi = wilson(real)
    metrics = [("fd", r"FD$_{\mathrm{DINO}}$ ($\downarrow$)", True), ("precision", r"Precision ($\uparrow$)", False),
               ("coverage", r"Coverage ($\uparrow$)", False), ("vendi", r"Vendi ($\uparrow$)", False),
               ("align_text", r"$A_{\mathrm{text}}$ ($\uparrow$)", False)]
    fig, axs = plt.subplots(1, 5, figsize=(7.6, 1.9), sharey=True)
    for ax, (col, lab, inv) in zip(axs, metrics):
        ax.axhspan(100 * lo, 100 * hi, color="#bbb", alpha=0.5, lw=0)
        ax.axhline(100 * real, color="#333", ls="--", lw=0.8)
        for fam in FAMS:
            pts = []
            for reg in ("P1", "P2"):
                r = t[(t.family == fam) & (t.regime == reg) & (t.selection == "Random-N")].iloc[0]
                a = r.accuracy; l, h = wilson(a)
                ax.errorbar(r[col], 100 * a, yerr=[[100 * (a - l)], [100 * (h - a)]], color=COL[fam], marker=MARK[fam], ms=5,
                            mfc=COL[fam] if reg == "P1" else "white", mew=1.1, lw=0.8, capsize=0)
                pts.append((r[col], 100 * a))
            ax.plot(*zip(*pts), color=COL[fam], lw=0.6, alpha=0.7)
        if inv:
            ax.invert_xaxis()
        ax.set_xlabel(lab)
    axs[0].set_ylabel("Real-test acc. (%)"); axs[0].text(0.02, 0.96, "Real-$N$", transform=axs[0].transAxes, fontsize=6.5, va="top", color="#555")
    hs = [plt.Line2D([], [], marker=MARK[f], color=COL[f], ls="", ms=5) for f in FAMS]
    hs += [plt.Line2D([], [], marker="o", color="#333", ls="", ms=5), plt.Line2D([], [], marker="o", mfc="white", color="#333", ls="", ms=5)]
    fig.legend(hs, [NICE[f] for f in FAMS] + ["P1 (class name)", "P2 (diversified)"], loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.12), fontsize=7)
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig2")


# ---------------------------------------------------------------- Figure 3 (collinearity)
def fig3():
    js = json.load(open(RUNS / "subsets" / "collinearity.json"))
    names = ["vendi", "coverage", "alignment", "precision"]; short = ["D", "C", "A", "F"]
    M = np.eye(4)
    for k, v in js["pairs"].items():
        a, b = k.split("-"); i, j = names.index(a), names.index(b); M[i, j] = M[j, i] = v
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.4, 1.7), gridspec_kw=dict(width_ratios=[1, 1.15]))
    a1.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1.0)
    a1.grid(False)
    for i in range(4):
        for j in range(4):
            a1.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(M[i, j]) > 0.6 else "black")
    a1.set_xticks(range(4)); a1.set_xticklabels(short)
    a1.set_yticks(range(4)); a1.set_yticklabels([f"{s} (VIF {js['vif'][n]:.0f})" for s, n in zip(short, names)], fontsize=6)
    a1.set_title("factor correlations", fontsize=7.5)
    for s in a1.spines.values(): s.set_visible(False)
    sub = pd.read_csv(RUNS / "subsets" / "subsets.csv")
    for fam in FAMS:
        d = sub[sub.family == fam]
        a2.scatter(d.coverage, 100 * d.accuracy, s=5, alpha=0.5, color=COL[fam], marker=MARK[fam], lw=0, label=NICE[fam])
    a2.set_xlabel("coverage of subset"); a2.set_ylabel("acc. (%)")
    a2.set_title(f"{sub.groupby('family').size().iloc[0]} random subsets / generator", fontsize=7.5)
    a2.legend(frameon=False, fontsize=6, markerscale=1.5, loc="lower left", handletextpad=0.1)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig3")


# ---------------------------------------------------------------- Figure 4 (per-class gap)
def fig4():
    pc = pd.read_csv(RUNS / "h3" / "per_class.csv"); rg = pd.read_csv(RUNS / "h3" / "regression.csv").set_index("family")
    fig, axs = plt.subplots(1, 3, figsize=(3.6, 1.75), sharey=True)
    for ax, fam in zip(axs, FAMS):
        d = pc[pc.family == fam]
        ax.scatter(d.familiarity, 100 * d.gap, s=9, color=COL[fam], lw=0, alpha=0.9)
        r = rg.loc[fam]; xs = np.array([0, 1])
        ax.plot(xs, 100 * (r.intercept + r.slope * xs), color="#333", lw=0.8)
        ax.text(0.97, 0.62, f"$R^2$={r.r2:.2f}", transform=ax.transAxes, ha="right", fontsize=6.5)
        w = d.loc[d.gap.idxmax()]
        ax.annotate(w.class_name.split(".", 1)[1].replace("_", "-"), (w.familiarity, 100 * w.gap), xytext=(0.15, 0.88),
                    textcoords="axes fraction", fontsize=5.5, arrowprops=dict(arrowstyle="-", lw=0.4, color="#666"))
        ax.set_title(NICE[fam], color=COL[fam], fontsize=8)
        ax.set_xlim(-0.05, 1.05); ax.set_xticks([0, 1]); ax.set_xlabel(r"familiarity (mean $A_{\mathrm{text}}$)", fontsize=6)
    axs[0].set_ylabel("per-class gap (pp)")
    fig.tight_layout(w_pad=0.4)
    save(fig, "fig4")


# ---------------------------------------------------------------- Figure 5 (DAS gains)
def fig5():
    d = pd.read_csv(RUNS / "das" / "table2.csv")
    meths = [("All-mN", "All-$mN$"), ("k-center only", r"$k$-ctr"), ("Filter only (A_text)", r"Filt$_{\mathrm{text}}$"),
             ("DAS-text", r"DAS$_{\mathrm{text}}$"), ("Filter only (A_real)", r"Filt$_{\mathrm{real}}$"), ("DAS-real", r"DAS$_{\mathrm{real}}$")]
    fig, ax = plt.subplots(figsize=(3.4, 1.8))
    xs = np.arange(len(meths)); w = 0.26
    ax.axvspan(3.5, 5.5, color="#ddd", alpha=0.6, lw=0)
    for i, fam in enumerate(FAMS):
        s = d[d.family == fam].set_index("method").accuracy
        ax.bar(xs + (i - 1) * w, [100 * (s[m] - s["Random-N"]) for m, _ in meths], w, color=COL[fam], label=NICE[fam])
    ax.axhline(0, color="#333", lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in meths], fontsize=6.5)
    ax.set_ylabel(r"acc. $-$ Random-$N$ (pp)")
    ax.text(1.5, 0.97, "no real images", transform=ax.get_xaxis_transform(), ha="center", fontsize=6, color="#555", va="top")
    ax.text(4.5, 0.97, "uses 5 real/class", transform=ax.get_xaxis_transform(), ha="center", fontsize=6, color="#555", va="top")
    ax.legend(frameon=False, ncol=3, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 1.22))
    fig.tight_layout()
    save(fig, "fig5")


# ---------------------------------------------------------------- Figure 6 (what the filter removes)
def fig6():
    from PIL import Image
    from synthgap.utils import stable_hash

    pool = ROOT / "data" / "pools" / "cub" / "sd15" / "p2" / "seed0"
    sc = np.load(ROOT / "data" / "features" / f"align__cub__{stable_hash('sd15', 'P2', 0, 1440)}__clip.npz")["real"]
    names = pd.read_csv(RUNS / "h3" / "per_class.csv").query("family=='sd15'").set_index("class_index").class_name
    K = 20; per = len(sc) // K
    rng = np.random.default_rng(0)
    classes = sorted(rng.choice(K, 3, replace=False).tolist())  # fixed rule: seed 0, not hand-picked
    fig, axs = plt.subplots(3, 6, figsize=(3.6, 2.0))
    for r, c in enumerate(classes):
        low = np.argsort(sc[c * per:(c + 1) * per])[:6]
        files = sorted((pool / f"{c:04d}").glob("*.jpg"))
        for k, j in enumerate(low):
            axs[r, k].imshow(Image.open(files[j]).convert("RGB")); axs[r, k].axis("off")
        nm = names[c].split(".", 1)[1].replace("_", " ").rsplit(" ", 1)
        axs[r, 0].text(-0.08, 0.5, "\n".join(nm), transform=axs[r, 0].transAxes, rotation=90,
                       ha="right", va="center", fontsize=5.5, linespacing=1.0)
    fig.subplots_adjust(wspace=0.03, hspace=0.03, left=0.08)
    save(fig, "fig6")
    print("fig6 classes:", [names[c] for c in classes])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--skip6", action="store_true"); a = ap.parse_args()
    fig1(); fig2(); fig3(); fig4(); fig5()
    if not a.skip6:
        fig6()
    PAPER.mkdir(parents=True, exist_ok=True)
    for n in range(1, 7):
        for ext in ("pdf", "png"):
            if (OUT / f"fig{n}.{ext}").exists():
                shutil.copy(OUT / f"fig{n}.{ext}", PAPER / f"fig{n}.{ext}")
