"""Recompute the statistics quoted in paper_cvpr/main.tex from results_pilot/ and compare.

Run from the repo root:  python scripts/verify_paper_numbers.py
Exit code 1 if any quoted number disagrees with the stored results.
"""
import json
import sys
from math import sqrt
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, kendalltau, norm

R = Path("results_pilot/runs")
N_TEST = 585
bad = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"{'ok ' if ok else 'BAD'}  {name:58s} computed {got:10.4f}  paper {want:10.4f}")
    if not ok:
        bad.append(name)


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def p2(p1, p2_, n=N_TEST):
    """Two-sided unpaired two-proportion z-test (the paper's conservative test)."""
    pp = (p1 + p2_) / 2
    se = sqrt(2 * pp * (1 - pp) / n)
    return 2 * (1 - norm.cdf(abs(p1 - p2_) / se)) if se > 0 else 1.0


t1 = pd.read_csv(R / "measure/table1.csv")
t3 = pd.read_csv(R / "das/table2.csv")
cells = pd.read_csv(R / "factorial/cells.csv")
sh = pd.read_csv(R / "subsets/shapley.csv")
col = json.load(open(R / "subsets/collinearity.json"))
h3 = pd.read_csv(R / "h3/regression.csv").set_index("family")
pc = pd.read_csv(R / "h3/per_class.csv")
abl = pd.read_csv(R / "ablation/ablation.csv")
mix = pd.read_csv(R / "mixing/mixing.csv")
acc = lambda f, r: t1[(t1.family == f) & (t1.regime == r)].accuracy.iloc[0]

print("--- Sec 6.1 cross-generator")
check("SDXL P1 acc", 100 * acc("sdxl", "P1"), 96.2, 0.05)
check("SD1.5 P1 acc", 100 * acc("sd15", "P1"), 90.9, 0.05)
check("SD-Turbo P1 acc", 100 * acc("sdturbo", "P1"), 91.5, 0.05)
print("     p(SDXL vs SD1.5) =", f"{p2(acc('sdxl','P1'), acc('sd15','P1')):.2e}", "(paper: <0.001)")
print("     p(SDXL vs Turbo) =", f"{p2(acc('sdxl','P1'), acc('sdturbo','P1')):.2e}", "(paper: <0.001)")
check("p(SD-Turbo vs SD1.5)", p2(acc("sdturbo", "P1"), acc("sd15", "P1")), 0.76, 0.01)
for r in ["P1", "P2"]:
    check(f"Kendall tau release order vs acc ({r})",
          kendalltau([1, 2, 3], [acc("sd15", r), acc("sdxl", r), acc("sdturbo", r)]).statistic, 0.33, 0.01)
for f, lo, hi in [("real", 0, 0)]:
    pass
for sel, lo, hi in [("Real-5", 96.4, 98.8), ("Real-10", 96.0, 98.6), ("Real-N", 97.3, 99.3)]:
    a = t1[t1.selection == sel].accuracy.iloc[0]
    l, h = wilson(a, N_TEST)
    check(f"Wilson lo {sel}", 100 * l, lo, 0.06); check(f"Wilson hi {sel}", 100 * h, hi, 0.06)
for f, r, lo, hi in [("sd15", "P1", 88.3, 93.0), ("sdturbo", "P2", 91.6, 95.5), ("sdxl", "P1", 94.4, 97.5)]:
    l, h = wilson(acc(f, r), N_TEST)
    check(f"Wilson lo {f} {r}", 100 * l, lo, 0.06); check(f"Wilson hi {f} {r}", 100 * h, hi, 0.06)
dv = [t1[(t1.family == f) & (t1.regime == "P2")].vendi.iloc[0] - t1[(t1.family == f) & (t1.regime == "P1")].vendi.iloc[0] for f in ["sd15", "sdturbo", "sdxl"]]
check("P2-P1 Vendi gain min", min(dv), 2.5, 0.1); check("P2-P1 Vendi gain max", max(dv), 3.9, 0.1)
check("max |P2-P1 acc| SD1.5/SDXL (pts)", max(abs(100 * (acc(f, "P2") - acc(f, "P1"))) for f in ["sd15", "sdxl"]), 0.4, 0.06)
real_n = t1[t1.selection == "Real-N"].accuracy.iloc[0]
vae = {f: t1[t1.family == "vae." + f].accuracy.iloc[0] for f in ["sd15", "sdturbo"]}
check("real - VAE ceiling max gap (pts)", 100 * max(real_n - v for v in vae.values()), 0.3, 0.06)
check("SD1.5/SD-Turbo gap to real, min (pts)", 100 * min(real_n - acc("sd15", "P1"), real_n - acc("sdturbo", "P1")), 7.2, 0.1)
check("SD1.5/SD-Turbo gap to real, max (pts)", 100 * max(real_n - acc("sd15", "P1"), real_n - acc("sdturbo", "P1")), 7.7, 0.15)
check("max improved recall < 0.05", t1[t1.regime.isin(["P1", "P2"])].recall.max(), 0.04, 0.01)

print("--- Sec 6.2 factorial (Table 2)")
want = {"sd15": (0.1, 12.2, 1.50, -4.33, 0.039, 0.199, 0.089, -0.023, 1.6),
        "sdturbo": (-0.7, 3.9, 1.42, -2.60, 0.108, -0.023, 0.102, 0.042, 0.3),
        "sdxl": (-0.5, 1.3, 1.37, -3.05, 0.127, -0.004, 0.137, -0.120, 0.9)}
for f, w in want.items():
    c = cells[cells.family == f].groupby("cell").mean(numeric_only=True)
    ef = lambda col_, arm: (c.loc[["high/low", "high/high"] if arm == "d" else ["low/high", "high/high"], col_].mean()
                            - c.loc[["low/low", "low/high"] if arm == "d" else ["low/low", "high/low"], col_].mean())
    inter = (c.loc["high/high", "accuracy"] - c.loc["high/low", "accuracy"]) - (c.loc["low/high", "accuracy"] - c.loc["low/low", "accuracy"])
    got = [100 * ef("accuracy", "d"), 100 * ef("accuracy", "a"), ef("vendi", "d"), ef("vendi", "a"),
           ef("coverage", "d"), ef("coverage", "a"), ef("precision", "d"), ef("precision", "a"), 100 * inter]
    names = ["div acc", "aln acc", "dVendi div", "dVendi aln", "dCov div", "dCov aln", "dPrec div", "dPrec aln", "interaction"]
    for n, g, v in zip(names, got, w):
        check(f"{f} {n}", g, v, 0.006 if abs(v) < 1 and "d" in n[:2] and n != "div acc" else 0.06)
    if f in ("sd15", "sdturbo"):
        hi = c.loc[["low/high", "high/high"], "accuracy"].mean(); lo = c.loc[["low/low", "high/low"], "accuracy"].mean()
        print(f"     p(alignment effect, {f}) = {p2(hi, lo):.2e} (paper: <0.01)")
    else:
        hi = c.loc[["low/high", "high/high"], "accuracy"].mean(); lo = c.loc[["low/low", "high/low"], "accuracy"].mean()
        print(f"     p(alignment effect, {f}) = {p2(hi, lo):.2f} (paper: not significant)")
    spread = cells[cells.family == f].groupby("cell").accuracy.agg(lambda s: s.max() - s.min()).max()
    check(f"{f} seed spread (pts) <= 0.9", 100 * spread, 0.0, 0.9)

print("--- Sec 6.2 Shapley / collinearity")
w = sh[sh.model == "within-pool"].set_index("factor")
check("within-pool R2 (%)", 100 * w.r2.iloc[0], 3.6, 0.05)
for k, v in [("C", 1.6), ("A", 1.0), ("D", 0.5), ("F", 0.5)]:
    check(f"Shapley share {k} (% of variance)", 100 * w.loc[k, "share"], v, 0.06)
check("VIF min", min(col["vif"].values()), 36, 0.5); check("VIF max", max(col["vif"].values()), 147, 0.5)
check("corr Vendi-coverage", col["pairs"]["vendi-coverage"], 0.91, 0.005)
check("corr Vendi-precision", col["pairs"]["vendi-precision"], 0.92, 0.005)

print("--- Sec 6.3 per-class gap")
for f, share, slope, r2, p in [("sd15", 65, -0.29, 0.16, 0.085), ("sdturbo", 82, -0.95, 0.58, 1e-4), ("sdxl", 58, -0.18, 0.49, 6e-4)]:
    d = pc[pc.family == f].sort_values("gap", ascending=False)
    check(f"{f} share of gap in worst two classes (%)", 100 * d.gap.head(2).sum() / d.gap.clip(lower=0).sum(), share, 1.0)
    print(f"     {f}: classes with gap<=0: {(d.gap <= 1e-9).mean() * 100:.0f}%  (paper: 55-65% overall)")
    check(f"{f} OLS slope", h3.loc[f, "slope"], slope, 0.006); check(f"{f} OLS R2", h3.loc[f, "r2"], r2, 0.006)
    print(f"     {f} p={h3.loc[f, 'p']:.4g} (paper {p})  bootstrap CI [{h3.loc[f, 'slope_lo']:.2f}, {h3.loc[f, 'slope_hi']:.2f}]")
bb = pc[pc.class_name.str.contains("Bay_breasted|Bay-breasted", regex=True)]
print(bb[["family", "gap", "familiarity"]].round(3).to_string(index=False), "(paper: SD1.5 73pt @0.68; SD-Turbo 30pt @0.995)")
print(pc[pc.class_name.str.contains("Chuck")][["family", "gap", "familiarity"]].round(3).to_string(index=False), "(paper: familiarity 0.11-0.25)")

print("--- Sec 6.4 DAS")
g = lambda f, m: t3[(t3.family == f) & (t3.method == m)].iloc[0]
gains = {}
for f in ["sd15", "sdturbo", "sdxl"]:
    for m in ["DAS-real", "DAS-text"]:
        gains[(f, m)] = 100 * (g(f, m).accuracy - g(f, "Random-N").accuracy)
for (f, m), v in zip(gains, [2.4, 3.4, 2.1, 1.2, 1.2, 1.2]):
    check(f"gain {f} {m}", gains[(f, m)], v, 0.06)
dl = [g(f, m).delta for f in ["sd15", "sdturbo", "sdxl"] for m in ["DAS-real", "DAS-text"]]
check("gap closed min (%)", 100 * min(dl), 25, 0.5); check("gap closed max (%)", 100 * max(dl), 44, 0.6)
check("sign test p (6/6 positive)", binomtest(6, 6, 0.5).pvalue, 0.03, 0.005)
for f in ["sd15", "sdturbo", "sdxl"]:
    for m in ["DAS-real", "DAS-text"]:
        pv = p2(g(f, m).accuracy, g(f, "Random-N").accuracy)
        print(f"     p({f} {m} vs Random-N) = {pv:.3f}{'  <-- significant' if pv < 0.05 else ''}")
beats_all = sum(g(f, m).accuracy > g(f, "All-mN").accuracy for f in ["sd15", "sdturbo", "sdxl"] for m in ["DAS-real", "DAS-text"])
check("DAS beats All-mN in k of 6", beats_all, 5, 0)
fb = sum(g(f, "Filter only (A_text)" if m == "DAS-text" else "Filter only (A_real)").accuracy >= g(f, m).accuracy - 1e-9
         for f in ["sd15", "sdturbo", "sdxl"] for m in ["DAS-real", "DAS-text"])
check("plain filter >= DAS in k of 6", fb, 3, 0)
check("k-center vs random SD1.5 (pts)", 100 * (g("sd15", "k-center only").accuracy - g("sd15", "Random-N").accuracy), -2.1, 0.06)
check("k-center vs random SD-Turbo (pts)", 100 * (g("sdturbo", "k-center only").accuracy - g("sdturbo", "Random-N").accuracy), -1.4, 0.06)
r5 = t3[t3.method == "Real 5-shot"].accuracy.iloc[0]
best_syn = max(g(f, m).accuracy for f in ["sd15", "sdturbo", "sdxl"] for m in t3.method.unique() if m not in ("Real 5-shot", "Real-N", "Real 5-shot + DAS-real"))
check("Real 5-shot (%)", 100 * r5, 97.8, 0.06); check("best synthetic-only (%)", 100 * best_syn, 97.3, 0.06)
hyb = [100 * (g(f, "Real 5-shot + DAS-real").accuracy - r5) for f in ["sd15", "sdturbo", "sdxl"]]
check("real5 + DAS-real vs real5: min (pts)", min(hyb), -0.2, 0.06); check("... max (pts)", max(hyb), 0.7, 0.06)
m_ = mix[mix.real_per_class.astype(str).isin(["5", "10"])]
d = []
for (f, k), x in m_.groupby(["family", "real_per_class"]):
    b = x[x.synthetic == "none"].accuracy.iloc[0]
    d += [100 * (r.accuracy - b) for r in x.itertuples() if r.synthetic != "none"]
check("mixing change min (pts)", min(d), -0.5, 0.06); check("mixing change max (pts)", max(d), 0.9, 0.06)
q = abl[abl.knob == "threshold_quantile"].groupby("family").accuracy.agg(lambda s: 100 * (s.max() - s.min()))
print(f"     tau range per generator (pts): {q.round(2).to_dict()}   (paper: 'at most 1.1')")
cen = abl[abl.knob == "first_centre"].set_index("family").accuracy
for f, v in [("sd15", 1.4), ("sdturbo", -0.2), ("sdxl", 0.2)]:
    check(f"centroid anchor - default ({f})", 100 * (cen[f] - g(f, "DAS-real").accuracy), v, 0.06)

print()
print("ALL CHECKS PASSED" if not bad else f"{len(bad)} MISMATCHES: {bad}")
sys.exit(1 if bad else 0)
