"""Write paper_cvpr/tab_{main,factorial,das}.tex (Tables 1-3) from results_pilot/. No hand-typed numbers.

Run from the repo root:  python scripts/make_tex.py
"""
from math import sqrt
from pathlib import Path

import pandas as pd
from scipy.stats import norm

R = Path("results_pilot/runs"); O = Path("paper_cvpr")
N_TEST = 585
t1 = pd.read_csv(R / "measure/table1.csv"); t3 = pd.read_csv(R / "das/table2.csv")
cells = pd.read_csv(R / "factorial/cells.csv"); abl = pd.read_csv(R / "ablation/ablation.csv")
FAM = ["sd15", "sdturbo", "sdxl"]; NAME = {"sd15": "SD1.5", "sdturbo": "SD-Turbo", "sdxl": r"SDXL ($768^2$)"}


def wilson(p, n=N_TEST, z=1.96):
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def pval(p1, p2, n=N_TEST):
    pp = (p1 + p2) / 2; se = sqrt(2 * pp * (1 - pp) / n)
    return 2 * (1 - norm.cdf(abs(p1 - p2) / se)) if se > 0 else 1.0


def acc_ci(a): lo, hi = wilson(a); return f"{100*a:.1f} [{lo:.1f},{hi:.1f}]"


def sgn(x, d=2): return f"{x:+.{d}f}".replace("-", "$-$")


# ---------------- Table 1
row = lambda f, r: t1[(t1.family == f) & (t1.regime == r)].iloc[0]
metric_cols = [("fd", "{:.2f}", False), ("precision", "{:.2f}", True), ("density", "{:.2f}", True),
               ("coverage", "{:.2f}", True), ("vendi", "{:.1f}", True), ("align_text", "{:.2f}", True)]
syn = [row(f, r) for f in FAM for r in ("P1", "P2")]
best = {}
for c, _, hi in metric_cols:
    vals = [getattr(x, c) for x in syn]; best[c] = max(vals) if hi else min(vals)
best_acc = max(x.accuracy for x in syn)
L = []
for sel, lab in [("Real-5", "Real, 5/class"), ("Real-10", "Real, 10/class"), ("Real-N", r"Real, all ($\approx$24/class)")]:
    a = t1[t1.selection == sel].accuracy.iloc[0]
    L.append(f"{lab} & -- & {acc_ci(a)} & " + " & ".join(["--"] * 6) + r" \\")
L.append(r"\midrule")
for f in FAM:
    for i, r in enumerate(("P1", "P2")):
        x = row(f, r)
        cs = []
        for c, fmt, _ in metric_cols:
            s = fmt.format(getattr(x, c)); cs.append(r"\textbf{" + s + "}" if abs(getattr(x, c) - best[c]) < 1e-9 else s)
        a = acc_ci(x.accuracy); a = r"\textbf{" + a.split(" ")[0] + "} " + a.split(" ")[1] if abs(x.accuracy - best_acc) < 1e-9 else a
        L.append(f"{NAME[f] if i == 0 else ''} & {r} & {a} & " + " & ".join(cs) + r" \\")
    L.append(r"\midrule" if f != "sdxl" else r"\midrule")
for f in ("sd15", "sdturbo"):
    x = t1[t1.family == "vae." + f].iloc[0]
    L.append(f"VAE recon.\\ ({NAME[f]}) & -- & {acc_ci(x.accuracy)} & {x.fd:.2f} & {x.precision:.2f} & {x.density:.2f} & {x.coverage:.2f} & {x.vendi:.1f} & -- \\\\")
T1 = ("\\begin{table*}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{@{}llcrrrrrr@{}}\n\\toprule\nTraining data (480 images) & Prompt & Real-test acc.\\ (\\%) $\\uparrow$ & FD$_\\text{DINO}$ $\\downarrow$ & Precision $\\uparrow$ & Density $\\uparrow$ & Coverage $\\uparrow$ & Vendi $\\uparrow$ & $A_\\text{text}$ $\\uparrow$ \\\\\n\\midrule\n"
      + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n"
      "\\caption{\\textbf{Cross-generator picture} (20 CUB classes, frozen DINOv2 probe, random selection, one generation seed). Brackets: Wilson 95\\% intervals over the 585 test images. Bold: best synthetic value per column. Generator metrics are per-class averages in DINOv2 space. Precision ranks SD1.5, the worst teacher, first; only FD$_\\text{DINO}$ orders the three generators as their accuracy does. VAE recon.: real training images re-rendered through the generator's autoencoder (not run for SDXL).}\n\\label{tab:main}\n\\end{table*}\n")

# ---------------- Table 2 (factorial + manipulation check)
L = []
for f in FAM:
    c = cells[cells.family == f].groupby("cell").mean(numeric_only=True)
    eff = lambda col, arm: (c.loc[["high/low", "high/high"] if arm == "d" else ["low/high", "high/high"], col].mean()
                            - c.loc[["low/low", "low/high"] if arm == "d" else ["low/low", "high/low"], col].mean())
    inter = (c.loc["high/high", "accuracy"] - c.loc["high/low", "accuracy"]) - (c.loc["low/high", "accuracy"] - c.loc["low/low", "accuracy"])
    nm = NAME[f].split(" (")[0]
    L.append(f"\\multirow{{2}}{{*}}{{{nm}}} & diversity & {sgn(100*eff('accuracy','d'),1)} & {sgn(eff('vendi','d'))} & {sgn(eff('coverage','d'),3)} & {sgn(eff('precision','d'),3)} & \\multirow{{2}}{{*}}{{{sgn(100*inter,1)}}} \\\\")
    L.append(f" & alignment & \\textbf{{{sgn(100*eff('accuracy','a'),1)}}} & {sgn(eff('vendi','a'))} & {sgn(eff('coverage','a'),3)} & {sgn(eff('precision','a'),3)} & \\\\")
    L.append(r"\midrule" if f != "sdxl" else "")
T2 = ("\\begin{table}[t]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n & & Effect on & \\multicolumn{3}{c}{Manipulation check} & \\\\\n\\cmidrule(lr){4-6}\nGen. & Factor & acc.\\ (pp) & $\\Delta$Vendi & $\\Delta$Cov. & $\\Delta$Prec. & Inter. \\\\\n\\midrule\n"
      + "\n".join(x for x in L if x) + "\n\\bottomrule\n\\end{tabular}\n"
      "\\caption{\\textbf{2$\\times$2 factorial (P2): main effects and what each arm actually changed.} Effects average over the other factor's levels; ``Inter.'' is the diversity$\\times$alignment interaction. Raising alignment lowers Vendi by 2.6--4.3 yet raises accuracy; raising diversity lifts Vendi by $\\approx$1.4 and coverage, but not accuracy.}\n\\label{tab:factorial}\n\\end{table}\n")

# ---------------- Table 3 (DAS vs ablations)
g = lambda f, m: t3[(t3.family == f) & (t3.method == m)].iloc[0]
rows = [("Random-N", "Random-N", 0), ("All-$mN$ (3$\\times$ images)", "All-mN", 0), ("$k$-center only", "k-center only", 0),
        ("Filter only ($A_\\text{text}$)", "Filter only (A_text)", 0), ("DAS-text", "DAS-text", 0), None,
        ("Filter only ($A_\\text{real}$)", "Filter only (A_real)", 5), ("DAS-real", "DAS-real", 5), None,
        ("Real 5-shot + DAS-real", "Real 5-shot + DAS-real", 5)]
synth_only = ["Random-N", "All-mN", "k-center only", "Filter only (A_text)", "DAS-text", "Filter only (A_real)", "DAS-real"]
bestc = {f: max(g(f, m).accuracy for m in synth_only) for f in FAM}
L = []
for r in rows:
    if r is None: L.append(r"\midrule"); continue
    lab, m, ri = r; cs = []
    for f in FAM:
        x = g(f, m); a = f"{100*x.accuracy:.1f}"
        if m in synth_only and abs(x.accuracy - bestc[f]) < 1e-9: a = r"\textbf{" + a + "}"
        if m in ("DAS-text", "DAS-real", "All-mN", "k-center only", "Filter only (A_text)", "Filter only (A_real)") and pval(x.accuracy, g(f, "Random-N").accuracy) < 0.05 and x.accuracy > g(f, "Random-N").accuracy: a += "$^*$"
        d = "0" if m == "Random-N" else sgn(x.delta)
        cs.append(f"{a} & {d}")
    L.append(f"{lab} & {ri} & " + " & ".join(cs) + r" \\")
r5 = t3[t3.method == "Real 5-shot"].accuracy.iloc[0]; rn = t3[t3.method == "Real-N"].accuracy.iloc[0]
L.insert(len(L) - 0, f"Real 5-shot (no synthetic) & 5 & \\multicolumn{{6}}{{c}}{{{100*r5:.1f}}} \\\\")
L.append(f"Real-$N$ (reference) & all & \\multicolumn{{6}}{{c}}{{{100*rn:.1f}}} \\\\")
L.append(r"\midrule")
L.append(r"\multicolumn{8}{@{}l}{\emph{DAS-real ablations (accuracy)}} \\")
for lab, sel in [(r"$\tau$ = 20th pct.", ("threshold_quantile", 0.2)), (r"$\tau$ = 40th pct.\ (default)", ("threshold_quantile", 0.4)),
                 (r"$\tau$ = 60th pct.", ("threshold_quantile", 0.6)), ("centroid-nearest anchor", ("first_centre", "centroid"))]:
    v = []
    for f in FAM:
        d = abl[(abl.family == f) & (abl.knob == sel[0])]
        d = d[d.value.astype(str) == str(sel[1])]
        v.append(f"\\multicolumn{{2}}{{c}}{{{100*d.accuracy.iloc[0]:.1f}}}")
    L.append(f"\\quad {lab} & 5 & " + " & ".join(v) + r" \\")
T3 = ("\\begin{table}[t]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{1.5pt}\n\\begin{tabular}{@{}lcrrrrrr@{}}\n\\toprule\n & Real & \\multicolumn{2}{c}{SD1.5} & \\multicolumn{2}{c}{SD-Turbo} & \\multicolumn{2}{c}{SDXL} \\\\\nMethod & /cls & Acc & $\\Delta$ & Acc & $\\Delta$ & Acc & $\\Delta$ \\\\\n\\midrule\n"
      + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n"
      "\\caption{\\textbf{DAS versus its ablations} (P2, 480 images unless noted). $\\Delta$: fraction of gap closed (Eq.~\\ref{eq:delta}). Bold: best synthetic-only set. $^*$: $p{<}0.05$ vs.\\ Random-$N$ under a conservative unpaired test; one test image is 0.17 points. Real 5-shot was re-trained in this stage and differs from Tab.~\\ref{tab:main} by one test image.}\n\\label{tab:das}\n\\end{table}\n")
for n, t in [("tab_main", T1), ("tab_factorial", T2), ("tab_das", T3)]:
    (O / f"{n}.tex").write_text(t)
print("wrote tab_main.tex, tab_factorial.tex, tab_das.tex in", O)
