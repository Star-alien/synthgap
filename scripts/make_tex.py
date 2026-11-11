"""Write paper_cvpr/gen_macros.tex and gen_tables.tex from results_pilot/ (no hand-typed numbers)."""
import json
import pandas as pd
from pathlib import Path
from scipy.stats import kendalltau

R = Path("results_pilot"); O = Path("paper_cvpr")
t1 = pd.read_csv(R/"runs/measure/table1.csv"); t2 = pd.read_csv(R/"runs/das/table2.csv")
eff = pd.read_csv(R/"runs/factorial/effects.csv"); h3 = pd.read_csv(R/"runs/h3/regression.csv")
sh = pd.read_csv(R/"runs/subsets/shapley.csv"); col = json.load(open(R/"runs/subsets/collinearity.json"))
abl = pd.read_csv(R/"runs/ablation/ablation.csv"); mix = pd.read_csv(R/"runs/mixing/mixing.csv")
pc = pd.read_csv(R/"runs/h3/per_class.csv")
G = {"sd15": "A", "sdturbo": "B", "sdxl": "C"}            # macro letters
NAME = {"sd15": "SD1.5", "sdturbo": "SD-Turbo", "sdxl": "SDXL"}
RG = {"P1": "a", "P2": "b"}
f1 = lambda x: f"{100*x:.1f}"; f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"
M = []
def mac(name, val): M.append(f"\\newcommand{{\\{name}}}{{{val}}}")
a = lambda g, r: t1[(t1.family == g) & (t1.regime == r)].iloc[0]
for g, L in G.items():
    for r, l in RG.items():
        x = a(g, r)
        mac(f"acc{L}{l}", f1(x.accuracy)); mac(f"vendi{L}{l}", f"{x.vendi:.1f}"); mac(f"cov{L}{l}", f2(x.coverage))
        mac(f"prec{L}{l}", f2(x.precision)); mac(f"rec{L}{l}", f3(x.recall)); mac(f"fd{L}{l}", f2(x.fd)); mac(f"atext{L}{l}", f2(x.align_text)); mac(f"areal{L}{l}", f3(x.align_real))
    e = lambda q: eff[(eff.family == g) & (eff.quantity == q)].iloc[0]
    for q, n in [("diversity", "Div"), ("alignment", "Aln"), ("interaction", "Int")]:
        mac(f"eff{n}{L}", f"{100*e(q)['mean']:+.1f}")
        mac(f"eff{n}Lo{L}", f"{100*e(q).lo:+.1f}"); mac(f"eff{n}Hi{L}", f"{100*e(q).hi:+.1f}")
    r_ = h3[h3.family == g].iloc[0]
    mac(f"hslope{L}", f2(r_.slope)); mac(f"hlo{L}", f2(r_.slope_lo)); mac(f"hhi{L}", f2(r_.slope_hi)); mac(f"hr{L}", f2(r_.r2)); mac(f"hp{L}", f"{r_.p:.4f}")
    low = pc[pc.family == g].sort_values("familiarity").iloc[0]
    mac(f"lowgap{L}", f1(low.gap)); mac(f"lowfam{L}", f2(low.familiarity))
for g, L in G.items():
    r = t1[t1.family == f"vae.{g}"]
    if len(r): mac(f"vae{L}", f1(r.accuracy.iloc[0]))
for sel, n in [("Real-5", "RealFive"), ("Real-10", "RealTen"), ("Real-N", "RealN")]:
    mac(n, f1(t1[t1.selection == sel].accuracy.iloc[0]))
mac("RealFiveTwo", f1(t2[t2.method == "Real 5-shot"].accuracy.iloc[0]))
get = lambda g, m: t2[(t2.family == g) & (t2.method == m)].iloc[0]
for g, L in G.items():
    for m, n in [("Random-N", "Rand"), ("All-mN", "All"), ("k-center only", "Kc"), ("Filter only (A_text)", "FiltT"), ("DAS-text", "DasT"), ("Filter only (A_real)", "FiltR"), ("DAS-real", "DasR"), ("Real 5-shot + DAS-real", "Hyb")]:
        x = get(g, m); mac(f"twacc{n}{L}", f1(x.accuracy)); mac(f"twd{n}{L}", f2(x.delta))
    mac(f"gainDasR{L}", f"{100*(get(g,'DAS-real').accuracy-get(g,'Random-N').accuracy):+.1f}")
    mac(f"gainDasT{L}", f"{100*(get(g,'DAS-text').accuracy-get(g,'Random-N').accuracy):+.1f}")
ds = [get(g, m).delta for g in G for m in ["DAS-real", "DAS-text"]]
mac("dasMin", f"{100*min(ds):.0f}"); mac("dasMax", f"{100*max(ds):.0f}")
w = sh[sh.model == "within-pool"].set_index("factor")
for k in "DCAF": mac(f"shw{k}", f"{100*w.loc[k,'share']:.1f}")
mac("withinRtwo", f1(w.r2.iloc[0])); mac("nsubsets", str(int(w.n.iloc[0]) // 3))
vif = col["vif"]; mac("vifMin", f"{min(vif.values()):.0f}"); mac("vifMax", f"{max(vif.values()):.0f}")
p_ = col["pairs"]; mac("corrVC", f2(p_["vendi-coverage"])); mac("corrVP", f2(p_["vendi-precision"])); mac("corrMax", f2(col["max_abs_pair"]))
acc = lambda r: [a(g, r).accuracy for g in ["sd15", "sdxl", "sdturbo"]]
mac("tauPa", f2(kendalltau([1, 2, 3], acc("P1")).statistic)); mac("tauPb", f2(kendalltau([1, 2, 3], acc("P2")).statistic))
# mixing: max abs change from adding synthetic to real
d = []
for (g, k), x in mix[mix.real_per_class.astype(str).isin(["5", "10"])].groupby(["family", "real_per_class"]):
    base = x[x.synthetic == "none"].accuracy.iloc[0]; d += [100*(r.accuracy-base) for r in x.itertuples() if r.synthetic != "none"]
mac("mixMin", f"{min(d):+.1f}"); mac("mixMax", f"{max(d):+.1f}")
for g, L in G.items():
    q = abl[(abl.family == g) & (abl.knob == "threshold_quantile")].accuracy
    mac(f"ablLo{L}", f1(q.min())); mac(f"ablHi{L}", f1(q.max()))
(O/"gen_macros.tex").write_text("\n".join(M) + "\n")

T = []
# Table 1 (wide)
rows = []
def r1(lab, reg, x):
    return f"{lab} & {reg} & {f1(x.accuracy)} & {x.vendi:.1f} & {x.coverage:.2f} & {x.precision:.2f} & {x.recall:.3f} & {x.fd:.2f} & {x.align_real:.3f} \\\\"
rows += [f"Real, 5/class & -- & {f1(t1[t1.selection=='Real-5'].accuracy.iloc[0])} & -- & -- & -- & -- & -- & -- \\\\",
         f"Real, 10/class & -- & {f1(t1[t1.selection=='Real-10'].accuracy.iloc[0])} & -- & -- & -- & -- & -- & -- \\\\",
         f"Real, all ($N{{=}}24$) & -- & {f1(t1[t1.selection=='Real-N'].accuracy.iloc[0])} & -- & -- & -- & -- & -- & -- \\\\", "\\midrule"]
for g in ["sd15", "sdturbo", "sdxl"]:
    for r in ["P1", "P2"]:
        rows.append(r1(NAME[g] + (r" ($768^2$)" if g == "sdxl" else ""), r, a(g, r)))
    rows.append("\\midrule") if g != "sdxl" else None
rows.append("\\midrule")
for g in ["sd15", "sdturbo"]:
    x = t1[t1.family == f"vae.{g}"].iloc[0]
    rows.append(f"VAE ceiling ({NAME[g]}) & -- & {f1(x.accuracy)} & {x.vendi:.1f} & {x.coverage:.2f} & {x.precision:.2f} & {x.recall:.2f} & {x.fd:.2f} & -- \\\\")
T.append("\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{llrrrrrrr}\n\\toprule\nTraining data & Prompt & Acc $\\uparrow$ & Vendi $\\uparrow$ & Cov. $\\uparrow$ & Prec. $\\uparrow$ & Recall $\\uparrow$ & FD$_\\text{DINO}$ $\\downarrow$ & $A_\\text{real}$ $\\uparrow$ \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
         "\\caption{\\textbf{Main results (CUB-200-2011, 20 classes).} Real-test accuracy (\\%) of a frozen DINOv2 linear probe trained on 480 random images (24 per class), with set-level measures in DINOv2 space (alignment in CLIP space, Sec.~\\ref{sec:measures}). One generation seed; one test image is 0.17 points. VAE ceiling: real images re-rendered through the generator's autoencoder (not run for SDXL). Aircraft, SD3.5, FLUX and P3 rows of the full design were not run.}\n\\label{tab:main}\n\\end{table*}\n")
# Table factorial
rows = []
for g in ["sd15", "sdturbo", "sdxl"]:
    L = G[g]
    rows.append(f"{NAME[g]} & \\fmt{{\\effDiv{L}}} [\\effDivLo{L}, \\effDivHi{L}] & \\effAln{L} [\\effAlnLo{L}, \\effAlnHi{L}] & \\effInt{L} [\\effIntLo{L}, \\effIntHi{L}] \\\\".replace("\\fmt{\\effDiv" + L + "}", "\\effDiv" + L))
T.append("\\begin{table}[t]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\nGenerator & Diversity & Alignment & Interaction \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
         "\\caption{\\textbf{2$\\times$2 factorial (P2), effects in accuracy points} with 95\\% intervals over selection and training seeds (not over generation seeds).}\n\\label{tab:factorial}\n\\end{table}\n")
# Table 2 DAS
rows = []
labels = [("Random-N", "Rand", 0), ("All-mN (3$\\times$ data)", "All", 0), ("$k$-center only", "Kc", 0), ("Filter only ($A_\\text{text}$)", "FiltT", 0), ("DAS-text", "DasT", 0),
          ("Filter only ($A_\\text{real}$)", "FiltR", 5), ("DAS-real", "DasR", 5), ("Real 5-shot + DAS-real", "Hyb", 5)]
for lab, n, ri in labels:
    rows.append(f"{lab} & {ri} & " + " & ".join(f"\\twacc{n}{L} & \\twd{n}{L}" for L in "ABC") + " \\\\")
rows.insert(len(rows), f"Real 5-shot & 5 & \\multicolumn{{6}}{{c}}{{\\RealFiveTwo}} \\\\")
rows.append("Real-$N$ & all & \\multicolumn{6}{c}{\\RealN} \\\\")
T.append("\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{@{}lccrrrrrr@{}}\n\\toprule\n& Real & \\multicolumn{2}{c}{SD1.5} & \\multicolumn{2}{c}{SD-Turbo} & \\multicolumn{2}{c}{SDXL} \\\\\nMethod & imgs & Acc & $\\Delta$ & Acc & $\\Delta$ & Acc & $\\Delta$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
         "\\caption{\\textbf{DAS versus baselines (CUB, 20 classes, P2).} $\\Delta$: fraction of the gap closed (Eq.~\\ref{eq:delta}). ``Real imgs'': real images per class used anywhere in the pipeline. One test image is 0.17 points.}\n\\label{tab:das}\n\\end{table*}\n")
(O/"gen_tables.tex").write_text("\n".join(T))
print(len(M), "macros;", len(T), "tables")
