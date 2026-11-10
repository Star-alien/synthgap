"""Build paper/paper.pdf (+ paper.md) from results_pilot/. Every table is read from the CSVs."""
import pandas as pd
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

R = Path("results_pilot"); OUT = Path("paper"); OUT.mkdir(exist_ok=True)
t1 = pd.read_csv(R/"runs/measure/table1.csv"); t2 = pd.read_csv(R/"runs/das/table2.csv")
eff = pd.read_csv(R/"runs/factorial/effects.csv"); h3 = pd.read_csv(R/"runs/h3/regression.csv")
abl = pd.read_csv(R/"runs/ablation/ablation.csv")
NAME = {"sd15": "SD1.5", "sdturbo": "SD-Turbo", "sdxl": "SDXL (768px)"}
GENS = ["sd15", "sdturbo", "sdxl"]
pct = lambda x: f"{100*x:.1f}"

ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontName="Times-Roman", fontSize=10.5, leading=13.5, spaceAfter=6)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Times-Bold", fontSize=19, leading=23, alignment=1, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=13, spaceBefore=10, spaceAfter=4)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontName="Times-BoldItalic", fontSize=11, spaceBefore=6, spaceAfter=3)
CAP = ParagraphStyle("CAP", parent=B, fontSize=9, leading=11.5, spaceAfter=10)
SUB = ParagraphStyle("SUB", parent=B, alignment=1, fontSize=10.5)
BOX = ParagraphStyle("BOX", parent=B, fontSize=9.5, leading=12.5, backColor=colors.HexColor("#F3F3F3"), borderPadding=6, spaceBefore=4, spaceAfter=10)

story, md = [], []
def h1(t): story.append(Paragraph(t, H1)); md.append(f"# {t}\n")
def h2(t): story.append(Paragraph(t, H2)); md.append(f"\n## {t}\n")
def h3_(t): story.append(Paragraph(t, H3)); md.append(f"\n### {t}\n")
def p(t, style=B):
    story.append(Paragraph(t, style)); md.append(t.replace("<b>", "**").replace("</b>", "**").replace("<i>", "*").replace("</i>", "*").replace("<br/>", "  \n") + "\n")
def table(rows, widths, caption, head=1):
    data = [[Paragraph(str(c), ParagraphStyle("c", parent=B, fontSize=8.6, leading=10.4, spaceAfter=0)) for c in r] for r in rows]
    t = Table(data, colWidths=[w*inch for w in widths], repeatRows=head)
    t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,head-1), colors.HexColor("#E6E6E6")), ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
                           ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("TOPPADDING", (0,0), (-1,-1), 2), ("BOTTOMPADDING", (0,0), (-1,-1), 2)]))
    story.append(KeepTogether([t, Spacer(1, 3), Paragraph(caption, CAP)]))
    md.append("| " + " | ".join(map(str, rows[0])) + " |\n|" + "---|"*len(rows[0]) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows[1:]) + f"\n\n*{caption}*\n")
def fig(path, w, caption):
    from PIL import Image as PI
    iw, ih = PI.open(path).size
    story.append(KeepTogether([Image(str(path), width=w*inch, height=w*inch*ih/iw), Paragraph(caption, CAP)]))
    md.append(f"![{caption}](../{path})\n")

# ---------------------------------------------------------------- title / abstract
h1("Do Better Generators Make Worse Teachers?<br/>A Compute-Limited Pilot on 20 CUB Classes")
p("Authors: [to be added]", SUB)
p("<b>Status: pilot study / working draft.</b> This is a small, honestly scoped study run on free GPUs. It is not a "
  "full-scale replication of the original design, and it should not be submitted as one without the extensions listed in Section 6.", BOX)

h2("Abstract")
real_n = t1[t1.selection == "Real-N"].accuracy.iloc[0]; real5 = t1[t1.selection == "Real-5"].accuracy.iloc[0]
p("Text-to-image models are widely used to create training data, and newer generators produce images that look better. "
  "Do they also teach better? We ask which measurable property of a synthetic training set (diversity, coverage, label "
  "alignment, fidelity) explains real-test accuracy. We hold the prompt and image pool fixed and build equal-sized training "
  "sets that differ in one property. Because of a hard compute limit (free Kaggle T4 GPUs), this pilot covers only 20 evenly "
  f"spaced classes of CUB-200-2011, three open generators (SD1.5, SD-Turbo, SDXL), two prompt styles and one generation seed. "
  f"A linear probe on frozen DINOv2 features trained on 480 synthetic images per generator reaches {pct(t1[(t1.family=='sd15')&(t1.regime=='P1')].accuracy.iloc[0])}% (SD1.5), "
  f"{pct(t1[(t1.family=='sdturbo')&(t1.regime=='P1')].accuracy.iloc[0])}% (SD-Turbo) and {pct(t1[(t1.family=='sdxl')&(t1.regime=='P1')].accuracy.iloc[0])}% (SDXL) on real test images, against {pct(real_n)}% for real data. "
  "<b>In this pilot the newer, larger generator is the better teacher; we do not observe the hypothesised inversion.</b> "
  "Within a generator, label alignment is the property that matters (+12.2 points for SD1.5, +1.3 for SDXL), while diversity has almost no effect. "
  "A training-free selection rule that filters by alignment and then spreads samples out (DAS) improves over random selection by 1.2 to 3.4 points, "
  "but a plain alignment filter does about as well. Most differences are only a few test images; we report them with that caveat.")

# ---------------------------------------------------------------- intro
h2("1. Introduction")
p("Synthetic images from text-to-image diffusion models are a cheap way to build classifiers when real data is scarce. A common assumption is "
  "that a generator which looks more realistic is a better teacher. This assumption is rarely tested directly, because generators differ in many "
  "ways at once. We try to isolate what matters by <i>intervening</i>: for one generator and one prompt, we build training sets of the same size "
  "that differ only in one measured property of the images, then compare the accuracy of a classifier trained on each.")
p("The original design of this project covers four generators, two datasets and three generation seeds (about 600,000 images, around 600 A100 GPU-hours). "
  "We could not run that. This report describes what a compute-limited version found, and is explicit about what that version can and cannot say.")
p("<b>Contributions.</b> (i) An open implementation of the full pipeline (generation, measurements, factorial and subset designs, selection rules), run end to end on real data. "
  "(ii) A pilot-scale answer to the title question on three generators. (iii) Evidence that, within a generator, label alignment dominates diversity as a lever, and that the four properties are too collinear to be separated by a regression. "
  "(iv) A clear list of what must be scaled up before any strong claim is made.")

# ---------------------------------------------------------------- setup
h2("2. Setup")
h3_("2.1 Data")
p("CUB-200-2011 (200 bird species). To fit our budget we use 20 classes chosen deterministically at evenly spaced class indices (no hand-picking). "
  "Each class has 24 real training images, 5 further training images per class are held out only to score label alignment (the 'alignment split'), "
  "and 585 real test images in total are used for evaluation. One test image is worth 0.17 percentage points.")
h3_("2.2 Generators and prompts")
p("<b>SD1.5</b> (512px, 25 steps, guidance 2.0), <b>SD-Turbo</b> (512px, 4 steps, no guidance; an adversarially distilled model, so a faster model, not simply a newer SD1.5), "
  "and <b>SDXL</b> (rendered at 768px rather than its native 1024px, 25 steps, guidance 2.0; this was needed to fit a 15 GB T4 GPU and the session time limit). "
  "Resolution, step count and guidance therefore differ across generators, and they are confounded with the generator itself. "
  "Prompt <b>P1</b> is one fixed template (\"a photo of a {class}, a type of bird\"). Prompt <b>P2</b> uses templates that vary pose, background, lighting, season and viewpoint. "
  "Each pool has 72 images per class (3 times the 24 real images), so 1,440 images, and equal-size training sets use 480 images (24 per class).")
h3_("2.3 Measurements")
p("Features come from DINOv2-base (used for the learner and all reported metrics) and CLIP ViT-L/14 (used only for selection and alignment, so selection cannot optimise the space it is scored in). "
  "Per class we compute Vendi score (diversity), k-NN coverage, improved precision/recall and density, and Frechet distance (FD_DINO). "
  "Label alignment is scored two ways: A_real, the probability of the prompted class under a nearest-class-mean classifier built from the 5-shot alignment split, and A_text, CLIP zero-shot "
  "probability from class names alone.")
h3_("2.4 Learner")
p("A frozen-feature linear (logistic regression) probe on DINOv2 features. Weight decay is chosen on a held-out 20% of the <i>synthetic</i> data, so no real validation data is used. "
  "The from-scratch ResNet control in the original design was not run.")
h3_("2.5 Interventional designs and DAS")
p("<b>2x2 factorial:</b> from the P2 pool we build equal-size sets that are low or high on diversity and low or high on alignment, and compare cells. "
  "<b>Random-subset design:</b> 150 balanced random subsets per generator (reduced from 500 for time), regressed on the four properties with a variance decomposition. "
  "<b>DAS</b> (diversity-matched, alignment-filtered sampling) first keeps the best-aligned images (top 60%, quantile 0.4) and then picks a spread-out subset with greedy k-center in CLIP space. "
  "Baselines are random selection, all 1,440 images (3x data), k-center only, alignment filter only, and real 5-shot data.")

# ---------------------------------------------------------------- results
h2("3. Results")
h3_("3.1 Which generator teaches best?")
rows = [["Generator", "Prompt", "Acc. (%)", "Vendi", "Coverage", "Precision", "Recall", "FD_DINO", "A_text"]]
for g in GENS:
    for rg in ["P1", "P2"]:
        r = t1[(t1.family == g) & (t1.regime == rg)].iloc[0]
        rows.append([NAME[g], rg, pct(r.accuracy), f"{r.vendi:.1f}", f"{r.coverage:.2f}", f"{r.precision:.2f}", f"{r.recall:.3f}", f"{r.fd:.2f}", f"{r.align_text:.2f}"])
for g in ["sd15", "sdturbo"]:
    r = t1[t1.family == f"vae.{g}"].iloc[0]; rows.append([f"VAE ref. ({NAME[g]})", "-", pct(r.accuracy), f"{r.vendi:.1f}", f"{r.coverage:.2f}", f"{r.precision:.2f}", f"{r.recall:.2f}", f"{r.fd:.2f}", "-"])
for sel, lab in [("Real-5", "Real, 5 per class"), ("Real-10", "Real, 10 per class"), ("Real-N", "Real, all 24 per class")]:
    rows.append([lab, "-", pct(t1[t1.selection == sel].accuracy.iloc[0]), "", "", "", "", "", ""])
table(rows, [1.4, .6, .6, .5, .7, .7, .6, .8, .6],
      "Table 1. Accuracy on real CUB test images of a probe trained on 480 images (random selection), plus fidelity/diversity measures. "
      "VAE ref. = real training images re-rendered through the generator's autoencoder (SDXL's was not run for memory reasons). One generation seed; one test image = 0.17 points.")
p("Under both prompts the order is SDXL, then SD-Turbo, then SD1.5 (P1: 96.2, 91.5, 90.9). The newest and largest generator is the best teacher here, so the hypothesised inversion "
  "(newer generators teach worse) is <b>not supported in this pilot</b>. SD-Turbo is not better than SD1.5 under P1 (91.5 vs 90.9 is about 3 test images), but is under P2 (93.8 vs 90.6). "
  "Re-rendering real images through the autoencoder gives 98.3 to 98.5%, essentially the real-data accuracy, so autoencoder artefacts are not what limits synthetic data. "
  "The ranking by FD_DINO (SDXL 0.74 best, SD-Turbo 0.81, SD1.5 0.93) matches the accuracy ranking, while improved precision does not: SD1.5 has the <i>highest</i> precision (0.62) but is the "
  "<i>worst</i> teacher. This is the one place where 'looks better, teaches worse' appears, and it rests on one measure at a small per-class sample size (precision and recall are computed over only 24 images per class and are noisy).")
fig(R/"figures/fig1_cub.png", 6.0, "Figure 1. Real-test accuracy (black) against precision (left) and FD_DINO (right, inverted), P1 prompts. Dotted grey: autoencoder reference. Green: real data.")

h3_("3.2 Alignment matters; diversity barely does")
e = lambda g, q: eff[(eff.family == g) & (eff.quantity == q)].iloc[0]
rows = [["Generator", "Diversity effect", "Alignment effect", "Interaction", "low div / low align", "high div / high align"]]
for g in GENS:
    rows.append([NAME[g]] + [f"{100*e(g,q)['mean']:+.1f} [{100*e(g,q).lo:+.1f}, {100*e(g,q).hi:+.1f}]" for q in ["diversity", "alignment", "interaction"]] +
                [f"{100*e(g,'cell:low/low')['mean']:.1f}", f"{100*e(g,'cell:high/high')['mean']:.1f}"])
table(rows, [1.15, 1.25, 1.25, 1.2, 1.0, 1.05],
      "Table 2. 2x2 factorial on the P2 pool, in accuracy points (95% interval over selection and training seeds, not over generation seeds). Cells use extreme quantile subsets, so they are not comparable to Table 1's random selection.")
p("Raising label alignment helps a lot for the weakest generator (+12.2 points for SD1.5), less for SD-Turbo (+3.9), and little for SDXL (+1.3). Raising diversity does nothing or slightly hurts (+0.1, -0.7, -0.5). "
  "A plausible reading is that a weaker generator often draws the wrong or an unrecognisable bird for a prompt, so filtering out those images helps a great deal, while SDXL already mostly draws the right bird. "
  "This is a hypothesis consistent with the data, not something we tested directly. The intervals are very narrow, but they only reflect selection and training randomness; with one pool per generator, the "
  "real uncertainty is larger than they suggest.")
fig(R/"figures/fig2a_cub.png", 5.8, "Figure 2. Factorial cells (diversity x alignment) per generator, P2 pool.")

h3_("3.3 The four properties cannot be separated here")
p("Across 150 random subsets per generator (450 in total), a regression on the four properties explains only <b>3.6%</b> of the variance in accuracy, and each property's share is at most 1.6%. "
  "The properties are strongly collinear: Vendi correlates 0.91 with coverage and 0.92 with precision, and variance inflation factors range from 36 to 147 (values above 10 are usually taken as serious). "
  "A between-generator version of the decomposition has only 3 data points and 4 predictors, so it is not identified (its R-squared of 1.0 is a mathematical artefact); "
  "<b>we do not interpret it</b>, and we do not report the 'share of accuracy explained by each property' that the original draft aimed for.")

h3_("3.4 DAS and baselines")
rows = [["Method", "SD1.5", "SD-Turbo", "SDXL"]]
order = ["Random-N", "All-mN", "k-center only", "Filter only (A_text)", "DAS-text", "Filter only (A_real)", "DAS-real", "Real 5-shot + DAS-real"]
get = lambda g, m: t2[(t2.family == g) & (t2.method == m)].iloc[0]
for m in order:
    rows.append([m + (" (3x data)" if m == "All-mN" else "")] + [f"{pct(get(g,m).accuracy)}" for g in GENS])
rows.append(["Real 5-shot (100 images)"] + [pct(t2[t2.method == "Real 5-shot"].accuracy.iloc[0])] * 3)
rows.append(["Real, all (480 images)"] + [pct(t2[t2.method == "Real-N"].accuracy.iloc[0])] * 3)
table(rows, [2.4, 1.3, 1.3, 1.3],
      "Table 3. Real-test accuracy (%) for each selection method, 480 synthetic images unless stated, P2 pool. Real 5-shot appears as 97.9 in Table 1 and 97.8 here; they come from separate runs of the probe (one test image = 0.17 points).")
p("DAS-real improves over random selection by +2.4 (SD1.5), +2.1 (SD-Turbo) and +1.2 points (SDXL); DAS-text by +3.4, +1.2 and +1.2. It beats using all 1,440 images (three times the data) in 5 of 6 cases. "
  "But filtering by alignment alone is about as good: it is equal to or better than DAS in 3 of 6 cases (SD-Turbo with A_text, SDXL with both scorers), so <b>we cannot credit DAS's diversity step with an improvement</b>. "
  "The diversity-only baseline (k-center) is below random for two generators. Adding DAS-selected synthetic images to the 5-shot real set changes accuracy by -0.2 to +0.7 points relative to real 5-shot alone, which is within a few test images. "
  "Real data with only 5 images per class (97.8%) still beats every synthetic-only set.")

h3_("3.5 Robustness checks and per-class effects")
p("<b>Ablations.</b> Changing the alignment-filter threshold (keep top 80%, 60%, 40%) moves DAS accuracy by at most about 1 point for SD1.5 (93.0 to 93.3), SD-Turbo (95.0 to 96.1) and SDXL (96.4 to 97.4); "
  "a different first-centre rule gives similar accuracy. <b>Mixing with real data.</b> Once a class has 5 or 10 real images, adding synthetic images changes accuracy by between -0.5 and +0.9 points "
  "(up to about 5 test images), with no consistent direction. We cannot say where synthetic data stops helping, because real-only accuracy is already near ceiling (about 98%).")
rows = [["Generator", "Slope of gap on familiarity", "95% bootstrap CI", "R-squared", "p-value"]]
for g in GENS:
    r = h3[h3.family == g].iloc[0]; rows.append([NAME[g], f"{r.slope:.2f}", f"[{r.slope_lo:.2f}, {r.slope_hi:.2f}]", f"{r.r2:.2f}", f"{r.p:.4f}"])
table(rows, [1.5, 1.6, 1.5, 0.9, 0.9], "Table 4. Per-class gap (real-data accuracy minus synthetic-data accuracy) regressed on class familiarity (mean CLIP zero-shot probability of the class name on P1 images); 20 classes.")
p("In all three generators, classes the generator 'knows' poorly have larger accuracy gaps (negative slope). The relationship is strong for SD-Turbo (R-squared 0.58) and SDXL (0.49) and weak for SD1.5 (0.16, p = 0.085). "
  "With 20 classes and many classes at 100% accuracy, this is suggestive evidence only.")
fig(R/"figures/fig3_cub.png", 6.0, "Figure 3. Accuracy when adding synthetic images to different numbers of real images per class.")

# ---------------------------------------------------------------- discussion / limits
h2("4. Discussion")
p("The simple story 'better-looking generators teach worse' is not what we see: the best generator by FD, by accuracy and by label alignment is the same one (SDXL). "
  "What we do see is (a) a precision measure that disagrees with the others, (b) a large benefit from alignment filtering for the weak generator that shrinks as the generator improves, and "
  "(c) no sign that diversity, as measured here, is the bottleneck. Practically, for a weak generator, checking that each image shows the prompted class (with CLIP, for free) is the most useful thing to do before training.")

h2("5. Limitations (please read)")
for t in [
    "<b>Scale.</b> 20 classes of one dataset, three generators, one generation seed per pool, 585 test images. Intervals reflect selection/training randomness only. One test image is 0.17 points, so most differences in Table 3 are 1 to 6 images.",
    "<b>Ceiling effect.</b> The 20 selected classes are easy: real data reaches 98.6%. This compresses differences and may hide effects visible on all 200 classes.",
    "<b>Confounded generators.</b> SD1.5, SD-Turbo and SDXL differ in resolution (512 / 512 / 768), steps (25 / 4 / 25) and guidance, not just in model quality. SDXL was not run at native resolution. SD-Turbo is a distilled model, not a 'next version' of SD1.5.",
    "<b>Old generators only.</b> SD3.5 and FLUX, which the original question is really about, were not run (they do not fit on free T4 GPUs).",
    "<b>Reduced designs.</b> 150 random subsets instead of 500; the from-scratch ResNet control, the second dataset (FGVC-Aircraft), the oracle caption prompt (P3) and additional generation seeds were not run. SDXL's autoencoder reference was not run.",
    "<b>Collinearity.</b> The four properties are too correlated to attribute accuracy to each; the between-generator decomposition has only three points.",
    "<b>Metric noise.</b> Per-class precision/recall use 24 images per class and are noisy; SD-Turbo's P1 precision (0.03) and coverage (0.03) look unusually low and should be checked before being relied on."]:
    p("&bull; " + t)

h2("6. What is needed for a full study")
p("A full version needs all 200 CUB classes and FGVC-Aircraft, three generation seeds, at least one modern generator (SD3.5 or FLUX) at native resolution, and a consistent step/guidance budget. "
  "The code in the accompanying repository does this (about 200 A100 GPU-hours for one seed; see docs/COMPUTE_PLAN.md). We expect it to need a university or cloud allocation, which we did not have.")

h2("7. Reproducibility and responsible use")
p("Code, configurations and all result tables: https://github.com/Star-alien/synthgap (a venue with double-blind review would need an anonymised copy). The pilot used the config files "
  "configs/exp/kaggle20.yaml and final20.yaml. Generation ran on free Kaggle Tesla T4 GPUs; analysis ran on a laptop. Generators are used under their own model licences; CUB-200-2011 is used under its research terms. No human-subject data were collected. All numbers in this report are produced by the code from the stored images; none were edited by hand.")

def foot(c, d):
    c.saveState(); c.setFont("Times-Roman", 8); c.drawCentredString(letter[0]/2, 0.5*inch, f"Pilot study draft - page {d.page}"); c.restoreState()
doc = SimpleDocTemplate(str(OUT/"paper.pdf"), pagesize=letter, leftMargin=0.9*inch, rightMargin=0.9*inch, topMargin=0.8*inch, bottomMargin=0.8*inch,
                        title="Do Better Generators Make Worse Teachers? A Compute-Limited Pilot", author="")
doc.build(story, onFirstPage=foot, onLaterPages=foot)
(OUT/"paper.md").write_text("\n".join(md))
print("built", OUT/"paper.pdf")
