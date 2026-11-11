# CVPR 2027 submission (review format)

**Title:** What Makes Synthetic Training Data Work? A Pilot Interventional Study on Fine-Grained Classification

| File | What it is |
|---|---|
| `main.tex` / `main.pdf` | LaTeX source and its build (official CVPR 2027 author kit: `cvpr.sty`, `ieeenat_fullname.bst`) |
| `synthgap_cvpr2027_REVIEW_anonymous.pdf` | The anonymous review copy this source was rebuilt from (same content) |
| `tab_main.tex`, `tab_factorial.tex`, `tab_das.tex` | Tables 1-3, **generated from `results_pilot/`** by `python scripts/make_tex.py` |
| `figs/fig1-6.pdf` (+ `.png` previews) | Figures 1-6 |
| `refs.bib` | Bibliography (the 40 cited entries are used; a few extra are unused) |

## Build
```bash
tectonic main.tex        # or upload this folder to Overleaf and compile main.tex
```

## Check the numbers
```bash
python scripts/verify_paper_numbers.py   # recomputes every statistic quoted in the text from results_pilot/
python scripts/make_tex.py               # regenerates Tables 1-3
```
`verify_paper_numbers.py` recomputes the Wilson intervals, two-proportion and sign tests, factorial effects and
manipulation checks, Shapley shares, collinearity, per-class gap shares, H3 regressions, DAS gains/ablations and mixing
ranges, and exits non-zero on any mismatch. The text numbers are typed in `main.tex`; the script is what ties them to the data.

## Figures: provenance
Figures 1-6 in `figs/` were cut (as vector graphics; Fig. 6 contains raster photos) out of the review PDF.
The generating plotting code for these specific versions is **not** in this repository, but every plotted value
comes from `results_pilot/runs/*.csv` (older, differently styled versions of some plots are in `results_pilot/figures/`).
If you need the figures regenerated from code before camera-ready, they should be re-plotted from those CSVs.

## Before submitting (authors)
- Confirm or rewrite the **LLM usage statement** (bracketed text) and name the LLM that wrote the P2 templates.
- Verify every citation against its original publication.
- This is a **pilot** (20 CUB classes, 3 generators, 1 generation seed); Sec. 7 lists what was not run.
- The review copy is anonymous: do not cite the repository URL in it.
