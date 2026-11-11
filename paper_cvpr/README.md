# CVPR 2027 submission (review format)

`main.pdf` is the compiled paper, built from `main.tex` with the official CVPR 2027 author kit (`cvpr.sty`, `ieeenat_fullname.bst`).

Numbers and tables are **generated from `results_pilot/`** by `python scripts/make_tex.py` (writes `gen_macros.tex`, `gen_tables.tex`). Nothing in the results is typed by hand.

Build: `tectonic main.tex` (or upload this folder to Overleaf and compile `main.tex`).

## Before submitting (authors)
- Confirm/rewrite the LLM usage statement (bracketed text) and verify every citation against its original publication.
- Add the name/version of the LLM that wrote the P2 templates.
- This version is a **pilot** (20 CUB classes, 3 generators, 1 seed). See Sec. 7 of the paper for what was not run.
- The paper is anonymous; do not cite this repository URL in the review copy.


## Restyled version
The paper keeps its original text and title but uses the figures, tables (Wilson CIs, bold best) and teaser/findings layout of the review version. Tables `tab_*.tex` come from `python scripts/make_tex_tables.py`; `gen_macros.tex` holds the inline numbers. Figures 1-6 are generated from the stored results by `scripts/make_figures.py` (Figs 1-5 from `results_pilot/runs/*`, Fig 6 from the SD1.5/P2 pool images and cached alignment scores; classes chosen by a seed-0 rule) and saved separately in `results_pilot/figures/fig{1..6}.{pdf,png}`. Body is 8 pages, references on 9-10.
