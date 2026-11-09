# synthgap

Code for *"Do Better Generators Make Worse Teachers? Disentangling Diversity, Label
Alignment and Fidelity in Synthetic Training Data"* (CVPR 2027 submission).

The paper asks **which property of a synthetic training set drives real-test accuracy**
and answers by *intervention*: hold the generator, prompt and image pool fixed, then
build equal-sized training sets that differ in exactly one measured property
(intrinsic diversity, coverage, label alignment, per-sample fidelity). It then turns
the decomposition into **DAS** (Diversity-matched, Alignment-filtered Sampling), a
training-free selection rule.

This repository implements every experiment in the draft: pool generation for four
open generators, all measurements, both interventional designs, the Shapley
decomposition, DAS and every baseline, the from-scratch control, H3, the ablations,
Figures 1–4 and the LaTeX-ready tables.

---

## Status: what has and has not been run

| Component | State |
|---|---|
| Full pipeline (all stages, Figures 1–4, tables, results document) | **implemented** |
| Unit tests for metrics, selection, decomposition, splits | **14 tests, passing** |
| End-to-end CPU run (procedural generators + cheap features) | **verified**, ~4 min |
| Real diffusion generation, DINOv2/CLIP features, ResNet control | **implemented, requires GPUs** (not run here) |
| Paper numbers in the abstract/tables | **not produced** — see below |

**No experimental numbers are fabricated anywhere in this repo.** Blank cells in the
draft are filled only by running the stages on real data; the mock mode writes its
outputs under an explicit "SMOKE-TEST OUTPUT — NOT PAPER RESULTS" banner and exists
only to prove the pipeline is wired correctly. A machine with GPUs and the datasets
is required for the real numbers (see `docs/COMPUTE_PLAN.md`).

---

## Quick start: verify the code in ~4 minutes (CPU only)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest -q                      # 14 unit tests
python -m synthgap.cli all --config configs/exp/smoke.yaml
```

That runs every stage on a small procedural dataset with cheap features and writes:

```
results/figures/fig1_mockbirds.pdf    Fig. 1  fidelity vs accuracy across generations
results/figures/fig2a_mockbirds.pdf   Fig. 2a 2x2 factorial
results/figures/fig2b_mockbirds.pdf   Fig. 2b Shapley shares, within vs between pools
results/figures/fig3_mockbirds.pdf    Fig. 3  mixing curves
results/figures/fig4_mockbirds.pdf    Fig. 4  lowest-alignment samples
results/tables/table1_mockbirds.md    Table 1
results/tables/table2_mockbirds.md     Table 2 (DAS vs baselines, with fraction of gap closed)
results/tables/shapley_mockbirds.md    variance decomposition
results/RESULTS.md                     everything assembled, with the smoke banner
```

## Running the real experiments

```bash
pip install -r requirements.txt -r requirements-gpu.txt   # torch/diffusers/transformers
python -m synthgap.cli download --config configs/exp/main.yaml
python -m synthgap.cli generate --config configs/exp/main.yaml     # 4 generators x P1,P2 x 3 seeds
python -m synthgap.cli measure  --config configs/exp/main.yaml     # Table 1 + Fig. 1
python -m synthgap.cli factorial --config configs/exp/main.yaml --config configs/exp/factorial.yaml
python -m synthgap.cli subsets   --config configs/exp/main.yaml --config configs/exp/subsets.yaml
python -m synthgap.cli das       --config configs/exp/main.yaml --config configs/exp/das.yaml
python -m synthgap.cli h3        --config configs/exp/main.yaml
python -m synthgap.cli mixing    --config configs/exp/main.yaml --config configs/exp/mixing.yaml
python -m synthgap.cli ablation  --config configs/exp/main.yaml --config configs/exp/ablation.yaml
python -m synthgap.cli scratch   --config configs/exp/main.yaml --config configs/exp/scratch.yaml
python -m synthgap.cli figures   --config configs/exp/main.yaml
python -m synthgap.cli report    --config configs/exp/main.yaml
```

`make` wraps the same commands. For FGVC-Aircraft add `--config configs/exp/aircraft.yaml`;
for the oracle P3 regime, first produce captions (`python -m synthgap.gen.caption`) and
then use `--config configs/exp/p3.yaml`.

Every stage is resumable: pools, features, alignment scores and subsets are cached on
disk and reused, so an interrupted run continues where it stopped. Any config value
can be overridden inline, e.g. `--set generation.seeds=[0]`.

---

## Repository layout

```
synthgap/
  data/       dataset indices (CUB, FGVC-Aircraft, HAM10000, mock), prompt regimes
  gen/        generator backends (SD1.5, SDXL, SD3.5, FLUX.1-dev), pools, VAE ceiling, P3 captioner
  features/   DINOv2 / CLIP / Inception / cheap feature extraction with caching
  metrics/    Vendi, coverage, improved precision/recall, density, FD/KID, alignment scorers
  select/     random, clustered, k-center, quantile bands, DAS and all baselines
  learn/      frozen linear probe, from-scratch ResNet control
  design/     2x2 factorial, balanced random-subset design, cluster bootstrap
  stats/      Eq. (1) fixed-effects regression, Shapley (LMG) decomposition, correlations
  plots/      Figures 1-4
  pipeline.py all experiment stages
  cli.py, report.py, config.py, utils.py
configs/      base defaults + experiment configs (smoke, main, aircraft, per-stage)
tests/        unit tests (metrics, selection, decomposition, splits)
docs/         DATA.md, COMPUTE_PLAN.md, PAPER_MAP.md
data/         datasets, pools, features (git-ignored)
results/      runs/, figures/, tables/, RESULTS.md
```

## Protocol decisions worth knowing

These follow the draft and are enforced in code, not left to convention:

1. **Leak-free alignment split.** The k=5 real images per class used by `A_real` and by
   the matched 5-shot baseline are carved *first*, then removed from the real training
   split. Real-N therefore never contains an image that a scorer has seen.
2. **Selection and measurement embeddings are disjoint.** Selection (DAS, k-center,
   quantile bands) operates in CLIP space; every reported metric and the learner use
   DINOv2. DAS cannot win by optimising the space it is scored in.
3. **No real data in the default protocol.** Weight decay is chosen on a held-out
   *synthetic* validation split. Using real validation data is available only as an
   explicitly flagged upper bound (`learner.protocol: real_val_upper_bound`).
4. **Per-class metrics at equal N.** Vendi, coverage, precision and density are computed
   per class and averaged, because several of them depend on sample size.
5. **Undefined quantities stay undefined.** The fraction of gap closed is reported as
   `--` when the denominator is near zero, as the paper commits to.

## Where each paper artefact comes from

`docs/PAPER_MAP.md` maps every `[xx]` and `[Result to write]` marker in the draft to the
file and command that fills it. Compute requirements and GPU-hour estimates are in
`docs/COMPUTE_PLAN.md`.

## Licences

Datasets are used under their research-use terms; generators under their own model
licences (non-commercial research). No new human-subject data is collected. The code is
released under MIT. Selection indices can be released without redistributing images.
