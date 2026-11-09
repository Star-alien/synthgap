# Paper marker → artefact map

Every `[xx]` and `[Result to write]` in `cvpr2027_submission_draft.pdf` maps to a file
produced by one command. All paths are relative to `results/`.

| Draft location | Marker | Artefact | Command |
|---|---|---|---|
| Abstract | "[coverage / alignment / fidelity] accounts for [xx]%" | `runs/subsets/shapley.csv` (within-pool shares, dataset column) | `synthgap.cli subsets` |
| Abstract | "FID ranks generators [...] than random" | `runs/measure/table1.csv` + `docs/` note on Kendall τ vs release order | `synthgap.cli measure` |
| Abstract | "DAS closes [xx]% (CUB) and [xx]% (Aircraft)" | `runs/das/table2.csv`, `delta` column, `DAS-real` / `DAS-text` rows | `synthgap.cli das` |
| Sec. 3.1 | "SD3.5-[Large/Medium]", "guidance [value]", "confirm SD3.5/FLUX" | `configs/base.yaml` (`generation.*`) and `runs/measure/table1.csv` metadata | — |
| Sec. 6.1 / Fig. 1 | "Does accuracy fall with newer generators under P1?" + Kendall τ with CI | `figures/fig1_<dataset>.pdf`, `runs/measure/table1.csv` | `synthgap.cli measure figures` |
| Sec. 6.1 | "precision and FD_DINO rank generators in the same or opposite order (H2)" | `runs/measure/table1.csv` (`precision`, `fd`) | `synthgap.cli measure` |
| Sec. 6.1 | "How much does P2 change the ranking, how close does P3 get" | rows with `regime=P2` / `P3` in `table1.csv` | `synthgap.cli measure` (+ `p3.yaml`) |
| Sec. 6.1 | "gap between Real-N and VAE-reconstructed real data" | `table1.csv` rows `selection=VAE-ceiling` vs `Real-N` | `synthgap.cli measure` |
| Sec. 6.2 / Fig. 2a | main effects of diversity, alignment, interaction with CI | `figures/fig2a_<dataset>.pdf`, `runs/factorial/effects.csv` | `synthgap.cli factorial` |
| Sec. 6.2 / Fig. 2b | within-pool vs between-generator Shapley shares, R² | `figures/fig2b_<dataset>.pdf`, `runs/subsets/shapley.csv` | `synthgap.cli subsets` |
| Sec. 6.2 | H3 per-class gap on class familiarity (slope, CI, R²) | `runs/h3/regression.csv`, scatter data in `runs/h3/per_class.csv` | `synthgap.cli h3` |
| Sec. 6.2 | robustness: CLIP instead of DINOv2, A_text instead of A_real, rank agreement | `runs/subsets/shapley.csv` re-run with `--set features.primary=clip`; scorer agreement from `--set selection.das.quantile` ablation + `runs/subsets/collinearity.json` | see below |
| Sec. 6.2 | collinearity of D, C, A, F | `runs/subsets/collinearity.json` | `synthgap.cli subsets` |
| Sec. 6.3 | Spearman between probe and from-scratch ResNet, sign of main effects | `runs/scratch/scratch.csv` (`probe_accuracy`, `scratch_accuracy`) | `synthgap.cli scratch` |
| Sec. 6.4 / Table 2 | Δ for DAS-real / DAS-text, beats All-mN, vs Real 5-shot | `runs/das/table2.csv` (`accuracy`, `delta`) | `synthgap.cli das` |
| Sec. 6.4 | ablations: m ∈ {2,3,5}, τ, selection embedding, first-centre rule | `runs/ablation/ablation.csv`; for m, regenerate pools with `--set generation.images_per_class_multiplier=2` and rerun `das` | `synthgap.cli ablation` |
| Sec. 6.4 / Fig. 3 | real budget at which synthetic stops helping; does DAS move it | `figures/fig3_<dataset>.pdf`, `runs/mixing/mixing.csv` | `synthgap.cli mixing` |
| Fig. 4 | qualitative failure modes, chosen by a stated rule | `figures/fig4_<dataset>.pdf` (3 random classes, 6 lowest `A_real` each) | `synthgap.cli figures` |
| Sec. 7 | implications paragraph | prose, written from the numbers above | — |
| LLM usage statement | P2 template generator, P3 captioner name/version | `configs/base.yaml`, `synthgap/data/prompts.py` header, `synthgap/gen/caption.py` | — |

## Robustness re-runs (Sec. 6.2)

The two embedding-space checks are single-command re-runs; they write to separate
directories because the cache keys include the embedding name:

```bash
# all metrics in CLIP space instead of DINOv2
python -m synthgap.cli subsets --config configs/exp/main.yaml \
  --set features.primary=clip --set paths.results=results_clip

# A_text instead of A_real for the alignment factor
python -m synthgap.cli subsets --config configs/exp/main.yaml \
  --set selection.score=a_text --set paths.results=results_atext
```

Scorer agreement (A_real vs A_text per image, expected to be weaker on Aircraft) is
available directly from the cached arrays in `data/features/align__*.npz`.

## What still requires human writing

The prose in Sec. 6.1–6.4 ("Result to write" paragraphs), the implications paragraph and
the final caption sentences are interpretive. The code produces the numbers, CIs and
figures; the sentences that say what they mean are the authors'.
