# Results status (updated)

**What was run (real data, no fabricated numbers):** a compute-limited pilot on 20 evenly spaced CUB-200-2011 classes,
three generators (SD1.5, SD-Turbo, SDXL at 768px), prompts P1/P2, one generation seed. Generation ran on free Kaggle T4
GPUs; analysis ran on a laptop (Apple GPU). Outputs: `results_pilot/` (tables, figures, CSVs) and the CVPR-format paper `paper_cvpr/main.pdf`.

**What was NOT run:** FGVC-Aircraft, SD3.5 / FLUX, 3 generation seeds, the oracle caption prompt P3, the from-scratch ResNet
control, SDXL's VAE reference, and the full 500-subset design (150 used). See Section 5 of the paper.

**Headline (pilot only):** SDXL is the best teacher (96.2% vs 91.5% SD-Turbo, 90.9% SD1.5, real 98.6%), so the hypothesised
"better generators teach worse" inversion is not observed here. Label alignment is the dominant lever within a generator;
diversity has almost no effect; the four properties are too collinear to attribute variance.

`results_smoke_NOT_PAPER/` and `paper_DRAFT_blank_cells.pdf` are the earlier mock-data smoke test and the original, unfilled draft.
