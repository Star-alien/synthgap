# Compute plan

Generation dominates the budget; everything else is cheap by comparison. The numbers
below are **estimates** from per-image timings typical of an A100-80GB at 50 steps and
batch size 8. The pipeline logs wall-clock time per class, so measure the first pool and
scale from there rather than trusting these figures.

## How many images

| | CUB-200-2011 | FGVC-Aircraft |
|---|---|---|
| classes | 200 | 100 |
| N per class (after carving k=5 alignment images) | ~27 | ~33 |
| pool size per (generator, prompt, seed) = m·N·K, m=3 | ~16,200 | ~9,900 |
| pools (4 generators × 2 prompts × 3 seeds) | 24 | 24 |
| images | ~389k | ~238k |
| **total** | **~627k** | |

Plus VAE-reconstruction ceiling pools (one per generator, N per class, no diffusion
steps: essentially free) and optional P3 oracle pools (SD1.5 and SDXL only, per the draft).

## Wall-clock per image (A100-80GB, 50 steps)

| generator | resolution | ~seconds/image |
|---|---|---|
| SD1.5 | 512² | 0.4 |
| SDXL | 1024² | 3.2 |
| SD3.5-Large | 1024² | 4.5 (Medium: 1.8) |
| FLUX.1-dev | 1024² | 6.0 |

## Estimated totals

| plan | scope | generation | everything else |
|---|---|---|---|
| **A — paper as designed** | both datasets, 4 generators, P1+P2, 3 seeds | ~610 GPU-h (~25 GPU-days) | features ~1 h, scratch control ~2 h, ~4.4k probes ≈ 50–75 CPU-h (≈3 h on 32 cores) |
| **A′ — SD3.5-Medium** | same, cheaper third generation | ~500 GPU-h | as above |
| **B — minimum viable** | both datasets, 4 generators, P1+P2, **1 seed** | ~205 GPU-h | 1/3 of the probes |
| **C — pilot** | SDXL + FLUX, 1 seed, both datasets | ~90 GPU-h | small |

The three generation seeds exist to put CIs on Fig. 1. If compute is tight, run plan B
and say so in the paper; the interventional designs (factorial, subsets, DAS) need only
one generation seed per pool because their variance comes from *selection* seeds, which
are free.

## Storage

627k images at 1024² PNG is ~0.9 TB. Options, in order of preference:

1. `generation.save_format: jpg` (quality 95) — ~150 GB, no measurable effect on
   DINOv2/CLIP features or on the 128² from-scratch control.
2. Generate at native resolution but keep only 512² copies for storage; the paper
   resizes to 224² before any learner anyway.
3. Keep PNG only for the SD1.5 pool.

Features add ~50 MB per pool (float32, DINOv2-B), which is negligible.

## Memory

* FLUX.1-dev (12B) and SD3.5-Large (~8B) need ≥24 GB in bf16. On 24 GB cards set
  `generation.low_vram: true`, which enables attention slicing and CPU offload at
  roughly 2–3× the per-image time.
* SDXL and SD1.5 fit comfortably on 16–24 GB.
* The analysis stages are CPU-only and fit in a few GB: features are cached as float32
  arrays and every metric is computed per class.

## Order of operations, and what can be parallelised

1. `download` (once, ~4 GB).
2. `generate` — embarrassingly parallel over (generator, prompt, seed); the natural unit
   is one GPU per generator. Resumable per class.
3. `measure`, `factorial`, `subsets`, `das`, `h3`, `mixing`, `ablation` — CPU, parallel
   over pools; all read cached features.
4. `scratch` — GPU, 24 short trainings.
5. `figures`, `report` — seconds.

Every stage caches to disk, so re-running after an interruption is cheap.
