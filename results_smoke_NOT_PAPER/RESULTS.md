# Results: Do Better Generators Make Worse Teachers?

> **SMOKE-TEST OUTPUT - NOT PAPER RESULTS**
>
> These numbers come from procedurally generated stand-in images and a cheap CPU
> feature extractor. They prove the pipeline runs end to end; they are not results
> for the paper. Re-run with a GPU config to obtain the real numbers.


## Table 1 - cross-generator picture (mockbirds)

| training data | acc (%) | Vendi | coverage | precision | A_real | n |
| --- | --- | --- | --- | --- | --- | --- |
| flux (P1) | 47.5 [47.5, 47.5] | 1.5 | 0.411 | 0.232 | 0.137 | 1 |
| flux (P2) | 41.2 [41.2, 41.2] | 1.5 | 0.381 | 0.232 | 0.137 | 1 |
| real | 52.9 [50.0, 58.8] | -- | -- | -- | -- | 3 |
| sd15 (P1) | 37.5 [37.5, 37.5] | 1.3 | 0.470 | 0.607 | 0.126 | 1 |
| sd15 (P2) | 35.0 [35.0, 35.0] | 1.3 | 0.506 | 0.512 | 0.127 | 1 |
| sd35 (P1) | 52.5 [52.5, 52.5] | 1.4 | 0.732 | 0.524 | 0.133 | 1 |
| sd35 (P2) | 56.2 [56.2, 56.2] | 1.4 | 0.702 | 0.631 | 0.133 | 1 |
| sdxl (P1) | 57.5 [57.5, 57.5] | 1.4 | 0.827 | 0.631 | 0.130 | 1 |
| sdxl (P2) | 52.5 [52.5, 52.5] | 1.3 | 0.786 | 0.649 | 0.130 | 1 |
| vae.flux (VAE) | 58.8 [58.8, 58.8] | 1.3 | 1.000 | 1.000 | -- | 1 |
| vae.sd15 (VAE) | 58.8 [58.8, 58.8] | 1.3 | 1.000 | 1.000 | -- | 1 |
| vae.sd35 (VAE) | 58.8 [58.8, 58.8] | 1.3 | 1.000 | 1.000 | -- | 1 |
| vae.sdxl (VAE) | 58.8 [58.8, 58.8] | 1.3 | 1.000 | 1.000 | -- | 1 |

## Factorial main effects and interaction (pp)

| generator | quantity | estimate (pp) |
| --- | --- | --- |
| flux | diversity | +1.88 [+0.00, +3.75] |
| flux | alignment | +23.12 [+21.25, +25.00] |
| flux | interaction | +8.75 [+5.00, +12.50] |
| flux | cell:low/low | +32.50 [+28.75, +36.25] |
| flux | cell:low/high | +51.25 [+51.25, +51.25] |
| flux | cell:high/low | +30.00 [+30.00, +30.00] |
| flux | cell:high/high | +57.50 [+57.50, +57.50] |
| sd15 | diversity | +12.50 [+11.88, +13.13] |
| sd15 | alignment | +30.00 [+29.38, +30.63] |
| sd15 | interaction | +22.50 [+21.25, +23.75] |
| sd15 | cell:low/low | +16.25 [+16.25, +16.25] |
| sd15 | cell:low/high | +35.00 [+35.00, +35.00] |
| sd15 | cell:high/low | +17.50 [+16.25, +18.75] |
| sd15 | cell:high/high | +58.75 [+58.75, +58.75] |
| sd35 | diversity | +0.63 [+0.00, +1.25] |
| sd35 | alignment | +25.63 [+25.00, +26.25] |
| sd35 | interaction | +6.25 [+5.00, +7.50] |
| sd35 | cell:low/low | +33.75 [+33.75, +33.75] |
| sd35 | cell:low/high | +56.25 [+56.25, +56.25] |
| sd35 | cell:high/low | +31.25 [+31.25, +31.25] |
| sd35 | cell:high/high | +60.00 [+58.75, +61.25] |
| sdxl | diversity | -0.31 [-1.25, +0.63] |
| sdxl | alignment | +25.31 [+25.00, +25.62] |
| sdxl | interaction | -6.87 [-7.50, -6.25] |
| sdxl | cell:low/low | +28.75 [+28.75, +28.75] |
| sdxl | cell:low/high | +57.50 [+57.50, +57.50] |
| sdxl | cell:high/low | +31.87 [+31.25, +32.50] |
| sdxl | cell:high/high | +53.75 [+52.50, +55.00] |

## Shapley share of R^2 for D, C, A, F

| dataset | model | n | A | C | D | F | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mockbirds | between-generator | 4 | 0.219 | 0.317 | 0.28 | 0.185 | 1.0 |
| mockbirds | within-pool | 160 | 0.027 | 0.042 | 0.011 | 0.006 | 0.086 |

## Table 2 - DAS versus baselines (mockbirds)

| generator | method | images | acc (%) | delta |
| --- | --- | --- | --- | --- |
| flux | Random-N | 168 | 41.2 | 0.00 |
| flux | All-mN | 504 | 52.5 | 0.64 |
| flux | k-center only | 168 | 55.0 | 0.79 |
| flux | Filter only (A_text) | 168 | 48.8 | 0.43 |
| flux | DAS-text | 168 | 52.5 | 0.64 |
| flux | Filter only (A_real) | 168 | 52.5 | 0.64 |
| flux | DAS-real | 168 | 60.0 | 1.07 |
| flux | Real 5-shot + DAS-real | 208 | 63.7 | 1.29 |
| real | Real 5-shot | 40 | 48.8 | -- |
| real | Real-N | 168 | 58.8 | -- |
| sd15 | Random-N | 168 | 35.0 | 0.00 |
| sd15 | All-mN | 504 | 48.8 | 0.58 |
| sd15 | k-center only | 168 | 55.0 | 0.84 |
| sd15 | Filter only (A_text) | 168 | 51.2 | 0.68 |
| sd15 | DAS-text | 168 | 63.7 | 1.21 |
| sd15 | Filter only (A_real) | 168 | 62.5 | 1.16 |
| sd15 | DAS-real | 168 | 53.8 | 0.79 |
| sd15 | Real 5-shot + DAS-real | 208 | 63.7 | 1.21 |
| sd35 | Random-N | 168 | 56.2 | 0.00 |
| sd35 | All-mN | 504 | 58.8 | 1.00 |
| sd35 | k-center only | 168 | 57.5 | 0.50 |
| sd35 | Filter only (A_text) | 168 | 56.2 | 0.00 |
| sd35 | DAS-text | 168 | 56.2 | 0.00 |
| sd35 | Filter only (A_real) | 168 | 62.5 | 2.50 |
| sd35 | DAS-real | 168 | 65.0 | 3.50 |
| sd35 | Real 5-shot + DAS-real | 208 | 58.8 | 1.00 |
| sdxl | Random-N | 168 | 52.5 | 0.00 |
| sdxl | All-mN | 504 | 60.0 | 1.20 |
| sdxl | k-center only | 168 | 41.2 | -1.80 |
| sdxl | Filter only (A_text) | 168 | 53.8 | 0.20 |
| sdxl | DAS-text | 168 | 58.8 | 1.00 |
| sdxl | Filter only (A_real) | 168 | 55.0 | 0.40 |
| sdxl | DAS-real | 168 | 58.8 | 1.00 |
| sdxl | Real 5-shot + DAS-real | 208 | 62.5 | 1.60 |

## Mixing with real data

| generator | synthetic | peak real/class | peak acc (%) |
| --- | --- | --- | --- |
| flux | DAS-real | 5 | 63.7 |
| flux | Random-N | 10 | 52.5 |
| flux | none | 5 | 50.0 |
| sd15 | DAS-real | 5 | 63.7 |
| sd15 | Random-N | 5 | 56.2 |
| sd15 | none | 5 | 50.0 |
| sd35 | DAS-real | 0 | 65.0 |
| sd35 | Random-N | 0 | 56.2 |
| sd35 | none | 5 | 50.0 |
| sdxl | DAS-real | 5 | 62.5 |
| sdxl | Random-N | 5 | 56.2 |
| sdxl | none | 5 | 50.0 |

## Ablations

| dataset | family | first_centre / centroid | threshold_quantile / 0.2 | threshold_quantile / 0.4 | threshold_quantile / 0.6 |
| --- | --- | --- | --- | --- | --- |
| mockbirds | flux | 0.575 | 0.625 | 0.6 | 0.5625 |
| mockbirds | sd15 | 0.5875 | 0.4875 | 0.5375 | 0.625 |
| mockbirds | sd35 | 0.6125 | 0.6375 | 0.65 | 0.6125 |
| mockbirds | sdxl | 0.525 | 0.5625 | 0.5875 | 0.5875 |

## Collinearity of the four factors

- max |rho| between factors: 0.90
- VIF(vendi) = 8.49
- VIF(coverage) = 8.04
- VIF(alignment) = 9.00
- VIF(precision) = 12.85

## How to fill the paper's blanks

Every `[xx]`/`[Result to write]` marker in the draft maps to one of the tables above;
see docs/PAPER_MAP.md for the marker-by-marker mapping and the command that produces it.
