# Results: Do Better Generators Make Worse Teachers?


## Table 1 - cross-generator picture (cub)

| training data | acc (%) | Vendi | coverage | precision | A_real | n |
| --- | --- | --- | --- | --- | --- | --- |
| real | 98.1 [97.6, 98.6] | -- | -- | -- | -- | 3 |
| sd15 (P1) | 90.9 [90.9, 90.9] | 11.2 | 0.700 | 0.622 | 0.068 | 1 |
| sd15 (P2) | 90.6 [90.6, 90.6] | 14.3 | 0.679 | 0.794 | 0.064 | 1 |
| sdturbo (P1) | 91.5 [91.5, 91.5] | 4.6 | 0.031 | 0.028 | 0.073 | 1 |
| sdturbo (P2) | 93.8 [93.8, 93.8] | 8.5 | 0.248 | 0.176 | 0.068 | 1 |
| sdxl (P1) | 96.2 [96.2, 96.2] | 6.5 | 0.358 | 0.268 | 0.074 | 1 |
| sdxl (P2) | 95.9 [95.9, 95.9] | 9.0 | 0.581 | 0.470 | 0.070 | 1 |
| vae.sd15 (VAE) | 98.3 [98.3, 98.3] | 5.2 | 1.000 | 0.976 | -- | 1 |
| vae.sdturbo (VAE) | 98.5 [98.5, 98.5] | 5.2 | 1.000 | 0.976 | -- | 1 |

## Factorial main effects and interaction (pp)

| generator | quantity | estimate (pp) |
| --- | --- | --- |
| sd15 | diversity | +0.06 [-0.09, +0.34] |
| sd15 | alignment | +12.19 [+12.05, +12.48] |
| sd15 | interaction | +1.60 [+1.03, +1.88] |
| sd15 | cell:low/low | +82.11 [+81.54, +82.39] |
| sd15 | cell:low/high | +93.50 [+93.50, +93.50] |
| sd15 | cell:high/low | +81.37 [+81.37, +81.37] |
| sd15 | cell:high/high | +94.36 [+94.36, +94.36] |
| sdturbo | diversity | -0.68 [-0.68, -0.68] |
| sdturbo | alignment | +3.93 [+3.93, +3.93] |
| sdturbo | interaction | +0.34 [+0.34, +0.34] |
| sdturbo | cell:low/low | +92.48 [+92.48, +92.48] |
| sdturbo | cell:low/high | +96.24 [+96.24, +96.24] |
| sdturbo | cell:high/low | +91.62 [+91.62, +91.62] |
| sdturbo | cell:high/high | +95.73 [+95.73, +95.73] |
| sdxl | diversity | -0.46 [-0.68, -0.34] |
| sdxl | alignment | +1.25 [+1.03, +1.37] |
| sdxl | interaction | +0.91 [+0.68, +1.37] |
| sdxl | cell:low/low | +96.47 [+96.24, +96.92] |
| sdxl | cell:low/high | +97.26 [+97.26, +97.26] |
| sdxl | cell:high/low | +95.56 [+95.56, +95.56] |
| sdxl | cell:high/high | +97.26 [+97.26, +97.26] |

## Shapley share of R^2 for D, C, A, F

| dataset | model | n | A | C | D | F | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cub | between-generator | 3 | 0.365 | 0.105 | 0.31 | 0.22 | 1.0 |
| cub | within-pool | 450 | 0.01 | 0.016 | 0.005 | 0.005 | 0.036 |

## Table 2 - DAS versus baselines (cub)

| generator | method | images | acc (%) | delta |
| --- | --- | --- | --- | --- |
| real | Real 5-shot | 100 | 97.8 | -- |
| real | Real-N | 480 | 98.6 | -- |
| sd15 | Random-N | 480 | 90.6 | 0.00 |
| sd15 | All-mN | 1440 | 90.4 | -0.02 |
| sd15 | k-center only | 480 | 88.5 | -0.26 |
| sd15 | Filter only (A_text) | 480 | 91.6 | 0.13 |
| sd15 | DAS-text | 480 | 94.0 | 0.43 |
| sd15 | Filter only (A_real) | 480 | 90.6 | 0.00 |
| sd15 | DAS-real | 480 | 93.0 | 0.30 |
| sd15 | Real 5-shot + DAS-real | 580 | 97.9 | 0.91 |
| sdturbo | Random-N | 480 | 93.8 | 0.00 |
| sdturbo | All-mN | 1440 | 95.4 | 0.32 |
| sdturbo | k-center only | 480 | 92.5 | -0.29 |
| sdturbo | Filter only (A_text) | 480 | 96.2 | 0.50 |
| sdturbo | DAS-text | 480 | 95.0 | 0.25 |
| sdturbo | Filter only (A_real) | 480 | 95.2 | 0.29 |
| sdturbo | DAS-real | 480 | 95.9 | 0.43 |
| sdturbo | Real 5-shot + DAS-real | 580 | 98.5 | 0.96 |
| sdxl | Random-N | 480 | 95.9 | 0.00 |
| sdxl | All-mN | 1440 | 96.2 | 0.12 |
| sdxl | k-center only | 480 | 96.6 | 0.25 |
| sdxl | Filter only (A_text) | 480 | 97.3 | 0.50 |
| sdxl | DAS-text | 480 | 97.1 | 0.44 |
| sdxl | Filter only (A_real) | 480 | 97.3 | 0.50 |
| sdxl | DAS-real | 480 | 97.1 | 0.44 |
| sdxl | Real 5-shot + DAS-real | 580 | 97.6 | 0.62 |

## Mixing with real data

| generator | synthetic | peak real/class | peak acc (%) |
| --- | --- | --- | --- |
| sd15 | DAS-real | 10 | 98.1 |
| sd15 | Random-N | 5 | 97.9 |
| sd15 | none | 5 | 97.9 |
| sdturbo | DAS-real | 5 | 98.5 |
| sdturbo | Random-N | 5 | 98.5 |
| sdturbo | none | 5 | 97.9 |
| sdxl | DAS-real | 5 | 97.6 |
| sdxl | Random-N | 10 | 97.6 |
| sdxl | none | 5 | 97.9 |

## Ablations

| dataset | family | first_centre / centroid | threshold_quantile / 0.2 | threshold_quantile / 0.4 | threshold_quantile / 0.6 |
| --- | --- | --- | --- | --- | --- |
| cub | sd15 | 0.9436 | 0.9316 | 0.9299 | 0.9333 |
| cub | sdturbo | 0.9573 | 0.9504 | 0.959 | 0.9607 |
| cub | sdxl | 0.9726 | 0.9641 | 0.9709 | 0.9744 |

## Collinearity of the four factors

- max |rho| between factors: 0.92
- VIF(vendi) = 147.05
- VIF(coverage) = 35.96
- VIF(alignment) = 62.79
- VIF(precision) = 38.56

## How to fill the paper's blanks

Every `[xx]`/`[Result to write]` marker in the draft maps to one of the tables above;
see docs/PAPER_MAP.md for the marker-by-marker mapping and the command that produces it.
