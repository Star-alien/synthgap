"""synthgap: interventional study of synthetic training data for fine-grained recognition.

Code for "Do Better Generators Make Worse Teachers? Disentangling Diversity, Label
Alignment and Fidelity in Synthetic Training Data".

Module map (stage -> module):
    data/      datasets, splits, alignment split A, prompt regimes P1/P2/P3
    gen/       generator backends (SD1.5, SDXL, SD3.5, FLUX) + mock + VAE ceiling
    features/  DINOv2 / CLIP / Inception feature extraction with on-disk cache
    metrics/   Vendi, coverage, precision/recall, density, FID/KID/FD-DINO, alignment
    select/    random, clustered, k-center, quantile, DAS and baselines
    learn/     frozen linear probe, from-scratch ResNet, real-only / hybrid baselines
    design/    2x2 factorial, balanced random-subset design, cluster bootstrap
    stats/     Eq. (1) fixed-effects regression, Shapley (LMG) decomposition, correlations
    plots/     Figures 1-4 and the LaTeX tables
"""

__version__ = "0.1.0"
