# Data

Two fine-grained domains carry the paper; HAM10000 is an exploratory third domain for
the supplement. Nothing is redistributed here — the downloaders fetch the public
archives, and HAM10000 must be obtained manually under its own terms.

## CUB-200-2011 (main)

* 200 bird species, 11,788 images; the official split is ~30 train images per class.
* Download (~1.1 GB):

```bash
python -m synthgap.cli download --config configs/exp/main.yaml
# or manually:
#   https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
```

Expected layout:

```
data/raw/CUB_200_2011/
  images.txt  image_class_labels.txt  train_test_split.txt  classes.txt
  images/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg
```

The loader uses the file lists rather than globbing, verifies that every listed image
exists, and refuses to run on a partial extraction.

## FGVC-Aircraft 2013b (main)

* 100 variants, 10,200 images, official variant-level train/val/test split.
* Download (~2.7 GB):

```bash
python -m synthgap.cli download --config configs/exp/aircraft.yaml
# or manually:
#   https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
```

Expected layout:

```
data/raw/fgvc-aircraft-2013b/data/
  variants.txt  images_variant_train.txt  images_variant_val.txt  images_variant_test.txt
  images/...
```

Archive mirrors differ in how the variant is encoded in image filenames; the resolver
tries the known conventions and raises a diagnostic listing what it tried rather than
silently skipping images.

## HAM10000 (supplementary, exploratory)

Requires a Kaggle account and manual download (no anonymous URL): `HAM10000_metadata.csv`
plus `HAM10000_images_part_1/` and `HAM10000_images_part_2/`, arranged as

```
data/raw/HAM10000/HAM10000_metadata.csv
data/raw/HAM10000/HAM10000_images_part_1/*.jpg
data/raw/HAM10000/HAM10000_images_part_2/*.jpg
```

The loader uses the `dx` column as the label and builds a stratified 70/15/15 split.
Only used to probe generator limits; no clinical claim is made.

## Splits produced by the code

For every dataset the pipeline writes a JSON index under `data/indices/` recording
`train`, `test`, `val` and `alignment`. Concretely:

1. `alignment` (k=5 per class, `dataset.alignment_k`) is carved **first** from the real
   train split with a seeded per-class shuffle.
2. `val` is the official validation split where one exists (Aircraft); otherwise it is
   carved from the remainder only if `dataset.real_val_k > 0`.
3. `train` is what is left. `N` — the per-class size of every synthetic set and of
   Real-N — is the minimum class size of `train`.

CUB has no official validation split, so by default `real_val_k: 0` and all remaining
train images are used for Real-N; the probe's weight decay is selected on a held-out
*synthetic* split, per the paper.

## Contamination

The generators and DINOv2 were trained on web-scale data and may have seen these
benchmarks. This cannot be ruled out. It would inflate synthetic accuracy and probe
accuracy alike, i.e. it biases *toward* synthetic data and against the paper's
conclusions. Report it as such.
