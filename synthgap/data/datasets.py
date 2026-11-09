"""Dataset indices: train / test / val plus the alignment split A (Sec. 3).

Conventions
-----------
* Every path is stored *relative* to ``root`` so indices stay portable.
* ``alignment`` (k images per class) is carved from the real train split *first*;
  ``train`` is what remains. Real-N therefore means "all remaining real training
  images", and no image used by a scorer or by the 5-shot baseline is ever seen
  by Real-N training. This is the leak-free version of the protocol in Sec. 3.
* ``val`` is either the official validation split (FGVC-Aircraft) or a carved
  slice of train (CUB). It is only used by the explicitly-flagged
  ``real_val_upper_bound`` protocol and by the from-scratch control.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..config import Cfg, resolve_dir
from ..utils import ensure_dir, json_dump, json_load, stable_hash


@dataclass
class DatasetIndex:
    name: str
    root: str
    classes: list[str]
    train: list[list[str]]
    test: list[list[str]]
    val: list[list[str]]
    alignment: list[list[str]]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.classes)
        for attr in ("train", "test", "val", "alignment"):
            seq = getattr(self, attr)
            if len(seq) != n:
                raise ValueError(f"{attr} has {len(seq)} entries for {n} classes")
        if any(len(c) < 2 for c in self.train):
            counts = [len(c) for c in self.train]
            raise ValueError(
                f"{self.name}: some classes have <2 training images after carving "
                f"(min={min(counts)}, max={max(counts)})"
            )

    # -- geometry -----------------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def per_class_train(self) -> int:
        """N: the matched per-class size of every synthetic set and of Real-N."""
        return min(len(c) for c in self.train)

    @property
    def alignment_size(self) -> int:
        counts = [len(c) for c in self.alignment]
        return min(counts) if counts else 0

    def path(self, relative: str) -> Path:
        return Path(self.root) / relative

    def class_paths(self, split: str, c: int) -> list[Path]:
        return [self.path(p) for p in getattr(self, split)[c]]

    # -- serialisation -------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "root": self.root,
            "classes": self.classes,
            "train": self.train,
            "test": self.test,
            "val": self.val,
            "alignment": self.alignment,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetIndex":
        return cls(**data)

    def summary(self) -> str:
        return (
            f"{self.name}: {self.num_classes} classes, N={self.per_class_train}/class train, "
            f"|A|={self.alignment_size}/class, "
            f"{sum(len(c) for c in self.test)} test images"
        )


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _carve(per_class: list[list[str]], k: int, rng: np.random.Generator) -> tuple[list[list[str]], list[list[str]]]:
    """Split each class into (first k, rest) with a deterministic per-class shuffle."""
    if k <= 0:
        return [[] for _ in per_class], per_class
    picked: list[list[str]] = []
    rest: list[list[str]] = []
    for items in per_class:
        order = rng.permutation(len(items))
        ordered = [items[i] for i in order]
        picked.append(ordered[:k])
        rest.append(ordered[k:])
    return picked, rest


def _read_lines(path: Path) -> list[str]:
    with open(path) as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def _index_cache_path(cfg: Cfg) -> Path:
    key = stable_hash(
        cfg.dataset.name,
        cfg.dataset.alignment_k,
        cfg.dataset.real_val_k,
        cfg.dataset.index_seed,
    )
    return resolve_dir(cfg, "indices") / f"{cfg.dataset.name}_{key}.json"


# --------------------------------------------------------------------------------------
# CUB-200-2011
# --------------------------------------------------------------------------------------
def load_cub(root: Path, alignment_k: int = 5, real_val_k: int = 0, seed: int = 0) -> DatasetIndex:
    root = Path(root) / "CUB_200_2011"
    for required in ("images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt"):
        if not (root / required).exists():
            raise FileNotFoundError(
                f"CUB-200-2011 not found at {root} (missing {required}). "
                "Run `python -m synthgap.cli download --config configs/exp/main.yaml`."
            )

    images = {}
    for line in _read_lines(root / "images.txt"):
        image_id, rel = line.split(" ", 1)
        images[int(image_id)] = rel
    labels = {}
    for line in _read_lines(root / "image_class_labels.txt"):
        image_id, cls = line.split()
        labels[int(image_id)] = int(cls) - 1
    split = {}
    for line in _read_lines(root / "train_test_split.txt"):
        image_id, flag = line.split()
        split[int(image_id)] = int(flag) == 1
    class_names = {}
    for line in _read_lines(root / "classes.txt"):
        cls, name = line.split(" ", 1)
        class_names[int(cls) - 1] = name

    n_classes = len(class_names)
    train = [[] for _ in range(n_classes)]
    test = [[] for _ in range(n_classes)]
    missing = 0
    for image_id, rel in images.items():
        if not (root / "images" / rel).exists():
            missing += 1
            continue
        bucket = train if split.get(image_id) else test
        bucket[labels[image_id]].append(str(Path("CUB_200_2011") / "images" / rel))
    if missing:
        raise FileNotFoundError(f"{missing} CUB images listed in images.txt are missing on disk")

    rng = np.random.default_rng(seed)
    alignment, rest = _carve(train, alignment_k, rng)
    val, rest = _carve(rest, real_val_k, rng)
    return DatasetIndex(
        name="cub",
        root=str(Path(root).parent),
        classes=[class_names[i] for i in range(n_classes)],
        train=rest,
        test=test,
        val=val,
        alignment=alignment,
        meta={"source": "CUB-200-2011", "official_val": False, "missing": missing},
    )


# --------------------------------------------------------------------------------------
# FGVC-Aircraft
# --------------------------------------------------------------------------------------
def _resolve_aircraft_image(images_dir: Path, image_id: str, variant: str) -> Path | None:
    """FGVC-Aircraft archives differ in how the variant is encoded in the filename."""
    candidates = [
        images_dir / f"{image_id}.jpg",
        images_dir / f"{image_id}_{variant.replace(' ', '_')}.jpg",
        images_dir / f"{image_id}_{variant}.jpg",
        images_dir / f"{image_id}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(images_dir.glob(f"{image_id}*"))
    return matches[0] if len(matches) == 1 else None


def load_aircraft(root: Path, alignment_k: int = 5, seed: int = 0) -> DatasetIndex:
    data_dir = Path(root) / "fgvc-aircraft-2013b" / "data"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"FGVC-Aircraft not found at {data_dir}. "
            "Run `python -m synthgap.cli download --config configs/exp/main.yaml`."
        )
    variants = [v.strip() for v in _read_lines(data_dir / "variants.txt")]
    variant_to_idx = {v: i for i, v in enumerate(variants)}
    images_dir = data_dir / "images"

    def read_split(filename: str) -> list[list[str]]:
        buckets: list[list[str]] = [[] for _ in variants]
        path = data_dir / filename
        if not path.exists():
            return buckets
        for line in _read_lines(path):
            image_id, variant = line.split(" ", 1)
            variant = variant.strip()
            if variant not in variant_to_idx:
                continue
            resolved = _resolve_aircraft_image(images_dir, image_id, variant)
            if resolved is None:
                raise FileNotFoundError(
                    f"image {image_id} ({variant}) listed in {filename} not found in {images_dir}"
                )
            buckets[variant_to_idx[variant]].append(str(resolved.relative_to(root)))
        return buckets

    train_all = read_split("images_variant_train.txt")
    val = read_split("images_variant_val.txt")
    test = read_split("images_variant_test.txt")
    if not val or not any(val):
        # Some mirrors ship trainval only; carve a validation slice instead.
        rng = np.random.default_rng(seed + 7)
        val, train_all = _carve(train_all, max(1, min(3, min(len(c) for c in train_all) // 5)), rng)

    rng = np.random.default_rng(seed)
    alignment, train = _carve(train_all, alignment_k, rng)
    return DatasetIndex(
        name="aircraft",
        root=str(root),
        classes=variants,
        train=train,
        test=test,
        val=val,
        alignment=alignment,
        meta={"source": "FGVC-Aircraft 2013b", "official_val": True},
    )


# --------------------------------------------------------------------------------------
# HAM10000 (exploratory, supplement)
# --------------------------------------------------------------------------------------
def load_ham10000(root: Path, alignment_k: int = 5, seed: int = 0) -> DatasetIndex:
    import pandas as pd

    root = Path(root) / "HAM10000"
    metadata = root / "HAM10000_metadata.csv"
    if not metadata.exists():
        raise FileNotFoundError(
            f"HAM10000 metadata not found at {metadata}. The dataset needs a Kaggle "
            "account; see docs/DATA.md for the manual download steps."
        )
    frame = pd.read_csv(metadata)
    classes = sorted(frame["dx"].unique())
    idx = {c: i for i, c in enumerate(classes)}
    all_images = {}
    for part in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
        for path in (root / part).glob("*.jpg"):
            all_images[path.stem] = str(path.relative_to(root))

    buckets: list[list[str]] = [[] for _ in classes]
    for _, row in frame.iterrows():
        rel = all_images.get(row["image_id"])
        if rel:
            buckets[idx[row["dx"]]].append(rel)

    rng = np.random.default_rng(seed)
    per_class_train, test, val = [], [], []
    for items in buckets:
        order = rng.permutation(len(items))
        ordered = [items[i] for i in order]
        n_test = max(2, int(0.15 * len(ordered)))
        n_val = max(2, int(0.1 * len(ordered)))
        test.append(ordered[:n_test])
        val.append(ordered[n_test : n_test + n_val])
        per_class_train.append(ordered[n_test + n_val :])
    alignment, train = _carve(per_class_train, alignment_k, rng)
    return DatasetIndex(
        name="ham10000",
        root=str(root),
        classes=classes,
        train=train,
        test=test,
        val=val,
        alignment=alignment,
        meta={"source": "HAM10000", "official_val": False, "exploratory": True},
    )


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------
def build_index(cfg: Cfg, refresh: bool = False) -> DatasetIndex:
    """Load (or load from cache) the dataset index described by `cfg`."""
    cache = _index_cache_path(cfg)
    if not refresh and cache.exists():
        return DatasetIndex.from_dict(json_load(cache))

    raw = resolve_dir(cfg, "raw")
    name = cfg.dataset.name
    k = int(cfg.dataset.alignment_k)
    seed = int(cfg.dataset.index_seed)
    if name == "cub":
        index = load_cub(raw, alignment_k=k, real_val_k=int(cfg.dataset.real_val_k), seed=seed)
    elif name == "aircraft":
        index = load_aircraft(raw, alignment_k=k, seed=seed)
    elif name == "ham10000":
        index = load_ham10000(raw, alignment_k=k, seed=seed)
    elif name in {"mockbirds", "mock"}:
        from .mock_dataset import build_mock_dataset

        root = build_mock_dataset(raw / "mockbirds", seed=seed)
        index = _load_flat_mock(root, alignment_k=k, real_val_k=int(cfg.dataset.real_val_k), seed=seed)
    else:
        raise ValueError(f"unknown dataset {name!r}")
    max_classes = cfg.get("dataset.max_classes", None)
    if max_classes:
        index = subset_classes(index, int(max_classes))
    json_dump(index.to_dict(), cache)
    return index


def subset_classes(index: "DatasetIndex", n: int) -> "DatasetIndex":
    """Deterministic, evenly spaced class subset (keeps taxonomic spread; no cherry-picking)."""
    total = index.num_classes
    if n >= total:
        return index
    keep = sorted({int(round(i * (total - 1) / (n - 1))) for i in range(n)}) if n > 1 else [0]
    pick = lambda seq: [seq[i] for i in keep]  # noqa: E731
    meta = dict(index.meta, subset_of=total, kept_original_ids=keep)
    return DatasetIndex(
        name=index.name, root=index.root, classes=pick(index.classes), train=pick(index.train),
        test=pick(index.test), val=pick(index.val), alignment=pick(index.alignment), meta=meta,
    )


def _load_flat_mock(root: Path, alignment_k: int, real_val_k: int, seed: int) -> DatasetIndex:
    """Mock dataset layout: root/<split>/<class>/*.png with classes.json alongside."""
    classes = json_load(Path(root) / "classes.json")
    if classes is None:
        raise FileNotFoundError(f"mock dataset classes.json missing in {root}")
    train, test, val = [], [], []
    for split, bucket in (("train", train), ("test", test)):
        for c in classes:
            folder = Path(root) / split / c
            bucket.append(sorted(str(p.relative_to(root)) for p in folder.glob("*.png")))
    rng = np.random.default_rng(seed)
    alignment, rest = _carve(train, alignment_k, rng)
    val, rest = _carve(rest, real_val_k, rng)
    return DatasetIndex(
        name="mockbirds",
        root=str(root),
        classes=classes,
        train=rest,
        test=test,
        val=val,
        alignment=alignment,
        meta={"source": "procedural mock dataset", "mock": True},
    )
