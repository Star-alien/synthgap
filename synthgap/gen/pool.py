"""Pool construction: P_{G,pi} holds m*N images per class for a fixed (G, pi) (Sec. 3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from ..config import Cfg, resolve_dir
from ..data.datasets import DatasetIndex
from ..data.prompts import p2_prompts, prompt_for
from ..utils import Logger, ensure_dir, json_dump, json_load, slugify
from .base import FAMILIES, build_backend


@dataclass
class Pool:
    dataset: str
    family: str
    regime: str
    seed: int
    root: str
    paths: list[str]
    classes: list[int]
    prompts: list[str]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (len(self.paths) == len(self.classes) == len(self.prompts)):
            raise ValueError(
                f"pool arrays disagree: {len(self.paths)} paths, {len(self.classes)} labels, "
                f"{len(self.prompts)} prompts"
            )

    @property
    def n(self) -> int:
        return len(self.paths)

    @property
    def n_classes(self) -> int:
        return len(set(self.classes))

    @property
    def per_class(self) -> int:
        counts = np.bincount(np.asarray(self.classes, dtype=int))
        return int(counts[counts > 0].min())

    def __repr__(self) -> str:
        return f"Pool({self.dataset}/{self.family}/{self.regime}/s{self.seed}, n={self.n})"

    def manifest(self) -> dict:
        return {
            "dataset": self.dataset,
            "family": self.family,
            "regime": self.regime,
            "seed": self.seed,
            "root": self.root,
            "paths": self.paths,
            "classes": self.classes,
            "prompts": self.prompts,
            "meta": self.meta,
        }

    @classmethod
    def from_manifest(cls, data: dict) -> "Pool":
        return cls(**data)


def pool_dir(cfg: Cfg, dataset: str, family: str, regime: str, seed: int) -> Path:
    return resolve_dir(cfg, "pools") / dataset / slugify(family) / slugify(regime) / f"seed{seed}"


def _class_prompts(
    cfg: Cfg,
    index: DatasetIndex,
    c: int,
    regime: str,
    total: int,
    captions: dict[str, str] | None,
) -> list[str]:
    """`total` prompts for class c under the given regime."""
    class_name = index.classes[c]
    if regime == "P1":
        return [prompt_for(index.name, "P1", class_name, 0)] * total
    if regime == "P2":
        templates = p2_prompts(index.name, class_name)
        return [templates[i % len(templates)] for i in range(total)]
    if regime == "P3":
        if not captions:
            raise ValueError("P3 needs a caption CSV; set prompts.p3_captions")
        real_images = index.train[c]
        texts = []
        for rel in real_images:
            caption = captions.get(Path(rel).name) or captions.get(rel)
            if caption:
                texts.append(prompt_for(index.name, "P3", class_name, 0, caption=caption))
        if not texts:
            raise ValueError(f"no captions found for class {class_name!r} in prompts.p3_captions")
        return [texts[i % len(texts)] for i in range(total)]
    raise ValueError(f"unknown prompt regime {regime!r}")


def load_captions(cfg: Cfg) -> dict[str, str] | None:
    path = cfg.get("prompts.p3_captions")
    if not path:
        return None
    import csv

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = resolve_dir(cfg, "captions") / resolved.name
    if not resolved.exists():
        raise FileNotFoundError(f"caption file not found: {resolved}")
    with open(resolved, newline="") as handle:
        return {row["image"]: row["caption"] for row in csv.DictReader(handle)}


def build_pool(
    cfg: Cfg,
    index: DatasetIndex,
    family: str,
    regime: str,
    seed: int,
    log: Logger | None = None,
    overwrite: bool = False,
) -> Pool:
    """Generate (or load) the pool for one (generator, prompt regime, seed)."""
    log = log or Logger("pool")
    directory = pool_dir(cfg, index.name, family, regime, seed)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and not overwrite:
        return Pool.from_manifest(json_load(manifest_path))

    ensure_dir(directory)
    multiplier = int(cfg.generation.images_per_class_multiplier)
    per_class = index.per_class_train
    total_per_class = multiplier * per_class
    captions = load_captions(cfg) if regime == "P3" else None

    backend = build_backend(family, cfg, log=log)
    suffix = str(cfg.get("generation.save_format", "png")).lower().lstrip(".")
    batch = int(cfg.generation.batch_size)
    paths: list[str] = []
    classes: list[int] = []
    prompts: list[str] = []
    log.section(f"generating {index.name}/{family}/{regime}/seed{seed}: {total_per_class}/class x {index.num_classes}")

    for c in range(index.num_classes):
        folder = ensure_dir(directory / f"{c:04d}")
        class_prompts = _class_prompts(cfg, index, c, regime, total_per_class, captions)
        written = 0
        for chunk_idx in range(0, len(class_prompts), batch):
            chunk = class_prompts[chunk_idx : chunk_idx + batch]
            images = backend.generate(chunk, seed=int(seed) * 100_003 + chunk_idx, images_per_prompt=1)
            for offset, image in enumerate(images):
                path = folder / f"{chunk_idx + offset:05d}.{suffix}"
                if suffix in {"jpg", "jpeg"}:
                    image.convert("RGB").save(path, quality=95)
                else:
                    image.save(path)
                paths.append(str(path))
                classes.append(c)
                prompts.append(chunk[offset % len(chunk)])
                written += 1
        log.info(f"  class {c + 1}/{index.num_classes} ({index.classes[c]}): {written} images")

    pool = Pool(
        dataset=index.name,
        family=family,
        regime=regime,
        seed=int(seed),
        root=str(directory),
        paths=paths,
        classes=classes,
        prompts=prompts,
        meta={
            "model_id": FAMILIES[family].model_id if family in FAMILIES else family,
            "native_resolution": FAMILIES[family].native_resolution if family in FAMILIES else None,
            "multiplier": multiplier,
            "per_class": total_per_class,
            "steps": int(cfg.generation.steps) if not cfg.runtime.mock else 0,
            "guidance": float(cfg.generation.guidance),
            "mock": bool(cfg.runtime.mock),
        },
    )
    json_dump(pool.manifest(), manifest_path)
    return pool


def load_pool(cfg: Cfg, dataset: str, family: str, regime: str, seed: int) -> Pool | None:
    manifest_path = pool_dir(cfg, dataset, family, regime, seed) / "manifest.json"
    if not manifest_path.exists():
        return None
    return Pool.from_manifest(json_load(manifest_path))


def build_vae_ceiling_pool(
    cfg: Cfg,
    index: DatasetIndex,
    family: str,
    seed: int = 0,
    log: Logger | None = None,
    overwrite: bool = False,
) -> Pool:
    """Autoencoder ceiling: real training images pushed through the generator's VAE.

    Bounds how much of the synthetic-real gap is explained by latent-space
    artifacts alone, at zero diffusion cost (Sec. 3.3).
    """
    log = log or Logger("vae")
    tag = f"vae.{family}"
    directory = pool_dir(cfg, index.name, tag, "VAE", seed)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and not overwrite:
        return Pool.from_manifest(json_load(manifest_path))

    ensure_dir(directory)
    backend = build_backend(family, cfg, log=log)
    suffix = str(cfg.get("generation.save_format", "png")).lower().lstrip(".")
    resolution = FAMILIES[family].native_resolution if family in FAMILIES else 512
    paths, classes, prompts = [], [], []
    for c in range(index.num_classes):
        folder = ensure_dir(directory / f"{c:04d}")
        real = index.class_paths("train", c)[: index.per_class_train]
        recon = backend.reconstruct([Image.open(p).convert("RGB") for p in real])
        for i, image in enumerate(recon):
            out = folder / f"{i:05d}.{suffix}"
            if not cfg.runtime.mock and image.size != (resolution, resolution):
                image = image.resize((resolution, resolution), Image.BICUBIC)
            if suffix in {"jpg", "jpeg"}:
                image.convert("RGB").save(out, quality=95)
            else:
                image.save(out)
            paths.append(str(out))
            classes.append(c)
            prompts.append(f"vae reconstruction of a real {index.classes[c]}")
    pool = Pool(
        dataset=index.name,
        family=tag,
        regime="VAE",
        seed=int(seed),
        root=str(directory),
        paths=paths,
        classes=classes,
        prompts=prompts,
        meta={"model_id": f"{FAMILIES[family].model_id} VAE" if family in FAMILIES else family, "ceiling": True},
    )
    json_dump(pool.manifest(), manifest_path)
    log.info(f"VAE ceiling pool for {family}: {pool.n} images")
    return pool
