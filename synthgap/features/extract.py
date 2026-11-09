"""Feature extraction with on-disk caching (Sec. 3.2).

Primary measurement embedding: DINOv2 ViT-B/14. Robustness embedding: CLIP
ViT-L/14. Selection always uses CLIP, measurement always uses DINOv2, so DAS
cannot win by optimising the space it is scored in.

The `cheap` backend is a CPU stand-in used only by the smoke test.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from ..config import Cfg, resolve_dir
from ..data.prompts import clean_class_name, clip_class_texts
from ..utils import Logger, chunked, stable_hash


def l2_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


class FeatureBackend(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, paths: Sequence[Path]) -> np.ndarray:
        """(n, d) float32 embeddings for image paths."""

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError(f"{self.name} has no text encoder")

    def close(self) -> None:  # pragma: no cover
        pass


# --------------------------------------------------------------------------------------
# real backends
# --------------------------------------------------------------------------------------
class Dinov2Backend(FeatureBackend):
    name = "dinov2"

    def __init__(self, model_id: str, device: str = "cuda", batch_size: int = 64):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.dim = int(self.model.config.hidden_size)
        self.batch_size = batch_size
        from torchvision import transforms

        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def embed(self, paths: Sequence[Path]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for batch in chunked(list(paths), self.batch_size):
                tensors = torch.stack([self.transform(Image.open(p).convert("RGB")) for p in batch])
                tensors = tensors.to(self.device)
                hidden = self.model(pixel_values=tensors).last_hidden_state[:, 0]
                out.append(hidden.float().cpu().numpy())
        return l2_normalize(np.concatenate(out, axis=0))


class ClipBackend(FeatureBackend):
    name = "clip"

    def __init__(self, model_name: str, pretrained: str, device: str = "cuda", batch_size: int = 64):
        import torch
        import open_clip

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.batch_size = batch_size
        with torch.no_grad():
            probe = self.model.encode_image(torch.zeros(1, 3, 224, 224, device=self.device))
        self.dim = int(probe.shape[-1])

    def embed(self, paths: Sequence[Path]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for batch in chunked(list(paths), self.batch_size):
                tensors = torch.stack([self.preprocess(Image.open(p).convert("RGB")) for p in batch])
                feats = self.model.encode_image(tensors.to(self.device))
                out.append(feats.float().cpu().numpy())
        return l2_normalize(np.concatenate(out, axis=0))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for batch in chunked(list(texts), self.batch_size):
                tokens = self.tokenizer(list(batch)).to(self.device)
                feats = self.model.encode_text(tokens)
                out.append(feats.float().cpu().numpy())
        return l2_normalize(np.concatenate(out, axis=0))


class InceptionBackend(FeatureBackend):
    """Inception-v3 pool3 features, for FID/KID comparability only."""

    name = "inception"
    dim = 2048

    def __init__(self, device: str = "cuda", batch_size: int = 32):
        import torch
        from pytorch_fid.inception import InceptionV3

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block], resize_input=True, normalize_input=True).to(self.device).eval()
        self.batch_size = batch_size

    def embed(self, paths: Sequence[Path]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for batch in chunked(list(paths), self.batch_size):
                arr = np.stack(
                    [
                        np.asarray(Image.open(p).convert("RGB").resize((299, 299), Image.BICUBIC), dtype=np.float32)
                        / 127.5
                        - 1.0
                        for p in batch
                    ]
                )
                tensors = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)
                feats = self.model(tensors)[0]
                out.append(feats.squeeze(-1).squeeze(-1).cpu().numpy())
        return np.concatenate(out, axis=0)


class CheapBackend(FeatureBackend):
    """Deterministic pixel projection. CPU smoke test only -- never for paper numbers."""

    name = "cheap"

    def __init__(self, dim: int = 96):
        rng = np.random.default_rng(20240101)
        self.dim = int(dim)
        self.proj = (rng.normal(size=(16 * 16 * 3, self.dim)) / np.sqrt(16 * 16 * 3)).astype(np.float32)

    def _embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        feats = []
        for image in images:
            small = image.convert("RGB").resize((16, 16), Image.BILINEAR)
            feats.append(np.asarray(small, dtype=np.float32).reshape(-1) / 255.0)
        return l2_normalize(np.stack(feats) @ self.proj)

    def embed(self, paths: Sequence[Path]) -> np.ndarray:
        return self._embed_images([Image.open(p) for p in paths])

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Stand-in for a text encoder: renders a canonical prototype per class name."""
        from ..data.mock_dataset import CLASS_SPECS, render_sample

        images: list[Image.Image] = []
        for text in texts:
            spec = None
            for candidate in CLASS_SPECS:
                if candidate["name"] in text or clean_class_name(candidate["name"]).lower() in text.lower():
                    spec = candidate
                    break
            if spec is None:
                spec = CLASS_SPECS[stable_hash(text) and int(stable_hash(text), 16) % len(CLASS_SPECS)]
            rng = np.random.default_rng(int(stable_hash(text), 16) % (2**31))
            images.append(
                render_sample(
                    rng, spec, size=64, scale_range=(0.7, 0.7), position_jitter=0.0, hue_jitter=0.0, blur=0.0, noise=0.0
                )
            )
        return self._embed_images(images)


def get_backend(name: str, cfg: Cfg, log: Logger | None = None) -> FeatureBackend:
    log = log or Logger("features")
    name = str(name)
    if name == "cheap":
        return CheapBackend(int(cfg.features.cheap_dim))
    if name == "dinov2":
        log.info(f"loading DINOv2 {cfg.features.dinov2_model}")
        return Dinov2Backend(cfg.features.dinov2_model, device=cfg.runtime.device, batch_size=int(cfg.features.batch_size))
    if name == "clip":
        log.info(f"loading CLIP {cfg.features.clip_model}/{cfg.features.clip_pretrained}")
        return ClipBackend(
            cfg.features.clip_model,
            cfg.features.clip_pretrained,
            device=cfg.runtime.device,
            batch_size=int(cfg.features.batch_size),
        )
    if name == "inception":
        return InceptionBackend(device=cfg.runtime.device)
    raise ValueError(f"unknown feature backend {name!r}")


# --------------------------------------------------------------------------------------
# caching
# --------------------------------------------------------------------------------------
def embed_paths(backend: FeatureBackend, paths: Sequence[Path], batch_size: int = 64, log: Logger | None = None) -> np.ndarray:
    if not paths:
        return np.zeros((0, backend.dim), dtype=np.float32)
    chunks = []
    iterator = chunked(list(paths), batch_size)
    for i, batch in enumerate(iterator):
        chunks.append(backend.embed(batch))
        if log and i % 20 == 0 and i:
            log.info(f"    embedded {i * batch_size}/{len(paths)}")
    return np.concatenate(chunks, axis=0)


def pool_feature_file(cfg: Cfg, pool, backend_name: str) -> Path:
    key = stable_hash(pool.dataset, pool.family, pool.regime, pool.seed, pool.n)
    return resolve_dir(cfg, "features") / f"pool__{pool.dataset}__{key}__{backend_name}.npz"


def dataset_feature_file(cfg: Cfg, dataset: str, backend_name: str, alignment_k: int) -> Path:
    key = stable_hash(dataset, alignment_k)
    return resolve_dir(cfg, "features") / f"real__{dataset}__{key}__{backend_name}.npz"


def compute_pool_features(
    cfg: Cfg, pool, backend_name: str, log: Logger | None = None, refresh: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    path = pool_feature_file(cfg, pool, backend_name)
    if path.exists() and not refresh:
        data = npz_load_cached(path)
        if len(data["y"]) == pool.n:
            return data["X"], data["y"]
    log = log or Logger("features")
    backend = get_backend(backend_name, cfg, log)
    X = embed_paths(backend, pool.paths, batch_size=int(cfg.features.batch_size), log=log)
    y = np.asarray(pool.classes, dtype=np.int64)
    save_npz(path, X=X, y=y)
    return X, y


def compute_dataset_features(
    cfg: Cfg, index, backend_name: str, log: Logger | None = None, refresh: bool = False
) -> dict[str, np.ndarray]:
    path = dataset_feature_file(cfg, index.name, backend_name, index.alignment_size)
    if path.exists() and not refresh:
        data = npz_load_cached(path)
        if len(data["y_test"]) == sum(len(c) for c in index.test):
            return data
    log = log or Logger("features")
    backend = get_backend(backend_name, cfg, log)
    out: dict[str, np.ndarray] = {}
    for split in ("train", "test", "val", "alignment"):
        paths, labels = [], []
        for c in range(index.num_classes):
            for p in index.class_paths(split, c):
                paths.append(p)
                labels.append(c)
        out[f"X_{split}"] = embed_paths(backend, paths, batch_size=int(cfg.features.batch_size), log=log)
        out[f"y_{split}"] = np.asarray(labels, dtype=np.int64)
        log.info(f"  {index.name}/{split}: {len(paths)} images -> {out[f'X_{split}'].shape}")
    save_npz(path, **out)
    return out


def compute_text_features(
    cfg: Cfg, index, backend_name: str, log: Logger | None = None, refresh: bool = False
) -> np.ndarray:
    """Class-name embeddings for the text-only scorer A_text (one per class)."""
    path = resolve_dir(cfg, "features") / f"text__{index.name}__{stable_hash(index.classes) }__{backend_name}.npz"
    if path.exists() and not refresh:
        data = npz_load_cached(path)
        if len(data["X"]) == index.num_classes:
            return data["X"]
    log = log or Logger("features")
    backend = get_backend(backend_name, cfg, log)
    per_class = []
    for c in index.classes:
        texts = clip_class_texts(index.name, c)
        per_class.append(l2_normalize(backend.embed_texts(texts)).mean(axis=0))
    X = l2_normalize(np.stack(per_class))
    save_npz(path, X=X)
    return X


# thin wrappers so the cache helpers live in one place
def npz_load_cached(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_npz(path: Path, **arrays: np.ndarray) -> Path:
    from ..utils import npz_save

    return npz_save(path, **arrays)
