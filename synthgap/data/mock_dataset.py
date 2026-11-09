"""Procedural stand-in "real" dataset used by the CPU smoke test.

This is *not* a benchmark. It exists so that every stage of the pipeline
(splits -> pools -> features -> metrics -> probes -> design -> plots) can be run
and checked without GPUs or dataset downloads. Each class is defined by a hue,
a body shape and a wing pattern, so class information is present in the pixels
and the cheap feature extractor can recover it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..utils import ensure_dir, json_dump

CLASS_SPECS = [
    {"name": "mock_azure_warbler", "hue": 0.58, "shape": 0, "pattern": 0},
    {"name": "mock_crimson_finch", "hue": 0.00, "shape": 1, "pattern": 1},
    {"name": "mock_ochre_sparrow", "hue": 0.10, "shape": 0, "pattern": 2},
    {"name": "mock_emerald_thrush", "hue": 0.33, "shape": 2, "pattern": 0},
    {"name": "mock_violet_wren", "hue": 0.75, "shape": 1, "pattern": 2},
    {"name": "mock_slate_heron", "hue": 0.55, "shape": 2, "pattern": 1},
    {"name": "mock_amber_plover", "hue": 0.08, "shape": 2, "pattern": 2},
    {"name": "mock_moss_creeper", "hue": 0.25, "shape": 1, "pattern": 0},
]

SHAPES = ["ellipse", "triangle", "rounded"]


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def render_sample(
    rng: np.random.Generator,
    spec: dict,
    size: int = 64,
    *,
    hue_jitter: float = 0.02,
    scale_range: tuple[float, float] = (0.55, 0.85),
    position_jitter: float = 0.25,
    blur: float = 0.0,
    noise: float = 4.0,
    background_saturation: float = 0.25,
) -> Image.Image:
    """Render one procedural "bird" of class `spec` with controllable nuisance."""
    hue = (spec["hue"] + rng.normal(0, hue_jitter)) % 1.0
    bg_hue = (hue + 0.5 + rng.normal(0, 0.08)) % 1.0
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    # smooth vertical background gradient
    top = np.array(_hsv_to_rgb(bg_hue, background_saturation, 0.85), dtype=np.float32)
    bottom = np.array(_hsv_to_rgb(bg_hue, background_saturation * 0.6, 0.45), dtype=np.float32)
    ramp = np.linspace(0, 1, size, dtype=np.float32)[:, None, None]
    canvas += top[None, None, :] * (1 - ramp) + bottom[None, None, :] * ramp

    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)

    scale = rng.uniform(*scale_range) * size
    cx = size / 2 + rng.uniform(-position_jitter, position_jitter) * size
    cy = size / 2 + rng.uniform(-position_jitter, position_jitter) * size
    body = _hsv_to_rgb(hue, 0.75, 0.85)
    accent = _hsv_to_rgb((hue + 0.06) % 1.0, 0.9, 0.55)
    dark = _hsv_to_rgb(hue, 0.85, 0.35)

    shape = SHAPES[spec["shape"]]
    half_w, half_h = scale * 0.42, scale * 0.28
    if shape == "ellipse":
        draw.ellipse([cx - half_w, cy - half_h, cx + half_w, cy + half_h], fill=body)
    elif shape == "triangle":
        draw.polygon(
            [(cx, cy - half_h * 1.2), (cx - half_w, cy + half_h), (cx + half_w, cy + half_h)],
            fill=body,
        )
    else:
        draw.rounded_rectangle([cx - half_w, cy - half_h, cx + half_w, cy + half_h], radius=int(scale * 0.12), fill=body)

    # head + beak give the class a consistent, discriminative silhouette
    draw.ellipse(
        [cx + half_w * 0.55, cy - half_h * 1.05, cx + half_w * 1.15, cy - half_h * 0.35],
        fill=accent,
    )
    draw.polygon(
        [
            (cx + half_w * 1.15, cy - half_h * 0.75),
            (cx + half_w * 1.45, cy - half_h * 0.62),
            (cx + half_w * 1.12, cy - half_h * 0.5),
        ],
        fill=dark,
    )

    pattern = spec["pattern"]
    if pattern == 0:  # wing bars
        for offset in (-0.35, 0.0, 0.35):
            y = cy + offset * half_h
            draw.line([(cx - half_w * 0.8, y), (cx + half_w * 0.4, y)], fill=accent, width=max(1, int(scale * 0.05)))
    elif pattern == 1:  # spots
        for _ in range(4):
            sx = cx + rng.uniform(-0.6, 0.3) * half_w
            sy = cy + rng.uniform(-0.7, 0.7) * half_h
            r = scale * rng.uniform(0.03, 0.06)
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=dark)
    else:  # solid breast patch
        draw.ellipse(
            [cx - half_w * 0.45, cy - half_h * 0.2, cx + half_w * 0.25, cy + half_h * 0.55],
            fill=accent,
        )

    if blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur))
    if noise > 0:
        arr = np.asarray(image, dtype=np.float32) + rng.normal(0, noise, (size, size, 3))
        image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return image


def build_mock_dataset(
    root: str | Path,
    *,
    n_train: int = 26,
    n_test: int = 10,
    size: int = 64,
    seed: int = 0,
) -> Path:
    """Create (once) the procedural dataset and return its root."""
    root = Path(root)
    stamp = root / "classes.json"
    if stamp.exists():
        return root
    ensure_dir(root)
    classes = [spec["name"] for spec in CLASS_SPECS]
    json_dump(classes, stamp)
    for split, n in (("train", n_train), ("test", n_test)):
        for i, spec in enumerate(CLASS_SPECS):
            folder = ensure_dir(root / split / spec["name"])
            rng = np.random.default_rng(seed * 100003 + i * 977 + (0 if split == "train" else 50000))
            for j in range(n):
                img = render_sample(rng, spec, size=size)
                img.save(folder / f"{j:04d}.png")
    return root
