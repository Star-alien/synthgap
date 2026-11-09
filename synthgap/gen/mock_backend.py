"""Procedural generator used by the CPU smoke test.

Each "family" renders the same procedural bird classes with a different mixture
of (i) coverage/diversity of pose and framing, (ii) label-alignment error rate
and (iii) per-sample fidelity (blur/noise). The profile ordering mirrors the
qualitative finding the paper studies -- newer, prettier, narrower -- so that
the analysis code is exercised on non-degenerate variation.

This is a *test double*, never a stand-in for a real diffusion model.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

from ..config import Cfg
from ..data.mock_dataset import CLASS_SPECS, render_sample
from ..data.prompts import clean_class_name
from ..utils import stable_hash
from .base import FAMILIES, GeneratorBackend, GenSpec


class MockGenerator(GeneratorBackend):
    PROFILES = {
        # alignment error, pose diversity, blur radius, pixel noise, bg saturation
        "sd15": dict(align_err=0.05, diversity=1.00, blur=0.8, noise=7.0, bg_sat=0.30),
        "sdxl": dict(align_err=0.10, diversity=0.62, blur=0.45, noise=5.0, bg_sat=0.40),
        "sd35": dict(align_err=0.15, diversity=0.35, blur=0.30, noise=3.5, bg_sat=0.50),
        "flux": dict(align_err=0.08, diversity=0.18, blur=0.00, noise=1.5, bg_sat=0.62),
    }

    def __init__(self, family: str, cfg: Cfg):
        self.spec = FAMILIES.get(family, GenSpec(family, "mock", 512))
        self.family = family if family in self.PROFILES else "sd15"
        self.profile = self.PROFILES[self.family]
        self.resolution = int(self.spec.native_resolution)

    # ------------------------------------------------------------------ helpers
    def _class_from_prompt(self, prompt: str) -> dict | None:
        for spec in CLASS_SPECS:
            if spec["name"] in prompt or clean_class_name(spec["name"]).lower() in prompt.lower():
                return spec
        return None

    def generate(self, prompts: Sequence[str], seed: int, images_per_prompt: int = 1) -> list[Image.Image]:
        rng = np.random.default_rng([int(seed), int(stable_hash(*prompts)[:8], 16) % (2**31)])
        profile = self.profile
        shape = int(self.resolution / 8)  # cheap: 64px for 512, 128px for 1024
        images: list[Image.Image] = []
        for prompt in prompts:
            target = self._class_from_prompt(prompt)
            for rep in range(images_per_prompt):
                spec = target
                if target is not None and rng.random() < profile["align_err"]:
                    others = [s for s in CLASS_SPECS if s is not target]
                    spec = others[int(rng.integers(len(others)))]
                if spec is None:
                    spec = CLASS_SPECS[int(rng.integers(len(CLASS_SPECS)))]
                diversity = float(profile["diversity"])
                low = 0.55 - 0.30 * diversity
                high = 0.85 - 0.10 * diversity
                images.append(
                    render_sample(
                        rng,
                        spec,
                        size=shape,
                        scale_range=(low, high),
                        position_jitter=0.08 + 0.30 * diversity,
                        hue_jitter=0.01 + 0.03 * diversity,
                        blur=float(profile["blur"]),
                        noise=float(profile["noise"]),
                        background_saturation=float(profile["bg_sat"]),
                    )
                )
        return images

    def reconstruct(self, images: Sequence[Image.Image]) -> list[Image.Image]:
        """Mock VAE round trip: a small identity-preserving degradation."""
        from PIL import ImageFilter

        return [im.filter(ImageFilter.GaussianBlur(0.6)) for im in images]
