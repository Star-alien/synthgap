"""Generator backends and the family registry (Sec. 3.1)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from PIL import Image

from ..config import Cfg


@dataclass(frozen=True)
class GenSpec:
    family: str
    model_id: str
    native_resolution: int
    steps: int = 50
    guidance: float = 2.0
    guidance_distilled: bool = False
    note: str = ""


FAMILIES: dict[str, GenSpec] = {
    "sd15": GenSpec(
        family="sd15",
        model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        native_resolution=512,
        note="CFG 2.0, 50 steps (low guidance favours diversity)",
    ),
    "sdxl": GenSpec(
        family="sdxl",
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        native_resolution=1024,
        note="native 1024^2",
    ),
    "sd35": GenSpec(
        family="sd35",
        model_id="stabilityai/stable-diffusion-3.5-large",
        native_resolution=1024,
        note="swap to stable-diffusion-3.5-medium if VRAM is tight",
    ),
    "flux": GenSpec(
        family="flux",
        model_id="black-forest-labs/FLUX.1-dev",
        native_resolution=1024,
        guidance_distilled=True,
        note="guidance-distilled: guidance embedding, not true CFG",
    ),
}


class GeneratorBackend(ABC):
    """Produces images for prompts, and (for the ceiling control) re-renders real images."""

    spec: GenSpec

    @abstractmethod
    def generate(self, prompts: Sequence[str], seed: int, images_per_prompt: int = 1) -> list[Image.Image]:
        ...

    @abstractmethod
    def reconstruct(self, images: Sequence[Image.Image]) -> list[Image.Image]:
        """Encode+decode real images through the generator's VAE (no diffusion steps)."""

    def close(self) -> None:  # pragma: no cover - backend specific
        pass


def build_backend(family: str, cfg: Cfg, log=None) -> GeneratorBackend:
    if cfg.runtime.mock:
        from .mock_backend import MockGenerator

        return MockGenerator(family, cfg)
    from .diffusers_backend import DiffusersBackend

    return DiffusersBackend(family, cfg, log=log)
