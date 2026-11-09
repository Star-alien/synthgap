"""Diffusers backends: SD1.5, SDXL, SD3.5 and FLUX.1-dev (Sec. 3.1).

Sampling settings follow the paper: 50 steps, classifier-free guidance 2.0 where
true CFG exists, FLUX uses its distilled guidance embedding instead. SDXL and
SD3.5/FLUX are generated at their native 1024^2, SD1.5 at 512^2; everything is
resized to 224^2 only at feature-extraction time.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

from ..config import Cfg
from ..utils import Logger, chunked
from .base import FAMILIES, GeneratorBackend, GenSpec

_PIPELINE_CLASS = {
    "sd15": "StableDiffusionPipeline",
    "sdxl": "StableDiffusionXLPipeline",
    "sd35": "StableDiffusion3Pipeline",
    "flux": "FluxPipeline",
}


class DiffusersBackend(GeneratorBackend):
    def __init__(self, family: str, cfg: Cfg, log: Logger | None = None):
        try:
            import torch
            import diffusers  # noqa: F401
        except ImportError as exc:  # pragma: no cover - GPU-only path
            raise ImportError(
                "The real generators need the GPU stack: pip install -r requirements-gpu.txt"
            ) from exc

        self.torch = torch
        self.cfg = cfg
        self.log = log or Logger("generator")
        self.spec: GenSpec = FAMILIES[family]
        self.device = cfg.runtime.device if torch.cuda.is_available() else "cpu"
        self.dtype = getattr(torch, cfg.runtime.dtype) if self.device == "cuda" else torch.float32
        self.pipe = self._load()

    # -- loading -------------------------------------------------------------------
    def _load(self):  # pragma: no cover - requires weights
        import diffusers

        cls = getattr(diffusers, _PIPELINE_CLASS[self.spec.family])
        self.log.info(f"loading {self.spec.model_id} ({self.spec.family}) on {self.device}")
        pipe = cls.from_pretrained(self.spec.model_id, torch_dtype=self.dtype, safety_checker=None)
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        if self.device == "cuda" and self.cfg.get("generation.low_vram", False):
            pipe.enable_attention_slicing()
            pipe.enable_model_cpu_offload()
        if self.spec.family == "flux":
            self.log.warn(
                "FLUX.1-dev is guidance-distilled; generation.guidance is used as the "
                "guidance embedding, not as true CFG."
            )
        return pipe

    # -- generation ----------------------------------------------------------------
    def generate(self, prompts: Sequence[str], seed: int, images_per_prompt: int = 1) -> list[Image.Image]:
        torch = self.torch
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        kwargs = dict(
            prompt=list(prompts),
            num_inference_steps=int(self.cfg.generation.steps),
            num_images_per_prompt=int(images_per_prompt),
            generator=generator,
        )
        if self.spec.guidance_distilled:
            kwargs["guidance_scale"] = float(self.cfg.get("generation.guidance_embed", 2.0))
        else:
            kwargs["guidance_scale"] = float(self.cfg.generation.guidance)
        output = self.pipe(**kwargs)
        return list(output.images)

    # -- autoencoder ceiling -------------------------------------------------------
    def reconstruct(self, images: Sequence[Image.Image]) -> list[Image.Image]:  # pragma: no cover
        torch = self.torch
        vae = self.pipe.vae
        vae.to(self.device)
        scaling = float(getattr(vae.config, "scaling_factor", 0.18215))
        shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
        resolution = self.spec.native_resolution
        out: list[Image.Image] = []
        with torch.no_grad():
            for batch in chunked(list(images), 8):
                array = np.stack(
                    [
                        np.asarray(im.convert("RGB").resize((resolution, resolution), Image.BICUBIC), dtype=np.float32)
                        / 127.5
                        - 1.0
                        for im in batch
                    ]
                )
                x = torch.from_numpy(array).permute(0, 3, 1, 2).to(self.device, self.dtype)
                moments = vae.encode(x)
                latents = moments.latent_dist.mode() if hasattr(moments, "latent_dist") else moments
                latents = (latents - shift) * scaling
                decoded = vae.decode(latents / scaling + shift).sample
                decoded = (decoded.float().clamp(-1, 1) + 1) * 127.5
                for tensor in decoded:
                    arr = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    out.append(Image.fromarray(arr))
        return out
