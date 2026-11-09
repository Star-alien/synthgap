"""Prompt regimes P1, P2 and P3 (Sec. 3.1 of the paper).

P1  class name plus a domain hint, single template.
P2  LLM-diversified templates varying pose, background, lighting, season and
    viewpoint. The class name is always present and the templates were written
    once per domain without looking at real images.
P3  detailed captions of real training images from a VLM (oracle regime).

Prompts are *deterministic functions of the class and the sample index*, so the
same pool can be regenerated bit-for-bit.
"""
from __future__ import annotations

import re
from typing import Sequence

DOMAIN_NOUN = {
    "cub": "bird",
    "mockbirds": "bird",
    "aircraft": "aircraft",
    "ham10000": "skin lesion",
}

P1_TEMPLATE = {
    "cub": "a photo of a {cls}, a type of bird",
    "mockbirds": "a photo of a {cls}, a type of bird",
    "aircraft": "a photo of a {cls}, a type of aircraft",
    "ham10000": "a dermoscopic photo of a {cls}",
}

# Written once per domain, before any real images were inspected.
P2_TEMPLATES = {
    "cub": [
        "a photo of a {cls} perched on a branch, side view, soft morning light",
        "a close-up photograph of a {cls} in flight, wings spread, clear sky",
        "a {cls} standing on grass, front view, overcast daylight",
        "a {cls} seen from behind, blurred forest background, golden hour",
        "a {cls} on a snowy branch in winter, telephoto lens",
        "a photo of a {cls} in a desert, harsh midday light, low angle",
        "a {cls} bathing in a puddle, spring, shallow depth of field",
        "a {cls} at a feeder, autumn foliage, side profile",
        "a {cls} in dense reeds at dawn, muted colours",
        "a photo of a {cls} on a rock by the sea, windy, distant background",
        "a {cls} flying against a cloudy sky, underside visible",
        "a {cls} perched high in a canopy, backlit, strong bokeh",
        "a {cls} on a fence post, rural background, late afternoon",
        "a photo of a {cls} preening, close-up of plumage detail",
        "a {cls} walking on sand, low camera angle, bright sunlight",
        "a {cls} in a tropical forest, dappled light, vertical framing",
        "a {cls} on a wire against a plain blue sky, wide shot",
        "a photo of a {cls} on water, ripples, soft reflections",
        "a {cls} among wildflowers in summer, saturated colours",
        "a {cls} on a mossy log, rainforest, humid haze",
    ],
    "mockbirds": [
        "a photo of a {cls} perched on a branch, side view, soft morning light",
        "a close-up photograph of a {cls} in flight, clear sky",
        "a {cls} standing on grass, front view, overcast daylight",
        "a {cls} seen from behind, blurred forest background",
        "a {cls} on a snowy branch in winter, telephoto lens",
        "a {cls} at a feeder, autumn foliage, side profile",
        "a {cls} in dense reeds at dawn, muted colours",
        "a {cls} perched high in a canopy, backlit, strong bokeh",
    ],
    "aircraft": [
        "a photo of a {cls} on the runway, side view, clear sky",
        "a {cls} in flight, banking, mountain background",
        "a close-up of a {cls} cockpit and nose, ground level",
        "a {cls} on approach, landing gear down, overcast",
        "a photo of a {cls} parked at a gate, jet bridge visible",
        "a {cls} taking off, rear view, heat haze",
        "a {cls} flying over the ocean, low altitude, sunset",
        "a {cls} on static display at an airshow, crowd blurred",
        "a photo of a {cls} against a cloudy sky, underside view",
        "a {cls} taxiing on wet tarmac, reflections",
        "a {cls} at high altitude leaving contrails, wide shot",
        "a photo of a {cls} from the tower, top-down angle",
        "a {cls} inside a hangar, artificial lighting",
        "a {cls} flying in formation, distant, hazy horizon",
        "a close-up of a {cls} wing and engine, detail",
        "a {cls} landing on a snowy runway, winter",
        "a photo of a {cls} at dusk, runway lights",
        "a {cls} on a desert airfield, strong sunlight",
        "a {cls} on a grass strip, vintage setting",
        "a {cls} climbing steeply after take-off, blue sky",
    ],
    "ham10000": [
        "a dermoscopic image of a {cls}, circular field of view",
        "a close-up dermatoscopic photo of a {cls}, uniform illumination",
        "a dermoscopic view of a {cls} on pale skin, ruler marks visible",
        "a {cls} imaged with a dermatoscope, high magnification",
    ],
}

# Text-side prompts used by the CLIP zero-shot scorer A_text. Averaged over the
# ensemble, as is standard for zero-shot CLIP.
CLIP_TEMPLATES = {
    "cub": ["a photo of a {cls}", "a photo of a {cls}, a type of bird", "a {cls} bird"],
    "mockbirds": ["a photo of a {cls}", "a photo of a {cls}, a type of bird"],
    "aircraft": ["a photo of a {cls}", "a photo of a {cls}, a type of aircraft", "a {cls} aircraft"],
    "ham10000": ["a dermoscopic image of a {cls}", "a photo of a {cls} skin lesion"],
}


def clean_class_name(raw: str) -> str:
    """'001.Black_footed_Albatross' -> 'Black footed Albatross'."""
    name = str(raw)
    name = re.sub(r"^\d+[._]", "", name)
    name = name.replace("_", " ").strip()
    return name


def p1_prompt(dataset: str, class_name: str) -> str:
    template = P1_TEMPLATE.get(dataset, "a photo of a {cls}")
    return template.format(cls=clean_class_name(class_name))


def p2_prompts(dataset: str, class_name: str) -> list[str]:
    templates = P2_TEMPLATES.get(dataset) or P2_TEMPLATES["cub"]
    cls = clean_class_name(class_name)
    return [t.format(cls=cls) for t in templates]


def prompt_for(dataset: str, regime: str, class_name: str, index: int, caption: str | None = None) -> str:
    """Prompt used for the `index`-th image of a class."""
    cls = clean_class_name(class_name)
    if regime == "P1":
        return p1_prompt(dataset, class_name)
    if regime == "P2":
        templates = p2_prompts(dataset, class_name)
        return templates[index % len(templates)]
    if regime == "P3":
        if not caption:
            raise ValueError(
                "P3 needs a VLM caption for every real image; pass prompts.p3_captions "
                "pointing at a CSV produced by the captioning script."
            )
        return caption.format(cls=cls) if "{cls}" in caption else caption
    raise ValueError(f"unknown prompt regime {regime!r}")


def clip_class_texts(dataset: str, class_name: str) -> list[str]:
    templates = CLIP_TEMPLATES.get(dataset) or CLIP_TEMPLATES["cub"]
    cls = clean_class_name(class_name)
    return [t.format(cls=cls) for t in templates]


def all_class_prompts(dataset: str, regime: str, classes: Sequence[str]) -> dict[str, str]:
    """One canonical prompt per class (used for the alignment scorer and tables)."""
    return {c: prompt_for(dataset, regime, c, 0) for c in classes}
