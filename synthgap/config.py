"""Layered YAML configuration with attribute access and deep merge."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .utils import REPO_ROOT, stable_hash

_DEFAULT_CONFIG = REPO_ROOT / "configs" / "base.yaml"


def _wrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return Cfg(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


class Cfg:
    """Read-only, attribute-access view over a nested dict.

    >>> cfg.dataset.name
    >>> cfg.learner.weight_decay_grid[0]
    >>> cfg.get("missing.key", 3)
    """

    __slots__ = ("_d",)

    def __init__(self, data: Mapping[str, Any] | None = None):
        object.__setattr__(self, "_d", dict(data or {}))

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_d")
        if name in data:
            return _wrap(data[name])
        raise AttributeError(f"config has no key {name!r}")

    def __getitem__(self, key: str) -> Any:
        return _wrap(self._d[key])

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def __iter__(self):
        return iter(self._d)

    def __repr__(self) -> str:
        return f"Cfg({self._d!r})"

    def get(self, dotted: str, default: Any = None) -> Any:
        """Dotted lookup with a default, e.g. ``cfg.get('metrics.knn_k', 5)``."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Cfg) or part not in node._d:  # noqa: SLF001
                return default
            node = _wrap(node._d[part])  # noqa: SLF001
        return node

    def keys(self):
        return self._d.keys()

    def items(self):
        return self._d.items()

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._d)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(*paths: str | Path) -> Cfg:
    """Load `base.yaml` then deep-merge each given config file over it."""
    files: list[Path] = [_DEFAULT_CONFIG]
    for path in paths:
        if path is None:
            continue
        candidate = Path(path)
        if not candidate.is_absolute() and not candidate.exists():
            candidate = REPO_ROOT / candidate
        files.append(candidate)
    merged: dict[str, Any] = {}
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        with open(path) as handle:
            data = yaml.safe_load(handle) or {}
        merged = deep_merge(merged, data)
    return Cfg(merged)


def config_hash(cfg: Cfg, *extra: Any) -> str:
    return stable_hash(cfg.to_dict(), *extra)


def resolve_dir(cfg: Cfg, key: str) -> Path:
    """Resolve a path from cfg['paths'][key] against the repo root and create it."""
    rel = cfg.get(f"paths.{key}")
    if rel is None:
        raise KeyError(f"paths.{key} not configured")
    path = Path(rel)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path
