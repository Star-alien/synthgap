"""Shared helpers: seeding, hashing, atomic IO, logging, bootstrap CIs."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed every RNG we touch. torch is seeded only if already imported."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def stable_hash(*parts: Any, length: int = 16) -> str:
    """Deterministic short hash over arbitrary repr-able parts (cache keys, run ids)."""
    h = hashlib.sha1()
    for part in parts:
        h.update(repr(part).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:length]


# --------------------------------------------------------------------------------------
# filesystem / IO
# --------------------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def json_dump(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Atomically write JSON (temp file + rename, so interrupted runs never leave half files)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as handle:
        json.dump(obj, handle, indent=indent, default=_json_default)
    os.replace(tmp, path)
    return path


def json_load(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path) as handle:
        return json.load(handle)


def npz_save(path: str | Path, **arrays: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npz")
    os.close(fd)
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path


def npz_load(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-").lower()


def human_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


class Logger:
    """Minimal logger: timestamped lines to stdout and (optionally) a file."""

    def __init__(self, name: str = "synthgap", logfile: str | Path | None = None):
        self.name = name
        self.logfile = Path(logfile) if logfile else None
        self.t0 = time.time()
        if self.logfile:
            self.logfile.parent.mkdir(parents=True, exist_ok=True)

    def _emit(self, level: str, message: str) -> None:
        line = f"[{time.time() - self.t0:8.1f}s] {level:<5} {message}"
        print(line, flush=True)
        if self.logfile:
            with open(self.logfile, "a") as handle:
                handle.write(line + "\n")

    def info(self, message: str) -> None:
        self._emit("INFO", message)

    def warn(self, message: str) -> None:
        self._emit("WARN", message)

    def error(self, message: str) -> None:
        self._emit("ERROR", message)

    def section(self, title: str) -> None:
        self._emit("----", f"== {title} " + "=" * max(0, 60 - len(title)))

    @contextlib.contextmanager
    def timed(self, what: str):
        start = time.time()
        self.info(f"{what} ...")
        try:
            yield
        finally:
            self.info(f"{what} done in {human_duration(time.time() - start)}")


# --------------------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------------------
def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    statistic=np.mean,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of `statistic` over i.i.d. units. Returns (lo, hi, point)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(statistic(values))
    if values.size == 1:
        return (point, point, point)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boots = statistic(values[idx], axis=1) if _supports_axis(statistic) else np.array(
        [statistic(values[i]) for i in idx]
    )
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi), point)


def _supports_axis(statistic) -> bool:
    try:
        statistic(np.zeros((2, 3)), axis=1)
        return True
    except TypeError:
        return False


def fraction_of_gap_closed(
    acc: np.ndarray | float,
    acc_random: np.ndarray | float,
    acc_real: np.ndarray | float,
    min_denominator: float = 1e-6,
) -> np.ndarray | float:
    """Eq. (2): (Acc(method) - Acc(Random-N)) / (Acc(Real-N) - Acc(Random-N)).

    Undefined (nan) when the denominator is near zero, as the paper commits to.
    """
    acc = np.asarray(acc, dtype=float)
    acc_random = np.asarray(acc_random, dtype=float)
    denominator = np.asarray(acc_real, dtype=float) - acc_random
    out = np.full(np.broadcast(acc, acc_random, denominator).shape, np.nan)
    valid = np.abs(denominator) >= min_denominator
    np.divide(acc - acc_random, denominator, out=out, where=valid)
    if out.ndim == 0:
        return float(out)
    return out


def format_pm(value: float, lo: float, hi: float, digits: int = 1) -> str:
    """Format `value` with a CI in the paper's style, e.g. ``57.3 [56.9, 57.7]``."""
    if not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
