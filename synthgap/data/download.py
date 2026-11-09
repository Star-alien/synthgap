"""Dataset acquisition.

CUB-200-2011 and FGVC-Aircraft are public downloads. HAM10000 lives behind a
Kaggle login and is therefore *not* auto-downloaded: see docs/DATA.md.
"""
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

from ..config import Cfg, resolve_dir
from ..utils import Logger, human_duration

CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
AIRCRAFT_URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"


def _download(url: str, dest: Path, log: Logger) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info(f"already downloaded: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
        return dest
    log.info(f"downloading {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    start = __import__("time").time()

    def hook(blocks: int, block_size: int, total: int) -> None:
        if total > 0 and blocks % 200 == 0:
            done = blocks * block_size
            log.info(f"  {done / 1e9:.2f}/{total / 1e9:.2f} GB")

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    tmp.rename(dest)
    log.info(f"finished in {human_duration(__import__('time').time() - start)}")
    return dest


def _extract(archive: Path, dest: Path, log: Logger) -> Path:
    marker = dest / (archive.name.replace(".tgz", "").replace(".tar.gz", ""))
    if marker.exists():
        log.info(f"already extracted: {marker}")
        return marker
    log.info(f"extracting {archive} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest)
    return marker


def download_cub(raw: Path, log: Logger) -> Path:
    archive = _download(CUB_URL, raw / "CUB_200_2011.tgz", log)
    return _extract(archive, raw, log)


def download_aircraft(raw: Path, log: Logger) -> Path:
    archive = _download(AIRCRAFT_URL, raw / "fgvc-aircraft-2013b.tar.gz", log)
    return _extract(archive, raw, log)


def download_all(cfg: Cfg, log: Logger | None = None) -> None:
    log = log or Logger("download")
    raw = resolve_dir(cfg, "raw")
    name = cfg.dataset.name
    if name in {"cub", "mockbirds", "mock"}:
        if name == "cub":
            download_cub(raw, log)
        else:
            from .mock_dataset import build_mock_dataset

            build_mock_dataset(raw / "mockbirds", seed=int(cfg.dataset.index_seed))
            log.info("procedural mock dataset ready")
    elif name == "aircraft":
        download_aircraft(raw, log)
    elif name == "ham10000":
        log.warn(
            "HAM10000 cannot be downloaded automatically (Kaggle login required). "
            "See docs/DATA.md; the main experiments do not need it."
        )
    else:
        raise ValueError(f"unknown dataset {name!r}")
