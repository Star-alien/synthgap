"""Command line interface: one subcommand per stage.

    python -m synthgap.cli measure --config configs/exp/main.yaml
    python -m synthgap.cli all --config configs/exp/smoke.yaml
    python -m synthgap.cli factorial --config configs/exp/main.yaml --config configs/exp/factorial.yaml
"""
from __future__ import annotations

import argparse

from .config import Cfg, load_config
from .utils import Logger

STAGES = [
    "download",
    "generate",
    "measure",
    "factorial",
    "subsets",
    "das",
    "mixing",
    "ablation",
    "scratch",
    "h3",
    "figures",
    "report",
]


def _apply_overrides(data: dict, overrides: list[str]) -> dict:
    import yaml

    for item in overrides:
        key, _, value = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthgap",
        description="Interventional study of synthetic training data (CVPR submission code).",
    )
    parser.add_argument("stage", choices=STAGES + ["all"])
    parser.add_argument("--config", action="append", default=[], help="YAML config, merged in order")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="dotted override, e.g. generation.seeds=[0]")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(*args.config)
    if args.overrides:
        cfg = Cfg(_apply_overrides(cfg.to_dict(), args.overrides))
    log = Logger("synthgap", f"{cfg.paths.results}/run.log")

    if args.stage == "download":
        from .data.download import download_all

        download_all(cfg, log)
        return 0

    from . import pipeline

    study = pipeline.Study(cfg, log)

    if args.stage == "generate":
        for family in study.families():
            for regime in study.regimes():
                for seed in study.pool_seeds():
                    study.pool(family, regime, seed)
            study.vae_pool(family)
        return 0

    if args.stage == "measure":
        pipeline.stage_measure(study)
    elif args.stage == "factorial":
        pipeline.stage_factorial(study)
    elif args.stage == "subsets":
        pipeline.stage_subsets(study)
    elif args.stage == "das":
        pipeline.stage_das(study)
    elif args.stage == "mixing":
        pipeline.stage_mixing(study)
    elif args.stage == "ablation":
        pipeline.stage_ablation(study)
    elif args.stage == "scratch":
        pipeline.stage_scratch(study)
    elif args.stage == "h3":
        pipeline.stage_h3(study)
    elif args.stage == "figures":
        from .plots.figures import make_all_figures

        make_all_figures(cfg, log)
    elif args.stage == "report":
        from .report import build_report

        build_report(cfg, log)
    elif args.stage == "all":
        for family in study.families():
            for regime in study.regimes():
                for seed in study.pool_seeds():
                    study.pool(family, regime, seed)
            study.vae_pool(family)
        pipeline.stage_measure(study)
        pipeline.stage_factorial(study)
        pipeline.stage_subsets(study)
        pipeline.stage_das(study)
        pipeline.stage_mixing(study)
        pipeline.stage_ablation(study)
        pipeline.stage_scratch(study)
        pipeline.stage_h3(study)
        from .plots.figures import make_all_figures
        from .report import build_report

        make_all_figures(cfg, log)
        build_report(cfg, log)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
