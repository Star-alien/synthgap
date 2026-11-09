"""Assemble tables and an auto-filled results document from the stage CSVs.

The report is generated from whatever stages have run. If ``runtime.mock`` is set
the document carries an explicit banner, because mock numbers are pipeline
checks, not paper results.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Cfg
from .utils import Logger, bootstrap_ci, ensure_dir, fraction_of_gap_closed


def _load(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if Path(path).exists() else None


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def _table1(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, regime), group in frame.groupby(["family", "regime"], dropna=False):
        accuracy = group.accuracy.to_numpy(dtype=float)
        lo, hi, mean = bootstrap_ci(accuracy, n_boot=1000, seed=0)
        rows.append(
            {
                "training data": family if family == "real" else f"{family} ({regime})",
                "acc (%)": f"{mean * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}]",
                "Vendi": f"{group.vendi.mean():.1f}" if group.vendi.notna().any() else "--",
                "coverage": f"{group.coverage.mean():.3f}" if group.coverage.notna().any() else "--",
                "precision": f"{group.precision.mean():.3f}" if group.precision.notna().any() else "--",
                "A_real": f"{group.align_real.mean():.3f}" if group.align_real.notna().any() else "--",
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _table2(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in frame.groupby("family"):
        for _, row in group.iterrows():
            delta = row.get("delta", np.nan)
            rows.append(
                {
                    "generator": family,
                    "method": row["method"],
                    "images": row.get("n_images", ""),
                    "acc (%)": f"{row.accuracy * 100:.1f}",
                    "delta": "--" if not np.isfinite(delta) else f"{delta:.2f}",
                }
            )
    return pd.DataFrame(rows)


def build_report(cfg: Cfg, log: Logger | None = None) -> Path:
    log = log or Logger("report")
    results = Path(cfg.paths.results)
    tables_dir = ensure_dir(results / "tables")
    runs = results / "runs"
    dataset = cfg.dataset.name
    banner = ""
    if bool(cfg.runtime.mock):
        banner = (
            f"> **{cfg.get('report.smoke_banner', 'SMOKE-TEST OUTPUT - NOT PAPER RESULTS')}**\n"
            ">\n"
            "> These numbers come from procedurally generated stand-in images and a cheap CPU\n"
            "> feature extractor. They prove the pipeline runs end to end; they are not results\n"
            "> for the paper. Re-run with a GPU config to obtain the real numbers.\n\n"
        )

    sections: list[str] = [f"# Results: {cfg.get('report.title')}\n", banner]

    measure = _load(runs / "measure" / "table1.csv")
    if measure is not None:
        table = _table1(measure)
        table.to_csv(tables_dir / f"table1_{dataset}.csv", index=False)
        (tables_dir / f"table1_{dataset}.md").write_text(_markdown_table(table) + "\n")
        sections.append(f"## Table 1 - cross-generator picture ({dataset})\n\n" + _markdown_table(table) + "\n")
        log.info("wrote Table 1")

    factorial = _load(runs / "factorial" / "effects.csv")
    if factorial is not None:
        rows = []
        for _, row in factorial.iterrows():
            rows.append(
                {
                    "generator": row["family"],
                    "quantity": row["quantity"],
                    "estimate (pp)": f"{row['mean'] * 100:+.2f} [{row['lo'] * 100:+.2f}, {row['hi'] * 100:+.2f}]",
                }
            )
        table = pd.DataFrame(rows)
        table.to_csv(tables_dir / f"factorial_{dataset}.csv", index=False)
        (tables_dir / f"factorial_{dataset}.md").write_text(_markdown_table(table) + "\n")
        sections.append("## Factorial main effects and interaction (pp)\n\n" + _markdown_table(table) + "\n")

    shapley = _load(runs / "subsets" / "shapley.csv")
    if shapley is not None:
        wide = shapley.pivot_table(index=["dataset", "model", "n"], columns="factor", values="share").reset_index()
        wide.columns = [str(c) for c in wide.columns]
        wide["R2"] = shapley.groupby(["dataset", "model"]).r2.first().values
        wide = wide.round(3)
        wide.to_csv(tables_dir / f"shapley_{dataset}.csv", index=False)
        (tables_dir / f"shapley_{dataset}.md").write_text(_markdown_table(wide) + "\n")
        sections.append("## Shapley share of R^2 for D, C, A, F\n\n" + _markdown_table(wide) + "\n")

    das = _load(runs / "das" / "table2.csv")
    if das is not None:
        table = _table2(das)
        table.to_csv(tables_dir / f"table2_{dataset}.csv", index=False)
        (tables_dir / f"table2_{dataset}.md").write_text(_markdown_table(table) + "\n")
        sections.append(f"## Table 2 - DAS versus baselines ({dataset})\n\n" + _markdown_table(table) + "\n")

    scratch = _load(runs / "scratch" / "scratch.csv")
    if scratch is not None and len(scratch) > 2:
        from scipy.stats import spearmanr

        rho, p = spearmanr(scratch.probe_accuracy, scratch.scratch_accuracy)
        sections.append(
            "## From-scratch control\n\n"
            f"Spearman(probe, ResNet) = {rho:.2f} (p = {p:.3f}, n = {len(scratch)} sets).\n"
        )

    mixing = _load(runs / "mixing" / "mixing.csv")
    if mixing is not None and len(mixing):
        rows = []
        for family, group in mixing.groupby("family"):
            for synthetic, sub in group.groupby("synthetic"):
                sub = sub.copy()
                sub["order"] = sub.real_per_class.astype(str).map({"0": 0, "5": 5, "10": 10, "all": 30})
                sub = sub.sort_values("order")
                best = sub.loc[sub.accuracy.idxmax()]
                rows.append(
                    {
                        "generator": family,
                        "synthetic": synthetic,
                        "peak real/class": best.real_per_class,
                        "peak acc (%)": f"{best.accuracy * 100:.1f}",
                    }
                )
        table = pd.DataFrame(rows)
        (tables_dir / f"mixing_{dataset}.md").write_text(_markdown_table(table) + "\n")
        sections.append("## Mixing with real data\n\n" + _markdown_table(table) + "\n")

    ablation = _load(runs / "ablation" / "ablation.csv")
    if ablation is not None and len(ablation):
        wide = ablation.pivot_table(index=["dataset", "family"], columns=["knob", "value"], values="accuracy").round(4)
        wide.columns = [" / ".join(str(x) for x in col) for col in wide.columns]
        (tables_dir / f"ablation_{dataset}.md").write_text(_markdown_table(wide.reset_index()) + "\n")
        sections.append("## Ablations\n\n" + _markdown_table(wide.reset_index()) + "\n")

    collinearity = runs / "subsets" / "collinearity.json"
    if collinearity.exists():
        import json

        data = json.loads(collinearity.read_text())
        lines = [f"- max |rho| between factors: {data['max_abs_pair']:.2f}"]
        for key, value in data["vif"].items():
            lines.append(f"- VIF({key}) = {value:.2f}")
        sections.append("## Collinearity of the four factors\n\n" + "\n".join(lines) + "\n")

    gaps = results / "gap_closed.md"
    if gaps.exists():
        sections.append("## Fraction of the real-data gap closed\n\n" + gaps.read_text() + "\n")

    sections.append(
        "## How to fill the paper's blanks\n\n"
        "Every `[xx]`/`[Result to write]` marker in the draft maps to one of the tables above;\n"
        "see docs/PAPER_MAP.md for the marker-by-marker mapping and the command that produces it.\n"
    )

    out = results / "RESULTS.md"
    out.write_text("\n".join(sections))
    log.info(f"wrote {out}")
    return out
