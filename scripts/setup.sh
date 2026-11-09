#!/usr/bin/env bash
# One-command setup + verification on any machine (CPU only, ~5 min).
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install -e ".[dev]"
python -m synthgap.cli all --config configs/exp/smoke.yaml
echo
echo "Smoke run complete. See results/RESULTS.md, results/figures, results/tables."
echo "For the real experiments: pip install -r requirements-gpu.txt, then read README.md."
