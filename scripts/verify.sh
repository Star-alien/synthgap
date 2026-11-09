#!/usr/bin/env bash
# Reproduce the CPU verification: unit tests + every pipeline stage on the toy setup.
set -euo pipefail
PY=${PY:-python}
$PY -m pytest -q tests
$PY -m synthgap.cli all --config configs/exp/smoke.yaml
echo
echo "OK: results/figures (Figs 1-4), results/tables (Tables 1-2, Shapley), results/RESULTS.md"
