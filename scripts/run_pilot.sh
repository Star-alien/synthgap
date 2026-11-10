#!/bin/bash
cd ~/Downloads/synthgap_work/synthgap && source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
C="--config configs/exp/main.yaml --config configs/exp/kaggle20.yaml --config configs/exp/local20.yaml --config configs/exp/final20.yaml --set paths.results=results_pilot"
for st in measure factorial subsets das h3 mixing ablation figures report; do
  echo "=== $st $(date +%T)"; python -m synthgap.cli $st $C 2>&1 | grep -v "Warning\|Loading\|Fetching\|deprec\|safety\|Redirects"; echo "=== exit $st ${PIPESTATUS[0]}"
done
echo ALLDONE
