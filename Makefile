# Convenience targets. Every target is a thin wrapper over `python -m synthgap.cli`.
PY ?= python

.PHONY: install install-gpu test smoke data generate features measure factorial subsets das scratch mixing ablation h3 figures report all clean-results

install:
	$(PY) -m pip install -e ".[dev]"

install-gpu:
	$(PY) -m pip install -e ".[dev,gpu]"

test:
	$(PY) -m pytest -q

# Full CPU-only pipeline on a toy setup (procedural "generators" + cheap features).
# Verifies every stage end to end in a few minutes. NOT paper results.
smoke:
	$(PY) -m synthgap.cli all --config configs/exp/smoke.yaml

data:
	$(PY) -m synthgap.cli download --config configs/exp/main.yaml

generate:
	$(PY) -m synthgap.cli generate --config configs/exp/main.yaml

features:
	$(PY) -m synthgap.cli features --config configs/exp/main.yaml

measure:
	$(PY) -m synthgap.cli measure --config configs/exp/main.yaml

factorial:
	$(PY) -m synthgap.cli factorial --config configs/exp/main.yaml --config configs/exp/factorial.yaml

subsets:
	$(PY) -m synthgap.cli subsets --config configs/exp/main.yaml --config configs/exp/subsets.yaml

das:
	$(PY) -m synthgap.cli das --config configs/exp/main.yaml --config configs/exp/das.yaml

scratch:
	$(PY) -m synthgap.cli scratch --config configs/exp/main.yaml --config configs/exp/scratch.yaml

mixing:
	$(PY) -m synthgap.cli mixing --config configs/exp/main.yaml --config configs/exp/mixing.yaml

ablation:
	$(PY) -m synthgap.cli ablation --config configs/exp/main.yaml --config configs/exp/ablation.yaml

h3:
	$(PY) -m synthgap.cli h3 --config configs/exp/main.yaml

figures:
	$(PY) -m synthgap.cli figures --config configs/exp/main.yaml

report:
	$(PY) -m synthgap.cli report --config configs/exp/main.yaml

# Everything that the paper needs, in dependency order. Requires GPUs + datasets.
all: data generate features measure factorial subsets das scratch mixing ablation h3 figures report

clean-results:
	rm -rf results/runs results/figures/*.pdf results/figures/*.png results/tables/*.csv
