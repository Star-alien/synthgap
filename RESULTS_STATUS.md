# Results status

* `pytest`: 14/14 passing.
* `python -m synthgap.cli all --config configs/exp/smoke.yaml`: ran end to end on CPU (~1 min).
  Outputs are in `results/` and are **SMOKE-TEST OUTPUT, NOT PAPER RESULTS** (procedural mock generators).
* The from-scratch ResNet control was skipped (torch not installed in this run).
* Real experiments (CUB / Aircraft, SD1.5/SDXL/SD3.5/FLUX; ~200-600 GPU-hours, see docs/COMPUTE_PLAN.md)
  have **not** been run. `paper_DRAFT_blank_cells.pdf` is the unmodified draft: its `[xx]` cells are still blank
  because no real numbers exist yet.
