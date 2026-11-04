"""Run an Optuna HPO study for PINTO on one PDE and parameter budget.

Usage: uv run hpo/tune_pinto.py --pde burgers --max_params 200000 --target_completed 15 --n_trials 100
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root

from architectures import PINTO
from hpo._cli import run_cli


if __name__ == "__main__":
    run_cli("pinto", PINTO, __doc__, has_sigma=False)
