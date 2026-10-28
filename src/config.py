"""Project-wide constants: output locations and the random seed."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "result"

RANDOM_STATE = 42
