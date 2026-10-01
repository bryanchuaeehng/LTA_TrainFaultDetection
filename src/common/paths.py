"""Single source of truth for the dataset location.

The PS3 datasets are ~6.1 GB and live OUTSIDE this repository. They are never
committed. Every module that touches data resolves its path through this file
and nowhere else, so that relocating the corpus is a one-line change here.

Resolution order:

1. The ``NEBULAX_DATA`` environment variable, if set and non-empty.
2. ``_DEFAULT_DATA_ROOT`` below.

If neither resolves to an existing directory, importing this module raises
``DataRootNotFound`` with a message naming both candidates. Failing loudly at
import is deliberate: a silent fallback to an empty or partial corpus would
produce validation numbers that look plausible and are meaningless, and with
zero leaderboard uploads nothing downstream would catch it.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "DataRootNotFound",
    "DATA_ROOT",
    "RAIL_DIR",
    "RAIL_TRAIN_DIR",
    "RAIL_TEST_DIR",
    "RAIL_TRAIN_LABELS",
    "REPO_ROOT",
    "REFERENCE_DIR",
    "EXAMPLE_SUBMISSION_DIR",
]

_ENV_VAR = "NEBULAX_DATA"

# Points at the 02_Datasets level: the four subsystem folders sit directly
# under it. Used only when NEBULAX_DATA is unset or empty.
_DEFAULT_DATA_ROOT = Path(
    r"C:\Users\qinhu\dev\nebulaxlta\nebulax-data\PS3\02_Datasets"
)


class DataRootNotFound(RuntimeError):
    """Raised at import when the dataset root cannot be located."""


def _resolve_data_root() -> Path:
    env_value = os.environ.get(_ENV_VAR, "").strip()

    if env_value:
        candidate = Path(env_value)
        if candidate.is_dir():
            return candidate
        raise DataRootNotFound(
            f"{_ENV_VAR} is set to {candidate!s} but that is not an existing "
            f"directory.\n"
            f"Point {_ENV_VAR} at the 02_Datasets folder (the one containing "
            f"ACV/, Door/, Rail_Corrugation/ and SHM/), or unset it to fall "
            f"back to the default at {_DEFAULT_DATA_ROOT!s}."
        )

    if _DEFAULT_DATA_ROOT.is_dir():
        return _DEFAULT_DATA_ROOT

    raise DataRootNotFound(
        f"Could not locate the PS3 dataset root.\n"
        f"  {_ENV_VAR} is unset or empty.\n"
        f"  The default path {_DEFAULT_DATA_ROOT!s} does not exist.\n"
        f"Set {_ENV_VAR} to the 02_Datasets folder (the one containing ACV/, "
        f"Door/, Rail_Corrugation/ and SHM/). The data is ~6.1 GB and is never "
        f"committed to this repository."
    )


DATA_ROOT: Path = _resolve_data_root()

# --- Rail Corrugation ------------------------------------------------------
# The only subsystem in scope this session. Door, ACV and SHM are deferred;
# their folders exist under DATA_ROOT but are intentionally not named here.
RAIL_DIR: Path = DATA_ROOT / "Rail_Corrugation"
RAIL_TRAIN_DIR: Path = RAIL_DIR / "Train"          # Train1.csv .. Train272.csv
RAIL_TEST_DIR: Path = RAIL_DIR / "Test"            # Test1.csv .. Test68.csv
RAIL_TRAIN_LABELS: Path = RAIL_DIR / "Train_Labels.csv"

# --- In-repo reference material (specs, not data) --------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
REFERENCE_DIR: Path = REPO_ROOT / "reference"
EXAMPLE_SUBMISSION_DIR: Path = REFERENCE_DIR / "04_Example_Submission"
