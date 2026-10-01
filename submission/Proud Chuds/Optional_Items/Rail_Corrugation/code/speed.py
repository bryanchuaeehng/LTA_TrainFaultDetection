"""Derived train speed from column 1, and the file enumeration helpers.

The derivation is plan section 5.3, from Info Kit section 2.1 lines 74-77: the
toothed-wheel sensor has 90 teeth and toggles 1/0 as each tooth enters and
leaves the detection point, so each tooth yields 2 transitions.

    T  = number of 0->1 and 1->0 transitions in column 1 over the file
    R  = T / (2 * 90)                     revolutions
    C  = pi * 0.85 m                      wheel circumference
    v  = R * C / t     with t = 1.0 s     metres per second

Column 1 is literally binary {0, 1} in all 272 Train files (plan section 7.4,
RESOLVED at Gate 1), so transitions are counted directly as ``count(diff != 0)``
and no thresholding is applied.

One implementation, used by Gate 1b, by feature extraction, and by
``predict()``. There is no second copy anywhere.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.rail.constants import (
    FILE_DURATION_S,
    TRANSITIONS_PER_REV,
    WHEEL_CIRCUMFERENCE_M,
)

__all__ = [
    "count_transitions",
    "speed_from_column1",
    "read_speed_column",
    "derive_speed_for_file",
    "file_index",
    "list_csv_files",
]


def count_transitions(col1: np.ndarray) -> int:
    """Number of level changes in the binary speed channel."""
    arr = np.asarray(col1)
    if arr.size < 2:
        return 0
    return int(np.count_nonzero(np.diff(arr) != 0))


def speed_from_column1(col1: np.ndarray) -> tuple[float, int]:
    """Return ``(v_mps, T)`` for one file's column 1."""
    transitions = count_transitions(col1)
    revolutions = transitions / TRANSITIONS_PER_REV
    v_mps = revolutions * WHEEL_CIRCUMFERENCE_M / FILE_DURATION_S
    return float(v_mps), transitions


def read_speed_column(path: Path) -> np.ndarray:
    """Read ONLY column 1 of a data file.

    ``usecols=[0]`` is mandatory on the Test path (plan section 4.8.3) and is
    used everywhere so there is a single code path.
    """
    frame = pd.read_csv(path, usecols=[0])
    return frame.iloc[:, 0].to_numpy()


def derive_speed_for_file(path: Path) -> tuple[float, int]:
    """Convenience: read column 1 of ``path`` and derive ``(v_mps, T)``."""
    return speed_from_column1(read_speed_column(path))


_INDEX_RE = re.compile(r"(\d+)")


def file_index(path: Path) -> int:
    """Integer index parsed from ``TrainN.csv`` / ``TestN.csv``."""
    match = _INDEX_RE.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot parse a numeric index from file name: {path.name}")
    return int(match.group(1))


def list_csv_files(directory: Path) -> list[Path]:
    """CSV files in natural NUMERIC order (Train1, Train2, ..., Train272).

    Never ``sorted()`` on the file name, which would give Train1, Train10,
    Train100 (plan section 5.4). Deterministic ordering is what makes the cache
    reproducible and the adjacency diagnostic meaningful.
    """
    return sorted(directory.glob("*.csv"), key=file_index)
