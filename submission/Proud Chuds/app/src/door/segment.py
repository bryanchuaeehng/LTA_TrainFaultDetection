"""Segmentation and per-cycle feature extraction for the Door subsystem.

The segmentation rule is a time-gap split, justified in docs/plans/door.md §1:
the idle time between door cycles is not sampled, so the stream arrives
pre-broken. On Train this reproduces all 110 ground-truth cycles with zero
boundary error, and it is stable for any threshold between 0.2 s and 2.0 s.

That claim is data-dependent, so :func:`segment_stream` reports what it found
and the caller is expected to sanity-check the count before trusting it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "GAP_THRESHOLD_S",
    "CURRENT_COL",
    "parse_datetime",
    "format_datetime",
    "load_stream",
    "segment_stream",
    "cycle_features",
]

#: Split the stream wherever consecutive readings are more than this far apart.
#: 1.0 s sits in the middle of the stable range (0.2-2.0 s), so the choice is
#: not near a cliff in either direction. Normal sampling is 20 ms.
GAP_THRESHOLD_S: float = 1.0

CURRENT_COL = "Motor current(mA)"
VOLTAGE_COL = "Motor Voltage(10mV)"
EMF_COL = "Motor electrodynamic force"
POSITION_COL = "Door leaf position"
OPENING_COL = "Door is opening"
CLOSING_COL = "Door is closing"
TIME_COL = "Datetime"


def parse_datetime(value: str) -> pd.Timestamp:
    """Parse the dataset's native timestamp format.

    ``Year-Month-Day-Hour-Minute-Second-Millisecond``, hyphen-separated and not
    zero-padded, e.g. ``2023-7-5-0-0-3-760`` (Info Kit §2.2). pandas cannot
    parse this directly because of the trailing millisecond field.
    """
    parts = [int(p) for p in str(value).split("-")]
    year, month, day, hour, minute, second, milli = parts
    return pd.Timestamp(year, month, day, hour, minute, second, milli * 1000)


def format_datetime(value: pd.Timestamp) -> str:
    """Render a timestamp back in the dataset's native format.

    Used only as a fallback. The pipeline echoes the original input strings
    wherever possible so that no formatting drift can occur between what we
    read and what we submit.
    """
    return (
        f"{value.year}-{value.month}-{value.day}-{value.hour}-"
        f"{value.minute}-{value.second}-{value.microsecond // 1000}"
    )


def load_stream(path: Path) -> pd.DataFrame:
    """Read a Door stream CSV and attach parsed timestamps.

    Keeps the original ``Datetime`` strings alongside the parsed values so the
    submission can echo them verbatim.
    """
    frame = pd.read_csv(path)
    if TIME_COL not in frame.columns:
        raise ValueError(
            f"Door telemetry requires a {TIME_COL!r} column."
        )
    required = [CURRENT_COL, VOLTAGE_COL, EMF_COL, POSITION_COL, OPENING_COL, CLOSING_COL]
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError("Door telemetry is missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("The Door telemetry file contains no readings.")
    try:
        frame["_t"] = frame[TIME_COL].map(parse_datetime)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Door timestamps must use the supplied year-month-day-hour-minute-second-millisecond format.") from exc
    if not frame["_t"].is_monotonic_increasing or frame["_t"].duplicated().any():
        raise ValueError("Door timestamps must be unique and in increasing order.")
    for col in required:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("Door telemetry contains missing or non-finite sensor values.")
    if not frame[[OPENING_COL, CLOSING_COL]].isin([0, 1]).all().all():
        raise ValueError("Door opening and closing flags must contain only 0 or 1.")
    return frame


def segment_stream(
    frame: pd.DataFrame, gap_threshold_s: float = GAP_THRESHOLD_S
) -> pd.Series:
    """Assign a cycle id to every row by splitting on inter-row time gaps.

    Returns a Series of integer cycle ids aligned to ``frame``'s index.
    """
    deltas = frame["_t"].diff().dt.total_seconds().fillna(0.0)
    return (deltas > gap_threshold_s).cumsum()


def _operation(block: pd.DataFrame) -> str:
    """Infer Open/Close from the controller's own flags.

    Operation is read from the stream, never predicted: ``Door is opening`` and
    ``Door is closing`` are columns in the data, present in Test as well as
    Train. The Info Kit says operation is informational and need not be
    predicted (§2.1); we use it only to pick which threshold applies.

    Ties and absences are explicit: the fitted model's global threshold is
    used, and the application shows an operation-ambiguity warning.
    """
    opening = int((block[OPENING_COL] == 1).sum())
    closing = int((block[CLOSING_COL] == 1).sum())
    if opening > closing:
        return "Open"
    if closing > opening:
        return "Close"
    return "Ambiguous"


def cycle_features(frame: pd.DataFrame, cycle_ids: pd.Series) -> pd.DataFrame:
    """One row per cycle, with the features docs/plans/door.md §3 specifies.

    ``current_sum`` is the primary discriminator: motor current integrated over
    the cycle, i.e. total work done against resistance. Peak current is
    deliberately NOT a discriminator (AUC 0.3735, below chance) but is emitted
    so the write-up can show that.
    """
    rows = []
    for cycle_id, block in frame.groupby(cycle_ids, sort=True):
        start_t = block["_t"].iloc[0]
        end_t = block["_t"].iloc[-1]
        rows.append(
            {
                "cycle_id": int(cycle_id),
                "start_time": block[TIME_COL].iloc[0],
                "end_time": block[TIME_COL].iloc[-1],
                "start_epoch": start_t.timestamp(),
                "end_epoch": end_t.timestamp(),
                "n_rows": len(block),
                "duration_s": (end_t - start_t).total_seconds(),
                "operation": _operation(block),
                "operation_ambiguous": _operation(block) == "Ambiguous",
                "current_sum": float(block[CURRENT_COL].sum()),
                "current_mean": float(block[CURRENT_COL].mean()),
                "current_max": float(block[CURRENT_COL].max()),
                "voltage_mean": float(block[VOLTAGE_COL].mean()),
                "emf_mean": float(block[EMF_COL].mean()),
                "position_range": float(
                    block[POSITION_COL].max() - block[POSITION_COL].min()
                ),
            }
        )
    return pd.DataFrame(rows)
