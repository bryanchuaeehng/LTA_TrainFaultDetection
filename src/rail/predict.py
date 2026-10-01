"""Rail corrugation inference.

Exposes ``predict(input_path) -> pd.DataFrame`` returning the exact submission
schema of plan section 3, plus a LOGIC-FREE ``--input``/``--output`` CLI wrapper
over that same function.

All preprocessing lives inside ``predict()``. The caller never renames,
reorders, casts or post-processes anything, and the CLI does no work beyond
parsing two arguments, calling ``predict()`` and writing the frame.

Order of operations per file, fixed by plan section 5.8:

  1. read the file, assert shape (10000, 129)
  2. derive v from column 1 (section 5.3)
  3. if v < V_MIN: emit ``Normal``, log it, count it, next file - the model is
     NOT invoked (section 5.3a.4, the low-speed domain rule)
  4. otherwise extract features, drop meta_/rejected columns, apply the fitted
     pipeline, argmax (section 5.6)

Two counters are reported separately and never conflated (section 5.8): the
low-speed rule firings, which are legitimate domain behaviour, and the failure
fallbacks, which mean something went wrong.

Usage:
    python -m src.rail.predict --input <path> --output <path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.common.metrics import RAIL_CLASSES
from src.rail.constants import MODEL_PATH, N_COLUMNS, N_SAMPLES, V_MIN
from src.rail.features import extract_file
from src.rail.speed import derive_speed_for_file, list_csv_files

__all__ = ["predict", "main"]

#: The submission schema, plan section 3.2. Exact names, exact order.
SUBMISSION_COLUMNS = ["file_id", "prediction"]

#: The fallback label for an unreadable/unprocessable file (plan section 5.8).
#: A missing row invalidates the whole submission; a wrong row costs one file.
FALLBACK_LABEL = "Normal"


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


def validate_input(path: Path) -> None:
    """Validate even when the low-speed rule applies."""
    data = pd.read_csv(path, dtype=np.float32)
    if data.shape != (N_SAMPLES, N_COLUMNS):
        raise ValueError(f"Expected {N_SAMPLES:,} readings and {N_COLUMNS} columns; received {data.shape}.")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Rail telemetry contains missing or non-finite sensor values.")
    if not data.iloc[:, 0].isin([0, 1]).all():
        raise ValueError("The first Rail column must be a binary speed signal (0 or 1).")


def _resolve_inputs(input_path: Path) -> list[Path]:
    """A single CSV, or every CSV in a directory, in natural numeric order."""
    if input_path.is_dir():
        return list_csv_files(input_path)
    return [input_path]


def predict(input_path: Path) -> pd.DataFrame:
    """Predict a corrugation class for every input file.

    ``input_path`` may be a single CSV or a directory of CSVs. The returned
    frame already matches the submission schema exactly: columns
    ``file_id,prediction``, both string dtype, one row per input file, in
    natural numeric order of the file index.
    """
    input_path = Path(input_path)
    paths = _resolve_inputs(input_path)
    if not paths:
        raise ValueError("No Rail CSV files were supplied.")

    artefact = load_model()
    model = artefact["model"]

    rows: list[dict[str, str]] = []
    low_speed_fired: list[tuple[str, float]] = []
    fallbacks: list[tuple[str, str]] = []

    for path in paths:
        try:
            validate_input(path)
            # Steps 1-2: shape assertion happens inside extract_file; the speed
            # derivation reads column 1 only, so it is cheap enough to do first.
            v_mps, _ = derive_speed_for_file(path)

            # Step 3: the low-speed domain rule. A branch on a computed
            # quantity, evaluated BEFORE the pipeline is invoked - not a
            # post-hoc edit of the output frame.
            if v_mps < V_MIN:
                low_speed_fired.append((path.name, v_mps))
                print(
                    f"[low-speed rule] {path.name}: v = {v_mps:.4f} m/s "
                    f"< V_MIN = {V_MIN} -> Normal (model not invoked)",
                    file=sys.stderr,
                )
                rows.append({"file_id": path.name, "prediction": "Normal"})
                continue

            # Step 4: features -> fitted pipeline -> argmax.
            features = pd.DataFrame([extract_file(path)])
            proba = model.predict_proba(features)[0]
            classes = list(model.named_steps["clf"].classes_)
            # argmax with RAIL_CLASSES order fixing the tie-break: on an exact
            # tie the lowest index wins (plan section 5.6).
            order = [classes.index(c) for c in RAIL_CLASSES if c in classes]
            best = max(order, key=lambda i: (proba[i], -order.index(i)))
            label = str(classes[best])
            rows.append({"file_id": path.name, "prediction": label})

        except Exception as exc:  # noqa: BLE001 - every file must get a row
            fallbacks.append((path.name, repr(exc)))
            print(
                f"[fallback] {path.name}: {exc!r} -> {FALLBACK_LABEL}",
                file=sys.stderr,
            )
            rows.append({"file_id": path.name, "prediction": FALLBACK_LABEL})

    # The two counters, reported separately and never conflated.
    print(f"low_speed_rule_fired: {len(low_speed_fired)}", file=sys.stderr)
    print(f"fallbacks: {len(fallbacks)}", file=sys.stderr)
    for name, v_mps in low_speed_fired:
        print(f"  low-speed: {name} v={v_mps:.4f} m/s", file=sys.stderr)

    frame = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
    frame = frame.astype({"file_id": "string", "prediction": "string"})
    frame.attrs["diagnostics"] = {
        "low_speed_rule": [{"file_id": n, "speed_mps": v} for n, v in low_speed_fired],
        "errors": [{"file_id": n, "detail": e} for n, e in fallbacks],
    }
    return frame


def main() -> None:
    """Logic-free CLI: parse two arguments, call predict(), write the frame."""
    parser = argparse.ArgumentParser(
        description="Predict rail corrugation classes for a CSV file or directory."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    predictions = predict(args.input)
    predictions.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
