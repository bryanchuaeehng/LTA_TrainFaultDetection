"""Door inference: segment a stream, classify each cycle, emit the submission.

    from src.door.predict import predict
    df = predict(Path("Test.csv"))          # already in submission schema

    python -m src.door.predict --input Test.csv --output door_predictions.csv

The CLI is a wrapper over ``predict()`` and contains no logic of its own, so
an app that imports ``predict()`` and the command line cannot diverge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

from src.door.model import ABNORMAL, NORMAL, DoorThresholdModel
from src.door.segment import (
    GAP_THRESHOLD_S,
    cycle_features,
    load_stream,
    segment_stream,
)

__all__ = ["predict", "MODEL_PATH", "SUBMISSION_COLUMNS"]

MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "door" / "model.json"

#: Exact submission schema (Info Kit §3). No file_id: Test is one stream.
SUBMISSION_COLUMNS = ["start_time", "end_time", "prediction"]


@lru_cache(maxsize=1)
def _load_model() -> DoorThresholdModel:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No fitted Door model at {MODEL_PATH}. Run `python -m src.door.train` "
            f"to fit it from Train.csv."
        )
    payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    return DoorThresholdModel(
        thresholds={k: float(v) for k, v in payload["thresholds"].items()},
        fallback=float(payload["fallback"]),
    )


def predict(input_path: Path) -> pd.DataFrame:
    """Segment and classify a Door stream.

    ``input_path`` is a continuous-stream CSV (``Test.csv``) or a directory
    containing exactly one. Returns a dataframe already matching the exact
    submission schema: ``start_time``, ``end_time``, ``prediction``, one row per
    predicted cycle, in stream order.

    Timestamps are echoed verbatim from the input rather than reformatted, so
    no formatting drift is possible between what was read and what is written.
    """
    input_path = Path(input_path)
    if input_path.is_dir():
        candidates = sorted(input_path.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV found in {input_path}")
        stream_path = candidates[0]
        if len(candidates) > 1:
            raise ValueError("Door inference requires exactly one continuous CSV stream.")
    else:
        stream_path = input_path

    frame = load_stream(stream_path)
    cycle_ids = segment_stream(frame)
    features = cycle_features(frame, cycle_ids)
    if (features["n_rows"] < 2).any() or (features["duration_s"] <= 0).any():
        raise ValueError("The uploaded stream produced an unexpected cycle structure: incomplete cycles.")
    if ((features["duration_s"] > 60) | (features["n_rows"] > 3000)).any():
        raise ValueError("The uploaded stream produced an unexpected cycle structure. Verify the Door telemetry format.")

    print(f"[door] rows: {len(frame)}")
    print(f"[door] segments found at gap > {GAP_THRESHOLD_S}s: {len(features)}")

    model = _load_model()
    predictions = model.predict(features)

    counts = {label: int((predictions == label).sum()) for label in (NORMAL, ABNORMAL)}
    print(f"[door] predicted: {counts}")
    by_op = features["operation"].value_counts().to_dict()
    print(f"[door] operations inferred: {by_op}")

    result = pd.DataFrame(
        {
            "start_time": features["start_time"].to_numpy(),
            "end_time": features["end_time"].to_numpy(),
            "prediction": predictions,
        }
    )[SUBMISSION_COLUMNS]
    features["threshold"] = features["operation"].map(model.thresholds).fillna(model.fallback)
    features["prediction"] = predictions
    result.attrs["diagnostics"] = {
        "input_rows": len(frame),
        "cycles": features.to_dict("records"),
        "ambiguous_operations": int(features["operation_ambiguous"].sum()),
        "gap_threshold_s": GAP_THRESHOLD_S,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Door: segment and classify a stream.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    frame = predict(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
