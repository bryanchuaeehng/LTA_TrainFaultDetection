"""Fit the Door thresholds on the full Train stream and save them.

    python -m src.door.train

Writes ``artifacts/door/model.json`` — two numbers and a fallback. The model is
small enough to be human-readable, which is a feature: a reviewer can check the
shipped thresholds against the validation output without loading a binary.
"""

from __future__ import annotations

import json

from src.common.paths import DATA_ROOT
from src.door.evaluate import TRAIN_PATH, load_truth
from src.door.model import ABNORMAL, DoorThresholdModel
from src.door.predict import MODEL_PATH
from src.door.segment import cycle_features, load_stream, segment_stream


def main() -> int:
    frame = load_stream(TRAIN_PATH)
    features = cycle_features(frame, segment_stream(frame))
    truth = load_truth()

    if len(features) != len(truth):
        print(
            f"Segment count {len(features)} does not match the answer key "
            f"{len(truth)}; refusing to fit against a guessed alignment."
        )
        return 1

    labels = truth["status"].to_numpy()
    model = DoorThresholdModel().fit(features, labels)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "thresholds": model.thresholds,
        "fallback": model.fallback,
        "feature": "current_sum",
        "n_train_cycles": int(len(features)),
        "n_abnormal": int((labels == ABNORMAL).sum()),
        "note": (
            "Per-operation threshold on integrated motor current. Fitted on all "
            "110 Train cycles. See docs/plans/door.md for validation."
        ),
    }
    MODEL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"fitted on {len(features)} cycles ({payload['n_abnormal']} abnormal)")
    for operation, cutoff in sorted(model.thresholds.items()):
        print(f"  {operation:<6} threshold {cutoff:,.0f}")
    print(f"written to {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
