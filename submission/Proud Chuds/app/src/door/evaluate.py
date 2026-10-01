"""Rolling-origin validation for Door, per docs/plans/door.md §4.

Contiguous splits only. The data is a temporal stream, so a random split would
let a threshold be tuned on cycles temporally interleaved with the ones it is
scored on. Splits are made at whole-cycle boundaries in stream order.

    python -m src.door.evaluate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.metrics import door_iou_f1
from src.common.paths import DATA_ROOT
from src.door.model import ABNORMAL, NORMAL, DoorThresholdModel, always_normal
from src.door.segment import (
    GAP_THRESHOLD_S,
    cycle_features,
    load_stream,
    parse_datetime,
    segment_stream,
)

DOOR_DIR = DATA_ROOT / "Door"
TRAIN_PATH = DOOR_DIR / "Train.csv"
ANSWER_PATH = DOOR_DIR / "Train_Segments_Answer.csv"

RULE = "=" * 76


def load_truth() -> pd.DataFrame:
    truth = pd.read_csv(ANSWER_PATH)
    truth["start_epoch"] = truth["start_time"].map(
        lambda s: parse_datetime(s).timestamp()
    )
    truth["end_epoch"] = truth["end_time"].map(lambda s: parse_datetime(s).timestamp())
    return truth


def as_segments(frame: pd.DataFrame, labels) -> list[tuple[float, float, str]]:
    return [
        (float(s), float(e), str(l))
        for s, e, l in zip(frame["start_epoch"], frame["end_epoch"], labels)
    ]


def main() -> int:
    print(RULE)
    print("DOOR — rolling-origin validation")
    print(RULE)

    frame = load_stream(TRAIN_PATH)
    cycles = segment_stream(frame)
    features = cycle_features(frame, cycles)
    truth = load_truth()

    print(f"stream rows        : {len(frame)}")
    print(f"gap threshold      : {GAP_THRESHOLD_S} s")
    print(f"segments found     : {len(features)}")
    print(f"true segments      : {len(truth)}")

    if len(features) != len(truth):
        print("\nSEGMENT COUNT MISMATCH — stopping rather than guessing an alignment.")
        return 1

    start_err = np.abs(features["start_epoch"].to_numpy() - truth["start_epoch"].to_numpy())
    end_err = np.abs(features["end_epoch"].to_numpy() - truth["end_epoch"].to_numpy())
    print(f"max start error    : {start_err.max():.6f} s")
    print(f"max end error      : {end_err.max():.6f} s")
    print(f"boundaries exact   : {start_err.max() == 0 and end_err.max() == 0}")

    labels = truth["status"].to_numpy()
    features = features.copy()
    features["status"] = labels
    n_abnormal = int((labels == ABNORMAL).sum())
    print(f"labels             : {len(labels) - n_abnormal} Normal / {n_abnormal} {ABNORMAL}")

    # Operation inference, checked against the answer key's informational column.
    agreement = (features["operation"].to_numpy() == truth["operation"].to_numpy()).mean()
    print(f"operation inferred correctly: {agreement * 100:.1f}%")

    print(f"\n{RULE}\nB0 — always Normal (the floor)\n{RULE}")
    b0 = door_iou_f1(
        as_segments(features, labels), as_segments(features, always_normal(features))
    )
    print(f"  IoU-weighted F1: {b0:.4f}")

    print(f"\n{RULE}\nB1 — per-operation threshold on integrated current\n{RULE}")
    print("  Rolling origin: train on everything before the cut, test on the")
    print("  next ~10% of cycles. Thresholds fitted inside the training part only.\n")

    n = len(features)
    scores: list[float] = []
    print(f"  {'fold':<5} {'train':<12} {'test':<12} {'abn':>4}  "
          f"{'thr Open':>10} {'thr Close':>10}  {'score':>7}")
    for k in range(5):
        cut = int(n * (0.5 + 0.1 * k))
        test_end = min(cut + int(n * 0.1) + 1, n)
        train_idx = np.arange(cut)
        test_idx = np.arange(cut, test_end)
        if len(test_idx) == 0:
            continue

        train_f = features.iloc[train_idx]
        test_f = features.iloc[test_idx]
        n_abn = int((test_f["status"] == ABNORMAL).sum())

        model = DoorThresholdModel().fit(train_f, train_f["status"].to_numpy())
        predicted = model.predict(test_f)
        score = door_iou_f1(
            as_segments(test_f, test_f["status"].to_numpy()),
            as_segments(test_f, predicted),
        )
        scores.append(score)
        t_open = model.thresholds.get("Open", float("nan"))
        t_close = model.thresholds.get("Close", float("nan"))
        print(f"  {k:<5} [0:{cut}]{'':<{max(0, 7 - len(str(cut)))}} "
              f"[{test_idx[0]}:{test_end}]{'':<{max(0, 6 - len(str(test_end)))}} "
              f"{n_abn:>4}  {t_open:>10.0f} {t_close:>10.0f}  {score:>7.4f}")

    arr = np.array(scores)
    print(f"\n  mean {arr.mean():.4f}   min {arr.min():.4f}   max {arr.max():.4f}"
          f"   std {arr.std():.4f}   folds {len(arr)}")
    print("\n  The MINIMUM is the honest headline, not the mean: with 3-4 abnormal")
    print("  cycles per fold, a single misclassification moves the score a long way.")

    print(f"\n{RULE}\nFull-data fit, for the shipped model\n{RULE}")
    final = DoorThresholdModel().fit(features, labels)
    for operation, cutoff in sorted(final.thresholds.items()):
        subset = features[features["operation"] == operation]
        is_abn = subset["status"] == ABNORMAL
        print(f"  {operation:<6} threshold {cutoff:>10.0f}   "
              f"normal max {subset.loc[~is_abn, 'current_sum'].max():>8.0f}   "
              f"abnormal min {subset.loc[is_abn, 'current_sum'].min():>8.0f}")
    resubmit = door_iou_f1(
        as_segments(features, labels), as_segments(features, final.predict(features))
    )
    print(f"\n  in-sample score {resubmit:.4f} (NOT a generalisation estimate —")
    print("  the thresholds were fitted on these same cycles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
