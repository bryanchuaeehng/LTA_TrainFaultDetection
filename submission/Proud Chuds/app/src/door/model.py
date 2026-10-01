"""The Door classifier: one integrated-current threshold per operation type.

Two scalar parameters in total. That is the whole model, and it is deliberate.
With 30 abnormal cycles and a feature that separates the classes perfectly
within each operation (AUC 1.0000, docs/plans/door.md §3), a learned model has
nothing to add and a great deal of variance to introduce.

The physical reading: abnormal resistance means the motor does more total work
to move the door. It is a cumulative-effort signal, not a spike -- peak current
is AUC 0.3735, *below* chance, so a spike detector is worse than useless here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.common.metrics import DOOR_CLASSES

__all__ = ["NORMAL", "ABNORMAL", "DoorThresholdModel", "always_normal"]

NORMAL, ABNORMAL = DOOR_CLASSES

#: The feature the thresholds are applied to.
FEATURE = "current_sum"


@dataclass
class DoorThresholdModel:
    """Per-operation threshold on integrated motor current.

    ``thresholds`` maps an operation ("Open"/"Close") to the cutoff above which
    a cycle is called abnormal. ``fallback`` covers an operation never seen
    during fitting.
    """

    thresholds: dict[str, float] = field(default_factory=dict)
    fallback: float = float("inf")

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> "DoorThresholdModel":
        """Fit one threshold per operation by maximising F1 on the training rows.

        Every threshold is chosen using only the rows passed in. The caller is
        responsible for passing a training fold, never a validation fold.
        """
        is_abnormal = np.asarray(labels) == ABNORMAL
        self.thresholds = {}

        for operation in sorted(set(features["operation"])):
            mask = (features["operation"] == operation).to_numpy()
            values = features.loc[mask, FEATURE].to_numpy(dtype=float)
            target = is_abnormal[mask]
            self.thresholds[operation] = _best_threshold(values, target)

        all_values = features[FEATURE].to_numpy(dtype=float)
        self.fallback = _best_threshold(all_values, is_abnormal)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Label each cycle. Above its operation's threshold means abnormal."""
        out = np.full(len(features), NORMAL, dtype=object)
        values = features[FEATURE].to_numpy(dtype=float)
        operations = features["operation"].to_numpy()
        for i, (value, operation) in enumerate(zip(values, operations)):
            cutoff = self.thresholds.get(operation, self.fallback)
            if value > cutoff:
                out[i] = ABNORMAL
        return out


def _best_threshold(values: np.ndarray, is_positive: np.ndarray) -> float:
    """The cutoff maximising F1 for ``values > cutoff`` predicting positive.

    Candidates are midpoints between adjacent observed values, so the chosen
    cutoff never sits exactly on a training point -- a threshold equal to an
    observed value would behave differently under floating-point noise.

    With no positives present, returns +inf: predict everything normal, which
    is the correct degenerate behaviour rather than an arbitrary guess.
    """
    if is_positive.sum() == 0 or len(values) == 0:
        return float("inf")

    order = np.unique(values)
    if len(order) == 1:
        # One distinct value: no threshold can separate anything.
        return float("inf") if not is_positive.all() else float(order[0] - 1.0)

    candidates = (order[:-1] + order[1:]) / 2.0
    best_cutoff = float("inf")
    best_f1 = -1.0
    for cutoff in candidates:
        predicted = values > cutoff
        tp = int((predicted & is_positive).sum())
        if tp == 0:
            continue
        fp = int((predicted & ~is_positive).sum())
        fn = int((~predicted & is_positive).sum())
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1 = f1
            best_cutoff = float(cutoff)
    return best_cutoff


def always_normal(features: pd.DataFrame) -> np.ndarray:
    """Baseline B0: predict Normal for everything.

    80 of 110 Train cycles are Normal, so this looks respectable on accuracy
    and scores 0 on the abnormal class. It exists to state the floor.
    """
    return np.full(len(features), NORMAL, dtype=object)
