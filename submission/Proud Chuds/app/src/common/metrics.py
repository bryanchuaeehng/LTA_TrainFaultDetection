"""Shared metric implementations.

Every subsystem and the tester score against the code in this file. A metric
computed two different ways in two places is a defect even when both are
correct, so nothing anywhere else may hand-roll or inline a metric.

Scope this session is rail only, so macro F1 is the only metric here. Door,
ACV and SHM metrics are deferred, not abandoned.

Reference for the rail metric --- Rail_Corrugation_Info_Kit.md section 4,
lines 127-129, quoted verbatim:

    Rail corrugation is graded on **macro F1** across the three classes
    (Normal, Side I, Side II) - not plain accuracy. Macro F1 computes the F1
    score for each class independently, then averages the three unweighted
    (each class counts equally regardless of how many examples it has).

Two consequences of that wording are load-bearing and are implemented
explicitly below:

* "each class counts equally regardless of how many examples it has" means the
  average is over the three CLASSES, not over the samples. A class absent from
  a particular fold still occupies a slot in the denominator.
* "a class the model never gets right (0 recall, 0 precision) contributes an
  F1 of 0 to the average" (line 134-135) fixes the degenerate case: an
  undefined 0/0 F1 is scored as 0.0, not dropped and not treated as 1.0.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import f1_score

__all__ = ["RAIL_CLASSES", "per_class_f1", "macro_f1", "macro_f1_from_per_class"]

#: The three rail classes, in the order the Info Kit lists them (section 4,
#: line 127). Fixing the order makes per-class output positionally stable and
#: guarantees every class occupies a slot in the macro average even when it is
#: absent from a given fold.
RAIL_CLASSES: tuple[str, ...] = ("Normal", "Side I", "Side II")


def per_class_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = RAIL_CLASSES,
) -> dict[str, float]:
    """F1 for each class independently, keyed by class label.

    A class with no true samples and no predicted samples scores 0.0, per the
    Info Kit's rule that a never-correct class contributes 0 to the average.
    """
    scores = f1_score(
        y_true,
        y_pred,
        labels=list(labels),
        average=None,
        zero_division=0.0,
    )
    return {label: float(score) for label, score in zip(labels, scores)}


def macro_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = RAIL_CLASSES,
) -> float:
    """Macro F1: per-class F1 averaged unweighted over ``labels``.

    This is the rail ``primary_metric`` (Info Kit section 4, line 152).

    ``labels`` is passed explicitly so the denominator is always the full class
    count. Letting it default to the classes present in the data would shrink
    the denominator on a fold missing a minority class and silently inflate the
    score -- with 14 Side I samples across 272 files, that is a realistic fold.
    """
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(labels),
            average="macro",
            zero_division=0.0,
        )
    )


def macro_f1_from_per_class(per_class_scores: Sequence[float]) -> float:
    """Macro-average a set of already-computed per-class F1 scores.

    Exists because the Info Kit's worked example (section 4, lines 137-145) is
    stated as three per-class F1 values rather than as predictions, so this is
    the exact operation that example pins down. Production code paths should
    call :func:`macro_f1` on labels instead.
    """
    scores = np.asarray(per_class_scores, dtype=float)
    if scores.size == 0:
        raise ValueError("per_class_scores must not be empty")
    return float(scores.mean())


# ===========================================================================
# Door — IoU-weighted F1
# ===========================================================================
#
# Reference: Door_Subsystem_Info_Kit.md section 4, quoted where it matters.
#
# The Door metric scores segmentation and labelling together. It is NOT plain
# F1, even though on the data we were given it reduces to plain F1 because
# gap-splitting recovers every boundary exactly (see docs/plans/door.md §1).
# The real matching procedure is implemented here regardless, so that a stream
# which does NOT segment cleanly is scored honestly rather than optimistically.

DOOR_CLASSES: tuple[str, ...] = ("Normal", "Abnormal resistance")


def segment_iou(
    true_start: float, true_end: float, pred_start: float, pred_end: float
) -> float:
    """Intersection-over-union of two time intervals.

    Info Kit section 4.1, verbatim:

        intersection = max(0, min(true_end, pred_end) - max(true_start, pred_start))
        union        = (true_end - true_start) + (pred_end - pred_start) - intersection
        IoU          = intersection / union   (0 if union <= 0)
    """
    intersection = max(0.0, min(true_end, pred_end) - max(true_start, pred_start))
    union = (true_end - true_start) + (pred_end - pred_start) - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def door_iou_f1(
    true_segments: Sequence[tuple[float, float, str]],
    pred_segments: Sequence[tuple[float, float, str]],
) -> float:
    """IoU-weighted F1 for Door (Info Kit section 4.2).

    Each segment is ``(start, end, label)`` with start/end as numeric times in
    any consistent unit (seconds since epoch is what the pipeline uses).

    The three rules that make this not-plain-F1, all from section 4.1:

    1. **Same label only.** "A segment with perfectly overlapping timing but
       the wrong label ... cannot match at all." A mislabelled segment is both
       a miss and a false positive, never a partial credit.
    2. **IoU > 0 required.** Touching-but-not-overlapping does not match.
    3. **One-to-one, greedy by highest IoU first.** "once a segment (true or
       predicted) is used, it's removed from further consideration."

    Credit for a match is the IoU value itself, not a flat 1.0, so sloppy
    boundaries cost score without being counted as a miss.
    """
    n_true = len(true_segments)
    n_pred = len(pred_segments)
    if n_true == 0 and n_pred == 0:
        return 0.0
    if n_true == 0 or n_pred == 0:
        return 0.0

    candidates = []
    for i, (ts_, te_, tl) in enumerate(true_segments):
        for j, (ps_, pe_, pl) in enumerate(pred_segments):
            if tl != pl:
                continue  # rule 1
            iou = segment_iou(ts_, te_, ps_, pe_)
            if iou > 0.0:  # rule 2
                candidates.append((iou, i, j))

    # Rule 3: greedy, highest IoU first. Ties broken by index for determinism,
    # so the same inputs always produce the same score.
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

    used_true: set[int] = set()
    used_pred: set[int] = set()
    iou_sum = 0.0
    for iou, i, j in candidates:
        if i in used_true or j in used_pred:
            continue
        used_true.add(i)
        used_pred.add(j)
        iou_sum += iou

    soft_recall = iou_sum / n_true
    soft_precision = iou_sum / n_pred
    if soft_recall + soft_precision <= 0:
        return 0.0
    return float(
        2 * soft_recall * soft_precision / (soft_recall + soft_precision)
    )
