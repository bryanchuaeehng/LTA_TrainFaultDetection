"""Numerical verification of the rail metric against the Info Kit.

With zero leaderboard uploads, nothing external will ever catch a metric that
disagrees with the organisers'. If macro F1 here is wrong, every number this
project produces is wrong and it will look fine. These tests are the only thing
standing between that and us, so they check the Info Kit's stated arithmetic
directly rather than checking our implementation against itself.

Source: reference/03_References/Rail_Corrugation/Rail_Corrugation_Info_Kit.md
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.metrics import (
    RAIL_CLASSES,
    macro_f1,
    macro_f1_from_per_class,
    per_class_f1,
)

# --- The Info Kit's worked example, section 4 lines 137-145 ----------------
# | Class   | F1 score |
# | Normal  | 0.97     |
# | Side I  | 0.40     |
# | Side II | 0.60     |
# **Macro F1 = (0.97 + 0.40 + 0.60) / 3 = 0.657**
KIT_PER_CLASS_F1 = {"Normal": 0.97, "Side I": 0.40, "Side II": 0.60}
KIT_MACRO_F1_STATED = 0.657


def test_worked_example_macro_average():
    """The Info Kit's headline number, reproduced exactly.

    (0.97 + 0.40 + 0.60) / 3 = 0.65666..., which the kit prints rounded to
    0.657. Asserting to 3dp confirms we round to the same published figure;
    asserting the unrounded value confirms we are not accidentally matching
    0.657 through a different route.
    """
    result = macro_f1_from_per_class(list(KIT_PER_CLASS_F1.values()))

    assert round(result, 3) == KIT_MACRO_F1_STATED
    assert result == pytest.approx((0.97 + 0.40 + 0.60) / 3, abs=1e-12)
    assert result == pytest.approx(0.6566666666666666, abs=1e-12)


def test_worked_example_class_order_does_not_matter():
    """Unweighted averaging must be order-independent."""
    forward = macro_f1_from_per_class([0.97, 0.40, 0.60])
    reversed_ = macro_f1_from_per_class([0.60, 0.40, 0.97])

    assert forward == pytest.approx(reversed_, abs=1e-12)


def test_always_predicting_normal_collapses_macro_f1():
    """The Info Kit's second worked case, section 4 lines 147-150.

    A model that always predicts "Normal" scores F1 = 0 on both fault classes.
    The kit writes this as "macro F1 = (1.0 + 0 + 0) / 3 ~= 0.33 even though its
    plain accuracy would be roughly 85-90%".

    NOTE: the kit's "1.0" for Normal is an approximation, not exact arithmetic.
    A model that predicts Normal for every file necessarily predicts Normal for
    the fault files too, so Normal's PRECISION cannot be 1.0 whenever any fault
    file exists. At the kit's own stated 85% accuracy:

        Normal precision = 85/100 = 0.85, recall = 85/85 = 1.0
        Normal F1        = 2(0.85)(1.0) / (0.85 + 1.0) = 0.918918...
        macro F1         = (0.918918... + 0 + 0) / 3   = 0.306306...

    So the true value is 0.306, not 0.333. The kit's figure is an illustrative
    round number for a point about class collapse, not a metric definition, and
    its actual definition (lines 127-129) is what we implement. We assert the
    exact arithmetic and keep the kit's qualitative claim -- macro F1 near 0.3
    against accuracy near 0.85 -- as the thing being verified.

    This is also where a wrong zero_division setting would hide: treating an
    undefined F1 as 1.0 would score this degenerate model ~0.64 instead of 0.31.
    """
    y_true = ["Normal"] * 85 + ["Side I"] * 5 + ["Side II"] * 10
    y_pred = ["Normal"] * 100

    per_class = per_class_f1(y_true, y_pred)
    result = macro_f1(y_true, y_pred)

    assert per_class["Side I"] == 0.0
    assert per_class["Side II"] == 0.0
    assert per_class["Normal"] == pytest.approx(0.9189189189189189, abs=1e-12)
    assert result == pytest.approx(0.9189189189189189 / 3.0, abs=1e-12)
    assert result == pytest.approx(0.3063063063063063, abs=1e-12)

    # The kit's qualitative point holds: accuracy 85%, macro F1 ~0.31.
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    assert accuracy == pytest.approx(0.85, abs=1e-12)
    assert result < 0.35


def test_end_to_end_agrees_with_per_class_average():
    """macro_f1 on labels == unweighted mean of per_class_f1 on the same labels.

    Ties the label-based path (what the pipeline calls) to the per-class path
    (what the worked example pins down), so the example verifies both.
    """
    y_true = ["Normal", "Normal", "Side I", "Side II", "Side II", "Normal"]
    y_pred = ["Normal", "Side I", "Side I", "Side II", "Normal", "Normal"]

    per_class = per_class_f1(y_true, y_pred)
    assert macro_f1(y_true, y_pred) == pytest.approx(
        macro_f1_from_per_class(list(per_class.values())), abs=1e-12
    )


def test_perfect_prediction_scores_one():
    y_true = ["Normal", "Side I", "Side II", "Normal"]
    assert macro_f1(y_true, list(y_true)) == pytest.approx(1.0, abs=1e-12)


def test_absent_class_still_occupies_a_denominator_slot():
    """A class missing from a fold must not shrink the denominator.

    With 14 Side I files across 272, a validation fold containing no Side I is
    realistic. If the absent class were dropped, the average would be over 2
    classes instead of 3 and the score would be silently inflated -- exactly
    the kind of optimistic number that zero leaderboard uploads guarantees
    nobody catches.
    """
    y_true = ["Normal", "Normal", "Side II"]
    y_pred = ["Normal", "Normal", "Side II"]

    per_class = per_class_f1(y_true, y_pred)
    assert set(per_class) == set(RAIL_CLASSES)
    assert per_class["Side I"] == 0.0

    # Normal and Side II are both perfect; a 2-class denominator would give 1.0
    assert macro_f1(y_true, y_pred) == pytest.approx(2.0 / 3.0, abs=1e-12)


def test_labels_are_the_three_info_kit_classes():
    """Guards the exact strings, including the space in 'Side I'/'Side II'."""
    assert RAIL_CLASSES == ("Normal", "Side I", "Side II")
