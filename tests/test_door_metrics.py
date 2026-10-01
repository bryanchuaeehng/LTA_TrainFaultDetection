"""Verification of the Door IoU-weighted F1 against the Info Kit's definition.

The Info Kit gives no worked numeric example for Door, unlike Rail. So instead
of reproducing a stated number, these tests pin down each behavioural clause of
section 4 separately, including every case where the metric differs from plain
F1. Those differences are the whole point of the metric; if they are wrong,
a submission could look fine locally and score badly.

Source: reference/03_References/Door/Door_Subsystem_Info_Kit.md section 4
"""

from __future__ import annotations

import pytest

from src.common.metrics import DOOR_CLASSES, door_iou_f1, segment_iou

N = "Normal"
A = "Abnormal resistance"


# --- segment_iou, the formula quoted verbatim at section 4.1 ---------------


def test_iou_identical_intervals_is_one():
    assert segment_iou(0.0, 10.0, 0.0, 10.0) == pytest.approx(1.0)


def test_iou_half_overlap():
    # true [0,10], pred [5,15]: intersection 5, union 15
    assert segment_iou(0.0, 10.0, 5.0, 15.0) == pytest.approx(5.0 / 15.0)


def test_iou_no_overlap_is_zero():
    assert segment_iou(0.0, 10.0, 20.0, 30.0) == 0.0


def test_iou_touching_endpoints_is_zero():
    """Abutting intervals share a point but no length, so IoU is 0."""
    assert segment_iou(0.0, 10.0, 10.0, 20.0) == 0.0


def test_iou_contained_interval():
    # true [0,10], pred [2,8]: intersection 6, union 10
    assert segment_iou(0.0, 10.0, 2.0, 8.0) == pytest.approx(0.6)


def test_iou_zero_length_union_is_zero():
    assert segment_iou(5.0, 5.0, 5.0, 5.0) == 0.0


# --- door_iou_f1, section 4.2 ---------------------------------------------


def test_perfect_submission_scores_one():
    """Info Kit 4.2: 'soft_recall = soft_precision = 1.0, score = 1.0'."""
    true = [(0.0, 10.0, N), (20.0, 30.0, A)]
    assert door_iou_f1(true, list(true)) == pytest.approx(1.0)


def test_wrong_label_with_perfect_overlap_scores_zero():
    """The clause that makes this not plain IoU.

    Section 4.1: 'A segment with perfectly overlapping timing but the wrong
    label ... cannot match at all — it contributes nothing, exactly as if it
    weren't submitted.' Section 4.2 adds that it 'scores exactly the same as
    missing that segment entirely and predicting a spurious extra one'.
    """
    true = [(0.0, 10.0, N)]
    pred = [(0.0, 10.0, A)]
    assert door_iou_f1(true, pred) == 0.0


def test_missing_segment_lowers_recall():
    """Two true, one found perfectly: recall 0.5, precision 1.0 -> F1 2/3."""
    true = [(0.0, 10.0, N), (20.0, 30.0, N)]
    pred = [(0.0, 10.0, N)]
    assert door_iou_f1(true, pred) == pytest.approx(2 / 3)


def test_spurious_segment_lowers_precision():
    """Over-segmenting is penalised, not free (section 4.2)."""
    true = [(0.0, 10.0, N)]
    pred = [(0.0, 10.0, N), (50.0, 60.0, N)]
    assert door_iou_f1(true, pred) == pytest.approx(2 / 3)


def test_sloppy_boundaries_reduce_the_score_without_being_a_miss():
    """Right cycle, right label, loose timing: still a match, worth less."""
    true = [(0.0, 10.0, N)]
    pred = [(2.0, 8.0, N)]  # IoU 0.6
    # recall = precision = 0.6, harmonic mean = 0.6
    assert door_iou_f1(true, pred) == pytest.approx(0.6)


def test_credit_is_the_iou_not_a_flat_point():
    """A loose match must score strictly less than a tight one."""
    true = [(0.0, 10.0, N)]
    tight = door_iou_f1(true, [(0.0, 10.0, N)])
    loose = door_iou_f1(true, [(4.0, 14.0, N)])
    assert loose < tight
    assert loose > 0.0


def test_matching_is_one_to_one_greedy_highest_iou_first():
    """Two predictions overlap one true segment; only the better one matches.

    Section 4.1 rule 3: 'once a segment (true or predicted) is used, it's
    removed from further consideration.' The weaker overlap becomes a false
    positive rather than a second partial match.
    """
    true = [(0.0, 10.0, N)]
    pred = [(0.0, 10.0, N), (9.0, 19.0, N)]  # IoU 1.0 and 0.05
    # only the 1.0 match counts: recall 1.0, precision 1/2
    assert door_iou_f1(true, pred) == pytest.approx(2 * 1.0 * 0.5 / 1.5)


def test_greedy_order_matters_and_picks_the_better_pairing():
    """A worse-overlapping pair must not claim a segment the better one needs."""
    true = [(0.0, 10.0, N), (10.0, 20.0, N)]
    # pred A overlaps true[0] strongly; pred B overlaps both but true[1] more
    pred = [(0.0, 10.0, N), (11.0, 20.0, N)]
    score = door_iou_f1(true, pred)
    # true[0]<->pred[0] IoU 1.0; true[1]<->pred[1] IoU 9/10
    expected_sum = 1.0 + 0.9
    r = expected_sum / 2
    p = expected_sum / 2
    assert score == pytest.approx(2 * r * p / (r + p))


def test_no_overlap_at_all_scores_zero():
    true = [(0.0, 10.0, N)]
    pred = [(100.0, 110.0, N)]
    assert door_iou_f1(true, pred) == 0.0


def test_empty_prediction_scores_zero():
    assert door_iou_f1([(0.0, 10.0, N)], []) == 0.0


def test_empty_truth_scores_zero():
    assert door_iou_f1([], [(0.0, 10.0, N)]) == 0.0


def test_mixed_labels_match_independently():
    """Labels partition the matching; a Normal cannot borrow an Abnormal."""
    true = [(0.0, 10.0, N), (20.0, 30.0, A)]
    pred = [(0.0, 10.0, A), (20.0, 30.0, N)]  # both labels swapped
    assert door_iou_f1(true, pred) == 0.0


def test_reduces_to_plain_f1_when_boundaries_are_exact():
    """The case that actually holds on our data (docs/plans/door.md section 1).

    With every IoU exactly 1.0, iou_sum is the match count, so soft_recall and
    soft_precision become ordinary recall and precision. Here: 3 true, 3
    predicted, 2 correctly labelled -> precision = recall = 2/3.
    """
    true = [(0.0, 10.0, N), (20.0, 30.0, A), (40.0, 50.0, A)]
    pred = [(0.0, 10.0, N), (20.0, 30.0, A), (40.0, 50.0, N)]
    assert door_iou_f1(true, pred) == pytest.approx(2 / 3)


def test_door_classes_are_the_two_info_kit_labels():
    """Guards the exact strings, including the space and lowercase 'r'."""
    assert DOOR_CLASSES == ("Normal", "Abnormal resistance")
