"""The door evidence margin: how close a call sat to the threshold.

These exist because the margin is shown to a technician deciding whether to
raise work. On the official test stream, flagged cycles span +0.04% to +36.5%
over the threshold; presenting those identically is the failure this guards
against.

The margin is presentation only. It must never alter a prediction.
"""

from __future__ import annotations

import pytest

from src.app.maintenance import (
    MARGINAL_BAND,
    evidence_label,
    evidence_note,
    margin_pct,
)

ABNORMAL = "Abnormal resistance"
NORMAL = "Normal"


def test_margin_is_zero_at_the_threshold():
    assert margin_pct(88402, 88402) == pytest.approx(0.0)


def test_margin_is_positive_above_and_negative_below():
    assert margin_pct(100000, 88402) > 0
    assert margin_pct(80000, 88402) < 0


def test_margin_matches_the_real_borderline_cycle():
    """The case that motivated this: 88438 against a threshold of 88402.

    36 units out of 88,000 -- a flagged fault on a 0.04% exceedance.
    """
    assert margin_pct(88438, 88402) == pytest.approx(0.0407, abs=1e-3)


def test_margin_matches_the_real_clear_cycle():
    assert margin_pct(130237, 100178) == pytest.approx(30.006, abs=1e-2)


def test_zero_threshold_does_not_divide_by_zero():
    assert margin_pct(1000, 0) == 0.0


def test_label_is_borderline_inside_the_band_either_side():
    assert evidence_label(0.04) == "Borderline"
    assert evidence_label(-0.08) == "Borderline"
    assert evidence_label(MARGINAL_BAND - 0.01) == "Borderline"
    assert evidence_label(-(MARGINAL_BAND - 0.01)) == "Borderline"


def test_label_is_clear_outside_the_band():
    assert evidence_label(36.5) == "Clear"
    assert evidence_label(-12.0) == "Clear"
    assert evidence_label(MARGINAL_BAND) == "Clear"


def test_borderline_fault_tells_the_technician_to_verify():
    note = evidence_note(0.04, ABNORMAL)
    assert "just over" in note.lower()
    assert "verify" in note.lower()


def test_clear_fault_does_not_hedge():
    note = evidence_note(36.5, ABNORMAL)
    assert "well above" in note.lower()
    assert "verify" not in note.lower()


def test_near_miss_normal_is_offered_as_worth_a_look():
    note = evidence_note(-0.08, NORMAL)
    assert "just under" in note.lower()


def test_clear_normal_is_not_flagged_for_attention():
    note = evidence_note(-30.0, NORMAL)
    assert "well below" in note.lower()
    assert "worth a look" not in note.lower()


def test_no_note_claims_a_probability():
    """The margin is a distance from a fitted threshold, not a calibrated
    probability, and the wording must not imply otherwise."""
    forbidden = ("probability", "confidence", "% likely", "chance")
    for margin in (-30.0, -0.08, 0.04, 36.5):
        for label in (NORMAL, ABNORMAL):
            note = evidence_note(margin, label).lower()
            for word in forbidden:
                assert word not in note, f"{note!r} implies calibration via {word!r}"
