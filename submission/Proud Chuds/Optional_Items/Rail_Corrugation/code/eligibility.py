"""AC-3b - realised eligible-set counts at the derived V_MIN, and B0.

Plan section 5.3a.3 / AC-3b: before any model is fitted the implementer MUST
compute and print, at ``V_MIN = 1.28`` exactly:

  * the number of Train files excluded, and their per-class breakdown;
  * the resulting eligible per-class counts;
  * the realised always-Normal macro F1 (B0) on the eligible set.

Those realised numbers supersede every figure in the plan.

MANDATORY ASSERTION: the excluded set must contain **zero Side I and zero
Side II** files. If it does not, this STOPS and returns to the planner - it
would contradict Gate 1's ``[0,2)`` histogram and mean the speed derivation had
changed.

Addition 1 (coordinator, Gate 2): the per-class breakdown of the excluded set is
also the EVIDENCE FOR the section 5.3a.4 low-speed inference rule, not merely a
count. "We predict Normal for low-speed inputs" is an assumption; "we predict
Normal for low-speed inputs, and every low-speed Train file is in fact Normal"
is a justified one.

This reads column 1 only. No channel data is touched.

Run:  python -m src.rail.eligibility
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.common.metrics import RAIL_CLASSES, macro_f1, per_class_f1
from src.common.paths import RAIL_TRAIN_DIR, RAIL_TRAIN_LABELS
from src.rail.constants import V_CUT, V_MIN
from src.rail.speed import derive_speed_for_file, file_index

__all__ = ["train_speed_table", "main"]


def train_speed_table() -> pd.DataFrame:
    """One row per Train file: name, index, label, derived v, T, eligibility.

    Column 1 only. Never touches columns 2-129.
    """
    labels = pd.read_csv(RAIL_TRAIN_LABELS)
    label_by_name = dict(zip(labels["filename"], labels["label"]))

    rows = []
    for name, label in label_by_name.items():
        path = RAIL_TRAIN_DIR / name
        v_mps, T = derive_speed_for_file(path)
        rows.append(
            {
                "filename": name,
                "index": file_index(path),
                "label": label,
                "v_mps": v_mps,
                "T": T,
            }
        )

    frame = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    frame["eligible"] = frame["v_mps"] >= V_MIN
    frame["restricted"] = frame["v_mps"] >= V_CUT
    return frame


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    counter = Counter(frame["label"])
    return {cls: int(counter.get(cls, 0)) for cls in RAIL_CLASSES}


def main() -> None:
    print("=" * 72)
    print("AC-3b - realised eligible counts at the DERIVED V_MIN, and B0")
    print("=" * 72)
    print("V_MIN is derived, not chosen (plan section 5.3a.2a):")
    print("    V_MIN = N * lambda_max / t = 2 * 0.64 m / 1.0 s = "
          f"{V_MIN} m/s  ({V_MIN * 3.6:.3f} km/h)")
    print()

    frame = train_speed_table()
    total = len(frame)
    print(f"Train files measured (column 1 only): {total}")

    full_counts = _counts(frame)
    print(f"Full-corpus label counts: "
          + ", ".join(f"{cls}: {n}" for cls, n in full_counts.items())
          + f", total {sum(full_counts.values())}")

    excluded = frame[~frame["eligible"]]
    eligible = frame[frame["eligible"]]

    exc_counts = _counts(excluded)
    eli_counts = _counts(eligible)

    print()
    print(f"--- Excluded (v < V_MIN = {V_MIN}) ---")
    print(f"  count: {len(excluded)}")
    print("  per-class: " + ", ".join(f"{cls}: {n}" for cls, n in exc_counts.items()))
    n_exact_zero = int((excluded["v_mps"] == 0.0).sum())
    print(f"  of which v == 0 exactly: {n_exact_zero}; 0 < v < V_MIN: "
          f"{len(excluded) - n_exact_zero}")
    if len(excluded):
        print(f"  fastest excluded file: {excluded['v_mps'].max():.4f} m/s")
    if len(eligible):
        print(f"  slowest retained file: {eligible['v_mps'].min():.4f} m/s")

    # The gap the plan asserts is empty (section 5.3a.3): nothing in [1.00, 1.28).
    in_gap = int(((frame["v_mps"] >= 1.00) & (frame["v_mps"] < V_MIN)).sum())
    print(f"  files in [1.00, {V_MIN}): {in_gap}   "
          f"(plan section 5.3a.3 asserts ZERO; threshold lands in an empty gap)")

    print()
    print("--- Eligible (training/CV population) ---")
    print("  per-class: " + ", ".join(f"{cls}: {n}" for cls, n in eli_counts.items()))
    print(f"  total: {len(eligible)}")

    # --- THE MANDATORY ASSERTION (AC-3b) ---------------------------------
    print()
    print("--- MANDATORY ASSERTION (AC-3b): zero fault files excluded ---")
    n_fault_excluded = exc_counts["Side I"] + exc_counts["Side II"]
    print(f"  Side I excluded:  {exc_counts['Side I']}")
    print(f"  Side II excluded: {exc_counts['Side II']}")
    if n_fault_excluded != 0:
        print("  RESULT: **FAILED**")
        raise SystemExit(
            f"STOP - AC-3b assertion FAILED: {n_fault_excluded} fault file(s) fall "
            f"below V_MIN = {V_MIN}. This contradicts Gate 1's [0,2) histogram and "
            f"would mean the speed derivation has changed. Returning to the planner "
            f"per plan section 5.3a.3; no model is fitted."
        )
    print("  RESULT: PASSED - the excluded set is entirely majority-class.")

    # --- Addition 1: evidence FOR the low-speed inference rule -------------
    print()
    print("--- Addition 1: empirical support for the section 5.3a.4 rule ---")
    n_normal_exc = exc_counts["Normal"]
    print(f"  Of the {len(excluded)} Train files below V_MIN, {n_normal_exc} are Normal.")
    print(f"  Breakdown: " + ", ".join(f"{cls}: {n}" for cls, n in exc_counts.items()))
    if len(excluded):
        frac = n_normal_exc / len(excluded)
        print(f"  => {n_normal_exc}/{len(excluded)} = {frac:.1%} of low-speed Train files are Normal.")
        if n_normal_exc == len(excluded):
            print("  The rule 'predict Normal below V_MIN' is therefore never once")
            print("  contradicted by a labelled low-speed Train file. This is EVIDENCE")
            print("  FOR the rule, not a claim that the rule is safe.")
        else:
            print("  *** FINDING: not all low-speed Train files are Normal. This")
            print("  *** UNDERCUTS the section 5.3a.4 inference rule and contradicts")
            print("  *** Gate 1's [0,2) histogram. Reporting loudly per the brief.")

    # --- B0, the always-Normal floor, on the realised eligible set ---------
    print()
    print("--- B0: always-Normal macro F1 on the REALISED eligible set ---")
    y_true = eligible["label"].tolist()
    y_pred = ["Normal"] * len(y_true)
    b0 = macro_f1(y_true, y_pred)
    b0_per_class = per_class_f1(y_true, y_pred)
    print("  per-class F1: " + ", ".join(f"{c}: {s:.6f}" for c, s in b0_per_class.items()))
    print(f"  B0 macro F1 (realised, eligible set): {b0:.6f}")
    print(f"  plan section 5.3a.3 expectation: ~0.3030")

    # For the record only, both other populations the plan names.
    b0_full = macro_f1(frame["label"].tolist(), ["Normal"] * total)
    print(f"  for reference, B0 on the full {total}: {b0_full:.6f}  (Gate 1 measured 0.3083)")

    # --- Restricted-domain composition, AC-28 first half ------------------
    restricted = eligible[eligible["restricted"]]
    res_counts = _counts(restricted)
    print()
    print(f"--- Restricted domain composition at V_CUT = {V_CUT} (AC-28, eligible only) ---")
    print("  per-class: " + ", ".join(f"{cls}: {n}" for cls, n in res_counts.items()))
    print(f"  total: {len(restricted)} of {len(eligible)} eligible "
          f"({len(restricted) / len(eligible):.1%})")
    print(f"  plan section 4.7.3 estimate: ~101 Normal / 14 Side I / 24 Side II, total ~139")

    # --- Speed by class, AC-14 -------------------------------------------
    print()
    print("--- AC-14: derived speed by class (full corpus, then eligible) ---")
    for name, subset in (("full", frame), ("eligible", eligible)):
        print(f"  [{name}]")
        for cls in RAIL_CLASSES:
            s = subset[subset["label"] == cls]["v_mps"]
            if len(s) == 0:
                continue
            print(f"    {cls:<8} n={len(s):>3}  min={s.min():8.4f}  med={s.median():8.4f}  "
                  f"max={s.max():8.4f} m/s   (med {s.median() * 3.6:.3f} km/h)")
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
