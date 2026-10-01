"""GATE 1b - the declared, one-time Test-set domain check (plan section 4.8).

This is a DELIBERATE, DECLARED, BOUNDED use of Test *input*. It reads
**column 1 only**, via ``usecols=[0]``, of each of the 68 Test files, and
computes exactly the six items enumerated in plan section 4.8.2 and nothing
else.

FORBIDDEN here, per plan section 4.8.3, and none of it is done:
  * reading columns 2-129 of any Test file;
  * feature extraction on Test;
  * fitting anything on Test - no scaler, imputer, selector, threshold, prior
    or calibration;
  * using the result to select a model, or to change V_MIN, V_CUT, the
    wavelength bands, the exclusion rule or the low-speed inference rule, all
    of which are fixed by the plan BEFORE this runs;
  * any expansion beyond the six items;
  * printing per-file Test values. Aggregate statistics and counts only.

What the result may inform is exactly two REPORTING questions (section 4.8.4):
whether the low-speed inference rule ever fires on Test, and whether the
restricted evaluation domain matches the deployment domain. Nothing else.

Run:  python -m src.rail.gate1b_test_domain
"""

from __future__ import annotations

import numpy as np

from src.common.paths import RAIL_TEST_DIR
from src.rail.constants import V_CUT, V_MIN
from src.rail.speed import list_csv_files, read_speed_column, speed_from_column1


def main() -> None:
    print("=" * 72)
    print("GATE 1b - Test-set domain check (plan section 4.8)")
    print("=" * 72)
    print("DECLARED Test access: column 1 only, read via usecols=[0].")
    print("No columns 2-129 read. Nothing fitted. No model choice informed.")
    print(f"V_MIN = {V_MIN} m/s   V_CUT = {V_CUT} m/s   (both fixed before this ran)")
    print()

    paths = list_csv_files(RAIL_TEST_DIR)
    print(f"Test files enumerated: {len(paths)}")

    speeds: list[float] = []
    transitions: list[int] = []
    non_binary_files = 0

    for path in paths:
        col1 = read_speed_column(path)
        distinct = np.unique(col1)
        if not np.all(np.isin(distinct, (0, 1))):
            non_binary_files += 1
        v_mps, T = speed_from_column1(col1)
        speeds.append(v_mps)
        transitions.append(T)

    v = np.asarray(speeds, dtype=float)
    T_arr = np.asarray(transitions, dtype=int)

    # --- Item 1: count with v == 0 exactly ---------------------------------
    n_zero = int(np.count_nonzero(v == 0.0))
    print()
    print(f"[1] Test files with v == 0 exactly (zero transitions): {n_zero} / {len(v)}")

    # --- Item 2: count with 0 < v < V_MIN ----------------------------------
    n_near_zero = int(np.count_nonzero((v > 0.0) & (v < V_MIN)))
    print(f"[2] Test files with 0 < v < V_MIN ({V_MIN} m/s):        {n_near_zero} / {len(v)}")
    print(f"    => total below V_MIN (low-speed rule would fire):   {n_zero + n_near_zero} / {len(v)}")

    # --- Item 3: counts above / below V_CUT --------------------------------
    n_above_cut = int(np.count_nonzero(v >= V_CUT))
    n_below_cut = int(len(v) - n_above_cut)
    print(f"[3] Test files with v >= V_CUT ({V_CUT} m/s):           {n_above_cut} / {len(v)}")
    print(f"    Test files with v <  V_CUT:                         {n_below_cut} / {len(v)}")

    # --- Item 4: five-number summary of v ----------------------------------
    q = np.percentile(v, [0, 25, 50, 75, 100])
    print("[4] Five-number summary of derived speed across the 68 Test files:")
    print("      stat    |    m/s    |   km/h")
    for name, value in zip(("min", "p25", "median", "p75", "max"), q):
        print(f"      {name:<7} | {value:9.4f} | {value * 3.6:8.3f}")

    # --- Item 5: transition count T summary --------------------------------
    print("[5] Transition count T: "
          f"min {int(T_arr.min())}, median {float(np.median(T_arr)):.1f}, max {int(T_arr.max())}")

    # --- Item 6: binary confirmation ---------------------------------------
    print(f"[6] Test files whose column 1 contains any non-binary value: {non_binary_files} / {len(v)}")
    print(f"    Column 1 binary {{0,1}} in Test as in Train: "
          f"{'CONFIRMED' if non_binary_files == 0 else 'NOT CONFIRMED'}")

    print()
    print("End of the six items. Nothing further computed on Test.")
    print("=" * 72)


if __name__ == "__main__":
    main()
