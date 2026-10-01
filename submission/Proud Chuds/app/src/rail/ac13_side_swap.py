"""AC-13 - the side-swap symmetry invariant test (plan sections 5.2, 6).

The claim the contrast design makes: because ``A_I`` and ``A_II`` are computed by
IDENTICAL code over identically structured channel groups, exchanging the two
sides' channels must flip a ``Side I`` prediction to ``Side II`` and leave a
``Normal`` prediction alone. A flat 128-channel model simply does not possess
this invariant; this design does, so it is testable.

Procedure: permute each file's channels by the section 5.2 index map, re-extract
features from the permuted input, and compare the model's prediction on the
original against its prediction on the swapped copy.

Run on at least 10 Train files spanning all three classes, ALL OF THEM ELIGIBLE
so the section 5.3a.4 low-speed rule cannot short-circuit the test.

Partial failure is a reportable finding, not a silent pass.

Run:  python -m src.rail.ac13_side_swap
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.metrics import RAIL_CLASSES
from src.common.paths import RAIL_TRAIN_DIR
from src.rail.constants import FEATURES_TRAIN_PATH, SEED, V_MIN
from src.rail.features import (
    channel_car,
    channel_kind,
    channel_position,
    channel_side,
    extract_file,
    side_swap_permutation,
)
from src.rail.model import build_pipeline, make_configurations

__all__ = ["main"]

#: The expected prediction on the swapped copy, given the prediction on the
#: original. Normal is its own mirror; the two fault classes exchange.
_MIRROR = {"Normal": "Normal", "Side I": "Side II", "Side II": "Side I"}


def _print_channel_map() -> None:
    """AC-12: the side mapping, shown for the first and last few indices."""
    print("--- AC-12: channel index arithmetic (plan section 5.2) ---")
    print("  j -> car = j//16+1, pos = (j%16)//2+1, kind = vib if j even, "
          "side = Side I if pos odd")
    for j in list(range(6)) + [None] + list(range(122, 128)):
        if j is None:
            print("    ...")
            continue
        print(f"    j={j:<4} car={channel_car(j)}  pos={channel_position(j)}  "
              f"{channel_kind(j):<9} {channel_side(j)}")
    sides = [channel_side(j) for j in range(128)]
    kinds = [channel_kind(j) for j in range(128)]
    n_i = sides.count("Side I")
    n_ii = sides.count("Side II")
    print(f"  Side I channels:  {n_i}  "
          f"({sum(1 for j in range(128) if channel_side(j)=='Side I' and channel_kind(j)=='vibration')} vib + "
          f"{sum(1 for j in range(128) if channel_side(j)=='Side I' and channel_kind(j)=='shock')} shock)")
    print(f"  Side II channels: {n_ii}  "
          f"({sum(1 for j in range(128) if channel_side(j)=='Side II' and channel_kind(j)=='vibration')} vib + "
          f"{sum(1 for j in range(128) if channel_side(j)=='Side II' and channel_kind(j)=='shock')} shock)")
    assert n_i == 64 and n_ii == 64, "side split must be 64/64"
    assert kinds.count("vibration") == 64 and kinds.count("shock") == 64
    print("  assertion 64/64 with 32 vib + 32 shock each: PASSED")


def main() -> None:
    print("=" * 72)
    print("AC-13 - side-swap symmetry invariant")
    print("=" * 72)

    _print_channel_map()

    perm = side_swap_permutation()
    print()
    print(f"--- side-swap permutation: j -> j XOR 2, e.g. "
          f"{[(j, int(perm[j])) for j in range(4)]} ---")

    train = pd.read_parquet(FEATURES_TRAIN_PATH)
    eligible = train[train["meta_eligible"]].copy()
    y = eligible["meta_label"].to_numpy()

    # Fit M1 once on the full ELIGIBLE TRAIN set. This is a diagnostic model for
    # the invariant test only - it is not a validation score and no number from
    # it is reported as performance.
    model = make_configurations()["M1"]
    model.fit(eligible, y)
    print(f"diagnostic model: M1 fitted on the {len(eligible)} eligible Train files "
          f"(for the invariant only; not a validation score)")

    # --- pick >=10 eligible files spanning all three classes ---------------
    rng = np.random.default_rng(SEED)
    chosen: list[tuple[str, str]] = []
    for cls in RAIL_CLASSES:
        pool = eligible[eligible["meta_label"] == cls]["file_id"].tolist()
        take = min(4, len(pool))
        picks = rng.choice(pool, size=take, replace=False)
        chosen.extend((name, cls) for name in picks)

    print(f"files tested: {len(chosen)} "
          f"({', '.join(f'{c}={sum(1 for _, k in chosen if k == c)}' for c in RAIL_CLASSES)})"
          f"  - all eligible, so the low-speed rule cannot short-circuit")
    print()

    n_pass = 0
    n_fail = 0
    rows = []
    for name, true_cls in chosen:
        path = RAIL_TRAIN_DIR / name
        orig = pd.DataFrame([extract_file(path, swap_sides=False)])
        swap = pd.DataFrame([extract_file(path, swap_sides=True)])
        # Align to the fitted model's expected columns.
        orig = orig.reindex(columns=eligible.columns.drop("meta_label"), fill_value=np.nan)
        swap = swap.reindex(columns=eligible.columns.drop("meta_label"), fill_value=np.nan)

        p_orig = str(model.predict(orig)[0])
        p_swap = str(model.predict(swap)[0])
        expected = _MIRROR[p_orig]
        ok = p_swap == expected
        n_pass += ok
        n_fail += not ok
        rows.append((name, true_cls, p_orig, p_swap, expected, ok))

    print(f"{'file':<14}{'true':<10}{'pred':<10}{'pred(swap)':<12}{'expected':<10}{'':<6}")
    for name, true_cls, p_orig, p_swap, expected, ok in rows:
        print(f"{name:<14}{true_cls:<10}{p_orig:<10}{p_swap:<12}{expected:<10}"
              f"{'PASS' if ok else 'FAIL'}")

    print()
    print(f"AC-13 RESULT: {n_pass} passed, {n_fail} failed, out of {len(rows)}")
    if n_fail:
        print("*** PARTIAL FAILURE - reportable finding, NOT a silent pass. ***")
        print("*** The side-swap invariant is the structural property the       ***")
        print("*** contrast design of section 5.2 claims; a failure means the    ***")
        print("*** feature set is not side-antisymmetric as designed.            ***")
    else:
        print("The side-swap invariant holds on every file tested.")
    print("=" * 72)


if __name__ == "__main__":
    main()
