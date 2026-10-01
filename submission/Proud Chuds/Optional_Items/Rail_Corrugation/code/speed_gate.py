"""The empirical speed-correlation gate (USER RULING, post-Gate-2).

WHY THIS EXISTS - the root cause, for the write-up.

The ``domwavelength`` family computed ``lambda = v / f_centroid``. Wherever
``f_centroid`` is near-constant across files - broadband shock with no dominant
spectral peak - ``lambda`` is almost exactly proportional to ``v``. So **the
metres mandate multiplied speed back in after mechanism 3 had divided it out**.
Measured |Spearman| against ``v`` reached 0.9982, and a depth-3 tree on that one
feature scored 0.3984 restricted macro F1, beating both B0 (0.2806) and raw
``v`` itself (0.3566) - the very feature plan section 5.3b struck.

The unit-suffix contract passed those columns as ``_m`` **correctly**: they ARE
metres. The lesson is that **the suffix contract checks DIMENSION, which is a
property of the formula, whereas leakage is a property of the DATA.** A static
naming rule cannot catch this. Only an empirical check against the data can,
which is what this module is.

WHAT IT DOES

For every feature surviving the suffix contract and the ``domwavelength`` ban,
compute ``|Spearman|`` against derived speed on the **restricted** eligible
training set (``v >= V_CUT``, the 139 files) and drop anything above
``SPEED_CORR_THRESHOLD``.

Restriction is deliberate: above the cut is where a speed proxy is most
dangerous, because it is the domain the headline metric is computed on. The
finding that drove this ruling was that restriction makes correlation WORSE
(49 features above 0.9 restricted, versus 6 on the full set), so gating on the
full set would have missed most of them.

TWO PROPERTIES THE REVIEWER SHOULD SEE STATED PLAINLY:

* **The 0.90 threshold is CHOSEN PRAGMATICALLY, NOT DERIVED.** No argument
  from physics or statistics fixes it.
* **The gate is a FIXED FEATURE-SET DECISION computed ONCE on training data,
  not a per-fold fit.** It uses labels nowhere - only speeds - so it is not
  label leakage; but it does see the whole eligible restricted training set, and
  that is stated rather than hidden.

Run:  python -m src.rail.speed_gate
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.rail.constants import (
    ADMISSIBLE_SUFFIXES,
    ARTIFACT_DIR,
    BANNED_FEATURE_STEMS,
    FEATURES_TRAIN_PATH,
    META_PREFIX,
    SPEED_CORR_PATH,
    SPEED_CORR_THRESHOLD,
    V_CUT,
)

__all__ = ["fit_speed_gate", "main"]


def _candidate_columns(names: list[str]) -> list[str]:
    """Columns after the suffix contract and the domwavelength ban, before the gate."""
    return [
        name
        for name in names
        if not name.startswith(META_PREFIX)
        and name.endswith(ADMISSIBLE_SUFFIXES)
        and not any(stem in name for stem in BANNED_FEATURE_STEMS)
    ]


def fit_speed_gate(train: pd.DataFrame) -> dict:
    """Compute the gate on the restricted eligible TRAINING set and persist it."""
    restricted = train[train["meta_eligible"] & train["meta_restricted"]]
    v = np.asarray(restricted["meta_v_mps"], dtype=float)

    candidates = _candidate_columns(list(train.columns))
    dropped: list[str] = []
    correlations: dict[str, float] = {}

    for name in candidates:
        x = np.asarray(restricted[name], dtype=float)
        finite = np.isfinite(x)
        if finite.sum() < 10 or np.unique(x[finite]).size < 2:
            rho = 0.0
        else:
            rho = sp_stats.spearmanr(x[finite], v[finite]).statistic
            if not np.isfinite(rho):
                rho = 0.0
        correlations[name] = abs(float(rho))
        if abs(rho) > SPEED_CORR_THRESHOLD:
            dropped.append(name)

    payload = {
        "threshold": SPEED_CORR_THRESHOLD,
        "threshold_note": (
            "CHOSEN PRAGMATICALLY, NOT DERIVED. Fixed feature-set decision "
            "computed ONCE on the restricted eligible training set; not a "
            "per-fold fit. Uses speeds only, never labels."
        ),
        "domain": f"restricted eligible training set (v >= V_CUT = {V_CUT})",
        "n_restricted_files": int(len(restricted)),
        "n_candidates": len(candidates),
        "n_dropped": len(dropped),
        "dropped": sorted(dropped),
        "max_abs_spearman_kept": max(
            (c for n, c in correlations.items() if n not in set(dropped)), default=0.0
        ),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    SPEED_CORR_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload | {"correlations": correlations}


def main() -> None:
    print("=" * 72)
    print("EMPIRICAL SPEED-CORRELATION GATE (user ruling, post-Gate-2)")
    print("=" * 72)
    print("Root cause: lambda = v / f_centroid meant the metres mandate")
    print("multiplied speed back in after mechanism 3 had divided it out.")
    print("The suffix contract checks DIMENSION (a property of the formula);")
    print("leakage is a property of the DATA. Only an empirical check catches it.")
    print()

    train = pd.read_parquet(FEATURES_TRAIN_PATH)
    all_feature_cols = [
        c for c in train.columns
        if not c.startswith(META_PREFIX) and c not in ("file_id", "meta_label")
    ]
    n_suffix_ok = len([c for c in all_feature_cols if c.endswith(ADMISSIBLE_SUFFIXES)])
    n_banned = len([c for c in all_feature_cols
                    if any(s in c for s in BANNED_FEATURE_STEMS)])

    print(f"[1] after suffix contract:                 {n_suffix_ok}")
    print(f"[2] DROPPED: domwavelength family (banned): {n_banned}")
    print(f"    -> remaining:                           {n_suffix_ok - n_banned}")

    result = fit_speed_gate(train)

    print()
    print(f"[3] correlation gate, |Spearman| vs v > {SPEED_CORR_THRESHOLD}")
    print(f"    domain: {result['domain']}, n = {result['n_restricted_files']} files")
    print(f"    THRESHOLD IS CHOSEN PRAGMATICALLY, NOT DERIVED.")
    print(f"    Fixed feature-set decision computed ONCE on training data,")
    print(f"    NOT a per-fold fit. Uses speeds only, never labels.")
    print(f"    candidates examined: {result['n_candidates']}")
    print(f"    **DROPPED BY THE GATE: {result['n_dropped']}**")
    print(f"    max |Spearman| among KEPT features: {result['max_abs_spearman_kept']:.4f}")

    final = result["n_candidates"] - result["n_dropped"]
    print()
    print(f"FINAL MODEL MATRIX: {final} features")
    print(f"  (1104 after suffix contract -> {n_suffix_ok - n_banned} after "
          f"domwavelength ban -> {final} after correlation gate)")

    corrs = result["correlations"]
    top = sorted(corrs.items(), key=lambda kv: -kv[1])[:10]
    print()
    print("  10 highest |Spearman| among candidates (dropped ones marked):")
    dropped_set = set(result["dropped"])
    for name, c in top:
        print(f"    {c:.4f}  {'DROP' if name in dropped_set else 'keep'}  {name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
