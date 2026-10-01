"""Nested-CV model search, per the pre-registration in docs/search-plan.md.

Read that file first. The configuration list was fixed before this ran and is
not extended here.

What this reports is the score of a SELECTION PROCEDURE, not of the best
configuration. For each outer fold, the eight configurations are scored on
inner folds carved from the outer training portion only; the inner winner is
refitted on the full outer-training portion and evaluated once on the held-out
outer fold. The mean of those outer scores is the nested estimate.

Per-configuration outer scores are also computed, purely as diagnostics for the
log. They are NOT a headline: the maximum over them is optimistically biased by
exactly the amount the search had room to overfit.

This module never writes to artifacts/rail/model.joblib or to predictions/.

    python -m src.rail.search
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.common.metrics import macro_f1
from src.rail.constants import ARTIFACT_DIR, SEED, V_CUT
from src.rail.features import model_matrix_columns

RULE = "=" * 78

N_SPLITS = 5
N_REPEATS = 10
INNER_SPLITS = 3

INCUMBENT = "C1"
MARGIN_MEAN = 0.05
MARGIN_WINS = 35
MARGIN_P10 = -0.05


def _pipe(clf):
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ]
    )


def configurations() -> dict[str, Pipeline]:
    """The eight pre-registered configurations. Not extended at run time."""
    return {
        "C1": _pipe(
            LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED
            )
        ),
        "C2": _pipe(
            LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=5000, random_state=SEED
            )
        ),
        "C3": _pipe(
            LogisticRegression(
                C=10.0, class_weight="balanced", max_iter=5000, random_state=SEED
            )
        ),
        "C4": _pipe(
            LogisticRegression(
                C=1.0,
                solver="saga",
                l1_ratio=1.0,
                class_weight="balanced",
                max_iter=3000,
                random_state=SEED,
            )
        ),
        "C5": _pipe(
            LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED)
        ),
        "C6": _pipe(
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=-1,
            )
        ),
        "C7": _pipe(
            ExtraTreesClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            )
        ),
        "C8": _pipe(RidgeClassifier(alpha=1.0, class_weight="balanced")),
    }


def load_matrix():
    tr = pd.read_parquet(ARTIFACT_DIR / "features_train.parquet")
    cols = model_matrix_columns(list(tr.columns))
    el = tr[tr["meta_eligible"].astype(bool)].reset_index(drop=True)
    X = el[cols].to_numpy(dtype=float)
    y = el["meta_label"].to_numpy()
    v = el["meta_v_mps"].to_numpy()
    return X, y, v, cols


def restricted_score(y_true, y_pred, v_fold) -> float | None:
    """Macro F1 on the restricted subset of a fold, or None if it is empty."""
    mask = v_fold >= V_CUT
    if mask.sum() == 0:
        return None
    return macro_f1(list(y_true[mask]), list(y_pred[mask]))


def main() -> int:
    t_start = time.perf_counter()
    print(RULE)
    print("NESTED-CV MODEL SEARCH — per docs/search-plan.md (pre-registered)")
    print(RULE)

    X, y, v, cols = load_matrix()
    print(f"eligible files : {len(y)}  features: {len(cols)}")
    print(f"class counts   : {dict(pd.Series(y).value_counts())}")
    print(f"restricted     : {int((v >= V_CUT).sum())} files at v >= {V_CUT}")
    print(f"seed           : {SEED}")
    print(f"outer          : RepeatedStratifiedKFold({N_SPLITS}x{N_REPEATS}) = 50 folds")
    print(f"inner          : StratifiedKFold({INNER_SPLITS}, shuffle=True)")
    print(f"incumbent      : {INCUMBENT} (restricted 0.7765 from the frozen run)")

    configs = configurations()
    print(f"\npre-registered configurations ({len(configs)}): {', '.join(configs)}")
    print("NOT extended at run time.\n")

    outer = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
    )

    nested_scores: list[float] = []
    chosen_counts: dict[str, int] = {k: 0 for k in configs}
    per_config_outer: dict[str, list[float]] = {k: [] for k in configs}
    inner_mean_log: list[dict] = []

    print(RULE)
    print("OUTER FOLDS")
    print(RULE)

    for fold_i, (tr_idx, te_idx) in enumerate(outer.split(X, y), start=1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te, v_te = X[te_idx], y[te_idx], v[te_idx]

        inner = StratifiedKFold(
            n_splits=INNER_SPLITS, shuffle=True, random_state=SEED
        )
        inner_means: dict[str, float] = {}

        for name, proto in configs.items():
            inner_scores = []
            for i_tr, i_te in inner.split(X_tr, y_tr):
                from sklearn.base import clone

                model = clone(proto)
                model.fit(X_tr[i_tr], y_tr[i_tr])
                pred = model.predict(X_tr[i_te])
                # Inner selection uses the restricted metric, matching the
                # headline, on the inner validation portion.
                s = restricted_score(y_tr[i_te], pred, v[tr_idx][i_te])
                if s is not None:
                    inner_scores.append(s)
            inner_means[name] = float(np.mean(inner_scores)) if inner_scores else 0.0

        winner = max(inner_means, key=inner_means.get)
        chosen_counts[winner] += 1
        inner_mean_log.append({"fold": fold_i, "winner": winner, **inner_means})

        # Refit the inner winner on the full outer-training portion, evaluate
        # ONCE on the untouched outer fold. This is the nested estimate.
        from sklearn.base import clone

        best = clone(configs[winner])
        best.fit(X_tr, y_tr)
        nested = restricted_score(y_te, best.predict(X_te), v_te)
        if nested is not None:
            nested_scores.append(nested)

        # Diagnostics only: every config's own outer score, for the log.
        for name, proto in configs.items():
            m = clone(proto)
            m.fit(X_tr, y_tr)
            s = restricted_score(y_te, m.predict(X_te), v_te)
            if s is not None:
                per_config_outer[name].append(s)

        if fold_i % 5 == 0 or fold_i == 1:
            print(
                f"  fold {fold_i:2d}/50  inner winner {winner}  "
                f"outer(nested) {nested:.4f}"
            )

    nested = np.array(nested_scores)

    print(f"\n{RULE}\nNESTED ESTIMATE — this is the headline for the procedure\n{RULE}")
    print(f"  mean   {nested.mean():.4f}")
    print(f"  std    {nested.std():.4f}")
    print(f"  p10    {np.percentile(nested, 10):.4f}")
    print(f"  p90    {np.percentile(nested, 90):.4f}")
    print(f"  min    {nested.min():.4f}")
    print(f"  max    {nested.max():.4f}")
    print(f"  folds  {len(nested)}")

    print(f"\n{RULE}\nINNER-FOLD SELECTION COUNTS (how often each config won)\n{RULE}")
    for name, n in sorted(chosen_counts.items(), key=lambda kv: -kv[1]):
        bar = "#" * n
        print(f"  {name}  chosen {n:2d}/50  {bar}")

    print(f"\n{RULE}\nPER-CONFIG OUTER SCORES — DIAGNOSTIC ONLY, NOT A HEADLINE\n{RULE}")
    print("  The maximum of this table is a biased estimate. It is logged")
    print("  because losers are evidence, not because it may be reported.\n")
    print(f"  {'cfg':<5} {'mean':>8} {'std':>8} {'p10':>8} {'min':>8} {'max':>8}")
    table = {}
    for name, scores in per_config_outer.items():
        a = np.array(scores)
        table[name] = a
        print(
            f"  {name:<5} {a.mean():>8.4f} {a.std():>8.4f} "
            f"{np.percentile(a, 10):>8.4f} {a.min():>8.4f} {a.max():>8.4f}"
        )

    print(f"\n{RULE}\nREPLACEMENT TEST vs INCUMBENT {INCUMBENT}\n{RULE}")
    inc = table[INCUMBENT]
    print(f"  incumbent {INCUMBENT} per-fold mean: {inc.mean():.4f}\n")
    print(
        f"  {'challenger':<11} {'mean(d)':>9} {'wins':>7} {'p10(d)':>9}  "
        f"{'c1':>3} {'c2':>3} {'c3':>3}  verdict"
    )

    any_pass = False
    for name in configs:
        if name == INCUMBENT:
            continue
        ch = table[name]
        n = min(len(ch), len(inc))
        d = ch[:n] - inc[:n]
        c1 = d.mean() > MARGIN_MEAN
        c2 = int((d > 0).sum()) >= MARGIN_WINS
        c3 = np.percentile(d, 10) > MARGIN_P10
        ok = c1 and c2 and c3
        any_pass = any_pass or ok
        print(
            f"  {name:<11} {d.mean():>+9.4f} {int((d > 0).sum()):>4}/{n:<2} "
            f"{np.percentile(d, 10):>+9.4f}  "
            f"{'Y' if c1 else 'n':>3} {'Y' if c2 else 'n':>3} {'Y' if c3 else 'n':>3}  "
            f"{'**PASS**' if ok else 'fail'}"
        )

    # The nested estimate against the incumbent, which is the comparison that
    # actually answers "is the search procedure better than just using M1".
    n = min(len(nested), len(inc))
    d_nested = nested[:n] - inc[:n]
    print(f"\n  nested procedure vs {INCUMBENT}:")
    print(f"    mean(d) {d_nested.mean():+.4f}   wins {int((d_nested > 0).sum())}/{n}")
    print(f"    p10(d)  {np.percentile(d_nested, 10):+.4f}")

    print(f"\n{RULE}\nVERDICT\n{RULE}")
    if any_pass:
        print("  At least one challenger passed all three conditions.")
        print("  The pre-registration requires the incumbent be replaced.")
        print("  NOTE: this script does not rewrite artefacts. Replacement is a")
        print("  deliberate follow-up step, reported for a human decision.")
    else:
        print("  NO challenger passed all three conditions.")
        print(f"  THE INCUMBENT {INCUMBENT} STANDS. Frozen artefacts unchanged.")
        print("  This is a real result: eight configurations, and the one already")
        print("  chosen is not beaten by a margin that survives paired testing.")

    out = ARTIFACT_DIR / "search_log.json"
    payload = {
        "nested_mean": float(nested.mean()),
        "nested_std": float(nested.std()),
        "nested_scores": [float(x) for x in nested],
        "chosen_counts": chosen_counts,
        "per_config_outer": {k: [float(x) for x in v_] for k, v_ in table.items()},
        "inner_selection_log": inner_mean_log,
        "incumbent": INCUMBENT,
        "any_challenger_passed": bool(any_pass),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  full log written to {out} (no losers deleted)")
    print(f"  wall clock: {time.perf_counter() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
