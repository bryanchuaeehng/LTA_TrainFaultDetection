"""Cross-validated evaluation (plan sections 4.1, 4.5, 4.7, 5.7a).

ONE splitter, TWO evaluations. The same 50 folds are scored twice: once over
the whole validation fold (``full``), once over the subset of that fold at or
above ``V_CUT`` (``restricted``). That keeps a single partition and therefore a
single source of fold-to-fold variance.

**The RESTRICTED macro F1 is the headline** (plan section 4.7.5): it is the only
domain in which a good score cannot be produced by the speed shortcut.

Everything that learns anything is fitted inside the fold, via the pipeline.
B1's threshold and B2's two thresholds are refitted in every fold on the
training portion only.

Run:  python -m src.rail.evaluate
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from src.common.metrics import RAIL_CLASSES, macro_f1, per_class_f1
from src.rail.constants import (
    FEATURES_TRAIN_PATH,
    MARGIN_MEAN_D,
    MARGIN_MIN_WIN_FOLDS,
    MARGIN_P10_D,
    N_REPEATS,
    N_SPLITS,
    PROXY_BAR_RESTRICTED,
    SEED,
    V_CUT,
)
from src.rail.model import make_configurations

__all__ = ["run_cv", "main"]

#: The configurations evaluated in this run. The Gate 3 brief caps this at
#: B0, B2 and ONE model - no hyperparameter search, no second family.
CONFIGS_THIS_RUN = ("B0", "B1", "B2", "M1")


def run_cv(eligible: pd.DataFrame) -> dict:
    """Score every configuration over the identical 50 folds, in both domains."""
    y = eligible["meta_label"].to_numpy()
    v = np.asarray(eligible["meta_v_mps"], dtype=float)

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
    )
    print(f"splitter: {splitter}")
    print(f"  n_splits={N_SPLITS}, n_repeats={N_REPEATS}, random_state={SEED}")

    configs = make_configurations()
    results: dict[str, dict[str, list]] = {
        name: {
            "full": [],
            "restricted": [],
            "full_per_class": [],
            "restricted_per_class": [],
            "pred_full": [],
            "true_full": [],
            "pred_restricted": [],
            "true_restricted": [],
        }
        for name in CONFIGS_THIS_RUN
    }
    fold_composition = []
    b2_thresholds = []
    b1_thresholds = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(eligible, y)):
        X_tr = eligible.iloc[tr_idx]
        X_va = eligible.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        mask = v[va_idx] >= V_CUT

        comp = {
            "fold": fold,
            "n_val": len(va_idx),
            "n_restricted": int(mask.sum()),
        }
        for cls in RAIL_CLASSES:
            comp[f"val_{cls}"] = int((y_va == cls).sum())
            comp[f"res_{cls}"] = int((y_va[mask] == cls).sum())
        fold_composition.append(comp)

        for name in CONFIGS_THIS_RUN:
            est = configs[name]
            est.fit(X_tr, y_tr)
            pred = np.asarray(est.predict(X_va), dtype=object)

            results[name]["full"].append(macro_f1(y_va, pred))
            results[name]["full_per_class"].append(per_class_f1(y_va, pred))
            results[name]["pred_full"].extend(pred.tolist())
            results[name]["true_full"].extend(y_va.tolist())

            if mask.any():
                results[name]["restricted"].append(macro_f1(y_va[mask], pred[mask]))
                results[name]["restricted_per_class"].append(
                    per_class_f1(y_va[mask], pred[mask])
                )
                results[name]["pred_restricted"].extend(pred[mask].tolist())
                results[name]["true_restricted"].extend(y_va[mask].tolist())

            if name == "B2":
                b2_thresholds.append((float(est.t_lo_), float(est.t_hi_)))
            if name == "B1":
                b1_thresholds.append(float(est.t_))

    return {
        "results": results,
        "composition": pd.DataFrame(fold_composition),
        "b2_thresholds": b2_thresholds,
        "b1_thresholds": b1_thresholds,
    }


def _summarise(scores: list[float]) -> str:
    a = np.asarray(scores, dtype=float)
    return (f"mean {a.mean():.4f}  std {a.std(ddof=1):.4f}  "
            f"p10 {np.percentile(a, 10):.4f}  p90 {np.percentile(a, 90):.4f}  "
            f"min {a.min():.4f}  max {a.max():.4f}")


def _paired_test(d: np.ndarray, label: str) -> bool:
    mean_d = float(d.mean())
    wins = int((d > 0).sum())
    p10 = float(np.percentile(d, 10))
    c1 = mean_d > MARGIN_MEAN_D
    c2 = wins >= MARGIN_MIN_WIN_FOLDS
    c3 = p10 > MARGIN_P10_D
    print(f"  vs {label}:  mean(d) {mean_d:+.4f}  std(d) {d.std(ddof=1):.4f}  "
          f"wins {wins}/{len(d)}  p10(d) {p10:+.4f}")
    print(f"    1) mean(d) > {MARGIN_MEAN_D}          : {'PASS' if c1 else 'FAIL'}")
    print(f"    2) d_i > 0 on >= {MARGIN_MIN_WIN_FOLDS}/50 folds : {'PASS' if c2 else 'FAIL'}")
    print(f"    3) p10(d) > {MARGIN_P10_D}           : {'PASS' if c3 else 'FAIL'}")
    verdict = c1 and c2 and c3
    print(f"    OVERALL vs {label}: {'PASS' if verdict else 'FAIL'}")
    return verdict


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 72)
    print("GATE 3 - cross-validated evaluation")
    print("=" * 72)

    train = pd.read_parquet(FEATURES_TRAIN_PATH)
    eligible = train[train["meta_eligible"]].copy()
    print(f"eligible set: {len(eligible)} files")
    print(f"configurations this run: {list(CONFIGS_THIS_RUN)} "
          f"(count = {len(CONFIGS_THIS_RUN)}; ONE model, no hyperparameter search)")
    print()

    out = run_cv(eligible)
    results = out["results"]
    comp = out["composition"]

    print(f"\nfolds completed: {len(results['M1']['full'])}")

    # --- AC-28: restricted composition per fold ---------------------------
    print()
    print("--- restricted-subset composition across the 50 folds ---")
    print(f"  restricted size: min {comp['n_restricted'].min()}, "
          f"median {comp['n_restricted'].median():.1f}, max {comp['n_restricted'].max()}")
    for cls in RAIL_CLASSES:
        print(f"  {cls:<8} per restricted fold: min {comp[f'res_{cls}'].min()}, "
              f"max {comp[f'res_{cls}'].max()}")
    missing = int(((comp[[f"res_{c}" for c in RAIL_CLASSES]] == 0).any(axis=1)).sum())
    print(f"  folds whose restricted subset is missing any class: {missing}/50")
    print()
    print("  first repeat (folds 0-4), validation composition:")
    for _, r in comp.head(5).iterrows():
        print(f"    fold {int(r['fold'])}: n_val {int(r['n_val'])} "
              f"(N {int(r['val_Normal'])}/SI {int(r['val_Side I'])}/SII {int(r['val_Side II'])})"
              f"  restricted {int(r['n_restricted'])} "
              f"(N {int(r['res_Normal'])}/SI {int(r['res_Side I'])}/SII {int(r['res_Side II'])})")

    # --- in-fold threshold evidence (AC-10, AC-33) ------------------------
    b2t = out["b2_thresholds"]
    print()
    print(f"--- B2 thresholds refitted per fold (first 5): "
          f"{[(round(a,2), round(b,2)) for a,b in b2t[:5]]}")
    print(f"    distinct B2 threshold pairs across 50 folds: {len(set(b2t))}")
    print(f"--- B1 thresholds refitted per fold (first 5): "
          f"{[round(t,3) for t in out['b1_thresholds'][:5]]}")
    print(f"    distinct B1 thresholds across 50 folds: {len(set(out['b1_thresholds']))}")

    # --- the score tables -------------------------------------------------
    for domain in ("restricted", "full"):
        tag = "RESTRICTED (v >= 9.70) -- THE HEADLINE" if domain == "restricted" else "FULL SET"
        print()
        print(f"=== {tag} ===")
        for name in CONFIGS_THIS_RUN:
            print(f"  {name}: {_summarise(results[name][domain])}")
        print(f"  per-class F1 ({domain}):")
        for name in CONFIGS_THIS_RUN:
            pcs = results[name][f"{domain}_per_class"]
            parts = []
            for cls in RAIL_CLASSES:
                a = np.asarray([p[cls] for p in pcs])
                parts.append(f"{cls} {a.mean():.3f}(min {a.min():.2f})")
            print(f"    {name}: " + "  ".join(parts))
        zeros = sum(1 for p in results["M1"][f"{domain}_per_class"] if p["Side I"] == 0.0)
        print(f"  M1 folds with Side I F1 == 0.0: {zeros}/50")

    # --- pooled confusion matrix + predicted-class counts (AC-17) ---------
    print()
    print("--- M1 pooled confusion matrix, restricted (rows=true, cols=pred) ---")
    yt = np.asarray(results["M1"]["true_restricted"])
    yp = np.asarray(results["M1"]["pred_restricted"])
    print("            " + "".join(f"{c:>10}" for c in RAIL_CLASSES))
    for t in RAIL_CLASSES:
        print(f"    {t:<8}" + "".join(
            f"{int(((yt == t) & (yp == p)).sum()):>10}" for p in RAIL_CLASSES))
    print(f"  pooled predicted-class counts (restricted): "
          f"{ {c: int((yp == c).sum()) for c in RAIL_CLASSES} }")
    ypf = np.asarray(results["M1"]["pred_full"])
    print(f"  pooled predicted-class counts (full):       "
          f"{ {c: int((ypf == c).sum()) for c in RAIL_CLASSES} }")

    # --- the three-condition paired margin test (section 5.7a, AC-35) ----
    print()
    print("=== PAIRED MARGIN TEST, RESTRICTED DOMAIN (plan section 5.7a) ===")
    m = np.asarray(results["M1"]["restricted"], dtype=float)
    verdicts = {}
    for base in ("B0", "B1", "B2"):
        b = np.asarray(results[base]["restricted"], dtype=float)
        verdicts[base] = _paired_test(m - b, base)

    # --- the proxy bar (user ruling) --------------------------------------
    print()
    print("=== THE PROXY BAR (user ruling) ===")
    headline = float(m.mean())
    print(f"  restricted headline (M1 mean over 50 folds): {headline:.4f}")
    print(f"  bar = leaking domwavelength proxy:           {PROXY_BAR_RESTRICTED:.4f}")
    print(f"  (for reference, raw v scored 0.3566 - NOT the bar)")
    print(f"  VERDICT: headline {'BEATS' if headline > PROXY_BAR_RESTRICTED else 'DOES NOT BEAT'} "
          f"the proxy bar of {PROXY_BAR_RESTRICTED}")

    print()
    print(f"elapsed: {time.perf_counter() - t0:.1f} s")
    print("=" * 72)


if __name__ == "__main__":
    main()
