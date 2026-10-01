"""Leave-one-case-out validation of the locked ACV ensemble (notebook section 12).

Whole cases are held out, never timestamp rows -- the independent unit is a
fault case, not a 30-second reading.

    python -m acv.evaluate --train-dir <ACV/Train> --labels <ACV/Train_Labels.csv>
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone

from .features import (ROBUST_FEATURES, LOCKED_METHOD_WEIGHTS, extract_case_features,
                       add_domain_score, normalized_rank_score, window_aggregate,
                       rank_decay_score, normalise_car_id)
from .train import build_models, build_feature_table


def ensemble_case(train_features, infer_features, infer_raw, weights=LOCKED_METHOD_WEIGHTS):
    result = infer_features[["car"]].copy()
    result["domain"] = normalized_rank_score(add_domain_score(infer_features)["domain_score"]).values

    Xtr = train_features[ROBUST_FEATURES].copy()
    med = Xtr.median(numeric_only=True)
    Xtr = Xtr.fillna(med).fillna(0)
    Xin = infer_features[ROBUST_FEATURES].fillna(med).fillna(0)
    y = train_features["faulty"]

    for name, base in build_models().items():
        m = clone(base); m.fit(Xtr, y)
        result[name] = normalized_rank_score(m.predict_proba(Xin)[:, 1]).values

    for minutes, name in [(30, "window30"), (60, "window60")]:
        ws = window_aggregate(infer_raw, minutes=minutes)
        merged = result[["car"]].merge(ws[["car", "window_score"]], on="car", how="left")
        result[name] = normalized_rank_score(merged["window_score"].fillna(0.5)).values

    result["ensemble_score"] = sum(weights[n] * result[n] for n in weights)
    return result.sort_values("ensemble_score", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--labels", required=True)
    a = ap.parse_args()

    labels = pd.read_csv(a.labels)
    labels["faulty_car"] = labels["faulty_car"].map(normalise_car_id)
    feature_df, cases, raw = build_feature_table(a.train_dir, labels)

    rows = []
    for holdout in cases:
        tr = feature_df[~feature_df["filename"].eq(holdout)]
        va = feature_df[feature_df["filename"].eq(holdout)]
        true_car = va["faulty_car"].iloc[0]
        ranking = ensemble_case(tr, va, raw[holdout])
        ranked = ranking["car"].tolist()
        rows.append({"filename": holdout, "true_faulty_car": true_car,
                     "true_rank": ranked.index(true_car) + 1,
                     "rank_decay_score": rank_decay_score(true_car, ranked),
                     "ranked_cars": "|".join(ranked)})
        print("  %-20s true=%s rank=%d score=%.4f" %
              (holdout, true_car, rows[-1]["true_rank"], rows[-1]["rank_decay_score"]), flush=True)

    v = pd.DataFrame(rows)
    print("\nmean rank-decay score : %.4f" % v.rank_decay_score.mean())
    print("top-1 cases           : %d / %d" % ((v.true_rank == 1).sum(), len(v)))
    print("mean true rank        : %.2f" % v.true_rank.mean())
    return v


if __name__ == "__main__":
    main()
