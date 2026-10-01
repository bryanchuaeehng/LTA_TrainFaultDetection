"""Rebuild the frozen ACV model bundle from the training cases.

Only needed if artifacts/acv/model_bundle.joblib is missing or must be
regenerated. Inference does not call this.

    python -m acv.train --train-dir <PS3/02_Datasets/ACV/Train> \
        --labels <PS3/02_Datasets/ACV/Train_Labels.csv> \
        --output artifacts/acv/model_bundle.joblib
"""
import argparse, json, os
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from .features import (CORE_STANDARD_PARAMETERS, ROBUST_FEATURES, LOCKED_METHOD_WEIGHTS,
                       RANDOM_STATE, extract_parameter_names, extract_case_features,
                       normalise_car_id)


def build_models():
    return {
        "logistic": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", C=0.5,
                                         max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "rf": RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=2,
                                     max_features="sqrt", class_weight="balanced",
                                     random_state=RANDOM_STATE),
        "extra": ExtraTreesClassifier(n_estimators=200, max_depth=3, min_samples_leaf=2,
                                      max_features="sqrt", class_weight="balanced",
                                      random_state=RANDOM_STATE),
    }


def standard_cases(train_dir):
    """Cases carrying the standard telemetry schema. Case 04's schema differs
    materially, so it is excluded rather than force-mapped."""
    out = []
    for p in sorted(Path(train_dir).glob("acv_case_*.xlsx")):
        params = extract_parameter_names(pd.read_excel(p, nrows=0).columns)
        if CORE_STANDARD_PARAMETERS.issubset(params):
            out.append(p.name)
    return sorted(out)


def build_feature_table(train_dir, labels):
    cases = standard_cases(train_dir)
    tables, raw = [], {}
    for fn in cases:
        df = pd.read_excel(Path(train_dir) / fn)
        raw[fn] = df
        faulty = labels.loc[labels["filename"].eq(fn), "faulty_car"].iloc[0]
        f = extract_case_features(df)
        f["filename"] = fn
        f["faulty_car"] = faulty
        f["faulty"] = f["car"].eq(faulty).astype(int)
        tables.append(f)
    return pd.concat(tables, ignore_index=True), cases, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output", default="artifacts/acv/model_bundle.joblib")
    a = ap.parse_args()

    labels = pd.read_csv(a.labels)
    labels["faulty_car"] = labels["faulty_car"].map(normalise_car_id)

    feature_df, cases, _ = build_feature_table(a.train_dir, labels)
    print("standard-schema cases:", cases)
    assert (feature_df.groupby("filename")["faulty"].sum() == 1).all()

    X = feature_df[ROBUST_FEATURES].copy()
    medians = X.median(numeric_only=True)
    Xc = X.fillna(medians).fillna(0)
    y = feature_df["faulty"]

    fitted = {}
    for name, base in build_models().items():
        m = clone(base); m.fit(Xc, y); fitted[name] = m

    bundle = {
        "models": fitted, "training_medians": medians,
        "model_features": ROBUST_FEATURES, "robust_features": ROBUST_FEATURES,
        "ensemble_weights": LOCKED_METHOD_WEIGHTS,
        "standard_training_cases": cases, "random_state": RANDOM_STATE,
    }
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    joblib.dump(bundle, a.output)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
