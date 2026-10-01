"""Fit the final model on the full eligible Train set and persist it.

This is NOT validation. The cross-validated numbers come from
``src.rail.evaluate``; this module exists only to produce the single fitted
artefact ``predict()`` loads. Every transformer is fitted here on the eligible
TRAINING set and only ever APPLIED to Test (plan section 5.4).

Run:  python -m src.rail.train
"""

from __future__ import annotations

import joblib
import pandas as pd

from src.rail.constants import ARTIFACT_DIR, FEATURES_TRAIN_PATH, MODEL_PATH, V_MIN
from src.rail.model import make_configurations

__all__ = ["main"]

#: The ONE model family carried forward (see the Gate 3 brief: one model, no
#: hyperparameter search, no second family). M1 = regularised multinomial
#: logistic regression: strongest choice at p >> n, and its coefficients on the
#: contrast features are directly interpretable as "Side I minus Side II".
SELECTED_MODEL = "M1"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(FEATURES_TRAIN_PATH)
    eligible = train[train["meta_eligible"]].copy()

    print(f"fitting {SELECTED_MODEL} on the {len(eligible)} eligible Train files "
          f"(v >= V_MIN = {V_MIN})")
    model = make_configurations()[SELECTED_MODEL]
    model.fit(eligible, eligible["meta_label"].to_numpy())

    n_features = len(model.named_steps["select_columns"].columns_)
    print(f"model matrix: {n_features} features")

    joblib.dump(
        {
            "model": model,
            "name": SELECTED_MODEL,
            "feature_columns": model.named_steps["select_columns"].columns_,
            "n_train": int(len(eligible)),
        },
        MODEL_PATH,
    )
    print(f"wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()
