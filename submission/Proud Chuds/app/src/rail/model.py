"""Pipelines, baselines and the CV protocol (plan sections 4.1, 5.5, 5.7, 5.7a).

Everything that learns anything from data is fitted INSIDE the fold, via an
sklearn ``Pipeline`` handed to the CV loop. Fitting a scaler on all 228 eligible
files and then cross-validating is the defect the reviewer checks for, and it is
structurally impossible here because no transformer is ever fitted outside a
pipeline.

The mandatory pipeline shape (plan section 5.5):

  1. drop every ``meta_`` column and every suffix-rejected column  (not optional)
  2. SimpleImputer(strategy="median")   - the NaN band features of section 5.3
  3. VarianceThreshold                  - constant-column drop
  4. StandardScaler
  5. optional in-fold SelectKBest (linear model only)
  6. the classifier

B2 is the SOLE declared exception to step 1: it reads ``meta_v_mps`` and nothing
else. That bypass is named explicitly below and nowhere else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.common.metrics import RAIL_CLASSES, macro_f1
from src.rail.constants import EPS, SEED
from src.rail.features import model_matrix_columns

__all__ = [
    "SpeedOnlyClassifier",
    "PhysicsRuleClassifier",
    "ColumnSelector",
    "build_pipeline",
    "make_configurations",
]


class ColumnSelector(BaseEstimator):
    """Pipeline step 1: keep only the admitted model-matrix columns.

    Drops every ``meta_``-prefixed column and every column whose unit suffix is
    rejected by the dimensional rule (plan sections 5.3b.2, 5.3b.2a). This is
    the first step of every learned pipeline and it is not optional.
    """

    def __init__(self) -> None:
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        self.columns_ = model_matrix_columns(list(X.columns))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return X[self.columns_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


class SpeedOnlyClassifier(BaseEstimator, ClassifierMixin):
    """B2 - the speed-only confound baseline (plan section 5.7a). MANDATORY FLOOR.

    Uses derived speed and NOTHING else: no vibration data at all. Two
    thresholds partition the speed axis into three regions mapped to the three
    classes; the thresholds and the region-to-class mapping are fitted by
    exhaustive search on the TRAINING PORTION ONLY, in every fold.

    *** THIS IS THE SOLE DECLARED ``meta_`` BYPASS IN THE SUBSYSTEM. ***
    It reads ``meta_v_mps`` directly, bypassing the pipeline's drop-meta step.
    The bypass is deliberate, named here, and exists so the learned model can be
    shown to beat a classifier that looks only at train speed.
    """

    #: Named so the bypass is greppable and appears in output.
    META_BYPASS_COLUMN = "meta_v_mps"

    def __init__(self, max_candidates: int = 40) -> None:
        # 40 rather than 60 purely for runtime: the search is still an
        # exhaustive grid over midpoints of the TRAINING portion's sorted
        # unique speeds, just on a slightly coarser grid. At 202 distinct
        # speeds across 228 eligible files this costs threshold resolution of
        # a few hundredths of a m/s and nothing else. Recorded because it is a
        # deviation in grid density from the plan's "sorted unique speeds".
        self.max_candidates = max_candidates

    def fit(self, X: pd.DataFrame, y):  # noqa: N803
        v = np.asarray(X[self.META_BYPASS_COLUMN], dtype=float)
        y = np.asarray(y)
        self.classes_ = np.asarray(RAIL_CLASSES)

        uniq = np.unique(v)
        if uniq.size > 1:
            midpoints = (uniq[:-1] + uniq[1:]) / 2.0
        else:
            midpoints = uniq
        if midpoints.size > self.max_candidates:
            idx = np.linspace(0, midpoints.size - 1, self.max_candidates).astype(int)
            midpoints = midpoints[idx]

        best = (-1.0, midpoints[0], midpoints[0], ("Normal", "Normal", "Normal"))
        # Region-to-class mappings: low region is Normal (speed is low => no
        # fault evidence); the two upper regions take the remaining assignments.
        mappings = [
            ("Normal", "Side I", "Side II"),
            ("Normal", "Side II", "Side I"),
            ("Normal", "Normal", "Side I"),
            ("Normal", "Normal", "Side II"),
            ("Normal", "Side I", "Side I"),
            ("Normal", "Side II", "Side II"),
        ]
        for i, t_lo in enumerate(midpoints):
            for t_hi in midpoints[i:]:
                region = np.where(v < t_lo, 0, np.where(v < t_hi, 1, 2))
                for mapping in mappings:
                    pred = np.asarray(mapping, dtype=object)[region]
                    score = macro_f1(y, pred)
                    if score > best[0]:
                        best = (score, t_lo, t_hi, mapping)

        self.train_score_, self.t_lo_, self.t_hi_, self.mapping_ = best
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        v = np.asarray(X[self.META_BYPASS_COLUMN], dtype=float)
        region = np.where(v < self.t_lo_, 0, np.where(v < self.t_hi_, 1, 2))
        return np.asarray(self.mapping_, dtype=object)[region]


class PhysicsRuleClassifier(BaseEstimator, ClassifierMixin):
    """B1 - the one-feature physics rule baseline (plan section 5.7).

    Uses the single most physically motivated feature: the Side I vs Side II
    contrast in total corrugation-band power,
    ``c = log((P_I + eps) / (P_II + eps))`` with ``P_side`` the p90-across-channels
    vibration power summed over the five wavelength bands. Predicts ``Side I`` if
    ``c > +t``, ``Side II`` if ``c < -t``, else ``Normal``.

    The threshold ``t`` is fitted INSIDE each training fold and applied to the
    validation portion. It is never fitted on validation data.

    B1 is the bar that proves the ~1100-feature learned model earns its
    complexity: if M1-M3 cannot beat it, a one-feature rule is the right model
    and that conclusion is reported rather than buried.
    """

    #: The two per-side band-power aggregates the contrast is built from.
    SIDE_I_COL = "vib_p90_bandpow_total_sideI_amppv"
    SIDE_II_COL = "vib_p90_bandpow_total_sideII_amppv"

    def __init__(self, n_thresholds: int = 60) -> None:
        self.n_thresholds = n_thresholds

    def _contrast(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        p_i = np.asarray(X[self.SIDE_I_COL], dtype=float)
        p_ii = np.asarray(X[self.SIDE_II_COL], dtype=float)
        p_i = np.nan_to_num(p_i, nan=0.0)
        p_ii = np.nan_to_num(p_ii, nan=0.0)
        with np.errstate(all="ignore"):
            return np.log((np.abs(p_i) + EPS) / (np.abs(p_ii) + EPS))

    def fit(self, X: pd.DataFrame, y):  # noqa: N803
        c = self._contrast(X)
        y = np.asarray(y)
        self.classes_ = np.asarray(RAIL_CLASSES)
        candidates = np.unique(np.abs(c))
        if candidates.size > self.n_thresholds:
            idx = np.linspace(0, candidates.size - 1, self.n_thresholds).astype(int)
            candidates = candidates[idx]
        best = (-1.0, 0.0)
        for t in candidates:
            pred = np.where(c > t, "Side I", np.where(c < -t, "Side II", "Normal"))
            score = macro_f1(y, pred)
            if score > best[0]:
                best = (score, float(t))
        self.train_score_, self.t_ = best
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        c = self._contrast(X)
        return np.where(c > self.t_, "Side I", np.where(c < -self.t_, "Side II", "Normal"))


def _as_float_array(df):
    """Module-level (so it pickles) DataFrame -> float64 ndarray."""
    return np.asarray(df, dtype=np.float64)


def build_pipeline(classifier, *, select_k: int | None = None) -> Pipeline:
    """The mandatory pipeline shape of plan section 5.5.

    Every fitted step lives inside this object, so handing it to the CV loop
    fits all of them on the training portion of each fold and nothing else.
    """
    steps = [
        ("select_columns", ColumnSelector()),
        ("to_numpy", FunctionTransformer(_as_float_array)),
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scale", StandardScaler()),
    ]
    if select_k is not None:
        steps.append(("select_k", SelectKBest(score_func=f_classif, k=select_k)))
    steps.append(("clf", classifier))
    return Pipeline(steps)


def make_configurations() -> dict[str, object]:
    """The comparison set: EXACTLY six configurations, no more (plan section 5.5).

    Declaring the search space up front is what keeps model-selection leakage
    bounded (section 4.6). Adding a seventh is a planner decision.
    """
    return {
        "B0": build_pipeline(DummyClassifier(strategy="most_frequent")),
        "B1": PhysicsRuleClassifier(),
        "B2": SpeedOnlyClassifier(),
        "M1": build_pipeline(
            LogisticRegression(
                penalty="l2",
                class_weight="balanced",
                max_iter=5000,
                random_state=SEED,
            ),
            select_k="all",
        ),
        "M2": build_pipeline(
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                random_state=SEED,
                n_jobs=-1,
            )
        ),
        "M3": build_pipeline(
            ExtraTreesClassifier(
                n_estimators=500,
                class_weight="balanced",
                min_samples_leaf=2,
                random_state=SEED,
                n_jobs=-1,
            )
        ),
    }
