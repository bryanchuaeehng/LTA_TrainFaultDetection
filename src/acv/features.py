"""ACV feature extraction.

Lifted verbatim from the frozen notebook pipeline (ACV.ipynb sections 4-10) so
that training and inference share one implementation. Changing anything here
changes the model, so treat it as frozen alongside the bundle.
"""
"""Fixed feature extraction from the ACV branch notebook; no training side effects."""
import re
import numpy as np
import pandas as pd

CORE_STANDARD_PARAMETERS = {
    "ACV Control Temperature (Cooling)",
    "ACV Control Temperature (Heating)",
    "ACV Information Valid",
    "ACV Running Mode",
    "ACV Setting Mode",
    "Indoor Average Temperature",
    "Load Halved",
}

ROBUST_FEATURES = [
    "indoor_mean",
    "cooling_gap_mean",
    "cooling_gap_p90",
    "active_gap_mean",
    "rel_indoor_mean",
    "rel_gap_mean",
    "rel_indoor_pos_frac",
    "rel_gap_pos_frac",
]

LOCKED_METHOD_WEIGHTS = {
    "domain": 1 / 6, "logistic": 1 / 6, "rf": 1 / 6,
    "extra": 1 / 6, "window30": 1 / 6, "window60": 1 / 6,
}

RANDOM_STATE = 42


def normalise_car_id(value):
    m = re.search(r"(\d+)", str(value))
    if not m:
        raise ValueError(f"Could not parse car ID from {value!r}")
    return f"{int(m.group(1)):02d}"


def extract_car_ids(columns):
    cars = set()
ROBUST_FEATURES = ['indoor_mean', 'cooling_gap_mean', 'cooling_gap_p90', 'active_gap_mean', 'rel_indoor_mean', 'rel_gap_mean', 'rel_indoor_pos_frac', 'rel_gap_pos_frac']

def extract_car_ids(columns):
    cars = set()

    for col in columns:
        m = re.match(r"Car (\d{2}) - ", str(col))
        if m:
            cars.add(m.group(1))
    return sorted(cars)


def extract_parameter_names(columns):
    params = set()
    for col in columns:
        m = re.match(r"Car \d{2} - (.+)", str(col))
        if m:
            params.add(m.group(1))
    return params


def find_column(df, car_id, possible_names):
    prefix = f"Car {car_id} - "
    for name in possible_names:
        col = prefix + name
        if col in df.columns:
            return col
    return None


    return sorted(cars)

def find_column(df, car_id, possible_names):
    prefix = f"Car {car_id} - "

    for name in possible_names:
        col = prefix + name

        if col in df.columns:
            return col

    return None

def numeric_series(df, column, valid_mask=None):
    if column is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    s = pd.to_numeric(df[column], errors="coerce")
    if valid_mask is not None:
        s = s.where(valid_mask)
    return s


def extract_case_features(df):
    """One row per car: thermal, peer-relative and persistence features."""
    cars = extract_car_ids(df.columns)
    indoor, cooling_target, outdoor, running, valid, cooling_gap = {}, {}, {}, {}, {}, {}

    s = pd.to_numeric(df[column], errors="coerce")

    if valid_mask is not None:
        s = s.where(valid_mask)

    return s

def extract_case_features(df):
    cars = extract_car_ids(df.columns)

    indoor = {}
    cooling_target = {}
    outdoor = {}
    running = {}
    valid = {}
    cooling_gap = {}

    for car in cars:
        valid_col = find_column(df, car, ["ACV Information Valid"])
        valid_mask = df[valid_col].eq("Valid") if valid_col else pd.Series(True, index=df.index)
        valid[car] = valid_mask
        indoor[car] = numeric_series(df, find_column(df, car, ["Indoor Average Temperature"]), valid_mask)
        cooling_target[car] = numeric_series(
            df, find_column(df, car, ["ACV Control Temperature (Cooling)"]), valid_mask)
        outdoor[car] = numeric_series(df, find_column(
            df, car, ["Outdoor Average Temperature", "Outside Temperature Sensor Reading"]), valid_mask)
        running_col = find_column(df, car, ["ACV Running Mode"])
        running[car] = (df[running_col].where(valid_mask) if running_col
                        else pd.Series(np.nan, index=df.index, dtype=object))
        cooling_gap[car] = indoor[car] - cooling_target[car]

    peer_indoor_median = pd.DataFrame(indoor).median(axis=1, skipna=True)
    peer_gap_median = pd.DataFrame(cooling_gap).median(axis=1, skipna=True)

    rows = []
    for car in cars:
        ind, target, out = indoor[car], cooling_target[car], outdoor[car]
        gap, run = cooling_gap[car], running[car]
        active = run.isin(["Automatic Cooling", "Full Cooling"])
        active_gap = gap.where(active)
        rel_indoor = ind - peer_indoor_median
        rel_gap = gap - peer_gap_median
        rows.append({
            "car": car,
            "valid_fraction": valid[car].mean(),
            "indoor_mean": ind.mean(), "indoor_median": ind.median(),
            "indoor_std": ind.std(), "indoor_p90": ind.quantile(0.90),
            "cooling_target_mean": target.mean(),
            "cooling_gap_mean": gap.mean(), "cooling_gap_median": gap.median(),
            "cooling_gap_std": gap.std(), "cooling_gap_p90": gap.quantile(0.90),
            "cooling_gap_p95": gap.quantile(0.95),
            "active_gap_mean": active_gap.mean(), "active_gap_p90": active_gap.quantile(0.90),
            "outdoor_indoor_gap_mean": (out - ind).mean(),
            "rel_indoor_mean": rel_indoor.mean(), "rel_indoor_p90": rel_indoor.quantile(0.90),
            "rel_gap_mean": rel_gap.mean(), "rel_gap_p90": rel_gap.quantile(0.90),
            "rel_indoor_pos_frac": (rel_indoor > 0).where(rel_indoor.notna()).mean(),
            "rel_gap_pos_frac": (rel_gap > 0).where(rel_gap.notna()).mean(),

        valid[car] = valid_mask

        indoor[car] = numeric_series(
            df,
            find_column(df, car, ["Indoor Average Temperature"]),
            valid_mask,
        )

        cooling_target[car] = numeric_series(
            df,
            find_column(df, car, ["ACV Control Temperature (Cooling)"]),
            valid_mask,
        )

        outdoor[car] = numeric_series(
            df,
            find_column(
                df,
                car,
                [
                    "Outdoor Average Temperature",
                    "Outside Temperature Sensor Reading",
                ],
            ),
            valid_mask,
        )

        running_col = find_column(df, car, ["ACV Running Mode"])

        if running_col:
            running[car] = df[running_col].where(valid_mask)
        else:
            running[car] = pd.Series(np.nan, index=df.index, dtype=object)

        cooling_gap[car] = indoor[car] - cooling_target[car]

    indoor_df = pd.DataFrame(indoor)
    gap_df = pd.DataFrame(cooling_gap)

    # Median across the train at the same timestamp.
    peer_indoor_median = indoor_df.median(axis=1, skipna=True)
    peer_gap_median = gap_df.median(axis=1, skipna=True)

    rows = []

    for car in cars:
        ind = indoor[car]
        target = cooling_target[car]
        out = outdoor[car]
        gap = cooling_gap[car]
        run = running[car]

        active = run.isin(["Automatic Cooling", "Full Cooling"])
        active_gap = gap.where(active)

        rel_indoor = ind - peer_indoor_median
        rel_gap = gap - peer_gap_median

        rows.append({
            "car": car,

            "valid_fraction": valid[car].mean(),

            "indoor_mean": ind.mean(),
            "indoor_median": ind.median(),
            "indoor_std": ind.std(),
            "indoor_p90": ind.quantile(0.90),

            "cooling_target_mean": target.mean(),

            "cooling_gap_mean": gap.mean(),
            "cooling_gap_median": gap.median(),
            "cooling_gap_std": gap.std(),
            "cooling_gap_p90": gap.quantile(0.90),
            "cooling_gap_p95": gap.quantile(0.95),

            "active_gap_mean": active_gap.mean(),
            "active_gap_p90": active_gap.quantile(0.90),

            "outdoor_indoor_gap_mean": (out - ind).mean(),

            "rel_indoor_mean": rel_indoor.mean(),
            "rel_indoor_p90": rel_indoor.quantile(0.90),
            "rel_gap_mean": rel_gap.mean(),
            "rel_gap_p90": rel_gap.quantile(0.90),

            # Persistence: how often this car is above its peers.
            "rel_indoor_pos_frac": (rel_indoor > 0).where(rel_indoor.notna()).mean(),
            "rel_gap_pos_frac": (rel_gap > 0).where(rel_gap.notna()).mean(),

            "automatic_cooling_fraction": run.eq("Automatic Cooling").mean(),
            "full_cooling_fraction": run.eq("Full Cooling").mean(),
            "stop_fraction": run.eq("Stop").mean(),
        })
    return pd.DataFrame(rows)


def add_domain_score(case_features, selected_features=None):
    selected_features = selected_features or ROBUST_FEATURES
    scored = case_features.copy()
    pct_cols = []
    for feature in selected_features:
        c = f"{feature}_pct"
        if scored[feature].notna().any():
            scored[c] = scored[feature].rank(pct=True, ascending=True, method="average").fillna(0.5)
        else:
            scored[c] = 0.5
        pct_cols.append(c)
    scored["domain_score"] = scored[pct_cols].mean(axis=1)
    return scored


def normalized_rank_score(values):
    values = pd.Series(values)
    ranks = values.rank(ascending=False, method="average")
    n = len(values)
    if n <= 1:
        return pd.Series(np.ones(n), index=values.index)
    return (n - ranks) / (n - 1)


def rank_decay_score(true_car, ranked_cars):
    if true_car not in ranked_cars:
        return 0.0
    n = len(ranked_cars)
    r = ranked_cars.index(true_car) + 1
    return (n - (r - 1)) / n


    return pd.DataFrame(rows)

def add_domain_score(case_features, selected_features=None):
    selected_features = selected_features or ROBUST_FEATURES

    scored = case_features.copy()
    percentile_cols = []

    for feature in selected_features:
        pct_col = f"{feature}_pct"

        if scored[feature].notna().any():
            scored[pct_col] = (
                scored[feature]
                .rank(pct=True, ascending=True, method="average")
                .fillna(0.5)
            )
        else:
            scored[pct_col] = 0.5

        percentile_cols.append(pct_col)

    scored["domain_score"] = scored[percentile_cols].mean(axis=1)

    return scored

def normalized_rank_score(values):
    values = pd.Series(values)

    ranks = values.rank(
        ascending=False,
        method="average",
    )

    n = len(values)

    if n <= 1:
        return pd.Series(np.ones(n), index=values.index)

    # 1.0 = first, 0.0 = last.
    return (n - ranks) / (n - 1)

def window_aggregate(df, minutes, min_rows=20):
    data = df.copy()
    data["Time"] = pd.to_datetime(data["Time"])
    window_rows = []
    for _, window in data.groupby(pd.Grouper(key="Time", freq=f"{minutes}min")):
        if len(window) < min_rows:
            continue
        feats = extract_case_features(window)
        if feats.empty:
            continue
        scored = add_domain_score(feats)
        scored["top1"] = scored["domain_score"].eq(scored["domain_score"].max()).astype(float)
        window_rows.append(scored[["car", "domain_score", "top1"]])

    cars = extract_car_ids(df.columns)
    if not window_rows:
        return pd.DataFrame({"car": cars, "window_score": 0.5, "top1_freq": np.nan, "windows": 0})

    allw = pd.concat(window_rows, ignore_index=True)
    result = (allw.groupby("car")
              .agg(mean_window_domain=("domain_score", "mean"),
                   top1_freq=("top1", "mean"), windows=("top1", "size"))
              .reset_index())
    result["window_score"] = 0.70 * result["mean_window_domain"] + 0.30 * result["top1_freq"]

    window_rows = []

    for _, window in data.groupby(
        pd.Grouper(
            key="Time",
            freq=f"{minutes}min",
        )
    ):
        if len(window) < min_rows:
            continue

        features = extract_case_features(window)

        if features.empty:
            continue

        scored = add_domain_score(features)

        scored["top1"] = scored["domain_score"].eq(
            scored["domain_score"].max()
        ).astype(float)

        window_rows.append(
            scored[["car", "domain_score", "top1"]]
        )

    cars = extract_car_ids(df.columns)

    if not window_rows:
        return pd.DataFrame({
            "car": cars,
            "window_score": 0.5,
            "top1_freq": np.nan,
            "windows": 0,
        })

    all_windows = pd.concat(window_rows, ignore_index=True)

    result = (
        all_windows
        .groupby("car")
        .agg(
            mean_window_domain=("domain_score", "mean"),
            top1_freq=("top1", "mean"),
            windows=("top1", "size"),
        )
        .reset_index()
    )

    # Fixed combination, not tuned on the test case.
    result["window_score"] = (
        0.70 * result["mean_window_domain"]
        + 0.30 * result["top1_freq"]
    )

    return result
