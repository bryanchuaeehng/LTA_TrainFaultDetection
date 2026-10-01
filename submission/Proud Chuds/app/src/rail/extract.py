"""Streaming feature extraction to the Parquet cache (plan section 5.4).

ONE FILE AT A TIME. Open it, read it, compute its feature vector, discard the
raw array, move on. Peak memory is one file's array (10,000 x 129 float32,
~5.2 MB) plus the accumulating feature list of a few hundred kilobytes. No list
of raw DataFrames is ever built and ``pd.concat`` never sees raw data.

4.5 GB of raw Train CSV collapses to under 2 MB of features. Once the cache
exists, the 50 CV fits never open a raw file again.

Features are extracted for ALL 272 Train files, not only the eligible ones
(plan section 5.4): the excluded files' ``meta_`` values document the exclusion,
and a future change to ``V_MIN`` must not require a 4.5 GB re-read. Eligibility
is recorded as the boolean column ``meta_eligible`` so the filter is explicit.

Run:  python -m src.rail.extract
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.common.paths import RAIL_TEST_DIR, RAIL_TRAIN_DIR, RAIL_TRAIN_LABELS
from src.rail.constants import (
    ADMISSIBLE_SUFFIXES,
    ARTIFACT_DIR,
    EPS,
    FEATURE_META_PATH,
    FEATURE_VERSION,
    FEATURES_TEST_PATH,
    FEATURES_TRAIN_PATH,
    FS_HZ,
    META_PREFIX,
    REJECTED_SUFFIXES,
    UNIT_SUFFIXES,
    V_CUT,
    V_MIN,
    V_MIN_LAMBDA_MAX_M,
    V_MIN_N_PERIODS,
    V_MIN_WINDOW_S,
    WAVELENGTH_BANDS_M,
    WELCH_DETREND,
    WELCH_NOVERLAP,
    WELCH_NPERSEG,
    WELCH_WINDOW,
)
from src.rail.features import (
    extract_file,
    model_matrix_columns,
    validate_feature_names,
)
from src.rail.speed import file_index, list_csv_files

__all__ = ["extract_directory", "load_cache", "cache_is_current", "main"]


def _extract_one(path: Path) -> dict[str, float]:
    row = extract_file(path)
    row["file_id"] = path.name
    return row


def extract_directory(directory: Path, *, n_jobs: int = -1) -> pd.DataFrame:
    """Stream over every CSV in ``directory`` and return the feature frame.

    Files are enumerated in natural NUMERIC order. Parallelism over files is
    permitted (each file is independent) and must not change the output, so
    results are reassembled into that deterministic order before returning.
    """
    paths = list_csv_files(directory)
    rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_extract_one)(path) for path in paths
    )
    # Reassemble deterministically regardless of completion order.
    rows = sorted(rows, key=lambda r: file_index(Path(r["file_id"])))

    frame = pd.DataFrame(rows)
    feature_cols = [c for c in frame.columns if c != "file_id"]

    # THE HARD ERROR (plan section 5.3b.2a) fires here, before anything is
    # written to the cache.
    validate_feature_names(feature_cols)

    frame["meta_eligible"] = frame["meta_v_mps"] >= V_MIN
    frame["meta_restricted"] = frame["meta_v_mps"] >= V_CUT

    ordered = ["file_id"] + sorted(c for c in frame.columns if c != "file_id")
    return frame[ordered]


def _write_meta(train_shape: tuple[int, int], test_shape: tuple[int, int]) -> None:
    meta = {
        "feature_version": FEATURE_VERSION,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "train_shape": list(train_shape),
        "test_shape": list(test_shape),
        "V_MIN": V_MIN,
        "V_MIN_derivation": {
            "formula": "V_MIN = N * lambda_max / t",
            "N": V_MIN_N_PERIODS,
            "lambda_max_m": V_MIN_LAMBDA_MAX_M,
            "t_s": V_MIN_WINDOW_S,
            "result_mps": V_MIN,
            "note": "Derived, not chosen. Plan section 5.3a.2a; N is the "
                    "assumption at section 7.14.",
        },
        "V_CUT": V_CUT,
        "wavelength_bands_m": [list(b) for b in WAVELENGTH_BANDS_M],
        "welch": {
            "nperseg": WELCH_NPERSEG,
            "noverlap": WELCH_NOVERLAP,
            "window": WELCH_WINDOW,
            "fs": FS_HZ,
            "detrend": WELCH_DETREND,
        },
        "eps": EPS,
        "unit_suffixes": list(UNIT_SUFFIXES),
        "admissible_suffixes": list(ADMISSIBLE_SUFFIXES),
        "rejected_suffixes": list(REJECTED_SUFFIXES),
    }
    FEATURE_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def cache_is_current() -> bool:
    """True when the cache exists AND its version matches this extractor.

    A cache whose version does not match is REBUILT, not used. Silent reuse of
    a stale cache would make every downstream number wrong.
    """
    if not (
        FEATURE_META_PATH.exists()
        and FEATURES_TRAIN_PATH.exists()
        and FEATURES_TEST_PATH.exists()
    ):
        return False
    try:
        meta = json.loads(FEATURE_META_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("feature_version") == FEATURE_VERSION


def load_cache() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the cached feature matrices, rebuilding first if stale."""
    if not cache_is_current():
        print(f"[cache] version mismatch or missing -> REBUILDING")
        main()
    meta = json.loads(FEATURE_META_PATH.read_text(encoding="utf-8"))
    print(f"[cache] loaded cache, version {meta['feature_version']}; raw files read: 0")
    return (
        pd.read_parquet(FEATURES_TRAIN_PATH),
        pd.read_parquet(FEATURES_TEST_PATH),
    )


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GATE 2 - streaming feature extraction to the Parquet cache")
    print("=" * 72)
    print(f"feature version: {FEATURE_VERSION}")
    print(f"V_MIN = {V_MIN} m/s (derived)   V_CUT = {V_CUT} m/s")
    print()

    # --- Train -------------------------------------------------------------
    t0 = time.perf_counter()
    train = extract_directory(RAIL_TRAIN_DIR)
    train_elapsed = time.perf_counter() - t0
    print(f"TRAIN: {train.shape[0]} files extracted in {train_elapsed:.1f} s "
          f"({train_elapsed / train.shape[0]:.3f} s/file)")

    labels = pd.read_csv(RAIL_TRAIN_LABELS)
    train = train.merge(
        labels.rename(columns={"filename": "file_id", "label": "meta_label"}),
        on="file_id",
        how="left",
    )
    if train["meta_label"].isna().any():
        missing = train.loc[train["meta_label"].isna(), "file_id"].tolist()
        raise ValueError(f"Unlabelled Train files: {missing}")

    # --- Test --------------------------------------------------------------
    t1 = time.perf_counter()
    test = extract_directory(RAIL_TEST_DIR)
    test_elapsed = time.perf_counter() - t1
    print(f"TEST:  {test.shape[0]} files extracted in {test_elapsed:.1f} s "
          f"({test_elapsed / test.shape[0]:.3f} s/file)")

    total_elapsed = time.perf_counter() - t0
    print()
    print(f"WALL-CLOCK, full pass over {train.shape[0]} Train files: "
          f"{train_elapsed:.1f} s = {train_elapsed / 60:.2f} min")
    print(f"WALL-CLOCK, Train + Test ({train.shape[0] + test.shape[0]} files): "
          f"{total_elapsed:.1f} s = {total_elapsed / 60:.2f} min")

    train.to_parquet(FEATURES_TRAIN_PATH, index=False)
    test.to_parquet(FEATURES_TEST_PATH, index=False)
    _write_meta(train.shape, test.shape)

    # --- Shapes and the suffix inventory (AC-4, AC-36) --------------------
    feature_cols = [c for c in train.columns if c not in ("file_id", "meta_label")]
    model_cols = model_matrix_columns(feature_cols)
    meta_cols = [c for c in feature_cols if c.startswith(META_PREFIX)]

    print()
    print(f"train shape: {train.shape}   test shape: {test.shape}")
    shared = set(train.columns) - {"meta_label"} == set(test.columns)
    print(f"train and test carry the same feature columns: {shared}")
    print(f"F (total feature columns, excluding file_id/meta_label): {len(feature_cols)}")
    print(f"  model matrix columns (admitted): {len(model_cols)}")
    print(f"  meta_ columns (dropped):         {len(meta_cols)} -> {sorted(meta_cols)}")
    print(f"AC-4 range 500-1,200: "
          f"{'WITHIN' if 500 <= len(model_cols) <= 1200 else 'OUTSIDE - reporting the discrepancy'}")

    print()
    print("--- AC-36: unit-suffix inventory ---")
    print(f"  closed vocabulary: {list(UNIT_SUFFIXES)}")
    for suffix in UNIT_SUFFIXES:
        others = [s for s in UNIT_SUFFIXES if s != suffix and s.endswith(suffix)]
        n = sum(
            1
            for c in feature_cols
            if not c.startswith(META_PREFIX)
            and c.endswith(suffix)
            and not any(c.endswith(o) for o in others)
        )
        verdict = "admitted" if suffix in ADMISSIBLE_SUFFIXES else "REJECTED"
        print(f"    {suffix:<9} {n:>5} columns   [{verdict}]")
    n_rejected = len([c for c in feature_cols
                      if not c.startswith(META_PREFIX) and c not in model_cols])
    print(f"  admitted to model matrix: {len(model_cols)}")
    print(f"  rejected by suffix rule:  {n_rejected}")

    for path in (FEATURES_TRAIN_PATH, FEATURES_TEST_PATH):
        print(f"  on disk: {path.name} = {path.stat().st_size / 1e6:.3f} MB")

    print("=" * 72)


if __name__ == "__main__":
    main()
