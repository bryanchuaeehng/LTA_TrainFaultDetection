"""Module constants for the rail corrugation subsystem.

Every numeric constant the plan (`docs/plans/rail.md`) fixes lives here and
nowhere else, so a reviewer can read the whole constant set in one screen and
so no value can drift between the training code and ``predict()``.

Plan section references are given per constant.
"""

from __future__ import annotations

__all__ = [
    "SEED",
    "V_MIN",
    "V_MIN_N_PERIODS",
    "V_MIN_LAMBDA_MAX_M",
    "V_MIN_WINDOW_S",
    "V_CUT",
    "WAVELENGTH_BANDS_M",
    "EPS",
    "WHEEL_DIAMETER_M",
    "WHEEL_CIRCUMFERENCE_M",
    "TEETH_PER_REV",
    "TRANSITIONS_PER_REV",
    "FILE_DURATION_S",
    "FS_HZ",
    "N_SAMPLES",
    "N_COLUMNS",
    "N_CHANNELS",
    "WELCH_NPERSEG",
    "WELCH_NOVERLAP",
    "WELCH_WINDOW",
    "WELCH_DETREND",
    "FEATURE_VERSION",
    "UNIT_SUFFIXES",
    "ADMISSIBLE_SUFFIXES",
    "REJECTED_SUFFIXES",
    "META_PREFIX",
    "N_SPLITS",
    "N_REPEATS",
    "INNER_N_SPLITS",
    "MARGIN_MEAN_D",
    "MARGIN_MIN_WIN_FOLDS",
    "MARGIN_P10_D",
    "ARTIFACT_DIR",
    "FEATURES_TRAIN_PATH",
    "FEATURES_TEST_PATH",
    "FEATURE_META_PATH",
    "MODEL_PATH",
    "SPEED_CORR_PATH",
    "BANNED_FEATURE_STEMS",
    "SPEED_CORR_THRESHOLD",
    "PROXY_BAR_RESTRICTED",
]

from pathlib import Path

# Inference must not resolve the training corpus at import time.
REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Seed (plan section 4.1). ONE definition site in the whole codebase. -----
SEED: int = 20260918

# --- Speed thresholds -------------------------------------------------------
# V_MIN is DERIVED, not chosen (plan section 5.3a.2a):
#     V_MIN = N * lambda_max / t
# with N = 2 (assumption, section 7.14), lambda_max = 0.64 m (the upper edge of
# the longest mandated wavelength band, section 7.5), t = 1.0 s (Info Kit
# section 2.1 line 81).
V_MIN_N_PERIODS: int = 2
V_MIN_LAMBDA_MAX_M: float = 0.64
V_MIN_WINDOW_S: float = 1.0
V_MIN: float = V_MIN_N_PERIODS * V_MIN_LAMBDA_MAX_M / V_MIN_WINDOW_S  # 1.28 m/s

# The restricted-evaluation cut (plan section 4.7.2). Reporting stratification
# only: it never enters a feature and never enters a decision rule.
V_CUT: float = 9.70

# --- Wavelength bands, in metres (plan section 5.3 mechanism 1, section 7.5) -
WAVELENGTH_BANDS_M: tuple[tuple[float, float], ...] = (
    (0.02, 0.04),
    (0.04, 0.08),
    (0.08, 0.16),
    (0.16, 0.32),
    (0.32, 0.64),
)

# Guards the log-ratio of plan section 5.2 step 4 only.
EPS: float = 1e-12

# --- Speed derivation (plan section 5.3, from Info Kit section 2.1:74-77) ----
WHEEL_DIAMETER_M: float = 0.85
WHEEL_CIRCUMFERENCE_M: float = 3.141592653589793 * WHEEL_DIAMETER_M
TEETH_PER_REV: int = 90
TRANSITIONS_PER_REV: int = 2 * TEETH_PER_REV  # 180
FILE_DURATION_S: float = 1.0

# --- File geometry ----------------------------------------------------------
FS_HZ: int = 10_000
N_SAMPLES: int = 10_000
N_COLUMNS: int = 129
N_CHANNELS: int = 128

# --- Welch PSD parameters (plan section 7.7) --------------------------------
WELCH_NPERSEG: int = 2048
WELCH_NOVERLAP: int = 1024
WELCH_WINDOW: str = "hann"
WELCH_DETREND: str = "constant"

# --- Feature cache version (plan section 5.4) -------------------------------
# Must be distinct from any iteration-1 cache. Bumping this invalidates the
# cache and forces a rebuild.
FEATURE_VERSION: str = "rail-feat-v2.0.0"

# --- The closed unit-suffix vocabulary (plan section 5.3b.2a) ---------------
# A feature column name that ends in none of these is a HARD ERROR aborting
# extraction. meta_-prefixed columns are exempt (they are not features).
UNIT_SUFFIXES: tuple[str, ...] = (
    "_m",        # metres (a wavelength or length)            -> admissible
    "_ratio",    # dimensionless ratio or normalised index     -> admissible
    "_amppv",    # amplitude normalised by v                   -> admissible
    "_amppv2",   # amplitude normalised by v^2                 -> admissible
    "_amp",      # amplitude-axis quantity, raw                -> REJECTED
    "_hz",       # frequency-axis rate                         -> REJECTED
    "_persec",   # any other per-second rate                   -> REJECTED
)
ADMISSIBLE_SUFFIXES: tuple[str, ...] = ("_m", "_ratio", "_amppv", "_amppv2")
REJECTED_SUFFIXES: tuple[str, ...] = ("_amp", "_hz", "_persec")
META_PREFIX: str = "meta_"

# --- Empirical speed-leakage controls (USER RULING, post-Gate-2) -----------
# The suffix contract checks DIMENSION, which is a property of the formula.
# Leakage is a property of the DATA. A static rule cannot catch the second.
#
# 1. The domwavelength family is dropped outright. It computed
#    lambda = v / f_centroid, so wherever f_centroid is near-constant across
#    files (broadband shock with no dominant peak) lambda is almost exactly
#    proportional to v -- the metres mandate multiplied speed back in after
#    mechanism 3 had divided it out. Measured |Spearman| vs v up to 0.9982.
BANNED_FEATURE_STEMS: tuple[str, ...] = ("domwavelength",)

# 2. A correlation gate on top: any surviving feature whose |Spearman| against
#    derived speed exceeds this threshold, computed on the RESTRICTED eligible
#    training set, is dropped.
#
#    THE THRESHOLD IS CHOSEN PRAGMATICALLY, NOT DERIVED. It is a fixed
#    feature-set decision computed ONCE on training data, not a per-fold fit.
SPEED_CORR_THRESHOLD: float = 0.90

# The bar the restricted headline must beat: the measured restricted macro F1
# of a depth-3 tree on the single worst leaking domwavelength feature. This is
# the bar, NOT raw v's 0.3566.
PROXY_BAR_RESTRICTED: float = 0.3984

# --- Validation protocol (plan section 4.1, 5.5, 5.7a) ----------------------
N_SPLITS: int = 5
N_REPEATS: int = 10
INNER_N_SPLITS: int = 3

# The three margin conditions of plan section 5.7a. Fixed before any model is
# fitted; relaxing any of them is a planner decision.
MARGIN_MEAN_D: float = 0.05
MARGIN_MIN_WIN_FOLDS: int = 35
MARGIN_P10_D: float = -0.05

# --- Artefact paths (plan section 5.4). All gitignored. ---------------------
ARTIFACT_DIR: Path = REPO_ROOT / "artifacts" / "rail"
FEATURES_TRAIN_PATH: Path = ARTIFACT_DIR / "features_train.parquet"
FEATURES_TEST_PATH: Path = ARTIFACT_DIR / "features_test.parquet"
FEATURE_META_PATH: Path = ARTIFACT_DIR / "feature_meta.json"
MODEL_PATH: Path = ARTIFACT_DIR / "model.joblib"
SPEED_CORR_PATH: Path = ARTIFACT_DIR / "speed_correlated_features.json"
