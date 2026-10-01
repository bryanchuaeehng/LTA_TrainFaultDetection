"""Per-file feature extraction (plan sections 5.2, 5.3, 5.3b, 5.4).

Design, in one paragraph. The same per-channel feature function ``f`` runs over
all 128 channels. Channels are grouped by (side, kind) - Side I is odd axle
positions, Side II even - and aggregated across the 32 channels of each group by
six order statistics. That yields the Side I aggregate ``A_I`` and the Side II
aggregate ``A_II`` with identical layout, from which the PRIMARY family is the
explicit contrast ``A_I - A_II`` and ``log((A_I+eps)/(A_II+eps))``. Speed sets
WHERE to look (the wavelength-band edges are ``v/lambda`` Hz, different for every
file) but never itself becomes a number the classifier can read.

Two rules govern what may be emitted, and both are enforced mechanically here:

* section 5.3b.2 - no feature computable from column 1 alone. Speed quantities go
  into ``meta_``-prefixed columns which are dropped before the pipeline.
* section 5.3b.2a - the dimensional rule. Every feature column name MUST end in a
  suffix from the closed vocabulary in ``constants.UNIT_SUFFIXES``. A column with
  no recognised suffix is a HARD ERROR that aborts extraction, naming the
  offender. ``_hz``, ``_persec`` and ``_amp`` may exist in the cache for
  diagnostics but are rejected from the model matrix.

Raw absolute-level amplitudes are struck (section 5.3b.3): amplitude-like
quantities are emitted ONLY in ``/v`` and ``/v^2`` normalised form, because
supplying both ``X`` and ``X/v`` would let a tree reconstruct ``v`` by division.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy import stats as sp_stats

from src.rail.constants import (
    ADMISSIBLE_SUFFIXES,
    BANNED_FEATURE_STEMS,
    SPEED_CORR_PATH,
    EPS,
    FS_HZ,
    META_PREFIX,
    N_CHANNELS,
    N_COLUMNS,
    N_SAMPLES,
    UNIT_SUFFIXES,
    WAVELENGTH_BANDS_M,
    WELCH_DETREND,
    WELCH_NOVERLAP,
    WELCH_NPERSEG,
    WELCH_WINDOW,
)
from src.rail.speed import file_index, speed_from_column1

__all__ = [
    "UnitSuffixError",
    "channel_side",
    "channel_kind",
    "channel_car",
    "channel_position",
    "side_swap_permutation",
    "band_edges_hz",
    "extract_file",
    "validate_feature_names",
    "model_matrix_columns",
    "load_speed_correlated_features",
]


class UnitSuffixError(ValueError):
    """Raised when a feature column name carries no recognised unit suffix.

    This is the HARD ERROR of plan section 5.3b.2a. It aborts extraction rather
    than warning, so a feature added without thinking about units cannot reach
    the cache, let alone the model.
    """


# --- Channel index arithmetic (plan section 5.2, CONFIRMED at Gate 1) -------
# For 0-based j over columns 2..129 (j = 0..127):
#   car  = j // 16 + 1 ; pos = (j % 16) // 2 + 1 ; kind = vibration if j even
#   side = Side I if pos odd else Side II


def channel_car(j: int) -> int:
    return j // 16 + 1


def channel_position(j: int) -> int:
    return (j % 16) // 2 + 1


def channel_kind(j: int) -> str:
    return "vibration" if j % 2 == 0 else "shock"


def channel_side(j: int) -> str:
    return "Side I" if channel_position(j) % 2 == 1 else "Side II"


def _group_indices() -> dict[tuple[str, str], np.ndarray]:
    """The four (side, kind) channel groups, 32 channels each."""
    groups: dict[tuple[str, str], list[int]] = {}
    for j in range(N_CHANNELS):
        groups.setdefault((channel_side(j), channel_kind(j)), []).append(j)
    return {key: np.asarray(value, dtype=int) for key, value in groups.items()}


GROUPS = _group_indices()


def side_swap_permutation() -> np.ndarray:
    """Column permutation exchanging Side I and Side II channels (AC-13).

    Channel ``j`` at position ``p`` maps to the channel of the same car and same
    kind at position ``p+1`` if ``p`` is odd, ``p-1`` if even. In index terms
    that is simply ``j XOR 2`` within each car block, since position occupies
    bits 1-3 of ``j % 16`` and swapping odd/even position flips bit 1.
    """
    perm = np.arange(N_CHANNELS)
    for j in range(N_CHANNELS):
        swapped = j ^ 2
        perm[j] = swapped
        # Sanity: the swap must preserve car and kind, and flip side.
        assert channel_car(swapped) == channel_car(j)
        assert channel_kind(swapped) == channel_kind(j)
        assert channel_side(swapped) != channel_side(j)
    return perm


def band_edges_hz(v_mps: float) -> list[tuple[float, float]]:
    """Speed-dependent Hz edges for each fixed WAVELENGTH band.

    A band fixed in metres maps to ``[v/lambda_hi, v/lambda_lo]`` Hz, so the Hz
    edges differ for every file (AC-15). That is the point: the same feature
    index always means the same physical wavelength.
    """
    return [(v_mps / hi, v_mps / lo) for (lo, hi) in WAVELENGTH_BANDS_M]


# --- The per-channel feature function --------------------------------------
# Names here are STEMS. The unit suffix is appended when the block is assembled,
# because the same stem appears in raw (_amp), /v (_amppv) and /v^2 (_amppv2)
# form. Scale-free stems carry their own suffix already.

_AMPLITUDE_STEMS: tuple[str, ...] = tuple(
    [f"bandpow_{lo:g}_{hi:g}" for (lo, hi) in WAVELENGTH_BANDS_M]
    + ["bandpow_total", "rms", "peak"]
)
_SCALEFREE_NAMES: tuple[str, ...] = (
    "kurtosis_ratio",
    "skewness_ratio",
    "crestfactor_ratio",
    "domwavelength_m",
    "spectralentropy_ratio",
)


def _per_channel_features(
    channels: np.ndarray, v_mps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``f`` for all 128 channels at once.

    Returns ``(amplitude_block, scalefree_block)`` with shapes
    ``(128, len(_AMPLITUDE_STEMS))`` and ``(128, len(_SCALEFREE_NAMES))``.

    ``channels`` is ``(N_SAMPLES, 128)``.
    """
    # Welch PSD for every channel in one call (axis=0 is time).
    freqs, psd = sp_signal.welch(
        channels,
        fs=FS_HZ,
        window=WELCH_WINDOW,
        nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NOVERLAP,
        detrend=WELCH_DETREND,
        axis=0,
    )
    # psd: (n_freqs, 128). df is uniform for Welch.
    df = float(freqs[1] - freqs[0])
    nyquist = FS_HZ / 2.0

    n_bands = len(WAVELENGTH_BANDS_M)
    band_power = np.full((N_CHANNELS, n_bands), np.nan, dtype=np.float64)

    for b, (lo_hz, hi_hz) in enumerate(band_edges_hz(v_mps)):
        # Plan section 5.3 mechanism 1: a band whose upper edge exceeds Nyquist,
        # or whose width falls below the frequency resolution, is NaN for this
        # file and imputed in-fold. Never silently zeroed.
        if hi_hz > nyquist or (hi_hz - lo_hz) < df or lo_hz <= 0.0:
            continue
        mask = (freqs >= lo_hz) & (freqs < hi_hz)
        if not mask.any():
            continue
        band_power[:, b] = psd[mask, :].sum(axis=0) * df

    total_power = psd.sum(axis=0) * df
    rms = np.sqrt(np.mean(channels.astype(np.float64) ** 2, axis=0))
    peak = np.max(np.abs(channels.astype(np.float64)), axis=0)

    amplitude = np.column_stack([band_power, total_power, rms, peak])

    # --- scale-free / dimensionless block ---------------------------------
    kurt = sp_stats.kurtosis(channels.astype(np.float64), axis=0, bias=False)
    skew = sp_stats.skew(channels.astype(np.float64), axis=0, bias=False)
    crest = np.divide(peak, rms, out=np.zeros_like(peak), where=rms > 0)

    # Spectral centroid, emitted in METRES as a dominant wavelength
    # (v / f_centroid), never in Hz - plan section 5.3b.2a struck the Hz form.
    psd_sum = psd.sum(axis=0)
    centroid_hz = np.divide(
        (psd * freqs[:, None]).sum(axis=0),
        psd_sum,
        out=np.zeros(N_CHANNELS),
        where=psd_sum > 0,
    )
    dom_wavelength_m = np.divide(
        np.full(N_CHANNELS, v_mps),
        centroid_hz,
        out=np.full(N_CHANNELS, np.nan),
        where=centroid_hz > 0,
    )

    # Normalised Shannon entropy of the PSD: dimensionless by construction, so
    # permitted as a frequency-axis quantity (plan section 5.3b.2a audit table).
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.divide(psd, psd_sum, out=np.zeros_like(psd), where=psd_sum > 0)
        log_p = np.where(p > 0, np.log(p), 0.0)
        entropy = -(p * log_p).sum(axis=0) / np.log(psd.shape[0])

    scalefree = np.column_stack([kurt, skew, crest, dom_wavelength_m, entropy])
    return amplitude, scalefree


# --- Aggregation across the 32 channels of a group -------------------------
# Order statistics, not just the mean (plan section 5.2 step 2), plus the two
# across-car dispersion aggregators mandated as the per-car hedge.

_AGGREGATORS: tuple[str, ...] = (
    "median",
    "p90",
    "max",
    "iqr",
    "stdcar",
    "maxmmed",
)


def _aggregate(block: np.ndarray) -> dict[str, np.ndarray]:
    """Aggregate a ``(32, d)`` block down the channel axis to six ``(d,)`` vectors."""
    with np.errstate(all="ignore"):
        q25, q50, q75, q90 = np.nanpercentile(block, [25, 50, 75, 90], axis=0)
        mx = np.nanmax(block, axis=0)
        return {
            "median": q50,
            "p90": q90,
            "max": mx,
            "iqr": q75 - q25,
            "stdcar": np.nanstd(block, axis=0),
            "maxmmed": mx - q50,
        }


_SIDE_TAG = {"Side I": "sideI", "Side II": "sideII"}
_KIND_TAG = {"vibration": "vib", "shock": "shk"}


def extract_file(path: Path, *, swap_sides: bool = False) -> dict[str, float]:
    """Extract the full feature vector for ONE file.

    Peak memory is one file's array. The raw array is discarded on return; no
    list of DataFrames is ever built and ``pd.concat`` never sees raw data.

    ``swap_sides`` exchanges the Side I and Side II channels before extraction
    and exists only for the AC-13 symmetry test.
    """
    frame = pd.read_csv(path, dtype=np.float32)
    if frame.shape != (N_SAMPLES, N_COLUMNS):
        raise ValueError(
            f"{path.name}: expected shape ({N_SAMPLES}, {N_COLUMNS}), got {frame.shape}. "
            f"A file failing this assertion is a hard error, not skipped."
        )

    values = frame.to_numpy(dtype=np.float32, copy=False)
    col1 = values[:, 0]
    channels = values[:, 1:N_COLUMNS]

    if swap_sides:
        channels = channels[:, side_swap_permutation()]

    v_mps, T = speed_from_column1(col1)

    out: dict[str, float] = {}

    # --- meta_ block: needed for the exclusion rule, the restricted domain,
    # the inference rule and B2. DROPPED before the pipeline (section 5.3b.2).
    out["meta_v_mps"] = float(v_mps)
    out["meta_v_kmh"] = float(v_mps * 3.6)
    out["meta_T"] = float(T)
    out["meta_index"] = float(file_index(path))

    # Below V_MIN the amplitude normalisations are not computed at all
    # (section 5.3b.3 numerical guard) and the band edges collapse. Such files
    # never reach a fit; predict() short-circuits them before this point.
    amplitude, scalefree = _per_channel_features(channels, v_mps)

    amp_pv = amplitude / v_mps if v_mps > 0 else np.full_like(amplitude, np.nan)
    amp_pv2 = amplitude / (v_mps**2) if v_mps > 0 else np.full_like(amplitude, np.nan)

    # --- per-(side, kind) aggregation ------------------------------------
    per_group: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}
    for (side, kind), idx in GROUPS.items():
        per_group[(side, kind)] = {
            "amppv": _aggregate(amp_pv[idx, :]),
            "amppv2": _aggregate(amp_pv2[idx, :]),
            "scalefree": _aggregate(scalefree[idx, :]),
        }

    def _emit(name: str, value: float) -> None:
        out[name] = float(value)

    # Absolute per-side blocks (normalised only - raw is STRUCK by 5.3b.3),
    # plus the symmetric mean, plus the contrast and log-ratio primary family.
    for kind, ktag in _KIND_TAG.items():
        gI = per_group[("Side I", kind)]
        gII = per_group[("Side II", kind)]

        for agg in _AGGREGATORS:
            # --- amplitude-like, in /v and /v^2 form only ---
            for norm, suffix in (("amppv", "_amppv"), ("amppv2", "_amppv2")):
                aI = gI[norm][agg]
                aII = gII[norm][agg]
                for s, stem in enumerate(_AMPLITUDE_STEMS):
                    base = f"{ktag}_{agg}_{stem}"
                    _emit(f"{base}_sideI{suffix}", aI[s])
                    _emit(f"{base}_sideII{suffix}", aII[s])
                    _emit(f"{base}_sym{suffix}", (aI[s] + aII[s]) / 2.0)
                    _emit(f"{base}_diff{suffix}", aI[s] - aII[s])
                    # Log ratio: dimensionless, and a multiplicative speed
                    # factor common to both sides cancels EXACTLY here.
                    with np.errstate(all="ignore"):
                        ratio = np.log((abs(aI[s]) + EPS) / (abs(aII[s]) + EPS))
                    _emit(f"{base}_logratio_ratio", ratio)

            # --- scale-free block, permitted raw ---
            sI = gI["scalefree"][agg]
            sII = gII["scalefree"][agg]
            for s, sname in enumerate(_SCALEFREE_NAMES):
                stem, suffix = sname.rsplit("_", 1)
                suffix = f"_{suffix}"
                base = f"{ktag}_{agg}_{stem}"
                _emit(f"{base}_sideI{suffix}", sI[s])
                _emit(f"{base}_sideII{suffix}", sII[s])
                _emit(f"{base}_sym{suffix}", (sI[s] + sII[s]) / 2.0)
                _emit(f"{base}_diff{suffix}", sI[s] - sII[s])

    return out


# --- The unit-suffix contract (plan section 5.3b.2a, AC-36) ----------------


def validate_feature_names(names: list[str]) -> None:
    """HARD ERROR on any column with no recognised unit suffix.

    ``meta_``-prefixed columns are exempt: they are not features and are dropped
    before the pipeline. Everything else must end in one of
    ``constants.UNIT_SUFFIXES``.
    """
    offenders = [
        name
        for name in names
        if not name.startswith(META_PREFIX)
        and not name.endswith(UNIT_SUFFIXES)
    ]
    if offenders:
        raise UnitSuffixError(
            "Unit-suffix contract violated (plan section 5.3b.2a). "
            f"{len(offenders)} column(s) carry no recognised suffix from "
            f"{list(UNIT_SUFFIXES)}. Extraction ABORTED. Offending column(s): "
            + ", ".join(offenders[:20])
            + ("..." if len(offenders) > 20 else "")
        )


def load_speed_correlated_features() -> list[str]:
    """Feature names dropped by the empirical speed-correlation gate.

    Computed ONCE on the restricted eligible TRAINING set by
    ``src.rail.speed_gate`` and persisted, so ``predict()`` applies exactly the
    same feature set the model was fitted on. Returns ``[]`` if the gate has not
    been run yet.
    """
    if not SPEED_CORR_PATH.exists():
        return []
    try:
        payload = json.loads(SPEED_CORR_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return list(payload.get("dropped", []))


def model_matrix_columns(names: list[str]) -> list[str]:
    """The columns admitted to the model matrix.

    Four filters, in order:

    1. drop every ``meta_`` column (plan section 5.3b.2);
    2. keep only admissible unit suffixes (plan section 5.3b.2a) -- this drops
       ``_amp``, ``_hz``, ``_persec``;
    3. drop every banned feature stem (USER RULING: the ``domwavelength``
       family, whose ``lambda = v / f_centroid`` form multiplied speed back in);
    4. drop every feature the empirical speed-correlation gate rejected.

    Filters 3 and 4 exist because the suffix contract checks DIMENSION, a
    property of the formula, while leakage is a property of the DATA. A static
    rule cannot catch the second; only an empirical check against the data can.
    """
    banned = load_speed_correlated_features()
    return [
        name
        for name in names
        if not name.startswith(META_PREFIX)
        and name.endswith(ADMISSIBLE_SUFFIXES)
        and not any(stem in name for stem in BANNED_FEATURE_STEMS)
        and name not in banned
    ]
