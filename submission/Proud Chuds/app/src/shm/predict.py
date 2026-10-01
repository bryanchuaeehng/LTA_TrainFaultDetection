"""Cumulative fatigue damage from the reconstructed, fitted Bryan ridge model."""
from __future__ import annotations
import argparse
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.shm.core import cycle_histogram, predict as predict_histogram

MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts/shm/model.joblib"


@lru_cache(maxsize=1)
def load_model():
    model = joblib.load(MODEL_PATH)
    if model.get("kind") != "ridge" or "_pipe" not in model:
        raise ValueError("The fitted SHM ridge pipeline is incomplete.")
    return model


def read_signal(path: Path) -> np.ndarray:
    try:
        values = np.loadtxt(path, delimiter=",", ndmin=2)
    except ValueError as exc:
        raise ValueError("SHM requires a headerless CSV with one numeric stress column.") from exc
    if values.shape[1] != 1 or values.shape[0] < 3:
        raise ValueError("SHM requires at least three readings in one stress column.")
    values = values[:, 0]
    if not np.isfinite(values).all():
        raise ValueError("SHM contains missing or non-finite stress readings.")
    if np.ptp(values) == 0:
        raise ValueError("SHM requires a varying dynamic-stress signal.")
    return values


def predict(input_path: Path) -> pd.DataFrame:
    path = Path(input_path)
    paths = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not paths:
        raise ValueError("No SHM CSV files were supplied.")
    model = load_model()
    rows, diagnostics = [], []
    for item in paths:
        signal = read_signal(item)
        H = cycle_histogram(signal, model["k"])
        if H.sum() <= 0:
            raise ValueError(f"{item.name}: no valid fatigue cycles were identified.")
        value = float(predict_histogram(H, model)[0])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{item.name}: the fitted model did not return a valid damage estimate.")
        rows.append({"file_id": item.name, "prediction": value})
        stride = max(1, len(signal) // 600)
        diagnostics.append({
            "file_id": item.name, "input_rows": len(signal),
            "counted_cycles": int(H.sum()), "stress_min": float(signal.min()),
            "stress_max": float(signal.max()),
            "trace_index": list(range(0, len(signal), stride)),
            "trace": signal[::stride].tolist(),
        })
    result = pd.DataFrame(rows)
    result.attrs["diagnostics"] = {"files": diagnostics}
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict cumulative fatigue damage.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = predict(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
