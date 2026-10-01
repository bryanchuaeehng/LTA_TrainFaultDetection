"""Reconstruct the missing fitted SHM artifact from official TRAINING data only.

Test predictions are compared with the already committed outputs as a parity
check, never used for fitting, configuration selection, or performance scoring.
Run: python -m scripts.reconstruct_shm --data /path/to/02_Datasets/SHM
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.shm.core import cycle_histogram, fit, predict
from src.shm.predict import read_signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "artifacts/shm"
    labels = pd.read_csv(args.data / "Train_Labels.csv")
    if len(labels) != 64 or labels["filename"].duplicated().any():
        raise ValueError("Expected the 64 official SHM training files.")
    cache = args.data / ".reconstruction_histograms.npz"
    digest = hashlib.sha256((args.data / "Train_Labels.csv").read_bytes())
    for name in labels.filename:
        digest.update((args.data / "Train" / name).read_bytes())
    source_hash = digest.hexdigest()
    if cache.exists():
        loaded = np.load(cache, allow_pickle=False)
        if str(loaded["source_hash"]) != source_hash:
            raise ValueError("SHM reconstruction cache does not match training inputs.")
        H = loaded["H"]
    else:
        histograms = []
        for i, name in enumerate(labels.filename, 1):
            histograms.append(cycle_histogram(read_signal(args.data / "Train" / name)))
            print(f"Training features {i}/64", flush=True)
        H = np.asarray(histograms)
        np.savez(cache, H=H, source_hash=source_hash)
    model = fit(H, labels.damage.to_numpy(dtype=float), kind="ridge")
    reference = json.loads((out / "reference_model.json").read_text())
    if not np.isclose(model["alpha"], reference["alpha"], rtol=1e-8):
        raise ValueError("Reconstruction selected a different ridge penalty.")
    np.testing.assert_allclose(model["coef"], reference["coef"], rtol=2e-5, atol=2e-7)
    expected = pd.read_csv(out / "reference_predictions.csv")
    test_H = np.asarray([cycle_histogram(read_signal(args.data / "Test" / name))
                         for name in expected.file_id])
    actual = predict(test_H, model)
    np.testing.assert_allclose(actual, expected.prediction, rtol=2e-6, atol=1e-9)
    model["reconstruction"] = {
        "training_files": 64, "training_source_sha256": source_hash,
        "parity_files": len(expected), "parity_max_relative_difference":
        float(np.max(np.abs(actual - expected.prediction.to_numpy()) / np.abs(expected.prediction))),
        "held_out_labels_used": False,
    }
    joblib.dump(model, out / "model.joblib")
    summary = {k: v for k, v in model.items() if k != "_pipe"}
    (out / "reconstruction.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(model["reconstruction"], indent=2))


if __name__ == "__main__":
    main()
