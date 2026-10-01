"""Replica of the organisers' SHM scorer (Info Kit Section 5).

score = max(0, 1 - MAPE),  MAPE = mean(|true - pred| / |true|)
"""
import sys, pandas as pd, numpy as np

def shm_score(pred_csv, truth_csv):
    p = pd.read_csv(pred_csv); t = pd.read_csv(truth_csv)
    p.columns = [c.strip().lower() for c in p.columns]
    t.columns = [c.strip().lower() for c in t.columns]
    tcol = "damage" if "damage" in t.columns else "prediction"
    fid = "file_id" if "file_id" in t.columns else "filename"
    m = t.rename(columns={fid: "file_id", tcol: "true"})[["file_id", "true"]].merge(
        p[["file_id", "prediction"]], on="file_id", how="left")
    if m.prediction.isna().any():
        missing = m[m.prediction.isna()].file_id.tolist()
        raise SystemExit(f"no prediction for: {missing}")
    mape = float(np.mean(np.abs(m.true - m.prediction) / np.abs(m.true)))
    return max(0.0, 1.0 - mape), mape, len(m)

if __name__ == "__main__":
    s, mape, n = shm_score(sys.argv[1], sys.argv[2])
    print(f"n={n}  MAPE={mape:.6f} ({mape*100:.3f}%)  primary_metric={s:.6f}")
