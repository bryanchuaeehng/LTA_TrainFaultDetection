"""Held-out evaluation: nested CV (k and m chosen inside every fold)."""
import numpy as np, pandas as pd, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.shm.core import cycle_histogram, pseudo_damage, fit, predict
from sklearn.model_selection import GroupKFold, KFold, LeaveOneOut

DATA = "/home/claude/repo/PS3/02_Datasets/SHM"
lab = pd.read_csv(f"{DATA}/Train_Labels.csv")
names = list(lab.filename); D = lab.damage.values.astype(float)
CACHE = "/home/claude/shm/cache/H_train_v2.npy"
if os.path.exists(CACHE):
    H = np.load(CACHE)
else:
    H = np.array([cycle_histogram(np.loadtxt(f"{DATA}/Train/{n}")) for n in names]); np.save(CACHE, H)
g = np.load("/home/claude/shm/cache/groups.npy")
print("condition groups (unsupervised, no labels):", np.bincount(g))

def run(spl, gr, tag):
    pred = np.zeros(len(D)); ms = []
    it = spl.split(H, D, gr) if gr is not None else spl.split(H)
    for tr, va in it:
        mdl = fit(H[tr], D[tr]); pred[va] = predict(H[va], mdl); ms.append(mdl["m"])
    v = np.mean(np.abs(pred - D) / D)
    print("%-26s MAPE=%.4f%%  score=%.5f   m %.3f-%.3f" % (tag, 100*v, max(0,1-v), min(ms), max(ms)))
    return pred, v

print("\n=== held-out evaluation (model refit inside every fold) ===")
p1, _ = run(GroupKFold(3), g, "GroupKFold by condition")
run(KFold(8, shuffle=True, random_state=0), None, "KFold(8) random")
run(LeaveOneOut(), None, "LeaveOneOut(64)")

full = fit(H, D)
print("\nfull fit: k=%d  m=%.4f  logC=%.5f  in-sample MAPE=%.4f%%" %
      (full["k"], full["m"], full["logC"], 100*full["train_mape"]))
json.dump(full, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json"), "w"), indent=2)

ape = np.abs(p1 - D) / D
rng = np.random.default_rng(0)
s = np.array([1 - ape[rng.integers(0, len(D), 16)].mean() for _ in range(200000)])
print("\nper-file abs error: median %.2f%%  90th %.2f%%  max %.2f%%" %
      (100*np.median(ape), 100*np.percentile(ape, 90), 100*ape.max()))
print("simulated score on a 16-file test set:")
for q in [5, 25, 50, 75, 95]:
    print("   %2dth pct: %.4f" % (q, np.percentile(s, q)))
print("   80%% central interval: %.4f - %.4f" % (np.percentile(s,10), np.percentile(s,90)))
print("   P(score < 0.98) = %.1f%%" % (100*(s < 0.98).mean()))
os.makedirs("/home/claude/shm/out", exist_ok=True)
pd.DataFrame({"file_id": names, "prediction": p1}).to_csv("/home/claude/shm/out/_cv_pred.csv", index=False)
lab.rename(columns={"filename": "file_id"}).to_csv("/home/claude/shm/out/_cv_truth.csv", index=False)
