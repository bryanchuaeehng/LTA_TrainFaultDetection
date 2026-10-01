"""Fair benchmark: is a properly-built ML regressor competitive with recovered physics?

Three families, all scored on the SAME grouped CV folds:
  A. frequency-domain fatigue (narrow-band, Dirlik) -- NO rainflow, cheap, "online"
  B. ML regression on features -- with and without rainflow-derived features
  C. the recovered physics model (reference)
"""
import numpy as np, pandas as pd, time
from scipy import signal, special
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATA = "/home/claude/repo/PS3/02_Datasets/SHM"
lab = pd.read_csv(f"{DATA}/Train_Labels.csv"); names = list(lab.filename)
D = lab.damage.values.astype(float); logD = np.log(D)
g = np.load("cache/groups.npy")
M_TRUE = 4.9845


def spectral_features(x):
    f, P = signal.welch(x, nperseg=8192, detrend="constant")
    m0 = np.trapezoid(P, f); m1 = np.trapezoid(f * P, f)
    m2 = np.trapezoid(f**2 * P, f); m4 = np.trapezoid(f**4 * P, f)
    nu0 = np.sqrt(m2 / m0); nup = np.sqrt(m4 / m2)
    gam = m2 / np.sqrt(m0 * m4)
    # narrow-band damage rate (up to the 1/C constant)
    nb = nu0 * (2 * np.sqrt(2 * m0)) ** M_TRUE * special.gamma(1 + M_TRUE / 2)
    # Dirlik
    xm = (m1 / m0) * np.sqrt(m2 / m4)
    D1 = 2 * (xm - gam**2) / (1 + gam**2)
    R = (gam - xm - D1**2) / (1 - gam - D1 + D1**2)
    D2 = (1 - gam - D1 + D1**2) / (1 - R)
    D3 = 1 - D1 - D2
    Q = 1.25 * (gam - D3 - D2 * R) / max(D1, 1e-9)
    Z = np.linspace(1e-4, 12, 4000)
    pz = (D1 / Q) * np.exp(-Z / Q) + (D2 * Z / R**2) * np.exp(-(Z**2) / (2 * R**2)) + D3 * Z * np.exp(-(Z**2) / 2)
    integ = np.trapezoid(Z**M_TRUE * pz, Z)
    dk = nup * (2 * np.sqrt(m0)) ** M_TRUE * integ
    return dict(m0=m0, m1=m1, m2=m2, m4=m4, nu0=nu0, nup=nup, gamma=gam,
                log_nb=np.log(max(nb, 1e-300)), log_dirlik=np.log(max(dk, 1e-300)))


t0 = time.time()
rows = []
for n in names:
    rows.append({"filename": n, **spectral_features(np.loadtxt(f"{DATA}/Train/{n}"))})
sp = pd.DataFrame(rows)
print("spectral features done", round(time.time() - t0, 1), "s", flush=True)

# rainflow-derived pseudo-damage at several exponents (cheap given the histogram)
H = np.load("cache/H_train_v2.npy")
EDG = np.exp(np.linspace(np.log(1e-4), np.log(500), 4001)); CEN = np.sqrt(EDG[:-1] * EDG[1:])
LC = np.log(CEN / 10.0)
pd_feats = {}
for m in [2, 3, 4, 5, 6, 7, 8]:
    la = m * LC; Mx = la.max()
    pd_feats[f"logpd_m{m}"] = np.log(H @ np.exp(la - Mx) + 1e-300) + Mx
rf = pd.DataFrame(pd_feats)

stat = pd.read_csv("cache/feat_Train.csv").set_index("filename").loc[names].reset_index()
SC = [c for c in stat.columns if c not in ("filename", "n")]
SPC = [c for c in sp.columns if c != "filename"]

def score_cv(X, model_fn, tag):
    X = np.asarray(X, float); p = np.zeros(64)
    for tr, va in GroupKFold(3).split(X, logD, g):
        mdl = model_fn(); mdl.fit(X[tr], logD[tr]); p[va] = mdl.predict(X[va])
    v = np.mean(np.abs(np.exp(p) - D) / D)
    print("  %-46s MAPE %7.3f%%   score %.5f" % (tag, 100 * v, max(0, 1 - v)), flush=True)
    return v

def scale_only(col, tag):
    """classical estimator: single free constant (log offset), fit per fold"""
    p = np.zeros(64)
    for tr, va in GroupKFold(3).split(col.reshape(-1, 1), logD, g):
        b = np.median(logD[tr] - col[tr]); p[va] = col[va] + b
    v = np.mean(np.abs(np.exp(p) - D) / D)
    print("  %-46s MAPE %7.3f%%   score %.5f" % (tag, 100 * v, max(0, 1 - v)), flush=True)
    return v

RID = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 50)))
GBR = lambda: GradientBoostingRegressor(random_state=0)
RFR = lambda: RandomForestRegressor(n_estimators=400, random_state=0)

print("\n=== A. frequency-domain fatigue, NO rainflow (1 free constant) ===")
scale_only(sp.log_nb.values, "narrow-band approximation")
scale_only(sp.log_dirlik.values, "Dirlik spectral method")

print("\n=== B. ML regression, NO rainflow features (online-capable) ===")
Xa = np.column_stack([stat[SC].values, sp[SPC].values])
score_cv(Xa, RID, "Ridge  on 24 stat + 9 spectral")
score_cv(Xa, GBR, "GBoost on 24 stat + 9 spectral")
score_cv(Xa, RFR, "RandomForest on 24 stat + 9 spectral")

print("\n=== C. ML regression WITH rainflow pseudo-damage features ===")
Xb = np.column_stack([rf.values])
score_cv(Xb, RID, "Ridge  on log-pseudo-damage m=2..8")
score_cv(Xb, GBR, "GBoost on log-pseudo-damage m=2..8")
Xc = np.column_stack([rf.values, stat[SC].values, sp[SPC].values])
score_cv(Xc, RID, "Ridge  on everything (rainflow+stat+spectral)")
score_cv(Xc, GBR, "GBoost on everything")

print("\n=== D. recovered physics (reference) ===")
import sys; sys.path.insert(0, "pkg")
from src.shm.core import fit as pfit, predict as ppred
p = np.zeros(64)
for tr, va in GroupKFold(3).split(H, D, g):
    mdl = pfit(H[tr], D[tr]); p[va] = ppred(H[va], mdl)
v = np.mean(np.abs(p - D) / D)
print("  %-46s MAPE %7.3f%%   score %.5f" % ("2-parameter recovered S-N law", 100 * v, 1 - v))
sp.to_csv("cache/spectral_Train.csv", index=False)
rf.to_csv("cache/pseudodamage_Train.csv", index=False)
