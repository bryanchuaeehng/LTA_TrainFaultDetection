# SHM — Cumulative Fatigue Damage Regression

**Shipped model: ridge regression on physics-derived features.**
**Expected leaderboard score `0.994` (80% interval 0.992 – 0.997).**

A forecast, not a result. Estimated by
grouped cross-validation and by 100 simulated 48/16 submissions in which the whole
model is re-derived from 48 files and scored on 16 it never saw.

| | Grouped CV | Simulated submission (100×) |
|---|---|---|
| **Ridge on pseudo-damage (shipped)** | **0.99507** | **0.99432** ± 0.00190 |
| 50/50 geometric blend | 0.99443 | 0.99412 ± 0.00208 |
| Physics, 2-parameter S-N law | 0.99117 | 0.99167 ± 0.00225 |

Ridge beat the physics model in 93/100 simulated submissions.

## Task framing

SHM is a **regression** task: one continuous cumulative-damage value per file. The
shipped model is a linear regression, fitted by ridge with the penalty chosen by
internal CV on the training fold.

```
log D = b₀ + Σₘ wₘ · log( Σᵢ nᵢ σᵢᵐ )        m = 2,3,…,8
```

Each feature is the rainflow **pseudo-damage** at a candidate S-N exponent. The
fitted weights are the interesting part:

| m | 2 | 3 | 4 | **5** | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| weight | +0.010 | −0.068 | +0.362 | **+0.486** | +0.316 | +0.102 | −0.082 |

They peak at m=5 and decay symmetrically — the regression independently re-derives
the physics (Miner's rule with m≈5) and then relaxes the single-exponent constraint
into a smooth blend of neighbouring power laws. Weighted-mean exponent 5.167.

The gain is **not** from capacity: 3 features (m=4,5,6) already give 0.514%, and
23 features give 0.587% — worse. There is a genuine optimum at 3–7 exponents.

## Feature extraction — where the accuracy actually came from

Two conventions dominate, both recovered empirically from the training labels.

**Repeat-history residue closure.** Rainflow counting leaves an unclosed residue —
only ~10 cycles per file, but carrying **20–74% of total σ⁵ damage**.

| Residue treatment | MAPE |
|---|---|
| discarded | 15.89% |
| counted as half-cycles (`rainflow` library default) | 2.46% |
| **closed by concatenating residue with itself** | **0.80%** |

**Sixty-four stress classes.** Binning reversals into k=64 classes — the classical
rainflow default — is an *isolated* optimum, not a trend:

| k | 48 | 56 | 60 | **64** | 68 | 72 | 80 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| MAPE | 1.33% | 1.26% | 1.19% | **0.80%** | 1.24% | 1.21% | 1.09% | 1.19% | 1.12% |

Nested CV re-selected k=64 independently in 11/11 folds.

## Model comparison

All scored on identical grouped-CV folds (`code/benchmark.py`).

| Family | Method | MAPE | Score |
|---|---|---|---|
| **Regression on physics features** | **Ridge, log-pseudo-damage m=2..8** | **0.49%** | **0.995** |
| Parametric physics | 2-parameter recovered S-N law | 0.88% | 0.991 |
| Regression on physics features | Ridge, all features pooled | 1.98% | 0.980 |
| Tree ensemble | GBoost on pseudo-damage | 9.11% | 0.909 |
| ML, no rainflow | GBoost, 24 statistical + 9 spectral | 20.00% | 0.800 |
| ML, no rainflow | Ridge, 24 statistical + 9 spectral | 23.12% | 0.769 |
| ML, no rainflow | RandomForest, same features | 30.87% | 0.691 |
| Frequency domain | Dirlik spectral method | 31.83% | 0.682 |
| Frequency domain | Narrow-band approximation | 45.10% | 0.549 |

Two readings worth stating in a write-up. Generic ML on raw signal statistics
plateaus around 20% MAPE — the target is a σ⁵-weighted functional that statistical
summaries cannot represent. But regression *on top of* physics-derived features beats
the pure physics model. The feature engineering carries the information; the
regression then exploits the residual flexibility the single-exponent law forbids.

## Tested and rejected

| Variant | Result |
|---|---|
| Endurance cutoff σ_cut ∈ {0.5 … 15} | 0.797% vs 0.802% — noise |
| Dual-slope / Haibach knee (8 knees × 6 deltas) | 0.801% vs 0.802% — noise |
| Goodman mean-stress, Su ∈ {200 … 10000} | monotonically → no-correction limit |
| Walker mean-stress, γ ∈ {0.5 … 1.0} | monotonically → γ=1 (no correction) |
| Linear mean-stress, a ∈ {−0.01 … +0.01} | monotonically → a=0 (no correction) |
| Nonparametric S-N, 2–8 knot log-log spline | LOO selected the 2-knot (power-law) solution |
| Residual correction from 24 signal features | CV R² = 0.075 — not predictable, excluded |

Each mean-stress family degrades *monotonically* away from its own no-correction
limit — strong evidence the reference used pure stress range.

## Train/validation split

No official split is given and file numbering is explicitly random. The data spans
two lines × two load conditions (AW0/AW4), so a random split risks leaking
condition-level structure.

Groups were built **without labels**: 24 statistical + spectral features →
standardise → PCA(4) → KMeans(4), any group under 4 files merged into its nearest
neighbour. Result: 3 groups of 22 / 28 / 14. GroupKFold over these is the headline,
with the model refit from scratch inside every fold.

Grouped CV (0.493%) and random CV (0.580%) agree closely — no condition-level
leakage available to exploit.

## Leakage and overfitting checks

| Check | Result |
|---|---|
| Fold partitions differ between seeds | verified, all distinct |
| Permutation test (labels shuffled) | MAPE 118% vs 0.58% real — **204× collapse** |
| Stability over 20 reshuffles | sd 0.016% |
| Nested CV, config re-searched per fold | 0.905% vs 0.883% — 0.02pp selection bias |
| Simulated 48/16 submission × 100 | 0.99432 ± 0.00190 |
| Capacity sweep | more features make it worse — not overfitting |
| Test features inside training envelope | **16/16 on all 7 features** — interpolating |

Not measured by any of the above: the 64 labels were visible while deciding *what to
try*. The automated search is priced in; human choices cannot be. Mitigation is that
the key discovery (residue carries most of the damage) is a label-free property of
the signal, and the fix is textbook practice rather than a tuned knob.

## Residual risk

Two test files (test03 at 73.8%, test02 at 62.3%) exceed the training maximum
residue share of 62.0%, so their answers lean hardest on the closure convention
matching the organisers'. Mitigating: residue share does not predict error on
training data (r = 0.07), and the ridge and physics models agree to within 0.45% and
0.24% on exactly those two files.

## Files

```
shm_predictions.csv         16 rows, file_id + prediction
predictions.zip             flat, ready to submit
code/shm_core.py            feature extraction + both models
code/predict.py             CLI: --input <dir|file> --output shm_predictions.csv
code/benchmark.py           reproduces the model-comparison table
code/evaluate.py            reproduces the validation numbers
code/judge_leaderboard.py   replica of the organisers' scorer
code/model.joblib           fitted pipeline
code/model.json             readable coefficients
```

```bash
python code/predict.py --input <test_dir> --output shm_predictions.csv
python code/judge_leaderboard.py shm_predictions.csv <truth.csv>
```

## Caveats

- The scorer is a replica built from the disclosed formula; the organisers'
  `judge_leaderboard.py` is not in the repo.
- Simulated-submission figures assume the 16 test files resemble the 64 training
  files. Feature-envelope checks support this but do not prove it.
- Per-file error on held-out folds: median 0.27%, 90th percentile 0.86%, worst 5.66%.
# Train Condition Intelligence

One Streamlit app covering all four LTA × NebulaX PS3 subsystems: Door cycle
detection, ACV fault localisation, Rail Corrugation classification and SHM
cumulative-damage estimation.

**Zero of our five leaderboard uploads were used.** Every number below comes
from local validation, which meant the validation had to be worth trusting.
That constraint shaped most of what follows.
# Final model submission evaluation scores:
<img width="1036" height="553" alt="image" src="https://github.com/user-attachments/assets/44363da7-83cd-4f04-9dfb-a0985d442ca3" />

Scores mentioned anywhere below this are from local eval. 

## Run locally

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m streamlit run app.py
```

Python 3.12 or 3.13. Inference needs only the application code, the committed
artifacts in `artifacts/` and the uploaded file — no training data, no
environment variables. Models load once and are cached; predictions stay in the
session and temporary uploads are deleted after processing.

---

## How we chose each model

The short version: **three of the four subsystems are too small to justify a
learned model, and we said so rather than fitting one anyway.**

### Rail — the only subsystem with a real model, and the only one with a trap
TLDR: **three of the four subsystems are too small to justify ML models so we just used chud small models**

### Rail — the only subsystem with a real model,

272 training files, 234 Normal against 14 Side I and 24 Side II.

Before modelling we derived train speed from the pulse-counter column and
checked it against the labels. **No fault file was recorded below 9.70 m/s,
while 133 of 234 Normal files were.** A rule using nothing but speed scores
0.509 macro F1 against an always-Normal floor of 0.308.

That is a confound, not a feature. A model could score respectably by learning
"fast means fault" and nothing about corrugation — and with no leaderboard
feedback, nothing would have caught it. So:

- Speed still sets the **wavelength band edges**, because corrugation excites
  vibration at `f = v/λ` and a fixed frequency band is physically wrong.
- Raw speed is **barred as a classifier input**.
- The headline is measured on the **restricted domain** (≥ 9.70 m/s), where
  every class spans the same speed range and the shortcut is unavailable.

Features are per-side aggregates built into a **Side I vs Side II contrast** —
differences and log-ratios between the two rails in the same file. The two fault
classes are the same phenomenon mirrored, and a common multiplicative speed
factor cancels exactly in a log-ratio. Model is logistic regression with balanced
class weights: at 982 features and 228 samples this is `p >> n`, where
regularised linear models are the defensible choice.

**We tried to beat it and failed.** Eight configurations, pre-registered before
running, under nested cross-validation. The nested estimate came out at **0.7558
— below** the incumbent's 0.7765, and no challenger passed a paired test. Random
forest was worst and was never once selected.

**Restricted macro F1 0.7765** (std 0.0833, 50 folds). Full-set 0.7858. We quote
the lower one.

### Door — the model is two numbers, on purpose

The Info Kit frames finding cycle boundaries as the hard part. Measurement
disagreed: the idle time between cycles **is not sampled**, so the stream
arrives pre-broken. Splitting on inter-row time gaps recovers all 110 training
cycles with **zero boundary error**, stable from 0.2 s to 2.0 s. Every IoU is
1.000, so IoU-weighted F1 collapses to plain binary F1 — the whole score is
classification.

The signal is **integrated motor current**: total work done against resistance.
AUC 0.9837, and 1.0000 within each operation type. Notably *peak* current is AUC
0.3735 — **below chance** — so a spike detector points the wrong way.

With 30 abnormal cycles and a feature that separates perfectly within operation,
a learned model adds nothing and contributes variance. So the model is one
threshold for Open and one for Close, fitted in-fold. Leave-one-out **109/110**.

### ACV — six cases cannot support learning

Six labelled cases exist, one of which (`acv_case_04`) ships with a different,
much richer schema and is outside the fitted model's validated scope. That
leaves five.

The approach is a frozen ensemble of six equally-weighted components — a domain
percentile score, three classifiers and two persistence scores — producing a
**ranking**, not a classification. Leave-one-case-out puts the faulty car first
in all five evaluated folds. A perfect score on five cases is encouraging, not
conclusive, and the app says so.

### SHM — the baseline is the ceiling

The Info Kit discloses that labels were generated by rainflow counting plus
Miner's rule. That makes an analytical reimplementation the **baseline rather
than an achievement**: a correct one should reproduce the labels almost exactly,
and a learned model that merely matches it has probably found a leak rather than
out-modelled the definition. Our ridge model over log-domain cycle-histogram
features reaches a train MAPE of 0.0047. No pass/fail limit was supplied, so the
app reports the estimate and asks for engineering review rather than inventing a
threshold.

## The leak we found by measuring, not reviewing

Worth its own section, because it nearly shipped.

We wrote a rule barring any feature whose units carry a per-second term, which
required expressing spectral centroid as a wavelength in metres. Dimensionally
correct — and wrong. The conversion is `λ = v/f_centroid`, so wherever the
centroid is roughly constant, **dividing by it multiplies speed back in.** One
such feature correlated with speed at **0.9982**, and a depth-3 tree on it alone
beat raw speed — the very thing we had banned. A second family, normalised by
`v²`, over-corrected and reintroduced speed with inverted sign.

> A unit contract checks a property of the **formula**. Leakage is a property of
> the **data**. No static rule catches this; only measuring against the data
> does.

122 features dropped, and an empirical gate added: any feature correlating with
speed above |Spearman| 0.90 on the restricted training set is removed. The
threshold is a pragmatic choice, not a derived one — the highest surviving
correlation is 0.8995.

---

## App interface decisions

### Four tabs, not one upload

The subsystems read genuinely unrelated inputs: 68 rail files of 129 columns, one
continuous door stream, 16 headerless SHM columns, one ACV workbook. No single
upload could serve all four, so the app does not pretend otherwise. What *is*
unified is the output — one validated `predictions.zip`.

### Show the evidence, not just the verdict

Our door model is a threshold, and on the test stream flagged cycles range from
**+0.04% to +36.5%** over it. One scored 88,438 against a threshold of 88,402 —
36 units out of 88,000. As a plain list of labels that looks identical to the
+36.5% case, and a technician dispatched on it is acting on a coin flip.

So cycles to inspect are **sorted strongest-evidence-first** and tagged `Clear`
or `Borderline`, with a caption naming how many sit inside the band.

### The "worth a look" list

Eight cycles on the test stream are `Normal` but within 5% of the threshold, the
closest at −0.08%. Under a pure label view they are invisible. They now get their
own list, explicitly *not* as faults — a threshold is a line through a
continuum, and which side each cycle landed on is not the whole story.

### What the interface refuses to say

- The margin is a **distance from a fitted threshold**, never a probability.
  There is a test asserting the wording never implies calibration.
- Rail distinguishes *"no corrugation detected"* from *"too slow to tell"*. The
  second shows as **Check again**, not a pass; the CSV still says `Normal` as the
  format requires, and the app says the classifier never ran.
- ACV grey means **lower priority, not confirmed healthy**.
- SHM asks for engineering review because no limit was supplied.
- Door recordings do not identify a car or door number, and the app says so.

Each is a place where a more confident-looking app would be a worse one.

### Accessibility is not a colour choice

Every status carries a **text label and a glyph** (`✓ ! ? —`) beside its colour,
so nothing depends on distinguishing red from green. The train and coach
diagrams are SVG with `role="img"` and descriptive labels. Focus rings are
visible, `prefers-reduced-motion` is respected, and the layout works to phone
width.

### A status strip, because partial submissions are easy

Four cards show what has been checked and what is outstanding, so nobody submits
three subsystems believing they submitted four.

---

## Verification

```bash
python -m pytest -q
```

**77 tests**, covering the scoring metrics' exact matching rules — including
every clause where Door's IoU-weighted F1 differs from plain F1 — and the
wording guarantees above. Measured runtimes on the official test inputs: Door
0.14 s, ACV 24.3 s, Rail 80.6 s, SHM 7.9 s. Those are predictions and timings,
not held-out accuracy claims; the organisers hold the answers.

## Known limitations

- **Rail Side I rests on 14 files.** Per-class F1 0.565, and 0.0 on 2 of 50
  folds. Fold scores range 0.575–0.946.
- **Two rail training pairs are byte-identical duplicates**
  (`Train107`/`Train115`, `Train165`/`Train187`), so 226 distinct recordings.
- **Door is one door on one day.** The Info Kit warns distributions differ
  between doors; we cannot test that.
- **SHM** currently runs the earlier ridge model; a newer one exists in a format
  the app's wrapper does not yet read.
- The speed composition of the rail train and test sets is near-identical, so
  the confound likely persists into the held-out set — which is exactly why the
  restricted figure is the headline.
