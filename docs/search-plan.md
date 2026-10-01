# Model search — PRE-REGISTERED

**Written before any configuration was scored. Not extended mid-run.**

This file is the commitment device. Everything below was fixed in advance so
that the reported number is the score of a *procedure*, not the maximum over a
list of things we tried.

## The incumbent

**M1, restricted macro F1 0.7765** (std 0.0833, 50 folds). Logistic regression,
L2, `class_weight="balanced"`, plan defaults, all 982 gated features.

The incumbent's artefacts (`artifacts/rail/model.joblib`,
`predictions/rail_predictions.csv`) are **frozen**. They change only if a
challenger passes the replacement test below. If nothing passes, they are not
touched at all, and that is a perfectly good outcome.

## Protocol: nested cross-validation

The thing being estimated is **the whole selection procedure**, not any one
configuration.

- **Outer loop:** `RepeatedStratifiedKFold(n_splits=5, n_repeats=10,
  random_state=20260918)` — the same 50 folds as the incumbent, so the
  comparison is paired fold-for-fold.
- **Inner loop:** `StratifiedKFold(n_splits=3, shuffle=True,
  random_state=20260918)` on the outer training portion only.
- **Selection happens in the inner folds.** For each outer fold, every
  pre-registered configuration is scored on the inner folds, the best is chosen,
  refitted on the full outer-training portion, and evaluated once on the
  held-out outer fold.
- **The reported number is the mean of the outer-fold scores.** That is an
  estimate of "run this selection procedure on data like this and see what you
  get".

**What is explicitly forbidden:** scoring every configuration on the outer folds
and reporting the best. That number is optimistically biased by exactly the
amount the search had room to overfit, and it is the standard way a model search
produces a result that does not survive contact with held-out data. Any
per-configuration outer scores computed for the log are **diagnostics, never a
headline**.

Selection metric in the inner loop: **restricted** macro F1 (`v >= V_CUT`),
matching the headline, so the procedure optimises the thing we report.

## Pre-registered configurations

Exactly these. **No additions once the run starts**, regardless of what the
results suggest.

| ID | Family | Settings |
|---|---|---|
| `C1` | Logistic regression | L2, `C=1.0`, balanced — **the incumbent's config** |
| `C2` | Logistic regression | L2, `C=0.1`, balanced — stronger regularisation |
| `C3` | Logistic regression | L2, `C=10.0`, balanced — weaker regularisation |
| `C4` | Logistic regression | L1 (`saga`), `C=1.0`, balanced — sparse, p>>n |
| `C5` | Linear SVM | `LinearSVC`, `C=1.0`, balanced |
| `C6` | Random forest | 500 trees, `min_samples_leaf=2`, balanced_subsample |
| `C7` | Extra trees | 500 trees, `min_samples_leaf=2`, balanced |
| `C8` | Ridge classifier | `alpha=1.0`, balanced |

Eight configurations. The rationale for the spread: at 982 features and 228
samples this is a `p >> n` problem, so the live question is how much
regularisation helps and whether a nonlinear model can use 38 fault examples at
all. Both directions are represented.

`C1` is included deliberately so the incumbent competes on identical terms
inside the same nested protocol rather than being compared across protocols.

## Replacement rule

A challenger replaces the incumbent **only** if it passes the **three-condition
paired test** on the outer folds, against the incumbent's per-fold restricted
scores:

1. `mean(d) > 0.05`
2. `d_i > 0` on at least 35 of 50 folds
3. `p10(d) > -0.05`

where `d_i` is the challenger's score minus the incumbent's on outer fold `i`.

These are the same three conditions used throughout the project, fixed before
this run. **If no challenger passes, the incumbent stands and the frozen
artefacts are not rewritten.** A null result is a real result.

## What this run may not touch

- `docs/plans/rail.md` — frozen
- The feature gates (domwavelength drop, |Spearman| > 0.90 gate) — fixed
- The exclusion rules (`V_MIN = 1.28`), `V_CUT = 9.70`, the low-speed
  inference rule
- The seed, fold count, repeat count
- The 982-feature matrix

## Logging

**Every configuration's score is recorded, winners and losers alike.** Inner
selection counts (how often each config was chosen), per-configuration
diagnostic outer scores, and the nested estimate all go to
`docs/tests/rail-search-raw.txt`. Nothing is deleted. A configuration that
performed badly is evidence about the problem, not clutter.
