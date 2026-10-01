# Rail Corrugation — Plan (iteration 2)

Scope: rail corrugation only. Door, ACV and SHM are out of scope this session.

Authoritative source, in precedence order:

1. `reference/03_References/Rail_Corrugation/Rail_Corrugation_Info_Kit.md` (the
   Info Kit — authoritative problem definition)
2. `reference/01_Problem_Statement_3_Specifications.md` (top-level spec)
3. `reference/04_Example_Submission/rail_predictions.csv` (output schema)

Where the Info Kit and the top-level spec disagree, the Info Kit wins. The
top-level spec itself says so, at
`reference/01_Problem_Statement_3_Specifications.md:25-27`:

> The table above is only a quick side-by-side summary — see each subsystem's
> own Info Kit (Section 2.2) for that subsystem's full business background,
> dataset, and defined task; the Info Kit is the authoritative problem
> definition, not this table.

Two contradictions already logged in `CLAUDE.md` OPEN QUESTIONS (the dangling
**Deliverables** pointer, and the dataset tracked in git) are NOT re-litigated
here. One further contradiction is internal to the Info Kit and IS resolved
here, in §7.1.

Hard constraints binding this plan: zero leaderboard uploads; no data file ever
read fully into context; models only, no app/UI/server; all data access through
`src/common/paths.py`; the metric through `src/common/metrics.py` and nowhere
else.

**Iteration 2 notice.** This plan was revised after Gate 1 measurement, in
response to `docs/plans/rail-objection.md`. The revised sections are §4.1, §4.2,
§4.7 (new), §4.8 (new), §5.3 (rewritten), §5.3a (new), §5.3b (new), §5.5, §5.7,
§5.7a (new), §5.8, §6 (extended), §7.3, §7.4, §7.6, §7.10-§7.13 (new), and the
Revision log. Every other section stands from iteration 1. The single largest
change: **derived speed separates the classes on its own and is therefore struck
as a classifier input.** See §5.3b.

**Amended after acceptance:** `V_MIN` is now **derived** (§5.3a.2a, §7.14) rather
than chosen, and mechanism 3 is generalised into a **dimensional rule**
(§5.3b.2a) enforced by a unit-suffix contract. New §5.3a.2a, §5.3b.2a, §7.14 and
AC-36. **THE PLAN IS FROZEN after this pass** — no amendment 3; anything noticed
from here goes to `docs/backlog.md` and is actioned only if the implementer
actually trips on it.

**Two numeric constants govern this iteration and appear throughout.** Both are
module constants:

- **`V_MIN = 1.28 m/s`** — the training-eligibility floor, **derived** in
  §5.3a.2a as `N · λ_max / t = 2 × 0.64 / 1.0`. 44 Train files fall below it, all
  Normal, leaving an eligible set of **228 = 190 Normal / 14 Side I / 24
  Side II**. Measured: no file lies in `[1.00, 1.28)`, so the derivation moved
  the constant without moving the eligible set.
- **`V_CUT = 9.70 m/s`** — the restricted-evaluation cut (§4.7). Roughly 139 of
  the 228 eligible files sit at or above it, including **all 38 fault files**.

Wherever this plan quotes an expected count, the implementer reports the
**realised** count and the realised number governs (AC-3b, AC-28).

---

## 1. Task type

**Multi-class classification. Three mutually exclusive classes, one prediction
per input file.**

Citation, Info Kit §3 lines 104-107:

> Build a model that classifies each 1-second axle-box vibration/shock recording
> as **Normal**, **Side I**, or **Side II** corrugation — a 3-class problem,
> judging the Side I and Side II rails' condition together from the same file

Corroborated by the top-level spec's summary table
(`01_Problem_Statement_3_Specifications.md:22`): "Multi-class classification —
Normal / Side I / Side II corrugation".

The classes are defined by the Info Kit §2.2 lines 87-90 and are exhaustive and
mutually exclusive as labelled:

- `Normal` — both Side I and Side II rails normal
- `Side I` — corrugation on Side I, Side II normal
- `Side II` — corrugation on Side II, Side I normal

Note what this label set does *not* contain: there is no "both sides corrugated"
class in the training data. See §7.2 — this is an ambiguity, not a settled fact
about the test set.

The unit of prediction is the **file**, not the sample, not the axle box, not
the second. One row of output per input file. This is not a segmentation task
(no time boundaries are scored) and not a ranking task (no ordering is scored).

---

## 2. Scoring metric

### 2.1 Verbatim definition

Info Kit §4, lines 127-129, quoted verbatim:

> Rail corrugation is graded on **macro F1** across the three classes (Normal,
> Side I, Side II) — not plain accuracy. Macro F1 computes the F1 score for each
> class independently, then averages the three unweighted (each class counts
> equally regardless of how many examples it has).

Info Kit §4, lines 133-135, verbatim, which fixes the degenerate case:

> A model that predicts "Normal" for every file could score well over 90% plain
> accuracy while never once correctly detecting a fault — macro F1 does not
> reward that: a class the model never gets right (0 recall, 0 precision)
> contributes an F1 of 0 to the average, regardless of how rare that class is.

Info Kit §4, line 152: "Your final Rail score (`primary_metric`) is this macro
F1 value."

### 2.2 Precise specification

Stated precisely enough to implement from this plan alone:

- **Label set and order**: exactly `("Normal", "Side I", "Side II")`, in the
  order the Info Kit lists them (§4 line 127). The order fixes positional
  stability of per-class output; it does not affect the macro average.
- **Per-class F1**: for class `c`, with `TP` = files whose true label is `c` and
  whose predicted label is `c`, `FP` = files predicted `c` whose true label is
  not `c`, `FN` = files whose true label is `c` predicted as something else:
  `F1_c = 2*TP / (2*TP + FP + FN)`.
- **Averaging mode**: unweighted arithmetic mean over the **three classes**, not
  over samples. `macro_f1 = (F1_Normal + F1_"Side I" + F1_"Side II") / 3`.
- **Denominator is always 3.** A class absent from a particular validation fold
  still occupies a slot in the denominator. This is load-bearing: letting the
  denominator shrink to the classes present would silently inflate any fold
  missing a minority class, and with 14 Side I files that is a realistic fold.
  This follows directly from "each class counts equally regardless of how many
  examples it has" (§4 line 129).
- **Zero division**: `F1_c = 0.0` when `2*TP + FP + FN == 0` (class neither
  present nor predicted), and when `TP == 0` with nonzero FP or FN. Not dropped,
  not treated as 1.0. Fixed by §4 lines 134-135 quoted above.
- **Clipping**: none. Macro F1 is already in `[0, 1]` by construction. No
  clipping, flooring or rescaling anywhere.
- **Tie handling**: the metric consumes hard labels, so there are no ties at the
  metric level. Ties in the *model's* decision (argmax over class scores) are a
  modelling concern and are specified in §5.6.
- **No thresholding, no probability calibration** enters the metric. It is
  computed from predicted labels against true labels, nothing else.

### 2.3 Worked numeric example, reproduced in full

Info Kit §4 lines 137-145. A hypothetical model scored on a small held-out set,
with per-class F1 scores of:

| Class | F1 score |
|---|---|
| Normal | 0.97 |
| Side I | 0.40 |
| Side II | 0.60 |

**Macro F1 = (0.97 + 0.40 + 0.60) / 3 = 0.657**

Arithmetic check: `0.97 + 0.40 + 0.60 = 1.97`; `1.97 / 3 = 0.65666...`, which
the Info Kit rounds to `0.657`. The tester must reproduce `0.657` to three
decimal places (i.e. `round(macro_f1_from_per_class([0.97, 0.40, 0.60]), 3) ==
0.657`).

Second worked figure, Info Kit §4 lines 147-150: an always-Normal model scores
F1 = 0 on both Side I and Side II, giving **macro F1 = (1.0 + 0 + 0) / 3 ≈
0.33**, against roughly 85-90% plain accuracy. Note the Info Kit's `1.0` for
Normal is itself a simplification — an always-Normal model has Normal recall 1.0
but precision equal to the Normal base rate, so the true figure on a 234/14/24
distribution is `F1_Normal = 2*0.8603/(1+0.8603) = 0.9249`, macro F1 = `0.3083`.
This was confirmed empirically at Gate 1: the implementer measured B0 at exactly
**0.3083** on the full 272 (objection §3).

**Iteration 2 correction.** Because iteration 2 excludes the 44 files below
`V_MIN` from training (§5.3a), the operative always-Normal floor is computed on
the **eligible** 190/14/24 set and is **≈0.3030** (arithmetic in §4.2.0). The
plan uses **≈0.3030** as the B0 floor, subject to the realised counts. The
full-corpus `0.3083` and the Info Kit's loose `0.33` are both recorded; neither
is the bar.

### 2.4 Implementation mandate

The metric comes from `src/common/metrics.py` and **nowhere else**. No module,
script, notebook or test may hand-roll, inline, or re-derive macro F1, and none
may call `sklearn.metrics.f1_score` directly. The functions to use:

- `macro_f1(y_true, y_pred)` — the production path, on hard labels.
- `per_class_f1(y_true, y_pred)` — for reporting the three components.
- `macro_f1_from_per_class(scores)` — only for reproducing the Info Kit's worked
  example, which is stated as three F1 values rather than as predictions.
- `RAIL_CLASSES` — the canonical label tuple. Do not redefine it locally.

`macro_f1` must always be called such that `labels=RAIL_CLASSES` is in force
(it is the default). Any call site that narrows `labels` is a defect.

---

## 3. Output schema

### 3.1 The example file, copied exactly

`reference/04_Example_Submission/rail_predictions.csv`, all 6 lines, verbatim:

```
file_id,prediction
example_test_01.csv,Normal
example_test_02.csv,Side I
example_test_03.csv,Normal
example_test_04.csv,Side II
example_test_05.csv,Normal
```

### 3.2 Requirements

- **Filename**: exactly `rail_predictions.csv`. Info Kit §3 line 110; top-level
  spec table line 133.
- **Header**: exactly `file_id,prediction` — two columns, this order, lowercase,
  no spaces, no BOM, no index column.
- **`file_id`**: the source file name **including its extension**, e.g.
  `Test1.csv`. Info Kit §3 line 114: "The source file name, including its
  extension — e.g. `Test1.csv`." Not a stem, not a path, not an integer.
- **`prediction`**: one of exactly `Normal`, `Side I`, `Side II`. Case-sensitive,
  single ASCII space inside `Side I` / `Side II`, no trailing whitespace. These
  are the exact strings in Info Kit §3 line 115 and in `RAIL_CLASSES`.
- **Dtypes**: both columns are strings. `prediction` must never be emitted as an
  integer class index.
- **Row count**: **68** — one row per Test file, `Test1.csv`..`Test68.csv`.
  The example file carries **5 illustrative placeholder rows**, not 68, and its
  `file_id` values (`example_test_01.csv` etc.) are placeholders. Info Kit §3
  lines 117-118: "the file names and values there are illustrative placeholders,
  not real data". Do not copy the example's rows or infer real file names from
  them.
- **Completeness**: every one of the 68 Test files appears exactly once. No
  duplicates, no omissions, no blank rows, no blank fields. A file the model
  cannot process still gets a row (see §5.8 fallback), and a file the model is
  not *trained* for still gets a row (see §5.3a.4, the low-speed inference
  rule). **Training eligibility never reduces the output row count.**
- **Ordering**: the Info Kit and the spec state no row ordering requirement, so
  ordering is not scored. Nevertheless the implementer MUST emit rows in
  **natural numeric order of the Test index** — `Test1.csv, Test2.csv, ...,
  Test68.csv` — not lexicographic order (`Test1, Test10, Test11, ...`). This is
  a determinism and diffability requirement of this plan, not of the spec.
- **Line endings / encoding**: UTF-8 without BOM. Trailing newline at end of
  file. Written via `DataFrame.to_csv(path, index=False)` or equivalent.

### 3.3 Validation of the emitted file

`scripts/validate_submission.py` already exists and checks filename, header,
valid labels, no empty rows/fields, unique `file_id`, and row count 68. Any
submission artefact this subsystem produces MUST pass it with default arguments
(`--expect-rows 68`). That script is the gate; do not reimplement its checks.

---

## 4. Validation split and its justification

### 4.1 The mandate — no discretion

**Splitter**: `sklearn.model_selection.RepeatedStratifiedKFold`

**Parameters, exactly:**

| Parameter | Value |
|---|---|
| `n_splits` | `5` |
| `n_repeats` | `10` |
| `random_state` | `20260918` |
| Stratification key `y` | the three-class label string from `Train_Labels.csv`, i.e. one of `Normal` / `Side I` / `Side II` |
| Grouping key | **none** — see §4.4 |
| Unit of splitting | the **file** (one training row = one `TrainN.csv`) |
| Population split | the **eligible** set of §4.2.0 — expected 228 files, the 44 files with `v < V_MIN = 1.28` m/s excluded; the realised count governs |
| Total fits | `5 × 10 = 50` |
| Shuffle | implied by `RepeatedStratifiedKFold`; do not set it separately |

This is not a default to be reconsidered. A single train/test split, a single
non-repeated `StratifiedKFold`, `train_test_split`, or any hold-out carved off
by hand is a defect and the reviewer rejects it.

The seed `20260918` is fixed and appears in exactly one place in the codebase
(a module-level constant in the rail package). It must not be passed on the
command line, must not be read from the environment, and must not be varied to
"see if the score improves". Any second seed used anywhere must be reported
alongside the first, never instead of it.

**One splitter, two evaluations.** Iteration 2 adds a *restricted* evaluation
(§4.7). It does **not** add a second splitter. The same 50 folds are scored
twice: once over the whole validation fold, and once over the subset of that
fold above the speed cut. This keeps a single partition and therefore a single
source of fold-to-fold variance.

### 4.2 Why stratified, and how the minority classes land

**Iteration 2 changes this section. Read §4.2.0 first.**

#### 4.2.0 The eligible training set — 228 files, not 272

The full labelled corpus is **234 Normal / 14 Side I / 24 Side II**, total 272,
verified against `Train_Labels.csv` and agreeing with Info Kit §2.2 line 86-87.

**Iteration 2 excludes the low-speed files from model TRAINING.** The exclusion
threshold is `v < V_MIN` with **`V_MIN = 1.28 m/s`**, derived in §5.3a.2a. The user's decision fixed
the exclusion of the **38 zero-speed** files; extending it to also cover the 6
near-zero files is **my decision**, reasoned in §5.3a.3 and flagged as an
assumption in §7.10. Gate 1 measured **44 Train files below 1.0 m/s** — 38 at
exactly zero plus 6 in `(0, 0.49]` — and **all 44 are Normal**. Direct
measurement at the derived threshold confirms the same 44 files fall below
1.28 m/s, because nothing lies in `[1.00, 1.28)`.

| Class | Full corpus | Excluded at `v < 1.28` | **Eligible (training)** |
|---|---|---|---|
| Normal | 234 | 44 | **190** |
| Side I | 14 | 0 | **14** |
| Side II | 24 | 0 | **24** |
| **Total** | **272** | **44** | **228** |

For reference, had the exclusion been zero-speed only (`v == 0`, 38 files), the
eligible set would be **196 / 14 / 24 = 234**. Both figures appear in the
Revision log so the reviewer can see exactly what the user decided and what I
decided. **228 at the derived `V_MIN = 1.28` is binding.** These counts are derived from Gate
1's reported statistics; the implementer MUST report the realised counts and use
those (AC-3b).

All 44 excluded files are Normal, so exclusion removes only majority-class files.

**What this does to the class balance.** The Normal share falls from
234/272 = 86.0% to 190/228 = 83.3%. The minority classes become *relatively*
less rare: Side I rises from 5.1% to 6.1%, Side II from 8.8% to 10.5%. The
imbalance ratio Normal:Side I improves from 16.7:1 to 13.6:1. This is a mild
improvement and is **not** a reason for the exclusion — the reason is stated in
§5.3a — but it must be recorded because it changes the `class_weight="balanced"`
weights computed in-fold (§5.7) and it changes the always-Normal baseline.

**Recomputed B0 on the eligible set.** An always-Normal model on 190/14/24 has
Normal recall 1.0 and precision 190/228 = 0.83333, so
`F1_Normal = 2 × 0.83333 / (1 + 0.83333) = 0.90909`, and
**macro F1 = 0.90909 / 3 = 0.30303 ≈ 0.3030**. (At the zero-only exclusion of
196/14/24 the same arithmetic gives ≈0.3039; on the full 272 Gate 1 measured
0.3083.) All three are recorded; **the figure B0 must reproduce inside the CV
loop is whichever corresponds to the realised eligible counts**, and the
implementer prints it rather than asserting this plan's estimate (AC-3b, AC-16).

**What this does to the already-thin Side I count — nothing, and that is the
point.** Side I stays at **14**. The exclusion buys no additional minority data.
The per-fold Side I quantisation described in §4.2.1 is **unchanged and remains
the binding constraint on this subsystem's statistical resolution.** Removing
majority files does not make 14 files into more than 14 files, and no claim to
the contrary may appear in the report. What the exclusion *does* change is that
Side I's 14 files are now a larger fraction of a smaller pool, which slightly
raises the prior a naive classifier places on the minority classes — a
second-order effect, stated so nobody mistakes it for a data gain.

**Exclusion from training is NOT exclusion from inference.** See §5.3a.4: every
one of the 68 Test files receives a prediction regardless of its speed.

#### 4.2.1 Fold composition on the eligible set

At `n_splits=5` over **228** eligible files, each validation fold holds
228/5 ≈ 45-46 files. Stratification distributes each class as evenly as integer
arithmetic permits:

| Class | Eligible total | Per validation fold at k=5 |
|---|---|---|
| Normal | 190 | 38 (190 = 5×38 exactly) |
| Side I | **14** | **2 or 3** (14 = 5×2 + 4, so four folds get 3 and one gets 2) |
| Side II | 24 | 4 or 5 (24 = 5×4 + 4) |
| Total | 228 | 45 or 46 |

**The Normal and total rows are derived from the expected eligible counts; the
implementer recomputes and prints them from the realised counts (AC-3b, AC-8).
The Side I and Side II rows do NOT depend on the threshold** — both fault
classes lie entirely above `V_MIN` under any reading — **so 2-or-3 Side I and
4-or-5 Side II per fold hold regardless, and that is the only fold statistic
anything in this plan binds to.**

**State this plainly: at k=5 a validation fold contains only 2 or 3 Side I
files.** The consequences are severe and must be written into the report, not
discovered later:

- Per-fold Side I recall is quantised to `{0, 1/3, 2/3, 1}` on a 3-file fold and
  `{0, 0.5, 1}` on a 2-file fold. There is no meaningful resolution between
  those steps. A single Side I file flipping from correct to incorrect moves
  per-fold Side I F1 by roughly 0.2-0.3 absolute.
- Consequently **per-fold macro F1 is not a reportable number.** Only the
  distribution across all 50 folds is.
- Side I F1 will legitimately be `0.0` on some folds. That is expected
  quantisation, not necessarily a broken model, and the report must not treat a
  single zero fold as a failure signal.

### 4.3 Why 10 repeats specifically

The repeat count is chosen for the **minority classes**, not for Normal.

- With `n_repeats=1` we get 5 fold-scores, each computed on 2-3 Side I files.
  The standard error of the mean over 5 such coarse, quantised scores is far too
  wide to distinguish a real improvement from reshuffling luck. A point estimate
  from a single 5-fold run is, at these counts, close to noise.
- With `n_repeats=10` every training file is placed in a validation fold **10
  times**, under 10 independent partitions. Each of the 14 Side I files is
  therefore evaluated 10 times against 10 different training sets, giving 140
  Side I validation *evaluations* in total rather than 14. The quantisation does
  not disappear — it is a property of each fold — but averaging over 50 folds
  converts it from a cliff into a usable distribution.
- 10 repeats also gives a **50-point sample** of macro F1, enough to report a
  mean, a standard deviation, and a 10th/90th percentile range that actually
  means something.
- The cost is bounded and known. Because features are cached to disk (§5.4),
  50 fits are 50 fits over a ~228×F float matrix of a few hundred kilobytes, not
  50 passes over 4.5 GB of CSV. That is seconds to low minutes, so 10 repeats is
  affordable where it would not be if raw data were re-read.

**Why not more than 10?** Beyond ~10 repeats the marginal reduction in the
standard error of the mean is small relative to the *irreducible* uncertainty
from having only 14 Side I files in existence. More repeats reduce
partition-choice variance; they cannot reduce sample-size variance. Ten is the
point where added compute buys reshuffling precision we do not need and cannot
convert into a better model. If the implementer finds 50 fits cheap enough to
want more, that is a change to this plan and comes back to the planner.

**Why not k=10 instead?** k=10 would put only 1-2 Side I files in each
validation fold — *worse* quantisation, and folds with a single minority example
make per-class F1 degenerate. k=5 is the largest fold count that still keeps
Side I above 1 per fold everywhere while leaving 80% of the data for training.
This trade-off is stated so the reviewer can check it, not so it can be
relitigated. **k=5, and the granularity cost is accepted and reported.**

### 4.4 Why no grouping key — and the honest caveat

`GroupKFold` and friends exist to prevent the same physical entity appearing in
both train and validation. The Info Kit gives us **no grouping metadata at all**:

- §2.2 lines 92-93 says `Train_Labels.csv` has "one row per file, with
  `filename` and `label` columns" — two columns, no run ID, no track section, no
  date, no line, no train set ID.
- §2.1 lines 74-82 describes the file's own columns: speed, then 128
  vibration/shock channels. No metadata columns.

So there is no key to group on. Grouping cannot be specified because the
information required to specify it has not been provided.

**This is a real and unquantifiable leakage risk, and it is stated, not hidden.**
Each file is 1 second of a moving train (§2.1 line 81). Consecutive files
plausibly come from consecutive seconds over the same track section, which would
make near-duplicate files land on both sides of a fold boundary and inflate
every number this plan produces. Nothing in the Info Kit confirms or denies
this. See §7.3 — this is the single largest methodological threat to the result
and it must appear in the final report as such, not only in this plan.

**Mandated mitigation (diagnostic, not a split change):** the implementer MUST
produce, as part of the test evidence, an **adjacency diagnostic**: for the
cached feature matrix, compute the correlation (or cosine similarity) between
each file's feature vector and that of file index `n+1`, and compare the
distribution of adjacent-index similarity against the distribution of
random-pair similarity. Report both distributions' median and interquartile
range.

- If adjacent-index similarity is not materially higher than random-pair
  similarity, the file ordering carries no obvious block structure and the
  plain stratified split is defensible on the evidence available.
- If adjacent-index similarity **is** materially higher, that is a finding that
  returns to the planner for a split revision (a blocked/contiguous split by file
  index) — it does NOT authorise the implementer to change the split unilaterally.

This diagnostic replaces a guess with a measurement. It does not eliminate the
risk, because similar files could be non-adjacent in index order.

#### 4.4.1 Fault-file clustering check — A HARD GATE. **PASSED at iteration 1.**

**Status: this gate ran at Gate 1 and PASSED. It is not re-run.** The evidence is
recorded in `docs/plans/rail-objection.md` §1 and is summarised here because the
result is now an established fact the plan depends on:

- Side I, 14 files, indices `[62, 83, 100, 106, 121, 137, 150, 163, 170, 180,
  185, 194, 202, 213]`. Largest run at gap ≤ 2: **1**. Runs of length ≥ 3: **0**.
- Side II, 24 files, indices `[2, 12, 26, 30, 43, 53, 72, 93, 103, 104, 111, 127,
  164, 179, 188, 198, 199, 207, 216, 217, 240, 248, 265, 267]`. Largest run at
  gap ≤ 2: **2**. Runs of length ≥ 3: **0**.
- Seeded random reference (seed `20260918`, 10,000 draws per class): largest-run
  median 2.0 for k=14 and 2.0 for k=24. Both observed values sit **at or below**
  the random median. `P(random largest_run >= observed)` = 1.0000 (Side I) and
  0.9911 (Side II).

**Verdict: PASSED. Observed clustering is less than chance.** The stratified
split of §4.1 is not challenged on clustering grounds. Per this section's own
wording this remains **weak** evidence — an interleaved acquisition order would
scatter files from one session — so the §7.3 independence assumption stands as an
assumption, not as an established fact.

The original gate specification is retained below for the record and because the
reviewer checks that the evidence produced matches what was demanded.

**What to compute.** Extract the integer index `n` from each `TrainN.csv` in
`Train_Labels.csv` and, separately for the 14 Side I files and the 24 Side II
files, report:

- the full sorted list of indices for each class (14 and 24 numbers — small
  enough to print in full, and the raw list is the primary evidence);
- the gaps between consecutive indices within each class;
- the number of maximal runs of consecutive or near-consecutive indices, where
  "near-consecutive" means a gap of 1 or 2;
- the size of the largest such run;
- for reference, the same statistics for a random sample of 14 and 24 indices
  drawn from 1..272, so the observed clustering is compared against what
  scattering actually looks like rather than against intuition.

**THE GATE (as originally written).** If the fault files of either class are
clustered — any run of 3 or more consecutive-or-near-consecutive indices, or a
largest-run length clearly outside what the random reference shows — the
implementer MUST report the finding and STOP. This condition did not fire.

#### 4.4.2 Side I index span — RESIDUAL CAVEAT, not a gate, not a blocker

The implementer reported (objection §1a) a clustering-adjacent statistic the gate
criterion does not cover: **Side I occupies indices 62 to 213 only, a span of 151
out of a possible 271**, with `P(random span ≤ 151) = 0.0026` at k=14 against the
same seeded reference. Side II shows nothing comparable (span 265, `P = 0.8865`).

**Decision: record as a residual caveat; do NOT block on it.** Reasoning:

1. The gate criterion in §4.4.1 is defined on **runs**, and runs are the
   statistic that bears on the concern the gate exists to catch — that the fault
   files are near-duplicate recordings of one or two physical events. A run of 3+
   would mean consecutive seconds of the same pass. Largest run = 1 rules that
   out directly. A wide-but-bounded span does not imply duplication.
2. The implementer was correct not to treat it as a gate failure. Inventing a
   criterion after seeing the data, then applying it to the same data, is
   exactly the post-hoc testing this plan forbids elsewhere. I am not going to
   do at the plan level what I forbade at the implementation level.
3. `p = 0.0026` is a single statistic on 14 files, selected *after* looking at
   the data. Under any reasonable accounting for the number of statistics one
   could have computed on this index list, it is not decisive.
4. Two innocent readings exist and cannot be distinguished: the Side I recording
   campaign occupied a bounded portion of the acquisition, or it is chance at
   k=14.

**What is mandated:** the caveat appears in §7.3 and MUST appear in the final
write-up, stated as "Side I file indices span 62-213 out of 1-272; under a
seeded random reference this span is unusually narrow (p = 0.0026). The
run-length gate passed, which rules out consecutive-second duplication, but a
bounded acquisition window for Side I cannot be excluded and would reduce the
effective number of independent Side I examples below 14." No further
computation is required. **AC-27.**

### 4.5 What is reported

For each of the 50 folds, record: macro F1, and the three per-class F1 values,
all from `src/common/metrics.py`, **computed twice — once on the full validation
fold and once on the restricted subset of §4.7.** Then report, for each of the
two evaluation domains separately:

- **Mean macro F1 across 50 folds.**
- **Standard deviation across 50 folds.**
- **10th and 90th percentile of macro F1 across 50 folds.**
- **Mean, std, and min of per-class F1 for each of the three classes
  separately.** Side I F1 is the number that matters; a good macro F1 carried
  entirely by Normal and Side II is not a success.
- **Count of folds where Side I F1 == 0.0**, out of 50.
- **A pooled confusion matrix**, summed over all 50 folds, 3×3, with rows = true
  and columns = predicted, class order `RAIL_CLASSES`. This is where a
  Side I ↔ Side II confusion shows up, which is the physically interesting
  failure mode (§5.2).

**The headline is the RESTRICTED number** (§4.7). The full-set number is
reported alongside it, in the same table, never alone.

No single fold's score is ever reported alone, and the maximum-over-folds score
is never reported as the result. Reporting the best fold is a defect.

### 4.6 What this split prevents, and what it does not

**Prevents:**

- *Single-split lottery.* With 14 Side I files, one random split's Side I F1 is
  dominated by which 2-3 files happened to land in validation. Repeating 10 times
  turns a lottery ticket into a distribution.
- *Absent-class folds.* Stratification guarantees every validation fold contains
  at least 2 Side I and 4 Side II files, so no fold silently scores macro F1 on a
  degenerate two-class problem.
- *Tuning-on-validation leakage.* All preprocessing that learns anything from
  data — scaling, imputation, feature selection, resampling, class weighting
  computed from counts — must be fitted **inside** the fold, on the training
  portion only, via an `sklearn` `Pipeline` passed to the CV loop. Fitting a
  scaler on all 228 eligible files and then cross-validating is a defect the
  reviewer checks for explicitly.
- *Test-set contamination.* No Test file is ever loaded during feature design,
  model selection, or threshold setting, with the **single declared exception**
  of the Gate 1b domain check (§4.8), which reads column 1 only and may not
  influence any model choice.
- *The speed shortcut* — newly, via §5.3b (speed struck as a classifier input)
  and §4.7 (restricted evaluation). Neither of these is a property of the
  splitter; both are properties of the protocol the splitter runs inside, and
  both are mandatory.

**Does not prevent:**

- *Temporal/spatial correlation between files*, because no grouping key exists
  (§4.4). Stated openly.
- *Model-selection overfitting across iterations.* Every time a human looks at
  the 50-fold mean and changes a hyperparameter, the mean becomes slightly
  optimistic. Mitigation is procedural and mandatory: the plan permits at most
  the model comparison enumerated in §5.5. The count of distinct configurations
  evaluated MUST be reported. An undeclared search is a defect.
- *Small-sample uncertainty in general.* 14 files is 14 files. No split fixes
  that, and the report must say so.
- *A speed confound operating through the vibration signal itself.* Striking `v`
  as a feature removes the direct shortcut. It does not remove the possibility
  that raw amplitude, which rises with speed, carries the same information
  indirectly. §4.7's restricted evaluation is the check on that residue, and it
  is not a complete one either. Stated plainly rather than claimed away.

### 4.7 Speed-stratified evaluation — the restricted domain is the headline

**NEW in iteration 2. This is a reporting and evaluation requirement, not a
splitter change.**

#### 4.7.1 Why

Gate 1 established that no fault file in Train sits below 9.7023 m/s, while 133
of 234 Normal files do (objection §3). A model can therefore reach macro F1
~0.509 by learning "fast ⇒ fault" and nothing about corrugation. Striking `v` as
a feature (§5.3b) removes the *direct* route to that shortcut but not every
route: amplitude-like features rise with speed, and any of them can proxy for
`v`.

The only way to know whether the model has learned corrugation rather than speed
is to **score it where speed cannot separate the classes.** Above the fault-class
speed floor, both fault classes are present and Normal files are present too, so
a speed-only rule has no purchase by construction. That subset is where the
honest number lives.

#### 4.7.2 The cut

**Speed cut `V_CUT = 9.70 m/s`**, a module constant. Files with derived
`v >= V_CUT` are in the restricted domain; files below are out of it.

The value is taken from the measured Side I minimum, 9.7023 m/s (objection §3),
rounded **down** to 9.70 so that all 14 Side I files fall inside the restricted
domain rather than one balancing on the boundary. This is an **assumption
derived from the training data, not from the spec** (§7.11). It is a
data-derived constant used for *reporting stratification only*: it never enters
a feature, never enters a decision rule, and is never fitted to maximise any
score. That distinction is what keeps it from being leakage: the cut partitions
the evaluation, it does not inform the model.

#### 4.7.3 Expected composition — derived, and to be verified

From the per-class distributions the implementer measured:

| Class | Eligible n | Min v (m/s) | Expected n with v ≥ 9.70 |
|---|---|---|---|
| Normal | 190 | ≥ 1.0 by construction | **≈101** |
| Side I | 14 | 9.7023 | **14 (all)** |
| Side II | 24 | 11.6902 | **24 (all)** |
| **Total** | **228** | — | **≈139** |

Derivation: all 38 fault files exceed 9.70 by measurement, so both fault classes
survive the cut entirely. For Normal, the objection states 133 of 234 fall below
the fault floor, leaving 234 − 133 = **101** above it. The 44 excluded low-speed
files are all inside those 133, so the eligible-Normal count above the cut is
**also 101** — the exclusion and the restricted cut remove only Normal files from
the same low-speed tail, and the cut is the stricter of the two. The restricted
domain is therefore ≈139 of 228 eligible files, **≈61%**.

**The ≈101 figure is DERIVED, not measured.** The objection's "133 of 234" was
stated against the Side I minimum 9.7023, and the cut here is 9.70. The two can
differ only if a Normal file sits in `[9.70, 9.7023)`, which is possible.
**The implementer MUST report the exact realised counts at `V_CUT = 9.70` before
any restricted metric is relied upon (AC-28), and MUST use the realised counts,
not this table's estimate, in all downstream reasoning.** If the realised Normal
count differs from 101 by more than a few files, that is a discrepancy to report,
not to absorb silently.

#### 4.7.4 Fold composition in the restricted domain, and why the splitter does not change

The restricted subset is **not** a separate split. The procedure is:

1. Build the 50 folds with `RepeatedStratifiedKFold` over all 228 eligible files,
   exactly as §4.1 says. Train on the full training portion of each fold —
   **including the below-cut files.** They are legitimate training data; only the
   *scoring* is restricted.
2. For each fold, take the validation portion and score it twice: once entire
   (the full-set number), and once on `{files in the validation portion with
   v >= V_CUT}` (the restricted number).

**Consequence for restricted fold size.** The restricted domain is ≈139 of 228
eligible files, ≈61%. A validation fold of 45-46 files therefore yields a
restricted validation subset of **roughly 26-30 files**, containing **all** the
fold's Side I and Side II files (both fault classes are entirely above the cut,
so the cut removes only Normal files) and **≈20-21 Normal files** instead of 38.

This is the important structural point and it must be understood before reading
any restricted number:

- **Side I per restricted fold: still 2 or 3.** Unchanged. The cut removes no
  Side I file. The quantisation of §4.2.1 applies identically.
- **Side II per restricted fold: still 4 or 5.** Unchanged.
- **Normal per restricted fold: ≈20-21 instead of 38.** Normal F1 becomes
  noisier in the restricted domain, and Normal precision in particular falls
  because there are about half as many Normal files for a false positive to be
  drawn from. **Expect the restricted macro F1 to be LOWER than the full-set
  macro F1 for a genuinely good model**, not because the model is worse there but
  because the class balance is harder and the Normal denominator is smaller. A
  restricted number materially *higher* than the full-set number is a signal to
  investigate, not to celebrate.

**Is the restricted subset too thin to support the planned splitter? No.** The
binding constraint on this subsystem was always Side I at 2-3 per fold, and the
cut does not touch it. Approximately 26-30 files per restricted fold, with all
three classes present in every one, is not degenerate. **The minimum viable k is
therefore unchanged at k=5**, for exactly the reason given in §4.3: k=10 would
put 1-2 Side I files per fold, and that is degenerate in both domains.

**Stratification is on the class label only, NOT on the speed bin.** Adding speed
as a second stratification key would guarantee an exact restricted-subset size
per fold, but it would also mean the splitter uses a data-derived speed quantity
to build the folds, which couples the evaluation design to the confound it is
meant to expose. The restricted fold sizes will vary slightly (roughly 26-30)
because Normal files land where the class stratification puts them. That
variation is accepted and reported. **The implementer MUST print the realised
restricted-subset size and per-class composition for every one of the 50 folds,
or at minimum the min/median/max across folds plus the full breakdown for one
repeat (AC-28).**

**Failure condition.** If any fold's restricted subset turns out to contain zero
files of some class — which the derivation above says cannot happen, since all
fault files are above the cut — that fold's restricted macro F1 is still computed
with a denominator of 3 per §2.2, and the occurrence MUST be reported as a count
out of 50. If it occurs on more than a handful of folds, stop and return to the
planner; it would mean the measured distributions are not what the objection
reported.

#### 4.7.5 What is reported, and which number is the headline

Every score table carries **two columns**: `full` and `restricted (v >= 9.70)`.
Both carry mean, std, p10, p90 over the 50 folds, and both carry the three
per-class F1 breakdowns.

**The RESTRICTED macro F1 is the headline number for this subsystem.** It is
what goes in the write-up's first sentence, what the model-selection rule of
§5.5 uses, and what the acceptance criteria are checked against. The full-set
number is reported immediately alongside it and is never suppressed.

Justification: the restricted domain is the only one in which a good score cannot
be produced by the speed shortcut. With zero leaderboard uploads, nothing
external will ever catch a speed-shortcut model, so the internal evaluation has
to be the one that catches it. A defensible 0.4 in the restricted domain is worth
more to this project than an undefended 0.6 on the full set — that is a direct
application of the session's "a defensible number beats a high one" constraint.

**Both numbers are reported for every model and every baseline**, B0, B1, B2 and
M1-M3 alike, so the comparison is like-for-like.

### 4.8 GATE 1b — the declared Test-set domain check

**NEW in iteration 2. This is a deliberate, one-time, declared use of Test
input.**

#### 4.8.1 What it is and why it is sanctioned

The 44 low-speed Train files force the plan to define an inference rule for
low-speed inputs (§5.3a.4), and §4.7 defines a restricted evaluation domain from
a speed cut. Both raise the same question: **does the deployment domain look like
the evaluation domain?** That question cannot be answered from Train alone.

This is Test *input*, not Test *labels*. It is nonetheless Test access and the
plan treats it as such: it is declared here in advance, it is bounded to a single
computation, it is logged, and it is surfaced in the write-up. Undeclared Test
inspection would be a defect; a declared, bounded, reported one is a
methodological disclosure. **The plan chooses disclosure over ignorance and says
so.**

#### 4.8.2 EXACTLY what is computed

For each of the 68 Test files, read **column 1 only** — via
`pandas.read_csv(path, usecols=[0])`, which is ~80 KB per file, not a full read —
and apply the §5.3 transition-counting derivation to obtain `v`. Then report:

1. The count of Test files with `v == 0` exactly (zero column-1 transitions).
2. The count with `0 < v < V_MIN` (1.0 m/s).
3. The count with `v >= V_CUT` (9.70 m/s), and the count below it.
4. The five-number summary of `v` across the 68 files: min, p25, median, p75,
   max, in m/s and km/h.
5. The transition count `T` summary: min, median, max.
6. Confirmation that column 1 in Test is binary `{0,1}` as it is in Train, with
   the count of Test files containing any non-binary value.

**That is the complete list. Nothing else.**

#### 4.8.3 EXACTLY what is forbidden

- **No reading of columns 2-129 of any Test file for this check.** The
  `usecols=[0]` restriction is mandatory and the reviewer checks for it.
- **No feature extraction on Test** as part of this check. Test feature
  extraction happens once, later, at prediction time (§5.4).
- **No fitting of anything on Test.** No scaler, no imputer, no selector, no
  threshold, no class prior, no calibration.
- **No use of the result to select a model.** The Gate 1b output must not change
  which of M1-M3 is chosen, must not change any hyperparameter, must not change
  `V_CUT`, must not change `V_MIN`, must not change the wavelength bands, must
  not change the exclusion rule of §5.3a, and must not change the inference rule
  of §5.3a.4. Those are all fixed by this plan **before** Gate 1b runs and they
  are fixed by Train evidence and by argument, not by Test.
- **No expansion.** This is one check, run once, producing the six items above.
  It is not a licence for ongoing Test inspection. Any further Test access is a
  new planner decision.
- **No per-file Test values printed.** Aggregate statistics and counts only, per
  the "report shapes, never contents" constraint.

#### 4.8.4 What the result IS allowed to inform

Exactly two things, both of them **reporting**, neither of them modelling:

1. **Whether the low-speed inference rule of §5.3a.4 ever fires.** If Test
   contains low-speed files, the write-up must say how many and must state that
   those predictions come from a stated rule rather than from the model. If Test
   contains none, the write-up says so and the rule is documented as a defensive
   path that was never exercised. Either way the rule itself is unchanged — it is
   fixed before this check runs.
2. **Whether the restricted evaluation domain of §4.7 matches the deployment
   domain.** If most Test files sit above 9.70 m/s, the restricted number is the
   directly relevant one and the write-up says so. If most sit below, the
   write-up must state plainly that the headline restricted metric was measured
   on a domain that is not where the model will mostly be applied, and that this
   weakens the connection between the local number and the held-out score. **This
   is a caveat to be written down, not a trigger to retrain.**

#### 4.8.5 When it runs, and the audit trail

Gate 1b runs **after** this plan is accepted and **before or alongside** feature
extraction — its ordering relative to modelling does not matter because it cannot
influence modelling. Its output is written verbatim to the test evidence file and
is reproduced in the write-up. **AC-29.**

---

## 5. Modelling approach

### 5.1 Overview

A **file-level feature extraction** stage producing a compact fixed-width
feature vector per file, cached to disk, followed by a **classical tabular
classifier** trained inside the CV loop. No deep learning, no raw-waveform
model. The reasons: 228 eligible training files with 14 minority examples cannot
support a high-capacity model, and the cached-feature design (§5.4) is what makes
50 CV fits affordable.

### 5.2 Physical structure of the signal — the central design decision

Info Kit §2.1 lines 66-68, verbatim:

> Positions 1, 3, 5, and 7 correspond to the Side I rail, and positions 2, 4, 6,
> and 8 correspond to the Side II rail; the states of the Side I and Side II
> rails must be judged separately.

Info Kit §1.2 lines 54-56, verbatim:

> Side I and Side II rails must be judged **independently from the same
> recording**: a file may show corrugation on one side while the other remains
> normal, so the model must localise the fault to a side rather than simply
> flagging the file as anomalous.

**Channel layout, per Info Kit §2.1 lines 78-82.** Columns 2-129 are ordered:
Car 1 Pos 1 vibration, Car 1 Pos 1 shock, Car 1 Pos 2 vibration, Car 1 Pos 2
shock, ..., Car 8 Pos 8 vibration, Car 8 Pos 8 shock. With 0-based column index
`j` over columns 2..129 (so `j = 0..127`):

- `car   = j // 16 + 1`   (1..8)
- `pos   = (j % 16) // 2 + 1`  (1..8)
- `kind  = "vibration" if j % 2 == 0 else "shock"`
- `side  = "Side I" if pos is odd else "Side II"`

This gives **64 Side I channels** (32 vibration + 32 shock, from positions
1,3,5,7 across 8 cars) and **64 Side II channels**, identically structured.

**CONFIRMED at Gate 1.** The implementer parsed all 128 channel header strings
and checked each against this arithmetic: **all 128 match** (objection §4). The
headers are descriptive — column 1 is `'Rotating speed'`, channels are
`'<Vibration|Shock> of bearing in position <p> of car <c>'` — and are
byte-identical across sampled Train and Test files. §7.6 is RESOLVED. The
positional-access mandate of §5.4 stands, and a name-based assertion is now
additionally available as a cheap extra check; the implementer MAY add one but
positional access remains the primary path.

**DECISION: per-side feature extraction with an explicit Side I vs Side II
comparison. NOT a single flat 128-channel model.**

The design:

1. Compute the same feature function `f(channel) -> R^d` for every one of the
   128 channels.
2. Aggregate across the 32 channels of each (side, kind) group using
   **order statistics**, not just the mean: median, 90th percentile, max, and
   interquartile range across channels. Order statistics matter because
   corrugation is a property of a *track section* that all 32 same-side boxes
   traverse, but sensor faults, wheel flats and per-box noise affect single
   channels; the median is robust to those while the 90th percentile and max
   retain sensitivity to a genuine strong signal.
3. This yields `A_I` = the Side I aggregate vector and `A_II` = the Side II
   aggregate vector, with identical layout and identical length.
4. **Emit, as explicit features, the contrast between them**: elementwise
   `A_I - A_II`, and a log ratio `log((A_I + eps) / (A_II + eps))` for the
   strictly positive features (energies, band powers, RMS). `eps` is a fixed
   small constant defined once as a module constant.
5. Also emit `A_I` and `A_II` themselves, and their symmetric combination
   `(A_I + A_II)/2`, so the model can use both absolute level and contrast —
   **subject to the speed-feature rule of §5.3b, which constrains which of these
   absolute-level blocks may enter the model matrix.**

**Why this and not a flat 128-channel model.**

The label is a *side*, and the difference between `Side I` and `Side II` is a
sign flip of exactly the same physical phenomenon. A per-side contrast feature
makes that antisymmetry explicit in one dimension: `A_I - A_II` is positive for
Side I, negative for Side II, near zero for Normal. A flat 128-channel model
would have to *learn* that antisymmetry from 14 Side I and 24 Side II examples,
discovering by itself that channel 0 and channel 2 belong together and channel 1
belongs to the other group. With 38 total fault examples it will not reliably do
so, and whatever it does learn is as likely to encode "which car" as "which
side".

The contrast design also buys **built-in symmetry as a diagnostic**. Because
`A_I` and `A_II` are computed by identical code over identically structured
channel groups, swapping the two sides' channels must flip a `Side I` prediction
to `Side II` and leave `Normal` alone. That is a testable invariant (§6, AC-13)
that a flat model simply does not possess.

**Iteration 2 strengthens this argument.** The contrast family is also the
**speed-robust** family (§5.3 mechanism 4): a multiplicative speed factor common
to both sides cancels exactly under a log ratio. Now that speed is known to be a
confound, the contrast design is no longer merely the better-conditioned choice
— it is the *confound-resistant* choice, and §5.3b promotes it accordingly.

**What the flat 128-channel alternative would cost.** It is not worthless — it
would preserve per-car information that the aggregation discards, which could
matter if corrugation registers more strongly under the leading bogie (the Info
Kit mentions attack angle of the leading wheelset at §1.1 lines 28-30, though it
says this about corrugation *formation*, not about detection strength, so this is
a conjecture and not a spec claim). The costs are: ~128 × d raw features against
228 samples, a dimensionality that invites overfitting; no built-in side
symmetry; no symmetry invariant to test; and interpretability loss, since a
learned feature importance over 128 anonymous channels tells a judge nothing. On
14 minority examples, the variance cost dominates the information gain.

**Mandated hedge.** Because the per-car question is genuinely open, the
aggregation MUST additionally emit, for each side and kind, the **across-car
dispersion** of the per-channel feature (standard deviation across the 32
channels, and max-minus-median). If corrugation is localised to particular cars,
that shows up as elevated dispersion, and the information is not wholly thrown
away. This is cheap — it is another order statistic over the same per-channel
values — and it keeps the discarded axis partly recoverable without a 128-wide
model.

### 5.3 Speed: derivation and the four mechanisms

This must not be left implicit. Info Kit §1.1 lines 45-46 states the problem
directly:

> This non-contact, efficient, online-monitoring method is, however, affected by
> train speed and ballast noise, and requires signal processing and machine
> learning to improve accuracy.

And §1.1 lines 42-44: "corrugation excitation generates characteristic-frequency
vibrations ... Time-frequency analysis extracts the dominant wavelength".

**The physics the spec gives us.** Corrugation is a *spatial* wear pattern with a
characteristic **wavelength** λ (Info Kit §1.1 lines 11-13: "wavelength typically
ranges from a few centimetres to dozens of centimetres"). A wheel rolling over it
at speed `v` excites a *temporal* frequency `f = v / λ`. Therefore a fixed
frequency band in Hz corresponds to a *different* wavelength at every speed, and
a fixed-band feature is measuring different physical things in fast and slow
files. **A fixed Hz band is wrong**, and the plan does not use one as its primary
representation.

**This argument is now supported by measurement, not only by physics.** Gate 1
found derived speed ranging from 0.0445 to 19.479 m/s across nonzero-speed files,
a factor of 438, and a factor of 3.4 across the interquartile range alone
(objection §4). Speed varies more than enough for the normalisation to matter.

**Speed derivation, from Info Kit §2.1 lines 74-77.** Column 1 is the output of a
toothed-wheel sensor: 90 teeth evenly distributed around the circumference,
output toggling between 1 and 0 as each tooth enters and leaves the detection
point, "by counting the 0/1 transitions over a specific time, the train running
speed can be calculated". Wheel diameter 0.85 m. Sampling 10,000 Hz, 1 s per
file.

The derivation the implementer must use, stated as specification:

- Count `T` = the number of 0→1 and 1→0 transitions in column 1 over the file.
- Each tooth produces 2 transitions (one entering, one leaving), so revolutions
  `R = T / (2 × 90) = T / 180`.
- Wheel circumference `C = π × 0.85 m ≈ 2.670354 m`.
- Over the file's duration `t = 1.0 s`: distance `= R × C`, speed
  `v = R × C / t` m/s.
- Report `v` in m/s and also `v × 3.6` in km/h.

**Column 1's encoding is RESOLVED, not assumed.** Gate 1 measured it across all
272 Train files: `dtype=int64`, values literally `{0, 1}`, zero NaNs, zero
non-binary files. No thresholding is required and none is performed; transitions
are counted directly as `count(diff != 0)`. Nyquist saturation is also ruled out:
the fastest file runs at 656.5 teeth/s, 15.23 samples per tooth period, 13.1% of
the 10,000-transitions-per-second capacity. §7.4 is RESOLVED on both its primary
question and its sub-ambiguity.

**The sanity band, restated for iteration 2.** The iteration-1 wording ("if
derived speeds fall outside a plausible metro operating range, the derivation is
wrong and the implementer must stop") fired at Gate 1 on the 38 zero-speed files,
and correctly so. That stop condition has now been **adjudicated by the planner
in §5.3a** and is replaced by an explicit rule. The implementer no longer stops
on zero speed; the implementer applies §5.3a. The sanity band remains live for
any *other* implausibility: a derived speed above the observed maximum by a wide
margin, or a negative value, or a NaN, remains a stop condition returning to the
planner.

**How speed is handled — the four mechanisms, as revised by §5.3b:**

1. **Order-domain / wavelength-domain band features (primary). SURVIVES,
   unchanged and mandatory.** Rather than fixed Hz bands, define bands in the
   **spatial-wavelength domain**. For a target wavelength band `[λ_lo, λ_hi]` in
   metres, the corresponding temporal band for a file at speed `v` is
   `[v/λ_hi, v/λ_lo]` Hz. Compute the power spectral density once per channel
   (Welch, parameters in §7.7), then integrate it over the **speed-dependent** Hz
   edges derived per file. The band edges are therefore different for every file,
   which is exactly the point: the same feature index always means the same
   physical wavelength.

   The wavelength bands are fixed by the Info Kit's own range, §1.1 lines 11-12
   ("a few centimetres to dozens of centimetres"). Mandated bands, in metres, as
   a module constant: `[0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [0.16, 0.32],
   [0.32, 0.64]` — five logarithmically spaced bands spanning 2 cm to 64 cm,
   covering "a few centimetres to dozens of centimetres" with margin at both
   ends. This band choice is an **assumption** derived from prose, not a spec
   constant (§7.5). Bands whose upper Hz edge exceeds Nyquist (5,000 Hz) or
   whose lower edge falls below the frequency resolution are set to `NaN` for
   that file and imputed inside the pipeline; they are not silently zeroed.

   Gate 1 confirmed this mechanism is sound at operating speeds: at the median
   9.799 m/s the bands are `[244.97, 489.94]`, `[122.48, 244.97]`,
   `[61.24, 122.48]`, `[30.62, 61.24]`, `[15.31, 30.62]` Hz, all resolvable at
   the 4.883 Hz Welch resolution and all far below Nyquist; at the maximum
   19.479 m/s the top band reaches 973.94 Hz, still well below 5,000 Hz. The
   mechanism broke only at near-zero speed, which §5.3a removes from training.

   **This mechanism is the canonical example of "normalise, not predict"
   (§5.3b): `v` sets where to look in the spectrum; it does not itself become a
   number the classifier can read off.**

2. **Speed as an explicit feature. STRUCK. See §5.3b.** Iteration 1 mandated
   emitting `v` and its within-file variability as model features. **That is
   revoked in its entirety.** Neither `v`, nor `v` in km/h, nor the transition
   count `T`, nor the windowed transition-rate statistics, nor any other
   quantity derived from column 1 alone, may enter the model feature matrix.

3. **Speed-normalised amplitude. PARTIALLY STRUCK — this is the subtle one and
   §5.3b.3 adjudicates it in full.** Iteration 1 mandated emitting every
   energy/RMS feature twice, raw and divided by `v` (and `v²`). Iteration 2
   permits the **speed-normalised** variant and strikes the **raw** variant for
   absolute-level features. Full reasoning and the exact rule in §5.3b.3.

4. **Contrast is inherently speed-robust. SURVIVES and is PROMOTED.** The
   `A_I - A_II` and `log(A_I / A_II)` features of §5.2 compare two sides of the
   *same* file at the *same* speed, so the speed dependence largely cancels. This
   is a second, structural reason the per-side contrast design was chosen over a
   flat model, and it is why the log-ratio form is mandated specifically: a
   multiplicative speed factor common to both sides vanishes exactly under a log
   ratio. Iteration 2 makes this the **primary** feature family (§5.3b.4).

**The residual risk is no longer hypothetical; it is measured.** §5.3b and §4.7
are the response.

### 5.3a Zero-speed and low-speed files — the exclusion rule

**NEW in iteration 2. This resolves implementer Objection A.**

#### 5.3a.1 The finding

Gate 1 measured, across all 272 Train files: **38 files have zero column-1
transitions**, so `T = 0` and `v = 0.0000 m/s` exactly. Their column 1 is
constant at `1` for all 10,000 samples. **All 38 are labelled Normal** (0 Side I,
0 Side II). A further **6 files** yield `0 < v < 1.0 m/s`, the smallest nonzero
being 0.0445 m/s; these are also all Normal. The vibration and shock channels in
the zero-speed files carry live signal — `Train5.csv` channels 1-4 have standard
deviations `[0.0361, 1.370, 0.0334, 1.524]` — so the recordings are not dead; it
is specifically the speed channel that is flat.

#### 5.3a.2 THE RULE — exclusion from training, with V_MIN DERIVED

**Files with `v < V_MIN` are EXCLUDED from the model's training and
cross-validation population.** `V_MIN` is **derived, not chosen** — see §5.3a.2a.
Its value is **`V_MIN = 1.28 m/s`**. That is 44 files — the 38 with `v = 0` plus
the 6 with `0 < v < 0.49` — all of them Normal. The eligible set is **228 files:
190 Normal / 14 Side I / 24 Side II** (§4.2.0).

**Reasoning — physical, and it is the whole justification.** The task defined by
the Info Kit is the detection of *rail corrugation*, a spatial wear pattern
detected through the vibration a wheel produces **while rolling over it**. Info
Kit §1.1 lines 42-44 makes the mechanism explicit: "corrugation excitation
generates characteristic-frequency vibrations" at `f = v/λ`. **At `v = 0` there
is no wheel-rail rolling interaction, so there is no corrugation excitation, so
the quantity the task asks the model to detect does not exist in the signal.**
The file may contain vibration from auxiliary machinery, from an adjacent train,
or from anything else, but it cannot contain corrugation excitation. Training a
corrugation detector on files where corrugation is physically undetectable
teaches it to map non-corrugation vibration to the label `Normal`, which is a
correlation of the acquisition, not of the physics.

**Two secondary reasons, which are consequences rather than justifications:**

- Mechanism 1 collapses: at `v = 0` every wavelength band edge is `v/λ = 0` Hz
  and all five bands are empty. The primary feature family is **entirely** NaN
  for these files, not partially degraded. §5.3's NaN-and-impute provision was
  written for occasional missing bands, not for a fifth of the corpus having none.
- Mechanism 3 is undefined: dividing by `v = 0` is a divide-by-zero, and an `eps`
  guard converts it into an arbitrary enormous number rather than a meaningful
  feature. The implementer was right that this is not a fix.

#### 5.3a.2a DERIVATION of V_MIN — auditable, not asserted

**Amendment A, iteration 2.** `V_MIN` was previously the round number 1.0 m/s,
defensible but chosen. It is now derived from the band design and one stated
assumption, so a reviewer can check the arithmetic rather than accept a
judgement.

**The requirement.** Corrugation is a **periodic** spatial pattern (Info Kit §1.1
lines 11-13, "wavelength typically ranges from a few centimetres to dozens of
centimetres"). For a file to carry evidence of a periodic pattern at wavelength
λ, the wheel must traverse **at least N full spatial periods of λ within the
observation window**. Below that, whatever is in the signal is not evidence of
periodicity.

**The formula.** A wheel at constant speed `v` covers `v · t` metres in an
observation window of length `t`. The number of spatial periods of wavelength λ
traversed is `v · t / λ`. Requiring at least `N` periods at the **longest**
wavelength the feature set claims to measure, `λ_max`:

```
v · t / λ_max  ≥  N
```

which rearranges to the definition:

```
V_MIN  =  N · λ_max / t
```

**The inputs, each traceable:**

| Symbol | Value | Source |
|---|---|---|
| `λ_max` | **0.64 m** | §7.5, the upper edge of the longest mandated wavelength band `[0.32, 0.64]` |
| `t` | **1.0 s** | Info Kit §2.1 line 81 — each file is 1 second |
| `N` | **2** | **Assumption, §7.14.** Minimum periods for periodicity to be evidenced rather than assumed |

**The arithmetic:**

```
V_MIN  =  2 × 0.64 m / 1.0 s  =  1.28 m/s   ( = 4.608 km/h )
```

**Why N = 2 and not 1.** At `N = 1` the window contains exactly one spatial
period. One period is indistinguishable from a single isolated transient — a rail
joint, a switch, a wheel flat impact, a one-off shock. Corrugation is defined by
its *periodicity*, and periodicity cannot be asserted from a single cycle: two
cycles is the smallest count at which the signal can evidence repetition, which
is the property actually being detected. `N = 1` would admit files in which the
longest band can only ever see a transient. `N = 3` or more would be defensible
too and would exclude more files; `N = 2` is the **least aggressive** choice that
still requires repetition, which is the right side to err on when the exclusion
discards real labelled data. **`N` is a choice and is flagged as an assumption in
§7.14**, exactly as the band edges are flagged in §7.5.

**On the choice of λ_max as the basis, recorded for the write-up.** Basing the
threshold on `λ_max` excludes a file whenever its *longest* band is unresolvable,
even though §5.3 mechanism 1 already handles unresolvable bands per-band via NaN
and in-fold imputation; a `λ_min` basis would instead give
`V_MIN = 2 × 0.02 / 1.0 = 0.04 m/s` and a nearly identical eligible set, and the
difference between the two bases is at most 7 Normal files and **zero** minority
files, so the choice is **not load-bearing**.

**Note on what is NOT used in this derivation.** An alternative derivation from
the **Welch frequency resolution** — requiring the lowest band edge `v/λ_max` to
sit at least one resolution bin `Δf = fs/nperseg = 10000/2048 = 4.8828 Hz` above
DC — would give `V_MIN = Δf · λ_max = 4.8828 × 0.64 = 3.125 m/s`, excluding
roughly 60 files rather than 44. **That derivation is deliberately NOT used**,
for two reasons. First, the user's ruling fixes the basis as the spatial-period
argument. Second, and independently, the Welch-resolution question is about
whether a *particular band* is resolvable in a *particular file*, and §5.3
mechanism 1 already handles it correctly and per-band by setting unresolvable
bands to `NaN` and imputing them in-fold. Folding it into a global eligibility
threshold would double-handle it and would discard files whose shorter bands are
perfectly resolvable. The two mechanisms address different questions and are kept
separate. Recorded here so the reviewer can see the alternative was considered
and why it was rejected.

#### 5.3a.3 What V_MIN = 1.28 excludes — measured, not estimated

**The threshold moved from 1.0 to 1.28 m/s under Amendment A. The eligible set
did not move at all.**

**What Gate 1 measured** (`docs/plans/rail-objection.md` §2, §3):

- **38 files at `v = 0`** exactly — all Normal.
- **6 files at `0 < v < 1.0`** — all Normal. So **44 below 1.0 m/s**, all Normal.
- **51 files in the `[0, 2)` m/s histogram bin — 51 Normal, 0 fault.**

**What direct measurement at 1.28 then established.** The draft of this amendment
carried a range of 221-228, because Gate 1 measured at the 1.0 and 2.0 boundaries
but never at 1.28. That gap has been closed: **there are ZERO files in
`[1.00, 1.28)`.** The seven files in `[1.0, 2.0)` sit at 1.2907, 1.2907, 1.3352,
1.6764, 1.7506, 1.8841 and 1.9138 m/s — every one of them **above** 1.28, and
every one retained.

**The speed distribution has a wide empty gap where the threshold lands.** The
fastest excluded file is at **0.4896 m/s**; the slowest retained one is at
**1.2907 m/s**. Nothing whatsoever lies between. Any threshold in
`(0.4896, 1.2907)` produces the identical eligible set, and the derived 1.28 sits
inside that interval. **The derivation buys auditability at zero cost to any
downstream number.**

**The eligible set, measured:**

| Class | Full corpus | Excluded at `v < 1.28` | **Eligible** |
|---|---|---|---|
| Normal | 234 | **44** (38 at `v = 0`, 6 in `(0, 0.49]`) | **190** |
| Side I | 14 | **0** (certain — no fault file below 2.0 m/s) | **14** |
| Side II | 24 | **0** (certain) | **24** |
| **Total** | **272** | **44** | **228** |

**B0, the always-Normal floor, on the eligible set.** For eligible counts
`(190, 14, 24)` with total 228, an always-Normal model has Normal precision
190/228 = 0.83333 and recall 1.0, so
`macro F1 = (2 × 0.83333 / (1 + 0.83333)) / 3 = ` **≈0.3030**. A single value,
not a range.

**MANDATORY MEASUREMENT (AC-3b) — still required.** Before any model is fitted
the implementer MUST compute and print, at `V_MIN = 1.28` exactly: the excluded
count, its per-class breakdown, the eligible per-class counts, and the realised
always-Normal macro F1 on the eligible set. **Those realised numbers supersede
every figure in this plan.** This measurement is required even though the
expected values are stated above, because a plan that asserts a count the
implementer has not reproduced is exactly what this project does not do. The
implementer MUST additionally assert that the excluded set contains **zero
Side I and zero Side II files** and **stop and return to the planner** if that
assertion fails — it would contradict Gate 1's `[0,2)` histogram and would mean
the speed derivation had changed.

**What is certain regardless, and it is the thing that matters.** The `[0, 2)`
bin contains **zero fault files**. Therefore every file excluded at any threshold
up to 2.0 m/s is Normal, so:

- **Side I remains 14. Side II remains 24. Under any `V_MIN ≤ 2.0`.**
- The per-fold minority arithmetic of §4.2.1 — **2 or 3 Side I, 4 or 5 Side II** —
  is **completely unaffected** by this amendment. That is the only fold statistic
  this plan binds to, and it does not move.

**Relationship to the user's rulings, for traceability.** The user fixed the
exclusion of the 38 zero-speed files (→ 196/14/24 = 234). I extended it to the
near-zero files, which at the chosen `V_MIN = 1.0` gave 190/14/24 = 228; the user
accepted that extension and its 4.45 cm basis. Amendment A now *derives* the
threshold from that same basis, moving it to 1.28 m/s while the eligible total
**stays at 228**. The physical argument is unchanged throughout; only the
precision of the constant has improved.

**Reasoning for excluding the near-zero files, retained from the pre-amendment
text because the derivation formalises it rather than replacing it:**

1. **The spatial-period argument, now the formal basis.** At 0.0445 m/s a train
   covers 4.45 cm in the whole 1-second file — less than one period of the
   longest wavelength band, and less than one period of most of the band set.
   There is no corrugation *cycle* in the recording to detect. §5.3a.2a turns
   exactly this observation into the threshold.
2. **The wavelength bands are numerically unusable at these speeds.** At
   0.0445 m/s the five bands span 0.07 Hz to 2.23 Hz, every one of them narrower
   than the 4.883 Hz Welch resolution (objection §2). All five would be NaN — the
   primary feature family entirely absent, not degraded.
3. **Mechanism 3 is numerically catastrophic.** Dividing by 0.0445 multiplies by
   22.5; by `v²`, by 505. Those files would dominate every scale-dependent
   feature's distribution and would drive the in-fold scaler.
4. **A threshold of exactly zero is arbitrary in the wrong direction.** `v = 0` is
   not physically special relative to `v = 0.0445`; what is special is "too slow
   for the phenomenon to register", which is what the derivation now expresses.
5. **It costs only majority-class data.** Every excluded file is Normal, certain
   from the `[0,2)` histogram. No minority example is lost.

**A note on what this exclusion is NOT justified by.** Excluding 44 slow Normal
files also happens to weaken the speed confound of §7.13, since the excluded
files sit at the extreme low end of the Normal speed distribution. That effect
is a side effect, not a motivation, and must not be presented as one. The
exclusion is justified on the physics in point 1; if the physics argument were
rejected, the exclusion would not be defensible on the confound-reduction
grounds alone, because removing inconvenient data to improve a metric is exactly
what this plan otherwise forbids.

#### 5.3a.4 INFERENCE — the low-speed rule. Every Test file gets a prediction.

**Exclusion from training is NOT exclusion from inference.** §3.2 requires 68
rows and that requirement is absolute. `predict()` must emit a label for every
input file whatever its speed.

**THE RULE, deterministic and stated in advance:**

> For any input file whose derived speed satisfies `v < V_MIN` (1.0 m/s),
> `predict()` returns **`Normal`** without invoking the learned model.

Implementation specification:

- The rule is evaluated **inside `predict()`**, on the per-file derived speed,
  before the feature vector is passed to the fitted pipeline. It is a branch on a
  computed quantity, not a post-hoc edit of the output frame.
- It is deterministic: the same file always produces `Normal`.
- Every firing is **logged to stderr** with the file name and the derived speed,
  and the **count of firings is reported** (AC-30). A silent firing is a defect.
- The count is reported separately from, and in addition to, the §5.8 failure
  fallback count. The two are different mechanisms and must not be conflated: the
  §5.8 fallback means "something went wrong", this rule means "the input is
  out-of-domain and the plan says what to do".

**Why `Normal` and not something else. The reasoning, since this is an
assumption:**

1. **It is what the training evidence supports.** Every one of the 44 low-speed
   Train files is labelled Normal — 44 out of 44, no fault file anywhere near the
   threshold. If the Test set was produced the same way, `Normal` is the correct
   label for a low-speed file. This is the strongest available argument and it is
   an empirical one about *this dataset*, not a claim about rails in general.
2. **It is what the physics supports under the standstill reading.** A stationary
   train's recording contains no corrugation excitation, so there is no evidence
   of a fault in it. Asserting a fault from a signal that cannot contain fault
   evidence would be unfounded.
3. **The alternative — running the model anyway — is worse.** The model would be
   fed a feature vector whose entire wavelength-band family is NaN and whose
   speed-normalised amplitudes are either undefined or astronomically scaled. The
   imputer would fill the NaNs with training medians, so the prediction would be
   driven almost entirely by imputed values, i.e. by the training-set median
   file, not by this file. That is a prediction with no information content, and
   it would depend on which fold's medians were fitted. A stated rule is more
   honest than an arbitrary one.
4. **The macro-F1 cost of being wrong is bounded and small.** If a low-speed Test
   file is truly a fault, this rule contributes one FN to that fault class and
   one FP to Normal. Given 68 Test files and a fault class likely holding ~4-7
   files (§7.9), that is real but bounded. Against it: predicting a fault on a
   signal with no fault evidence risks an FP on a class with maybe 4 true members,
   which costs precision on the metric's most fragile component.
5. **It is NOT chosen because Normal is the majority class.** Defaulting to the
   majority is exactly the collapse this subsystem is built to avoid (§5.7). The
   rule is justified by points 1-3, and it applies to a narrow, physically
   defined subset of inputs identified by a threshold fixed before any Test file
   was examined. If the rule applied broadly it would be majority-class collapse
   in disguise; applied to `v < 1.0 m/s` it is a domain boundary.

**This rule is an ASSUMPTION and is recorded as one in §7.10.** It is fixed
before Gate 1b runs and Gate 1b may not change it (§4.8.3).

**Applies identically at inference on Train.** If any diagnostic runs `predict()`
over Train files, the same rule fires on the same 44 files. No Train/Test branch
exists anywhere.

### 5.3b Speed as a feature — the NORMALISE / PREDICT line

**NEW in iteration 2. This resolves implementer Objection B. This is the section
the reviewer checks hardest.**

#### 5.3b.1 The finding, restated as fact

Gate 1 measured derived speed by class over all 272 Train files:

| Class | n | min m/s | p25 | median | p75 | max m/s |
|---|---|---|---|---|---|---|
| Normal | 234 | 0.000 | 2.663 | 8.130 | 12.896 | 19.479 |
| Side I | 14 | **9.702** | 12.725 | 12.959 | 13.567 | 18.603 |
| Side II | 24 | **11.690** | 13.278 | 13.997 | 15.881 | 18.514 |

**No fault file falls below 9.7023 m/s. 133 of 234 Normal files do.**
Mann-Whitney Side I vs Normal `p = 2.514e-04`, AUC 0.7766; Side II vs Normal
`p = 9.080e-08`, AUC 0.8230. A two-threshold rule on speed **alone** reaches
macro F1 **0.5089** (thresholds 12.25 and 13.00 m/s), fitted on all 272 labels
and therefore an optimistic upper bound, against the always-Normal floor 0.3083.

A scalar with nothing to do with corrugation clears the plan's own floor baseline
by ~0.20 macro F1. With zero leaderboard uploads, nothing external would ever
catch a model that learned this and nothing else.

#### 5.3b.2 THE RULE — the line is normalise versus predict

**Speed may be used to decide WHERE IN THE SIGNAL TO LOOK. Speed may not be used
as EVIDENCE OF THE LABEL.**

Stated mechanically, so a violation is visible by inspecting the feature list:

> **No feature whose value is a monotone function of `v` alone may enter the
> model matrix.** Equivalently: if a feature's value can be computed from column
> 1 without reading any of columns 2-129, it is forbidden. If a feature requires
> the vibration/shock channels and uses `v` only to select or rescale what is
> read from them, it is permitted.

This is a **mechanically checkable** criterion, deliberately. The test is not
"does this feel like a shortcut"; the test is "can this number be produced from
the speed channel alone". `v`, `v` in km/h, `T`, `T`'s windowed std, `T`'s
windowed min/max, any binning or transform of any of these — all computable from
column 1 alone, all forbidden. A wavelength-band power requires the PSD of a
vibration channel and therefore is not.

**Mandated enforcement, so this is checkable and not merely asserted:**

- The feature extractor emits the speed block into a **separate, clearly named
  set of columns prefixed `meta_`** (e.g. `meta_v_mps`, `meta_T`,
  `meta_v_kmh`, `meta_T_win_std`). These are written to the feature cache
  because they are needed for the exclusion rule (§5.3a), the restricted
  evaluation (§4.7), the inference rule (§5.3a.4) and the diagnostics — **but
  they are dropped before the matrix reaches the pipeline.**
- The model matrix is constructed by an explicit selection that **excludes every
  column whose name starts with `meta_`**, and the count of dropped columns is
  printed.
- The implementer prints the **full list of `meta_`-prefixed columns** and
  asserts that none of them appears in the fitted estimator's input feature
  names. **AC-31.**
- The reviewer checks that no non-`meta_` feature is computable from column 1
  alone.

#### 5.3b.2a THE GENERAL RULE — a dimensional criterion, not a list

**Amendment B, iteration 2.** §5.3b.2's column-1 test catches features computable
from the speed channel alone. It does **not** catch a feature computed from the
vibration channels whose *units* encode rate, and the spectral centroid in Hz
nearly slipped through for exactly that reason. A struck-feature list enumerates
instances; it does not state the rule. The rule is stated here, and it governs.

**THE RULE:**

> **Any feature whose value is a position, width, or rate along the signal's
> time or frequency axis is a disguised speed feature. It must be expressed in
> METRES (as a wavelength) or as a DIMENSIONLESS RATIO before it may enter the
> model matrix. A feature carrying a per-second term derived from that axis —
> anything in Hz, s⁻¹, or a count-per-second — is rejected.**

**The necessary distinction, stated because a naive reading over-strikes.** A
vibration channel is measured in m/s² (Info Kit §2.1 line 82), so *amplitude*
features inevitably carry per-second terms: RMS is m·s⁻², band power is
m²·s⁻⁴. **Those per-second terms come from the amplitude axis, not the frequency
axis, and they do not encode train speed.** They are governed separately by
§5.3b.3's raw-versus-normalised adjudication. The rule above applies to the
**time/frequency axis only** — to *where* in the spectrum something sits or *how
fast* something recurs, never to *how hard* the axle box is shaking.

Stated as a two-question test the implementer applies to every feature:

1. **Is this quantity a location, width, or rate on the time/frequency axis?**
   If no, the rule does not apply; §5.3b.3 governs instead.
2. **If yes, is it expressed in metres or as a dimensionless ratio?** If no, it
   is rejected.

**Application to every feature currently in the plan — a full re-audit, done
against the general rule rather than against the old list:**

| Feature | Axis | Units as emitted | Verdict under the general rule |
|---|---|---|---|
| Spectral centroid **in Hz** | frequency, a *location* | s⁻¹ | **STRUCK.** Rate on the frequency axis. |
| Spectral centroid **in metres** (`v / f_centroid`, dominant wavelength) | frequency, a *location* | m | **SURVIVES.** Expressed as a wavelength. Matches Info Kit §1.1:43-44. |
| Wavelength-band powers (5 bands) | amplitude *value*; band *edges* are on the frequency axis | m²·s⁻⁴, band labelled by λ in metres | **SURVIVES.** The emitted value is an amplitude, not an axis position. The edges are derived per-file as `v/λ` precisely so the band is fixed in **metres**; the feature name carries the λ in metres, never the Hz edges. |
| Total band power, RMS, peak | amplitude | m²·s⁻⁴, m·s⁻² | **SURVIVES** (in `/v`, `/v²` form per §5.3b.3). Amplitude axis — rule does not apply. |
| `X/v`, `X/v²` | amplitude | m·s⁻³, dimensionless-in-`v` | **SURVIVES.** Confirmed dimensionally acceptable: the general rule does not contradict §5.3b.3's adjudication, because these are amplitude-axis quantities. |
| Kurtosis, skewness | amplitude distribution shape | dimensionless | **SURVIVES.** |
| Crest factor (peak/RMS) | amplitude ratio | dimensionless | **SURVIVES.** |
| Spectral entropy | frequency axis, but a normalised Shannon entropy of the PSD | dimensionless | **SURVIVES.** Frequency-axis quantity, but dimensionless by construction, which the rule explicitly permits. |
| Across-car dispersion (std, max-minus-median of a per-channel feature) | inherits the underlying feature's axis | inherits | **INHERITS** the verdict of whatever feature it aggregates. Dispersion of a Hz-valued quantity would be struck; of a metre- or ratio-valued one, permitted. |
| `v`, `v_kmh`, `T`, windowed `T` statistics | time axis, a *rate* | m·s⁻¹, s⁻¹ | **STRUCK** — already struck by §5.3b.2, and struck again independently here. |

**Outcome of the re-audit: no feature previously permitted is struck by the
general rule.** The rule reproduces every earlier per-case adjudication,
including the spectral-centroid decision that motivated it, and confirms
`X/v`/`X/v²`. Its value is **prospective**: it governs features not yet written.

**ENFORCEMENT — mechanical, and it fails loudly at extraction time.**

A membership check against a list cannot catch the next case. The enforcement is
therefore a **naming contract checked at extraction**, not at review:

- **Every feature column name MUST end with a unit suffix drawn from a closed
  vocabulary**, declared once as a module constant:

  | Suffix | Meaning | Admissible to the model matrix? |
  |---|---|---|
  | `_m` | metres (a wavelength or length) | **yes** |
  | `_ratio` | dimensionless ratio or normalised index | **yes** |
  | `_amp` | amplitude-axis quantity, raw | **no** — struck by §5.3b.3 |
  | `_amppv` | amplitude normalised by `v` | **yes** |
  | `_amppv2` | amplitude normalised by `v²` | **yes** |
  | `_hz` | frequency-axis rate | **no** — rejected by this rule |
  | `_persec` | any other per-second rate | **no** — rejected by this rule |

- **The extractor validates every emitted column name against this vocabulary at
  the moment the feature matrix is assembled**, before anything is written to the
  cache. A column whose name carries **no recognised suffix is a HARD ERROR**
  that aborts extraction, naming the offending column. It is never a warning and
  never a silent pass. This is the property that makes the rule future-proof: a
  new feature added without thinking about units **cannot reach the cache**, let
  alone the model.
- **Columns suffixed `_hz`, `_persec` or `_amp` are rejected from the model
  matrix** by the same pipeline step that drops `meta_` columns (§5.5 step 1).
  They may exist in the cache for diagnostics; they may not be fitted on.
- The implementer **prints the full suffix inventory** — every distinct suffix
  present and the column count under each — plus the count admitted and the count
  rejected, and asserts that the fitted estimator's input feature names contain
  **zero** columns suffixed `_hz`, `_persec`, `_amp`, or prefixed `meta_`.
  **AC-36.**
- **B2 remains the sole declared exception** (§5.7a), reading `meta_v_mps` only,
  with its bypass named explicitly.

**Why a naming contract rather than a units library.** A full dimensional-analysis
library would be stronger but is a dependency this plan does not add for one
subsystem. The suffix vocabulary is checkable from printed output alone, which is
what the reviewer needs, and the hard-error-on-unknown-suffix behaviour gives the
loud failure the rule requires. Recorded as a deliberate trade-off.

#### 5.3b.3 The subtle case — speed-normalised amplitude (mechanism 3)

Iteration 1 mandated emitting every energy/RMS feature **twice**: raw, and
divided by `v` (and `v²`). Iteration 2 must decide whether `X/v` is "normalise"
or "predict in disguise". **It is adjudicated as follows, and the two halves of
mechanism 3 are treated differently.**

**The raw absolute-level features `X` are STRUCK from the model matrix.**

Reasoning. Vibration amplitude rises with speed independently of corrugation —
the Info Kit says so directly at §1.1 lines 45-46 ("affected by train speed"). So
a raw amplitude feature `X` is, to first order, `X ≈ g(condition) · h(v)`. In a
dataset where `v` separates the classes, `h(v)` alone carries the shortcut
signal. A tree ensemble splitting on raw RMS is splitting on speed by proxy, and
the mechanical rule of §5.3b.2 does not catch it, because `X` does require the
vibration channels. **This is precisely the loophole the reviewer is expected to
probe, and the plan closes it by removing the raw absolute-level block rather
than by hoping the model ignores it.**

**The speed-normalised features `X/v` and `X/v²` are PERMITTED — they are
"normalise", not "predict". Reasoning:**

1. `X/v` is **not** a monotone function of `v` alone. It requires the vibration
   channels. Holding the channels fixed and increasing `v` *decreases* it;
   holding `v` fixed and increasing the vibration energy increases it. It carries
   information from columns 2-129 that no function of column 1 can reproduce.
2. Its *purpose* is to remove the speed dependence, not to encode it. If the
   model `X ≈ g(condition) · v` holds, then `X/v ≈ g(condition)` and the
   normalisation has stripped the confound out. That is the definition of
   normalising.
3. The direction of the residual risk is **against** the shortcut, not for it.
   Fault files are *fast*. If normalisation is imperfect and `X/v` retains some
   speed dependence, it retains it with the **opposite sign** to the raw feature:
   `X/v` is *lower* for fast files if the normalisation overcorrects. A model
   exploiting that would have to learn "low normalised amplitude ⇒ fault", which
   is the reverse of the physically expected direction and would be visible as an
   implausible coefficient sign on M1. **Mandated: the implementer reports the
   sign of M1's coefficients on the top-magnitude normalised-amplitude features,
   and a systematic "lower ⇒ fault" pattern is a finding for the planner
   (AC-32).**
4. Emitting the raw variant *as well* would defeat the whole exercise. A tree can
   recover `v` to good accuracy from the pair `(X, X/v)` by division, so
   supplying both hands the model the confound in a form no naming convention
   catches. **This is the decisive argument for striking raw rather than keeping
   both**, and it is why iteration 1's "emit both and let the model choose" is
   revoked: "let the model choose" is exactly what we cannot afford when one of
   the choices is a shortcut we cannot detect downstream.

**The honest caveat, which must be in the write-up.** `X/v` is a *first-order*
correction under an *assumed* multiplicative model. The Info Kit gives no
functional form for the speed dependence of axle-box vibration; the plan does not
know whether it is linear, quadratic, or neither. This is why **both** `X/v` and
`X/v²` are emitted: they bracket the plausible exponents. If the true dependence
is `v^1.5`, neither fully removes it and a residue remains. **The restricted
evaluation of §4.7 is the check on that residue**, which is why §4.7 is mandatory
alongside this section and why the restricted number is the headline. Neither
mechanism alone is sufficient; together they are the best available and the plan
claims no more than that. Recorded as an assumption in §7.12.

**Numerical guard.** `X/v` and `X/v²` are computed only for files with
`v >= V_MIN = 1.28 m/s`, which by §5.3a is every file that enters training and
every file that reaches the pipeline at inference. Division is therefore never by
a value below 1.0, the multiplier is never above 1.0, and no `eps` guard is
needed on this path. **The iteration-1 statement "`eps` guards division at
implausibly low `v`" is revoked** — the implementer correctly objected that an
`eps` guard converts a divide-by-zero into an arbitrary huge number. The
exclusion rule replaces the guard. `eps` retains its role only in the log-ratio
of §5.2 step 4, where it guards a ratio of two non-negative powers.

#### 5.3b.4 Summary — what survives, what is struck

| §5.3 mechanism | Status | Reason |
|---|---|---|
| 1. Wavelength-domain band edges `[v/λ_hi, v/λ_lo]` | **SURVIVES**, mandatory, primary | `v` selects where to look. The band power itself needs the PSD of a vibration channel. Canonical "normalise". |
| 2. `v` and column-1 derivatives as model features | **STRUCK entirely** | Computable from column 1 alone. Canonical "predict". This is the shortcut, handed to the model directly. |
| 3a. Raw absolute-level amplitude/energy features | **STRUCK from the model matrix** | `X ≈ g(condition)·h(v)`; carries the shortcut by proxy. Also, `(X, X/v)` together let a tree reconstruct `v`. |
| 3b. Speed-normalised `X/v`, `X/v²` | **SURVIVES** | Not a function of `v` alone; purpose is confound removal; residual risk points the wrong way for a shortcut and is detectable via AC-32. |
| 4. Side I vs Side II contrast and log-ratio | **SURVIVES and is PROMOTED to the primary family** | Compares two sides of the same file at the same speed; a common multiplicative speed factor cancels exactly under the log ratio. Structurally confound-free. |
| **The general dimensional rule (§5.3b.2a)** | **GOVERNS ALL OF THE ABOVE** | The mechanism-by-mechanism verdicts in this table are instances; §5.3b.2a states the rule they follow and governs any feature added later. Where this table and §5.3b.2a could be read to differ, **§5.3b.2a governs.** |

**What the feature matrix therefore contains** (revising §5.4's arithmetic):

- The **contrast block** — `A_I − A_II` and `log((A_I+eps)/(A_II+eps))` — in
  full. Primary family.
- The **speed-normalised per-side blocks** `A_I/v`-type and `A_II/v`-type
  aggregates, and their symmetric mean.
- The **wavelength-band power features**, which are amplitudes and so appear in
  normalised form and in contrast form, not in raw form.
- The **shape/dimensionless features** — kurtosis, skewness, crest factor,
  spectral centroid, spectral entropy — which are **scale-free by construction**
  and therefore are **not** absolute-level features. These are PERMITTED in raw
  form, subject to the dimensional rule. **The full per-feature adjudication —
  including the spectral centroid, which MUST be emitted in metres
  (`v / f_centroid`, a dominant wavelength) and never in Hz — lives in
  §5.3b.2a's audit table.** It is stated there and only there, so the
  adjudication has exactly one home.
- **No `meta_` column.**

**Consequence for the feature count.** Striking the raw absolute-level block
removes roughly the raw half of the scale-dependent per-channel values across
all 24 aggregate blocks, plus the ~6 speed features. The expected total falls
from ~1,120 to roughly **700-900**. The implementer reports the exact number;
**AC-4's accepted range is revised to 500-1,200.** This is a welcome side effect
at `p >> n`, not the reason for the change.

#### 5.3b.5 What this does NOT fix, stated plainly

- It does not remove a speed confound operating through signal *shape* rather
  than amplitude. If corrugation excitation lands in a different part of the
  spectrum at different speeds in a way the wavelength normalisation does not
  fully capture, a residue remains.
- It does not make the 133 slow Normal files' labels reliable. If corrugation is
  simply undetectable below some speed, those labels may be wrong (§7.13), and no
  feature rule can repair a wrong label.
- It does not tell us which way the causation runs — acquisition artefact or
  physical detectability threshold. That is an organiser question (§7.13) and the
  plan does not speculate.
- **Therefore the restricted evaluation of §4.7 and the mandatory B2 floor of
  §5.7a are not optional companions to this section. All three are required
  together**, because each catches something the others miss.

### 5.4 Memory strategy — streaming extraction and the feature cache

272 files × ~17 MB ≈ **4.5 GB** for Train; 68 × 17 MB ≈ **1.1 GB** for Test.
Neither may be held in memory, and the raw corpus must never be re-read during
model iteration.

**Note on which files are extracted.** Features are extracted for **all 272**
Train files, not only the eligible ones. The exclusion of §5.3a happens at the
*modelling* stage, by filtering the cached matrix on `meta_v_mps >= V_MIN`, not
at the extraction stage. Reasons: the excluded files' `meta_` values are needed
to document the exclusion; a future planner decision to change `V_MIN` must not
require a 4.5 GB re-read; and the diagnostics of §4.4 are more informative over
the full index range. The excluded rows are present in `features_train.parquet`
and carry a boolean column **`meta_eligible`** so the filter is explicit and
auditable rather than recomputed at each call site. Wavelength-band features for
excluded files will be NaN, which is expected and harmless because those rows
never reach a fit.

**Mandated design: one-pass streaming extraction to a disk cache.**

- **Process exactly one file at a time.** Open it, read it, compute its feature
  vector, discard the raw array, move on. Peak memory is one file's array
  (10,000 × 129 float64 ≈ 10.3 MB, or ~5.2 MB as float32) plus the accumulating
  feature list. Never build a list of raw DataFrames. Never `pd.concat` raw data.
- **Read with `pandas.read_csv(path, dtype=np.float32)`** (or an equivalent
  explicit dtype), and **honour the header row**: every file is 10,001 lines =
  1 header + 10,000 data rows. Do NOT pass `header=None`; doing so would coerce
  the header into a row of `NaN` and shift everything. This is a concrete,
  checkable failure mode (AC-2). Gate 1 confirmed the 1-header + 10,000-data-row
  structure on sampled Train and Test files.
- **Column access is positional, not by name.** Column 1 is speed, columns 2-129
  are the channels, per Info Kit §2.1 lines 74-81. The implementer MUST index by
  position (`iloc[:, 0]` and `iloc[:, 1:129]`) and MUST assert
  `df.shape == (10000, 129)` for every file. A file failing that assertion is a
  hard error, reported by filename, not skipped. Gate 1 verified the header
  strings match §5.2's arithmetic for all 128 channels (§7.6 RESOLVED), so a
  name-based assertion MAY additionally be added as a cheap consistency check;
  positional access remains the primary path.
- **Deterministic file ordering.** Files are enumerated in natural numeric order
  (`Train1..Train272`, `Test1..Test68`) via a numeric sort of the parsed index,
  never via `sorted()` on the filename, which yields `Train1, Train10, Train100`.
  Deterministic ordering is what makes the cache reproducible and the adjacency
  diagnostic (§4.4) meaningful.
- **Parallelism** via `joblib` over files is permitted and encouraged
  (`n_jobs=-1`), since each file is independent. It must not change the output:
  results are reassembled into the deterministic order above before writing.

**Cache format and path.**

- Format: **Parquet** (`.parquet`), via pandas. Chosen over CSV because it
  preserves float dtypes exactly and stores column names; chosen over `.npy`
  because named columns are what makes the feature matrix auditable and what
  lets a reviewer read a feature importance table.
- Paths, all **inside the repo** and all **gitignored**:
  - `artifacts/rail/features_train.parquet`
  - `artifacts/rail/features_test.parquet`
  - `artifacts/rail/feature_meta.json` — the feature-extractor version string,
    the wavelength band constants, the Welch parameters, the `eps` value,
    **`V_MIN`, `V_CUT`**, and the extraction timestamp.
- `artifacts/` must be added to `.gitignore` if not already present. The cache is
  a derived artefact; it is never committed.
- **Cache invalidation**: the extractor writes a version string into
  `feature_meta.json`. Any change to feature code increments it. A cache whose
  version does not match the current extractor's is **rebuilt**, not used. Silent
  reuse of a stale cache would make every downstream number wrong in a way
  nothing else catches. **Iteration 2's feature changes (§5.3b) REQUIRE a version
  bump; any cache built under iteration 1's feature set is invalid.**

**Expected feature matrix size.**

Per channel, the per-channel feature function `f` produces: 5 wavelength-band
powers + total band power + RMS + peak (all amplitude-like, emitted **only** in
`/v` and `/v²` normalised form per §5.3b.3, not raw) + kurtosis + skewness +
crest factor + dominant-wavelength centroid in metres + spectral entropy (all
scale-free, emitted raw). That is ~8 amplitude-like values × 2 normalisations =
16, plus ~5 scale-free = **`d ≈ 21`** per channel, with the raw amplitude block
struck.

Aggregation: 2 sides × 2 kinds (vibration, shock) × 6 aggregators (median, p90,
max, IQR, std-across-cars, max-minus-median) = 24 aggregate blocks of `d`.
Contrast block, symmetric-mean block, as in §5.2.

**Expected total: roughly 700-900 features.** The implementer reports the exact
number; AC-4 checks it against the range **500-1,200**, not against a precise
target. If it falls outside, report the number and the discrepancy rather than
silently proceeding.

**Every emitted column carries a unit suffix from §5.3b.2a's closed vocabulary**
(`_m`, `_ratio`, `_amp`, `_amppv`, `_amppv2`, `_hz`, `_persec`), or a `meta_`
prefix. **A column with no recognised suffix is a HARD ERROR that aborts
extraction**, naming the offender — it is never a warning and never written to
the cache. `feature_meta.json` records `N` and the derived `V_MIN` alongside the
other constants, so the cache carries the derivation that produced it.

Sizes:

| Matrix | Rows | Cols | float32 bytes | On disk (Parquet, compressed) |
|---|---|---|---|---|
| Train | 272 | ~800 + `meta_` + label | 272 × 800 × 4 ≈ **0.87 MB** | ~0.4-1.2 MB |
| Test | 68 | ~800 + `meta_` + id | 68 × 800 × 4 ≈ **0.22 MB** | ~0.2-0.5 MB |

That is the whole point: **4.5 GB of raw CSV collapses to under 1 MB of
features.** Once the cache exists, 50 CV fits touch under 1 MB, and model
iteration never opens a raw file again.

**Note on the feature-to-sample ratio.** ~800 features against 228 eligible
samples, with 14 minority examples, is `p >> n`. This is a deliberate, declared
trade-off: a wide, cheap, physically-motivated feature set combined with
**aggressive in-fold regularisation and in-fold feature selection** (§5.5),
rather than a hand-picked narrow set chosen by peeking at the labels. Choosing
features by looking at all labels before splitting would be leakage and is
forbidden. Any feature selection happens **inside the pipeline, inside the
fold**. If the reviewer judges ~800 too wide, the remedy is a reduced `d` or
fewer aggregators specified by the planner, not ad-hoc pruning by the
implementer.

**Test-side behaviour.** The Test corpus goes through the **identical**
extraction function, same code path, same constants, same version string — no
Test-specific branch anywhere. Test extraction happens **once**, and produces
`features_test.parquet`. Critically: no transformer is ever *fitted* on Test.
Scaling, imputation and selection are fitted on the full eligible Train set
(after CV has settled the configuration) and only *applied* to Test.
`predict(input_path)` for a single file runs the same per-file extractor and the
same fitted pipeline, after the §5.3a.4 low-speed branch.

### 5.5 Model family and the comparison set

All models are wrapped in an `sklearn.pipeline.Pipeline` so that every fitted
step is fitted inside the fold. Mandatory pipeline shape:

1. Drop every `meta_`-prefixed column (§5.3b.2). This is the first step and it is
   not optional.
2. `SimpleImputer(strategy="median")` — handles `NaN` band features from §5.3.
3. Variance threshold / constant-column drop.
4. `StandardScaler` (or `RobustScaler`).
5. Optional in-fold univariate selection (`SelectKBest`) for the linear model.
6. The classifier.

**The comparison set is exactly these six configurations. No more.** Declaring
the search space up front is what keeps §4.6's model-selection leakage bounded.

| # | Model | Imbalance handling | Why it is here |
|---|---|---|---|
| B0 | `DummyClassifier(strategy="most_frequent")` | none | The always-Normal baseline, ≈0.3030 on the eligible set. Must be beaten. |
| B1 | Physics rule baseline (§5.7) | none | A non-learned, explainable baseline the learned model must beat. |
| **B2** | **Speed-only two-threshold rule (§5.7a), fitted in-fold** | none | **MANDATORY FLOOR. The confound baseline.** |
| M1 | `LogisticRegression(penalty="l2", class_weight="balanced", multi_class="multinomial", max_iter=5000)` with in-fold `SelectKBest` | `class_weight="balanced"` | Strong regulariser, `p >> n`-appropriate, and its coefficients on contrast features are directly interpretable as "Side I minus Side II". |
| M2 | `RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=2)` | `class_weight="balanced_subsample"` | Non-linear, robust to the wide feature set, gives feature importances. |
| M3 | `ExtraTreesClassifier(n_estimators=500, class_weight="balanced", min_samples_leaf=2)` | `class_weight="balanced"` | More variance reduction than RF at these sample sizes; cheap to add given M2's plumbing. |

**B2 renumbering, to avoid two different B2s.** Iteration 1's B2 was
`DummyClassifier(strategy="stratified")`, reported "for context only, not a bar".
**That estimator is REMOVED from the comparison set entirely.** It was never a
bar, it consumed a configuration slot, and keeping it under a different letter
would invite confusion with the mandatory floor. **From iteration 2 onward, "B2"
means the speed-only rule and nothing else.** There is exactly one B2. If a
chance-level reference is wanted in the write-up, the always-Normal B0 serves
that purpose and is already reported.

All of M1-M3 take `random_state` from the single plan seed.

**Dependency decision: xgboost and lightgbm are NOT installed, and this plan does
NOT add them.** Justification: gradient-boosted trees are strongest when there
are enough minority examples to fit many sequential residual corrections, and
with 14 Side I files they overfit readily; their advantage over a well-regularised
RF/ExtraTrees at n = 228 is not established and would have to be demonstrated,
which costs model-selection budget we are deliberately conserving. Adding a
dependency also adds an install-reproducibility risk for zero demonstrated gain.
If a later iteration shows M1-M3 all failing on Side I specifically, adding
LightGBM is a planner decision, not an implementer decision.

**Hyperparameter tuning is capped.** Beyond the values fixed above, the only
tuning permitted is `LogisticRegression`'s `C` over the fixed grid
`{0.01, 0.1, 1.0, 10.0}` and `SelectKBest`'s `k` over `{50, 100, 200, "all"}`,
and it MUST be done as a **nested** search (`GridSearchCV` as the estimator
*inside* the outer `RepeatedStratifiedKFold`), with an inner
`StratifiedKFold(n_splits=3)` — inner k=3 because 80% of 14 Side I is 11, and
k=3 leaves ~3-4 per inner fold, whereas inner k=5 would leave ~2. Tuning on the
outer folds and reporting the outer score is leakage and is forbidden. If nested
search proves too slow, the fallback is **fixed hyperparameters**, not
non-nested tuning.

**The inner search optimises the FULL-set macro F1, not the restricted one.**
Reason: the inner folds are small (roughly one third of 80% of 228 ≈ 61 files),
and restricting them further would leave too few files to select on. The
restricted metric is an *evaluation* instrument, not a selection objective, at
the inner level. At the **outer** level it is the selection objective — see
below. This asymmetry is deliberate and must be stated in the report.

**Final model selection rule, stated in advance so it cannot be rationalised
afterwards:** the winner is the configuration with the highest **mean RESTRICTED
macro F1 across the 50 outer folds** (§4.7), with ties (within 0.01) broken by
**higher mean restricted Side I F1**, then by **higher mean full-set macro F1**,
and further ties broken in favour of the **simpler** model (M1 over M2 over M3).
This rule is fixed now. Iteration 1's rule selected on the full-set mean; that is
superseded, because selecting on the full set would select for the confound.

### 5.6 Decision rule and tie handling

Predictions are `argmax` over the classifier's predicted class probabilities,
with `RAIL_CLASSES` order fixing the tie-break: on an exact probability tie the
lowest index wins, i.e. `Normal` beats `Side I` beats `Side II`. Exact ties are
essentially impossible with float probabilities but the behaviour is pinned so
it is deterministic rather than dependent on `numpy` internals.

**No post-hoc threshold tuning on the validation folds.** Shifting a decision
threshold to maximise macro F1 on the same folds used to report macro F1 is
leakage. If a later iteration wants threshold optimisation, it must be fitted
inside the inner CV, and that is a planner decision.

**The §5.3a.4 low-speed rule takes precedence over the argmax**, and fires before
the pipeline is invoked. It is not a tie-break and not a post-hoc override; it is
a domain gate at the front of `predict()`.

### 5.7 Class imbalance handling and the baselines

**The concrete failure mode.** With 190/14/24 eligible, a model predicting
`Normal` for everything achieves ~83.3% accuracy and macro F1 ≈ **0.3030**
(§4.2.0 arithmetic; ≈0.3039 under a zero-only exclusion; ≈0.3083 on the full 272,
measured at Gate 1; the Info Kit's own looser figure is ~0.33 at §4 line 148).
Optimising accuracy — or using an unweighted loss, which is the same thing by
another name — lands exactly there. **Accuracy is never reported as a headline
number in this subsystem.** It may appear in a diagnostics table, clearly
subordinate.

**Why the chosen handling suits macro F1 specifically.** Macro F1 gives Side I —
14 files, ~6% of the eligible data — the same one-third weight as Normal's 190.
The loss the model minimises must reflect that or the model will optimise
something other than the thing being scored. Concretely:

- **`class_weight="balanced"` is mandated on every learned model.** It weights
  each class inversely to its frequency, so one Side I file counts roughly 190/14
  ≈ 13.6× a Normal file in the loss. That is the closest standard, in-fold,
  leakage-free approximation to "each class counts equally" that the metric
  demands. It requires no synthetic data and no resampling, and it is computed
  **inside the fold** from the training portion's counts only (sklearn does this
  automatically when `class_weight="balanced"` is set on the estimator inside a
  `Pipeline`) — computing weights once from all eligible labels and hardcoding
  them would be a mild leak and is forbidden.
- **`RandomForest` uses `class_weight="balanced_subsample"`**, which recomputes
  the balance per bootstrap sample, which is the correct analogue for a bagged
  ensemble.
- **No SMOTE, no random oversampling, no synthetic minority generation.**
  Reason: with 14 real Side I files in a ~800-dimensional feature space,
  synthetic interpolation between them generates points in a space where 14
  samples define essentially nothing, and it inflates CV scores by making
  minority validation points look like interpolations of training points. If
  oversampling is used at all it must be inside the fold via an
  `imbalanced-learn` pipeline — but `imbalanced-learn` is not installed, and this
  plan does not add it. That is a deliberate decision, not an omission.
- **Stratification in the splitter (§4.1)** ensures the imbalance is *identical*
  in every fold, so the weighting is stable across folds.

**The three baselines, all of which must be computed and reported, under BOTH
evaluation domains of §4.7:**

- **B0 — always-Normal (`DummyClassifier(strategy="most_frequent")`).** Expected
  full-set macro F1 ≈ **0.3030** on the eligible 190/14/24 set (≈0.3039 if only
  exact zeros are excluded); the implementer prints the realised value. This is
  the absolute floor. A learned model that does not clear it is worthless; a
  learned model that merely matches it has collapsed to the majority class.
- **B1 — the physics rule baseline, non-learned and explainable.** Specification:
  using only the single most physically motivated feature — the Side I vs Side II
  contrast in total corrugation-band power, i.e.
  `c = log((P_I + eps) / (P_II + eps))` where `P_side` is the p90-across-channels
  vibration power summed over the five wavelength bands — predict `Side I` if
  `c > +t`, `Side II` if `c < -t`, else `Normal`. The threshold `t` is fitted
  **inside each training fold** (chosen to maximise macro F1 on the training
  portion only) and applied to the validation portion. It is never fitted on
  validation data. B1 is the bar that proves the ~800-feature learned model is
  earning its complexity: **if M1-M3 cannot beat B1's mean macro F1, the honest
  conclusion is that a one-feature rule is the right model**, and that conclusion
  is reported rather than buried.
- **B2 — the SPEED-ONLY rule. MANDATORY FLOOR. New in iteration 2.** See §5.7a.

#### 5.7a B2 — the speed-only confound baseline, and the margin the model must clear

**What B2 is.** A classifier using **derived speed and nothing else** — no
vibration data at all. Two thresholds `t_lo <= t_hi` partition the speed axis
into three regions mapped to the three classes. Formally: predict `Normal` if
`v < t_lo`; otherwise predict `Side I` or `Side II` according to which of the two
remaining regions `v` falls in, with the region-to-class mapping chosen along
with the thresholds.

**Exact fitting specification — this is what makes the comparison honest.**

Gate 1's 0.5089 figure was fitted on **all 272 labels** and is therefore an
**optimistic upper bound**, not a CV score. It may not be used as the bar. B2
must be computed the same way every other configuration is:

- B2 is implemented as an `sklearn`-compatible estimator whose `fit(X, y)` uses
  **only the `meta_v_mps` column** of `X` and whose `predict(X)` likewise. It is
  the **one and only** configuration permitted to see a `meta_` column; the
  §5.5 drop-`meta_` step is bypassed for B2 alone, and that bypass is explicit
  and named in the code, not incidental.
- `fit` performs an exhaustive search over candidate threshold pairs
  `(t_lo, t_hi)` drawn from the **sorted unique speeds of the training portion
  only**, at their midpoints, together with all valid region-to-class mappings,
  selecting the pair maximising `macro_f1` on **the training portion only**.
- `predict` applies the fitted thresholds to the validation portion.
- B2 therefore runs inside the **identical** 50 outer folds as everything else,
  with thresholds refitted in every fold, and is scored in **both** the full and
  restricted domains.
- **The in-CV B2 number, not 0.509, is the operative floor.** The implementer
  reports both: the in-CV mean (the bar) and Gate 1's all-data 0.5089 (labelled
  explicitly as an optimistic upper bound and as the historical finding that
  motivated this section). **AC-33.**

**Expected behaviour, stated as a prediction so a surprise is visible.** In the
**full** domain, B2 should score in the neighbourhood of 0.45-0.51 — somewhat
below the all-data 0.5089 because the thresholds are fitted out-of-fold, and
somewhat shifted because the eligible set excludes the slowest 44 Normal files,
which were B2's easiest wins. The direction of the net effect is not predicted
here; the implementer reports what it is. In the **restricted** domain
(`v >= 9.70`), B2 should collapse toward B0, because the cut removes exactly the
region where speed separates the classes. **If restricted B2 does NOT collapse —
if it stays well above B0 — that is a finding for the planner**, because it would
mean speed still separates the classes above the cut and the restricted domain is
not the confound-free zone this plan assumes it is. **AC-34.**

**THE MARGIN — what "beat B2 by a real margin" means, concretely.**

A point estimate comparison is not enough at these sample sizes. The rule is
stated on the **paired 50-fold distribution**, because B2 and the candidate model
are evaluated on **the same 50 folds** and the per-fold differences are therefore
paired.

For the selected learned model `M` and baseline `B`, define the per-fold paired
difference `d_i = macroF1_M(fold i) − macroF1_B(fold i)` for `i = 1..50`, computed
in the **restricted** domain. Report `mean(d)`, `std(d)`, and the count of folds
with `d_i > 0`. Then:

> **PASS on B2 requires ALL THREE of:**
> 1. `mean(d) > 0.05` — the mean restricted macro F1 exceeds B2's by more than
>    0.05 absolute;
> 2. `d_i > 0` on **at least 35 of the 50 folds** (70%) — the win is not carried
>    by a handful of folds;
> 3. the **10th percentile** of `d` is `> −0.05` — the model is not catastrophically
>    worse than B2 on its bad folds.
>
> The same three conditions are evaluated and reported against **B0** and **B1**
> as well, in the same table. Against B0 the required `mean(d)` is `> 0.05`; the
> same fold-count and p10 conditions apply.

The 0.05 and 35/50 figures are **my choice** and are stated as such (§7.11). The
reasoning: a per-fold Side I flip moves per-fold macro F1 by ~0.07-0.10 (one of
2-3 files, one third of the macro average), so a mean advantage below 0.05 is
within the resolution of a single file's fate; and a win on 35 of 50 paired folds
is a margin that a coin flip would produce with negligible probability, while
requiring a sweep of all 50 would be unattainable at this quantisation. They are
fixed **before** any model is fitted, which is the property that matters more
than the exact values.

**What happens if the model fails to clear B2.** It is reported as the result.
**A model that does not beat the speed-only rule by the stated margin is not
promoted, and the write-up says plainly that the learned model did not
demonstrably outperform a classifier that looks only at train speed.** That is a
defensible, publishable finding and is preferable to a headline number that a
speed rule could have produced. The plan explicitly refuses to relax the margin
after seeing the numbers; relaxation is a planner decision requiring a new
iteration and a Revision log entry.

**The learned model must beat B0, B1 and B2 on mean restricted macro F1 under the
three-condition test above, and must beat B0 on mean Side I F1 specifically.**
Beating B0 on macro F1 while scoring 0.0 mean Side I F1 is not a pass (AC-7).

**Collapse detection is explicit, not assumed.** The report MUST include the
count of each predicted class pooled over all 50 validation folds, in both
domains. If the model predicts `Side I` on fewer than a handful of the 140 Side I
evaluation opportunities, it has effectively collapsed even if macro F1 looks
acceptable, and that must be visible in the pooled confusion matrix (§4.5).

### 5.8 Prediction path and failure behaviour

`src/rail/predict.py` exposes `predict(input_path: Path) -> pd.DataFrame`
returning the exact submission schema (§3), plus a **logic-free**
`--input`/`--output` CLI wrapper over that same function. Per `CLAUDE.md`, the
wrapper contains no behaviour: argument parsing, a call to `predict()`, a
`to_csv`. The reviewer checks this specifically.

`input_path` may be a single CSV file or a directory of CSV files; in both cases
the returned frame has one row per input file, with `file_id` the file's name
including extension.

**Order of operations inside `predict()` for each file**, fixed:

1. Read the file, assert shape `(10000, 129)`.
2. Derive `v` from column 1 (§5.3).
3. **If `v < V_MIN` (1.0 m/s): emit `Normal`, log it, count it, next file.**
   (§5.3a.4.) The model is not invoked.
4. Otherwise extract features, drop `meta_` columns, apply the fitted pipeline,
   `argmax` (§5.6).

**Failure behaviour.** If any of steps 1, 2 or 4 fails for a file (unreadable,
wrong shape, extraction error), the file still gets a row, predicted `Normal`,
and the failure is logged to stderr with the filename and the reason. Rationale:
a missing row makes the whole submission invalid under §3.2's completeness
requirement, which costs the entire subsystem's score; a wrong row costs at most
that file. The count of such fallbacks MUST be reported and MUST be zero on the
real Test set for the run that produces the submission (AC-11).

**The §5.3a.4 low-speed rule is NOT a failure fallback and its count is reported
separately** (AC-30). Conflating the two would hide either a genuine bug behind a
legitimate domain rule, or vice versa.

---

## 6. Acceptance criteria

Every item is observable in command output. The tester verifies each and records
the evidence verbatim in `docs/tests/rail-raw.txt`; the reviewer checks the same
list. An item that cannot be checked from output is a defect in this plan.

**Environment and plumbing**

- **AC-1** — No hardcoded data path anywhere in `src/rail/` or `scripts/`. A
  grep for the literal string `02_Datasets`, for `nebulax-data`, and for
  `NEBULAX_DATA` across `src/` and `scripts/` returns **zero hits outside
  `src/common/paths.py`**. All rail data access resolves through
  `RAIL_TRAIN_DIR`, `RAIL_TEST_DIR`, `RAIL_TRAIN_LABELS`.
- **AC-2** — The loader honours the header row. Output shows, for at least 3
  named Train files and 1 named Test file, `df.shape == (10000, 129)` — not
  `(10001, 129)`. Shapes only; no file contents printed.
- **AC-3** — Label counts read from `RAIL_TRAIN_LABELS` print exactly
  `Normal: 234`, `Side I: 14`, `Side II: 24`, total `272`. Test file count prints
  `68`.
- **AC-3b** — **Eligible-set counts printed, measured at the derived threshold.**
  With **`V_MIN = 1.28` m/s** (derived, §5.3a.2a), output prints: the number of
  Train files excluded, their per-class breakdown, the resulting eligible
  per-class counts, **and the realised always-Normal macro F1 on that set**.
  Expected: **44 excluded, all Normal, eligible 190/14/24 = 228, B0 ≈ 0.3030** —
  identical to the pre-amendment figures, because no file lies in `[1.00, 1.28)`.
  The excluded set MUST contain **zero Side I and zero Side II files**; the
  implementer asserts this and **stops and returns to the planner if the
  assertion fails**, since that would contradict Gate 1's `[0,2)` histogram and
  mean the speed derivation had changed. The realised counts are what all
  downstream fold arithmetic uses, and the report states the realised numbers
  rather than this plan's estimates.

**Feature cache**

- **AC-4** — The streaming extractor completes over all 272 Train files and all
  68 Test files and writes `artifacts/rail/features_train.parquet`,
  `artifacts/rail/features_test.parquet` and `artifacts/rail/feature_meta.json`.
  Output prints the shapes: train `(272, F)` and test `(68, F)` with the **same**
  `F` columns, and prints `F`. **`F` is in the range 500-1,200**; if it falls
  outside, the implementer reports the number and the discrepancy rather than
  silently proceeding. `feature_meta.json` contains `V_MIN`, `V_CUT`, the
  wavelength bands, the Welch parameters, `eps`, **`N` and the derived `V_MIN`
  with its formula**, and a version string **distinct from any iteration-1
  cache**.
- **AC-5** — Peak resident memory during extraction is reported and is **under
  1 GB**. Combined with AC-4 this demonstrates streaming rather than bulk load.
  The on-disk size of each parquet file is printed and each is **under 20 MB**.
- **AC-6** — Re-running any modelling step does **not** re-read raw CSVs: output
  shows the cache being loaded (a printed "loaded cache" line with the version
  string from `feature_meta.json`) and the raw-file-read counter at 0 for that
  step. A stale cache (version mismatch) triggers a rebuild, demonstrated by
  bumping the version string and observing the rebuild message.

**Validation**

- **AC-7** — The CV run uses `RepeatedStratifiedKFold(n_splits=5, n_repeats=10,
  random_state=20260918)` — the splitter's repr, or the three parameter values,
  appear in the output — over the **eligible** set only, and produces exactly
  **50** fold scores. Output reports, **for both the full and the restricted
  domain**: mean macro F1, std, p10, p90 across the 50 folds; mean/std/min
  per-class F1 for each of the three classes; the count of folds with Side I
  F1 == 0.0; and the pooled 3×3 confusion matrix.
- **AC-8** — Realised class counts per validation fold are printed for at least
  one repeat and match §4.2.1 recomputed at the realised eligible counts: each
  fold has **2 or 3 Side I**, **4 or 5 Side II**, and a Normal count consistent
  with the realised eligible Normal total.
- **AC-9** — Every metric value in every output traces to
  `src/common/metrics.py`. Grep shows **zero** direct `f1_score` imports or calls
  outside `src/common/metrics.py`, and zero hand-rolled F1 arithmetic. The
  Info Kit's worked example reproduces: `macro_f1_from_per_class([0.97, 0.40,
  0.60])` prints `0.657` at 3 dp.
- **AC-10** — No leakage in the pipeline: every fitted transformer (imputer,
  scaler, selector, class weights, B1's threshold `t`, **B2's thresholds
  `t_lo`/`t_hi`**) is inside the `Pipeline`/fold. Evidence: the estimator passed
  to the CV loop is printed and is a `Pipeline`, and a grep shows no `.fit(` or
  `.fit_transform(` call on the full training matrix before the CV loop. B1's and
  B2's thresholds are shown being fitted per fold (per-fold values printed, and
  they differ across folds).
- **AC-11** — **Zero** Test files hit the `Normal` *failure* fallback path of
  §5.8 in the run that produces the submission. Printed as `fallbacks: 0`. (This
  is distinct from AC-30's low-speed rule count, which may legitimately be
  nonzero.)

**Physical structure, speed, imbalance**

- **AC-12** — The side mapping is verified in output: a printed table, or an
  assertion, showing that the 128 channels split into exactly **64 Side I**
  (positions 1,3,5,7) and **64 Side II** (positions 2,4,6,8), each 32 vibration +
  32 shock, with the column index arithmetic of §5.2 shown for at least the first
  and last few indices.
- **AC-13** — **Side-swap symmetry test.** With Side I and Side II channels
  exchanged in a copy of the feature extractor's input (columns permuted by the
  §5.2 index map), a file the model predicts `Side I` is predicted `Side II`, and
  vice versa, and a `Normal` prediction stays `Normal`. Run on at least 10 Train
  files spanning all three classes, **all of them eligible files** (so the
  §5.3a.4 rule does not short-circuit the test). The pass/fail count is printed.
  Partial failure is a reportable finding, not a silent pass.
- **AC-14** — Derived speed is reported: min, median and max in m/s and km/h
  across all 272 Train files, **and broken down by the three classes**, and
  again for the eligible subset. The transition-counting derivation of §5.3 is
  shown. Column 1's dtype and distinct-value set are reported, confirming Gate
  1's `{0,1}` finding.
- **AC-15** — Wavelength-domain band features are demonstrably speed-dependent:
  for at least two eligible Train files with materially different derived speeds,
  the printed Hz band edges for the same wavelength band differ. This is the
  check that the bands are not secretly fixed in Hz.
- **AC-16** — Baselines are computed under the identical CV protocol and their
  mean macro F1 is printed **in both domains**: **B0** (always-Normal, full-domain
  ≈ **0.3030** on the realised eligible set), **B1** (physics rule), **B2**
  (speed-only). The selected learned model's mean **restricted** macro F1
  **exceeds B0, B1 and B2 under the three-condition margin test of §5.7a**, and
  its mean Side I F1 **exceeds B0's** (which is 0.0). If it does not, that is
  reported as the result, not hidden.
- **AC-17** — Pooled predicted-class counts across all 50 validation folds are
  printed for both domains, showing the model predicts each of the three classes
  a non-zero number of times. A model never predicting `Side I` fails this
  criterion.
- **AC-18** — The **adjacency diagnostic** of §4.4 is reported: median and IQR of
  adjacent-index feature similarity versus random-pair feature similarity, with
  the implementer's plain statement of whether adjacent files look materially
  more similar.
- **AC-18a** — The **fault-file clustering check** of §4.4.1 is reported. **This
  was satisfied at Gate 1 and PASSED**; the evidence in
  `docs/plans/rail-objection.md` §1 is carried forward and the test report cites
  it. It need not be recomputed, but if it is, the result must match.

**Iteration-2 criteria — speed handling, exclusion, restricted evaluation**

- **AC-27** — The **Side I index span caveat** (§4.4.2) appears in the test report
  and in the write-up, with the span 62-213 and `p = 0.0026`, and with the
  statement that the run-length gate passed so consecutive-second duplication is
  ruled out while a bounded acquisition window is not.
- **AC-28** — **Restricted-domain composition is reported before any restricted
  metric is relied upon.** Output prints: the realised per-class counts of
  eligible files with `v >= V_CUT = 9.70` (expected ≈101 Normal / 14 Side I /
  24 Side II, total ≈139 — the implementer prints the **realised** numbers and
  flags any material difference from these estimates); and the realised
  restricted-subset size and per-class composition per validation fold, either
  for all 50 folds or as min/median/max across folds plus a full breakdown for
  one repeat. The count of folds whose restricted subset is missing any class is
  printed out of 50.
- **AC-29** — **Gate 1b, the Test domain check, is reported.** Output contains
  exactly the six items of §4.8.2 and no more: counts at `v == 0`, at
  `0 < v < V_MIN`, above and below `V_CUT`; the five-number speed summary in m/s
  and km/h; the `T` summary; and the binary-column-1 confirmation. Evidence shows
  the read used `usecols=[0]`. **No Test per-file values and no Test columns
  2-129 appear anywhere in the output.** The write-up states what Gate 1b found
  and the two things it was allowed to inform (§4.8.4), and states that it
  informed nothing else.
- **AC-30** — **The low-speed inference rule's firing count is reported
  separately** from the failure-fallback count, for the Test run: printed as
  something like `low_speed_rule_fired: N`. If `N > 0` the affected `file_id`s
  are listed with their derived speeds, and the write-up states that those N
  predictions came from a stated rule rather than from the model.
- **AC-31** — **No speed feature reaches the model.** Output prints (a) the full
  list of `meta_`-prefixed columns, (b) the count of columns dropped by the
  pipeline's first step, and (c) an assertion, evaluated and printed as passing,
  that no `meta_`-prefixed name appears among the fitted estimator's input
  feature names. Additionally the implementer states, and the reviewer verifies,
  that no surviving feature is computable from column 1 alone. **B2 is the sole
  declared exception and its bypass is named explicitly in the output.** The
  check is against the **unit-suffix vocabulary of §5.3b.2a**, not against an
  enumerated feature list — see AC-36.
- **AC-32** — **Normalised-amplitude coefficient signs are reported.** For M1
  fitted on the full eligible set, the top-20-by-magnitude coefficients are
  printed with their feature names and signs. The implementer states plainly
  whether a systematic "lower normalised amplitude ⇒ fault" pattern is present.
  Such a pattern is a finding for the planner, not something to be absorbed.
- **AC-33** — **B2 is fitted in-fold and both figures are reported.** Output
  shows B2's per-fold thresholds differing across folds, its mean restricted and
  full macro F1 across the 50 folds, and, labelled explicitly as an optimistic
  all-data upper bound and not a bar, Gate 1's 0.5089.
- **AC-34** — **B2's restricted score is reported and interpreted.** If
  restricted B2 does not fall substantially toward B0, the implementer states so
  explicitly and raises it as a finding for the planner rather than proceeding
  as though the restricted domain were confound-free.
- **AC-35** — **The paired margin test of §5.7a is computed and printed** for the
  selected model against each of B0, B1 and B2, in the restricted domain:
  `mean(d)`, `std(d)`, the count of folds with `d_i > 0` out of 50, and the 10th
  percentile of `d`. The three PASS conditions are each evaluated and printed as
  pass/fail. The overall verdict against B2 is printed in one line.
- **AC-36** — **The unit-suffix contract is enforced and printed** (§5.3b.2a).
  Output shows: the closed suffix vocabulary; the count of feature columns under
  each suffix; the counts admitted to and rejected from the model matrix; and an
  assertion, evaluated and printed as passing, that the fitted estimator's input
  feature names contain **zero** columns suffixed `_hz`, `_persec` or `_amp` and
  zero prefixed `meta_`. The implementer additionally demonstrates the hard-error
  path: a deliberately mis-suffixed column aborts extraction with the offending
  name reported. B2's declared bypass is named explicitly in the output.

**Model selection discipline**

- **AC-19** — The number of distinct model configurations evaluated against the
  outer CV is printed and is **at most 6** (B0, B1, B2, M1, M2, M3). Any
  hyperparameter tuning is shown to be **nested** (`GridSearchCV` inside the
  outer loop), with the inner splitter's parameters printed. Non-nested tuning
  reported as an outer score fails this criterion. The inner search is shown to
  optimise the **full-set** macro F1 and the outer selection the **restricted**
  one, per §5.5.
- **AC-20** — The seed `20260918` appears in exactly one definition site, printed
  from that constant. Re-running the full CV twice yields **bit-identical** mean
  macro F1 to at least 6 decimal places, printed both times, in both domains.

**Output**

- **AC-21** — `predict()` on `RAIL_TEST_DIR` produces a file that
  `python scripts/validate_submission.py <path>` reports as `VALID` with default
  arguments (68 rows). The validator's `VALID:` line appears verbatim in the
  output.
- **AC-22** — The emitted CSV's header line is exactly `file_id,prediction`; its
  `file_id` values are exactly `Test1.csv`..`Test68.csv` in natural numeric
  order; and its distinct `prediction` values are a subset of `{Normal, Side I,
  Side II}`. Printed as the first 3 and last 3 lines of the file plus a value
  count — **no data-file contents are printed anywhere**. **Row count is exactly
  68 regardless of how many Test files were low-speed** (§5.3a.4).
- **AC-23** — The `--input`/`--output` CLI is logic-free: it parses arguments,
  calls `predict()`, writes the frame. Evidence is the wrapper's source being
  short enough to read in the review and containing no feature, model, metric, or
  speed-rule logic. The §5.3a.4 rule lives inside `predict()`, not in the wrapper.
- **AC-24** — **Zero leaderboard uploads.** No network call, no upload step, no
  submission-portal code anywhere in the subsystem. Verified by inspection.

**Reporting honesty**

- **AC-25** — The reported headline is the **mean RESTRICTED macro F1 across 50
  folds**, never a single fold and never the maximum fold. Its std and p10-p90
  range are reported alongside it in the same sentence or table, and the full-set
  mean is reported immediately adjacent, never suppressed.
- **AC-26** — The limitations of §4.2.1, §4.4, §4.4.2, §5.3a, §5.3b.5 and §7.13
  appear explicitly in the test report, not only in this plan. Specifically: 2-3
  Side I files per fold; no grouping key available; the Side I span caveat; the
  44 excluded low-speed files and what their exclusion assumes; the measured
  speed/class confound and the fact that `X/v` is only a first-order correction;
  and the open question about whether slow Normal labels are trustworthy.

**Total: 38 acceptance criteria.** The enumeration is AC-1, AC-2, AC-3, AC-3b,
AC-4 through AC-36 inclusive, and AC-18a — that is 36 plain-numbered items plus
AC-3b and AC-18a. Of these, **13 are new or materially rewritten in iteration
2**: AC-3b (rewritten again in the amendment pass), AC-27, AC-28, AC-29, AC-30,
AC-31, AC-32, AC-33, AC-34, AC-35, AC-36 (new in the amendment pass), AC-4
(amended), and AC-25 (headline redefined to the restricted metric). The
enumerated list above is definitive; any discrepancy between a count stated in
prose anywhere and the enumerated list is resolved in favour of the list.

**The acceptance criteria are INSTRUMENTATION, not the deliverable.** A working
end-to-end pipeline that reaches Gate 3 comes first. If an AC's output is
expensive to produce and the model is not yet running, the implementer **defers
that AC to the end and says so explicitly** in the gate report, naming which
criteria were deferred and why. A deferred AC is not a skipped AC: it is
reported as deferred, and it is satisfied before the subsystem is considered
complete. What is never acceptable is silently omitting one.

---

## 7. Ambiguities and assumptions

Each is flagged, not resolved, except where explicitly marked RESOLVED with the
reason. Source file and line given for each.

### 7.1 RESOLVED — the Info Kit contradicts itself on class counts

`Rail_Corrugation_Info_Kit.md:131-132` (§4) says:

> only ~9 Side I examples against ~190 Normal in the training data

`Rail_Corrugation_Info_Kit.md:86-87` (§2.2) says:

> 272 data files in total (`Train1.csv`–`Train272.csv`) - 234 normal files, 14
> Side I fault files, and 24 Side II fault files

These disagree. **RESOLVED in favour of §2.2: 234 / 14 / 24.** Reasons: §2.2 is
the dataset-description section and is the specific claim; §4's figures are
hedged with "~" inside a passage whose purpose is rhetorical (explaining why
macro F1 is used, not stating counts); §2.2's three numbers sum to 272, which
matches the stated file count and the verified file listing, whereas §4's ~9 +
~190 does not; and **`Train_Labels.csv` itself agrees with §2.2**, confirmed by
direct measurement at Gate 1. §4 lines 131-132 is stale prose from an earlier
dataset version. Nothing in this plan depends on §4's figures.

Note the same passage's "roughly 85-90% plain accuracy" (line 149) is consistent
with 234/272 = 86.0%, which is further evidence the ~9/~190 numbers are the stale
part rather than the whole passage.

**Iteration 2 note:** the plan's *training* population is now the eligible subset
(§4.2.0, §5.3a), which is a plan decision and not a contradiction of the Info
Kit. The corpus is still 234/14/24; the plan simply does not train on all of it,
and says so. **Coincidentally the eligible Normal count, 190, is numerically
close to §4's stale "~190". That is a coincidence and must not be read as
corroboration of the stale passage.**

### 7.2 No "both sides corrugated" class

`Rail_Corrugation_Info_Kit.md:87-90` defines exactly three labels, and "Side I"
explicitly means Side II is normal. `Rail_Corrugation_Info_Kit.md:54-56` however
says the two sides "must be judged **independently**" and that the model "must
localise the fault to a side rather than simply flagging the file as anomalous",
which is the language of two independent binary decisions — and two independent
binary decisions admit a fourth state, both-sides-corrugated.

**Two readings.** (a) The label space is genuinely three-valued and both-sides
cases do not occur in this dataset, in which case a 3-class classifier is exactly
right. (b) The task is conceptually two binary judgements collapsed into three
labels because the training set happens to contain no both-sides case, in which
case a both-sides Test file would exist and be unrepresentable.

**Not resolved.** The output schema permits only the three strings
(`Rail_Corrugation_Info_Kit.md:115`; example CSV), so a 3-class model is the only
*submittable* form regardless, and this plan builds one. But the per-side
contrast design of §5.2 is chosen partly because it degrades more gracefully
under reading (b) — a both-sides file has near-zero contrast and would be
predicted `Normal`, which is at least a defined behaviour. This is an argument
for the design, not a resolution of the ambiguity.

### 7.3 No grouping metadata — file independence is assumed, not established

`Rail_Corrugation_Info_Kit.md:92-93` states `Train_Labels.csv` has only
`filename` and `label`. `Rail_Corrugation_Info_Kit.md:81` states each file is 1
second. Nothing states whether files are independent recordings, consecutive
seconds of one run, or multiple passes over the same track sections.

**ASSUMPTION, explicitly flagged: files are treated as independent samples.**
This is forced — no grouping key exists — but if false it inflates every number
this plan produces, because near-duplicate files would straddle fold boundaries.
Two diagnostics mitigate this, and both measure rather than assume:

1. The **adjacency diagnostic** (§4.4, AC-18), on feature similarity between
   index-adjacent files. Partial only, since similar files need not be adjacent.
   **Not yet run** — it requires the feature cache.
2. The **fault-file clustering check** (§4.4.1, AC-18a), on filename index order.
   **RUN AT GATE 1 AND PASSED.** Largest run at gap ≤ 2 is 1 for Side I and 2 for
   Side II; neither class has a run of 3+; both sit at or below the seeded random
   median. Consecutive-second duplication of fault recordings is ruled out.

**RESIDUAL CAVEAT (iteration 2, §4.4.2): the Side I index span.** Side I occupies
indices 62-213 only, span 151 of a possible 271, with `P(random span ≤ 151) =
0.0026` at k=14 under the same seeded reference. Side II shows nothing comparable
(span 265, `P = 0.8865`). The run-length gate passed, which is the statistic that
bears on the duplication concern, so **this does not block**. But it is
consistent with the Side I recordings having come from a bounded acquisition
window, which would reduce the effective number of independent Side I examples
below 14. It is a single post-hoc statistic on 14 files and could easily be
chance. **It MUST appear in the write-up (AC-27).** No further computation is
mandated, because inventing and then applying a new criterion to the same data
is the post-hoc testing this plan forbids elsewhere.

Neither diagnostic eliminates the risk. **This remains the largest methodological
risk in the subsystem and must be stated in the final report.**

### 7.4 RESOLVED — column 1's encoding, and Nyquist saturation

`Rail_Corrugation_Info_Kit.md:74-77` says the speed sensor output "toggles
between 1 and 0" and that speed is obtained "by counting the 0/1 transitions over
a specific time", without stating whether the stored column is literally binary,
whether it is sampled at 10,000 Hz like the vibration channels, or whether any
conditioning has been applied.

**RESOLVED by measurement at Gate 1** (`docs/plans/rail-objection.md` §4):

- **dtype** `int64` in every file sampled; **distinct values** literally `{0, 1}`
  across all 272 Train files; **zero** non-binary files; no NaNs. The
  "threshold at the midpoint" contingency does not apply and was not used.
  Transitions are counted as `count(diff != 0)`.
- **No Nyquist saturation.** Tooth passage rate maxes at 656.5 teeth/s, i.e.
  15.23 samples per tooth period at the fastest file and 30.28 at the median. Max
  `T` = 1313 against a 10,000 ceiling, 13.1% of capacity. Level-run lengths
  (3-9 samples at the fastest sampled file) confirm it directly. **The derivation
  is not invalidated by saturation.**

**What is NOT resolved and moves to §7.10:** why 38 files have a column 1 that
never transitions.

### 7.5 ASSUMPTION — the wavelength band edges

`Rail_Corrugation_Info_Kit.md:11-13` gives the corrugation wavelength range as
"a few centimetres to dozens of centimetres". The five bands mandated in §5.3
(`0.02-0.04, 0.04-0.08, 0.08-0.16, 0.16-0.32, 0.32-0.64` m) are **my
interpretation of that prose**, not a spec constant. "A few centimetres" is read
as ~2-5 cm and "dozens of centimetres" as ~20-60 cm; the logarithmic spacing is a
choice, not a specified one.

Flagged as an assumption. It is a module constant precisely so it is visible and
changeable. Gate 1 confirmed the bands are numerically well-behaved at operating
speeds (all five resolvable and sub-Nyquist from ~9.8 to ~19.5 m/s), which
validates the *engineering* of the choice but not the physical *edges*.

### 7.6 RESOLVED — the files' header row strings

`Rail_Corrugation_Info_Kit.md:74-81` describes column *meanings* and their *order*
but never quotes the literal header text. **Gate 1 read them.** Column 1 is
`'Rotating speed'`; columns 2-129 are
`'<Vibration|Shock> of bearing in position <p> of car <c>'`; **all 128 channel
headers match §5.2's index arithmetic**, and headers are byte-identical across
the sampled Train and Test files with no BOM. The Info Kit's stated ordering is
confirmed against the actual files, so **the §5.2 side mapping is valid**.

The plan continues to mandate positional access plus a shape assertion (§5.4),
because positional access is what the Info Kit's column *order* guarantees. A
name-based assertion is now additionally available and MAY be added.

### 7.7 ASSUMPTION — per-channel feature function contents

The Info Kit mandates "signal processing and machine learning"
(`Rail_Corrugation_Info_Kit.md:45-46`) and "time-frequency analysis" to extract
"the dominant wavelength" (`:43-44`), but specifies no particular features. The
per-channel statistics named in §5.4 (band powers, RMS, peak, kurtosis, skewness,
crest factor, dominant-wavelength centroid, spectral entropy) are **my choices**,
not spec-derived. Only the wavelength-band-power family and the
dominant-wavelength centroid are directly traceable to the Info Kit's wavelength
discussion — and the centroid is now mandated **in metres rather than Hz**
(§5.3b.4), which brings it closer to the Info Kit's own "dominant wavelength"
language as well as removing a disguised speed feature.

Flagged. The implementer may not add features beyond this set without returning
to the planner, because silently widening the feature set changes the `p >> n`
trade-off of §5.4 — and, newly, because a casually added feature could reintroduce
the speed shortcut §5.3b exists to remove.

Welch PSD parameters are likewise unspecified by the Info Kit and are set by this
plan as constants to be recorded in `feature_meta.json`: `nperseg=2048`,
`noverlap=1024`, `window="hann"`, `fs=10000`, `detrend="constant"`. Gate 1
confirmed the resulting 4.883 Hz resolution is fine enough for all five bands at
all eligible speeds (the lowest band edge at the slowest eligible file must be
checked by the implementer and printed; AC-15 exercises the speed-dependence, and
the NaN-and-impute path of §5.3 handles any band that still falls below
resolution).

### 7.8 The `predict.py` / app contradiction — logged, not re-litigated

`CLAUDE.md` OPEN QUESTIONS §1 records that the Info Kits demand a `predict.py`
with an `--input`/`--output` CLI
(`Rail_Corrugation_Info_Kit.md:109-110`, `:121-123`) pointing at a **Deliverables**
section that does not exist, while
`01_Problem_Statement_3_Specifications.md:150-155` mandates an app and never
mentions `predict.py`. That is already resolved repo-wide by satisfying both:
`predict(input_path) -> pd.DataFrame` plus a logic-free CLI wrapper. This plan
adopts that (§5.8) and does not reopen it. **No app is built this session**
(models only), which is a session-scope decision recorded in `CLAUDE.md`, not a
rail-specific one.

### 7.9 Test set class distribution is unknown

`Rail_Corrugation_Info_Kit.md:97-100` says the 68 Test files' reference labels are
held by the organising committee. Nothing states whether the Test set has the same
234:14:24 proportions, is balanced, or is enriched for faults. 68 files at Train
proportions would be ~58 Normal / ~3.5 Side I / ~6 Side II — under 4 Side I files,
which would make the held-out macro F1 extremely high-variance regardless of model
quality.

**Not resolved, and it cannot be.** It is recorded because it bounds how much any
local number can predict the leaderboard number, which matters given the zero-
upload constraint: even a perfectly validated model has a wide confidence interval
on 68 files. The plan's response is to report the distribution rather than a point
estimate (§4.5) and to make no claim about the held-out score.

**Note that Gate 1b (§4.8) does NOT resolve this.** Gate 1b reads Test *speeds*,
not Test *labels*. It can say how many Test files are slow; it cannot say how many
are faults.

### 7.10 ASSUMPTION (iteration 2) — the low-speed files and the exclusion rule

Gate 1 established, as fact: 38 Train files have a column 1 constant at `1` with
zero transitions, all labelled Normal, all with live vibration channels; 6 more
yield `0 < v < 1` m/s, also all Normal.

**The Info Kit says nothing about standstill recordings, sensor dropout, a
minimum speed, or excluding files.** Two readings of the flat speed channel are
live and the spec settles neither:

- **(a) Genuine standstill.** The train was stopped, the vibration channels kept
  recording ambient and auxiliary vibration. The file is out of domain.
- **(b) Speed-sensor dropout.** The train was moving; the toothed-wheel sensor
  failed or its output was stuck. The file is in domain but its speed is
  unrecoverable.

**Three assumptions are recorded here, all of them mine, all flagged:**

1. **Excluding `v < V_MIN` from training is correct under both readings.** Under
   (a) the files are out of domain; under (b) their speed is unknown, so the
   wavelength-domain representation and the speed normalisation — the plan's
   primary and secondary feature families — cannot be computed for them. The rule
   does not depend on resolving the ambiguity. **This is the assumption that
   allows the plan to proceed without an organiser answer.**
2. **`V_MIN` is now DERIVED, not chosen** (§5.3a.2a): `N · λ_max / t = 1.28 m/s`,
   with `N` the remaining assumption, flagged at §7.14. It was formerly the
   round number 1.0 m/s. It is still not a spec constant and not fitted to any
   score, and it goes beyond the user's directive, which covered only the 38
   exact-zero files. Reasoning in §5.3a.3: at 0.0445 m/s a file traverses 4.45
   cm, less than one period of the longest wavelength band, and all five bands
   fall below the Welch resolution. Any value in roughly `[0.5, 3.0]` would
   exclude a similar set. It is an order of magnitude below the lowest fault-file
   speed. The 6 extra excluded files are all Normal.
3. **Predicting `Normal` for low-speed inference inputs is my choice** (§5.3a.4),
   justified by: 44/44 low-speed Train files are Normal; under reading (a) no
   corrugation evidence can be present; and the alternative — running a model on
   an almost-entirely-imputed feature vector — produces a prediction with no
   information content. It is explicitly **not** justified by Normal being the
   majority class, and it is confined to a narrow, physically defined region
   fixed before any Test file was inspected.

Under reading (b), assumption 3 is the weakest of the three: a moving train with
a dead speed sensor could be running over corrugated rail, and this rule would
mislabel it. The plan accepts that cost, bounds it (§5.3a.4 point 4), logs every
firing, and reports the count (AC-30). **If Gate 1b reveals many low-speed Test
files, the write-up must say prominently that a material fraction of the
submission came from a rule rather than a model** — but the rule does not change,
because changing it in response to Test would be fitting to Test.

### 7.11 ASSUMPTION (iteration 2) — the evaluation constants

Three numbers in this plan are chosen by me, are not in the spec, and are fixed
before any model is fitted. They are recorded together so a reviewer can see them
all at once:

1. **`V_CUT = 9.70 m/s`** (§4.7.2), the restricted-evaluation cut. Derived from
   the measured Side I minimum 9.7023 m/s, rounded down so all 14 Side I files
   fall inside. **It is derived from training data**, which would be leakage if
   it entered a feature or a decision rule — it does neither. It partitions the
   *reporting*, nothing else.
2. **`mean(d) > 0.05`** (§5.7a), the required mean paired margin over a baseline.
   Rationale: one Side I file flipping in a fold moves per-fold macro F1 by
   ~0.07-0.10, so a smaller mean advantage is within the resolution of a single
   file's fate.
3. **`d_i > 0` on ≥ 35 of 50 folds, and `p10(d) > −0.05`** (§5.7a). Rationale: a
   70% paired win rate is far outside coin-flip behaviour while remaining
   attainable at this quantisation; requiring all 50 would be unattainable.

(`V_MIN = 1.28 m/s` is now derived rather than chosen — it has moved out of
this chosen-constant family into §5.3a.2a, with its remaining assumption `N` at
§7.14; it is recorded separately at
§7.10, because it is a *training-population* decision rather than an evaluation
one.)

All are **stated in advance**, which is the property that matters more than
their exact values. Relaxing any of them after seeing results is forbidden to the
implementer and is a planner decision requiring a Revision log entry.

### 7.12 ASSUMPTION (iteration 2) — the functional form of the speed dependence

§5.3b.3 permits `X/v` and `X/v²` and strikes raw `X`, on the model
`X ≈ g(condition) · h(v)`. **The Info Kit gives no functional form for `h`.** It
says only that the method "is affected by train speed"
(`Rail_Corrugation_Info_Kit.md:45-46`). Whether axle-box vibration amplitude
scales with `v`, `v²`, or some other exponent is **not established by any source
this plan is allowed to use**, and I am not asserting it from outside knowledge.

Emitting both `/v` and `/v²` brackets the two most common first-order
possibilities; it does not guarantee either is right. If the true dependence is,
say, `v^1.5`, a residue remains in every normalised feature. **The restricted
evaluation of §4.7 exists precisely because this assumption may be wrong**, and
the write-up must state that the normalisation is a first-order correction under
an assumed form, not a proof of speed-invariance.

### 7.13 OPEN QUESTION FOR THE ORGANISERS (iteration 2) — speed-matched acquisition and the meaning of a slow "Normal"

**Destined for `CLAUDE.md` OPEN QUESTIONS.** This is the one genuinely
unresolvable ambiguity iteration 2 surfaced, and it is a question for the
organisers, not for us.

**The measurement.** In `Train_Labels.csv` plus the derived speeds (Gate 1):

- Not one of the 38 fault files (14 Side I, 24 Side II) has a derived speed below
  **9.7023 m/s**.
- **133 of the 234 Normal files do.**
- **38 Normal files have zero speed entirely** (column 1 constant, no
  transitions); 44 sit below 1.0 m/s.
- Mann-Whitney speed-vs-class: Side I vs Normal `p = 2.514e-04` (AUC 0.7766);
  Side II vs Normal `p = 9.080e-08` (AUC 0.8230).
- A speed-only classifier reaches macro F1 ≈ 0.509 against an always-Normal floor
  of ≈ 0.308.

**The two questions to ask, verbatim:**

1. **Were the fault recordings and the normal recordings collected under matched
   speed conditions?** If not, train speed is a confound that separates the
   classes in the training data without any relationship to corrugation.
2. **Does a `Normal` label at low speed mean "this rail was inspected and found
   sound", or does it mean "no corrugation was detected in this recording"?**

**Why it matters — the two readings have opposite implications:**

- **Reading A, acquisition artefact.** Fault sections happened to be recorded on
  faster runs. Speed is pure confound, the labels are all trustworthy, and the
  correct response is exactly what §5.3b and §4.7 do: remove speed from the
  features and evaluate where speed cannot separate.
- **Reading B, physical detectability threshold.** Corrugation excitation is
  simply not detectable below some speed, so a slow recording over corrugated
  rail gets labelled `Normal` because nothing was seen. Under this reading **the
  133 slow `Normal` labels are unreliable as ground truth** — some unknown
  fraction are corrugated rails recorded too slowly to tell. Training on them
  teaches the model to call corrugation "normal", and every metric computed
  against them is measuring agreement with a partly wrong label set.

**Neither reading is supported or excluded by anything in the Info Kit.**
`Rail_Corrugation_Info_Kit.md:45-46`'s "affected by train speed and ballast
noise" is consistent with both. I am not guessing which.

**What the plan does in the meantime**, and why it is the right hedge under
uncertainty: excluding the slowest files from training (§5.3a) removes the most
suspect labels under Reading B while being independently justified under Reading
A; striking speed features (§5.3b) is correct under both; and the restricted
evaluation (§4.7) reports the number on the subset where, under Reading B, the
labels are most trustworthy, since above 9.70 m/s corrugation is demonstrably
detectable (all 38 fault files were detected there). **This is a hedge, not a
resolution, and the write-up must say so.**

**Note the limit of the hedge.** If Reading B is true, even the restricted
domain's Normal labels could include undetected faults recorded at speeds just
above the cut. Nothing in this plan can detect that. Only the organisers can
answer it.

### 7.14 ASSUMPTION (iteration 2, amendment) — N, the minimum period count in V_MIN

§5.3a.2a derives `V_MIN = N · λ_max / t` and takes **`N = 2`**. Neither the Info
Kit nor the top-level spec states how many spatial periods of a corrugation
pattern must be observed for the pattern to be detectable. **`N` is my choice**,
in exactly the sense the wavelength band edges of §7.5 are my choice.

**Reasoning for N = 2**, restated so it can be challenged: corrugation is defined
by *periodicity*; one period in the window is indistinguishable from a single
transient (a joint, a switch, a wheel-flat impact); two periods is the smallest
integer count at which repetition can be evidenced rather than assumed.

**Sensitivity, so the reviewer can see what rides on it.** `V_MIN = 0.64·N`:
`N=1` → 0.64 m/s; `N=2` → **1.28**; `N=3` → 1.92 m/s. All three lie inside the
`[0, 2)` histogram bin that Gate 1 measured as containing **51 Normal and zero
fault files**, so **the choice of `N` in `{1, 2, 3}` cannot remove a single
minority example** and cannot alter the per-fold Side I/Side II arithmetic.

**Stronger than that, by direct measurement:** the eligible set is **identical at
`N=1` and `N=2`** — 228 = 190/14/24 in both cases — because no file at all sits
between 0.4896 and 1.2907 m/s. Only `N=3` would move anything, and then only 7
Normal files (to 221 = 183/14/24) and B0 by ~0.001. **Nothing in this plan is
sensitive to `N` within `{1,2,3}`**, which is why the choice is recorded as an
assumption and not escalated.

`λ_max = 0.64 m` is itself the §7.5 assumption, so `V_MIN` inherits that
assumption's status: if the band design changes, `V_MIN` is recomputed from the
formula rather than re-chosen.

---

## Revision log

**Iteration 1 (first pass)** — no prior findings. Plan written from
`Rail_Corrugation_Info_Kit.md`, `01_Problem_Statement_3_Specifications.md`, and
`04_Example_Submission/rail_predictions.csv`. One internal Info Kit contradiction
resolved (§7.1, class counts) with reasoning. Nine ambiguities/assumptions
recorded in §7. Nothing rejected yet.

**Iteration 2 (rework after Gate 1)** — entered on the implementer's objection at
`docs/plans/rail-objection.md`, raised at two §5.3 stop conditions the plan
itself defined. Nothing had been implemented; `src/rail/` held only an empty
`__init__.py`. All Gate 1 measurements below were independently re-derived and
confirmed by the caller and are treated here as established fact.

*Findings accepted, and what changed:*

1. **§4.4.1 clustering gate PASSED.** Side I largest run 1, Side II largest run
   2, both at or below the seeded random median over 10,000 draws; no run of 3+.
   **Changed:** §4.4.1 now records the result as established and marks the gate
   as satisfied rather than pending; §7.3's diagnostic 2 is marked run and
   passed. The gate specification is retained for the reviewer's check.
   **Not changed:** the split. The gate's purpose was to decide whether grouping
   was needed; it passed, so the stratified split stands unmodified.

2. **§7.6 headers RESOLVED.** All 128 channel headers match §5.2's arithmetic;
   64/64 side split confirmed; headers byte-identical across sampled Train and
   Test. **Changed:** §7.6 moved from open ambiguity to RESOLVED; §5.2 records
   the confirmation; §5.4 notes a name-based assertion is now optionally
   available. **Not changed:** the positional-access mandate stays primary,
   because column *order* is what the Info Kit guarantees and positional access
   is what that guarantee supports.

3. **§7.4 column 1 RESOLVED.** Literally binary `{0,1}` in all 272 files, no
   NaNs, no Nyquist saturation (13.1% of capacity at the fastest file).
   **Changed:** §7.4 marked RESOLVED on both its primary question and its
   sub-ambiguity; §5.3 drops the thresholding contingency as inapplicable; AC-14
   simplified accordingly.

4. **38 zero-speed files, all Normal (Objection A).** **Changed, substantially:**
   new §5.3a defines the exclusion rule with its physical reasoning (no
   wheel-rail interaction ⇒ no corrugation excitation ⇒ the task is undefined
   there), sets `V_MIN = 1.28 m/s` (derived), and defines the inference rule (predict
   `Normal`, logged and counted). §4.2.0 recomputes the eligible counts and the
   B0 floor, states the effect on class balance, and states plainly that Side I
   stays at 14 so the binding constraint is unchanged. §2.3, §4.1, §4.2.1, §4.3,
   §4.6, §5.1, §5.2, §5.4, §5.5, §5.7 updated for the new population. §5.8 fixes
   the order of operations inside `predict()`. §7.10 records all three
   assumptions. New AC-3b, AC-30.
   **Decided and stated where the user did not rule:** the 6 near-zero files
   (`0 < v < 1.0`) are folded into the same exclusion, flagged as **my** decision
   in §5.3a.3 with six reasons and again in §7.10. The user's directive fixed the
   38 zero-speed files, giving 196/14/24 = 234; my extension gives **190/14/24 =
   228**, a delta of 6 Normal files. **228 is binding throughout this plan**;
   both numbers are recorded so the reviewer can separate the user's decision
   from mine, and AC-3b requires the implementer to report realised counts rather
   than trust either estimate. Corresponding B0 floors: ≈0.3030 (228), ≈0.3039
   (234), 0.3083 (full 272, measured).

5. **Speed separates the classes; speed-only macro F1 ≈ 0.509 (Objection B).**
   **Changed, this is the largest change:** new §5.3b states the
   normalise-versus-predict line mechanically ("no feature computable from column
   1 alone"), enforces it structurally via the `meta_` prefix and an explicit
   drop step, strikes §5.3 mechanism 2 entirely, and adjudicates mechanism 3 by
   **striking raw absolute-level features and keeping the `/v` and `/v²`
   normalised ones** — the decisive argument being that supplying both lets a
   tree reconstruct `v` by division. The spectral centroid is moved from Hz to
   metres because a centroid in Hz is a disguised speed feature. §5.3b.5 states
   what this does not fix. §5.4's feature arithmetic and AC-4's range are revised
   downward (≈700-900 expected; 500-1,200 accepted). New AC-31, AC-32.

6. **B2 promoted to a mandatory floor.** **Changed:** iteration 1's B2
   (stratified dummy) is **removed from the comparison set entirely** so there is
   exactly one B2; §5.5's table and §5.7's baseline list are updated. New §5.7a
   specifies B2 as an in-fold-fitted speed-only estimator (the sole declared
   `meta_` consumer), predicts its expected behaviour in both domains so a
   surprise is visible, and defines "real margin" as a **three-condition paired
   test over the 50 folds** — `mean(d) > 0.05`, `d_i > 0` on ≥35/50, and
   `p10(d) > −0.05` — rather than a point estimate. Gate 1's 0.509 is explicitly
   demoted to "optimistic all-data upper bound, not the bar". New AC-33, AC-34,
   AC-35.

7. **Speed-stratified evaluation added.** **Changed:** new §4.7 defines
   `V_CUT = 9.70 m/s`, derives the restricted composition (≈101/14/24 ≈ 139 of
   228, with all fault files surviving because both class minima exceed the cut),
   works out that restricted folds hold ≈26-30 files with Side I and Side II
   counts **unchanged** at 2-3 and 4-5, and concludes the minimum viable k is
   therefore **unchanged at 5**. States that restricted macro F1 should be
   *lower* than full-set for a good model and that a higher one is a warning.
   Declines to stratify folds on speed, with reasoning. Makes the restricted
   number the **headline** and changes §5.5's outer selection rule to select on
   it, while keeping the inner search on the full set because inner folds are too
   small to restrict. §4.5 now reports both domains. New AC-28; AC-7, AC-16,
   AC-17, AC-25 extended to both domains. The ≈101 figure is marked DERIVED and
   the implementer must report realised counts before relying on it.

8. **Gate 1b added.** **Changed:** new §4.8 declares the Test column-1-only speed
   check, enumerates exactly six computed items, enumerates six prohibitions
   including `usecols=[0]` and "no use of the result to select a model", and
   confines what it may inform to two reporting questions. New AC-29.

9. **Side I index span, p = 0.0026.** **Changed:** recorded as a residual caveat
   in new §4.4.2 and in §7.3, mandated into the write-up via new AC-27.
   **Deliberately NOT changed:** the split, the gate criterion, and the
   independence assumption's status. Reasons in §4.4.2: the gate is defined on
   runs and runs are the statistic that bears on the duplication concern;
   inventing a span criterion after seeing the data and applying it to the same
   data is the post-hoc testing this plan forbids elsewhere; and a single
   post-hoc statistic on 14 files at p = 0.0026 is not decisive once the
   selection effect is acknowledged. The implementer was right not to treat it as
   a gate failure.

10. **New open question for the organisers.** **Changed:** new §7.13 records the
    speed-matched-acquisition question and the meaning-of-a-slow-Normal question,
    with both readings and their opposite implications, marked as destined for
    `CLAUDE.md` OPEN QUESTIONS. New §7.11 and §7.12 record the evaluation
    constants and the assumed functional form of the speed dependence as
    assumptions.

*Findings noted but deliberately NOT acted on, with reasons:*

- **The within-file speed stability observation** (objection §4: windowed
  transition-count spread has median 2, max 9, so files are near-constant speed,
  and iteration 1's acceleration-proxy feature would have little variance). **Not
  acted on as a feature question, because the feature is struck anyway** by
  §5.3b: any windowed transition-rate statistic is computable from column 1 alone
  and is therefore forbidden. The observation is recorded here rather than in the
  plan body because it no longer bears on any live decision. It does have one
  positive consequence worth stating: near-constant within-file speed means a
  single scalar `v` per file is an adequate basis for the band-edge derivation of
  mechanism 1, which the plan relies on.

- **The implementer's suggestion that the contrast family be made the *only*
  permitted feature family** (objection §3, third bullet). **Rejected, with
  reason:** it would discard the speed-normalised per-side absolute levels, which
  carry genuine information — a rail corrugated on one side raises that side's
  band power in absolute terms, and a file where both sides are quiet is
  different from a file where both are loud, which no contrast feature can
  express. The chosen position (strike raw levels, keep normalised levels,
  promote contrast to primary) retains that information while closing the
  shortcut. §5.3b.3 argues this explicitly and AC-32 gives the reviewer a
  concrete check on whether the normalised levels are behaving as expected. If
  AC-32 shows a systematic wrong-sign pattern, restricting to contrast-only
  becomes the natural iteration-3 response.

- **Re-running the §4.4.1 clustering gate.** Not mandated. It passed on evidence
  that is recorded, reproducible from a seeded reference, and derived from a
  single small CSV. AC-18a now cites the Gate 1 evidence rather than requiring
  recomputation.

- **Adding LightGBM/xgboost.** Still not added. The iteration-1 reasoning is
  unchanged and iteration 2 gives no new evidence bearing on it. Striking
  features made the matrix narrower, which if anything strengthens the case for
  the existing regularised models.

- **Excluding the 133 sub-9.70 m/s Normal files from training** (the natural
  extension of §7.13 Reading B). **Rejected, with reason:** it would discard 133
  of 190 remaining Normal files on the strength of an unresolved organiser
  question, leaving ~57 Normal against 38 fault — a different problem from the
  one posed. Under Reading A those labels are perfectly good data. The plan
  instead *evaluates* on the restricted domain (§4.7) while *training* on all
  eligible files, which is the choice that stays correct under both readings.
  Recorded so the option is visibly considered rather than overlooked.

### Iteration 2 — amendment pass (post-acceptance)

Iteration 2 was accepted with two amendments. The user also ruled on the two
items surfaced: **228 accepted and binding** ("the planner's physical cut is
better than my categorical one, and the 4.45 cm argument is the right basis"),
and **rejecting the 133-file training exclusion accepted** for the stated reason
that it is correct under both readings of §7.13, so the plan is not blocked
awaiting an organiser reply — that reasoning is retained explicitly because it
is load-bearing.

**Amendment A — V_MIN derived, not chosen.** `V_MIN` was the round number
1.0 m/s. It is now **derived** in new §5.3a.2a as `V_MIN = N · λ_max / t`, with
`λ_max = 0.64 m` (§7.5 band design), `t = 1.0 s` (Info Kit §2.1:81), and `N = 2`
the minimum spatial periods for periodicity to be evidenced rather than assumed
(new assumption §7.14). **Arithmetic: 2 × 0.64 / 1.0 = 1.28 m/s.**

*Changed:* threshold 1.0 → **1.28 m/s**. *Not changed:* **the eligible set
remains exactly 228 = 190 Normal / 14 Side I / 24 Side II.** The draft of this
amendment carried a range of 221-228 because Gate 1 measured counts at the 1.0
and 2.0 boundaries but never at 1.28. **That gap has since been closed by direct
measurement: there are ZERO files in `[1.00, 1.28)`.** The seven files in
`[1.0, 2.0)` sit at 1.2907, 1.2907, 1.3352, 1.6764, 1.7506, 1.8841 and
1.9138 m/s — all above 1.28, all retained. The speed distribution has a wide
empty gap between 0.4896 m/s (the fastest excluded file) and 1.2907 m/s (the
slowest retained one), so the derived threshold lands inside a region containing
no data at all. **B0 remains ≈0.3030, not a range.** All movement would have
been Normal in any case — the `[0,2)` bin is 51 Normal, 0 fault — so Side I
stays 14, Side II stays 24, and the per-fold minority arithmetic is untouched.

The derivation therefore buys auditability at **zero cost to any downstream
number**. AC-3b still requires the implementer to measure at 1.28 and assert
zero fault files excluded, because a plan that asserts a count the implementer
has not reproduced is exactly what this project does not do.

*Considered and rejected:* a Welch-resolution derivation,
`V_MIN = Δf · λ_max = 4.8828 × 0.64 = 3.125 m/s`, which would exclude ~60 files.
Rejected because the user's ruling fixes the spatial-period basis and,
independently, because per-band resolvability is already handled correctly and
per-band by §5.3's NaN-and-impute path; folding it into a global threshold would
double-handle it and discard files whose shorter bands are fine.

**Amendment B — mechanism 3 generalised to a dimensional rule.** The
struck-feature list was replaced by a general rule in new §5.3b.2a: *any feature
that is a position, width, or rate on the signal's time or frequency axis must be
expressed in metres or as a dimensionless ratio; anything in Hz, s⁻¹ or a
count-per-second is rejected.*

*Changed:* the rule explicitly distinguishes **frequency-axis** per-second terms
(which encode speed, and are rejected) from **amplitude-axis** per-second terms
(which come from the m/s² unit of the channels, encode shaking not speed, and are
governed by §5.3b.3 instead) — a naive dimensional test would have over-struck
band power and RMS. Enforcement moved from list-membership to a **closed
unit-suffix vocabulary** (`_m`, `_ratio`, `_amppv`, `_amppv2` admissible; `_hz`,
`_persec`, `_amp` rejected) validated **at extraction time**, with an
**unrecognised suffix a hard error that aborts extraction**, so a future feature
violating the rule fails loudly rather than silently at review. New AC-36; AC-31
and AC-4 amended.

*Full re-audit result:* **no previously permitted feature is struck.** The rule
reproduces every earlier per-case adjudication — spectral centroid struck in Hz,
survives in metres; band powers survive because the emitted value is an amplitude
and the band is labelled by λ in metres; `X/v` and `X/v²` confirmed acceptable as
amplitude-axis quantities. Spectral entropy was re-examined as a frequency-axis
quantity and **survives** because it is a normalised Shannon entropy and
dimensionless by construction. Across-car dispersion now explicitly **inherits**
the verdict of the feature it aggregates. The rule's value is prospective: it
governs features not yet written.

*Not changed, with reasons:* the restricted domain (§4.7) is untouched —
`V_CUT = 9.70` sits far above `V_MIN` so the restricted composition is unaffected
by Amendment A. The §5.7a margin conditions are unchanged. The split, the seed,
the fold count and the repeat count are unchanged. `N ∈ {1,2,3}` all fall inside
the zero-fault `[0,2)` bin, so the choice of `N` in that range cannot alter any
minority count — recorded in §7.14 rather than escalated.

**Acceptance criteria after this pass: 38.**

**THE PLAN IS FROZEN AT THIS POINT.** No amendment 3. Anything noticed from here
goes to `docs/backlog.md` and is actioned only if the implementer actually trips
on it.

*Iteration count: this is iteration 2 of a maximum 3. If iteration 3 does not
produce a plan that survives review, a blocker is written to `docs/blockers.md`
per the workflow and no fourth pass is started.*
