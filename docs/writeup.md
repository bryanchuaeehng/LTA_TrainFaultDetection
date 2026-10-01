# Rail Corrugation — Approach and Findings

**Headline: restricted macro F1 0.7765** (std 0.0833) over 50 cross-validation
folds. We used **zero of our five leaderboard uploads**. Every number below
comes from local validation, which means the validation itself had to be
trustworthy — that is what most of this write-up is about.

## In sixty seconds

- **The data contains a trap.** No fault file was recorded below 9.70 m/s,
  while 133 of 234 Normal files were. A model could score 0.509 on speed alone
  and learn nothing about corrugation.
- **So we report the harder number.** Macro F1 restricted to files above that
  speed, where every class spans the same range and the shortcut is unavailable
  by construction. It is the lower of our two figures, and it is the one we
  quote.
- **A speed-only baseline collapses from 0.3996 to 0.3078 under that
  restriction**, confirming the restricted domain does what we claim.
- **We found two leaks by measuring, not reviewing.** Both families of features
  were dimensionally correct and both smuggled speed back in; one correlated
  with speed at 0.998. We dropped 122 features and added an empirical gate.
- **We found two byte-identical duplicate file pairs** in the training data, and
  ruled out the block-structure risk we were most worried about.
- **Honest limits:** 14 Side I examples, per-fold scores ranging 0.575–0.946,
  and a correlation threshold we chose pragmatically rather than derived.

The rest of this document is the evidence for those six claims.

---

## The task

Each file is one second of vibration and shock data from 64 axle boxes on an
8-car train, sampled at 10 kHz: 10,000 rows × 129 columns, about 17 MB. We
classify each file as `Normal`, `Side I` (corrugation on the odd-numbered
axle positions' rail) or `Side II` (the even-numbered positions' rail).

Training data is 272 files: **234 Normal, 14 Side I, 24 Side II**. Fourteen
examples of a class is the constraint that shapes every decision that follows.

---

## The finding that changed the approach

Before modelling, we derived train speed from column 1 (a 90-tooth pulse
counter; transitions ÷ 180 × wheel circumference) and checked it against the
labels. The result was uncomfortable:

| Class | n | min speed | median | max |
|---|---|---|---|---|
| Normal | 234 | **0.00 m/s** | 8.13 | 19.48 |
| Side I | 14 | **9.70 m/s** | 12.96 | 18.60 |
| Side II | 24 | **11.69 m/s** | 14.00 | 18.51 |

**No fault file was recorded below 9.70 m/s. 133 of the 234 Normal files
were.** A rule using nothing but speed scores 0.509 macro F1 against an
always-Normal floor of 0.308.

This is a confound, not a feature. A model could score respectably by learning
"fast means fault" while learning nothing whatsoever about corrugation — and
with no leaderboard feedback, nothing downstream would ever have told us.

We responded in three ways.

**Speed still sets the wavelength bands.** Corrugation excites vibration at
frequency `f = v/λ`, so a fixed frequency band is physically wrong; the band
edges must move with speed. That use of speed is physics and it stays.

**Speed is barred as a classifier input.** Using `v` to decide *where to look*
is normalisation. Handing `v` to the model is the shortcut. Only the first is
allowed.

**The headline is measured on the restricted domain.** We report macro F1
twice: once over all eligible files, and once over only the files at or above
9.70 m/s (139 of 228, including *all* 38 fault files). Above that cut, every
class spans the same speed range, so the shortcut is unavailable by
construction. **The restricted number is the one we quote**, and it is the
lower of the two.

We also excluded 44 very slow files from training (all Normal, all below
1.28 m/s). At walking pace a wheel travels a few centimetres in one second —
less than a single wavelength of the patterns we are looking for — so
corrugation is not merely hard to detect there, it is absent from the signal.
The threshold is derived, not chosen: `V_MIN = N·λ_max/t = 2 × 0.64 / 1.0`,
requiring two full spatial periods of the longest band inside the one-second
window. Those files still receive a prediction at inference time (`Normal`, by
an explicit stated rule); all 44 low-speed training files are in fact Normal,
so the rule is never once contradicted by the data.

---

## Four baselines, and what each one rules out

A single score means little. Each baseline exists to close off a specific
"but maybe it's just…" objection.

| Baseline | Restricted macro F1 | What it rules out |
|---|---|---|
| **B0** always predict Normal | 0.2798 | That the class imbalance alone produces a good-looking score. It does not: 86% accuracy maps to 0.28 macro F1. |
| **B1** physics rule (Side I vs Side II band-power contrast, threshold fitted in-fold) | 0.5381 | That the learned model adds nothing over the obvious physical signal. |
| **B2** speed only | 0.3078 | **The confound itself.** This is the important one. |
| **M1** our model | **0.7765** | — |

**B2 is the key result, and the direction it moved matters.** On the full
eligible set the speed-only baseline scores 0.3996. Restricted, it collapses
to **0.3078** — essentially B0. That is the restricted domain doing exactly its
job: the speed shortcut stops working there. Had B2 stayed high after
restriction, the entire headline would have been suspect.

M1 beats all three on a paired test across the 50 folds (mean difference
> 0.05, wins on ≥35/50 folds, 10th percentile of the difference > −0.05, all
fixed before running). Against B2: **mean difference +0.4687, winning 50 of 50
folds**.

---

## The model

Per-channel features (band powers in five wavelength bands, RMS, peak,
kurtosis, skewness, crest factor, spectral entropy), aggregated across the 32
odd and 32 even axle positions, then combined into a **Side I vs Side II
contrast**: differences and log-ratios between the two sides of the same file.

The contrast family is the design's centrepiece. The two fault classes are the
same phenomenon mirrored, and a common multiplicative speed factor cancels
exactly in a log-ratio. We verified the extractor is *exactly* side-symmetric:
swapping the channel groups negates every difference feature and leaves every
symmetric feature untouched, with zero violations across 600 checks.

The *fitted model* does not inherit that symmetry: mirroring a fault file's
channels flips its prediction to the opposite side on only 8 of 12 test cases.
With 14 Side I against 24 Side II examples, the classifier learns side-specific
coefficients, so a mirrored fault vector falls outside the region it was fitted
on. We report this rather than enforce symmetry by augmentation, because the
fix is a modelling change and the honest reading is that it is a symptom of the
sample size. The property the design actually claims — that the *features* are
side-symmetric — holds exactly.

Classifier: logistic regression with balanced class weights, every transformer
fitted inside the training fold only. Validation is
`RepeatedStratifiedKFold(5 folds × 10 repeats, seed 20260918)` — 50 folds, so
each of the 14 Side I files is evaluated ten times under ten independent
partitions rather than once.

---

## The leaked features

This is the episode we would most want a judge to read.

We wrote a rule barring any feature whose units carry a per-second term on the
frequency axis, on the grounds that such a feature encodes speed. The rule
required the spectral centroid to be expressed as a wavelength in **metres**
rather than a frequency in Hz. Dimensionally, that is correct.

It was also wrong, and measurement caught it where review did not.

The conversion is `λ = v/f_centroid`. Where the centroid is roughly constant
across files — which it is, for broadband shock data with no dominant peak —
**dividing by it simply multiplies speed back in**. One such feature correlated
with speed at **0.9982**. A depth-3 decision tree on that single feature scored
0.3984 on the restricted domain: *better* than raw speed itself (0.3566), the
very thing we had banned.

Checking further, a second family had the same disease in the opposite
direction. Features normalised as `X/v²` **over-corrected**: where amplitude
already scales roughly as `v²`, dividing by it reintroduces a speed signal with
inverted sign. The worst sat at 0.9879.

Both families passed the dimensional rule. Both were genuinely correct in their
units. The lesson is compact:

> **A unit contract checks a property of the formula. Leakage is a property of
> the data.** No static rule can catch this; only measuring against the actual
> data can.

We dropped the wavelength family outright (48 features) and added an empirical
gate: any feature correlating with speed above **|Spearman| 0.90** on the
restricted training set is removed. That dropped a further 74. Feature count
went **1,104 → 1,056 → 982**.

**The 0.90 threshold was chosen pragmatically, not derived**, and we say so
plainly. The highest correlation among surviving features is **0.8995** — just
under the line. A stricter threshold would have removed more; we have not
established that 0.90 is optimal, only that it is defensible and applied
consistently. The gate uses speeds and features only, never labels, and is a
single fixed feature-set decision rather than a per-fold fit.

---

## Two data findings

**Duplicate files.** While running the adjacency diagnostic we found a pair of
training files with a feature-space similarity of exactly 1.000000. They are
**byte-identical**. Hashing the corpus found a second such pair:
`Train107.csv` = `Train115.csv`, and `Train165.csv` = `Train187.csv`. So the
272 training files are 270 distinct recordings, and our 228 eligible files are
226. All four are Normal, there are no duplicates within the test set, and no
training file matches any test file — so there is no train-on-test
contamination. The effect on the headline is small and confined to the majority
class, but it is real and we report it rather than quietly deduplicating.

**No block structure.** Our largest untested worry was that consecutive
filenames might be consecutive seconds of the same recording, which would put
near-identical files on both sides of a fold boundary and inflate everything.
We tested it: cosine similarity between index-adjacent file pairs has median
**−0.0692**, against **−0.0292** for random pairs. Adjacent files are, if
anything, *less* alike than random ones (Mann-Whitney p = 0.885). Only 1.6% of
adjacent pairs exceed the random 95th percentile, against 5% expected under no
structure. Separately, the 14 Side I files are scattered through the index with
a longest consecutive run of 1 — at or below what random placement produces.
**The concern does not materialise.**

---

## The numbers

| Metric | Restricted (headline) | Full eligible set |
|---|---|---|
| **M1 macro F1** | **0.7765** | 0.7858 |
| Standard deviation | 0.0833 | 0.0812 |
| 10th / 90th percentile | 0.6483 / 0.8744 | 0.6571 / 0.8806 |
| Min / max fold | 0.5753 / 0.9456 | 0.5810 / 0.9479 |
| B0 / B1 / B2 | 0.2798 / 0.5381 / 0.3078 | 0.3030 / 0.5335 / 0.3996 |

Per-class F1, restricted: **Normal 0.939, Side I 0.565, Side II 0.826.**

Evaluation population 228 eligible training files (190/14/24); restricted
subset 139 (101/14/24). 50 folds; 2–3 Side I and 4–5 Side II files per
restricted validation fold.

---

## We tried to beat it and could not

After freezing the result we ran a model search, pre-registered in
`docs/search-plan.md` **before** any configuration was scored: eight
configurations spanning three regularisation strengths, L1 and L2, a linear
SVM, ridge, random forest and extra trees.

It used **nested cross-validation**: configurations compete inside inner folds
carved from the training portion, and the winner is evaluated once on the
untouched outer fold. What that estimates is the whole selection procedure, not
the luckiest configuration.

**The nested estimate came out at 0.7558 — below the incumbent's 0.7765.** No
challenger passed the three-condition paired test; the closest, stronger L2
regularisation, won on only 8 of 50 folds. Random forest was worst at 0.6774
and was never once chosen by inner selection, which is what 38 fault examples
in 982 dimensions predicts.

The comparison worth noting: the best *per-configuration* outer mean was
0.7773, marginally above the incumbent. Quoting that as an improvement would
mean selecting a configuration on the same folds used to measure it. The nested
estimate, which does not do that, lands lower than both — and that gap is
roughly the size of the bias such a search introduces.

So the incumbent stands and the frozen artefacts were never rewritten. We
report this because a null result is evidence: the model we shipped is not a
lucky draw from a wide search, and we have the full log of every configuration
that lost.

## What we would not want overlooked

**Side I is the weak class and the honest number is 0.565.** With 14 examples,
a validation fold holds two or three of them, so per-fold Side I F1 can only
take a handful of discrete values. It is 0.0 on **2 of 50 folds**. That is
quantisation at tiny sample size, not a collapse — but it means the Side I
figure carries real uncertainty and a single-split estimate would have been
meaningless.

**The fold spread is wide, and that is the honest picture.** Restricted macro
F1 ranges from 0.575 to 0.946 across folds. We report the distribution rather
than a point estimate precisely because the range is this wide; quoting only
the mean would overstate what 14 minority files can support.

**The speed confound probably persists into the test set.** Train and test have
near-identical speed composition (files below the low-speed cut: 16.2% vs
14.7%; above the restricted cut: ~61% vs 63.2%). If a model partly exploits the
shortcut, it will still be available at test time — which is exactly why we
quote the restricted figure rather than the flattering full-set one.

**The low-speed rule fires on 10 of 68 test files.** Roughly one submitted row
in seven comes from a stated rule rather than from the model. We think that is
the right call physically, but it should be visible, not buried.

**One methodological caveat we decided not to hide:** the correlation gate was
computed once on the restricted training set rather than re-derived inside each
fold. It uses no labels, so it is not label leakage, but it is a data-derived
decision made with all the training features in view, and a fully conservative
protocol would nest it.

---

## Open questions for the organisers

**Were fault and normal recordings collected under matched speed conditions?**
No fault file sits below 9.70 m/s while 133 Normal files do, and 38 Normal
files have no wheel motion at all. Two readings have opposite implications:
either an artefact of how data was gathered, or corrugation genuinely is not
detectable below some speed — in which case the slow `Normal` labels mean "not
detected here" rather than "rail inspected and sound", and 133 training labels
are less reliable than they appear. We could not resolve this from the
documentation, and it materially affects how the results should be read.

**Are the duplicate file pairs intentional?** `Train107`/`Train115` and
`Train165`/`Train187` are byte-identical.

**Can a file show corrugation on both sides?** The three labels are mutually
exclusive, but the documentation elsewhere describes judging the two rails
independently, which would admit a fourth state. We built a three-class model,
as the output schema requires.

One further discrepancy, resolved locally: the Info Kit's scoring section cites
"~9 Side I against ~190 Normal", while its dataset section and the label file
both give 234/14/24. We used the label file. Its worked macro-F1 example
(0.97/0.40/0.60 → 0.657) we reproduce exactly; a second illustrative figure in
that section (0.33 for an always-Normal model) is arithmetically loose — the
true value on these counts is 0.308, since such a model cannot have perfect
precision on Normal.

---

## Reproducing

```
python -m src.rail.predict --input <test_dir> --output rail_predictions.csv
python scripts/validate_submission.py rail_predictions.csv
```

`predict()` is self-contained: it loads its own model and needs neither the
training corpus nor any environment configuration. The command-line wrapper
contains no logic beyond argument parsing, so the app and the CLI cannot drift
apart. Full methodology is in `docs/plans/rail.md`; verbatim run output for
every claim above is in `docs/tests/`; known limitations are tracked in
`docs/backlog.md`.
