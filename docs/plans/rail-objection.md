# Rail Corrugation — Implementer objection (Gate 1, iteration 1)

Raised by the implementer under rule 3, at the end of **Gate 1 (measurement and
diagnostics only)**. No feature extraction and no modelling was performed.

**Summary.** The §4.4.1 hard gate (fault-file clustering) **PASSES** — see §1
below, it is recorded here for completeness and is not the objection. The
objection is raised by **two separate §5.3 clauses that the plan itself makes
stop conditions**, both triggered by measurements of column 1:

- **Objection A — degenerate speed.** 38 of 272 Train files have a column 1 that
  never transitions, so the plan's derivation yields `v = 0.0000 m/s` for them.
  A further 6 yield `0 < v < 1 m/s`. §5.3's mandatory sanity band is violated,
  and `v` appears in a denominator in §5.3 mechanism 3.
- **Objection B — speed/class confound.** Derived speed separates the classes on
  its own. Every one of the 38 fault files has `v >= 9.70 m/s`, while 133 of 234
  Normal files fall below that. A speed-only rule reaches macro F1 **0.509**
  against the always-Normal floor of **0.308**. §5.3 names this outcome
  explicitly as "a finding for the planner, not something the implementer
  patches".

Both are decisions the plan reserves to the planner. Control returns to the
planner.

---

## 1. §4.4.1 hard gate — PASSED (not the objection)

Recorded first because §4.4.1 gates everything else and AC-18a requires the
evidence be reported before any feature extraction.

Source: `RAIL_TRAIN_LABELS`, shape `(272, 2)`, columns `['filename', 'label']`.
Label counts `Normal: 234`, `Side I: 14`, `Side II: 24`, total 272 — matching
§4.2 and §7.1's resolution exactly. Filenames are exactly `Train1.csv` ..
`Train272.csv`, verified as a complete index set over 1..272.

**Side I — 14 files, full sorted index list:**

```
[62, 83, 100, 106, 121, 137, 150, 163, 170, 180, 185, 194, 202, 213]
```

Gaps: `[21, 17, 6, 15, 16, 13, 13, 7, 10, 5, 9, 8, 11]`
(min 5, median 11.0, max 21).
Maximal runs at gap <= 2: **14** (i.e. every file is its own run).
Runs of length >= 2: **0**. Runs of length >= 3: **0**.
**Largest run length: 1.**

**Side II — 24 files, full sorted index list:**

```
[2, 12, 26, 30, 43, 53, 72, 93, 103, 104, 111, 127, 164, 179, 188, 198, 199,
 207, 216, 217, 240, 248, 265, 267]
```

Gaps: `[10, 14, 4, 13, 10, 19, 21, 10, 1, 7, 16, 37, 15, 9, 10, 1, 8, 9, 1, 23,
8, 17, 2]` (min 1, median 10.0, max 37).
Maximal runs at gap <= 2: **20**.
Runs of length >= 2: **4** — `[103,104]`, `[198,199]`, `[216,217]`, `[265,267]`.
Runs of length >= 3: **0**.
**Largest run length: 2.**

**Random reference.** Seed `20260918`, 10,000 draws per class of k indices
without replacement from 1..272, same run statistic:

| k | largest-run mean | median | p5 | p95 | max | P(>=1 run of len>=3) |
|---|---|---|---|---|---|---|
| 14 (Side I ref) | 1.865 | 2.0 | 1.0 | 3.0 | 5 | 0.0996 |
| 24 (Side II ref) | 2.503 | 2.0 | 2.0 | 4.0 | 8 | 0.4261 |

Full largest-run distributions:
k=14 `{1: 2432, 2: 6572, 3: 912, 4: 77, 5: 7}`;
k=24 `{1: 89, 2: 5650, 3: 3542, 4: 598, 5: 107, 6: 12, 7: 1, 8: 1}`.

`P(random largest_run >= observed)` is 1.0000 for Side I (observed 1) and
0.9911 for Side II (observed 2). Both observed values sit **at or below** the
random median. Observed clustering is if anything *less* than chance.

**Verdict: GATE PASSED.** Neither fault class shows a run of 3 or more, and
neither largest-run length is outside the random reference. Per §4.4.1 this is
**weak** evidence: it is consistent with independent recordings but does not
prove independence, since an interleaved acquisition order would scatter files
from one session.

### 1a. One clustering-adjacent observation, offered as context, NOT as a gate failure

The gate criterion in §4.4.1 is defined on *runs*, and on runs both classes pass
cleanly. A different statistic is worth the planner's attention:

**Side I occupies indices 62 to 213 only — span 151 out of a possible 271.**
Against the same seeded random reference, `P(random span <= 151) = 0.0026` for
k=14. No Side I file appears below index 62 or above index 213. Side II shows
nothing comparable (span 265, `P = 0.8865`).

I am **not** treating this as a gate failure, because inventing a span-based
criterion the plan does not contain would be deviating from the plan. It is
reported so the planner can decide whether it changes the §7.3 independence
assumption. It is a single observation on 14 files and could easily be chance
at these counts.

---

## 2. Objection A — the §5.3 speed derivation is degenerate on 38 files

### What the plan asks for

§5.3 specifies the derivation exactly: count transitions `T` in column 1,
`R = T/180` revolutions, `C = pi * 0.85 = 2.670354 m`, `v = R * C / 1.0 s`.

§5.3 then makes a stop condition mandatory, verbatim:

> **A sanity band is mandatory.** Derived speed must be reported as a
> distribution across all 272 training files (min, median, max). If derived
> speeds fall outside a plausible metro operating range, the derivation is wrong
> and the implementer must stop and report rather than proceed.

§5.3 also makes `v` a divisor, mechanism 3:

> Every energy/RMS feature is emitted **twice**: raw, and divided by `v` (and
> `v²` for energy-like quantities).

### What I observed

Column 1 is clean and unambiguous, which resolves the §7.4 primary ambiguity in
the plan's favour (see §4 below): `dtype=int64`, literally binary `{0, 1}`, zero
NaNs, **0 of 272 files non-binary**, so no thresholding is needed.

The derivation applied to all 272 files gives:

| Statistic | v (m/s) | v (km/h) |
|---|---|---|
| min | **0.0000** | **0.000** |
| p25 | 4.0612 | 14.620 |
| median | 9.7987 | 35.275 |
| p75 | 13.7449 | 49.482 |
| max | 19.4787 | 70.123 |
| std | 5.9493 | 21.418 |

Transition count `T`: min **0**, median 660.5, max 1313.

**38 of 272 files have `T == 0`.** Their column 1 is constant at the value `1`
for all 10,000 samples — a single distinct value, no transitions at all. Their
indices:

```
[5, 11, 13, 23, 25, 27, 38, 49, 55, 75, 78, 86, 90, 92, 138, 148, 155, 157,
 161, 162, 165, 178, 181, 184, 187, 191, 192, 196, 206, 219, 228, 229, 239,
 244, 245, 249, 252, 257]
```

All 38 are labelled **Normal** (0 Side I, 0 Side II).

A further 6 files give `0 < v < 1.0 m/s` (44 files below 1 m/s in total, again
all Normal); 60 files give `v < 3.0 m/s`. The smallest nonzero speed is
**0.0445 m/s = 0.160 km/h**, from `T` values as low as single digits. The ratio
of max to smallest-nonzero speed is **437.7**.

These are not dead files. On a sample of the `T == 0` files the vibration and
shock channels carry live signal — e.g. `Train5.csv` channels 1-4 have standard
deviations `[0.0361, 1.370, 0.0334, 1.524]`, and the array is not all-zero.
`Train11.csv` is materially identical. So the recordings exist; it is
specifically the speed channel that is flat.

### Why this stops me

Three distinct problems, none of which I may resolve myself:

1. **The sanity band fails as stated.** `v = 0` is not within any plausible
   metro operating range. §5.3 says that outcome means "the derivation is wrong
   and the implementer must stop and report". I am doing exactly that.
2. **Division by `v` is undefined.** §5.3 mechanism 3 mandates dividing by `v`
   and `v²`. On 38 files that is division by zero; on the 0.0445 m/s file it is
   multiplication by ~22.5 (and ~505 for `v²`), which would produce feature
   values orders of magnitude outside the rest of the distribution. §5.3 says
   "`eps` guards division at implausibly low `v`" — but an `eps` guard converts
   a divide-by-zero into an arbitrary huge number rather than into a meaningful
   feature, and the plan does not specify an `eps` value or a floor for `v`.
3. **The wavelength bands collapse.** §5.3 mechanism 1 sets band edges at
   `v/λ`. At `v = 0` every edge is 0 Hz and every band is empty. Even at the
   smallest nonzero speed 0.0445 m/s, all five bands fall below the Welch
   resolution: with `nperseg=2048, fs=10000` the resolution is 4.883 Hz, and the
   five bands span 0.07 Hz to 2.23 Hz — all narrower than a single bin, let
   alone two. §5.3 says out-of-range bands are set to `NaN` and imputed, so the
   mechanism exists, but at these speeds **all five bands are NaN for roughly a
   fifth of the training set**, which means the plan's primary feature family
   is entirely absent for 38+ files rather than partially degraded. That is a
   different situation from the occasional NaN §5.3 anticipated.

   For contrast, at the median speed 9.799 m/s the bands are
   `[244.97, 489.94]`, `[122.48, 244.97]`, `[61.24, 122.48]`, `[30.62, 61.24]`,
   `[15.31, 30.62]` Hz — all comfortably resolvable and all below Nyquist. At
   the max 19.479 m/s the highest band reaches 973.94 Hz, still well below the
   5,000 Hz Nyquist. So the band design is sound at normal speeds; it is
   specifically the near-zero-speed files that break it.

### What I need from the planner

A decision on how the 38 zero-speed files (and the low-speed tail) are handled.
I am not choosing between these; I am listing them so the decision is concrete:

- Are they stationary/idling recordings that should be excluded from Train
  entirely? That would change the class counts to 196/14/24 and change every
  number in §4.2, so it is plainly a planner decision.
- Should `v` be floored at some minimum for the normalisation mechanisms, and
  if so at what value, with the floor recorded as an assumption?
- Should a "speed invalid" indicator become an explicit feature, with the
  wavelength-band family NaN-imputed as §5.3 already allows?
- Is the derivation itself suspect — e.g. is a constant column 1 a sensor
  dropout rather than a genuine standstill? I cannot tell from the Info Kit,
  which says nothing about standstill or sensor failure.

Note the Test set almost certainly contains such files too. I did **not**
measure Test column 1 to check, because that would be using Test as a source of
truth for a modelling decision. Whatever is decided must be a rule `predict()`
can apply to an unseen file without knowing its label.

---

## 3. Objection B — derived speed separates the classes on its own

### What the plan asks for

§5.3, "The residual risk, stated", verbatim:

> If corrugation severity correlates with speed in the dataset (e.g. fault files
> were all recorded on a slow section), the model can learn speed as a shortcut.
> Mitigation is diagnostic and mandatory: report the derived-speed distribution
> **broken down by class** (min/median/max for Normal, Side I, Side II
> separately). **If the classes are separable on speed alone, that is a confound
> and a finding for the planner, not something the implementer patches.**

I ran the mandated diagnostic. The classes are separable on speed alone.

### What I observed

Derived speed by class, all 272 Train files:

| Class | n | min m/s | p25 | median | p75 | max m/s | median km/h |
|---|---|---|---|---|---|---|---|
| Normal | 234 | **0.000** | 2.663 | 8.130 | 12.896 | 19.479 | 29.27 |
| Side I | 14 | **9.702** | 12.725 | 12.959 | 13.567 | 18.603 | 46.65 |
| Side II | 24 | **11.690** | 13.278 | 13.997 | 15.881 | 18.514 | 50.39 |

**Not one of the 38 fault files falls below 9.70 m/s. 133 of the 234 Normal
files do.** So the rule "`v < 9.702 m/s` implies Normal" is correct on 133
Normal files and is never once wrong on a fault file, across the whole training
set.

Histogram, faults versus Normal (2 m/s bins):

```
bin m/s      Normal   Fault (Side I + Side II)
[ 0, 2)        51        0
[ 2, 4)        16        0
[ 4, 6)        42        0
[ 6, 8)         8        0
[ 8,10)        25        1
[10,12)        22        2
[12,14)        22       20
[14,16)        21        6
[16,18)        18        4
[18,20)         9        5
```

Rank statistics, speed versus Normal:

- Side I vs Normal: Mann-Whitney `p = 2.514e-04`, AUC **0.7766**
- Side II vs Normal: Mann-Whitney `p = 9.080e-08`, AUC **0.8230**

**The decisive number.** A classifier using **nothing but derived speed** — two
thresholds, no vibration data at all — reaches **macro F1 0.5089** (thresholds
12.25 and 13.00 m/s). Scored with `macro_f1` from `src/common/metrics.py`.
Against:

- always-Normal baseline B0: macro F1 **0.3083** (which reproduces §2.3's
  ~0.31 figure exactly)
- the speed-only rule: **0.5089**

That 0.51 figure is fitted on all 272 labels and is therefore an optimistic
upper bound, not a CV score — I am reporting it as what it is. But it does not
need to be a CV score to make the point: a single scalar that has nothing to do
with corrugation clears the plan's own floor baseline by ~0.20 macro F1.

### Why this stops me

§5.3 states the consequence itself: the model can learn speed as a shortcut.
The concrete danger is that the plan mandates speed as an explicit feature
(§5.3 mechanism 2: "Derived `v` is emitted as a feature in its own right"), so
the shortcut is not merely available to the model, it is **handed to it
directly**. With 14 Side I files, a tree ensemble will find a 0.78-AUC scalar
long before it finds a subtle per-side spectral contrast.

The result would be a validated macro F1 that partly measures "was this train
going fast", and §4.4's caveat applies with full force: with zero leaderboard
uploads, nothing downstream would ever catch it. The Test set's speed
distribution is unknown (§7.9 already records that its class distribution is
unknown), so a speed-shortcut model could score well in CV and fail entirely on
Test.

I also cannot tell from the data which way the causation runs, and I will not
speculate. Both readings are live:

- **Artefact:** the fault files happen to have been recorded on faster sections
  or during faster runs, and speed is pure confound.
- **Physical:** corrugation-induced vibration is only detectable above some
  speed, so low-speed recordings are labelled Normal because the fault is
  invisible, not because the rail is sound. Under this reading the 133
  sub-9.7 m/s Normal files carry unreliable labels.

These have opposite implications for what should be built, and the Info Kit
says nothing that settles it. §1.1's note that the method "is affected by train
speed" is consistent with both.

Note also that Objections A and B interact: the 38 zero-speed files are all
Normal, so whatever the planner decides about excluding or flooring them
directly changes the class-conditional speed distribution above.

### What I need from the planner

A decision on how the speed confound is handled, for example (again, listing,
not choosing):

- Whether `v` remains an explicit model feature at all (§5.3 mechanism 2), or is
  used only for band-edge derivation and normalisation.
- Whether a speed-matched or speed-stratified analysis is required, so the
  reported macro F1 is not carried by the confound.
- Whether the contrast features of §5.2, which are speed-robust by construction
  (mechanism 4), should be the *only* permitted feature family, so the shortcut
  is structurally unavailable.
- Whether a "speed-only" model should be added as a mandatory extra baseline
  alongside B0/B1, so that any learned model must be shown to beat 0.509 and not
  merely 0.308. This one seems to me the cheapest way to keep the issue visible
  whatever else is decided, but it is the planner's call, and B1's specification
  and the AC-19 configuration cap would both need amending.

---

## 4. §7.4 and §7.6 ambiguities — resolved by measurement, reported for the record

These are not objections. Both plan ambiguities are settled by Gate 1's
measurements, in the plan's favour.

### §7.6 — header strings (AC-14). RESOLVED, no conflict.

Every file carries a descriptive header, 129 columns, no BOM. Column 1 is named
`'Rotating speed'`. Columns 2-129 are named
`'<Vibration|Shock> of bearing in position <p> of car <c>'`.

First ten channel columns (1-based column index):

```
  1: 'Vibration of bearing in position 1 of car 1'
  2: 'Shock of bearing in position 1 of car 1'
  3: 'Vibration of bearing in position 2 of car 1'
  4: 'Shock of bearing in position 2 of car 1'
  5: 'Vibration of bearing in position 3 of car 1'
  6: 'Shock of bearing in position 3 of car 1'
  7: 'Vibration of bearing in position 4 of car 1'
  8: 'Shock of bearing in position 4 of car 1'
  9: 'Vibration of bearing in position 5 of car 1'
 10: 'Shock of bearing in position 5 of car 1'
```

Last six:

```
123: 'Vibration of bearing in position 6 of car 8'
124: 'Shock of bearing in position 6 of car 8'
125: 'Vibration of bearing in position 7 of car 8'
126: 'Shock of bearing in position 7 of car 8'
127: 'Vibration of bearing in position 8 of car 8'
128: 'Shock of bearing in position 8 of car 8'
```

I parsed all 128 channel headers and checked each against §5.2's index
arithmetic (`car = j//16 + 1`, `pos = (j%16)//2 + 1`, `kind = vibration if j
even`, `j = 0..127`). **All 128 match.** The ordering claimed in Info Kit §2.1
is confirmed against the actual files, so the §5.2 side mapping is valid: **64
Side I channels** (positions 1,3,5,7 — 32 vibration + 32 shock) and **64 Side
II channels** (positions 2,4,6,8 — 32 vibration + 32 shock).

Because the headers are in fact descriptive and self-documenting, §5.4's
positional-access mandate is safe but no longer the only option. I am not
changing it — the plan mandates positional access and I would not deviate — but
the planner may wish to know that a name-based assertion is now available as an
extra check.

### Shape (AC-2). Confirmed.

Every sampled file: 129 columns, 10,001 total lines = 1 header + **10,000 data
rows**, no BOM, and headers **byte-identical to `Train1.csv`**. Sampled
`Train1, Train2, Train62, Train103, Train137, Train200, Train272` and
`Test1, Test2, Test34, Test68`. Train directory contains 272 files, Test 68.

(The Test files were opened only to read the header and count lines — a schema
consistency check explicitly requested by the Gate 1 brief. No Test column
values were inspected and no Test-derived quantity informs any modelling
decision.)

### §7.4 — column 1 encoding. RESOLVED on the primary question.

- **dtype:** `int64` in every file sampled.
- **Distinct values:** `{0, 1}` — literally binary. Across all 272 files the
  per-file distinct-value count is between 1 and 2, and **zero files** contain a
  non-binary value. No NaNs.
- **Consequence:** the §5.3/§7.4 contingency "if column 1 is not binary,
  threshold at the midpoint of its observed range" **does not apply**. No
  thresholding is needed and none was done. Transitions are counted directly as
  `count(diff != 0)`.
- The files with a single distinct value are the 38 `T == 0` files of Objection
  A, all stuck at `1`.

### §7.4 sub-ambiguity — Nyquist saturation. RESOLVED: no saturation.

This is the sub-ambiguity §7.4 says "returns to the planner if saturation would
invalidate the derivation". It does not.

- Tooth passage rate (`T/2` teeth per second): min 0.0, median 330.2, max
  **656.5** teeth/s.
- At 10,000 Hz that is **15.23 samples per tooth period at the fastest file**
  and 30.28 at the median — far from the 2-sample Nyquist limit.
- Max `T` is 1313 against a hard ceiling of 10,000 transitions per second, i.e.
  **13.1% of capacity**.
- Level-run lengths confirm this directly: the fastest sampled file
  (`Train103.csv`, v = 17.28 m/s) has runs of 3 to 9 samples; `Train2.csv`
  (13.35 m/s) has runs of 2 to 12, median 11.

The square wave is comfortably oversampled at every observed speed. **The
derivation is not invalidated by saturation.** The derivation is invalidated on
the 38 zero-transition files, but by degeneracy, not by saturation — a
different cause, and the subject of Objection A.

### Speed variation across files — §5.3 normalisation does matter

Excluding the zero-speed files, derived speed ranges from 0.0445 m/s to 19.479
m/s, a ratio of 437.7; across the interquartile range alone, 4.06 to 13.74 m/s,
a factor of 3.4. **Speed varies more than enough for §5.3's speed normalisation
to matter**, and a fixed-Hz band would indeed be measuring different physical
wavelengths in different files. §5.3's core argument for wavelength-domain bands
is supported by the measurement. That part of the plan is vindicated; it is the
zero-speed tail and the class confound that are not.

Within-file speed stability is good: over 10 windows of 0.1 s, the per-file
spread `(win_max - win_min)` in transition count has median 2 and max 9, and the
per-file std of windowed counts has median 0.600 and max 2.685. Files are close
to constant speed, so §5.3 mechanism 2's acceleration-proxy feature will have
little variance to work with. Recorded as an observation, not an objection.

---

## 5. What I did not do

- **No feature extraction and no modelling.** Gate 1 was measurement only.
- **No split was run, no CV, no model fitted.** The only classifier-shaped thing
  in this document is the two-threshold speed-only rule of Objection B, which is
  a *diagnostic* computed to quantify the confound the plan told me to quantify,
  fitted on all 272 labels and reported as an upper bound, never as a validation
  result.
- **No Test data used for any decision.** Test files were read for header text
  and line count only (Task B, explicitly requested).
- **No metric hand-rolled.** Every macro F1 figure here comes from
  `macro_f1` in `src/common/metrics.py`, and `RAIL_CLASSES` was read from that
  module rather than redefined.
- **Nothing written to `src/rail/`.** All diagnostic scripts are in the session
  scratchpad.

## 6. Decision required

The §4.4.1 hard gate passes, so the fault files are not index-clustered and the
stratified split of §4.1 is not challenged on that ground.

I need the planner's decision on:

1. **Objection A** — the 38 zero-speed (and 6 near-zero-speed) Train files:
   exclude, floor, flag, or re-derive.
2. **Objection B** — the speed/class confound: whether `v` stays a feature,
   whether an extra speed-only baseline is mandated, and whether the reported
   macro F1 must be shown to be robust to it.
3. Optionally, whether the Side I index span of §1a changes anything about the
   §7.3 independence assumption.

I have not implemented a workaround for any of these and will not until the plan
is revised. This is **iteration 1**.
