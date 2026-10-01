# Proud Chuds — Problem Statement 3

Four subsystems, one app. **Zero of our five leaderboard uploads were used** —
every number below comes from local validation, which meant the validation
itself had to be trustworthy. That is what most of this write-up is about.

## In sixty seconds

- **Rail:** restricted macro F1 **0.7765** (std 0.0833, 50 folds). We report the
  *harder* of our two numbers, because train speed correlates with the label and
  the easier number flatters us.
- **Door:** segmentation turned out to be exact — all 110 training cycles
  recovered with zero boundary error — so the score is really classification.
  Leave-one-out 109/110.
- **ACV:** 6 labelled cases total, so no learned model is defensible. A
  rule-based ranking puts the faulty car first in all 5 leave-one-case-out
  folds.
- **SHM:** the labels come from a disclosed procedure, so an analytical
  reimplementation is the baseline rather than a target; train MAPE 0.0047.
- **We found two feature leaks by measuring, not reviewing.** Both were
  dimensionally correct. One correlated with speed at 0.998.
- **The app shows how close each call was to its threshold**, so a technician
  can tell a +36% exceedance from a +0.04% one.

---

## Rail corrugation

**The data contains a trap.** No fault file in training was recorded below
9.70 m/s, while 133 of 234 Normal files were. A model using nothing but speed
scores 0.509 macro F1 against an always-Normal floor of 0.308. It could look
respectable while learning nothing about corrugation — and with no leaderboard
feedback, nothing would have told us.

We responded three ways. Speed still sets the wavelength band edges, because
corrugation excites vibration at `f = v/λ` and a fixed frequency band is
physically wrong. Raw speed is **barred** as a classifier input. And the
headline is measured on the **restricted domain** — only files at or above
9.70 m/s, where every class spans the same speed range and the shortcut is
unavailable by construction.

| Baseline | Restricted macro F1 | What it rules out |
|---|---|---|
| Always Normal | 0.2798 | That imbalance alone produces a good score |
| Physics rule | 0.5381 | That the learned model adds nothing |
| **Speed only** | **0.3078** | **The confound itself** |
| **Our model** | **0.7765** | — |

The speed-only baseline is the important one: it scores 0.3996 on the full set
but collapses to **0.3078** once restricted — essentially the always-Normal
floor. That is the restricted domain doing its job. Had it stayed high, the
whole headline would have been suspect.

**The leaked features.** We wrote a rule barring any feature whose units carry a
per-second term, which required expressing spectral centroid as a wavelength in
metres. Dimensionally correct — and wrong. The conversion is `λ = v/f_centroid`,
so wherever the centroid is roughly constant, **dividing by it multiplies speed
back in**. One such feature correlated with speed at **0.9982**, and a depth-3
tree on that single feature beat raw speed itself. A second family, normalised
by `v²`, over-corrected and reintroduced speed with inverted sign.

> A unit contract checks a property of the *formula*. Leakage is a property of
> the *data*. No static rule catches this; only measuring against the data does.

We dropped 122 features and added an empirical gate (|Spearman| vs speed > 0.90,
chosen pragmatically — the highest surviving correlation is 0.8995).

**We then tried to beat the model and could not.** Eight configurations,
pre-registered before running, scored under nested cross-validation. The nested
estimate came out at **0.7558 — below** the incumbent's 0.7765, and no challenger
passed a three-condition paired test. Worth noting: the best *per-configuration*
score was 0.7773, marginally above the incumbent. Quoting that would mean
selecting on the same folds used to measure it; the nested estimate, which does
not, lands lower than both.

**Honest limits.** Side I rests on 14 files; its per-class F1 is 0.565 and is
0.0 on 2 of 50 folds. Fold scores range 0.575–0.946. Two training file pairs are
byte-identical duplicates, so 226 distinct recordings, not 228.

## Door

The Info Kit frames finding cycle boundaries as the central difficulty. It is
not, for the data given: the idle time between cycles **is not sampled**, so the
stream arrives pre-broken. Splitting on inter-row time gaps recovers all 110
training cycles with **zero boundary error**, stable anywhere from 0.2 s to
2.0 s. Every IoU is 1.000, so IoU-weighted F1 collapses to plain binary F1 and
the whole score is classification.

The signal is **integrated motor current** — total work done against
resistance — at AUC 0.9837, and 1.0000 within each operation type. Note that
*peak* current is AUC 0.3735, **below chance**: a spike detector points the
wrong way. The model is two numbers, one threshold each for Open and Close.

Rolling-origin validation over five contiguous folds scores 1.0000 throughout,
but that is not the number we quote. **Leave-one-out finds 109/110** — the one
failure being the abnormal cycle nearest the normal maximum, which forward-only
validation can never catch because it sits early in the stream. Reversed-order
validation gives 0.982–0.987. **The honest range is ~0.98–0.99.**

## ACV

Six labelled cases exist. That cannot support a learned model, and saying so is
the first honest step. The approach is rule-based, using components from the
Info Kit, and validated leave-one-case-out: **the faulty car is ranked first in
all 5 evaluated folds.**

The sixth, `acv_case_04`, is **deliberately excluded** — it ships with a richer
schema than the other five (33 MB against roughly 2 MB) and sits outside the
fitted model's validated scope. Including it would mean either extending the
model to a schema we could not validate on one example, or silently scoring it
with a model that does not fit it. Both are worse than leaving it out and saying
so. It does mean the ACV result rests on **five** cases, which is thin, and a
perfect score on five is encouraging rather than conclusive.

The output is a suspicion ranking, not a calibrated probability, and
lower-ranked cars are explicitly *not* confirmed healthy.

## SHM

The Info Kit discloses that the labels were generated by rainflow counting plus
Miner's rule. That makes an analytical reimplementation the **baseline**, not an
achievement — a correct reimplementation should reproduce the labels almost
exactly, and a learned model that merely matches it has probably found a leak
rather than out-modelled the definition. Our ridge model over log-domain cycle
histogram features reaches a train MAPE of 0.0047. There is no supplied pass/fail
limit, so the app reports the estimate and asks for engineering review rather
than inventing a threshold.

## The app

Four models behind one interface a technician can use without knowing they
exist. The subsystems read genuinely unrelated inputs — 68 rail files of 129
columns, one continuous door stream, 16 headerless SHM columns, one ACV
workbook — so there are four tabs. What is unified is the output: one
`predictions.zip`, validated before it is written.

**The interface shows evidence, not just verdicts.** Our door model is a
threshold, and on the test stream flagged cycles range from **+0.04% to +36.5%**
over it. One scored 88,438 against a threshold of 88,402 — 36 units out of
88,000. Presented as a list of labels, that looks identical to the +36.5% case,
and a technician dispatched on it is acting on a coin flip. So cycles are sorted
strongest-evidence-first and tagged `Clear` or `Borderline`, and the 8 cycles
that are *Normal but within 5% of the line* get their own "worth a look" list
rather than vanishing.

**What we refuse to say** matters as much. The margin is a distance from a
fitted threshold, never presented as a probability — there is a test asserting
the wording never drifts into implying calibration. Rail distinguishes "no
corrugation detected" from "too slow to tell". ACV grey means lower priority,
not healthy. SHM asks for review because no limit was supplied. Every status
carries a text label and a glyph, so nothing depends on telling red from green.

77 automated tests cover the scoring metrics' exact matching rules and those
wording guarantees.

## Open questions for the organisers

**Were fault and normal rail recordings collected under matched speed
conditions?** No fault file sits below 9.70 m/s while 133 Normal files do. Two
readings have opposite implications: an acquisition artefact, or corrugation
genuinely being undetectable below some speed — in which case 133 slow "Normal"
labels mean "not detected here" rather than "inspected and sound".

**Are the duplicate rail file pairs intentional?** `Train107`/`Train115` and
`Train165`/`Train187` are byte-identical.

**Is the Door framing aspirational?** Segmentation is presented as the core
difficulty but is exact on the data supplied. If the real stream is sampled
continuously, a gap-splitter fails completely.

One discrepancy we resolved locally: the Rail Info Kit's scoring section cites
"~9 Side I against ~190 Normal", while its dataset section and the label file
both say 234/14/24. We used the label file.

---

**The short version.** Most of the work went into not overstating what four
models know: reporting the harder number where a confound exists, finding two
leaks by measurement after review had passed them, failing to beat our own
model and saying so, and building an interface that shows how sure each call is.
A model that is right 98% of the time and an interface that hides which 2% is a
worse tool than the same model with its uncertainty on the surface.
