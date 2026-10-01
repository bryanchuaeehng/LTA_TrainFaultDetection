# Door — Plan

**Status: implemented and integrated.** QH's fitted model and inference pipeline are used in the unified Streamlit app. The 2026-09-19 integration check reproduced 110 training boundaries exactly and detected 38 official test cycles (28 Normal, 10 Abnormal resistance; 18 Open, 20 Close).

Authoritative source:
`reference/03_References/Door/Door_Subsystem_Info_Kit.md`. Where it and the
top-level specification disagree on submission requirements, the current top-level specification wins. The Info Kit defines the subsystem task and metric.

---

## 1. The headline finding: this subsystem is far easier than it looks

The Info Kit frames Door as the hard one — "find where each cycle starts and
ends *before* it can even ask whether that cycle looks normal" (§1.1), and it
warns against assuming any single column marks the boundaries (§2.2).

**Measurement contradicts that framing for the data we were actually given.**

`Train.csv` is 18,036 rows at a 20 ms sampling interval. The idle time between
door cycles **is not sampled at all** — it is absent from the file rather than
recorded as quiet rows. The stream therefore arrives pre-broken:

- 110 true cycles, **109 inter-row time gaps greater than 1 second**
- **100.0%** of rows fall inside a labelled cycle; there is no "between cycles"
  data to segment away

Splitting the stream wherever consecutive rows are more than a threshold apart
reproduces the ground truth **exactly**:

| Gap threshold | Segments found | True |
|---|---|---|
| 0.2 s | 110 | 110 |
| 0.5 s | 110 | 110 |
| 1.0 s | 110 | 110 |
| 2.0 s | 110 | 110 |

At a 1.0 s threshold, **maximum start-boundary error 0.0000 s and maximum
end-boundary error 0.0000 s across all 110 cycles.** Not approximately — every
boundary is identical to the answer key.

**Consequence for the metric.** IoU is 1.000 on every matched segment, so
IoU-weighted F1 collapses to ordinary binary F1 on the `status` label:

```
soft_recall    = (#matched) / (#true)        [since every IoU = 1]
soft_precision = (#matched) / (#predicted)
score          = harmonic mean               = plain F1
```

**The entire Door score is therefore a classification problem, not a
segmentation one.** Effort spent on clever change-point detection is effort
spent on a solved sub-problem.

### The risk this creates, stated plainly

This conclusion is measured on Train only. If `Test.csv` were built
differently — idle periods actually sampled, or cycles butted together with no
gap — gap-splitting would fail and the score would collapse. **This must be
checked on Test before submitting**, and it is checkable without labels: count
the gaps and confirm the implied cycle count is plausible. `Test.csv` is 6,253
rows, about a third of Train, so roughly 35-40 cycles are expected. That check
is a declared, labels-free structural inspection of the input, the same kind
already used for Rail's Gate 1b.

---

## 2. Task, metric, output

**Task.** Segment a continuous stream into door-open/close cycles, then
classify each as `Normal` or `Abnormal resistance` (Info Kit §3).

**Metric.** IoU-weighted F1, defined in Info Kit §4.2. Implemented once in
`src/common/metrics.py` as `door_iou_f1`, never inlined. Given §1 above it will
in practice equal binary F1, but the implementation must be the real
IoU-matching metric — greedy highest-IoU-first, one-to-one, same-label-only —
so that a Test set which *doesn't* segment cleanly is scored honestly rather
than optimistically.

**Output schema**, from `reference/04_Example_Submission/door_predictions.csv`:

```
start_time,end_time,prediction
2023-7-5-0-0-0-0,2023-7-5-0-0-4-500,Normal
```

Exactly three columns, **no `file_id`**, one row per predicted segment.
Timestamps in the dataset's native
`Year-Month-Day-Hour-Minute-Second-Millisecond` form (accepted per §3), echoed
from the input rather than reconstructed, so no formatting drift is possible.

---

## 3. The classification signal

Per-cycle features were computed and tested univariately against the label.

| Feature | AUC vs abnormal |
|---|---|
| **`cur_sum`** — motor current summed over the cycle | **0.9837** |
| `cur_mean` | 0.8258 |
| `vol_mean` | 0.7496 |
| `n_rows` / duration | 0.5850 |
| `cur_std` | 0.4237 |
| `cur_p90` | 0.4429 |
| `cur_max` | 0.3735 |

**Integrated current is the signal, and this is physically what abnormal
resistance means:** the motor does more total work to move the door against an
obstruction. Note that *peak* current is AUC 0.37 — **below chance** — so a
naive "spike detector" is not merely weak, it points the wrong way. That is
almost certainly the trap a first attempt falls into.

`cur_sum` is not a duration artefact: `n_rows` alone is AUC 0.585, and
`corr(cur_sum, n_rows) = -0.28`, i.e. slightly *negative*.

**Split by operation.** Open and Close are mechanically different (gravity,
latch engagement, seal compression), and their current profiles differ. Within
each operation, `cur_sum` separates the classes **perfectly**:

| Operation | n | Normal median | Abnormal median | AUC |
|---|---|---|---|---|
| Open | 55 | 92,044 | 117,519 | **1.0000** |
| Close | 55 | 78,477 | 111,432 | **1.0000** |

Globally the ranges overlap slightly (normal max 94,041 vs abnormal min
90,126), which is exactly what a per-operation threshold removes.

**Operation is inferred, not given.** The Info Kit says operation is
"informational only, not something you need to predict" (§2.1), and `Test.csv`
carries no labels — but `Door is opening` / `Door is closing` are *columns in
the stream*, so operation is read directly from the data at inference time, not
predicted.

---

## 4. Validation design

**Contiguous holdout, never random rows.** The data is a temporal stream and
cycles are ordered in time; a random split would let a threshold be tuned on
cycles temporally interleaved with the ones it is scored on. The unit of
splitting is the **whole cycle**, and splits are made at cycle boundaries in
stream order.

**Rolling-origin evaluation.** A single holdout on 110 cycles with 30 abnormal
is a small sample, so validate over five successive contiguous folds, each
training on everything before a cut point and testing on the next ~10% of
cycles. Already run as a feasibility check:

| Fold | Train | Test | Abnormal in test | F1 |
|---|---|---|---|---|
| 0 | cycles 0-54 | 55-66 | 4 | 1.0000 |
| 1 | 0-65 | 66-77 | 3 | 0.8571 |
| 2 | 0-76 | 77-88 | 4 | 1.0000 |
| 3 | 0-87 | 88-99 | 4 | 1.0000 |
| 4 | 0-98 | 99-109 | 4 | 1.0000 |

**Mean 0.9714, worst fold 0.8571.** Report the distribution, never a single
number, and never the best fold. The 0.857 fold is the honest one: it is where
the global class overlap bites, and it is the reason for the per-operation
split rather than a single global threshold.

### Results as implemented — and why the perfect folds are not the headline

With the per-operation split in place, **all five rolling-origin folds score
1.0000**. That is not the number to quote, and the reason matters.

Per-operation, the classes are **linearly separable with a gap**:

| Operation | Normal max | Abnormal min | Empty gap |
|---|---|---|---|
| Close | 86,678 | 90,126 | 3,448 (3.9% of mean) |
| Open | 94,041 | 106,315 | 12,274 (12.4% of mean) |

The threshold sits in empty space, which is why forward-only validation never
misclassifies anything.

**Leave-one-out finds the case forward validation cannot.** Refitting with each
cycle held out in turn: **109/110 correct, one failure** — cycle index 4, a
Close at `current_sum` 90,126, which is precisely the abnormal minimum sitting
nearest the normal maximum. Rolling-origin never catches it because index 4 is
always inside the training portion.

**Reversed-order validation** (train on late cycles, test on early ones) scores
0.9818 / 0.9848 / 0.9870 at three cut points.

**So the honest range is ~0.98-0.99, not 1.0.** The perfect rolling-origin
folds are an artefact of where the single hard case happens to sit in time.
Both figures are reported; the leave-one-out result is the one that should
appear in the write-up.

**The threshold is fitted inside the training portion only**, per fold. No
threshold may be chosen by looking at the fold it is scored on.

---

## 5. Approach

**Baseline B0 — always `Normal`.** 80/110 cycles are Normal. Scores F1 = 0 on
the abnormal class. States the floor.

**B1 — the primary model: per-operation threshold on `cur_sum`.**
For each operation in {Open, Close}, fit one scalar threshold on the training
cycles by maximising F1, and predict `Abnormal resistance` above it. Two
parameters total, both fitted in-fold.

This is deliberately the whole model. With 30 abnormal cycles and a feature
that achieves AUC 1.0000 within operation, a learned model has nothing to add
and a great deal of variance to introduce.

**B2 — regularised logistic regression** on a small feature set (`cur_sum`,
`cur_mean`, `vol_mean`, `emf_mean`, duration, operation) as a check that the
simple rule is not leaving signal behind. **It ships only if it beats B1 on the
rolling-origin folds by a margin that survives a paired comparison** — the same
bar used for Rail. Expected outcome: it does not.

---

## 6. Acceptance criteria

1. Gap-splitting reproduces all 110 Train cycles with zero boundary error.
2. The same split runs on `Test.csv`; row, cycle, label and operation counts are reported. Implausible giant or incomplete cycles require investigation. No expected cycle count is enforced.
3. `door_iou_f1` in `src/common/metrics.py` reproduces the Info Kit's §4.2
   definition, verified on hand-built cases including: perfect match, wrong
   label with perfect overlap (must score 0), partial overlap, and a
   one-to-one contention case where greedy highest-IoU-first matters.
4. Rolling-origin F1 reported as **per-fold scores, mean, and minimum** — never
   a point estimate.
5. Thresholds differ across folds in the printed output, demonstrating in-fold
   fitting.
6. `predict()` returns the exact three-column schema; the CLI is logic-free.
7. Output passes a `validate_submission.py` Door mode (implemented: exactly three columns, valid labels, parseable timestamps, `start_time < end_time`, no overlapping segments, and no empty rows).
8. Peak current (`cur_max`) is **not** used as a primary discriminator, and the
   write-up states why — it is below chance.

---

## 7. Ambiguities and assumptions

**7.1 — The Info Kit's framing does not match the data.** §1.1 and §1.2 present
segmentation as the central difficulty. Measured, it is exact and trivial. Two
readings: either the provided Train/Test are a simplified slice of a harder
production stream, or the framing is aspirational. **This should be asked of
the organisers**, because if the real held-out stream is sampled continuously,
a gap-splitter fails completely.

**7.2 — Stream-format assumption.** The current Info Kit states that Test is built in the same way. The official test structural check passed with 38 cycles; this does not establish generalisation to unrelated streams.

**7.3 — 30 abnormal cycles is a small sample.** A perfect within-operation AUC
on 15 abnormal Opens and 15 abnormal Closes is encouraging, not conclusive. The
reproduced rolling-origin minimum is 1.000 across five small folds. This is validation evidence, not a guaranteed test score.

**7.4 — Header documentation reviewed.** Controller switch definitions were read during integration. The existing opening/closing flags are retained. Tied flags produce an explicit Ambiguous operation and use the committed global fallback threshold.

**7.5 — Single door, single day.** The whole stream is 2023-07-05, 00:00 to
01:10. Info Kit §1.2 warns that "data distributions differ among doors". The
threshold is fitted on one door's data; generalisation across doors is untested
and untestable with what we have.
