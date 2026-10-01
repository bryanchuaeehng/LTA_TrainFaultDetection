# Backlog — rail corrugation

**The plan is FROZEN** as of the iteration-2 amendment pass
(`docs/plans/rail.md`, Revision log). There is no amendment 3.

Anything noticed from here — by any agent or by the user — goes in this file
rather than into the plan. An item is actioned **only if the implementer
actually trips on it**, meaning it blocks a gate or makes a plan instruction
impossible to follow. Items that are merely interesting, tidier, or
theoretically better stay here unactioned until the subsystem is complete.

The point of the freeze: methodology churn has a cost, and with zero leaderboard
uploads the thing that matters is getting one honest end-to-end number, not the
most elegant possible plan.

## Format

```
### <short title>
- **Noticed by / when:** 
- **What:** 
- **Would it block a gate?** yes / no
- **Status:** open / actioned / closed-not-actioned
```

## Items

### Side I index span (p = 0.0026)
- **Noticed by / when:** implementer, Gate 1
- **What:** Side I files span indices 62-213 only; a random 14-of-272 draw gives
  a span that small with probability 0.0026. The run-length gate passed cleanly
  (largest run 1), which rules out the strong duplicate-recording concern, but
  the span is unexplained.
- **Would it block a gate?** No. Recorded as a residual caveat in §4.4.2 and
  §7.3, mandated into the write-up by AC-27.
- **Status:** closed-not-actioned — user ruled "note as a residual caveat in the
  write-up, do not block on it."

### Speed confound likely persists into Test — write-up framing
- **Noticed by / when:** user, after Gate 1b
- **What:** Train and Test have near-identical speed composition. Low-speed
  fraction: Train 44/272 = 16.2%, Test 10/68 = 14.7%. Above `V_CUT`: Train
  ≈61% (139/228 eligible), Test 63.2% (43/68). Because the two corpora look
  alike at both ends, **the speed confound of §7.13 very likely persists into
  Test.**
- **Why it matters:** if a model partly exploits "fast ⇒ fault", that shortcut
  will still be available on Test, so **the full-set macro F1 will flatter us**.
  The restricted number (`v ≥ V_CUT`, where the shortcut is unavailable by
  construction) is the honest one. This is precisely why §4.7.5 makes the
  restricted figure the headline, and this observation is independent evidence
  for that choice.
- **Would it block a gate?** No. It changes nothing about the model, the split,
  the features or any threshold.
- **Status:** open — **for write-up framing only, NOT for the model.** User
  ruled: note, do not action. Carry into the final write-up alongside AC-25's
  headline definition.

### AC-13 side-swap invariant fails on the MODEL, not on the FEATURES
- **Noticed by / when:** implementer, Gate 2
- **What:** AC-13 run on 12 eligible Train files (4 per class) with a diagnostic
  M1 fitted on all 228 eligible files: **7 passed, 5 failed**. Every failure is
  a fault file whose mirrored copy collapses to `Normal` instead of flipping to
  the opposite side. All 4 `Normal` files passed.
- **Root cause, measured, and it matters:** the **feature extractor is exactly
  side-antisymmetric**. On `Train185.csv` and `Train150.csv`, checking every
  emitted feature: `_diff` features negate exactly (0/252 violations), `_sym`
  features are invariant (0/252), `_logratio` features negate exactly (0/96),
  and `sideI(original) == sideII(swapped)` exactly (0/252). The structural
  property §5.2 claims for the contrast design **holds perfectly.**
  The asymmetry is entirely in the **fitted classifier**: M1 learns
  side-specific coefficients from 14 Side I vs 24 Side II examples, so a
  mirrored fault vector falls outside the learned region. Predicted
  probabilities on swapped copies collapse hard rather than sitting near a
  boundary — `Train185` goes `[0.002, 0.998, 0.000]` to `[1.000, 0.000, 0.000]`.
- **Reading:** this is a symptom of 38 fault examples in a 1,104-dimensional
  space, not of a broken feature set. AC-13's wording ("a file the model
  predicts Side I is predicted Side II") tests the model; the design property it
  was motivated by (§5.2, "swapping the two sides' channels must flip a Side I
  prediction") is a property of the features, and that part is verified clean.
- **Options, all of which are PLANNER decisions and none of which I have taken:**
  symmetry augmentation (train on each file plus its mirrored copy with the
  label flipped, which would enforce the invariant and double the fault count to
  76); restricting to the antisymmetric contrast family; or accepting the
  finding and reporting the 7/12 as measured.
- **Would it block a gate?** **No.** Gate 2's deliverables all completed:
  extraction, the AC-3b counts and assertion, B0, the feature matrix shape and
  the wall-clock time. AC-13 is instrumentation and its result is *reported as
  measured*, which is exactly what the plan asks for ("partial failure is a
  reportable finding, not a silent pass"). I have not modified the features, the
  model, the split or any threshold in response.
- **Status:** open — reported at Gate 2, awaiting planner decision. Default if
  the planner does not act: carry the result into the write-up as a measured
  limitation.
- **UPDATE (deferred-AC re-run):** the re-run scores **8/12, not 7/12**. Exactly
  one file changed verdict: `Train267.csv` (Side II) went FAIL -> PASS; the
  other eleven are identical. **This is not run-to-run instability** — the model
  is deterministic under the fixed seed. The cause is the feature set:
  `model_matrix_columns()` reads `artifacts/rail/speed_correlated_features.json`
  at call time, and that file **did not exist** when Gate 2 ran AC-13 (it was
  written later, during the Gate 3 leak ruling). So Gate 2's AC-13 used the
  pre-gate **1,104** features and the re-run uses the post-gate **982**.
  Verified: `model_matrix_columns()` today returns 982, with zero
  `domwavelength` and zero speed-gated columns surviving.
  **Reading:** removing 122 speed-correlated features moved one borderline
  mirrored prediction across a decision boundary. It does not change the
  diagnosis — the extractor remains exactly antisymmetric (0 violations) and the
  asymmetry remains in the fitted classifier. Quote **8/12** as the current
  figure and note that the Gate 2 raw file records 7/12 against the older,
  wider feature set. Both raw files are retained.

### ROOT CAUSE: dimensional correctness does not imply statistical independence
- **Noticed by / when:** coordinator verification, post-Gate-2; user ruled immediately
- **What (for the write-up, one paragraph):** the `domwavelength` family computed
  `λ = v / f_centroid`. Wherever `f_centroid` is near-constant across files —
  broadband shock with no dominant spectral peak — `λ` is almost exactly
  proportional to `v`. **The metres mandate multiplied speed back in after
  mechanism 3 had divided it out.** Measured `|Spearman|` against `v` reached
  **0.9982** (`shk_median_domwavelength_sideI_m`), and a depth-3 tree on that
  single feature scored **0.3984** restricted macro F1 — beating B0 (0.2806) and
  beating raw `v` itself (0.3566), the very feature §5.3b struck. Restriction
  made it *worse*: 49 features above 0.9 correlation in the restricted domain
  versus 6 on the full set, so a full-set gate would have missed most of them.
- **The lesson, which is the quotable part:** the unit-suffix contract passed
  these columns as `_m` **correctly** — they genuinely are metres. **The suffix
  contract checks DIMENSION, which is a property of the formula; leakage is a
  property of the DATA.** A static naming rule cannot catch this. Only an
  empirical check against the data can.
- **Actioned:** `domwavelength` family dropped outright (48 features, no
  residualisation, no rescue attempt). An empirical correlation gate added:
  `|Spearman|` vs `v` on the restricted eligible training set, threshold 0.90,
  **chosen pragmatically, not derived**, applied once as a fixed feature-set
  decision on training data only (speeds, never labels), not a per-fold fit.
  It dropped a further **74**. 1,104 → 1,056 → **982 features**.
- **Would it block a gate?** No — applied and shipped.
- **Status:** actioned. Carry the paragraph above into the write-up.

### B2 threshold grid coarsened from 60 to 40 candidates (runtime)
- **Noticed by / when:** implementer, Gate 3
- **What:** §5.7a specifies exhaustive search over midpoints of the training
  portion's sorted unique speeds. At 202 distinct speeds the O(n²×6 mappings)
  search with a `macro_f1` call per candidate cost **38 s/fold = 32 min** for 50
  folds, against ~20 s total for B0+B1+M1. Reduced the candidate cap 60 → 40,
  giving 17.3 s/fold ≈ 14.4 min.
- **Effect:** still an exhaustive grid over training-portion speed midpoints,
  just coarser. Costs threshold resolution of a few hundredths of a m/s.
  Recorded because it is a deviation in grid density from the plan's wording.
- **Would it block a gate?** No.
- **Status:** open — noted, applied for runtime under the Gate 3 time budget.

### Model search ran and returned a null result — incumbent unchanged
- **Noticed by / when:** overnight run, after the plan was frozen
- **What:** eight pre-registered configurations (`docs/search-plan.md`) scored
  under nested CV. **Nested estimate of the selection procedure: 0.7558**
  (std 0.0977), below the incumbent's 0.7765. No challenger passed the
  three-condition paired test; closest was C2 at mean(d) +0.0008 on 8/50 folds.
- **Why it matters:** this is evidence the incumbent is not a lucky pick — a
  spread of regularisation strengths, a linear SVM, ridge, random forest and
  extra trees were all tried and none beat it by a margin that survives paired
  testing. Random forest was worst (0.6774, chosen 0/50 by inner selection),
  which is consistent with 38 fault examples being too few for a nonlinear
  model at 982 dimensions.
- **Would it block a gate?** No. Frozen artefacts verified byte-identical
  after the run.
- **Status:** closed-not-actioned — the incumbent stands, as the
  pre-registration requires. Full log retained at
  `artifacts/rail/search_log.json`.
