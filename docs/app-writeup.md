# Why this app

Four models behind one interface a maintenance technician can use without
knowing any of them exist.

The models are the easy part to talk about. What follows is about the harder
part: turning four numeric outputs into something a person can act on at 6am
without misleading them.

---

## The problem with a prediction

A model says `Abnormal resistance`. A technician reads that and walks to a door.

But our door model is a threshold on integrated motor current, and on the
official test stream the flagged cycles range from **+0.04% to +36.5%** over
that threshold. One cycle scored 88,438 against a threshold of 88,402 — a
margin of 36 units out of 88,000.

Presented as a list of labels, that cycle looks exactly like the +36.5% one. A
technician dispatched on it is acting on a coin flip and doesn't know it.

**So the app shows the evidence, not just the verdict.** Cycles to inspect are
sorted strongest-first and tagged `Clear` or `Borderline`:

| Cycle | Movement | Over threshold | Evidence |
|---|---|---|---|
| 34 | Close | +36.5% | Clear |
| 19 | Open | +30.0% | Clear |
| … | | | |
| 22 | Close | **+0.04%** | **Borderline** |

> *2 of these sit within 5% of the threshold. Verify those before raising work.*

The work order changes: start at cycle 34, treat the last two as verify-first.
Same model, same submitted CSV, materially better decision.

## The near-misses nobody was shown

The same stream has **8 cycles labelled `Normal` that sit within 5% of the
threshold** — the closest at −0.08%. Under a pure label view they are invisible.

The app lists them under **Worth a look**, explicitly *not* as faults:

> Not faults. Listed because they are close enough that a check while you are at
> the door costs little.

A threshold is a line drawn through a continuum. Showing only which side of it
each cycle landed on throws away the information that some landed almost exactly
on it.

## What we refuse to say

The margin is a distance from a fitted threshold. It is **not** a calibrated
probability, and the app never presents it as one — no "87% confident", no
percentage that implies a likelihood we cannot back. There is a test asserting
the wording never drifts into that claim.

The same discipline runs through the rest of the interface:

- **Air conditioning** ranks all eight cars by suspicion. Lower-ranked cars are
  shown in grey and captioned *"lower priority, not confirmed healthy"* — a
  ranking is not a clean bill of health.
- **Rail** distinguishes *"no corrugation detected"* from *"the train was moving
  too slowly to tell"*. The second is shown as **Check again**, not as a pass.
  The submitted CSV still says `Normal`, as the format requires, and the app
  says plainly that the classifier never ran.
- **Structural health** returns a cumulative-damage estimate and asks for
  engineering review, because no pass/fail limit was supplied with the data.
  Inventing one would be the easiest way to make the app look decisive and the
  fastest way to make it wrong.
- **Door** recordings do not identify which car or door they came from, and the
  app says so rather than implying a location it cannot know.

Every one of those is a place where a more confident-looking app would be a
worse one.

## Accessibility is not a colour choice

Every status carries a **text label and a glyph** alongside its colour — `✓`,
`!`, `?`, `—`. Nothing in the interface depends on distinguishing red from green,
which matters in a depot and matters for around one in twelve men.

The train and coach diagrams are SVG with `role="img"` and descriptive
`aria-label`s. Focus rings are visible on every interactive element.
`prefers-reduced-motion` is respected. The layout works down to phone width.

## One archive, correctly built

The four subsystems read genuinely unrelated inputs — 68 rail CSVs of 129
columns, one continuous door stream, 16 headerless SHM columns, one ACV
spreadsheet. There is no single upload that could serve all four, so the app
does not pretend otherwise: four tabs, four uploads.

What it *does* unify is the output. **Download predictions.zip** produces exactly
what the specification asks for — one archive, one CSV per subsystem attempted,
flat at the top level, no subfolders — and validates every file before writing
it: header, labels, timestamp ordering, no overlapping segments.

That validation also protects the technician. If a result fails its schema
check, the app says so and keeps working, rather than handing over an archive
that would be rejected.

## Built to be checked

- **77 automated tests**, including the scoring metric's exact matching rules
  and the wording guarantees above.
- **Session integrity**: changing, removing or duplicating an input invalidates
  its result rather than leaving a stale one on screen. Other subsystems keep
  theirs.
- **A status strip** showing what has been checked and what is outstanding, so
  nobody submits three subsystems thinking they submitted four.
- **A printable summary report** with source filenames, check times, findings
  and actions — the thing that actually gets handed to a supervisor.

---

## The short version

Most of the work in this app went into *not* overstating what four models know.
It shows how close each call was, names the near-misses, distinguishes "nothing
found" from "could not tell", refuses to convert a threshold distance into a
fake probability, and never relies on colour alone to say something important.

A model that is right 98% of the time and an interface that hides which 2% is a
worse tool than the same model with its uncertainty on the surface.
