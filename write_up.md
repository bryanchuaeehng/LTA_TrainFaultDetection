# Train Condition Monitoring — Project Write-Up

## Overview

We built a unified condition-monitoring application covering all four rail subsystems in
this challenge: **Door**, **ACV**, **Rail Corrugation**, and **SHM**. Maintenance staff
select a subsystem, upload sensor data, and receive plain-language findings and
recommended inspection priorities — with full prediction tables available underneath for
anyone who wants the detail. The goal was to make four structurally different modelling
problems feel like one coherent tool, rather than four disconnected scripts bolted
together behind a single UI.

## Approach by Subsystem

- **Door** — temporal segment detection plus binary classification. Cycles are
  identified via threshold-based segmentation of the continuous sensor stream, then each
  detected cycle is classified as Normal or Abnormal resistance from its motor
  current/voltage/back-EMF signature.
- **ACV** — fault localisation via ensemble ranking. All eight cars on a train are
  scored and ranked by likelihood of a refrigerant leak, combining cabin-temperature
  deviation from setpoint with any available fault-flag signals, so the output degrades
  gracefully even when a file's telemetry schema is sparser than others.
- **Rail Corrugation** — multi-class classification (Normal / Side I / Side II) from
  multi-channel axle-box vibration and shock data.
- **SHM** — regression, estimating cumulative fatigue damage from dynamic stress
  time series using rainflow cycle counting (via Fatpack) to convert raw stress signals
  into damage-relevant cycle features ahead of the regression step.

## Tech Stack

Built in Python 3.12 with **Streamlit** as the web interface. **Pandas**/**NumPy** handle
telemetry processing; **SciPy**, **scikit-learn**, and **Joblib** support feature
engineering, fitted models, and inference. **OpenPyXL** handles ACV's Excel inputs,
**Fatpack** performs rainflow cycle counting for SHM, and custom segmentation/validation
logic handles Door's cycle detection and per-subsystem input/output contract checking.
Model families span threshold-based classification (Door), ensemble ranking (ACV),
multi-class classification (Rail), and ridge regression (SHM) — chosen per task type
rather than forcing one model family across all four. The maintenance-facing
visualisation and printable summary report use HTML/CSS/SVG for a lightweight,
dependency-free train diagram. **Pytest** and **Streamlit AppTest** cover model
contracts, upload handling, error states, and downloads. The app is containerised with
**Docker** and deployed on **Google Cloud Run** via Cloud Build.

## Unique Technology & Why It's User-Friendly

Two design choices set this submission apart from a typical "four scripts behind a
form" hackathon entry: the **train visualisation** and the **choice of model
archetypes**, both aimed squarely at Ease of Use and Explainability rather than at
squeezing out marginal leaderboard points.

**The train model (visual status diagram).** Rather than returning a raw prediction
table as the primary output, results are rendered onto a lightweight SVG/HTML/CSS
diagram of the physical train itself — each car and subsystem coloured by status
(e.g. green/amber/red) with plain-language labels layered on top. This is the single
biggest usability decision in the app: a maintenance technician doesn't need to parse a
`ranked_cars` string or a confidence score to know where to look — they see the train,
see which car is flagged, and see why in one glance. Full prediction tables remain
available underneath for anyone who wants the numeric detail, so nothing is hidden, but
nothing is *required* reading either. Building this as inline SVG/HTML rather than a
charting library kept it dependency-free, fast to render even on the low-powered devices
maintenance staff might use trackside, and easy to keep visually consistent across all
four subsystems' very different underlying data.

**Model archetypes chosen for transparency, not just accuracy.** Every subsystem uses a
comparatively simple, explainable model family rather than a black-box alternative,
which matters directly for a maintenance audience that needs to trust and act on a
result, not just receive one:

- **Door** — threshold-based segmentation and classification. The decision boundary
  (what counts as "abnormal resistance") can be described in one sentence and checked
  against the raw current/voltage trace, rather than requiring a technician to trust an
  opaque score.
- **ACV** — ensemble ranking built from interpretable component scores (temperature
  deviation from setpoint, plus fault flags where available). Because each car's score
  is a transparent combination of a handful of named factors, the app can show *why* a
  car was ranked first, not just that it was.
- **Rail Corrugation** — a multi-class classifier over engineered vibration/shock
  features rather than a raw deep-learning model over the signal itself, keeping the
  Normal/Side I/Side II decision traceable back to specific signal characteristics.
- **SHM** — ridge regression. Deliberately the simplest model in the stack: fatigue
  damage estimation is safety-relevant, and a linear model with a reconstructed
  scaler/intercept is auditable end-to-end — every coefficient's contribution to the
  final damage estimate can be inspected — in a way a more complex regressor would not
  be.

Taken together, the visualisation turns four different statistical outputs into one
shared visual language, and the model choices mean that whenever a user (or a judge)
asks "why did it say that," the app has a genuine, inspectable answer rather than a
shrug.

## Metrics & Validation

Each subsystem's headline metric follows its own Info Kit exactly, since each was
already fixed and disclosed rather than left for us to choose: IoU-weighted F1 for Door,
linear rank-decay for ACV, macro F1 for Rail (so rare Side I/Side II cases are credited,
not just the dominant Normal class), and 1 − MAPE for SHM. We validated each model
against its own labelled training data before touching the held-out test files, and
specifically checked for schema drift across files within a subsystem (most notably in
ACV, where car telemetry column names differ between train/car models) so the pipeline
doesn't silently mis-map columns on an unfamiliar file.

## Key Assumptions & Design Decisions

Where the documentation left a design choice open, we made an explicit, justified call
rather than defaulting silently:
- **Door** segmentation uses gaps in the timestamp stream rather than the
  opening/closing status flags, since those flags are set on every row in the provided
  data and carry no boundary information on their own.
- **ACV** ranking falls back to temperature-deviation scoring when a file's schema
  lacks explicit fault flags, and we verified this generalises across the two telemetry
  schemas present in the training files, with one edge case documented rather than
  silently smoothed over.
- **Rail** and **SHM** splits were constructed by file/recording rather than by row, to
  avoid leaking a single physical event across train and validation.

## Challenges & How We Solved Them

The core challenge was integration, not any single model: four independent datasets,
task types, and output schemas needed to sit behind one reliable app without cross-
contaminating each other's logic. We addressed this with a shared inference layer plus
subsystem-specific validation and exact CSV formatting per the required schema. Along
the way we removed Rail's dependency on a developer's local training folder so the app
runs standalone, reconstructed SHM's scaler and intercept from the supplied training
data after the originals went missing (verifying the reconstruction against previously
committed predictions for parity), and resolved a scikit-learn version mismatch in the
saved ACV model by comparing rankings and component scores across runtimes to confirm
consistent behaviour. Large Rail batches also pushed memory usage higher than expected;
we addressed this by processing files sequentially and provisioning Cloud Run with 8 GiB
of memory. On the interface side, we simplified down to four direct subsystem tabs with
plain-language findings, accessible status colours, and a downloadable maintenance
summary report, so the tool stays usable by non-technical maintenance staff without
requiring them to interpret raw model output.
