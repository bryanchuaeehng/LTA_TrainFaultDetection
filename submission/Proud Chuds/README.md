# Proud Chuds — LTA NebulaX Problem Statement 3

All four subsystems attempted: Door, ACV, Rail Corrugation, SHM.

## Contents

| Item | Status |
|---|---|
| `demo_video.mp4` — item 1 | Submitted separately, outside this folder. |
| `predictions.zip` — item 2 | Ready. Four CSVs, flat at the top level. |
| `app/` — item 3 | Ready. Streamlit app covering all four subsystems. |
| `Optional_Items/` | One write-up covering all four subsystems and the app, plus code and model per subsystem. |

The demo video was uploaded separately rather than placed in this folder.
Per §4.1 all three compulsory items must reach the organisers; only the
delivery route differs.

## predictions.zip

Generated from the organisers' held-out test inputs in `02_Datasets/*/Test/`
(§2.3: the input files are distributed unlabelled; only the answers are
withheld).

| File | Rows | Source |
|---|---|---|
| `rail_predictions.csv` | 68 | `Rail_Corrugation/Test/` — 68 files |
| `door_predictions.csv` | 38 | `Door/Test.csv` — one continuous stream, 38 detected cycles |
| `acv_predictions.csv` | 1 | `ACV/Test/acv_test_case.xlsx` |
| `shm_predictions.csv` | 16 | `SHM/Test/` — 16 files |

Rail and SHM `file_id` values were checked to match their `Test/` directory
listings exactly. Every CSV passes both `scripts/validate_submission.py` and the
app's own submission validator: header, labels, row structure, and for Door,
`start_time < end_time` with no overlapping segments.

## Running the app

```bash
cd app
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m streamlit run app.py
```

Python 3.12 or 3.13. The app is self-contained: it loads its own fitted models
from `app/artifacts/` and needs neither the training corpus nor any environment
variable. Four tabs, one per subsystem; upload, check, download. The
**Download predictions.zip** control rebuilds the archive from whatever has been
checked in the session.

## Notes on the results

- **Rail** — restricted macro F1 0.7765 (std 0.0833) over 50 folds. The
  restricted figure is the headline because train speed correlates with the
  label; see `Optional_Items/write_up.md`.
- **Door** — segmentation is exact on the training stream (all 110 cycles, zero
  boundary error), so the score rests on classification. Leave-one-out 109/110.
- **SHM** — the app currently runs the earlier ridge model
  (`artifacts/shm/model.joblib`). A newer model exists at
  `artifacts/shm/model.json` in a different format that the app's wrapper does
  not yet read. Both are included.
- **ACV** — six labelled cases only, so the output is a suspicion ranking, not a
  calibrated probability. Lower-ranked cars are not confirmed healthy.
