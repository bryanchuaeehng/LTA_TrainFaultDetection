"""Check interpretation boundaries and report integrity, independent of models."""
import pandas as pd
import pytest
from src.app.maintenance import findings, summary_html, train_view
from src.app.submission import AnalysisResult


def result(key, table, detail=None):
    frame = pd.DataFrame(table)
    names = frame.file_id.tolist() if "file_id" in frame else ["Test.csv"]
    return AnalysisResult(key, frame, {"by_file": {name: detail or {} for name in names}}, names, "digest", 0.1)


def test_low_speed_normal_is_not_a_green_rail_result():
    output = result("rail", {"file_id": ["test.csv"], "prediction": ["Normal"]}, {"low_speed_rule": [{"file_id": "test.csv", "speed_mps": 0}]})
    assert findings(output)[0]["tone"] == "review"
    assert findings(output)[0]["status"] == "Check again"
    assert output.table.iloc[0].prediction == "Normal"


@pytest.mark.parametrize("label,tone", [("Normal", "normal"), ("Side I", "attention"), ("Side II", "attention")])
def test_rail_result_status(label, tone):
    assert findings(result("rail", {"file_id": ["test.csv"], "prediction": [label]}))[0]["tone"] == tone


def test_acv_report_keeps_native_car_ids_and_relative_ranking():
    output = result("acv", {"file_id": ["test.xlsx"], "ranked_cars": ["03|01|02|04|05|06|07|08"]})
    html = summary_html({"acv": output}).decode()
    assert "Car 03" in html and "not confirmed healthy" in html
    assert "1 of 4 subsystems checked" in html and html.count("Not checked") == 3


@pytest.mark.parametrize("value", [0, 0.9, 20])
def test_shm_does_not_invent_pass_fail_thresholds(value):
    output = result("shm", {"file_id": ["test.csv"], "prediction": [value]})
    assert findings(output)[0]["status"] == "Engineering review"


def test_report_escapes_filenames_and_preserves_predictions():
    output = result("shm", {"file_id": ["<script>.csv"], "prediction": [0.0000001234]})
    original = output.csv_bytes
    html = summary_html({"shm": output}).decode()
    assert "<script>" not in html and "&lt;script&gt;.csv" in html
    assert "1.234e-07" in html and output.csv_bytes == original


def test_report_rejects_invalid_output():
    output = result("shm", {"file_id": ["test.csv"], "prediction": [float('nan')]})
    with pytest.raises(ValueError, match="Cannot create a report"):
        summary_html({"shm": output})


def test_train_callouts_escape_labels_and_include_accessible_text():
    html = train_view([("<Car>", "Inspect first", "attention")])
    assert '<Car>' not in html and '&lt;Car&gt;' in html
    assert 'aria-label="&lt;Car&gt;: Inspect first"' in html
