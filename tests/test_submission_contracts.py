"""Contract tests for the official output formats and failure boundaries."""
import io
import zipfile
import pytest
from src.app.validators import validate_csv, validate_zip
from src.app.service import analyse
from src.door.segment import _operation, OPENING_COL, CLOSING_COL
import pandas as pd

DOOR = b"start_time,end_time,prediction\n2023-7-5-0-0-0-0,2023-7-5-0-0-1-0,Normal\n"


def test_native_and_iso_door_timestamps_are_accepted():
    assert not validate_csv("door", "door_predictions.csv", DOOR)
    assert not validate_csv("door", "door_predictions.csv",
        b"start_time,end_time,prediction\n2023-07-05T00:00:00Z,2023-07-05T00:00:01Z,Normal\n")


@pytest.mark.parametrize("data", [
    DOOR.replace(b",Normal\n", b",Normal,extra\n"),
    DOOR + b"\n",
    b"start_time,end_time,prediction\n",
    DOOR.replace(b"0-0-1-0,Normal", b"0-0-0-0,Normal"),
    DOOR.replace(b",Normal", b",normal"),
    DOOR + b"2023-7-5-0-0-0-500,2023-7-5-0-0-2-0,Normal\n",
    DOOR.replace(b"2023-7-5-0-0-0-0", b"invalid"),
])
def test_invalid_door_outputs_are_rejected(data):
    assert validate_csv("door", "door_predictions.csv", data)


def test_door_filename_is_exact():
    assert validate_csv("door", "result.csv", DOOR)


def test_ambiguous_operation_does_not_silently_become_close():
    assert _operation(pd.DataFrame({OPENING_COL:[0,0], CLOSING_COL:[0,0]})) == "Ambiguous"
    assert _operation(pd.DataFrame({OPENING_COL:[1,0], CLOSING_COL:[0,1]})) == "Ambiguous"


def test_car_identifiers_are_verified_against_the_input():
    data = b"file_id,ranked_cars\ncase.xlsx,11|12|13|14|15|16|17|18\n"
    assert not validate_csv("acv","acv_predictions.csv",data,expected_cars={"case.xlsx":[str(i) for i in range(11,19)]})
    assert validate_csv("acv","acv_predictions.csv",data,expected_cars={"case.xlsx":[f"{i:02}" for i in range(1,9)]})


@pytest.mark.parametrize("ranking", ["01|02|03", "01|02|03|04|05|06|07|07",
                                    "1|02|03|04|05|06|07|08", "01,02,03,04,05,06,07,08"])
def test_invalid_rankings_are_rejected(ranking):
    assert validate_csv("acv","acv_predictions.csv",f"file_id,ranked_cars\ncase.xlsx,{ranking}\n".encode())


@pytest.mark.parametrize("value",["NaN","inf","-inf","not-a-number"])
def test_shm_requires_finite_numbers(value):
    assert validate_csv("shm","shm_predictions.csv",f"file_id,prediction\ntest01.csv,{value}\n".encode())


def test_missing_and_duplicate_rail_files_are_rejected():
    data=b"file_id,prediction\nTest1.csv,Normal\n"
    assert validate_csv("rail","rail_predictions.csv",data,expected_files=["Test1.csv","Test2.csv"])
    assert validate_csv("rail","rail_predictions.csv",data+b"Test1.csv,Side I\n")


@pytest.mark.parametrize("filename",["nested/door_predictions.csv","Train.csv"])
def test_zip_rejects_subfolders_and_unexpected_files(filename):
    target=io.BytesIO()
    with zipfile.ZipFile(target,"w") as archive:
        archive.writestr(filename,DOOR)
    assert validate_zip(target.getvalue())


def test_service_does_not_report_bad_low_speed_rail_data_as_normal():
    with pytest.raises(ValueError,match="could not be processed"):
        analyse("rail",{"Test1.csv":b"speed,vibration\n0,1\n0,2\n"})


@pytest.mark.parametrize("name",["../Test.csv","folder/Test.csv","wrong.xlsx"])
def test_upload_path_and_extension_validation(name):
    with pytest.raises(ValueError):
        analyse("door",{name:b"data"})


def test_concurrent_analysis_gets_a_friendly_retry():
    from src.app.service import _ANALYSIS_SLOT
    assert _ANALYSIS_SLOT.acquire(blocking=False)
    try:
        with pytest.raises(ValueError, match="Another analysis is running"):
            analyse("door", {"Test.csv": b"unused"})
    finally:
        _ANALYSIS_SLOT.release()
