"""Official PS3 CSV and archive schemas, shared by the UI and CLI."""
from __future__ import annotations
import csv
import io
import math
import re
import zipfile
import pandas as pd

SCHEMAS = {
    "door": ["start_time", "end_time", "prediction"],
    "acv": ["file_id", "ranked_cars"],
    "rail": ["file_id", "prediction"],
    "shm": ["file_id", "prediction"],
}
FILENAMES = {key: f"{key}_predictions.csv" for key in SCHEMAS}
LABELS = {"door": {"Normal", "Abnormal resistance"}, "rail": {"Normal", "Side I", "Side II"}}


def timestamp(value: str) -> float:
    if re.fullmatch(r"\d{4}(?:-\d{1,3}){6}", value):
        y, mo, d, h, mi, sec, ms = map(int, value.split("-"))
        if not 0 <= ms <= 999:
            raise ValueError("Milliseconds must be between 0 and 999.")
        return pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi,
                            second=sec, microsecond=ms * 1000, tz="UTC").timestamp()
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("Timestamp is missing.")
    parsed = parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")
    return parsed.timestamp()


def validate_csv(subsystem: str, filename: str, data: bytes, *,
                 expected_files=None, expected_cars=None, expected_rows=None) -> list[str]:
    if subsystem not in SCHEMAS:
        return ["Unknown subsystem."]
    errors = []
    if filename != FILENAMES[subsystem]:
        errors.append(f"Filename must be {FILENAMES[subsystem]}.")
    try:
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig")), strict=True))
    except (UnicodeError, csv.Error):
        return errors + ["File must be a valid UTF-8 CSV."]
    if not rows or rows[0] != SCHEMAS[subsystem]:
        return errors + [f"Header must be {','.join(SCHEMAS[subsystem])}."]
    if len(rows) < 2:
        errors.append("At least one prediction row is required.")
    if expected_rows is not None and len(rows) - 1 != expected_rows:
        errors.append(f"Expected {expected_rows} prediction rows, received {len(rows) - 1}.")
    seen, spans = set(), []
    for line, row in enumerate(rows[1:], 2):
        if not row or not any(field.strip() for field in row):
            errors.append(f"Line {line}: empty row.")
            continue
        # Door is exactly THREE columns. Extra fields are rejected.
        if len(row) != len(SCHEMAS[subsystem]):
            errors.append(f"Line {line}: expected exactly {len(SCHEMAS[subsystem])} fields.")
            continue
        if any(not field.strip() for field in row):
            errors.append(f"Line {line}: empty field.")
            continue
        if subsystem == "door":
            if row[2] not in LABELS["door"]:
                errors.append(f"Line {line}: invalid Door prediction.")
            try:
                start, end = timestamp(row[0]), timestamp(row[1])
                if end <= start:
                    errors.append(f"Line {line}: start_time must precede end_time.")
                else:
                    spans.append((start, end, line))
            except (ValueError, OverflowError, TypeError):
                errors.append(f"Line {line}: unparseable timestamp.")
            continue
        name, value = row
        if name in seen:
            errors.append(f"Line {line}: duplicate file_id {name!r}.")
        seen.add(name)
        if any(c in name for c in "/\\\n\r"):
            errors.append(f"Line {line}: file_id must be a source filename.")
        ext = ".xlsx" if subsystem == "acv" else ".csv"
        if not name.lower().endswith(ext):
            errors.append(f"Line {line}: file_id must include the {ext} extension.")
        if subsystem == "rail" and value not in LABELS["rail"]:
            errors.append(f"Line {line}: invalid Rail prediction.")
        if subsystem == "shm":
            try:
                if not math.isfinite(float(value)):
                    raise ValueError
            except ValueError:
                errors.append(f"Line {line}: SHM prediction must be finite numeric data.")
        if subsystem == "acv":
            cars = value.split("|")
            if len(cars) != 8 or len(set(cars)) != 8 or not all(re.fullmatch(r"\d{2}", c) for c in cars):
                errors.append(f"Line {line}: rank eight unique two-digit car IDs separated by |.")
            if expected_cars is not None and set(cars) != set(expected_cars.get(name, [])):
                errors.append(f"Line {line}: ranked cars differ from the uploaded file headers.")
    spans.sort()
    for left, right in zip(spans, spans[1:]):
        if right[0] < left[1]:
            errors.append(f"Line {right[2]}: overlapping Door segments.")
    if expected_files is not None and subsystem != "door" and seen != set(expected_files):
        errors.append("Prediction file IDs do not match every uploaded source file exactly.")
    return errors


def validate_zip(data: bytes) -> list[str]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            if not names or len(set(names)) != len(names):
                return ["Archive is empty or contains duplicate filenames."]
            errors = []
            for entry in archive.infolist():
                name = entry.filename
                if name not in FILENAMES.values():
                    errors.append(f"Unexpected archive entry: {name}.")
                    continue
                if entry.file_size > 5_000_000:
                    errors.append(f"Prediction CSV is unexpectedly large: {name}.")
                    continue
                errors.extend(validate_csv(name.removesuffix("_predictions.csv"), name, archive.read(entry)))
            return errors
    except (zipfile.BadZipFile, RuntimeError, OSError):
        return ["Invalid predictions.zip archive."]
