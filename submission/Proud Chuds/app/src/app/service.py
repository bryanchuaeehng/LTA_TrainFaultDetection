"""Shared application inference service."""
from __future__ import annotations
import hashlib
import importlib
import logging
from functools import wraps
from threading import BoundedSemaphore
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import pandas as pd
from src.app.submission import AnalysisResult

LOG = logging.getLogger(__name__)
SUBSYSTEMS = ("door", "acv", "rail", "shm")
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_BATCH_BYTES = 1400 * 1024 * 1024
_ANALYSIS_SLOT = BoundedSemaphore(1)


def one_analysis_at_a_time(function):
    """Keep expensive model work bounded while uploads/WebSockets stay usable."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if not _ANALYSIS_SLOT.acquire(blocking=False):
            raise ValueError("Another analysis is running on this server. Please try again shortly.")
        try:
            return function(*args, **kwargs)
        finally:
            _ANALYSIS_SLOT.release()
    return wrapped


def model_status(subsystem: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(f"src.{subsystem}.predict")
        loader = getattr(module, "load_model", None) or getattr(module, "_load_model")
        loader()
        return True, "Operational"
    except Exception:
        LOG.exception("Cannot initialise %s backend", subsystem)
        return False, "Model unavailable"


def source_digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(files[name]).digest())
    return digest.hexdigest()


@one_analysis_at_a_time
def analyse(subsystem: str, files: dict[str, bytes], progress=None) -> AnalysisResult:
    started = perf_counter()
    if subsystem not in SUBSYSTEMS:
        raise ValueError("Choose one of the four supported subsystems.")
    if not files:
        raise ValueError("Upload a telemetry file first.")
    if subsystem == "door" and len(files) != 1:
        raise ValueError("Door requires one continuous-stream CSV.")
    if len(files) > 128 or sum(len(data) for data in files.values()) > MAX_BATCH_BYTES:
        raise ValueError("Use at most 128 files and 1,400 MB per analysis.")
    extension = ".xlsx" if subsystem == "acv" else ".csv"
    for name, data in files.items():
        if name != Path(name).name or any(char in name for char in "\\/\0\n\r") or name in (".", ".."):
            raise ValueError("Upload files with plain filenames, without folder paths.")
        if Path(name).suffix.lower() != extension:
            raise ValueError(f"{subsystem.upper()} requires {extension} telemetry files.")
        if not data or len(data) > MAX_FILE_BYTES:
            raise ValueError(f"{name}: file is empty or exceeds the 64 MB upload limit.")
    backend = importlib.import_module(f"src.{subsystem}.predict")
    frames, by_file, expected_cars = [], {}, {}
    with TemporaryDirectory(prefix=f"ps3-{subsystem}-") as temporary:
        for i, (name, data) in enumerate(files.items(), 1):
            path = Path(temporary) / name
            path.write_bytes(data)
            try:
                result = backend.predict(path)
                detail = result.attrs.get("diagnostics", {})
                if detail.get("errors"):
                    raise ValueError("This file could not be processed normally. Check the Rail CSV shape and sensor values.")
                by_file[name] = detail
                if subsystem == "acv":
                    expected_cars[name] = detail["car_ids"]
                result.attrs = {}
                frames.append(result)
            except ValueError as exc:
                raise ValueError(f"{name}: {exc}") from exc
            except Exception as exc:
                LOG.exception("Inference failed for %s / %s", subsystem, name)
                raise ValueError(f"{name}: analysis failed. Check that this is an official-format {subsystem.upper()} telemetry file.") from exc
            finally:
                path.unlink(missing_ok=True)
            if progress:
                progress(i, len(files), name)
    output = AnalysisResult(
        subsystem, pd.concat(frames, ignore_index=True),
        {"by_file": by_file, "expected_cars": expected_cars} if subsystem == "acv" else {"by_file": by_file},
        list(files), source_digest(files), perf_counter() - started,
    )
    errors = output.validation_errors()
    if errors:
        raise ValueError("Prediction output failed validation: " + " ".join(errors))
    return output
