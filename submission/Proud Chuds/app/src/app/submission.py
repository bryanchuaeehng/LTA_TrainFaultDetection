"""Session-local inference results and deterministic official output archives."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import io
import zipfile
import pandas as pd
from src.app.validators import FILENAMES, validate_csv, validate_zip


@dataclass
class AnalysisResult:
    subsystem: str
    table: pd.DataFrame
    diagnostics: dict
    source_files: list[str]
    source_digest: str
    elapsed_seconds: float
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC"))

    @property
    def filename(self):
        return FILENAMES[self.subsystem]

    @property
    def csv_bytes(self):
        return self.table.to_csv(index=False, lineterminator="\n").encode("utf-8")

    def validation_errors(self):
        return validate_csv(self.subsystem, self.filename, self.csv_bytes,
                            expected_files=self.source_files,
                            expected_cars=self.diagnostics.get("expected_cars"))


def predictions_zip(results: dict[str, AnalysisResult]) -> bytes:
    if not results:
        raise ValueError("Generate at least one prediction before downloading the archive.")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key in FILENAMES:
            if key not in results:
                continue
            result = results[key]
            errors = result.validation_errors()
            if errors:
                raise ValueError(" ".join(errors))
            entry = zipfile.ZipInfo(result.filename, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, result.csv_bytes)
    data = buffer.getvalue()
    errors = validate_zip(data)
    if errors:
        raise ValueError(" ".join(errors))
    return data
