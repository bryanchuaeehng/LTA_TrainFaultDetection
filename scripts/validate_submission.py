"""Validate official PS3 CSV files or predictions.zip with the same UI checks."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.app.validators import FILENAMES, validate_csv, validate_zip


def validate(path: Path, expected_rows: int | None = 68) -> list[str]:
    path = Path(path)
    if not path.is_file():
        return [f"File does not exist: {path}"]
    return validate_csv("rail", path.name, path.read_bytes(), expected_rows=expected_rows)


def validate_door(path: Path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        return [f"File does not exist: {path}"]
    return validate_csv("door", path.name, path.read_bytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--subsystem", choices=["auto", *FILENAMES], default="auto")
    parser.add_argument("--expect-rows", type=int)
    parser.add_argument("--allow-any-rows", action="store_true")
    args = parser.parse_args(argv)
    if not args.path.is_file():
        errors = [f"File does not exist: {args.path}"]
    elif args.path.name == "predictions.zip":
        errors = validate_zip(args.path.read_bytes())
    else:
        subsystem = args.subsystem if args.subsystem != "auto" else args.path.stem.removesuffix("_predictions")
        rows = args.expect_rows
        if subsystem == "rail" and rows is None and not args.allow_any_rows:
            rows = 68
        errors = validate_csv(subsystem, args.path.name, args.path.read_bytes(),
                              expected_rows=None if args.allow_any_rows else rows)
    if errors:
        print("INVALID\n" + "\n".join(errors))
        return 1
    print(f"VALID: {args.path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
