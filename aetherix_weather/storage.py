"""Atomic local CSV persistence and JSON manifest helpers."""

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows, returning an empty list when the file does not exist."""
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _atomic_write(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", newline="", encoding="utf-8") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def upsert_csv(
    path: Path,
    incoming_rows: list[dict[str, Any]],
    key: str = "timestamp_utc",
) -> int:
    """Atomically merge rows by key; incoming rows replace existing timestamps."""
    existing_rows = read_csv(path)
    by_key: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        value = row.get(key)
        if value:
            by_key[value] = row
    for row in incoming_rows:
        value = row.get(key)
        if not value:
            raise ValueError(f"Each row must contain a non-empty {key!r}.")
        by_key[str(value)] = dict(row)
    ordered_rows = [by_key[stamp] for stamp in sorted(by_key)]
    fieldnames: list[str] = []
    for row in ordered_rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)

    def write(handle: Any) -> None:
        if not fieldnames:
            return
        output = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        output.writeheader()
        output.writerows(ordered_rows)

    _atomic_write(path, write)
    return len(ordered_rows)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically replace a CSV dataset with the supplied rows."""
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)

    def write(handle: Any) -> None:
        if not fieldnames:
            return
        output = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        output.writeheader()
        output.writerows(rows)

    _atomic_write(path, write)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Write JSON metadata through a same-directory atomic replacement."""
    def write(handle: Any) -> None:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")

    _atomic_write(path, write)


def write_text_atomic(path: Path, value: str) -> None:
    """Atomically replace a UTF-8 text file."""
    def write(handle: Any) -> None:
        handle.write(value)

    _atomic_write(path, write)
