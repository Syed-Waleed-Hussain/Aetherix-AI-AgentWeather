"""Bounded operator checks for scheduled ingestion and retraining eligibility."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import requests

from aetherix_weather.air_quality import _AIR_QUALITY_FIELDS
from aetherix_weather.api import OpenMeteoAPIError, hourly_records, request_json
from aetherix_weather.config import AIR_QUALITY_API_URL, LATITUDE, LONGITUDE, WEATHER_API_URL
from aetherix_weather.storage import write_json_atomic
from aetherix_weather.weather import _WEATHER_FIELDS
from dashboard.data import read_object, replay, run_history

ROOT = Path(__file__).resolve().parents[1]


def fetch_recent(session: requests.Session, now: datetime) -> list[dict]:
    """Read and align the latest complete provider hours with stable UTC keys."""
    current_hour = now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = current_hour - timedelta(hours=24)
    end = current_hour - timedelta(hours=1)
    params = {
        "latitude": LATITUDE, "longitude": LONGITUDE, "timezone": "GMT",
        "start_hour": start.strftime("%Y-%m-%dT%H:%M"), "end_hour": end.strftime("%Y-%m-%dT%H:%M"),
    }

    def read(url: str, fields: tuple[str, ...], label: str) -> dict:
        body = request_json(session, url, {**params, "hourly": ",".join(fields)}, timeout=6)
        rows = hourly_records(body, label, fields, timezone_name="GMT")
        times = [row["time"].astimezone(timezone.utc) for row in rows]
        if times != [start + timedelta(hours=offset) for offset in range(24)]:
            raise OpenMeteoAPIError("Provider timestamps differ from the requested completed UTC hours.")
        return {stamp: {field: row[field] for field in fields} for stamp, row in zip(times, rows)}

    weather = read(WEATHER_API_URL, _WEATHER_FIELDS, "Weather")
    air = read(AIR_QUALITY_API_URL, _AIR_QUALITY_FIELDS, "Air quality")
    result = [
        {"city": "karachi", "timestamp_utc": stamp.isoformat(), "weather": values,
         "air_quality": air[stamp], "source": "Open-Meteo modeled hourly estimates"}
        for stamp, values in weather.items()
        if values["temperature_2m"] is not None and air[stamp]["us_aqi"] is not None
    ]
    if not result:
        raise OpenMeteoAPIError("No completed hours contain both temperature and AQI.")
    return result


def ingestion_check(now: datetime) -> dict:
    result = {"run_at_utc": now.isoformat(), "persistent_write": False, "rows": []}
    try:
        with requests.Session() as session:
            result["rows"] = fetch_recent(session, now)
    except (OpenMeteoAPIError, requests.RequestException):
        return result | {"status": "provider_unavailable_or_invalid", "row_count": 0}
    return result | {"status": "validated_ephemeral", "row_count": len(result["rows"])}


def training_gate(root: Path, *, persistent_storage_verified: bool = False) -> dict:
    dataset = root / "data/processed/features.csv"
    manifest = read_object(root / "artifacts/day5c/champion_selection.json")
    path = root / "outputs/agents/decisions.sqlite3"
    try:
        latest = next((entry for entry in run_history(path) if entry["mode"] == "local"
                       and entry["completed_at_utc"]), {})
        state = replay(path, latest["request_id"]) if latest else {}
    except (sqlite3.Error, ValueError):
        state = {}
    drift = state.get("drift", {})
    dataset_present = dataset.is_file()
    dataset_sha256 = None
    if dataset_present:
        with dataset.open("rb") as handle:
            dataset_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    validation_manifest = bool(dataset_sha256 and manifest.get("selection_split") == "validation"
                               and manifest.get("dataset_sha256") == dataset_sha256
                               and manifest.get("decisions"))
    matching_drift = bool(drift and dataset_sha256
                          and state.get("model_provenance", {}).get("dataset_sha256") == dataset_sha256)
    review_requested = matching_drift and drift.get("retraining_review_requested") is True
    if not dataset_present:
        decision = "blocked_missing_persistent_dataset"
    elif not validation_manifest:
        decision = "blocked_missing_compatible_validation_manifest"
    elif not matching_drift:
        decision = "blocked_missing_compatible_drift_evidence"
    elif not review_requested:
        decision = "no_retraining_evidence"
    elif not persistent_storage_verified:
        decision = "blocked_unverified_persistent_storage"
    else:
        decision = "manual_retraining_review_required"
    return {
        "decision": decision, "data_available": dataset_present, "validation_manifest": validation_manifest,
        "drift_status": drift.get("status", "unavailable") if matching_drift else "unavailable",
        "manual_review_requested": review_requested, "training_executed": False,
        "champion_changed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("ingestion", "training-gate"))
    parser.add_argument("--output", type=Path, help="Optional local JSON output.")
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc)
    try:
        result = ingestion_check(now) if args.mode == "ingestion" else training_gate(ROOT)
        result.setdefault("run_at_utc", now.isoformat())
        if args.output:
            write_json_atomic(args.output, result)
    except (ValueError, OSError) as exc:
        print(f"MLOps check failed ({type(exc).__name__}).", file=sys.stderr)
        return 1
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))
    return int(result.get("status") == "provider_unavailable_or_invalid")


if __name__ == "__main__":
    raise SystemExit(main())
