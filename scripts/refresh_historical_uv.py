"""Refresh historical AQI/PM2.5/UV rows without re-fetching weather data."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aetherix_weather.api import OpenMeteoAPIError
from aetherix_weather.historical import HistoricalAirQualityClient
from aetherix_weather.storage import read_csv, upsert_csv, write_json_atomic


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    raw_dir = args.data_dir / "raw"
    manifest_path = raw_dir / "manifest.json"

    try:
        manifest = _read_manifest(manifest_path)
        date_range = manifest.get("requested_date_range", {})
        start = date_range.get("start")
        end = date_range.get("end")
        if not start or not end:
            raise ValueError("The raw manifest must contain the requested date range.")
        raw_path = raw_dir / "air_quality.csv"
        if not read_csv(raw_path):
            raise ValueError("raw/air_quality.csv is missing or empty.")

        dataset = HistoricalAirQualityClient().fetch(start, end)
        row_count = upsert_csv(raw_path, dataset.records)
        manifest["availability"].update(dataset.availability)
        manifest["units"].update(dataset.units)
        manifest["source_schemas"]["air_quality"] = sorted(dataset.records[0])
        manifest["stored_row_counts"]["air_quality"] = row_count
        manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["sources"]["air_quality"]["returned_grid_coordinates"] = {
            "latitude": dataset.records[0]["api_latitude"],
            "longitude": dataset.records[0]["api_longitude"],
        }
        manifest["sources"]["air_quality"]["description"] = (
            "Modeled US AQI/PM2.5 and hourly cloud-aware UV index from "
            "Open-Meteo's Air Quality API."
        )
        write_json_atomic(manifest_path, manifest)
    except (OpenMeteoAPIError, ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"Historical UV refresh failed: {exc}", file=sys.stderr)
        return 1

    uv_count = sum(row.get("uv_index") is not None for row in dataset.records)
    print(
        f"Updated {row_count} air-quality rows for {start} through {end}; "
        f"UV values returned: {uv_count}."
    )
    print("Run python -m scripts.build_features to refresh features and reports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
