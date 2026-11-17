"""Persist historical forecast probabilities separately from observation data."""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from aetherix_weather.api import OpenMeteoAPIError
from aetherix_weather.config import LATITUDE, LONGITUDE, TIMEZONE
from aetherix_weather.historical import (
    HistoricalForecastClient,
    latest_full_historical_day,
    validate_date_range,
)
from aetherix_weather.dataset import add_historical_forecast_information
from aetherix_weather.storage import (
    upsert_csv,
    write_json_atomic,
    write_text_atomic,
)

from .backfill import _read_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    if bool(args.start_date) != bool(args.end_date):
        parser.error("--start-date and --end-date must be provided together.")

    raw_dir = args.data_dir / "raw"
    try:
        base_manifest = _read_manifest(raw_dir / "manifest.json")
        if args.start_date:
            start_text, end_text = args.start_date, args.end_date
        else:
            requested = base_manifest.get("requested_date_range", {})
            if requested.get("start") and requested.get("end"):
                start_text, end_text = requested["start"], requested["end"]
            else:
                latest = latest_full_historical_day()
                start_text = (latest - timedelta(days=89)).isoformat()
                end_text = latest.isoformat()
        start, end = validate_date_range(
            start_text, end_text, latest_allowed=latest_full_historical_day()
        )
        dataset = HistoricalForecastClient().fetch(start, end)
        output_path = raw_dir / "historical_forecast_precipitation_probability.csv"
        row_count = upsert_csv(output_path, dataset.records)
        records = dataset.records
        values = [
            row["forecast_precipitation_probability_percent"]
            for row in records
            if row["forecast_precipitation_probability_percent"] is not None
        ]
        metadata = [
            json.loads(row["forecast_model_metadata"]) for row in records
        ]
        model_identity_available = any(
            any(value is not None for value in item.values()) for item in metadata
        )
        issued_at_available = any(
            row.get("forecast_issued_at_utc") is not None for row in records
        )
        run_metadata_available = any(
            row.get("forecast_run_id") is not None for row in records
        )
        lead_time_available = any(
            row.get("forecast_lead_hours") is not None for row in records
        )
        returned_range = {
            "start": records[0]["timestamp_local"][:10] if records else None,
            "end": records[-1]["timestamp_local"][:10] if records else None,
        }
        coordinates = {
            "latitude": records[0].get("forecast_api_latitude") if records else None,
            "longitude": records[0].get("forecast_api_longitude") if records else None,
        }
        issue_missing_reason = (
            "The Historical Forecast API returns a stitched continuous series but "
            "does not return per-value model run, issuance time, or lead time. "
            "Therefore these probabilities are kept separate and excluded from "
            "model inputs and observed-rain targets."
        )
        report_summary = {
            "status": dataset.availability["forecast_precipitation_probability_percent"],
            "record_count": len(records),
            "stored_record_count": row_count,
            "probability_values_available": len(values),
            "probability_values_missing": len(records) - len(values),
            "valid_time_range_local": returned_range,
            "model_selection": "default_best_match",
            "model_identity_available": model_identity_available,
            "run_metadata_available": run_metadata_available,
            "forecast_issue_time_available": issued_at_available,
            "lead_time_available": lead_time_available,
            "used_as_feature": False,
            "exclusion_reason": issue_missing_reason,
        }
        forecast_manifest = {
            "city": "Karachi, Pakistan",
            "requested_coordinates": {"latitude": LATITUDE, "longitude": LONGITUDE},
            "returned_grid_coordinates": coordinates,
            "timezone": TIMEZONE,
            "source": "Open-Meteo Historical Forecast API",
            "documentation": "https://open-meteo.com/en/docs/historical-forecast-api",
            "endpoint": "https://historical-forecast-api.open-meteo.com/v1/forecast",
            "requested_hourly_variables": ["precipitation_probability"],
            "units": dataset.units,
            "requested_date_range": {"start": start.isoformat(), "end": end.isoformat()},
            "returned_date_range_local": returned_range,
            "rows": len(records),
            "stored_rows": row_count,
            "model_selection": "default Best Match; selected model is not returned",
            "model_identity_available": model_identity_available,
            "run_metadata_available": run_metadata_available,
            "forecast_issue_time_available": issued_at_available,
            "lead_time_available": lead_time_available,
            "forecast_series_semantics": (
                "Open-Meteo documents the Historical Forecast API as stitching "
                "the first hours of successive model runs into a continuous series."
            ),
            "feature_policy": {
                "used_as_feature": False,
                "reason": issue_missing_reason,
            },
            "report_summary": report_summary,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = raw_dir / "historical_forecast_manifest.json"
        write_json_atomic(manifest_path, forecast_manifest)

        report_path = args.data_dir / "reports" / "quality_report.json"
        markdown_path = args.data_dir / "reports" / "quality_report.md"
        if report_path.exists():
            quality_report = _read_manifest(report_path)
            add_historical_forecast_information(quality_report, forecast_manifest)
            write_json_atomic(report_path, quality_report)
            markdown = markdown_path.read_text(encoding="utf-8")
            markdown = markdown.split(
                "\n## Separate historical forecast information\n", maxsplit=1
            )[0]
            markdown += (
                "\n## Separate historical forecast information\n\n"
                f"- Probability values: {len(values)}/{row_count}; units: `%`.\n"
                "- Forecast probabilities are not observed rainfall labels and are "
                "not included in model features.\n"
                f"- Model/run identity available: {model_identity_available}; "
                f"issue time available: {issued_at_available}; "
                f"lead time available: {lead_time_available}.\n"
                f"- {issue_missing_reason}\n"
            )
            write_text_atomic(markdown_path, markdown)
    except (
        OpenMeteoAPIError,
        ValueError,
        KeyError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"Historical forecast ingestion failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Stored {row_count} historical forecast rows, including {len(values)} "
        f"precipitation-probability values, separately from model features."
    )
    print(f"Forecast metadata: {raw_dir / 'historical_forecast_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
