"""Download and persist an inclusive historical window for Karachi."""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from aetherix_weather.api import OpenMeteoAPIError
from aetherix_weather.config import LATITUDE, LONGITUDE, TIMEZONE
from aetherix_weather.dataset import (
    add_historical_forecast_information,
    build_feature_dataset,
)
from aetherix_weather.historical import (
    AIR_QUALITY_HOURLY_FIELDS,
    WEATHER_HOURLY_FIELDS,
    HistoricalAirQualityClient,
    HistoricalWeatherClient,
    latest_full_historical_day,
    parse_date,
    validate_date_range,
)
from aetherix_weather.storage import (
    read_csv,
    upsert_csv,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _local_range(rows: list[dict[str, Any]]) -> dict[str, str | None]:
    if not rows:
        return {"start": None, "end": None}
    return {
        "start": rows[0]["timestamp_local"][:10],
        "end": rows[-1]["timestamp_local"][:10],
    }


def _report_markdown(report: dict[str, Any]) -> str:
    missing = report["missing_after_cleaning"]["metrics"]
    missing_before = report["missing_before_cleaning"]["metrics"]
    unavailable = report["unavailable_variables"]
    target_counts = report["row_counts"]
    target_stats = report["target_availability"]
    available_target_cells = sum(
        values["available"] for values in target_stats.values()
    )
    possible_target_cells = sum(
        values["available"] + values["missing"] for values in target_stats.values()
    )
    imputed_counts = report["imputation"]["imputed_counts"]
    imputed_total = sum(imputed_counts.values())
    lines = [
        "# AETHERIX AI historical data quality report",
        "",
        f"- City: {report['city']}",
        f"- Coordinates: {report['coordinates']['latitude']}, "
        f"{report['coordinates']['longitude']}",
        f"- Requested dates: {report['requested_date_range']['start']} to "
        f"{report['requested_date_range']['end']}",
        f"- Returned UTC timestamps: {report['actual_date_range_utc']['start']} to "
        f"{report['actual_date_range_utc']['end']}",
        f"- Rows: weather {target_counts['weather']}, air quality "
        f"{target_counts['air_quality']}, aligned {target_counts['aligned']}",
        f"- Rows with at least one future target: "
        f"{target_counts['final_usable_rows_with_any_target']}",
        f"- Available variables: {', '.join(report['available_variables']) or 'none'}",
        f"- Target columns: {len(target_stats)}; observed target values: "
        f"{available_target_cells}/{possible_target_cells}",
        f"- Duplicate timestamps: weather "
        f"{report['duplicate_timestamps']['weather']}, air quality "
        f"{report['duplicate_timestamps']['air_quality']}",
        f"- Missing hourly timestamps: {report['missing_hour_count']}",
        "",
        "## Metric missingness after cleaning",
        "",
        "| Metric | Before missing | Before % | After missing | After % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric, stats in missing.items():
        before_stats = missing_before[metric]
        lines.append(
            f"| {metric} | {before_stats['missing_count']} | "
            f"{before_stats['missing_percent']:.2f}% | {stats['missing_count']} | "
            f"{stats['missing_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Imputation",
            "",
            f"Forward-fill window: {report['imputation']['max_forward_fill_hours']} hours; "
            f"filled values: {imputed_total}.",
            "",
            "## API availability",
            "",
        ]
    )
    lines.extend(
        f"- {item['name']}: {item['status']}" for item in unavailable
    )
    forecast_info = report.get("historical_forecast_information", {})
    if forecast_info.get("status") != "not_ingested":
        lines.extend(
            [
                "",
                "## Separate historical forecast information",
                "",
                f"- Probability values: "
                f"{forecast_info.get('probability_values_available', 0)}/"
                f"{forecast_info.get('record_count', 0)}; not observed rainfall.",
                f"- Model/run identity: {forecast_info.get('model_identity_available')}; "
                f"issue time: {forecast_info.get('forecast_issue_time_available')}; "
                f"lead time: {forecast_info.get('lead_time_available')}.",
                "- Used as a feature: "
                f"{forecast_info.get('used_as_feature', False)}.",
                f"- {forecast_info.get('exclusion_reason', '')}",
            ]
        )
    coverage_limitations = report.get("coverage_limitations", [])
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "No source-range limitation detected."
            if not coverage_limitations
            else "A source returned a shorter range than requested:",
        ]
    )
    for item in coverage_limitations:
        lines.append(
            f"- {item['source']}: requested {item['requested']}; "
            f"returned {item['returned']}."
        )
    quality_issues = report.get("quality_issues", [])
    lines.extend(["", "## Quality issues", ""])
    if quality_issues:
        lines.extend(
            f"- **{issue['severity']} — {issue['code']}:** {issue['detail']}"
            for issue in quality_issues
        )
    else:
        lines.append("No issues reported.")
    lines.extend(
        [
            "",
            "## Sources and limitations",
            "",
            "- Weather is Open-Meteo reanalysis/model data; air quality is modeled CAMS data.",
            "- Historical-weather precipitation probability is missing; archived forecasts are stored separately and excluded.",
            "- Missing rainfall and AQI values remain missing; default forward-fill is disabled.",
            f"- Rain-occurrence label: precipitation exceeds {report['target_definitions']['rain_occurred']['threshold_mm']} mm; reanalysis proxy, not station ground truth.",
            "- Wind direction targets use sine/cosine components.",
            "",
            "See `quality_report.json` for detailed gaps, missing percentages, and target availability.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=90, help="Inclusive days to fetch (default: 90).")
    parser.add_argument(
        "--end-date",
        help="Last requested local date (default: the latest conservatively complete day).",
    )
    parser.add_argument(
        "--max-forward-fill-hours",
        type=int,
        default=0,
        help="Maximum causal fill duration for eligible weather fields (default: 0).",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    if args.days < 1:
        parser.error("--days must be at least 1.")
    try:
        end = parse_date(args.end_date) if args.end_date else latest_full_historical_day()
        start = end - timedelta(days=args.days - 1)
        validate_date_range(
            start,
            end,
            latest_allowed=latest_full_historical_day(),
        )
        if args.max_forward_fill_hours < 0:
            parser.error("--max-forward-fill-hours must be zero or greater.")

        raw_dir = args.data_dir / "raw"
        processed_dir = args.data_dir / "processed"
        reports_dir = args.data_dir / "reports"
        manifest_path = raw_dir / "manifest.json"
        previous_manifest = _read_manifest(manifest_path)

        weather = HistoricalWeatherClient().fetch(start, end)
        air_quality = HistoricalAirQualityClient().fetch(start, end)
        weather_path = raw_dir / "weather.csv"
        air_quality_path = raw_dir / "air_quality.csv"
        weather_total = upsert_csv(weather_path, weather.records)
        air_quality_total = upsert_csv(air_quality_path, air_quality.records)

        stored_weather = read_csv(weather_path)
        stored_air_quality = read_csv(air_quality_path)

        availability = {**weather.availability, **air_quality.availability}
        current_schemas = {
            "weather": sorted(weather.records[0]) if weather.records else [],
            "air_quality": sorted(air_quality.records[0])
            if air_quality.records
            else [],
        }
        processed, report = build_feature_dataset(
            stored_weather,
            stored_air_quality,
            max_forward_fill_hours=args.max_forward_fill_hours,
            availability=availability,
            requested_start_date=start.isoformat(),
            requested_end_date=end.isoformat(),
        )
        forecast_manifest = _read_manifest(
            raw_dir / "historical_forecast_manifest.json"
        )
        add_historical_forecast_information(report, forecast_manifest)
        schema_changes = []
        prior_schemas = previous_manifest.get("source_schemas", {})
        for source, fields in current_schemas.items():
            previous = prior_schemas.get(source)
            if previous is not None and previous != fields:
                schema_changes.append(
                    {
                        "severity": "warning",
                        "code": f"{source}_schema_changed",
                        "detail": f"{source} source columns changed from {previous} to {fields}.",
                    }
                )
        current_feature_schema = report["feature_schema"]
        previous_feature_schema = previous_manifest.get("feature_schema")
        if (
            previous_feature_schema is not None
            and previous_feature_schema != current_feature_schema
        ):
            schema_changes.append(
                {
                    "severity": "warning",
                    "code": "feature_schema_changed",
                    "detail": "The generated feature columns changed from the previous manifest.",
                }
            )
        if schema_changes:
            report["quality_issues"].extend(schema_changes)
        report["returned_date_ranges_local"] = {
            "weather": _local_range(weather.records),
            "air_quality": _local_range(air_quality.records),
        }
        report["coverage_limitations"] = []
        for source, data in (
            ("weather", weather),
            ("air_quality", air_quality),
        ):
            actual = _local_range(data.records)
            if actual["start"] != start.isoformat() or actual["end"] != end.isoformat():
                report["coverage_limitations"].append(
                    {
                        "source": source,
                        "requested": {"start": start.isoformat(), "end": end.isoformat()},
                        "returned": actual,
                    }
                )
        write_csv_atomic(processed_dir / "features.csv", processed)
        write_json_atomic(reports_dir / "quality_report.json", report)
        write_text_atomic(
            reports_dir / "quality_report.md", _report_markdown(report)
        )

        manifest = {
            "city": "Karachi, Pakistan",
            "coordinates": {"latitude": LATITUDE, "longitude": LONGITUDE},
            "timezone": TIMEZONE,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "requested_date_range": {"start": start.isoformat(), "end": end.isoformat()},
            "sources": {
                "weather": {
                    "url": "https://open-meteo.com/en/docs/historical-weather-api",
                    "endpoint": "https://archive-api.open-meteo.com/v1/archive",
                    "description": "Historical weather reanalysis/model estimates",
                    "requested_hourly_variables": list(WEATHER_HOURLY_FIELDS),
                    "returned_grid_coordinates": {
                        "latitude": weather.records[0]["api_latitude"],
                        "longitude": weather.records[0]["api_longitude"],
                    },
                },
                "air_quality": {
                    "url": "https://open-meteo.com/en/docs/air-quality-api",
                    "endpoint": "https://air-quality-api.open-meteo.com/v1/air-quality",
                    "description": (
                        "Modeled US AQI/PM2.5 and hourly cloud-aware UV index; "
                        "not local station observations"
                    ),
                    "requested_hourly_variables": list(AIR_QUALITY_HOURLY_FIELDS),
                    "returned_grid_coordinates": {
                        "latitude": air_quality.records[0]["api_latitude"],
                        "longitude": air_quality.records[0]["api_longitude"],
                    },
                },
            },
            "availability": availability,
            "units": {**weather.units, **air_quality.units},
            "source_schemas": current_schemas,
            "feature_schema": current_feature_schema,
            "returned_date_ranges_local": report["returned_date_ranges_local"],
            "coverage_limitations": report["coverage_limitations"],
            "stored_row_counts": {
                "weather": weather_total,
                "air_quality": air_quality_total,
                "processed": len(processed),
            },
        }
        write_json_atomic(manifest_path, manifest)
    except (OpenMeteoAPIError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Historical backfill failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Backfill complete for {start.isoformat()} through {end.isoformat()} "
        f"({args.days} requested days)."
    )
    print(
        f"Stored weather rows: {weather_total}; air-quality rows: "
        f"{air_quality_total}; processed rows: {len(processed)}."
    )
    print(f"Quality report: {reports_dir / 'quality_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
