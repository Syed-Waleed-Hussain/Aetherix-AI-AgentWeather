"""Timestamp alignment, quality checks, imputation, features, and targets."""

from datetime import datetime, timedelta, timezone
from math import cos, isfinite, pi, sin
from statistics import mean, stdev
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import LATITUDE, LONGITUDE, TIMEZONE

RAIN_OCCURRENCE_THRESHOLD_MM = 0.0
METRIC_COLUMNS = (
    "temperature_c",
    "relative_humidity_percent",
    "precipitation_mm",
    "precipitation_probability_percent",
    "wind_speed_kmh",
    "wind_direction_degrees",
    "uv_index",
    "us_aqi",
    "pm2_5_ug_m3",
)

_WEATHER_COLUMNS = (
    "temperature_c",
    "relative_humidity_percent",
    "precipitation_mm",
    "precipitation_probability_percent",
    "wind_speed_kmh",
    "wind_direction_degrees",
    "uv_index",
)
_AIR_QUALITY_COLUMNS = ("us_aqi", "pm2_5_ug_m3")
_SAFE_FORWARD_FILL_COLUMNS = (
    "temperature_c",
    "relative_humidity_percent",
    "wind_speed_kmh",
    "wind_direction_degrees",
    "uv_index",
)
_FEATURE_METRICS = (
    "temperature_c",
    "relative_humidity_percent",
    "precipitation_mm",
    "precipitation_probability_percent",
    "wind_speed_kmh",
    "uv_index",
    "us_aqi",
    "pm2_5_ug_m3",
)
_TARGET_METRICS = {
    "temperature_c": "temperature",
    "relative_humidity_percent": "relative_humidity_percent",
    "precipitation_mm": "precipitation_mm",
    "precipitation_probability_percent": "precipitation_probability_percent",
    "wind_speed_kmh": "wind_speed_kmh",
    "uv_index": "uv_index",
    "us_aqi": "us_aqi",
    "pm2_5_ug_m3": "pm2_5_ug_m3",
}


class DataQualityError(ValueError):
    """Raised when timestamp identity or ordering makes alignment unsafe."""


def _utc_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise DataQualityError("timestamp_utc must be an ISO-8601 string.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise DataQualityError(f"Invalid UTC timestamp: {value!r}.") from exc
    if parsed.tzinfo is None:
        raise DataQualityError("timestamp_utc must be timezone-aware.")
    return parsed.astimezone(timezone.utc)


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _check_unique_monotonic(
    rows: list[dict[str, Any]], source_name: str
) -> list[datetime]:
    stamps = [_utc_datetime(row.get("timestamp_utc")) for row in rows]
    if len(set(stamps)) != len(stamps):
        raise DataQualityError(f"{source_name} contains duplicate timestamps.")
    if stamps != sorted(stamps):
        raise DataQualityError(f"{source_name} timestamps are not monotonic.")
    return stamps


def align_hourly(
    weather_rows: list[dict[str, Any]], air_quality_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Outer-join sources by exact UTC instant without shifting or dropping rows."""
    weather_stamps = _check_unique_monotonic(weather_rows, "Weather data")
    air_stamps = _check_unique_monotonic(air_quality_rows, "Air-quality data")
    weather_by_stamp = dict(zip(weather_stamps, weather_rows))
    air_by_stamp = dict(zip(air_stamps, air_quality_rows))
    all_stamps = sorted(set(weather_by_stamp) | set(air_by_stamp))
    try:
        local_zone = ZoneInfo(TIMEZONE)
    except ZoneInfoNotFoundError as exc:
        raise DataQualityError(f"Timezone data is unavailable for {TIMEZONE}.") from exc

    combined: list[dict[str, Any]] = []
    for stamp in all_stamps:
        weather = weather_by_stamp.get(stamp, {})
        air = air_by_stamp.get(stamp, {})
        for source_name, source_row in (("weather", weather), ("air quality", air)):
            local_value = source_row.get("timestamp_local")
            if local_value:
                try:
                    source_local = datetime.fromisoformat(str(local_value))
                except ValueError as exc:
                    raise DataQualityError(
                        f"{source_name} contains an invalid local timestamp."
                    ) from exc
                if (
                    source_local.tzinfo is None
                    or source_local.astimezone(timezone.utc) != stamp
                    or source_local.utcoffset()
                    != stamp.astimezone(local_zone).utcoffset()
                ):
                    raise DataQualityError(
                        f"{source_name} UTC/local timestamp conversion is inconsistent."
                    )
        row: dict[str, Any] = {
            "timestamp_utc": stamp.isoformat(),
            "timestamp_local": stamp.astimezone(local_zone).isoformat(),
            "weather_source_timestamp": weather.get("source_timestamp"),
            "air_quality_source_timestamp": air.get("source_timestamp"),
            "weather_source": weather.get("source"),
            "air_quality_source": air.get("source"),
            "weather_ingested_at_utc": weather.get("ingested_at_utc"),
            "air_quality_ingested_at_utc": air.get("ingested_at_utc"),
        }
        for column in _WEATHER_COLUMNS:
            if column != "uv_index":
                row[column] = _number(weather.get(column))
        for column in _AIR_QUALITY_COLUMNS:
            row[column] = _number(air.get(column))
        weather_uv = _number(weather.get("uv_index"))
        air_quality_uv = _number(air.get("uv_index"))
        row["weather_uv_index"] = weather_uv
        row["air_quality_uv_index"] = air_quality_uv
        row["uv_index"] = weather_uv if weather_uv is not None else air_quality_uv
        row["uv_index_source"] = (
            weather.get("source")
            if weather_uv is not None
            else air.get("source")
            if air_quality_uv is not None
            else None
        )
        combined.append(row)

    weather_only = sorted(set(weather_stamps) - set(air_stamps))
    air_only = sorted(set(air_stamps) - set(weather_stamps))
    gaps = _missing_hour_timestamps(all_stamps)
    return combined, {
        "weather_rows": len(weather_rows),
        "air_quality_rows": len(air_quality_rows),
        "aligned_rows": len(combined),
        "weather_only_timestamps": [value.isoformat() for value in weather_only],
        "air_quality_only_timestamps": [value.isoformat() for value in air_only],
        "missing_hourly_timestamps": [value.isoformat() for value in gaps],
        "duplicate_weather_timestamps": 0,
        "duplicate_air_quality_timestamps": 0,
    }


def _missing_hour_timestamps(stamps: list[datetime]) -> list[datetime]:
    missing: list[datetime] = []
    for previous, following in zip(stamps, stamps[1:]):
        cursor = previous + timedelta(hours=1)
        while cursor < following:
            missing.append(cursor)
            cursor += timedelta(hours=1)
    return missing


def missing_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report metric null counts/percentages without mutating records."""
    total = len(rows)
    metrics: dict[str, dict[str, float | int]] = {}
    for column in METRIC_COLUMNS:
        missing_count = sum(_number(row.get(column)) is None for row in rows)
        metrics[column] = {
            "missing_count": missing_count,
            "missing_percent": round(missing_count * 100 / total, 2) if total else 0.0,
        }
    return {"rows": total, "metrics": metrics}


def handle_missing(
    rows: list[dict[str, Any]], max_forward_fill_hours: int = 0
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Add missing/imputed flags; optionally forward-fill only safe weather fields."""
    if max_forward_fill_hours < 0:
        raise ValueError("max_forward_fill_hours must be zero or greater.")
    stamps = _check_unique_monotonic(rows, "Dataset")
    before = missing_summary(rows)
    result = [dict(row) for row in rows]
    imputed_counts = {column: 0 for column in _SAFE_FORWARD_FILL_COLUMNS}
    previous_values: dict[str, tuple[datetime, float]] = {}

    for row, stamp in zip(result, stamps):
        for column in METRIC_COLUMNS:
            original = _number(row.get(column))
            row[f"{column}_was_missing"] = original is None
            row[f"{column}_imputed"] = False
        for column in _SAFE_FORWARD_FILL_COLUMNS:
            current = _number(row.get(column))
            if current is not None:
                previous_values[column] = (stamp, current)
                continue
            previous = previous_values.get(column)
            if previous is None or max_forward_fill_hours == 0:
                continue
            elapsed = (stamp - previous[0]).total_seconds() / 3600
            if 0 < elapsed <= max_forward_fill_hours:
                row[column] = previous[1]
                row[f"{column}_imputed"] = True
                imputed_counts[column] += 1

    after = missing_summary(result)
    return result, {
        "max_forward_fill_hours": max_forward_fill_hours,
        "eligible_columns": list(_SAFE_FORWARD_FILL_COLUMNS),
        "imputed_counts": imputed_counts,
        "excluded_columns": [
            "precipitation_mm",
            "precipitation_probability_percent",
            "us_aqi",
            "pm2_5_ug_m3",
        ],
    }, {"before": before, "after": after}


def _feature_float(row: dict[str, Any], column: str) -> float | None:
    return _number(row.get(column))


def _has_contiguous_window(stamps: list[datetime], start: int, end: int) -> bool:
    return all(
        stamps[index] - stamps[index - 1] == timedelta(hours=1)
        for index in range(start + 1, end + 1)
    )


def create_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build causal calendar, lag, rolling, change, and wind-component features."""
    stamps = _check_unique_monotonic(rows, "Dataset")
    if not rows:
        return []
    derived_metrics = [
        column
        for column in _FEATURE_METRICS
        if any(_feature_float(row, column) is not None for row in rows)
    ]
    values: dict[str, list[float | None]] = {
        column: [_feature_float(row, column) for row in rows]
        for column in derived_metrics
    }
    if any(_feature_float(row, "wind_direction_degrees") is not None for row in rows):
        directions = [_feature_float(row, "wind_direction_degrees") for row in rows]
        values["wind_direction_sin"] = [
            sin(value * pi / 180) if value is not None else None
            for value in directions
        ]
        values["wind_direction_cos"] = [
            cos(value * pi / 180) if value is not None else None
            for value in directions
        ]

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        output = dict(row)
        try:
            local = datetime.fromisoformat(str(row["timestamp_local"]))
        except (KeyError, ValueError) as exc:
            raise DataQualityError("timestamp_local must be a valid ISO timestamp.") from exc
        if local.tzinfo is None:
            raise DataQualityError("timestamp_local must be timezone-aware.")
        day_of_year = local.timetuple().tm_yday
        days_in_year = 366 if _is_leap_year(local.year) else 365
        output.update(
            {
                "hour_of_day": local.hour,
                "day_of_week": local.weekday(),
                "month": local.month,
                "is_weekend": int(local.weekday() >= 5),
                "hour_sin": sin(2 * pi * local.hour / 24),
                "hour_cos": cos(2 * pi * local.hour / 24),
                "day_of_year_sin": sin(2 * pi * (day_of_year - 1) / days_in_year),
                "day_of_year_cos": cos(2 * pi * (day_of_year - 1) / days_in_year),
            }
        )

        for metric, series in values.items():
            current = series[index]
            if metric in ("wind_direction_sin", "wind_direction_cos"):
                output[metric] = current
            for lag in (1, 3, 6, 12, 24, 48):
                prior_index = index - lag
                value = None
                if (
                    prior_index >= 0
                    and stamps[index] - stamps[prior_index] == timedelta(hours=lag)
                ):
                    value = series[prior_index]
                output[f"{metric}_lag_{lag}h"] = value
            for window in (3, 6, 12, 24):
                start = index - window + 1
                window_values: list[float | None] = []
                if start >= 0 and _has_contiguous_window(stamps, start, index):
                    window_values = series[start : index + 1]
                complete = len(window_values) == window and all(
                    value is not None for value in window_values
                )
                numeric_window = [value for value in window_values if value is not None]
                output[f"{metric}_rolling_mean_{window}h"] = (
                    mean(numeric_window) if complete else None
                )
                if window == 24:
                    output[f"{metric}_rolling_std_24h"] = (
                        stdev(numeric_window) if complete else None
                    )
            for change in (1, 3, 24):
                prior_index = index - change
                prior = (
                    series[prior_index]
                    if prior_index >= 0
                    and stamps[index] - stamps[prior_index] == timedelta(hours=change)
                    else None
                )
                output[f"{metric}_change_{change}h"] = (
                    current - prior
                    if current is not None and prior is not None
                    else None
                )
        results.append(output)
    return results


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def create_targets(
    rows: list[dict[str, Any]], include_rain_occurrence: bool = True
) -> list[dict[str, Any]]:
    """Attach future-only targets by exact UTC hour; never invent missing targets."""
    stamps = _check_unique_monotonic(rows, "Dataset")
    by_stamp = {stamp: row for stamp, row in zip(stamps, rows)}
    result = [dict(row) for row in rows]
    available_target_metrics = {
        column: target_name
        for column, target_name in _TARGET_METRICS.items()
        if any(_number(row.get(column)) is not None for row in rows)
    }
    has_direction = any(
        _number(row.get("wind_direction_degrees")) is not None for row in rows
    )
    has_rain = include_rain_occurrence and any(
        _number(row.get("precipitation_mm")) is not None for row in rows
    )
    for output in result:
        for horizon in range(1, 25):
            future = by_stamp.get(
                _utc_datetime(output["timestamp_utc"]) + timedelta(hours=horizon)
            )
            for column, target_name in available_target_metrics.items():
                output[f"target_{target_name}_h{horizon:02}"] = (
                    _number(future.get(column)) if future is not None else None
                )
            if has_direction:
                direction = (
                    _number(future.get("wind_direction_degrees"))
                    if future is not None
                    else None
                )
                output[f"target_wind_direction_sin_h{horizon:02}"] = (
                    sin(direction * pi / 180) if direction is not None else None
                )
                output[f"target_wind_direction_cos_h{horizon:02}"] = (
                    cos(direction * pi / 180) if direction is not None else None
                )
            if has_rain:
                precipitation = (
                    _number(future.get("precipitation_mm"))
                    if future is not None
                    else None
                )
                output[f"target_rain_occurred_h{horizon:02}"] = (
                    int(precipitation > RAIN_OCCURRENCE_THRESHOLD_MM)
                    if precipitation is not None
                    else None
                )
    return result


def target_availability(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count observed and missing target values for every generated target."""
    if not rows:
        return {}
    output: dict[str, dict[str, int]] = {}
    target_keys = [
        key for key in rows[0] if key.startswith("target_") and "_h" in key
    ]
    for key in target_keys:
        available = sum(_number(row.get(key)) is not None for row in rows)
        output[key] = {"available": available, "missing": len(rows) - available}
    return output


def data_quality_report(
    rows: list[dict[str, Any]],
    alignment: dict[str, Any],
    missing: dict[str, Any],
    imputation: dict[str, Any],
    targets: list[dict[str, Any]],
    availability: dict[str, str],
    requested_start_date: str,
    requested_end_date: str,
) -> dict[str, Any]:
    """Build a transparent report from the actual aligned and processed rows."""
    stamps = [_utc_datetime(row["timestamp_utc"]) for row in rows]
    issues: list[dict[str, str]] = []
    for column, minimum, maximum in (
        ("relative_humidity_percent", 0, 100),
        ("precipitation_mm", 0, None),
        ("wind_speed_kmh", 0, None),
        ("wind_direction_degrees", 0, 360),
        ("us_aqi", 0, 500),
    ):
        for index, row in enumerate(rows):
            value = _number(row.get(column))
            if value is not None and (
                value < minimum or (maximum is not None and value > maximum)
            ):
                issues.append(
                    {
                        "severity": "warning",
                        "code": f"invalid_{column}",
                        "timestamp_utc": stamps[index].isoformat(),
                        "detail": f"{column}={value} is outside [{minimum}, {maximum}].",
                    }
                )
    gaps = _missing_hour_timestamps(stamps)
    for stamp in gaps:
        issues.append(
            {
                "severity": "warning",
                "code": "missing_hour",
                "timestamp_utc": stamp.isoformat(),
                "detail": "No source has a row for this hourly timestamp.",
            }
        )
    for source_name, key in (
        ("weather", "weather_only_timestamps"),
        ("air_quality", "air_quality_only_timestamps"),
    ):
        for stamp in alignment.get(key, []):
            issues.append(
                {
                    "severity": "warning",
                    "code": f"{source_name}_timestamp_not_in_other_source",
                    "timestamp_utc": stamp,
                    "detail": f"Timestamp exists only in {source_name} data.",
                }
            )
    available_variables = [
        name for name, status in availability.items() if status.startswith("available")
    ]
    unavailable_variables = [
        {"name": name, "status": status}
        for name, status in availability.items()
        if not status.startswith("available")
    ]
    return {
        "city": "Karachi, Pakistan",
        "coordinates": {"latitude": LATITUDE, "longitude": LONGITUDE},
        "timezone": TIMEZONE,
        "providers": {
            "weather": "Open-Meteo Historical Weather API; weather reanalysis/model estimates",
            "air_quality": "Open-Meteo Air Quality API; CAMS modeled air-quality data",
        },
        "requested_date_range": {
            "start": requested_start_date,
            "end": requested_end_date,
        },
        "actual_date_range_utc": {
            "start": stamps[0].isoformat() if stamps else None,
            "end": stamps[-1].isoformat() if stamps else None,
        },
        "row_counts": {
            "weather": alignment["weather_rows"],
            "air_quality": alignment["air_quality_rows"],
            "aligned": alignment["aligned_rows"],
            "final_usable_rows_with_any_target": sum(
                any(_number(value) is not None for key, value in row.items()
                    if key.startswith("target_"))
                for row in targets
            ),
        },
        "source_alignment": {
            "weather_only_count": len(alignment.get("weather_only_timestamps", [])),
            "weather_only_timestamps": alignment.get("weather_only_timestamps", []),
            "air_quality_only_count": len(
                alignment.get("air_quality_only_timestamps", [])
            ),
            "air_quality_only_timestamps": alignment.get(
                "air_quality_only_timestamps", []
            ),
        },
        "available_variables": available_variables,
        "unavailable_variables": unavailable_variables,
        "missing_before_cleaning": missing["before"],
        "missing_after_cleaning": missing["after"],
        "imputation": imputation,
        "duplicate_timestamps": {
            "weather": alignment.get("duplicate_weather_timestamps", 0),
            "air_quality": alignment.get("duplicate_air_quality_timestamps", 0),
        },
        "missing_hour_count": len(gaps),
        "quality_issues": issues,
        "target_availability": target_availability(targets),
        "target_definitions": {
            "rain_occurred": {
                "threshold_mm": RAIN_OCCURRENCE_THRESHOLD_MM,
                "positive_rule": (
                    f"precipitation_mm > {RAIN_OCCURRENCE_THRESHOLD_MM}"
                ),
                "missing_rule": "Missing future precipitation produces a missing target.",
                "source_semantics": "Historical weather reanalysis/model proxy, not station-observed truth.",
            }
        },
        "metric_provenance": {
            "uv_index": (
                "Open-Meteo Air Quality API hourly uv_index, cloud-aware; "
                "uv_index_clear_sky is not used. Historical weather archive "
                "value is preferred only when it is present."
            ),
            "rain_occurred": (
                f"Derived from historical weather precipitation_mm > "
                f"{RAIN_OCCURRENCE_THRESHOLD_MM} mm; this is a reanalysis/model "
                "proxy, not station-observed ground truth."
            ),
        },
        "feature_schema": sorted(
            key
            for key in targets[0]
            if not key.startswith("target_")
        )
        if targets
        else [],
        "limitations": [
            "The historical weather archive is reanalysis/model data, not station-only observations.",
            "Air-quality values, including US AQI and PM2.5, are modeled estimates rather than direct observations.",
            "Historical fields with all-null arrays, including precipitation probability or UV when unavailable, remain missing and are not given targets.",
            "Rain-occurrence labels are derived from historical weather reanalysis precipitation, not station-observed ground truth.",
            "Wind direction targets are sine/cosine components to preserve circularity.",
        ],
    }


def add_historical_forecast_information(
    report: dict[str, Any], manifest: dict[str, Any] | None = None
) -> None:
    """Keep archived forecast provenance distinct from observation features."""
    summary = manifest.get("report_summary") if manifest is not None else None
    report["historical_forecast_information"] = (
        summary
        if isinstance(summary, dict)
        else {"status": "not_ingested", "used_as_feature": False}
    )


def build_feature_dataset(
    weather_rows: list[dict[str, Any]],
    air_quality_rows: list[dict[str, Any]],
    max_forward_fill_hours: int = 0,
    availability: dict[str, str] | None = None,
    requested_start_date: str = "",
    requested_end_date: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run alignment, missing handling, causal features, targets, and reporting."""
    aligned, alignment = align_hourly(weather_rows, air_quality_rows)
    if not aligned:
        raise DataQualityError("No hourly records are available to process.")
    cleaned, imputation, missing = handle_missing(
        aligned, max_forward_fill_hours=max_forward_fill_hours
    )
    features = create_features(cleaned)
    targets = create_targets(features)
    report = data_quality_report(
        cleaned,
        alignment,
        missing,
        imputation,
        targets,
        availability or {},
        requested_start_date,
        requested_end_date,
    )
    add_historical_forecast_information(report)
    returned_ranges = {
        "weather": _source_local_date_range(weather_rows),
        "air_quality": _source_local_date_range(air_quality_rows),
    }
    coverage_limitations = []
    if requested_start_date and requested_end_date:
        for source, actual in returned_ranges.items():
            if (
                actual["start"] != requested_start_date
                or actual["end"] != requested_end_date
            ):
                coverage_limitations.append(
                    {
                        "source": source,
                        "requested": {
                            "start": requested_start_date,
                            "end": requested_end_date,
                        },
                        "returned": actual,
                    }
                )
    report["returned_date_ranges_local"] = returned_ranges
    report["coverage_limitations"] = coverage_limitations
    return targets, report


def _source_local_date_range(
    rows: list[dict[str, Any]],
) -> dict[str, str | None]:
    if not rows:
        return {"start": None, "end": None}
    local_stamps = [
        datetime.fromisoformat(str(row["timestamp_local"])) for row in rows
    ]
    return {
        "start": min(local_stamps).date().isoformat(),
        "end": max(local_stamps).date().isoformat(),
    }
