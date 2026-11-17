"""Historical Open-Meteo clients for weather reanalysis and air-quality models."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
from math import isfinite
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from .api import OpenMeteoAPIError, request_json
from .config import (
    AIR_QUALITY_API_URL,
    LATITUDE,
    LONGITUDE,
    REQUEST_TIMEOUT_SECONDS,
    TIMEZONE,
)

ARCHIVE_WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
HISTORICAL_SAFETY_LAG_DAYS = 6
WEATHER_HOURLY_FIELDS = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "wind_speed_10m",
    "wind_direction_10m",
    "uv_index",
)
AIR_QUALITY_HOURLY_FIELDS = ("us_aqi", "pm2_5", "uv_index")
HISTORICAL_FORECAST_HOURLY_FIELDS = ("precipitation_probability",)

_WEATHER_COLUMN_NAMES = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "relative_humidity_percent",
    "precipitation": "precipitation_mm",
    "precipitation_probability": "precipitation_probability_percent",
    "wind_speed_10m": "wind_speed_kmh",
    "wind_direction_10m": "wind_direction_degrees",
    "uv_index": "uv_index",
}
_AIR_QUALITY_COLUMN_NAMES = {
    "us_aqi": "us_aqi",
    "pm2_5": "pm2_5_ug_m3",
    "uv_index": "uv_index",
}


@dataclass(frozen=True)
class HistoricalDataset:
    """Normalized source rows and provenance for one historical API response."""

    records: list[dict[str, Any]]
    source: str
    requested_start_date: str
    requested_end_date: str
    availability: dict[str, str]
    units: dict[str, str]


def parse_date(value: str | date) -> date:
    """Parse an ISO date or date object, rejecting datetimes and invalid dates."""
    if isinstance(value, datetime):
        raise ValueError("Dates must not include a time component.")
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise TypeError("Date must be a date object or an ISO YYYY-MM-DD string.")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date {value!r}; expected YYYY-MM-DD.") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"Invalid date {value!r}; expected YYYY-MM-DD.")
    return parsed


def validate_date_range(
    start_date: str | date,
    end_date: str | date,
    latest_allowed: date | None = None,
) -> tuple[date, date]:
    """Return parsed inclusive dates after validating their ordering and limit."""
    start = parse_date(start_date)
    end = parse_date(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date.")
    if latest_allowed is not None and end > latest_allowed:
        raise ValueError(
            f"end_date must be on or before {latest_allowed.isoformat()}."
        )
    return start, end


def latest_full_historical_day(today: date | None = None) -> date:
    """Use a conservative buffer for the archive's published update delays."""
    reference = today or date.today()
    return reference - timedelta(days=HISTORICAL_SAFETY_LAG_DAYS)


def _normalise_historical_payload(
    payload: dict[str, Any],
    endpoint: str,
    timezone_name: str,
    requested_fields: tuple[str, ...],
    column_names: dict[str, str],
    source: str,
    start: date,
    end: date,
    availability: dict[str, str],
    units: dict[str, str],
) -> HistoricalDataset:
    if payload.get("timezone") != timezone_name:
        raise OpenMeteoAPIError(
            f"{endpoint} response timezone must be {timezone_name!r}."
        )
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise OpenMeteoAPIError(f"{endpoint} response is missing hourly data.")
    times = hourly.get("time")
    if not isinstance(times, list) or not times:
        raise OpenMeteoAPIError(f"{endpoint} response has no hourly timestamps.")

    try:
        local_zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise OpenMeteoAPIError(
            f"Timezone data is unavailable for {timezone_name}."
        ) from exc

    parsed_times: list[datetime] = []
    source_times: list[str] = []
    for raw_time in times:
        if not isinstance(raw_time, str):
            raise OpenMeteoAPIError(
                f"{endpoint} response contains an invalid timestamp."
            )
        try:
            parsed = datetime.fromisoformat(raw_time)
        except ValueError as exc:
            raise OpenMeteoAPIError(
                f"{endpoint} response contains an invalid timestamp: {raw_time!r}."
            ) from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=local_zone)
        parsed_times.append(parsed.astimezone(timezone.utc))
        source_times.append(raw_time)

    if len(set(parsed_times)) != len(parsed_times):
        raise OpenMeteoAPIError(f"{endpoint} response contains duplicate timestamps.")
    if parsed_times != sorted(parsed_times):
        raise OpenMeteoAPIError(f"{endpoint} timestamps are not monotonic.")

    arrays: dict[str, list[Any]] = {}
    for field in requested_fields:
        values = hourly.get(field)
        if not isinstance(values, list) or len(values) != len(parsed_times):
            raise OpenMeteoAPIError(
                f"{endpoint} field {field!r} must contain "
                f"{len(parsed_times)} hourly values."
            )
        arrays[field] = values

    ingestion_time = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    for index, timestamp_utc in enumerate(parsed_times):
        timestamp_local = timestamp_utc.astimezone(local_zone)
        row: dict[str, Any] = {
            "timestamp_utc": timestamp_utc.isoformat(),
            "timestamp_local": timestamp_local.isoformat(),
            "source_timestamp": source_times[index],
            "source": source,
            "ingested_at_utc": ingestion_time,
            "api_latitude": payload.get("latitude"),
            "api_longitude": payload.get("longitude"),
            "api_timezone": payload.get("timezone"),
        }
        for field in column_names.values():
            row[field] = None
        for field, column in column_names.items():
            value = arrays[field][index]
            if value is None:
                row[column] = None
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OpenMeteoAPIError(
                    f"{endpoint} field {field!r} contains a non-numeric value."
                )
            elif not isfinite(value):
                raise OpenMeteoAPIError(
                    f"{endpoint} field {field!r} contains a non-finite value."
                )
            else:
                row[column] = float(value)
        records.append(row)

    observed_availability = {
        column: (
            status
            if any(row.get(column) is not None for row in records)
            else "no_historical_values"
        )
        if status.startswith("available")
        else status
        for column, status in availability.items()
    }
    return HistoricalDataset(
        records=records,
        source=source,
        requested_start_date=start.isoformat(),
        requested_end_date=end.isoformat(),
        availability=observed_availability,
        units=units,
    )


class HistoricalWeatherClient:
    """Fetch hourly weather reanalysis from Open-Meteo's archive endpoint."""

    def __init__(
        self,
        latitude: float = LATITUDE,
        longitude: float = LONGITUDE,
        timezone_name: str = TIMEZONE,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        if not -90 <= latitude <= 90:
            raise ValueError("latitude must be between -90 and 90.")
        if not -180 <= longitude <= 180:
            raise ValueError("longitude must be between -180 and 180.")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")
        self.latitude = latitude
        self.longitude = longitude
        self.timezone_name = timezone_name
        self.timeout = timeout
        self.session = session or requests.Session()

    def fetch(
        self, start_date: str | date, end_date: str | date
    ) -> HistoricalDataset:
        start, end = validate_date_range(start_date, end_date)
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(WEATHER_HOURLY_FIELDS),
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
            "timezone": self.timezone_name,
        }
        payload = request_json(
            self.session, ARCHIVE_WEATHER_API_URL, params, self.timeout
        )
        availability = {
            "temperature_c": "available",
            "relative_humidity_percent": "available",
            "precipitation_mm": "available",
            "precipitation_probability_percent": "available",
            "wind_speed_kmh": "available",
            "wind_direction_degrees": "available",
            "uv_index": "available",
        }
        units = {
            "temperature_c": "°C",
            "relative_humidity_percent": "%",
            "precipitation_mm": "mm",
            "precipitation_probability_percent": "%",
            "wind_speed_kmh": "km/h",
            "wind_direction_degrees": "degrees",
            "uv_index": "index",
        }
        return _normalise_historical_payload(
            payload,
            "Historical weather",
            self.timezone_name,
            WEATHER_HOURLY_FIELDS,
            _WEATHER_COLUMN_NAMES,
            "open-meteo-historical-weather",
            start,
            end,
            availability,
            units,
        )


class HistoricalAirQualityClient:
    """Fetch past modeled AQI and PM2.5 from the Open-Meteo air-quality API."""

    def __init__(
        self,
        latitude: float = LATITUDE,
        longitude: float = LONGITUDE,
        timezone_name: str = TIMEZONE,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        if not -90 <= latitude <= 90:
            raise ValueError("latitude must be between -90 and 90.")
        if not -180 <= longitude <= 180:
            raise ValueError("longitude must be between -180 and 180.")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")
        self.latitude = latitude
        self.longitude = longitude
        self.timezone_name = timezone_name
        self.timeout = timeout
        self.session = session or requests.Session()

    def fetch(
        self, start_date: str | date, end_date: str | date
    ) -> HistoricalDataset:
        start, end = validate_date_range(start_date, end_date)
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(AIR_QUALITY_HOURLY_FIELDS),
            "timezone": self.timezone_name,
        }
        payload = request_json(
            self.session, AIR_QUALITY_API_URL, params, self.timeout
        )
        return _normalise_historical_payload(
            payload,
            "Historical air quality",
            self.timezone_name,
            AIR_QUALITY_HOURLY_FIELDS,
            _AIR_QUALITY_COLUMN_NAMES,
            "open-meteo-modeled-air-quality",
            start,
            end,
            {
                "us_aqi": "available_modeled",
                "pm2_5_ug_m3": "available_modeled",
                "uv_index": "available_modeled",
            },
            {
                "us_aqi": "US AQI index",
                "pm2_5_ug_m3": "µg/m³",
                "uv_index": "index",
            },
        )


class HistoricalForecastClient:
    """Fetch stitched historical forecasts; output is not observation truth."""

    def __init__(
        self,
        latitude: float = LATITUDE,
        longitude: float = LONGITUDE,
        timezone_name: str = TIMEZONE,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        if not -90 <= latitude <= 90:
            raise ValueError("latitude must be between -90 and 90.")
        if not -180 <= longitude <= 180:
            raise ValueError("longitude must be between -180 and 180.")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")
        self.latitude = latitude
        self.longitude = longitude
        self.timezone_name = timezone_name
        self.timeout = timeout
        self.session = session or requests.Session()

    def fetch(
        self, start_date: str | date, end_date: str | date
    ) -> HistoricalDataset:
        start, end = validate_date_range(start_date, end_date)
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(HISTORICAL_FORECAST_HOURLY_FIELDS),
            "timezone": self.timezone_name,
        }
        payload = request_json(
            self.session, HISTORICAL_FORECAST_API_URL, params, self.timeout
        )
        dataset = _normalise_historical_payload(
            payload,
            "Historical forecast",
            self.timezone_name,
            HISTORICAL_FORECAST_HOURLY_FIELDS,
            {
                "precipitation_probability": (
                    "forecast_precipitation_probability_percent"
                )
            },
            "open-meteo-historical-forecast",
            start,
            end,
            {"forecast_precipitation_probability_percent": "available_forecast"},
            {"forecast_precipitation_probability_percent": "%"},
        )
        for row in dataset.records:
            probability = row["forecast_precipitation_probability_percent"]
            if probability is not None and not 0 <= probability <= 100:
                raise OpenMeteoAPIError(
                    "Historical forecast precipitation probability must be between 0 and 100."
                )
            row.update(
                {
                    "forecast_valid_timestamp_utc": row["timestamp_utc"],
                    "forecast_issued_at_utc": payload.get("forecast_issued_at_utc"),
                    "forecast_run_id": payload.get("run"),
                    "forecast_api_latitude": payload.get("latitude"),
                    "forecast_api_longitude": payload.get("longitude"),
                    "forecast_api_timezone": payload.get("timezone"),
                    "forecast_model_metadata": json.dumps(
                        {
                            "model": payload.get("model"),
                            "models": payload.get("models"),
                        },
                        sort_keys=True,
                    ),
                    "forecast_model_selection": "default_best_match",
                    "forecast_lead_hours": None,
                    "forecast_value_type": "historical_model_forecast",
                }
            )
        return dataset
