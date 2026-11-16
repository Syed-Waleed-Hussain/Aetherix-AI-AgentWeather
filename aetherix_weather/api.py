"""Shared HTTP and response-validation helpers for Open-Meteo clients."""

from datetime import datetime
from math import isfinite
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from .config import HOURLY_FORECAST_HOURS, REQUEST_TIMEOUT_SECONDS, TIMEZONE


class OpenMeteoAPIError(RuntimeError):
    """Raised when an Open-Meteo request or response is unusable."""


def request_json(
    session: requests.Session,
    url: str,
    params: dict[str, str | float | int],
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Request a JSON object and turn network/API failures into a clear error."""
    try:
        response = session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise OpenMeteoAPIError(f"Open-Meteo request timed out after {timeout:g}s.") from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise OpenMeteoAPIError(f"Open-Meteo returned HTTP {status}.") from exc
    except requests.RequestException as exc:
        raise OpenMeteoAPIError(f"Open-Meteo request failed: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise OpenMeteoAPIError("Open-Meteo returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise OpenMeteoAPIError("Open-Meteo returned an unexpected response format.")
    if payload.get("error") is True:
        reason = payload.get("reason", "No reason provided.")
        raise OpenMeteoAPIError(f"Open-Meteo API error: {reason}")
    return payload


def hourly_records(
    payload: dict[str, Any],
    endpoint: str,
    fields: tuple[str, ...],
    timezone_name: str = TIMEZONE,
) -> list[dict[str, datetime | float | None]]:
    """Validate hourly arrays and return aligned rows with local-aware times."""
    if payload.get("timezone") != timezone_name:
        raise OpenMeteoAPIError(
            f"{endpoint} response timezone must be {timezone_name!r}."
        )

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise OpenMeteoAPIError(f"{endpoint} response is missing hourly data.")

    times = hourly.get("time")
    if not isinstance(times, list) or len(times) != HOURLY_FORECAST_HOURS:
        raise OpenMeteoAPIError(
            f"{endpoint} response must contain exactly "
            f"{HOURLY_FORECAST_HOURS} hourly timestamps."
        )

    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise OpenMeteoAPIError(
            f"Timezone data is unavailable for {timezone_name}."
        ) from exc

    parsed_times: list[datetime] = []
    for value in times:
        if not isinstance(value, str):
            raise OpenMeteoAPIError(f"{endpoint} response contains an invalid timestamp.")
        try:
            timestamp = datetime.fromisoformat(value)
        except ValueError as exc:
            raise OpenMeteoAPIError(
                f"{endpoint} response contains an invalid timestamp: {value!r}."
            ) from exc
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone)
        parsed_times.append(timestamp)

    arrays: dict[str, list[Any]] = {}
    for field in fields:
        values = hourly.get(field)
        if not isinstance(values, list) or len(values) != len(parsed_times):
            raise OpenMeteoAPIError(
                f"{endpoint} response field {field!r} must contain "
                f"{len(parsed_times)} hourly values."
            )
        arrays[field] = values

    rows: list[dict[str, datetime | float | None]] = []
    for index, timestamp in enumerate(parsed_times):
        row: dict[str, datetime | float | None] = {"time": timestamp}
        for field in fields:
            value = arrays[field][index]
            if value is None:
                row[field] = None
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OpenMeteoAPIError(
                    f"{endpoint} response field {field!r} contains a non-numeric value."
                )
            elif not isfinite(value):
                raise OpenMeteoAPIError(
                    f"{endpoint} response field {field!r} contains a non-finite value."
                )
            else:
                row[field] = float(value)
        rows.append(row)
    return rows
