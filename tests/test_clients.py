"""Offline tests for Open-Meteo clients and response handling."""

from datetime import datetime
from unittest.mock import Mock

import pytest
import requests

from aetherix_weather import AirQualityClient, WeatherClient
from aetherix_weather.api import OpenMeteoAPIError
from aetherix_weather.config import (
    AIR_QUALITY_API_URL,
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    WEATHER_API_URL,
)


def make_hourly_payload(fields: tuple[str, ...]) -> dict[str, object]:
    hourly: dict[str, list[object]] = {
        "time": [f"2026-01-01T{hour:02}:00" for hour in range(24)]
    }
    for field in fields:
        hourly[field] = [float(index) for index in range(24)]
    return {"timezone": TIMEZONE, "hourly": hourly}


def mock_session(payload: object) -> tuple[Mock, Mock]:
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    session = Mock(spec=requests.Session)
    session.get.return_value = response
    return session, response


def test_weather_client_returns_24_local_hourly_values() -> None:
    fields = (
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "precipitation_probability",
        "wind_speed_10m",
        "wind_direction_10m",
        "uv_index",
    )
    session, _ = mock_session(make_hourly_payload(fields))

    result = WeatherClient(session=session).get_hourly_forecast()

    assert len(result) == 24
    assert result[0].time == datetime.fromisoformat("2026-01-01T00:00+05:00")
    for index, record in enumerate(result):
        assert record.time == datetime.fromisoformat(
            f"2026-01-01T{index:02}:00+05:00"
        )
        assert record.temperature_c == float(index)
        assert record.relative_humidity_percent == float(index)
        assert record.precipitation_mm == float(index)
        assert record.precipitation_probability_percent == float(index)
        assert record.wind_speed_kmh == float(index)
        assert record.wind_direction_degrees == float(index)
        assert record.uv_index == float(index)
    session.get.assert_called_once()
    assert session.get.call_args.args[0] == WEATHER_API_URL
    assert session.get.call_args.kwargs["params"]["latitude"] == LATITUDE
    assert session.get.call_args.kwargs["params"]["longitude"] == LONGITUDE
    assert session.get.call_args.kwargs["params"]["timezone"] == TIMEZONE
    assert session.get.call_args.kwargs["params"]["hourly"] == (
        "temperature_2m,relative_humidity_2m,precipitation,"
        "precipitation_probability,wind_speed_10m,wind_direction_10m,uv_index"
    )
    assert session.get.call_args.kwargs["params"]["forecast_hours"] == 24
    assert session.get.call_args.kwargs["params"]["temperature_unit"] == "celsius"
    assert session.get.call_args.kwargs["params"]["wind_speed_unit"] == "kmh"
    assert session.get.call_args.kwargs["params"]["precipitation_unit"] == "mm"


def test_air_quality_client_preserves_missing_values() -> None:
    fields = ("us_aqi", "pm2_5", "uv_index")
    payload = make_hourly_payload(fields)
    payload["hourly"]["us_aqi"][0] = None  # type: ignore[index]
    session, _ = mock_session(payload)

    result = AirQualityClient(session=session).get_hourly_forecast()

    assert len(result) == 24
    for index, record in enumerate(result):
        assert record.time == datetime.fromisoformat(
            f"2026-01-01T{index:02}:00+05:00"
        )
        assert record.us_aqi == (None if index == 0 else float(index))
        assert record.pm2_5_ug_m3 == float(index)
        assert record.uv_index == float(index)
    assert session.get.call_args.args[0] == AIR_QUALITY_API_URL
    assert session.get.call_args.kwargs["params"]["hourly"] == "us_aqi,pm2_5,uv_index"
    assert session.get.call_args.kwargs["params"]["forecast_hours"] == 24


def test_client_raises_clear_error_for_http_failure() -> None:
    session, response = mock_session({})
    response.raise_for_status.side_effect = requests.HTTPError(
        response=Mock(status_code=503)
    )

    with pytest.raises(OpenMeteoAPIError, match="HTTP 503"):
        WeatherClient(session=session).get_hourly_forecast()


def test_client_raises_clear_error_for_timeout() -> None:
    session, _ = mock_session({})
    session.get.side_effect = requests.Timeout("slow")

    with pytest.raises(OpenMeteoAPIError, match="timed out"):
        AirQualityClient(session=session).get_hourly_forecast()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "unexpected response format"),
        ({"error": True, "reason": "Invalid parameter"}, "Invalid parameter"),
        (
            {"timezone": TIMEZONE, "hourly": {"time": ["2026-01-01T00:00"]}},
            "exactly 24 hourly timestamps",
        ),
    ],
)
def test_client_rejects_invalid_api_responses(payload: object, message: str) -> None:
    session, _ = mock_session(payload)

    with pytest.raises(OpenMeteoAPIError, match=message):
        WeatherClient(session=session).get_hourly_forecast()
