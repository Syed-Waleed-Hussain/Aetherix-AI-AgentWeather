"""Open-Meteo hourly weather forecast client."""

import requests

from .api import hourly_records, request_json
from .config import (
    LATITUDE,
    LONGITUDE,
    REQUEST_TIMEOUT_SECONDS,
    TIMEZONE,
    WEATHER_API_URL,
)
from .models import HourlyWeather

_WEATHER_FIELDS = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "wind_speed_10m",
    "wind_direction_10m",
    "uv_index",
)


class WeatherClient:
    """Fetch 24 hourly weather values for a location from Open-Meteo."""

    def __init__(
        self,
        latitude: float = LATITUDE,
        longitude: float = LONGITUDE,
        timezone: str = TIMEZONE,
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
        self.timezone = timezone
        self.timeout = timeout
        self.session = session or requests.Session()

    def get_hourly_forecast(self) -> list[HourlyWeather]:
        """Return 24 local-time forecast entries from the Open-Meteo API."""
        payload = request_json(
            self.session,
            WEATHER_API_URL,
            {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "hourly": ",".join(_WEATHER_FIELDS),
                "forecast_hours": 24,
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
                "precipitation_unit": "mm",
                "timezone": self.timezone,
            },
            self.timeout,
        )
        rows = hourly_records(
            payload, "Weather", _WEATHER_FIELDS, timezone_name=self.timezone
        )
        return [
            HourlyWeather(
                time=row["time"],
                temperature_c=row["temperature_2m"],
                relative_humidity_percent=row["relative_humidity_2m"],
                precipitation_mm=row["precipitation"],
                precipitation_probability_percent=row["precipitation_probability"],
                wind_speed_kmh=row["wind_speed_10m"],
                wind_direction_degrees=row["wind_direction_10m"],
                uv_index=row["uv_index"],
            )
            for row in rows
        ]
