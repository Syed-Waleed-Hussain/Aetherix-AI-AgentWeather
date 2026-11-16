"""Open-Meteo hourly air-quality forecast client."""

import requests

from .api import hourly_records, request_json
from .config import (
    AIR_QUALITY_API_URL,
    LATITUDE,
    LONGITUDE,
    REQUEST_TIMEOUT_SECONDS,
    TIMEZONE,
)
from .models import HourlyAirQuality

_AIR_QUALITY_FIELDS = ("us_aqi", "pm2_5", "uv_index")


class AirQualityClient:
    """Fetch 24 hourly US AQI, PM2.5, and UV Index values from Open-Meteo."""

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

    def get_hourly_forecast(self) -> list[HourlyAirQuality]:
        """Return 24 local-time forecast entries from the Open-Meteo API."""
        payload = request_json(
            self.session,
            AIR_QUALITY_API_URL,
            {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "hourly": ",".join(_AIR_QUALITY_FIELDS),
                "forecast_hours": 24,
                "timezone": self.timezone,
            },
            self.timeout,
        )
        rows = hourly_records(
            payload, "Air quality", _AIR_QUALITY_FIELDS, timezone_name=self.timezone
        )
        return [
            HourlyAirQuality(
                time=row["time"],
                us_aqi=row["us_aqi"],
                pm2_5_ug_m3=row["pm2_5"],
                uv_index=row["uv_index"],
            )
            for row in rows
        ]
