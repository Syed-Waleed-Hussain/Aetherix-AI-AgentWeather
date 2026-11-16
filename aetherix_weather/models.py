"""Typed records returned by the hourly data clients."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class HourlyWeather:
    """Weather measurements for one local hour; unavailable values are None."""

    time: datetime
    temperature_c: float | None
    relative_humidity_percent: float | None
    precipitation_mm: float | None
    precipitation_probability_percent: float | None
    wind_speed_kmh: float | None
    wind_direction_degrees: float | None
    uv_index: float | None


@dataclass(frozen=True)
class HourlyAirQuality:
    """Air-quality measurements for one local hour; unavailable values are None."""

    time: datetime
    us_aqi: float | None
    pm2_5_ug_m3: float | None
    uv_index: float | None
