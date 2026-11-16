"""Weather and air-quality data clients for AETHERIX AI."""

from .air_quality import AirQualityClient
from .weather import WeatherClient

__all__ = ["AirQualityClient", "WeatherClient"]
