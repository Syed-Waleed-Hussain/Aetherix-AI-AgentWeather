"""Fetch live Open-Meteo data to verify network access and API configuration."""

import sys

from aetherix_weather import AirQualityClient, WeatherClient
from aetherix_weather.api import OpenMeteoAPIError


def main() -> int:
    try:
        weather = WeatherClient().get_hourly_forecast()
        air_quality = AirQualityClient().get_hourly_forecast()
    except OpenMeteoAPIError as exc:
        print(f"Open-Meteo smoke test failed: {exc}", file=sys.stderr)
        return 1

    first_weather = weather[0]
    first_air_quality = air_quality[0]
    print(f"Weather: {len(weather)} hourly records; first at {first_weather.time:%Y-%m-%d %H:%M}")
    print(
        "  "
        f"temperature={first_weather.temperature_c} C, "
        f"humidity={first_weather.relative_humidity_percent}%, "
        f"precipitation={first_weather.precipitation_mm} mm, "
        f"precipitation_probability={first_weather.precipitation_probability_percent}%, "
        f"wind={first_weather.wind_speed_kmh} km/h at "
        f"{first_weather.wind_direction_degrees} degrees, "
        f"UV={first_weather.uv_index}"
    )
    print(
        f"Air quality: {len(air_quality)} hourly records; "
        f"first at {first_air_quality.time:%Y-%m-%d %H:%M}"
    )
    print(
        f"  US AQI={first_air_quality.us_aqi}, "
        f"PM2.5={first_air_quality.pm2_5_ug_m3} ug/m3, "
        f"UV index={first_air_quality.uv_index}"
    )
    print("These are Open-Meteo forecasts, not predictions from an AETHERIX AI model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
