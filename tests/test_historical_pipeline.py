"""Deterministic tests for historical ingestion and time-series processing."""

import json
from datetime import date, datetime, timedelta, timezone
from math import isclose
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from aetherix_weather.api import OpenMeteoAPIError
from aetherix_weather.dataset import (
    DataQualityError,
    align_hourly,
    build_feature_dataset,
    create_features,
    create_targets,
    data_quality_report,
    handle_missing,
    missing_summary,
)
from aetherix_weather.historical import (
    AIR_QUALITY_HOURLY_FIELDS,
    ARCHIVE_WEATHER_API_URL,
    HISTORICAL_FORECAST_API_URL,
    HISTORICAL_FORECAST_HOURLY_FIELDS,
    WEATHER_HOURLY_FIELDS,
    HistoricalAirQualityClient,
    HistoricalForecastClient,
    HistoricalWeatherClient,
    latest_full_historical_day,
    parse_date,
    validate_date_range,
)
from aetherix_weather.storage import upsert_csv

START = "2026-01-01"
LOCAL_ZONE = timezone(timedelta(hours=5))


def _session_for(payload: object) -> tuple[Mock, Mock]:
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    session = Mock(spec=requests.Session)
    session.get.return_value = response
    return session, response


def _payload(fields: tuple[str, ...], count: int = 24) -> dict[str, object]:
    hourly: dict[str, list[object]] = {
        "time": [
            f"2026-01-{1 + index // 24:02}T{index % 24:02}:00"
            for index in range(count)
        ]
    }
    for field in fields:
        hourly[field] = [float(index) for index in range(count)]
    return {"timezone": "Asia/Karachi", "hourly": hourly}


def _source_rows(count: int = 30) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    weather: list[dict[str, object]] = []
    air: list[dict[str, object]] = []
    local_start = datetime(2026, 1, 1, tzinfo=LOCAL_ZONE)
    for index in range(count):
        local = local_start + timedelta(hours=index)
        utc = local.astimezone(timezone.utc)
        base = {
            "timestamp_utc": utc.isoformat(),
            "timestamp_local": local.isoformat(),
        }
        weather.append(
            {
                **base,
                "source": "weather",
                "ingested_at_utc": "2026-01-02T00:00:00+00:00",
                "temperature_c": float(index),
                "relative_humidity_percent": 50.0,
                "precipitation_mm": 1.0 if index == 2 else 0.0,
                "precipitation_probability_percent": None,
                "wind_speed_kmh": 3.0,
                "wind_direction_degrees": 359.0 if index == 0 else 0.0,
                "uv_index": 1.0,
            }
        )
        air.append(
            {
                **base,
                "source": "air",
                "ingested_at_utc": "2026-01-02T00:00:00+00:00",
                "us_aqi": 20.0 + index,
                "pm2_5_ug_m3": 5.0 + index,
            }
        )
    return weather, air


def test_historical_weather_request_uses_archive_endpoint_and_supported_fields() -> None:
    session, _ = _session_for(_payload(WEATHER_HOURLY_FIELDS))

    result = HistoricalWeatherClient(session=session).fetch(START, "2026-01-01")

    assert len(result.records) == 24
    assert result.records[0]["temperature_c"] == 0.0
    assert result.records[0]["uv_index"] == 0.0
    assert result.records[0]["precipitation_probability_percent"] == 0.0
    assert result.availability["precipitation_probability_percent"] == "available"
    assert session.get.call_args.args[0] == ARCHIVE_WEATHER_API_URL
    params = session.get.call_args.kwargs["params"]
    assert params["hourly"] == ",".join(WEATHER_HOURLY_FIELDS)
    assert params["start_date"] == START
    assert params["end_date"] == START


def test_missing_historical_probability_values_are_reported() -> None:
    payload = _payload(WEATHER_HOURLY_FIELDS)
    payload["hourly"]["precipitation_probability"] = [None] * 24  # type: ignore[index]
    session, _ = _session_for(payload)

    result = HistoricalWeatherClient(session=session).fetch(START, START)

    assert result.availability["precipitation_probability_percent"] == (
        "no_historical_values"
    )
    assert all(
        row["precipitation_probability_percent"] is None for row in result.records
    )


def test_available_api_field_with_null_values_is_marked_unavailable() -> None:
    payload = _payload(WEATHER_HOURLY_FIELDS)
    payload["hourly"]["uv_index"] = [None] * 24  # type: ignore[index]
    session, _ = _session_for(payload)

    result = HistoricalWeatherClient(session=session).fetch(START, START)

    assert result.availability["uv_index"] == "no_historical_values"
    assert all(row["uv_index"] is None for row in result.records)


def test_historical_air_quality_request_includes_aqi_and_pm25() -> None:
    session, _ = _session_for(_payload(AIR_QUALITY_HOURLY_FIELDS))

    result = HistoricalAirQualityClient(session=session).fetch(START, START)

    params = session.get.call_args.kwargs["params"]
    assert params["hourly"] == "us_aqi,pm2_5,uv_index"
    assert result.records[2]["us_aqi"] == 2.0
    assert result.records[2]["pm2_5_ug_m3"] == 2.0
    assert result.records[2]["uv_index"] == 2.0
    assert result.availability["us_aqi"] == "available_modeled"
    assert result.availability["uv_index"] == "available_modeled"
    assert result.units["uv_index"] == "index"


def test_unavailable_historical_air_quality_uv_is_not_filled() -> None:
    payload = _payload(AIR_QUALITY_HOURLY_FIELDS)
    payload["hourly"]["uv_index"] = [None] * 24  # type: ignore[index]
    session, _ = _session_for(payload)

    result = HistoricalAirQualityClient(session=session).fetch(START, START)

    assert result.availability["uv_index"] == "no_historical_values"
    assert all(row["uv_index"] is None for row in result.records)


def test_historical_forecast_preserves_valid_time_and_unknown_issue_metadata() -> None:
    payload = _payload(HISTORICAL_FORECAST_HOURLY_FIELDS)
    payload.update({"latitude": 24.85, "longitude": 66.99})
    session, _ = _session_for(payload)

    result = HistoricalForecastClient(session=session).fetch(START, START)
    row = result.records[0]

    assert session.get.call_args.args[0] == HISTORICAL_FORECAST_API_URL
    assert session.get.call_args.kwargs["params"]["hourly"] == (
        "precipitation_probability"
    )
    assert row["forecast_valid_timestamp_utc"] == row["timestamp_utc"]
    assert row["forecast_issued_at_utc"] is None
    assert row["forecast_run_id"] is None
    assert row["forecast_lead_hours"] is None
    assert row["forecast_model_selection"] == "default_best_match"
    assert row["forecast_api_latitude"] == 24.85
    assert result.availability["forecast_precipitation_probability_percent"] == (
        "available_forecast"
    )
    assert json.loads(row["forecast_model_metadata"]) == {
        "model": None,
        "models": None,
    }
    assert "target_precipitation_probability_percent_h01" not in create_targets(
        result.records
    )[0]
    assert "target_rain_occurred_h01" not in create_targets(result.records)[0]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(START, date(2026, 1, 1)), (date(2026, 1, 2), date(2026, 1, 2))],
)
def test_date_range_parsing_and_validation(value: object, expected: date) -> None:
    assert parse_date(value) == expected  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="on or after"):
        validate_date_range("2026-01-02", START)
    with pytest.raises(ValueError, match="on or before"):
        validate_date_range(START, "2026-01-03", latest_allowed=date(2026, 1, 2))
    with pytest.raises(ValueError, match="time component"):
        parse_date(datetime(2026, 1, 1))


def test_latest_historical_date_uses_conservative_six_day_buffer() -> None:
    assert latest_full_historical_day(date(2026, 1, 10)) == date(2026, 1, 4)


def test_historical_client_rejects_missing_or_malformed_variable_arrays() -> None:
    payload = _payload(WEATHER_HOURLY_FIELDS)
    payload["hourly"]["temperature_2m"].pop()  # type: ignore[index]
    session, _ = _session_for(payload)

    with pytest.raises(OpenMeteoAPIError, match="hourly values"):
        HistoricalWeatherClient(session=session).fetch(START, START)


def test_historical_client_rejects_duplicate_timestamps() -> None:
    payload = _payload(WEATHER_HOURLY_FIELDS)
    payload["hourly"]["time"][1] = payload["hourly"]["time"][0]  # type: ignore[index]
    session, _ = _session_for(payload)

    with pytest.raises(OpenMeteoAPIError, match="duplicate timestamps"):
        HistoricalWeatherClient(session=session).fetch(START, START)


def test_historical_client_rejects_invalid_json() -> None:
    session, response = _session_for({})
    response.json.side_effect = ValueError("not JSON")

    with pytest.raises(OpenMeteoAPIError, match="invalid JSON"):
        HistoricalWeatherClient(session=session).fetch(START, START)


def test_weather_local_timestamps_are_converted_to_utc_without_shifting() -> None:
    session, _ = _session_for(_payload(WEATHER_HOURLY_FIELDS))
    record = HistoricalWeatherClient(session=session).fetch(START, START).records[0]

    assert record["timestamp_local"] == "2026-01-01T00:00:00+05:00"
    assert record["timestamp_utc"] == "2025-12-31T19:00:00+00:00"


def test_alignment_uses_exact_utc_outer_join_and_preserves_source_only_rows() -> None:
    weather, air = _source_rows(3)
    air = air[1:]
    combined, info = align_hourly(weather, air)

    assert len(combined) == 3
    assert combined[0]["us_aqi"] is None
    assert combined[0]["temperature_c"] == 0.0
    assert combined[2]["timestamp_utc"] == weather[2]["timestamp_utc"]
    assert info["weather_only_timestamps"] == [weather[0]["timestamp_utc"]]
    assert info["air_quality_only_timestamps"] == []


def test_alignment_uses_cloud_aware_air_quality_uv_with_provenance() -> None:
    weather, air = _source_rows(1)
    weather[0]["uv_index"] = None
    air[0]["uv_index"] = 5.9
    air[0]["source"] = "open-meteo-modeled-air-quality"

    combined, _ = align_hourly(weather, air)

    assert combined[0]["uv_index"] == 5.9
    assert combined[0]["weather_uv_index"] is None
    assert combined[0]["air_quality_uv_index"] == 5.9
    assert combined[0]["uv_index_source"] == "open-meteo-modeled-air-quality"


def test_alignment_rejects_duplicate_and_inconsistent_timezone_timestamps() -> None:
    weather, air = _source_rows(2)
    with pytest.raises(DataQualityError, match="duplicate"):
        align_hourly(weather + [weather[0]], air)

    broken = [dict(weather[0]), weather[1]]
    broken[0]["timestamp_local"] = "2026-01-01T00:00:00+00:00"
    with pytest.raises(DataQualityError, match="UTC/local"):
        align_hourly(broken, air)


def test_missing_summary_and_forward_fill_keep_rain_and_aqi_unfilled() -> None:
    weather, air = _source_rows(3)
    weather[1]["temperature_c"] = None
    weather[1]["precipitation_mm"] = None
    air[1]["us_aqi"] = None
    combined, _ = align_hourly(weather, air)

    cleaned, policy, summaries = handle_missing(combined, max_forward_fill_hours=1)

    assert summaries["before"]["metrics"]["temperature_c"]["missing_count"] == 1
    assert summaries["after"]["metrics"]["temperature_c"]["missing_count"] == 0
    assert cleaned[1]["temperature_c"] == 0.0
    assert cleaned[1]["temperature_c_imputed"] is True
    assert cleaned[1]["precipitation_mm"] is None
    assert cleaned[1]["us_aqi"] is None
    assert policy["imputed_counts"]["temperature_c"] == 1
    assert missing_summary(cleaned)["metrics"]["us_aqi"]["missing_count"] == 1


def test_features_include_calendar_lag_rolling_and_rate_of_change() -> None:
    weather, air = _source_rows(30)
    combined, _ = align_hourly(weather, air)
    features = create_features(combined)
    row = features[10]

    assert row["hour_of_day"] == 10
    assert row["day_of_week"] == 3
    assert row["is_weekend"] == 0
    assert isclose(row["temperature_c_lag_3h"], 7.0)
    assert isclose(row["temperature_c_rolling_mean_3h"], 9.0)
    assert isclose(row["temperature_c_change_1h"], 1.0)
    assert row["temperature_c_rolling_mean_24h"] is None
    assert row["temperature_c_rolling_std_24h"] is None
    assert "hour_sin" in row and "day_of_year_cos" in row


def test_lags_and_rolling_features_do_not_bridge_hourly_gaps() -> None:
    weather, air = _source_rows(5)
    weather.pop(2)
    air.pop(2)
    combined, _ = align_hourly(weather, air)
    features = create_features(combined)

    assert features[2]["temperature_c_lag_1h"] is None
    assert features[2]["temperature_c_rolling_mean_3h"] is None


def test_wind_direction_features_and_targets_use_circular_components() -> None:
    weather, air = _source_rows(30)
    combined, _ = align_hourly(weather, air)
    features = create_features(combined)
    targets = create_targets(features)

    assert abs(features[0]["wind_direction_sin"]) < 0.02
    assert features[0]["wind_direction_cos"] > 0.99
    assert isclose(
        targets[0]["target_wind_direction_sin_h01"],
        0.0,
        abs_tol=1e-12,
    )
    assert targets[0]["target_wind_direction_cos_h01"] > 0.99
    assert "target_precipitation_probability_percent_h01" not in targets[0]


def test_targets_use_exact_future_hours_and_leave_incomplete_tail_missing() -> None:
    weather, air = _source_rows(30)
    combined, _ = align_hourly(weather, air)
    targets = create_targets(combined)

    assert targets[0]["target_temperature_h01"] == 1.0
    assert targets[0]["target_temperature_h24"] == 24.0
    assert targets[-1]["target_temperature_h01"] is None
    assert all(
        row["target_temperature_h24"] is None for row in targets[-24:]
    )
    assert targets[0]["target_rain_occurred_h01"] == 0
    assert targets[0]["target_rain_occurred_h02"] == 1


def test_missing_future_precipitation_keeps_rain_occurrence_target_missing() -> None:
    weather, air = _source_rows(3)
    weather[1]["precipitation_mm"] = None
    combined, _ = align_hourly(weather, air)

    targets = create_targets(combined)

    assert targets[0]["target_rain_occurred_h01"] is None


def test_features_are_causal_when_future_values_change() -> None:
    weather, air = _source_rows(30)
    combined, _ = align_hourly(weather, air)
    baseline = create_features(combined)
    changed = [dict(row) for row in combined]
    changed[10]["temperature_c"] = 9999.0
    recalculated = create_features(changed)

    assert baseline[9]["temperature_c_rolling_mean_3h"] == recalculated[9][
        "temperature_c_rolling_mean_3h"
    ]
    assert baseline[9]["temperature_c_lag_1h"] == recalculated[9][
        "temperature_c_lag_1h"
    ]
    assert baseline[9]["temperature_c_lag_1h"] == 8.0


def test_repeated_ingestion_upserts_without_duplicate_timestamps(tmp_path: Path) -> None:
    weather, _ = _source_rows(2)
    path = tmp_path / "weather.csv"

    assert upsert_csv(path, weather) == 2
    assert upsert_csv(path, weather) == 2
    weather[0]["temperature_c"] = 123.0
    assert upsert_csv(path, weather[:1]) == 2
    contents = path.read_text(encoding="utf-8")
    assert contents.count("2025-12-31T19:00:00+00:00") == 1
    assert "123.0" in contents


def test_quality_report_flags_invalid_values_and_hourly_gaps() -> None:
    weather, air = _source_rows(3)
    weather[0]["relative_humidity_percent"] = 120.0
    weather.pop(1)
    air.pop(1)
    combined, alignment = align_hourly(weather, air)
    cleaned, imputation, missing = handle_missing(combined)
    report = data_quality_report(
        cleaned,
        alignment,
        missing,
        imputation,
        create_targets(cleaned),
        {"precipitation_probability_percent": "no_historical_values"},
        START,
        START,
    )

    assert report["missing_hour_count"] == 1
    assert any(issue["code"] == "missing_hour" for issue in report["quality_issues"])
    assert any(
        issue["code"] == "invalid_relative_humidity_percent"
        for issue in report["quality_issues"]
    )
    assert report["duplicate_timestamps"] == {"weather": 0, "air_quality": 0}
    assert report["unavailable_variables"] == [
        {
            "name": "precipitation_probability_percent",
            "status": "no_historical_values",
        }
    ]
    assert report["target_definitions"]["rain_occurred"]["threshold_mm"] == 0.0
    assert "not station-observed" in report["target_definitions"]["rain_occurred"][
        "source_semantics"
    ]


def test_quality_report_flags_negative_rain_wind_and_out_of_range_aqi() -> None:
    weather, air = _source_rows(5)
    weather[0]["precipitation_mm"] = -0.1
    weather[1]["wind_speed_kmh"] = -1.0
    weather[2]["wind_direction_degrees"] = 361.0
    air[3]["us_aqi"] = 501.0
    combined, alignment = align_hourly(weather, air)
    cleaned, imputation, missing = handle_missing(combined)
    report = data_quality_report(
        cleaned,
        alignment,
        missing,
        imputation,
        create_targets(cleaned),
        {},
        START,
        START,
    )

    codes = {issue["code"] for issue in report["quality_issues"]}
    assert "invalid_precipitation_mm" in codes
    assert "invalid_wind_speed_kmh" in codes
    assert "invalid_wind_direction_degrees" in codes
    assert "invalid_us_aqi" in codes


def test_short_source_coverage_is_reported_after_feature_rebuild() -> None:
    weather, air = _source_rows(3)

    _, report = build_feature_dataset(
        weather,
        air,
        requested_start_date="2026-01-01",
        requested_end_date="2026-01-02",
    )

    assert report["returned_date_ranges_local"]["weather"] == {
        "start": "2026-01-01",
        "end": "2026-01-01",
    }
    assert len(report["coverage_limitations"]) == 2
