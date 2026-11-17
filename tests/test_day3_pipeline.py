"""Focused tests for Day 3 EDA and leakage-safe modeling helpers."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from aetherix_weather.eda import load_and_validate_features
from aetherix_weather.modeling import (
    _make_rain_classifier,
    _make_regressor,
    _metrics,
    _predict_probability,
    _target_values,
    _write_test_predictions,
    chronological_split,
    model_feature_columns,
    seasonal_naive_values,
    select_best_candidate,
    split_indices,
)


def _hourly_frame(count: int = 100) -> pd.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=index) for index in range(count)]
    return pd.DataFrame(
        {
            "timestamp_utc": [stamp.isoformat() for stamp in timestamps],
            "_timestamp_utc": timestamps,
            "temperature_c": np.arange(count, dtype=float),
            "precipitation_mm": np.zeros(count, dtype=float),
            "target_temperature_h01": np.arange(count, dtype=float) + 1,
            "target_wind_direction_sin_h01": np.zeros(count, dtype=float),
            "target_wind_direction_cos_h01": np.ones(count, dtype=float),
            "target_rain_occurred_h01": np.zeros(count, dtype=float),
        }
    )


def test_chronological_partitions_purge_each_horizon_label_window() -> None:
    boundaries = chronological_split(1000)
    splits = split_indices(boundaries, horizon=24)

    assert splits["train"][-1] + 24 < boundaries.train_end
    assert splits["validation"][-1] + 24 < boundaries.validation_end
    assert splits["train"][-1] < splits["validation"][0]
    assert splits["validation"][-1] < splits["test"][0]
    assert splits["test"][0] == boundaries.validation_end


def test_model_inputs_exclude_targets_and_archived_forecasts() -> None:
    frame = pd.DataFrame(
        {
            "temperature_c": [10.0, 11.0],
            "hour_sin": [0.0, 0.1],
            "target_temperature_h01": [11.0, 12.0],
            "historical_forecast_probability": [30.0, 40.0],
            "weather_source": ["archive", "archive"],
            "timestamp_utc": ["2026-01-01T00:00:00+00:00"] * 2,
        }
    )

    assert model_feature_columns(frame) == ["temperature_c", "hour_sin"]


def test_preprocessing_statistics_are_fitted_from_training_rows_only() -> None:
    train_features = pd.DataFrame({"signal": [1.0, 3.0, np.nan]})
    estimator = _make_regressor("ridge", alpha=1.0)
    estimator.fit(train_features, np.array([1.0, 3.0, 2.0]))
    imputer = estimator.named_steps["imputer"]

    assert imputer.statistics_.tolist() == [2.0]
    assert 1000.0 not in imputer.statistics_


def test_seasonal_naive_uses_exact_same_hour_previous_day() -> None:
    frame = _hourly_frame(80)
    origin = np.array([20])

    prediction = seasonal_naive_values(
        frame, origin_indices=origin, horizon=5, metric="temperature_c"
    )

    # Target index 25 minus 24 hours is index 1, not issue index 20.
    assert prediction.tolist() == [1.0]


def test_seasonal_naive_missing_previous_day_value_stays_unavailable() -> None:
    frame = _hourly_frame(80)
    frame.loc[1, "temperature_c"] = np.nan

    prediction = seasonal_naive_values(
        frame, origin_indices=np.array([20]), horizon=5, metric="temperature_c"
    )

    assert np.isnan(prediction[0])


def test_missing_targets_are_marked_invalid_not_imputed() -> None:
    frame = _hourly_frame(3)
    frame.loc[1, "target_temperature_h01"] = np.nan

    values, valid = _target_values(frame, "temperature_c", 1)

    assert valid.tolist() == [True, False, True]
    assert np.isnan(values[1])


def test_rain_labels_and_classifier_probabilities_are_valid() -> None:
    frame = _hourly_frame(3)
    frame["target_rain_occurred_h01"] = [0.0, 1.0, np.nan]
    labels, valid = _target_values(frame, "rain_occurred", 1)
    classifier = _make_rain_classifier()
    features = pd.DataFrame({"signal": np.arange(20, dtype=float)})
    classes = np.tile([0, 1], 10)
    classifier.fit(features, classes)
    probabilities = _predict_probability(classifier, features)

    assert labels[valid].astype(int).tolist() == [0, 1]
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_rain_classification_metrics_use_probability_scores() -> None:
    result = _metrics(
        "rain_occurred",
        np.array([0, 1, 1, 0]),
        np.array([0.1, 0.9, 0.8, 0.2]),
    )

    assert np.isclose(result["brier_score"], 0.025)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_regression_metrics_and_constant_target_r2_handling() -> None:
    result = _metrics("temperature_c", np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    constant = _metrics("precipitation_mm", np.zeros(3), np.zeros(3))

    assert result["mae"] == 0.5
    assert np.isclose(result["rmse"], np.sqrt(0.5))
    assert result["r2"] == -1.0
    assert constant["r2"] is None


def test_wind_direction_metrics_respect_zero_degree_wraparound() -> None:
    actual = np.array([[np.sin(np.deg2rad(359)), np.cos(np.deg2rad(359))]])
    predicted = np.array([[np.sin(np.deg2rad(1)), np.cos(np.deg2rad(1))]])

    result = _metrics("wind_direction_degrees", actual, predicted)

    assert np.isclose(result["circular_mae_degrees"], 2.0)
    assert result["r2"] is None


def test_selection_uses_validation_only_and_ignores_test_scores() -> None:
    selected = select_best_candidate(
        [
            {"split": "validation", "model": "ridge", "mae": 2.0},
            {"split": "validation", "model": "seasonal_naive", "mae": 3.0},
            {"split": "test", "model": "seasonal_naive", "mae": 0.0},
        ],
        "mae",
    )

    assert selected["model"] == "ridge"


def test_test_predictions_include_issue_and_target_utc_times() -> None:
    frame = _hourly_frame(80)
    predictions: list[dict[str, object]] = []
    _write_test_predictions(
        frame,
        task="temperature_c",
        horizon=2,
        origin_indices=np.array([30]),
        selected_name="ridge_alpha_1",
        predicted=np.array([32.5]),
        baseline=np.array([8.0]),
        actual=np.array([32.0]),
        output=predictions,
    )

    row = predictions[0]
    assert row["issue_time_utc"] == frame.iloc[30]["_timestamp_utc"].isoformat()
    assert row["target_time_utc"] == frame.iloc[32]["_timestamp_utc"].isoformat()
    assert row["forecast_horizon_hours"] == 2


def test_processed_dataset_rejects_non_hourly_rows(tmp_path) -> None:
    frame = _hourly_frame(3)
    frame.loc[2, "timestamp_utc"] = "2026-01-01T04:00:00+00:00"
    frame["timestamp_local"] = frame["timestamp_utc"]
    path = tmp_path / "features.csv"
    frame.drop(columns="_timestamp_utc").to_csv(path, index=False)

    with pytest.raises(ValueError, match="hourly timeline"):
        load_and_validate_features(path)
