from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from aetherix_weather.statistical import (
    DEFAULT_CONFIG,
    SarimaxConfig,
    fit_sarimax,
    forecast_issue_origins,
    load_sarimax_artifact,
    regression_metrics,
    run_sarimax_experiment,
)


def _series(length: int) -> np.ndarray:
    index = np.arange(length, dtype=float)
    noise = np.random.default_rng(42).normal(0, 0.1, length)
    return 20 + 0.02 * index + 2 * np.sin(2 * np.pi * index / 24) + noise


def _config() -> SarimaxConfig:
    return SarimaxConfig(
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
        maxiter=50,
    )


def test_sarimax_forecast_shape_and_reproducibility() -> None:
    observations = _series(180)
    first = fit_sarimax(observations[:140], _config())
    second = fit_sarimax(observations[:140], _config())

    first_forecast = np.asarray(first.get_forecast(steps=24).predicted_mean)
    second_forecast = np.asarray(second.get_forecast(steps=24).predicted_mean)

    assert first_forecast.shape == (24,)
    np.testing.assert_allclose(first_forecast, second_forecast, rtol=0, atol=1e-8)


def test_forecast_state_uses_no_observations_after_issue_time() -> None:
    observations = _series(180)
    fitted = fit_sarimax(observations[:120], _config())
    changed_future = observations.copy()
    changed_future[151:] += 1000

    original = forecast_issue_origins(fitted, observations, [150], max_horizon=6)[150]
    changed = forecast_issue_origins(
        fitted, changed_future, [150], max_horizon=6
    )[150]

    np.testing.assert_allclose(original, changed, rtol=0, atol=1e-10)


def test_missing_observations_and_metric_calculation() -> None:
    observations = _series(180)
    observations[130] = np.nan
    fitted = fit_sarimax(observations[:140], _config())
    forecasts = forecast_issue_origins(fitted, observations, [150], max_horizon=24)

    assert np.isfinite(forecasts[150]).all()
    metrics = regression_metrics(
        np.array([1.0, 2.0, np.nan]), np.array([1.0, 3.0, 4.0])
    )
    assert metrics["n"] == 2
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(1 / np.sqrt(2))
    assert metrics["r2"] == pytest.approx(-1.0)
    assert regression_metrics(np.ones(2), np.ones(2))["r2"] is None


def test_sarimax_experiment_writes_aligned_forecasts_and_metadata(
    tmp_path,
) -> None:
    length = 600
    observations = _series(length)
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                "2024-01-01", periods=length, freq="h", tz="UTC"
            ).astype(str),
            "temperature_c": observations,
        }
    )
    for horizon in (1, 6):
        frame[f"target_temperature_h{horizon:02}"] = np.r_[
            observations[horizon:],
            np.full(horizon, np.nan),
        ]
    dataset_path = tmp_path / "features.csv"
    frame.to_csv(dataset_path, index=False)
    result = run_sarimax_experiment(
        frame,
        dataset_path,
        tmp_path / "output",
        horizons=(1, 6),
        max_origins_per_split=12,
        config=_config(),
    )

    assert result["metadata"]["model"] == "sarimax_ar1"
    assert result["metadata"]["config"]["seasonal_order"] == (0, 0, 0, 0)
    assert result["metadata"]["optimizer_converged"] is True
    assert result["metadata"]["dataset_sha256"]
    expected_schema_hash = hashlib.sha256(
        json.dumps(list(frame.columns), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert result["metadata"]["feature_schema_sha256"] == expected_schema_hash
    assert len(result["metadata"]["model_artifact_sha256"]) == 64
    persisted_payload = joblib.load(result["artifact_path"])
    assert persisted_payload["format_version"] == 1
    persisted_model = load_sarimax_artifact(result["artifact_path"])
    assert (
        persisted_model.nobs
        == result["metadata"]["training_observation_count"]
    )
    assert np.isfinite(persisted_model.get_forecast(steps=2).predicted_mean).all()
    assert (tmp_path / "output" / "statistical_evaluation.csv").is_file()
    assert (tmp_path / "output" / "statistical_metadata.json").is_file()
    assert (
        tmp_path / "output" / "temperature_c_sarimax_ar1.joblib"
    ).is_file()
    assert len(result["forecasts"]) > 0
    first = result["forecasts"][0]
    issue = pd.Timestamp(first["issue_time_utc"])
    target = pd.Timestamp(first["target_time_utc"])
    assert target - issue == pd.Timedelta(
        first["horizon_hours"], unit="h"
    )
    assert first["metric"] == "temperature_c"
    assert first["model_version"] == "1"


def test_sarimax_rejects_insufficient_training_observations() -> None:
    with pytest.raises(ValueError, match="100 training observations"):
        fit_sarimax(np.arange(99, dtype=float), DEFAULT_CONFIG)
