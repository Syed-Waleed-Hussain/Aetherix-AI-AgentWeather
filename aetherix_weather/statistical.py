"""Leakage-safe statistical time-series forecasts for supported numeric targets."""

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .modeling import (
    MAX_HORIZON_HOURS,
    _target_values,
    chronological_split,
    seasonal_naive_values,
    split_indices,
)
from .storage import write_csv_atomic, write_json_atomic, write_text_atomic

STATISTICAL_MODEL_ID = "sarimax_daily_ar1"
STATISTICAL_MODEL_VERSION = "1"
SUPPORTED_METRICS = (
    "temperature_c",
    "relative_humidity_percent",
    "wind_speed_kmh",
    "us_aqi",
    "uv_index",
)


@dataclass(frozen=True)
class SarimaxConfig:
    order: tuple[int, int, int] = (1, 0, 0)
    seasonal_order: tuple[int, int, int, int] = (1, 0, 0, 24)
    trend: str = "c"
    maxiter: int = 40
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False


DEFAULT_CONFIG = SarimaxConfig()


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Return regression metrics after excluding non-finite target/prediction pairs."""
    truth = np.asarray(actual, dtype=float).reshape(-1)
    estimate = np.asarray(predicted, dtype=float).reshape(-1)
    if truth.shape != estimate.shape:
        raise ValueError("actual and predicted arrays must have the same shape.")
    valid = np.isfinite(truth) & np.isfinite(estimate)
    truth = truth[valid]
    estimate = estimate[valid]
    if not len(truth):
        raise ValueError("At least one finite actual/prediction pair is required.")
    errors = estimate - truth
    return {
        "n": int(len(truth)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "r2": (
            float(1 - np.sum(np.square(errors)) / np.sum(np.square(truth - truth.mean())))
            if len(truth) > 1 and np.ptp(truth) > 0
            else None
        ),
    }


def fit_sarimax(
    observations: np.ndarray, config: SarimaxConfig = DEFAULT_CONFIG
) -> Any:
    """Fit the prespecified SARIMAX configuration; NaNs are handled by the Kalman filter."""
    values = np.asarray(observations, dtype=float).reshape(-1)
    if len(values) < 100:
        raise ValueError("At least 100 training observations are required.")
    if np.isinf(values).any():
        raise ValueError("Training observations must not contain infinite values.")
    if np.count_nonzero(np.isfinite(values)) < 100:
        raise ValueError("At least 100 finite training observations are required.")
    if config.maxiter < 1:
        raise ValueError("maxiter must be positive.")
    model = SARIMAX(
        values,
        order=config.order,
        seasonal_order=config.seasonal_order,
        trend=config.trend,
        enforce_stationarity=config.enforce_stationarity,
        enforce_invertibility=config.enforce_invertibility,
    )
    fitted = model.fit(disp=False, maxiter=config.maxiter)
    if not bool(fitted.mle_retvals.get("converged", False)):
        iterations = fitted.mle_retvals.get("iterations")
        raise RuntimeError(
            "SARIMAX optimizer did not converge "
            f"within {config.maxiter} iterations (completed {iterations})."
        )
    return fitted


def forecast_issue_origins(
    fitted: Any,
    observations: np.ndarray,
    issue_indices: list[int] | tuple[int, ...] | np.ndarray,
    max_horizon: int = MAX_HORIZON_HOURS,
) -> dict[int, np.ndarray]:
    """Forecast from each issue time after updating state only with observations through it."""
    values = np.asarray(observations, dtype=float).reshape(-1)
    if not 1 <= max_horizon <= MAX_HORIZON_HOURS:
        raise ValueError("max_horizon must be between 1 and 24.")
    origins = [int(index) for index in issue_indices]
    if origins != sorted(set(origins)):
        raise ValueError("issue_indices must be unique and strictly increasing.")
    if any(index < 0 or index >= len(values) for index in origins):
        raise ValueError("An issue index is outside the observation series.")
    next_observation = int(fitted.nobs)
    if origins and origins[0] < next_observation - 1:
        raise ValueError("Issue time precedes the fitted training observations.")

    state = fitted
    forecasts: dict[int, np.ndarray] = {}
    for origin in origins:
        while next_observation <= origin:
            state = state.extend(np.asarray([values[next_observation]], dtype=float))
            next_observation += 1
        prediction = np.asarray(
            state.get_forecast(steps=max_horizon).predicted_mean, dtype=float
        ).reshape(-1)
        if len(prediction) != max_horizon or not np.isfinite(prediction).all():
            raise RuntimeError(
                f"SARIMAX returned invalid forecast output at issue index {origin}."
            )
        forecasts[origin] = prediction
    return forecasts


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump_joblib_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".joblib", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        joblib.dump(value, temporary_path)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def load_sarimax_artifact(path: Path | str) -> Any:
    """Rebuild a filtered SARIMAX result from its compact saved parameters and history."""
    payload = joblib.load(path)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Unsupported or invalid SARIMAX artifact format.")
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("SARIMAX artifact is missing its configuration.")
    try:
        config = SarimaxConfig(
            order=tuple(raw_config["order"]),
            seasonal_order=tuple(raw_config["seasonal_order"]),
            trend=raw_config["trend"],
            maxiter=int(raw_config["maxiter"]),
            enforce_stationarity=bool(raw_config["enforce_stationarity"]),
            enforce_invertibility=bool(raw_config["enforce_invertibility"]),
        )
        observations = np.asarray(payload["training_observations"], dtype=float)
        params = np.asarray(payload["params"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("SARIMAX artifact has invalid model data.") from exc
    model = SARIMAX(
        observations,
        order=config.order,
        seasonal_order=config.seasonal_order,
        trend=config.trend,
        enforce_stationarity=config.enforce_stationarity,
        enforce_invertibility=config.enforce_invertibility,
    )
    return model.filter(params)


def run_sarimax_experiment(
    frame: pd.DataFrame,
    dataset_path: Path,
    output_dir: Path,
    metric: str = "temperature_c",
    horizons: tuple[int, ...] = (1, 6, 12, 24),
    max_origins_per_split: int | None = None,
    config: SarimaxConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Fit once on purged training observations and evaluate rolling issue-time forecasts."""
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported SARIMAX metric {metric!r}; supported metrics: "
            f"{', '.join(SUPPORTED_METRICS)}."
        )
    model_id = (
        STATISTICAL_MODEL_ID
        if config.seasonal_order[3] == 24
        else "sarimax_ar1"
    )
    if not horizons or any(
        horizon < 1 or horizon > MAX_HORIZON_HOURS for horizon in horizons
    ):
        raise ValueError("horizons must be selected from 1 through 24.")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must not contain duplicates.")
    if max_origins_per_split is not None and max_origins_per_split < 1:
        raise ValueError("max_origins_per_split must be positive when specified.")
    if "timestamp_utc" not in frame or metric not in frame:
        raise ValueError("timestamp_utc and the selected observed metric are required.")

    working = frame.reset_index(drop=True).copy()
    timestamps = pd.to_datetime(working["timestamp_utc"], utc=True, errors="raise")
    working["_timestamp_utc"] = timestamps
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("SARIMAX requires unique, ordered UTC timestamps.")
    elapsed_seconds = timestamps.diff().dropna().dt.total_seconds().to_numpy()
    if not np.all(elapsed_seconds == 3600):
        raise ValueError("SARIMAX requires a continuous hourly observation series.")
    observations = pd.to_numeric(working[metric], errors="coerce").to_numpy(dtype=float)
    if np.isinf(observations).any():
        raise ValueError(f"{metric} contains infinite observations.")

    started_at = time.monotonic()
    boundaries = chronological_split(len(working))
    maximum_horizon = max(horizons)
    training_indices = split_indices(boundaries, maximum_horizon)["train"]
    training_observations = observations[training_indices]
    print(
        f"[sarimax] Fitting {model_id} for {metric} on "
        f"{len(training_observations)} purged training observations",
        flush=True,
    )
    fit_started = time.monotonic()
    fitted = fit_sarimax(training_observations, config)
    fit_runtime = time.monotonic() - fit_started
    print(
        f"[sarimax] Fit converged in {fit_runtime:.1f}s; building issue-time forecasts",
        flush=True,
    )

    split_origins: dict[str, dict[int, np.ndarray]] = {}
    unique_origins: set[int] = set()
    actual_values: dict[tuple[str, int], np.ndarray] = {}
    for split_name in ("validation", "test"):
        split_origins[split_name] = {}
        for horizon in horizons:
            targets, target_valid = _target_values(working, metric, horizon)
            candidates = split_indices(boundaries, horizon)[split_name]
            baseline = seasonal_naive_values(working, candidates, horizon, metric)
            eligible = candidates[target_valid[candidates] & np.isfinite(baseline)]
            if max_origins_per_split is not None:
                eligible = eligible[:max_origins_per_split]
            split_origins[split_name][horizon] = eligible
            actual_values[(split_name, horizon)] = targets[eligible]
            unique_origins.update(int(index) for index in eligible)

    ordered_origins = sorted(unique_origins)
    all_forecasts = forecast_issue_origins(
        fitted, observations, ordered_origins, max_horizon=maximum_horizon
    )
    print(
        f"[sarimax] Forecasted {len(ordered_origins)} issue times across "
        f"{len(horizons)} requested horizons",
        flush=True,
    )

    metric_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    for split_name in ("validation", "test"):
        for horizon in horizons:
            origins = split_origins[split_name][horizon]
            actual = actual_values[(split_name, horizon)]
            predicted = np.asarray(
                [all_forecasts[int(origin)][horizon - 1] for origin in origins],
                dtype=float,
            )
            metrics = regression_metrics(actual, predicted)
            metric_rows.append(
                {
                    "split": split_name,
                    "metric": metric,
                    "horizon_hours": horizon,
                    "model": model_id,
                    "model_version": STATISTICAL_MODEL_VERSION,
                    **metrics,
                }
            )
            for position, origin in enumerate(origins):
                issue_time = timestamps.iloc[int(origin)]
                forecast_rows.append(
                    {
                        "issue_time_utc": issue_time.isoformat(),
                        "target_time_utc": (
                            issue_time + pd.Timedelta(horizon, unit="h")
                        ).isoformat(),
                        "horizon_hours": horizon,
                        "metric": metric,
                        "model": model_id,
                        "model_version": STATISTICAL_MODEL_VERSION,
                        "split": split_name,
                        "actual": float(actual[position]),
                        "forecast": float(predicted[position]),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "statistical_evaluation.csv"
    forecasts_path = output_dir / "statistical_forecasts.csv"
    metadata_path = output_dir / "statistical_metadata.json"
    artifact_path = output_dir / f"{metric}_{model_id}.joblib"
    write_csv_atomic(metrics_path, metric_rows)
    write_csv_atomic(forecasts_path, forecast_rows)
    schema_hash = hashlib.sha256(
        json.dumps(list(frame.columns), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    training_end_index = int(training_indices[-1])
    metadata = {
        "model": model_id,
        "model_version": STATISTICAL_MODEL_VERSION,
        "model_family": "SARIMAX",
        "metric": metric,
        "supported_metrics": list(SUPPORTED_METRICS),
        "unsupported_metrics": {
            "precipitation_mm": "intermittent, zero-inflated target; not modeled in this univariate Gaussian pilot",
            "wind_direction_degrees": "circular target; requires circular-specific modeling",
            "rain_occurred": "binary probability target; not suitable for Gaussian SARIMAX",
        },
        "dataset_sha256": _sha256_file(dataset_path),
        "feature_schema_sha256": schema_hash,
        "row_count": len(working),
        "training_observation_count": len(training_observations),
        "training_observation_end_index": training_end_index,
        "training_observation_end_utc": timestamps.iloc[training_end_index].isoformat(),
        "training_observations_end_before_validation": True,
        "forecast_input_policy": (
            "Parameters fitted on purged training observations only. At each issue time, "
            "the SARIMAX state is extended only with observed values through that issue time."
        ),
        "split": {
            "train_end_index_exclusive": boundaries.train_end,
            "validation_end_index_exclusive": boundaries.validation_end,
            "train_fraction": boundaries.train_fraction,
            "validation_fraction": boundaries.validation_fraction,
            "max_horizon_purge": maximum_horizon,
        },
        "config": asdict(config),
        "artifact_format": "sarimax-parameters-and-training-history-v1",
        "optimizer_converged": bool(fitted.mle_retvals.get("converged", False)),
        "optimizer_iterations": int(fitted.mle_retvals.get("iterations", 0)),
        "statsmodels_version": statsmodels.__version__,
        "python_version": os.sys.version.split()[0],
        "horizons_hours": list(horizons),
        "max_origins_per_split": max_origins_per_split,
        "fit_runtime_seconds": round(fit_runtime, 3),
        "evaluation_path": str(metrics_path),
        "forecasts_path": str(forecasts_path),
        "artifact_path": str(artifact_path),
    }
    _dump_joblib_atomic(
        artifact_path,
        {
            "format_version": 1,
            "model": model_id,
            "model_version": STATISTICAL_MODEL_VERSION,
            "metric": metric,
            "config": asdict(config),
            "training_observations": training_observations,
            "params": np.asarray(fitted.params, dtype=float),
            "statsmodels_version": statsmodels.__version__,
        },
    )
    metadata["model_artifact_sha256"] = _sha256_file(artifact_path)
    metadata["total_runtime_seconds"] = round(time.monotonic() - started_at, 3)
    write_json_atomic(metadata_path, metadata)
    write_text_atomic(
        output_dir / "statistical_evaluation.md",
        _evaluation_markdown(metric_rows, metadata),
    )
    return {
        "metrics": metric_rows,
        "forecasts": forecast_rows,
        "metadata": metadata,
        "metrics_path": str(metrics_path),
        "forecasts_path": str(forecasts_path),
        "metadata_path": str(metadata_path),
        "artifact_path": str(artifact_path),
    }


def _evaluation_markdown(
    metric_rows: list[dict[str, Any]], metadata: dict[str, Any]
) -> str:
    lines = [
        f"# {metadata['model_family']} {metadata['metric']} evaluation",
        "",
        f"- Model: `{metadata['model']}` version `{metadata['model_version']}`",
        f"- Dataset SHA-256: `{metadata['dataset_sha256']}`",
        f"- Config: `{metadata['config']}`",
        f"- Fit runtime: {metadata['fit_runtime_seconds']} seconds",
        f"- Experiment runtime through artifact hashing: "
        f"{metadata['total_runtime_seconds']} seconds",
        f"- Fitting converged: {metadata['optimizer_converged']}",
        "",
        "Parameters were fitted only on the purged training segment. At every "
        "forecast issue time, the filtered state uses observations through that "
        "time and never later observations.",
        "",
        "| Split | Horizon | N | MAE | RMSE | R2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metric_rows:
        r2 = "n/a" if row["r2"] is None else f"{row['r2']:.6f}"
        lines.append(
            f"| {row['split']} | h{row['horizon_hours']:02} | {row['n']} | "
            f"{row['mae']:.6f} | {row['rmse']:.6f} | {r2} |"
        )
    return "\n".join(lines) + "\n"
