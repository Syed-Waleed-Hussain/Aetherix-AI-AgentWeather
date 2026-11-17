"""Chronological, leakage-aware Day 3 forecasting experiments."""

import hashlib
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import RAIN_OCCURRENCE_THRESHOLD_MM
from .storage import write_csv_atomic, write_json_atomic, write_text_atomic

MAX_HORIZON_HOURS = 24
RIDGE_ALPHAS = (0.1, 1.0, 10.0)
RANDOM_FOREST_PARAMETERS: dict[str, Any] = {
    "n_estimators": 40,
    "max_depth": 10,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": 1,
}

REGRESSION_TASKS = {
    "temperature_c": {
        "target_prefix": "target_temperature",
        "raw_column": "temperature_c",
        "display_name": "temperature_c",
    },
    "relative_humidity_percent": {
        "target_prefix": "target_relative_humidity_percent",
        "raw_column": "relative_humidity_percent",
        "display_name": "relative_humidity_percent",
    },
    "precipitation_mm": {
        "target_prefix": "target_precipitation_mm",
        "raw_column": "precipitation_mm",
        "display_name": "precipitation_mm",
    },
    "wind_speed_kmh": {
        "target_prefix": "target_wind_speed_kmh",
        "raw_column": "wind_speed_kmh",
        "display_name": "wind_speed_kmh",
    },
    "us_aqi": {
        "target_prefix": "target_us_aqi",
        "raw_column": "us_aqi",
        "display_name": "us_aqi",
    },
    "uv_index": {
        "target_prefix": "target_uv_index",
        "raw_column": "uv_index",
        "display_name": "uv_index",
    },
}


@dataclass(frozen=True)
class SplitBoundaries:
    """Chronological origin-time boundaries shared by all forecast tasks."""

    row_count: int
    train_end: int
    validation_end: int
    train_fraction: float
    validation_fraction: float


def chronological_split(
    row_count: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    max_horizon: int = MAX_HORIZON_HOURS,
) -> SplitBoundaries:
    """Create chronological boundaries, leaving label-horizon purges per task."""
    if row_count <= 0:
        raise ValueError("row_count must be positive.")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than one.")
    if max_horizon < 1:
        raise ValueError("max_horizon must be at least one.")
    train_end = int(row_count * train_fraction)
    validation_end = int(row_count * (train_fraction + validation_fraction))
    if train_end <= max_horizon or validation_end - train_end <= max_horizon:
        raise ValueError(
            "Dataset is too short for the requested split and horizon purge."
        )
    return SplitBoundaries(
        row_count=row_count,
        train_end=train_end,
        validation_end=validation_end,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )


def split_indices(
    boundaries: SplitBoundaries, horizon: int
) -> dict[str, np.ndarray]:
    """Return origin indices whose labels end before the following partition."""
    if not 1 <= horizon <= MAX_HORIZON_HOURS:
        raise ValueError(f"horizon must be between 1 and {MAX_HORIZON_HOURS}.")
    return {
        "train": np.arange(0, boundaries.train_end - horizon, dtype=int),
        "validation": np.arange(
            boundaries.train_end,
            boundaries.validation_end - horizon,
            dtype=int,
        ),
        "test": np.arange(
            boundaries.validation_end, boundaries.row_count, dtype=int
        ),
    }


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Select numeric issue-time inputs and explicitly reject targets/forecasts."""
    excluded = {
        "timestamp_utc",
        "timestamp_local",
        "_timestamp_utc",
        "_timestamp_local",
    }
    columns = [
        column
        for column in frame.columns
        if column not in excluded
        and not column.startswith("_")
        and not column.startswith("target_")
        and "forecast" not in column.lower()
        and "_source_" not in column
        and not column.endswith("_source")
        and not column.endswith("_ingested_at_utc")
        and not column.endswith("_source_timestamp")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not columns:
        raise ValueError("No numeric issue-time feature columns were found.")
    if any(column.startswith("target_") for column in columns):
        raise ValueError("A future target column was selected as a model input.")
    if any("forecast" in column.lower() for column in columns):
        raise ValueError("Archived forecast information cannot be a model input.")
    return columns


def seasonal_naive_values(
    frame: pd.DataFrame,
    origin_indices: np.ndarray,
    horizon: int,
    metric: str,
) -> np.ndarray:
    """Use the exact same-hour value 24 hours before target time, if present."""
    if not 1 <= horizon <= MAX_HORIZON_HOURS:
        raise ValueError(f"horizon must be between 1 and {MAX_HORIZON_HOURS}.")
    if metric not in REGRESSION_TASKS and metric not in (
        "wind_direction_degrees",
        "rain_occurred",
    ):
        raise ValueError(f"Unsupported seasonal-naive metric: {metric}")
    raw_column = (
        "precipitation_mm"
        if metric == "rain_occurred"
        else REGRESSION_TASKS.get(metric, {}).get("raw_column", metric)
    )
    if raw_column not in frame.columns:
        raise ValueError(f"Seasonal-naive source column is missing: {raw_column}")
    timestamps = pd.to_datetime(frame["_timestamp_utc"], utc=True)
    position_by_time = {stamp: position for position, stamp in enumerate(timestamps)}
    values = np.full(len(origin_indices), np.nan, dtype=float)
    for output_index, origin_index in enumerate(origin_indices):
        target_time = timestamps.iloc[int(origin_index)] + pd.Timedelta(
            int(horizon), unit="h"
        )
        prior_day_time = target_time - pd.Timedelta(24, unit="h")
        prior_position = position_by_time.get(prior_day_time)
        if prior_position is None:
            continue
        raw_value = frame.iloc[prior_position][raw_column]
        if pd.isna(raw_value):
            continue
        value = float(raw_value)
        if metric == "rain_occurred":
            value = float(value > RAIN_OCCURRENCE_THRESHOLD_MM)
        values[output_index] = value
    return values


def select_best_candidate(
    validation_results: list[dict[str, Any]], selection_metric: str
) -> dict[str, Any]:
    """Select one candidate using validation scores only."""
    eligible = [
        result
        for result in validation_results
        if result.get("split") == "validation"
        and result.get(selection_metric) is not None
        and math.isfinite(float(result[selection_metric]))
    ]
    if not eligible:
        raise ValueError("No finite validation scores are available for selection.")
    return min(eligible, key=lambda result: float(result[selection_metric]))


def _make_regressor(name: str, alpha: float | None = None) -> Pipeline:
    imputer = SimpleImputer(strategy="median")
    if name == "ridge":
        if alpha is None:
            raise ValueError("Ridge candidates require an alpha.")
        return Pipeline(
            [
                ("imputer", imputer),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha)),
            ]
        )
    if name == "random_forest":
        return Pipeline(
            [
                ("imputer", imputer),
                (
                    "model",
                    RandomForestRegressor(**RANDOM_FOREST_PARAMETERS),
                ),
            ]
        )
    raise ValueError(f"Unknown regressor: {name}")


def _make_rain_classifier() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(**RANDOM_FOREST_PARAMETERS),
            ),
        ]
    )


def _target_column(metric: str, horizon: int) -> str:
    if metric == "wind_direction_degrees":
        return ""
    if metric == "rain_occurred":
        return f"target_rain_occurred_h{horizon:02}"
    return f"{REGRESSION_TASKS[metric]['target_prefix']}_h{horizon:02}"


def _target_values(
    frame: pd.DataFrame, metric: str, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    if metric == "wind_direction_degrees":
        columns = (
            f"target_wind_direction_sin_h{horizon:02}",
            f"target_wind_direction_cos_h{horizon:02}",
        )
        if any(column not in frame.columns for column in columns):
            raise ValueError(f"Wind-direction target columns are missing: {columns}")
        values = frame.loc[:, list(columns)].to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1)
        return values, valid
    column = _target_column(metric, horizon)
    if column not in frame.columns:
        raise ValueError(f"Required target column is missing: {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values, np.isfinite(values)


def _metrics(
    metric: str, actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float | None]:
    if metric == "rain_occurred":
        probabilities = np.asarray(predicted, dtype=float).reshape(-1)
        labels = (probabilities >= 0.5).astype(int)
        truth = np.asarray(actual, dtype=int).reshape(-1)
        return {
            "brier_score": float(brier_score_loss(truth, probabilities)),
            "precision": float(precision_score(truth, labels, zero_division=0)),
            "recall": float(recall_score(truth, labels, zero_division=0)),
            "f1": float(f1_score(truth, labels, zero_division=0)),
        }
    if metric == "wind_direction_degrees":
        truth_degrees = np.degrees(np.arctan2(actual[:, 0], actual[:, 1])) % 360
        predicted_degrees = np.degrees(
            np.arctan2(predicted[:, 0], predicted[:, 1])
        ) % 360
        errors = (predicted_degrees - truth_degrees + 180) % 360 - 180
        absolute_error = np.abs(errors)
        return {
            "circular_mae_degrees": float(absolute_error.mean()),
            "circular_rmse_degrees": float(np.sqrt(np.mean(np.square(errors)))),
            "mae": None,
            "rmse": None,
            "r2": None,
        }
    truth = np.asarray(actual, dtype=float).reshape(-1)
    estimate = np.asarray(predicted, dtype=float).reshape(-1)
    if metric == "precipitation_mm":
        estimate = np.maximum(estimate, 0.0)
    return {
        "mae": float(mean_absolute_error(truth, estimate)),
        "rmse": float(np.sqrt(mean_squared_error(truth, estimate))),
        "r2": (
            float(r2_score(truth, estimate))
            if len(truth) > 1 and np.ptp(truth) > 0
            else None
        ),
    }


def _direction_components(degrees: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(np.asarray(degrees, dtype=float))
    return np.column_stack((np.sin(radians), np.cos(radians)))


def _selection_metric(metric: str) -> str:
    if metric == "rain_occurred":
        return "brier_score"
    if metric == "wind_direction_degrees":
        return "circular_mae_degrees"
    return "mae"


def _predict_probability(estimator: Pipeline, features: pd.DataFrame) -> np.ndarray:
    classifier = estimator.named_steps["model"]
    probabilities = estimator.predict_proba(features)
    positive = np.flatnonzero(classifier.classes_ == 1)
    if len(positive) == 0:
        return np.zeros(len(features), dtype=float)
    return probabilities[:, int(positive[0])]


def _candidate_names(metric: str) -> list[tuple[str, float | None]]:
    if metric == "rain_occurred":
        return [("random_forest_classifier", None)]
    return [
        *((f"ridge_alpha_{alpha:g}", alpha) for alpha in RIDGE_ALPHAS),
        ("random_forest", None),
    ]


def _fit_candidate(
    name: str,
    alpha: float | None,
    metric: str,
    features: pd.DataFrame,
    target: np.ndarray,
) -> Pipeline:
    if metric == "rain_occurred":
        estimator = _make_rain_classifier()
        estimator.fit(features, target.astype(int))
        return estimator
    estimator = _make_regressor(
        "ridge" if name.startswith("ridge_") else "random_forest",
        alpha=alpha,
    )
    estimator.fit(features, target)
    return estimator


def _candidate_predictions(
    estimator: Pipeline, metric: str, features: pd.DataFrame
) -> np.ndarray:
    if metric == "rain_occurred":
        return _predict_probability(estimator, features)
    predictions = np.asarray(estimator.predict(features), dtype=float)
    if metric == "precipitation_mm":
        predictions = np.maximum(predictions, 0.0)
    return predictions


def _metric_row(
    split: str,
    metric: str,
    horizon: int,
    model_name: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    selection_metric: str,
    selected: bool,
    train_count: int,
) -> dict[str, Any]:
    values = _metrics(metric, actual, predicted)
    return {
        "split": split,
        "metric": metric,
        "horizon_hours": horizon,
        "model": model_name,
        "n": int(len(actual)),
        "n_train": int(train_count),
        "selection_metric": selection_metric,
        "is_selected": bool(selected),
        **values,
    }


def _write_test_predictions(
    frame: pd.DataFrame,
    task: str,
    horizon: int,
    origin_indices: np.ndarray,
    selected_name: str,
    predicted: np.ndarray,
    baseline: np.ndarray,
    actual: np.ndarray,
    output: list[dict[str, Any]],
) -> None:
    timestamps = pd.to_datetime(frame["_timestamp_utc"], utc=True)
    for position, origin in enumerate(origin_indices):
        issue_time = timestamps.iloc[int(origin)]
        target_time = issue_time + pd.Timedelta(int(horizon), unit="h")
        baseline_value = baseline[position]
        if task == "wind_direction_degrees":
            actual_value = float(
                np.degrees(np.arctan2(actual[position, 0], actual[position, 1]))
                % 360
            )
            predicted_value = float(
                np.degrees(np.arctan2(predicted[position, 0], predicted[position, 1]))
                % 360
            )
        elif task == "rain_occurred":
            actual_value = int(actual[position])
            predicted_value = float(predicted[position])
        else:
            actual_value = float(actual[position])
            predicted_value = float(predicted[position])
        baseline_value = (
            None if not np.isfinite(baseline_value) else float(baseline_value)
        )
        output.append(
            {
                "issue_time_utc": issue_time.isoformat(),
                "target_time_utc": target_time.isoformat(),
                "forecast_horizon_hours": horizon,
                "metric": task,
                "target_value": actual_value,
                "selected_model": selected_name,
                "prediction": predicted_value,
                "seasonal_naive_prediction": baseline_value,
                "rain_probability": (
                    predicted_value if task == "rain_occurred" else None
                ),
                "rain_class_at_0_5": (
                    int(predicted_value >= 0.5) if task == "rain_occurred" else None
                ),
            }
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_experiments(
    frame: pd.DataFrame,
    dataset_path: Path,
    output_dir: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Train candidates, select on validation, and evaluate selected models once."""
    if len(frame) < 100:
        raise ValueError("At least 100 hourly rows are required for Day 3 experiments.")
    working = frame.reset_index(drop=True).copy()
    if "_timestamp_utc" not in working:
        working["_timestamp_utc"] = pd.to_datetime(
            working["timestamp_utc"], utc=True, errors="raise"
        )
    boundaries = chronological_split(len(working))
    features = model_feature_columns(working)
    forbidden = [
        column
        for column in features
        if column.startswith("target_") or "forecast" in column.lower()
    ]
    if forbidden:
        raise ValueError(f"Leakage-prone model inputs found: {forbidden}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, Any]] = []
    selection_metadata: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selected_estimators: dict[str, Pipeline] = {}
    pending_test_evaluations: list[dict[str, Any]] = []
    task_names = [*REGRESSION_TASKS, "wind_direction_degrees", "rain_occurred"]

    for task in task_names:
        for horizon in range(1, MAX_HORIZON_HOURS + 1):
            indices = split_indices(boundaries, horizon)
            targets, target_valid = _target_values(working, task, horizon)
            train_indices = indices["train"][target_valid[indices["train"]]]
            validation_origins = indices["validation"]
            if len(train_indices) < 20:
                continue
            baseline_validation_all = seasonal_naive_values(
                working, validation_origins, horizon, task
            )
            target_validation_mask = target_valid[validation_origins]
            baseline_validation_mask = np.isfinite(baseline_validation_all)
            comparable_validation_mask = (
                target_validation_mask & baseline_validation_mask
            )
            validation_mask = (
                comparable_validation_mask
                if comparable_validation_mask.any()
                else target_validation_mask
            )
            validation_indices = validation_origins[validation_mask]
            if len(validation_indices) == 0:
                continue
            has_validation_baseline = bool(comparable_validation_mask.any())
            baseline_validation = (
                baseline_validation_all[validation_mask]
                if has_validation_baseline
                else np.full(len(validation_indices), np.nan)
            )
            baseline_validation_for_metrics = (
                _direction_components(baseline_validation)
                if task == "wind_direction_degrees"
                else baseline_validation
            )

            target_train = targets[train_indices]
            if task == "rain_occurred":
                unique_classes = np.unique(target_train.astype(int))
                if len(unique_classes) < 2:
                    candidate_specs: list[tuple[str, float | None]] = []
                else:
                    candidate_specs = _candidate_names(task)
            else:
                candidate_specs = _candidate_names(task)

            train_features = working.loc[train_indices, features]
            observed_train = train_features.notna().any(axis=0)
            selected_features = observed_train[observed_train].index.tolist()
            if not selected_features:
                raise ValueError(f"No training-observed features for task {task}.")
            train_features = train_features[selected_features]
            validation_features = working.loc[
                validation_indices, selected_features
            ]
            baseline_actual = targets[validation_indices]
            selection_metric = _selection_metric(task)
            task_validation_results: list[dict[str, Any]] = []

            if has_validation_baseline:
                baseline_row = _metric_row(
                    "validation",
                    task,
                    horizon,
                    "seasonal_naive",
                    baseline_actual,
                    baseline_validation_for_metrics,
                    selection_metric,
                    False,
                    len(train_indices),
                )
                task_validation_results.append(baseline_row)
                all_metrics.append(baseline_row)
            fitted_candidates: dict[str, Pipeline] = {}
            for candidate_name, alpha in candidate_specs:
                estimator = _fit_candidate(
                    candidate_name,
                    alpha,
                    task,
                    train_features,
                    target_train,
                )
                if task == "rain_occurred":
                    fitted_classes = estimator.named_steps["model"].classes_
                    if 1 not in fitted_classes:
                        continue
                predictions = _candidate_predictions(
                    estimator, task, validation_features
                )
                row = _metric_row(
                    "validation",
                    task,
                    horizon,
                    candidate_name,
                    baseline_actual,
                    predictions,
                    selection_metric,
                    False,
                    len(train_indices),
                )
                task_validation_results.append(row)
                all_metrics.append(row)
                fitted_candidates[candidate_name] = estimator

            if not task_validation_results:
                continue
            selected = select_best_candidate(
                task_validation_results, selection_metric
            )
            selected_name = str(selected["model"])
            selected["is_selected"] = True
            for row in task_validation_results:
                row["is_selected"] = row["model"] == selected_name
            estimator = fitted_candidates.get(selected_name)
            if estimator is not None:
                selected_estimators[f"{task}|h{horizon:02}"] = estimator

            train_issue_times = pd.to_datetime(
                working.loc[train_indices, "timestamp_utc"], utc=True
            )
            train_target_times = train_issue_times + pd.Timedelta(
                int(horizon), unit="h"
            )
            selection_metadata.append(
                {
                    "metric": task,
                    "horizon_hours": horizon,
                    "selected_model": selected_name,
                    "selection_metric": selection_metric,
                    "validation_score": float(selected[selection_metric]),
                    "validation_rows": int(selected["n"]),
                    "seasonal_naive_validation_rows": int(
                        comparable_validation_mask.sum()
                    ),
                    "seasonal_naive_unavailable_validation_rows": int(
                        (
                            target_validation_mask
                            & ~baseline_validation_mask
                        ).sum()
                    ),
                    "training_rows": int(len(train_indices)),
                    "validation_issue_start_utc": pd.to_datetime(
                        working.loc[validation_indices, "timestamp_utc"], utc=True
                    ).iloc[0].isoformat(),
                    "validation_issue_end_utc": pd.to_datetime(
                        working.loc[validation_indices, "timestamp_utc"], utc=True
                    ).iloc[-1].isoformat(),
                    "training_issue_start_utc": train_issue_times.iloc[0].isoformat(),
                    "training_issue_end_utc": train_issue_times.iloc[-1].isoformat(),
                    "training_target_end_utc": train_target_times.iloc[-1].isoformat(),
                    "feature_columns": selected_features,
                    "dropped_all_missing_training_features": [
                        column for column in features if column not in selected_features
                    ],
                    "model_parameters": (
                        {"alpha": float(selected_name.removeprefix("ridge_alpha_"))}
                        if selected_name.startswith("ridge_alpha_")
                        else RANDOM_FOREST_PARAMETERS
                        if "random_forest" in selected_name
                        else {"method": "same hour from previous day"}
                    ),
                }
            )
            pending_test_evaluations.append(
                {
                    "task": task,
                    "horizon": horizon,
                    "selected_name": selected_name,
                    "estimator": estimator,
                    "test_origins": indices["test"],
                    "targets": targets,
                    "target_valid": target_valid,
                    "selected_features": selected_features,
                    "train_count": len(train_indices),
                    "selection_metric": selection_metric,
                }
            )

    if not selection_metadata:
        raise ValueError("No eligible model tasks were trained and validated.")

    for pending in pending_test_evaluations:
        task = pending["task"]
        horizon = pending["horizon"]
        test_origins = pending["test_origins"]
        baseline_test_all = seasonal_naive_values(
            working, test_origins, horizon, task
        )
        target_test_mask = pending["target_valid"][test_origins]
        baseline_test_mask = np.isfinite(baseline_test_all)
        comparable_test_mask = target_test_mask & baseline_test_mask
        test_mask = (
            comparable_test_mask
            if comparable_test_mask.any()
            else target_test_mask
        )
        test_indices = test_origins[test_mask]
        baseline_test = (
            baseline_test_all[test_mask]
            if comparable_test_mask.any()
            else np.full(len(test_indices), np.nan)
        )
        if len(test_indices) == 0:
            continue
        test_actual = pending["targets"][test_indices]
        estimator = pending["estimator"]
        selected_name = pending["selected_name"]
        if selected_name == "seasonal_naive" and not comparable_test_mask.any():
            continue
        if estimator is None:
            test_predictions = (
                _direction_components(baseline_test)
                if task == "wind_direction_degrees"
                else baseline_test
            )
        else:
            test_features = working.loc[
                test_indices, pending["selected_features"]
            ]
            test_predictions = _candidate_predictions(estimator, task, test_features)
        baseline_for_metrics = (
            _direction_components(baseline_test)
            if task == "wind_direction_degrees"
            else baseline_test
        )
        all_metrics.append(
            _metric_row(
                "test",
                task,
                horizon,
                selected_name,
                test_actual,
                test_predictions,
                pending["selection_metric"],
                True,
                pending["train_count"],
            )
        )
        if selected_name != "seasonal_naive" and comparable_test_mask.any():
            all_metrics.append(
                _metric_row(
                    "test",
                    task,
                    horizon,
                    "seasonal_naive",
                    test_actual,
                    baseline_for_metrics,
                    pending["selection_metric"],
                    False,
                    pending["train_count"],
                )
            )
        _write_test_predictions(
            working,
            task,
            horizon,
            test_indices,
            selected_name,
            test_predictions,
            baseline_test,
            test_actual,
            prediction_rows,
        )
    metrics_path = output_dir / "model_evaluation.csv"
    write_csv_atomic(metrics_path, all_metrics)
    predictions_path = output_dir / "test_predictions.csv"
    write_csv_atomic(predictions_path, prediction_rows)
    rain_label_distribution: dict[str, dict[str, dict[str, int]]] = {}
    for horizon in range(1, MAX_HORIZON_HOURS + 1):
        labels, valid = _target_values(working, "rain_occurred", horizon)
        task_splits = split_indices(boundaries, horizon)
        horizon_counts: dict[str, dict[str, int]] = {}
        for split_name, row_indices in task_splits.items():
            observed = labels[row_indices][valid[row_indices]].astype(int)
            horizon_counts[split_name] = {
                "positive": int(np.count_nonzero(observed == 1)),
                "negative": int(np.count_nonzero(observed == 0)),
                "missing": int(len(row_indices) - len(observed)),
            }
        rain_label_distribution[f"h{horizon:02}"] = horizon_counts
    utc_issue_times = pd.to_datetime(working["timestamp_utc"], utc=True)
    selection_path = output_dir / "selected_models.json"
    write_json_atomic(
        selection_path,
        {
            "selection_rule": (
                "Select the lowest validation MAE for continuous targets, "
                "circular MAE for wind direction, or Brier score for rain. "
                "Test metrics are calculated only after selection and are not "
                "used to choose models."
            ),
            "split": {
                "row_count": boundaries.row_count,
                "train_end_index_exclusive": boundaries.train_end,
                "validation_end_index_exclusive": boundaries.validation_end,
                "train_fraction": boundaries.train_fraction,
                "validation_fraction": boundaries.validation_fraction,
                "test_fraction": 1
                - boundaries.train_fraction
                - boundaries.validation_fraction,
                "purge_hours": (
                    "Each horizon h purges the final h issue rows of train and "
                    "validation so their labels remain inside their partition."
                ),
                "base_interval_rows": {
                    "train": boundaries.train_end,
                    "validation": boundaries.validation_end - boundaries.train_end,
                    "test": boundaries.row_count - boundaries.validation_end,
                },
                "base_issue_time_ranges_utc": {
                    "train": {
                        "start": utc_issue_times.iloc[0].isoformat(),
                        "end": utc_issue_times.iloc[
                            boundaries.train_end - 1
                        ].isoformat(),
                    },
                    "validation": {
                        "start": utc_issue_times.iloc[
                            boundaries.train_end
                        ].isoformat(),
                        "end": utc_issue_times.iloc[
                            boundaries.validation_end - 1
                        ].isoformat(),
                    },
                    "test": {
                        "start": utc_issue_times.iloc[
                            boundaries.validation_end
                        ].isoformat(),
                        "end": utc_issue_times.iloc[-1].isoformat(),
                    },
                },
            },
            "training_metadata": {
                "dataset_sha256": _sha256(dataset_path),
                "manifest_sha256": (
                    _sha256(manifest_path)
                    if manifest_path is not None and manifest_path.is_file()
                    else None
                ),
                "python_version": platform.python_version(),
                "pandas_version": pd.__version__,
                "scikit_learn_version": sklearn.__version__,
                "input_feature_count": len(features),
                "rain_label_distribution_by_horizon": rain_label_distribution,
                "input_features": features,
                "excluded_target_columns": [
                    column
                    for column in working.columns
                    if column.startswith("target_")
                ],
                "historical_forecast_probabilities_used": False,
                "model_seed": 42,
            },
            "selected_models": selection_metadata,
        },
    )
    if selected_estimators:
        joblib.dump(selected_estimators, output_dir / "selected_models.joblib")
    report = _evaluation_markdown(selection_metadata)
    write_text_atomic(output_dir / "model_evaluation.md", report)
    return {
        "selected_models": selection_metadata,
        "metrics": all_metrics,
        "test_predictions": prediction_rows,
        "base_split_rows": {
            "train": boundaries.train_end,
            "validation": boundaries.validation_end - boundaries.train_end,
            "test": boundaries.row_count - boundaries.validation_end,
        },
        "input_feature_count": len(features),
        "rain_label_distribution_by_horizon": rain_label_distribution,
        "evaluation_path": str(metrics_path),
        "prediction_path": str(predictions_path),
        "model_artifact_path": (
            str(output_dir / "selected_models.joblib")
            if selected_estimators
            else None
        ),
    }


def _evaluation_markdown(
    selected: list[dict[str, Any]]
) -> str:
    lines = [
        "# Day 3 model evaluation",
        "",
        "Validation scores select models. Test scores are reported only for the "
        "selected candidate and seasonal-naive baseline; they do not influence "
        "model selection.",
        "",
        "Rain classification uses Brier score, precision, recall, and F1 at a "
        "0.5 threshold. Wind direction uses circular angular error in degrees; "
        "R² is intentionally omitted for this circular target and is undefined "
        "for constant target samples.",
        "",
        "Precipitation predictions are floored at 0 mm. Archived Open-Meteo "
        "Historical Forecast probabilities are excluded from inputs.",
        "",
        "The seasonal-naive baseline uses the exact previous-day valid-time "
        "record. If that input is absent, it is not imputed; that row is omitted "
        "from the like-for-like model comparison.",
        "",
        "## Validation-selected model by metric and horizon",
        "",
        "| Metric | Horizon | Selected model | Selection metric | Validation score | Train n | Validation n |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for item in selected:
        lines.append(
            f"| {item['metric']} | {item['horizon_hours']} | "
            f"{item['selected_model']} | {item['selection_metric']} | "
            f"{item['validation_score']:.6f} | {item['training_rows']} | "
            f"{item['validation_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Test performance",
            "",
            "See `model_evaluation.csv` for each per-metric, per-horizon score and "
            "`test_predictions.csv` for issue/target timestamps and predictions. "
            "Only the validation-selected model and seasonal-naive baseline are "
            "evaluated on the final test partition.",
            "",
            "Models are trained on the training partition only; imputation and "
            "Ridge scaling are fitted inside training-only scikit-learn pipelines.",
        ]
    )
    return "\n".join(lines) + "\n"
