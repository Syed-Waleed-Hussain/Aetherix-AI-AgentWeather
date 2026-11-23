"""Inference from the local, validation-selected forecast artifacts."""

from functools import cached_property
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge

from .hopsworks_features import MODEL_FEATURES, sha256_file
from .model_registry import feature_schema_hash
from .modeling import _candidate_predictions, _selection_metric, seasonal_naive_values

UNITS = {
    "temperature_c": "°C",
    "relative_humidity_percent": "%",
    "wind_speed_kmh": "km/h",
    "wind_direction_degrees": "°",
    "precipitation_mm": "mm",
    "rain_occurred": "probability",
    "uv_index": "UV index",
    "us_aqi": "US AQI",
}
SOURCE_COLUMNS = ("weather_source", "air_quality_source", "uv_index_source")


class InputError(ValueError):
    """An error with a fixed, credential-free explanation."""


def utc_hour(value: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("UTC")
            if stamp == stamp.floor("h"):
                return stamp
    except (ValueError, TypeError, OverflowError):
        pass
    raise InputError("Issue time must be an ISO-8601 hour with an explicit UTC offset.")


def prediction(metric: str, horizon: int, issue: pd.Timestamp, value: float, model: dict) -> dict:
    if not math.isfinite(value):
        raise InputError("Inference produced an invalid value.")
    bounded = value
    if metric in ("relative_humidity_percent", "rain_occurred"):
        bounded = min(100.0 if metric == "relative_humidity_percent" else 1.0, max(0.0, value))
    elif metric in ("wind_speed_kmh", "precipitation_mm", "uv_index", "us_aqi"):
        bounded = max(0.0, value)
    result = {
        "metric": metric, "horizon": horizon,
        "valid_at_utc": (issue + pd.Timedelta(horizon, unit="h")).isoformat(),
        "value": bounded, "unit": UNITS[metric], "model": model,
    }
    if bounded != value:
        result["physical_bounds_applied"] = True
    return result


class LocalForecastSource:
    def __init__(self, root: Path) -> None:
        self.dataset = root / "data/processed/features.csv"
        self.baselines = root / "artifacts/day3_5"
        self.manifest_path = root / "artifacts/day5c/champion_selection.json"

    @cached_property
    def metadata(self) -> dict:
        return json.loads((self.baselines / "selected_models.json").read_text(encoding="utf-8"))

    @cached_property
    def dataset_hash(self) -> str:
        digest = sha256_file(self.dataset)
        if digest != self.metadata["training_metadata"]["dataset_sha256"]:
            raise InputError("Local dataset SHA-256 differs from the validated training artifact.")
        return digest

    @cached_property
    def selections(self) -> dict:
        entries = self.metadata["selected_models"]
        indexed = {(row["metric"], row["horizon_hours"]): row for row in entries}
        if len(indexed) != len(entries):
            raise InputError("Saved selections have duplicate target/horizon keys.")
        return indexed

    @cached_property
    def manifest(self) -> dict | None:
        if not self.manifest_path.is_file():
            return None
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest["dataset_sha256"] != self.dataset_hash or manifest["selection_split"] != "validation":
            raise InputError("Champion manifest does not match the validated dataset and split.")
        decisions = manifest["decisions"]
        indexed = {(row["champion"]["target"], row["champion"]["horizon_hours"]):
                   row["champion"] for row in decisions}
        if len(indexed) != len(decisions):
            raise InputError("Champion manifest has duplicate target/horizon keys.")
        return indexed

    @cached_property
    def frame(self) -> pd.DataFrame:
        self.dataset_hash
        columns = {"timestamp_utc", *SOURCE_COLUMNS, *MODEL_FEATURES, *UNITS}
        for row in self.selections.values():
            columns.update(row["feature_columns"])
        frame = pd.read_csv(self.dataset, usecols=lambda name: name in columns)
        frame["_timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, format="mixed")
        stamps = frame["_timestamp_utc"]
        if stamps.empty or not stamps.is_unique or not stamps.diff().iloc[1:].eq(pd.Timedelta(1, unit="h")).all():
            raise InputError("The feature snapshot must contain consecutive unique UTC hours.")
        return frame

    @cached_property
    def estimators(self) -> dict:
        if self.metadata["training_metadata"]["scikit_learn_version"] != sklearn.__version__:
            raise InputError("The saved estimator bundle requires its original scikit-learn version.")
        # This fixed local path holds project-generated joblib artifacts.
        return joblib.load(self.baselines / "selected_models.joblib", mmap_mode="r")

    def issue_index(self, requested: str | None) -> int:
        if requested is None:
            return len(self.frame) - 1
        stamp = utc_hour(requested)
        result = self.frame.index[self.frame["_timestamp_utc"].eq(stamp)]
        if len(result) != 1:
            raise InputError("Issue hour is missing from the local feature snapshot.")
        if result[0] < self.metadata["split"]["validation_end_index_exclusive"]:
            raise InputError("Requested issue hour predates the end of model selection.")
        return int(result[0])

    def _forecast_value(self, metric: str, hour: int, index: int, selected: dict) -> float:
        name = selected["selected_model"]
        if name == "seasonal_naive":
            # Slice through issue time to enforce availability even on historical replay.
            history = self.frame.iloc[max(0, index - 23):index + 1]
            values = seasonal_naive_values(history, np.array([len(history) - 1]), hour, metric)
            return float(values[0])
        columns = selected["feature_columns"]
        if len(set(columns)) != len(columns) or any(
            col.startswith("target_") or "forecast" in col.lower() for col in columns
        ):
            raise InputError("Saved model declares invalid or leakage-prone predictors.")
        features = self.frame.iloc[[index]].loc[:, columns]
        estimator = self.estimators[f"{metric}|h{hour:02}"]
        if list(estimator.feature_names_in_) != columns:
            raise InputError("Estimator input order differs from the validated feature schema.")
        model = estimator.named_steps["model"]
        compatible = (
            (name.startswith("ridge_alpha_") and isinstance(model, Ridge)
             and float(name.removeprefix("ridge_alpha_")) == model.alpha)
            or (name == "random_forest" and isinstance(model, RandomForestRegressor))
            or (name == "random_forest_classifier" and isinstance(model, RandomForestClassifier))
        )
        if not compatible:
            raise InputError("Saved estimator family differs from the selected model.")
        values = _candidate_predictions(estimator, metric, features)
        if metric == "wind_direction_degrees":
            sine, cosine = values[0]
            if np.hypot(sine, cosine) < 1e-10:
                raise InputError("Wind direction is undefined for the predicted components.")
            return float(np.degrees(np.arctan2(sine, cosine)) % 360)
        return float(values[0])

    def forecast(self, requested: str | None, horizon: int) -> dict:
        index = self.issue_index(requested)
        issue = self.frame.at[index, "_timestamp_utc"]
        row = self.frame.iloc[index]
        manifest = self.manifest
        forecasts, gaps, fallbacks = [], [], []
        for metric in UNITS:
            for hour in range(1, horizon + 1):
                target = {"metric": metric, "horizon": hour}
                selected = self.selections.get((metric, hour))
                if selected is None:
                    gaps.append({**target, "reason": "No validated model selection."})
                    continue
                schema = feature_schema_hash(selected["feature_columns"])
                name = selected["selected_model"]
                champion = manifest.get((metric, hour)) if manifest is not None else None
                if manifest is not None and (
                    champion is None or champion["model_name"] != name
                    or champion["dataset_sha256"] != self.dataset_hash
                    or champion["feature_schema_sha256"] != schema
                    or champion["metric"] != _selection_metric(metric)
                ):
                    gaps.append({**target, "reason": "Champion lacks a compatible local inference artifact."})
                    continue
                provenance = {
                    "name": name, "dataset_sha256": self.dataset_hash,
                    "feature_schema_sha256": schema, "selection": "validation",
                    "artifact": ("historical_features"
                                 if name == "seasonal_naive" else "selected_models.joblib"),
                }
                try:
                    value = self._forecast_value(metric, hour, index, selected)
                    forecasts.append(prediction(metric, hour, issue, value, provenance))
                except Exception as exc:
                    reason = str(exc) if isinstance(exc, InputError) else type(exc).__name__
                    fallback = (
                        name != "seasonal_naive"
                        and selected["seasonal_naive_validation_rows"] == selected["validation_rows"]
                        and selected["validation_rows"] > 0
                    )
                    if fallback:
                        try:
                            value = self._forecast_value(
                                metric, hour, index, {"selected_model": "seasonal_naive"}
                            )
                            forecasts.append(prediction(metric, hour, issue, value, {
                                **provenance, "name": "seasonal_naive", "fallback_for": name,
                                "artifact": "historical_features",
                            }))
                            fallbacks.append({**target, "model": name, "reason": reason})
                            continue
                        except Exception:
                            reason += "; seasonal baseline unavailable"
                    gaps.append({**target, "model": name, "reason": reason})
        return {
            "issue_utc": issue.isoformat(), "predictions": forecasts, "coverage_gaps": gaps,
            "forecast_fallbacks": fallbacks,
            "weather_aqi_inputs": {
                metric: float(row[metric]) if metric in row and pd.notna(row[metric]) else None
                for metric in UNITS if metric != "rain_occurred"
            },
            "model_provenance": {
                "mode": "historical_local", "dataset_sha256": self.dataset_hash,
                "selection_source": "champion_manifest" if manifest is not None else "validated_baselines",
                "sources": {col: str(row[col]) for col in SOURCE_COLUMNS if col in row},
            },
        }
