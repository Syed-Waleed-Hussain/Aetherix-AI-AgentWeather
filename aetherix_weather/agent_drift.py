"""Monitor input shift and recently matured holdout forecast errors."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .agent_forecast import LocalForecastSource
from .dataset import RAIN_OCCURRENCE_THRESHOLD_MM
from .hopsworks_features import MODEL_FEATURES

THRESHOLDS = {
    "recent_hours": 168, "min_observations": 24, "ks_statistic": 0.25,
    "missing_rate_change": 0.10, "error_ratio": 1.50,
}


def _feature_drift(reference: pd.DataFrame, recent: pd.DataFrame) -> list[dict]:
    checks = []
    for name in MODEL_FEATURES:
        if name not in reference or name not in recent:
            continue
        before, after = reference[name], recent[name]
        if min(len(before), len(after)) < THRESHOLDS["min_observations"]:
            continue
        clean_before, clean_after = before.dropna(), after.dropna()
        shift = None
        if min(len(clean_before), len(clean_after)) >= THRESHOLDS["min_observations"]:
            shift = float(ks_2samp(clean_before, clean_after, method="asymp").statistic)
        previous_missing, recent_missing = float(before.isna().mean()), float(after.isna().mean())
        checks.append({
            "feature": name, "ks": shift,
            "reference_missing_rate": previous_missing, "recent_missing_rate": recent_missing,
            "missing_rate_change": recent_missing - previous_missing,
            "reference_rows": len(before), "recent_rows": len(after),
            "flagged": (
                abs(recent_missing - previous_missing) >= THRESHOLDS["missing_rate_change"]
                or (shift is not None and shift >= THRESHOLDS["ks_statistic"])
            ),
        })
    return checks


def _forecast_error_checks(archive: pd.DataFrame, observed: pd.DataFrame, selections: dict) -> list[dict]:
    if archive.empty:
        return []
    actual = observed.set_index("_timestamp_utc")
    checks = []
    for (metric, hour), rows in archive.groupby(["metric", "forecast_horizon_hours"]):
        selection = selections.get((metric, hour))
        if selection is None:
            continue
        model = selection["selected_model"]
        rows = rows.loc[rows["selected_model"].eq(model)]
        if rows["target_time_utc"].duplicated().any():
            continue
        column = "precipitation_mm" if metric == "rain_occurred" else metric
        if column not in actual:
            continue
        truth_values = actual[column]
        missing_flag = f"{column}_was_missing"
        if missing_flag in actual:
            truth_values = truth_values.mask(actual[missing_flag].eq(True))
        truth = rows["target_time_utc"].map(truth_values)
        if metric == "rain_occurred":
            truth = truth.gt(RAIN_OCCURRENCE_THRESHOLD_MM).astype(float).where(truth.notna())
        delta = (rows["prediction"] - truth).abs()
        delta = delta[np.isfinite(delta)]
        if len(delta) < THRESHOLDS["min_observations"]:
            continue
        if metric == "rain_occurred":
            delta = delta.pow(2)
        if metric == "wind_direction_degrees":
            angle = delta.mod(360)
            delta = np.minimum(angle, 360 - angle)
        current = float(delta.mean())
        reference = float(selection["validation_score"])
        checks.append({
            "metric": metric, "horizon": int(hour), "model": model,
            "error_metric": selection["selection_metric"], "observations": len(delta),
            "reference_validation_error": reference, "recent_holdout_error": current,
            "ratio": current / reference if reference > 0 else None,
            "flagged": current > reference and current >= reference * THRESHOLDS["error_ratio"],
        })
    return checks


def drift_result(features: list[dict], errors: list[dict], stamp: str) -> dict:
    affected_features = [row["feature"] for row in features if row["flagged"]]
    affected_models = [
        {key: row[key] for key in ("metric", "horizon", "model")}
        for row in errors if row["flagged"]
    ]
    if affected_models:
        status = "forecast_error_increase"
        action = "Review the matured errors and repeat on another holdout window before approving retraining."
    elif affected_features:
        status = "input_shift"
        action = "Inspect source quality and seasonal changes. Collect matured errors before considering retraining."
    elif not features and not errors:
        status = "insufficient_evidence"
        action = "Collect enough recent features or timestamp-matched actual observations."
    else:
        status = "no_threshold_breach"
        action = "Continue monitoring."
    return {
        "status": status, "evaluated_as_of_utc": stamp, "thresholds": THRESHOLDS,
        "features": features, "forecast_errors": errors, "affected_features": affected_features,
        "affected_models": affected_models, "recommended_action": action,
        "retraining_review_requested": bool(affected_models),
        "explanation": (
            "Input shift is measured against the training distribution. Error comparisons use "
            "matured holdout forecasts joined to observed targets and the model's validation score."
            if errors else "Input shift alone does not establish performance degradation. "
            "Matched forecast-error evidence is unavailable."
        ),
    }


def monitor_local(source: LocalForecastSource, requested: str | None) -> dict:
    frame = source.frame
    index = source.issue_index(requested)
    train_end = source.metadata["split"]["train_end_index_exclusive"]
    recent = frame.iloc[max(train_end, index - THRESHOLDS["recent_hours"] + 1):index + 1]
    issue = frame.at[index, "_timestamp_utc"]
    features = _feature_drift(frame.iloc[:train_end], recent)
    archive_path = source.baselines / "test_predictions.csv"
    errors = []
    if archive_path.is_file() and not recent.empty:
        chunks = []
        holdout_start = frame.at[source.metadata["split"]["validation_end_index_exclusive"],
                                 "_timestamp_utc"]
        columns = ["metric", "forecast_horizon_hours", "issue_time_utc", "target_time_utc",
                   "selected_model", "prediction"]
        for chunk in pd.read_csv(archive_path, usecols=columns, chunksize=40_000):
            times = pd.to_datetime(chunk["target_time_utc"], utc=True, format="mixed")
            candidates = chunk.loc[times.between(recent["_timestamp_utc"].iloc[0], issue)].copy()
            if candidates.empty:
                continue
            candidates["target_time_utc"] = times.loc[candidates.index]
            origins = pd.to_datetime(candidates["issue_time_utc"], utc=True, format="mixed")
            valid = (
                candidates["target_time_utc"] - origins
            ).eq(pd.to_timedelta(candidates["forecast_horizon_hours"], unit="hours"))
            valid &= origins.ge(holdout_start)
            chunks.append(candidates.loc[valid])
        if chunks:
            errors = _forecast_error_checks(pd.concat(chunks, ignore_index=True), recent, source.selections)
    return drift_result(features, errors, issue.isoformat())
