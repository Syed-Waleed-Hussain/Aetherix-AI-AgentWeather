"""Validation-only, per-horizon decisions for the existing forecast artifacts."""

import math
from pathlib import Path
from typing import Any

import pandas as pd

from aetherix_weather.model_registry import _read_json, feature_schema_hash


def validation_metric(target: str) -> str:
    return {
        "rain_occurred": "brier_score",
        "wind_direction_degrees": "circular_mae_degrees",
    }.get(target, "mae")


def decide_champion(
    incumbent: dict[str, Any], challengers: list[dict[str, Any]]
) -> dict[str, Any]:
    metric = validation_metric(incumbent["target"])
    score = incumbent["validation_score"]
    if incumbent["metric"] != metric or not math.isfinite(score) or score < 0:
        raise ValueError("Incumbent must have a finite score for the target validation metric.")
    winner = incumbent
    decisions = []
    ordered = sorted(challengers, key=lambda row: row["model_name"])
    for candidate in ordered:
        mismatches = [
            field for field in (
                "target", "horizon_hours", "dataset_sha256", "feature_schema_sha256",
                "validation_observations", "metric",
            )
            if incumbent.get(field) is None or candidate.get(field) != incumbent[field]
        ]
        score = candidate.get("validation_score")
        if mismatches:
            reason = "Incompatible " + ", ".join(mismatches)
        elif score is None or not math.isfinite(score) or score < 0:
            reason = "Missing or invalid validation score"
        elif score < winner["validation_score"]:
            winner = candidate
            reason = "Lower compatible validation score"
        else:
            reason = "Compatible validation score did not improve on the current best"
        decisions.append({**candidate, "reason": reason})
    for decision, candidate in zip(decisions, ordered):
        decision["selected"] = candidate is winner
        if not decision["selected"] and decision["reason"] == "Lower compatible validation score":
            decision["reason"] = "Another compatible model had a lower validation score"
    return {
        "champion": winner,
        "incumbent": incumbent["model_name"],
        "reason": (
            "Retained the existing validation selection"
            if winner is incumbent else "Promoted on lower compatible validation score"
        ),
        "challengers": decisions,
    }


def _hourly_window(start: str, end: str, count: int) -> dict[str, Any] | None:
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if (
        first.tzinfo is None or last.tzinfo is None or count < 1
        or last - first != pd.Timedelta(hours=count - 1)
    ):
        return None
    return {
        "start": first.tz_convert("UTC").isoformat(),
        "end": last.tz_convert("UTC").isoformat(),
        "count": count,
        "frequency": "hourly",
    }


def _statistical_candidates(root: Path) -> dict[tuple[str, int], list[dict]]:
    candidates: dict[tuple[str, int], list[dict]] = {}
    for path in sorted(root.glob("*/statistical_metadata.json")):
        metadata = _read_json(path)
        forecasts = pd.read_csv(
            path.with_name("statistical_forecasts.csv"),
            usecols=["split", "horizon_hours", "issue_time_utc", "actual", "forecast"],
        )
        validation = forecasts.loc[forecasts["split"].eq("validation")]
        for horizon, group in validation.groupby("horizon_hours"):
            times = pd.to_datetime(group["issue_time_utc"], utc=True)
            window = None
            if times.is_unique and times.eq(times.dt.floor("h")).all():
                window = _hourly_window(str(times.min()), str(times.max()), len(times))
            errors = (group["actual"] - group["forecast"]).abs()
            target = metadata["metric"]
            candidates.setdefault((target, horizon), []).append({
                "model_name": metadata["model"],
                "evaluation_run": path.parent.name,
                "target": target,
                "horizon_hours": int(horizon),
                "dataset_sha256": metadata["dataset_sha256"],
                "feature_schema_sha256": metadata["feature_schema_sha256"],
                "validation_observations": window,
                "metric": "mae",
                "validation_score": float(errors.mean()) if errors.notna().all() else None,
            })
    return candidates


def build_champion_manifest(
    baseline_path: Path, statistical_root: Path, lstm: dict[str, Any]
) -> dict[str, Any]:
    baseline = _read_json(baseline_path)
    dataset_hash = lstm["dataset_sha256"]
    if baseline["training_metadata"]["dataset_sha256"] != dataset_hash:
        raise ValueError("Saved baseline dataset hash differs from the verified LSTM dataset.")
    statistical = _statistical_candidates(statistical_root)
    period = lstm["validation_period"]
    lstm_window = _hourly_window(period["start"], period["end"], period["sequence_count"])
    entries = []
    seen = set()
    for row in baseline["selected_models"]:
        target, horizon = row["metric"], row["horizon_hours"]
        if (target, horizon) in seen:
            raise ValueError("Baseline selections contain duplicate target/horizon pairs.")
        seen.add((target, horizon))
        incumbent = {
            "model_name": row["selected_model"],
            "target": target,
            "horizon_hours": horizon,
            "metric": row["selection_metric"],
            "validation_score": row["validation_score"],
            "dataset_sha256": dataset_hash,
            "feature_schema_sha256": feature_schema_hash(row["feature_columns"]),
            "validation_observations": _hourly_window(
                row["validation_issue_start_utc"], row["validation_issue_end_utc"],
                row["validation_rows"],
            ),
        }
        challengers = list(statistical.get((target, horizon), ()))
        if target == lstm["target"] and horizon in lstm["horizons_hours"]:
            challengers.append({
                "model_name": lstm["model_name"],
                "target": target,
                "horizon_hours": horizon,
                "dataset_sha256": dataset_hash,
                "feature_schema_sha256": lstm["feature_schema_sha256"],
                "validation_observations": lstm_window,
                "metric": "scaled_mse_across_all_horizons",
                "validation_score": lstm["validation_metrics"]["scaled_mse"]["value"],
            })
        entries.append(decide_champion(incumbent, challengers))
    if not entries:
        raise ValueError("No saved baseline selections are available.")
    return {
        "manifest_version": 1,
        "selection_split": "validation",
        "dataset_sha256": dataset_hash,
        "decisions": entries,
    }
