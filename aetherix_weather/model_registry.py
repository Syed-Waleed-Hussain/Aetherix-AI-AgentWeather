"""Local model metadata preparation and validation-based champion selection."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aetherix_weather.hopsworks_features import sha256_file


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def feature_schema_hash(features: list[str]) -> str:
    return hashlib.sha256("\n".join(features).encode("utf-8")).hexdigest()


def _validation_metric(
    candidate: dict[str, Any], metric_name: str
) -> tuple[float, str]:
    metrics = candidate.get("validation_metrics")
    if not isinstance(metrics, dict) or metric_name not in metrics:
        raise ValueError(
            f"{candidate.get('model_name', 'candidate')} has no "
            f"validation metric {metric_name}."
        )
    metric = metrics[metric_name]
    try:
        score = float(metric["value"])
        direction = metric["direction"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Validation metric metadata is malformed.") from exc
    if not math.isfinite(score) or direction not in ("minimize", "maximize"):
        raise ValueError("Validation metric must be finite and have a valid direction.")
    return score, direction


def select_champion(
    candidates: list[dict[str, Any]], metric_name: str = "mae"
) -> dict[str, Any]:
    """Choose one comparable candidate using validation metrics only."""
    if not candidates:
        raise ValueError("At least one model candidate is required.")
    reference = candidates[0]
    required_identity = (
        "target",
        "horizons_hours",
        "dataset_sha256",
        "validation_period",
    )
    if any(reference.get(key) is None for key in required_identity):
        raise ValueError("Candidate validation identity is incomplete.")
    if any(
        candidate.get(key) != reference.get(key)
        for candidate in candidates[1:]
        for key in required_identity
    ):
        raise ValueError(
            "Candidates must share target, horizons, dataset, and validation period."
        )

    scored: list[tuple[float, str, dict[str, Any]]] = []
    expected_direction: str | None = None
    for candidate in candidates:
        score, direction = _validation_metric(candidate, metric_name)
        if expected_direction is None:
            expected_direction = direction
        elif direction != expected_direction:
            raise ValueError("Candidates must use the same metric direction.")
        name = candidate.get("model_name")
        if not isinstance(name, str) or not name:
            raise ValueError("Every candidate must have a model_name.")
        scored.append((score, name, candidate))

    if expected_direction == "maximize":
        scored.sort(key=lambda item: (-item[0], item[1]))
    else:
        scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][2]


def prepare_temperature_lstm_metadata(
    dataset_path: Path,
    model_metadata_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Build local registry-ready metadata without loading or uploading weights."""
    training = _read_json(model_metadata_path)
    dataset_hash = sha256_file(dataset_path)
    if training.get("dataset_sha256") != dataset_hash:
        raise ValueError("LSTM metadata dataset hash does not match features.csv.")

    input_features = training.get("input_features")
    if (
        not isinstance(input_features, list)
        or not input_features
        or not all(isinstance(name, str) and name for name in input_features)
        or len(set(input_features)) != len(input_features)
    ):
        raise ValueError("LSTM metadata has no valid input feature schema.")
    schema_hash = feature_schema_hash(input_features)
    if training.get("feature_schema_hash") != schema_hash:
        raise ValueError("LSTM metadata feature schema hash is invalid.")

    training_details = training.get("training", {})
    best_epoch = training_details.get("best_epoch")
    loss_history = training_details.get("validation_loss_history")
    if (
        not isinstance(best_epoch, int)
        or not isinstance(loss_history, list)
        or not 1 <= best_epoch <= len(loss_history)
    ):
        raise ValueError("LSTM metadata has no valid best validation epoch.")
    losses = [float(loss) for loss in loss_history]
    validation_loss = losses[best_epoch - 1]
    if not all(math.isfinite(loss) and loss >= 0 for loss in losses):
        raise ValueError("LSTM validation loss must be finite and non-negative.")
    if validation_loss != min(losses):
        raise ValueError("LSTM best epoch does not match the validation loss history.")

    boundaries = training.get("boundaries", {})
    train_end = boundaries.get("train_end")
    validation_end = boundaries.get("validation_end")
    row_count = boundaries.get("row_count")
    if not all(isinstance(value, int) for value in (train_end, validation_end, row_count)):
        raise ValueError("LSTM metadata has invalid chronological boundaries.")
    if not 0 < train_end < validation_end <= row_count:
        raise ValueError("LSTM chronological boundaries are inconsistent.")
    architecture = training["architecture"]
    horizon_count = architecture["output_size"]
    if type(horizon_count) is not int or horizon_count < 1:
        raise ValueError("LSTM architecture must specify positive forecast horizons.")
    horizons = list(range(1, horizon_count + 1))
    validation_rows = validation_end - train_end - horizon_count
    if train_end < 23 or validation_rows < 1:
        raise ValueError("LSTM validation partition contains no complete sequences.")

    validation_period = {
        "start": boundaries["validation_issue_start"],
        "end": boundaries["validation_issue_end"],
        "sequence_count": validation_rows,
    }
    common = {
        "target": "temperature_c",
        "horizons_hours": horizons,
        "dataset_sha256": dataset_hash,
        "validation_period": validation_period,
    }
    checkpoint_hash = sha256_file(checkpoint_path)
    repository_root = dataset_path.resolve().parents[2]
    try:
        relative_checkpoint_path = checkpoint_path.resolve().relative_to(
            repository_root
        ).as_posix()
    except ValueError:
        relative_checkpoint_path = checkpoint_path.name
    lstm_candidate = {
        **common,
        "model_name": "aetherix_temperature_lstm",
        "model_family": "pytorch_lstm",
        "version": 1,
        "architecture": architecture,
        "feature_schema": input_features,
        "feature_schema_sha256": schema_hash,
        "validation_metrics": {
            "scaled_mse": {
                "value": validation_loss,
                "direction": "minimize",
                "scope": "all forecast horizons",
                "best_epoch": best_epoch,
            }
        },
        "artifact": {
            "path": relative_checkpoint_path,
            "sha256": checkpoint_hash,
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "test_metrics": [
            {
                "horizon_hours": result["horizon_hours"],
                "mae": result["mae"],
                "rmse": result["rmse"],
                "r2": result["r2"],
                "samples": result["n"],
            }
            for result in training.get("evaluation", [])
            if result.get("split") == "test"
            and result.get("horizon_hours") in (1, 6, 12, 24)
        ],
    }

    data = pd.read_csv(dataset_path, usecols=["temperature_c"])
    if len(data) != row_count or data["temperature_c"].isna().any():
        raise ValueError("Temperature observations do not match the LSTM dataset.")
    target_scaler = training.get("target_scaler", {})
    target_mean = np.asarray(target_scaler.get("mean"), dtype=np.float64)
    target_scale = np.asarray(target_scaler.get("scale"), dtype=np.float64)
    if (
        target_mean.shape != (len(horizons),)
        or target_scale.shape != (len(horizons),)
        or not np.isfinite(target_mean).all()
        or not np.isfinite(target_scale).all()
        or (target_scale <= 0).any()
    ):
        raise ValueError("LSTM target scaler metadata is invalid.")

    origins = np.arange(train_end, validation_end - max(horizons))
    target_indices = origins[:, None] + np.arange(1, len(horizons) + 1)[None, :]
    temperatures = data["temperature_c"].to_numpy(dtype=np.float64)
    actual = temperatures[target_indices]
    naive = temperatures[target_indices - 24]
    actual_scaled = (actual - target_mean) / target_scale
    naive_scaled = (naive - target_mean) / target_scale
    naive_validation_loss = float(np.mean(np.square(naive_scaled - actual_scaled)))
    naive_candidate = {
        **common,
        "model_name": "seasonal_naive_24h",
        "model_family": "seasonal_naive_baseline",
        "version": 1,
        "feature_schema": ["temperature_c"],
        "validation_metrics": {
            "scaled_mse": {
                "value": naive_validation_loss,
                "direction": "minimize",
                "scope": "all forecast horizons",
            }
        },
        "artifact": None,
    }
    candidates = [lstm_candidate, naive_candidate]
    champion = select_champion(candidates, metric_name="scaled_mse")
    return {
        "metadata_schema_version": 1,
        "selection_metric": "scaled_mse",
        "selection_split": "validation",
        "dataset_sha256": dataset_hash,
        "feature_schema_sha256": schema_hash,
        "candidates": candidates,
        "champion": {
            "model_name": champion["model_name"],
            "model_family": champion["model_family"],
            "version": champion["version"],
            "validation_score": champion["validation_metrics"]["scaled_mse"]["value"],
        },
        "registered": False,
    }
