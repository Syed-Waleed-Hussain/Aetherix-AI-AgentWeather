"""Leakage-safe PyTorch LSTM sequence and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn


SEQUENCE_LENGTH = 48
FORECAST_HORIZON = 24
TARGET_COLUMN = "temperature_c"


@dataclass(frozen=True)
class SequenceData:
    x: np.ndarray
    y: np.ndarray
    origins: np.ndarray
    target_times: np.ndarray


class TemperatureLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        output_size: int = FORECAST_HORIZON,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.output(hidden[-1])


def default_lstm_features(frame: pd.DataFrame) -> list[str]:
    names = [
        "temperature_c",
        "relative_humidity_percent",
        "precipitation_mm",
        "wind_speed_kmh",
        "uv_index",
        "us_aqi",
        "pm2_5_ug_m3",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "temperature_c_lag_24h",
        "temperature_c_lag_48h",
        "relative_humidity_percent_lag_24h",
        "wind_speed_kmh_lag_24h",
        "uv_index_lag_24h",
        "us_aqi_lag_24h",
    ]
    return [name for name in names if name in frame.columns]


def chronological_sequence_origins(
    row_count: int,
    train_end: int,
    validation_end: int,
    sequence_length: int = SEQUENCE_LENGTH,
    forecast_horizon: int = FORECAST_HORIZON,
) -> dict[str, np.ndarray]:
    if row_count <= sequence_length + forecast_horizon:
        raise ValueError("Dataset is too short for 48-hour sequences and 24-hour targets.")
    return {
        "train": np.arange(sequence_length - 1, train_end - forecast_horizon, dtype=int),
        "validation": np.arange(
            train_end,
            validation_end - forecast_horizon,
            dtype=int,
        ),
        "test": np.arange(validation_end, row_count - forecast_horizon, dtype=int),
    }


def build_sequences(
    frame: pd.DataFrame,
    origins: np.ndarray,
    feature_columns: Iterable[str],
    sequence_length: int = SEQUENCE_LENGTH,
    forecast_horizon: int = FORECAST_HORIZON,
) -> SequenceData:
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    if timestamps.is_monotonic_increasing is False or timestamps.duplicated().any():
        raise ValueError("timestamp_utc must be unique and strictly chronological.")
    features = list(feature_columns)
    if not features:
        raise ValueError("At least one input feature is required.")
    values = frame[features].to_numpy(dtype=np.float32)
    target = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    valid_origins: list[int] = []
    target_times: list[pd.Timestamp] = []
    for origin in origins.tolist():
        if origin < sequence_length - 1:
            continue
        start = origin - sequence_length + 1
        end = origin + 1
        y_indices = np.arange(origin + 1, origin + forecast_horizon + 1)
        if y_indices[-1] >= len(frame):
            continue
        y = target[y_indices]
        if not np.isfinite(y).all():
            continue
        x_rows.append(values[start:end])
        y_rows.append(y)
        valid_origins.append(origin)
        target_times.extend(timestamps.iloc[index] for index in y_indices)
    if not x_rows:
        return SequenceData(
            x=np.empty((0, sequence_length, len(features)), dtype=np.float32),
            y=np.empty((0, forecast_horizon), dtype=np.float32),
            origins=np.empty((0,), dtype=int),
            target_times=np.empty((0,), dtype="datetime64[ns, UTC]"),
        )
    target_matrix = np.stack(y_rows)
    target_times_array = np.asarray(
        [timestamp.to_datetime64() for timestamp in target_times],
        dtype="datetime64[ns]",
    ).reshape(-1, forecast_horizon)
    target_times_array = pd.to_datetime(
        target_times_array.ravel(), utc=True
    ).to_numpy().reshape(-1, forecast_horizon)
    return SequenceData(
        x=np.stack(x_rows),
        y=target_matrix,
        origins=np.asarray(valid_origins, dtype=int),
        target_times=target_times_array,
    )


def fit_training_scalers(
    train_x: np.ndarray,
    train_y: np.ndarray,
) -> tuple[SimpleImputer, StandardScaler, SimpleImputer, StandardScaler]:
    shape = train_x.shape
    flat_x = train_x.reshape(-1, shape[-1])
    feature_imputer = SimpleImputer(strategy="median")
    flat_imputed = feature_imputer.fit_transform(flat_x)
    feature_scaler = StandardScaler()
    feature_scaler.fit(flat_imputed)

    target_imputer = SimpleImputer(strategy="median")
    target_imputed = target_imputer.fit_transform(train_y)
    target_scaler = StandardScaler()
    target_scaler.fit(target_imputed)
    return feature_imputer, feature_scaler, target_imputer, target_scaler


def transform_sequences(
    sequences: np.ndarray,
    feature_imputer: SimpleImputer,
    feature_scaler: StandardScaler,
) -> np.ndarray:
    shape = sequences.shape
    flat = sequences.reshape(-1, shape[-1])
    imputed = feature_imputer.transform(flat)
    scaled = feature_scaler.transform(imputed)
    return scaled.reshape(shape).astype(np.float32)


def inverse_transform_targets(
    targets: np.ndarray,
    target_scaler: StandardScaler,
) -> np.ndarray:
    if targets.shape[-1] != target_scaler.n_features_in_:
        raise ValueError("Target width does not match the fitted target scaler.")
    return target_scaler.inverse_transform(targets)


def set_reproducible_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
