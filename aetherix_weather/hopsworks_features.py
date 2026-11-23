"""Offline feature-store schema and validation for historical weather data."""

import hashlib
import io
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

FEATURE_GROUP_NAME = "aetherix_hourly_weather_features"
FEATURE_GROUP_VERSION = 1
FEATURE_GROUP_DESCRIPTION = (
    "Offline-only Karachi hourly weather and modeled air-quality inference "
    "features with UTC event time and source dataset provenance."
)
CITY_ID = "karachi_pk"
SOURCE_VERSION = "aetherix-weather-aqi-v1"
MAX_SAMPLE_ROWS = 24
FULL_BATCH_SIZE = 5000
FULL_QUOTA_SAFETY_FACTOR = 4
PRIMARY_KEY = ("city_id", "event_timestamp")

MODEL_FEATURES = (
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
)
SOURCE_COLUMNS = ("weather_source", "air_quality_source", "uv_index_source")
VERSION_COLUMNS = ("source_version", "source_dataset_sha256")
FEATURE_COLUMNS = (
    "city_id",
    "event_timestamp",
    *MODEL_FEATURES,
    *SOURCE_COLUMNS,
    *VERSION_COLUMNS,
)
FEATURE_TYPES = {
    "city_id": "string",
    "event_timestamp": "timestamp",
    **{column: "double" for column in MODEL_FEATURES},
    **{column: "string" for column in (*SOURCE_COLUMNS, *VERSION_COLUMNS)},
}
_LAG_HOURS = {
    column: int(column.rsplit("_lag_", 1)[1][:-1])
    for column in MODEL_FEATURES
    if "_lag_" in column
}
_REQUIRED_SOURCE_COLUMNS = (
    "timestamp_utc",
    *MODEL_FEATURES,
    *SOURCE_COLUMNS,
)


class FeatureDataError(ValueError):
    """Raised when source data cannot safely satisfy the Feature Group schema."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_feature_frame(path: Path) -> tuple[pd.DataFrame, str]:
    """Read only inference features and provenance, excluding all future labels."""
    # Parse and hash the same bytes, including source columns outside the projection.
    source = path.read_bytes()
    dataset_hash = hashlib.sha256(source).hexdigest()
    raw = pd.read_csv(
        io.BytesIO(source), usecols=lambda name: name in _REQUIRED_SOURCE_COLUMNS
    )
    missing_columns = sorted(set(_REQUIRED_SOURCE_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise FeatureDataError(
            "Source dataset is missing required columns: "
            + ", ".join(missing_columns)
        )
    timestamps = raw["timestamp_utc"].astype("string")
    explicit_timezone = timestamps.str.contains(
        r"(?:[Zz]|[+-]\d{2}:\d{2})$", regex=True, na=False
    )
    if not explicit_timezone.all():
        raise FeatureDataError("Every timestamp_utc value must include a UTC offset.")
    try:
        event_timestamps = pd.to_datetime(
            timestamps, format="mixed", utc=True, errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise FeatureDataError("timestamp_utc contains an invalid timestamp.") from exc

    features = pd.DataFrame(index=raw.index)
    features["city_id"] = CITY_ID
    features["event_timestamp"] = event_timestamps
    for column in MODEL_FEATURES:
        try:
            features[column] = pd.to_numeric(raw[column], errors="raise").astype(
                "float64"
            )
        except (TypeError, ValueError) as exc:
            raise FeatureDataError(
                f"Feature {column} must contain numeric values."
            ) from exc
    for column in SOURCE_COLUMNS:
        if raw[column].isna().any():
            raise FeatureDataError(f"Source metadata {column} cannot be missing.")
        features[column] = raw[column].astype(str)
    features["source_version"] = SOURCE_VERSION
    features["source_dataset_sha256"] = dataset_hash
    features = features.loc[:, FEATURE_COLUMNS].reset_index(drop=True)
    validate_feature_frame(features)
    return features, dataset_hash


def validate_feature_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate schema, UTC hourly alignment, missingness, and composite keys."""
    if frame.empty:
        raise FeatureDataError("The source dataset contains no rows.")
    if tuple(frame.columns) != FEATURE_COLUMNS:
        raise FeatureDataError("Feature columns do not match the declared schema.")
    if any(column.startswith("target_") for column in frame.columns):
        raise FeatureDataError("Future target columns cannot enter inference features.")

    if not frame["city_id"].eq(CITY_ID).all():
        raise FeatureDataError(f"city_id must be {CITY_ID}.")
    for column in (*SOURCE_COLUMNS, *VERSION_COLUMNS):
        if frame[column].isna().any() or not frame[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise FeatureDataError(f"String metadata {column} must be non-empty.")

    timestamps = pd.to_datetime(frame["event_timestamp"], utc=True, errors="raise")
    if timestamps.isna().any():
        raise FeatureDataError("event_timestamp cannot be missing.")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise FeatureDataError("event_timestamp values must be unique and increasing.")
    if not timestamps.eq(timestamps.dt.floor("h")).all() or not (
        timestamps.diff().iloc[1:].eq(timedelta(hours=1)).all()
    ):
        raise FeatureDataError("event_timestamp values must be aligned to hourly steps.")

    missing_counts: dict[str, int] = {}
    for column in MODEL_FEATURES:
        values = frame[column]
        if not is_numeric_dtype(values.dtype) or is_bool_dtype(values.dtype):
            raise FeatureDataError(f"Feature {column} must be numeric.")
        non_null = values.dropna().to_numpy(dtype=np.float64)
        if not np.isfinite(non_null).all():
            raise FeatureDataError(f"Feature {column} contains non-finite values.")
        missing_mask = values.isna().to_numpy()
        missing_counts[column] = int(missing_mask.sum())
        if column in _LAG_HOURS:
            expected_missing = np.arange(len(frame)) < _LAG_HOURS[column]
            if not np.array_equal(missing_mask, expected_missing):
                raise FeatureDataError(
                    f"Missing values in {column} must be limited to its "
                    "leading history warm-up rows."
                )
        elif missing_mask.any():
            raise FeatureDataError(f"Feature {column} contains unexpected missing values.")

    if frame.duplicated(subset=list(PRIMARY_KEY)).any():
        raise FeatureDataError("Composite primary keys must be unique.")

    return {
        "rows": len(frame),
        "feature_columns": len(MODEL_FEATURES),
        "primary_key_columns": list(PRIMARY_KEY),
        "timestamp_start_utc": timestamps.iloc[0].isoformat(),
        "timestamp_end_utc": timestamps.iloc[-1].isoformat(),
        "missing_values": missing_counts,
    }


def select_sync_rows(frame: pd.DataFrame, sample: bool) -> pd.DataFrame:
    """Return all rows or a deterministic, complete-feature sample of at most 24."""
    if not sample:
        return frame.copy()
    complete = frame.loc[:, MODEL_FEATURES].notna().all(axis=1)
    selected = frame.loc[complete].head(MAX_SAMPLE_ROWS).copy()
    if selected.empty:
        raise FeatureDataError("No complete-feature rows are available for the sample.")
    return selected.reset_index(drop=True)


def estimate_csv_payload_bytes(frame: pd.DataFrame) -> int:
    """Return exact UTF-8 CSV-equivalent bytes, excluding Hopsworks/Parquet overhead."""
    buffer = io.StringIO()
    frame.to_csv(
        buffer,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S%z",
        lineterminator="\n",
    )
    return len(buffer.getvalue().encode("utf-8"))


def verify_secret_ignored(repository_root: Path) -> None:
    """Require .env to be ignored and untracked without reading its contents."""
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", ".env"],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    if ignored.returncode != 0:
        raise FeatureDataError(".env must be Git-ignored before syncing.")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", ".env"],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    if tracked.returncode == 0:
        raise FeatureDataError(".env must not be tracked by Git.")
    if tracked.returncode != 1:
        raise FeatureDataError("Could not verify whether .env is tracked.")


def verify_round_trip(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    """Compare keys, inference features, and metadata for a cloud read-back."""
    if actual.empty:
        raise FeatureDataError("Feature Group read-back returned no rows.")
    for column in FEATURE_COLUMNS:
        if column not in actual:
            raise FeatureDataError(f"Read-back is missing column {column}.")
    expected_rows = expected.loc[:, FEATURE_COLUMNS].copy()
    actual_rows = actual.loc[:, FEATURE_COLUMNS].copy()
    expected_rows["event_timestamp"] = pd.to_datetime(
        expected_rows["event_timestamp"], utc=True, errors="raise"
    )
    actual_rows["event_timestamp"] = pd.to_datetime(
        actual_rows["event_timestamp"], utc=True, errors="raise"
    )
    if actual_rows.duplicated(subset=list(PRIMARY_KEY)).any():
        raise FeatureDataError("Feature Group read-back contains duplicate primary keys.")
    expected_rows = expected_rows.sort_values(list(PRIMARY_KEY)).reset_index(drop=True)
    actual_rows = actual_rows.sort_values(list(PRIMARY_KEY)).reset_index(drop=True)
    if len(expected_rows) != len(actual_rows):
        raise FeatureDataError(
            f"Read-back row count {len(actual_rows)} does not match "
            f"expected count {len(expected_rows)}."
        )
    for column in FEATURE_COLUMNS:
        if column in MODEL_FEATURES:
            if not np.allclose(
                expected_rows[column].to_numpy(dtype=np.float64, na_value=np.nan),
                actual_rows[column].to_numpy(dtype=np.float64, na_value=np.nan),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            ):
                raise FeatureDataError(f"Read-back values differ for {column}.")
        elif not expected_rows[column].eq(actual_rows[column]).fillna(False).all():
            raise FeatureDataError(f"Read-back values differ for {column}.")
