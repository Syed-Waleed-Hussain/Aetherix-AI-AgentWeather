from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aetherix_weather.hopsworks_features import (
    FEATURE_COLUMNS,
    FEATURE_GROUP_NAME,
    FEATURE_TYPES,
    MODEL_FEATURES,
    PRIMARY_KEY,
    SOURCE_COLUMNS,
    FeatureDataError,
    estimate_csv_payload_bytes,
    read_feature_frame,
    select_sync_rows,
    validate_feature_frame,
    verify_round_trip,
)
from scripts import sync_hopsworks_features as sync


def _write_source(path: Path, count: int = 80) -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    frame = pd.DataFrame(
        {
            "timestamp_utc": [
                (start + timedelta(hours=index)).isoformat()
                for index in range(count)
            ],
            **{
                column: np.arange(count, dtype=float) + 10
                for column in MODEL_FEATURES
            },
            "weather_source": ["weather-archive"] * count,
            "air_quality_source": ["modeled-air-quality"] * count,
            "uv_index_source": ["modeled-air-quality"] * count,
            "target_temperature_h01": np.arange(count, dtype=float),
            "precipitation_probability_percent": [np.nan] * count,
        }
    )
    for column in MODEL_FEATURES:
        if "_lag_" in column:
            lag_hours = int(column.rsplit("_lag_", 1)[1][:-1])
            frame.loc[: lag_hours - 1, column] = np.nan
    frame.to_csv(path, index=False)
    return frame


def _feature_frame(tmp_path: Path, count: int = 80) -> pd.DataFrame:
    source = tmp_path / "features.csv"
    _write_source(source, count)
    return read_feature_frame(source)[0]


def test_feature_schema_excludes_targets_and_all_null_source_columns(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "features.csv"
    _write_source(source_path)
    frame, digest = read_feature_frame(source_path)

    report = validate_feature_frame(frame)

    assert tuple(frame.columns) == FEATURE_COLUMNS
    assert len(MODEL_FEATURES) == 17
    assert len(frame.columns) == 24
    assert frame["event_timestamp"].dt.tz is not None
    assert not any(column.startswith("target_") for column in frame.columns)
    assert "precipitation_probability_percent" not in frame
    assert FEATURE_TYPES["event_timestamp"] == "timestamp"
    assert report["rows"] == 80
    assert digest == frame["source_dataset_sha256"].iloc[0]


def test_timestamp_alignment_rejects_missing_timezone_and_hourly_gaps(
    tmp_path: Path,
) -> None:
    frame = _feature_frame(tmp_path)
    frame.loc[30:, "event_timestamp"] += timedelta(hours=1)

    with pytest.raises(FeatureDataError, match="hourly steps"):
        validate_feature_frame(frame)

    frame = _feature_frame(tmp_path)
    frame["event_timestamp"] += timedelta(minutes=30)
    with pytest.raises(FeatureDataError, match="hourly steps"):
        validate_feature_frame(frame)

    source = tmp_path / "naive.csv"
    original = _write_source(source)
    original["timestamp_utc"] = original["timestamp_utc"].str.replace(
        "+00:00", "", regex=False
    )
    original.to_csv(source, index=False)
    with pytest.raises(FeatureDataError, match="UTC offset"):
        read_feature_frame(source)


def test_primary_key_uniqueness_and_lag_missingness_are_enforced(
    tmp_path: Path,
) -> None:
    frame = _feature_frame(tmp_path)
    frame.loc[35, "event_timestamp"] = frame.loc[34, "event_timestamp"]
    with pytest.raises(FeatureDataError, match="unique and increasing"):
        validate_feature_frame(frame)

    frame = _feature_frame(tmp_path)
    frame.loc[55, "temperature_c_lag_24h"] = np.nan
    with pytest.raises(FeatureDataError, match="leading history warm-up"):
        validate_feature_frame(frame)


def test_sample_is_bounded_deterministic_and_complete(tmp_path: Path) -> None:
    frame = _feature_frame(tmp_path, count=100)

    sample = select_sync_rows(frame, sample=True)

    assert len(sample) == 24
    assert sample.loc[:, MODEL_FEATURES].notna().all().all()
    assert sample["event_timestamp"].iloc[0] == frame["event_timestamp"].iloc[48]
    assert estimate_csv_payload_bytes(sample) > 0


def test_round_trip_compares_every_value_and_detects_missing_or_duplicate_rows(
    tmp_path: Path,
) -> None:
    expected = select_sync_rows(_feature_frame(tmp_path), sample=True)
    verify_round_trip(expected, expected.copy())
    actual = expected.astype({"city_id": "string"})
    actual["event_timestamp"] = actual["event_timestamp"].dt.tz_localize(None)
    verify_round_trip(expected, actual)

    with pytest.raises(FeatureDataError, match="row count"):
        verify_round_trip(expected, expected.iloc[:-1].copy())

    duplicate = pd.concat([expected, expected.iloc[[0]]], ignore_index=True)
    with pytest.raises(FeatureDataError, match="duplicate primary keys"):
        verify_round_trip(expected, duplicate)

    mismatch = expected.copy()
    mismatch.loc[0, "temperature_c"] += 1
    with pytest.raises(FeatureDataError, match="values differ"):
        verify_round_trip(expected, mismatch)


def test_offline_feature_group_schema_and_primary_key_are_explicit() -> None:
    feature_group = SimpleNamespace(
        columns=[
            SimpleNamespace(name=name, type=value)
            for name, value in FEATURE_TYPES.items()
        ],
        primary_key=list(PRIMARY_KEY),
        event_time="event_timestamp",
        online_enabled=False,
        time_travel_format="HUDI",
        stream=True,
    )

    sync._verify_feature_group_schema(feature_group)
    assert FEATURE_GROUP_NAME == "aetherix_hourly_weather_features"

    feature_group.online_enabled = True
    with pytest.raises(sync.SyncOperationError, match="offline-only"):
        sync._verify_feature_group_schema(feature_group)

    feature_group.online_enabled = False
    feature_group.columns[2].type = "float"
    with pytest.raises(sync.SyncOperationError, match="schema does not match"):
        sync._verify_feature_group_schema(feature_group)


def test_dry_run_never_connects_or_creates_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    source = tmp_path / "features.csv"
    _write_source(source)
    monkeypatch.setattr(sync, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sync, "verify_secret_ignored", lambda _root: None)
    monkeypatch.setattr(
        sync,
        "_connect_to_feature_store",
        lambda: pytest.fail("dry-run must not connect to Hopsworks"),
    )
    monkeypatch.setattr(
        sync,
        "_execute_sync",
        lambda *args, **kwargs: pytest.fail("dry-run must not execute a write"),
    )

    result = sync.main(["--dry-run", "--data", str(source)])
    output = capsys.readouterr().out

    assert result == 0
    assert "no Hopsworks connection or remote writes" in output
    assert not (tmp_path / "outputs").exists()
    assert "aetherix_hourly_weather_features" in output


def test_writes_require_explicit_confirmation_and_full_quota() -> None:
    with pytest.raises(ValueError, match="--confirm-write"):
        sync.validate_execution_request(
            sample=True,
            full=False,
            dry_run=False,
            execute=True,
            confirm_write=False,
            free_storage_remaining_bytes=None,
        )
    with pytest.raises(ValueError, match="Free-plan quota"):
        sync.validate_execution_request(
            sample=False,
            full=True,
            dry_run=False,
            execute=True,
            confirm_write=True,
            free_storage_remaining_bytes=None,
        )
    sync.validate_execution_request(
        sample=True,
        full=False,
        dry_run=False,
        execute=True,
        confirm_write=True,
        free_storage_remaining_bytes=None,
    )
    with pytest.raises(ValueError, match="--confirm-free-plan"):
        sync.validate_execution_request(
            sample=False,
            full=True,
            dry_run=False,
            execute=True,
            confirm_write=True,
            free_storage_remaining_bytes=100_000_000,
        )


def test_upsert_is_idempotent_and_cloud_errors_are_sanitized(
    tmp_path: Path,
) -> None:
    rows = select_sync_rows(_feature_frame(tmp_path), sample=True)

    class FakeFeatureGroup:
        def __init__(self) -> None:
            self.rows: dict[tuple[str, pd.Timestamp], dict] = {}
            self.calls = []

        def insert(self, frame, **kwargs):
            self.calls.append(kwargs)
            for record in frame.to_dict(orient="records"):
                key = (record["city_id"], record["event_timestamp"])
                self.rows[key] = record

        def read(self, *, start_time, end_time, online, dataframe_type):
            assert online is False
            assert dataframe_type == "pandas"
            return pd.DataFrame(
                [row for row in self.rows.values()
                 if start_time <= row["event_timestamp"] < end_time],
                columns=FEATURE_COLUMNS,
            )

    fake = FakeFeatureGroup()
    sync._upsert_and_verify(fake, rows)
    sync._upsert_and_verify(fake, rows)

    assert len(fake.rows) == len(rows)
    assert len(fake.calls) == 2
    assert all(call == {
        "overwrite": False,
        "operation": "upsert",
        "storage": "offline",
        "write_options": {
            "start_offline_materialization": True,
            "wait_for_job": True,
            "wait_for_online_ingestion": False,
        },
    } for call in fake.calls)

    class BrokenFeatureGroup:
        def insert(self, _frame, **_kwargs):
            raise RuntimeError("sensitive-placeholder")

    with pytest.raises(sync.SyncOperationError) as error:
        sync._upsert_and_verify(BrokenFeatureGroup(), rows)
    assert "RuntimeError" in str(error.value)
    assert "sensitive-placeholder" not in str(error.value)


def test_full_sync_progress_is_hash_bound_and_restartable(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    batch_size = sync.FULL_BATCH_SIZE
    row_count = 2 * batch_size + 1
    sync._write_progress(progress, "a" * 64, row_count, {0, batch_size})

    assert sync._read_progress(progress, "a" * 64, row_count) == {0, batch_size}
    with pytest.raises(sync.SyncOperationError, match="does not match"):
        sync._read_progress(progress, "b" * 64, row_count)


def test_full_sync_resumes_after_unverified_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _feature_frame(tmp_path)
    progress = tmp_path / "progress.json"
    digest = rows["source_dataset_sha256"].iloc[0]
    written, verified = [], []
    monkeypatch.setattr(sync, "FULL_BATCH_SIZE", 40)
    monkeypatch.setattr(sync, "_connect_to_feature_store", lambda: None)
    monkeypatch.setattr(sync, "_create_or_reuse_feature_group", lambda _store: None)
    monkeypatch.setattr(
        sync, "_wait_and_verify_read_back", lambda _group, batch: verified.append(batch.index[0])
    )

    def write_batch(_group, batch):
        start = batch.index[0]
        written.append(start)
        if written == [0, 40]:
            raise sync.SyncOperationError("read-back failed after write")

    monkeypatch.setattr(sync, "_upsert_and_verify", write_batch)
    with pytest.raises(sync.SyncOperationError, match="read-back failed"):
        sync._execute_sync(rows, digest, full=True, progress_path=progress)
    assert sync._read_progress(progress, digest, len(rows)) == {0}

    sync._execute_sync(rows, digest, full=True, progress_path=progress)
    assert written == [0, 40, 40]
    assert verified == [0]
    assert sync._read_progress(progress, digest, len(rows)) == {0, 40}


def test_read_back_retries_until_visible_and_fails_on_incorrect_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = select_sync_rows(_feature_frame(tmp_path), sample=True)
    responses = iter((rows.iloc[:0], rows))
    monkeypatch.setattr(sync, "READ_BACK_ATTEMPTS", 2)
    monkeypatch.setattr(sync.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(sync, "_read_expected_window", lambda *_args: next(responses))
    sync._wait_and_verify_read_back(None, rows)

    wrong = rows.assign(us_aqi=rows["us_aqi"] + 1)
    monkeypatch.setattr(sync, "_read_expected_window", lambda *_args: wrong)
    with pytest.raises(sync.SyncOperationError, match="values differ for us_aqi"):
        sync._wait_and_verify_read_back(None, rows)


def test_full_batch_encodes_warm_up_values_as_nullable_doubles(tmp_path: Path) -> None:
    rows = _feature_frame(tmp_path)

    class FakeFeatureGroup:
        def insert(self, payload, **_kwargs):
            self.payload = payload
            assert payload["temperature_c_lag_48h"].dtype == pd.Float64Dtype()
            assert payload["temperature_c_lag_48h"].iloc[0] is pd.NA

        def read(self, **_kwargs):
            return self.payload.copy()

    sync._upsert_and_verify(FakeFeatureGroup(), rows)
    assert rows["temperature_c_lag_48h"].dtype == np.dtype("float64")
