"""Dry-run or explicitly authorized sync of historical features to Hopsworks."""

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from aetherix_weather.hopsworks_features import (
    CITY_ID,
    FEATURE_COLUMNS,
    FEATURE_GROUP_DESCRIPTION,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURE_TYPES,
    FeatureDataError,
    FULL_BATCH_SIZE,
    FULL_QUOTA_SAFETY_FACTOR,
    MAX_SAMPLE_ROWS,
    MODEL_FEATURES,
    PRIMARY_KEY,
    estimate_csv_payload_bytes,
    read_feature_frame,
    select_sync_rows,
    validate_feature_frame,
    verify_round_trip,
    verify_secret_ignored,
)
from scripts.hopsworks_connection_check import _quiet_sdk_output

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "aetherix_agentweather"
READ_BACK_ATTEMPTS = 6
READ_BACK_DELAY_SECONDS = 5


class SyncOperationError(RuntimeError):
    """Raised for sanitized remote operation or read-back failures."""


def validate_execution_request(
    *,
    sample: bool,
    full: bool,
    dry_run: bool,
    execute: bool,
    confirm_write: bool,
    free_storage_remaining_bytes: int | None,
    confirm_free_plan: bool = False,
) -> None:
    if sample and full:
        raise ValueError("--sample and --full cannot be combined.")
    if dry_run and execute:
        raise ValueError("--dry-run and --execute cannot be combined.")
    if not (dry_run or sample or full):
        raise ValueError("Choose --dry-run, --sample, or --full.")
    if execute and not (sample or full):
        raise ValueError("--execute requires --sample or --full.")
    if execute and not confirm_write:
        raise ValueError("--execute requires the explicit --confirm-write flag.")
    if confirm_write and not execute:
        raise ValueError("--confirm-write is valid only with --execute.")
    if full and execute and free_storage_remaining_bytes is None:
        raise ValueError(
            "--full --execute requires --free-storage-remaining-bytes "
            "copied from the Hopsworks Free-plan quota page."
        )
    if full and execute and not confirm_free_plan:
        raise ValueError(
            "--full --execute requires --confirm-free-plan after checking that "
            "offline materialization and read-back fit the project's Free-plan compute quota."
        )
    if confirm_free_plan and not (full and execute):
        raise ValueError("--confirm-free-plan is valid only with --full --execute.")
    if free_storage_remaining_bytes is not None and free_storage_remaining_bytes < 0:
        raise ValueError("--free-storage-remaining-bytes must be zero or greater.")
    if sample and free_storage_remaining_bytes is not None:
        raise ValueError("The free-storage quota option applies only to --full.")


def _quiet_sdk_call(action: Callable[[], Any]) -> Any:
    with _quiet_sdk_output():
        return action()


def _connect_to_project() -> Any:
    try:
        from dotenv import load_dotenv

        _quiet_sdk_call(lambda: load_dotenv(PROJECT_ROOT / ".env"))
    except Exception as exc:
        raise SyncOperationError(
            f"Hopsworks configuration could not be loaded ({type(exc).__name__})."
        ) from None
    api_key = os.environ.get("HOPSWORKS_API_KEY", "")
    project_name = os.environ.get("HOPSWORKS_PROJECT", "").strip()
    host = os.environ.get("HOPSWORKS_HOST", "").strip()
    missing = [
        name
        for name, value in (
            ("HOPSWORKS_API_KEY", api_key.strip()),
            ("HOPSWORKS_PROJECT", project_name),
            ("HOPSWORKS_HOST", host),
        )
        if not value
    ]
    if missing:
        raise SyncOperationError(
            "Missing Hopsworks configuration: " + ", ".join(missing)
        )

    if project_name != PROJECT_NAME:
        raise SyncOperationError(f"HOPSWORKS_PROJECT must be {PROJECT_NAME}.")
    try:
        if version("hopsworks") != "5.2.3":
            raise SyncOperationError("Run cloud operations with Hopsworks SDK 5.2.3.")

        with _quiet_sdk_output():
            import hopsworks

            project = hopsworks.login(
                project=project_name,
                api_key_value=api_key,
                host=host,
                engine="python",
            )
        if project.name != project_name:
            raise SyncOperationError(
                "The connected Hopsworks project does not match HOPSWORKS_PROJECT."
            )
        return project
    except SyncOperationError:
        raise
    except Exception as exc:
        raise SyncOperationError(
            f"Hopsworks login failed "
            f"({type(exc).__name__})."
        ) from None


def _connect_to_feature_store() -> Any:
    project = _connect_to_project()
    try:
        return _quiet_sdk_call(project.get_feature_store)
    except Exception as exc:
        raise SyncOperationError(
            f"Feature Store access failed ({type(exc).__name__})."
        ) from None


def _create_or_reuse_feature_group(feature_store: Any) -> Any:
    try:
        from hsfs.feature import Feature

        features = [
            Feature(
                name=name,
                type=feature_type,
                description=f"AETHERIX offline inference feature: {name}.",
            )
            for name, feature_type in FEATURE_TYPES.items()
        ]
        feature_group = _quiet_sdk_call(
            lambda: feature_store.get_or_create_feature_group(
                name=FEATURE_GROUP_NAME,
                version=FEATURE_GROUP_VERSION,
                description=FEATURE_GROUP_DESCRIPTION,
                primary_key=list(PRIMARY_KEY),
                event_time="event_timestamp",
                online_enabled=False,
                time_travel_format="HUDI",
                stream=True,
                statistics_config=False,
                features=features,
            )
        )
        _quiet_sdk_call(lambda: _verify_feature_group_schema(feature_group))
        return feature_group
    except SyncOperationError:
        raise
    except Exception as exc:
        raise SyncOperationError(
            f"Feature Group create/reuse failed ({type(exc).__name__})."
        ) from None


def _feature_name(value: Any) -> str:
    return str(getattr(value, "name", value))


def _verify_feature_group_schema(feature_group: Any) -> None:
    actual_features = getattr(feature_group, "columns", None)
    if actual_features is None:
        raise SyncOperationError("Feature Group schema could not be inspected.")
    actual_schema = {
        feature.name: str(feature.type).lower() for feature in actual_features
    }
    if actual_schema != FEATURE_TYPES:
        raise SyncOperationError("Existing Feature Group schema does not match.")

    actual_keys = getattr(feature_group, "primary_key", None)
    if actual_keys is None or set(map(_feature_name, actual_keys)) != set(PRIMARY_KEY):
        raise SyncOperationError("Feature Group primary key does not match.")
    if _feature_name(getattr(feature_group, "event_time", None)) != "event_timestamp":
        raise SyncOperationError("Feature Group event-time column does not match.")
    if getattr(feature_group, "online_enabled", None) is not False:
        raise SyncOperationError(
            "Feature Group is not confirmed offline-only; refusing to write."
        )
    if (
        getattr(feature_group, "time_travel_format", None) != "HUDI"
        or getattr(feature_group, "stream", None) is not True
        or getattr(feature_group, "sink_enabled", False)
        or getattr(feature_group, "transformation_functions", ())
    ):
        raise SyncOperationError(
            "Feature Group must use Hudi streaming without a sink or transformations."
        )


def _read_expected_window(
    feature_group: Any, expected: pd.DataFrame
) -> pd.DataFrame:
    first = expected["event_timestamp"].min()
    last_exclusive = expected["event_timestamp"].max() + timedelta(hours=1)
    actual = _quiet_sdk_call(
        lambda: feature_group.read(
            online=False,
            start_time=first.to_pydatetime(),
            end_time=last_exclusive.to_pydatetime(),
            dataframe_type="pandas",
        )
    )
    if not isinstance(actual, pd.DataFrame):
        raise FeatureDataError("Feature Group read-back was not a Pandas DataFrame.")
    if not set(FEATURE_COLUMNS).issubset(actual.columns):
        raise FeatureDataError("Feature Group read-back schema is incomplete.")

    return actual.loc[actual["city_id"].eq(CITY_ID), FEATURE_COLUMNS].reset_index(drop=True)


def _wait_and_verify_read_back(
    feature_group: Any, expected: pd.DataFrame
) -> None:
    last_error = "no matching records became visible"
    for attempt in range(READ_BACK_ATTEMPTS):
        try:
            actual = _read_expected_window(feature_group, expected)
            verify_round_trip(expected, actual)
            return
        except FeatureDataError as exc:
            last_error = str(exc)
        except Exception as exc:
            last_error = type(exc).__name__
        if attempt + 1 < READ_BACK_ATTEMPTS:
            time.sleep(READ_BACK_DELAY_SECONDS)
    raise SyncOperationError(
        f"Feature Group read-back verification failed ({last_error})."
    )


def _read_offline_group(feature_group: Any) -> pd.DataFrame:
    try:
        return _quiet_sdk_call(
            lambda: feature_group.read(online=False, dataframe_type="pandas")
        )
    except Exception as exc:
        raise SyncOperationError(
            f"Offline Feature Group read failed ({type(exc).__name__})."
        ) from None


def _verify_existing_history(feature_group: Any, expected: pd.DataFrame) -> pd.DataFrame:
    actual = _read_offline_group(feature_group)
    if not isinstance(actual, pd.DataFrame) or not set(FEATURE_COLUMNS).issubset(actual.columns):
        raise FeatureDataError("Offline Feature Group read-back schema is incomplete.")
    timestamps = pd.to_datetime(actual["event_timestamp"], utc=True, errors="raise")
    verify_round_trip(
        expected.loc[expected["event_timestamp"].isin(timestamps)],
        actual,
    )
    sample = select_sync_rows(expected, sample=True)
    verify_round_trip(sample, actual.loc[timestamps.isin(sample["event_timestamp"])])
    return actual.assign(event_timestamp=timestamps)


def _get_existing_feature_group(feature_store: Any) -> Any:
    try:
        group = _quiet_sdk_call(
            lambda: feature_store.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        )
    except Exception as exc:
        raise SyncOperationError(
            f"Existing Feature Group lookup failed ({type(exc).__name__})."
        ) from None
    _verify_feature_group_schema(group)
    return group


def _upsert_and_verify(feature_group: Any, rows: pd.DataFrame) -> None:
    # SDK Avro serialization maps pandas.NA to null, while float NaN stays NaN.
    nullable = {name: "Float64" for name in MODEL_FEATURES if rows[name].isna().any()}
    payload = rows.astype(nullable) if nullable else rows
    try:
        _quiet_sdk_call(
            lambda: feature_group.insert(
                payload,
                overwrite=False,
                operation="upsert",
                storage="offline",
                write_options={
                    "start_offline_materialization": True,
                    "wait_for_job": True,
                    "wait_for_online_ingestion": False,
                },
            )
        )
    except Exception as exc:
        raise SyncOperationError(
            f"Feature Group upsert failed ({type(exc).__name__})."
        ) from None
    _wait_and_verify_read_back(feature_group, rows)


def _read_progress(
    path: Path, dataset_hash: str, row_count: int
) -> set[int]:
    if not path.exists():
        return set()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyncOperationError("Sync progress file is unreadable.") from exc
    expected_state = {
        "feature_group": FEATURE_GROUP_NAME,
        "feature_group_version": FEATURE_GROUP_VERSION,
        "dataset_sha256": dataset_hash,
        "row_count": row_count,
        "batch_size": FULL_BATCH_SIZE,
    }
    if not isinstance(state, dict) or any(
        state.get(key) != value for key, value in expected_state.items()
    ):
        raise SyncOperationError(
            "Sync progress does not match this dataset and schema; inspect it "
            "before continuing."
        )
    completed = state.get("completed_batch_starts")
    if not isinstance(completed, list) or any(
        not isinstance(value, int)
        or value < 0
        or value >= row_count
        or value % FULL_BATCH_SIZE != 0
        for value in completed
    ):
        raise SyncOperationError("Sync progress batch list is invalid.")
    return set(completed)


def _write_progress(
    path: Path, dataset_hash: str, row_count: int, completed: set[int]
) -> None:
    state = {
        "feature_group": FEATURE_GROUP_NAME,
        "feature_group_version": FEATURE_GROUP_VERSION,
        "dataset_sha256": dataset_hash,
        "row_count": row_count,
        "batch_size": FULL_BATCH_SIZE,
        "completed_batch_starts": sorted(completed),
    }
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(state, handle, indent=2)
            handle.write("\n")
        temporary.replace(path)
    except OSError:
        raise SyncOperationError("Verified batch could not be checkpointed.") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _execute_sync(
    rows: pd.DataFrame,
    dataset_hash: str,
    *,
    full: bool,
    progress_path: Path,
) -> None:
    if not full and not 1 <= len(rows) <= MAX_SAMPLE_ROWS:
        raise SyncOperationError("Sample must contain between 1 and 24 rows.")
    completed = _read_progress(progress_path, dataset_hash, len(rows)) if full else set()
    feature_store = _connect_to_feature_store()
    if not full:
        feature_group = _create_or_reuse_feature_group(feature_store)
        _upsert_and_verify(feature_group, rows)
        print(f"Sample cloud round trip: verified {len(rows)} rows.")
        return

    feature_group = _get_existing_feature_group(feature_store)
    existing = _verify_existing_history(feature_group, rows)
    for start in range(0, len(rows), FULL_BATCH_SIZE):
        batch = rows.iloc[start : start + FULL_BATCH_SIZE].copy()
        try:
            if start in completed:
                _wait_and_verify_read_back(feature_group, batch)
                continue
            recovered = existing.loc[existing["event_timestamp"].isin(batch["event_timestamp"])]
            if len(recovered) == len(batch):
                verify_round_trip(batch, recovered)
            else:
                _upsert_and_verify(feature_group, batch)
        except SyncOperationError as exc:
            raise SyncOperationError(
                f"Batch starting at row offset {start}: {exc} "
                "Resume with the same full-sync command and current remaining quota."
            ) from None
        completed.add(start)
        _write_progress(progress_path, dataset_hash, len(rows), completed)
    verify_round_trip(rows, _read_offline_group(feature_group))
    print(f"Full cloud round trip: verified {len(rows)} rows.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--sample", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument(
        "--preflight", action="store_true",
        help="Inspect the existing offline group and verify its rows without writing.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-write", action="store_true")
    parser.add_argument(
        "--free-storage-remaining-bytes",
        type=int,
        help="Current remaining Free-plan storage from the Hopsworks quota page.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "features.csv",
        help="Processed feature CSV (default: repository data/processed/features.csv).",
    )
    parser.add_argument(
        "--confirm-free-plan",
        action="store_true",
        help="Confirm Free-plan offline materialization and query compute capacity.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.preflight and (not args.full or args.execute):
            raise ValueError("--preflight requires --full and runs read-only.")
        validate_execution_request(
            sample=args.sample,
            full=args.full,
            dry_run=args.dry_run,
            execute=args.execute,
            confirm_write=args.confirm_write,
            free_storage_remaining_bytes=args.free_storage_remaining_bytes,
            confirm_free_plan=args.confirm_free_plan,
        )
        verify_secret_ignored(PROJECT_ROOT)
        features, dataset_hash = read_feature_frame(args.data)
        report = validate_feature_frame(features)
        selected = select_sync_rows(features, sample=args.sample)
        payload_bytes = estimate_csv_payload_bytes(selected)
        quota_estimate = payload_bytes * FULL_QUOTA_SAFETY_FACTOR
        if (
            args.full
            and args.free_storage_remaining_bytes is not None
            and quota_estimate > args.free_storage_remaining_bytes
        ):
            raise ValueError(
                "Estimated storage budget exceeds the supplied remaining quota."
            )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 2

    scope = "sample" if args.sample else "full history"
    print(f"Feature Group: {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION} (offline only)")
    print(f"Dataset rows: {report['rows']}; selected rows: {len(selected)} ({scope})")
    print(f"Dataset SHA-256: {dataset_hash}")
    print(
        "Event time: "
        f"{selected['event_timestamp'].iloc[0].isoformat()} through "
        f"{selected['event_timestamp'].iloc[-1].isoformat()}"
    )
    print(
        "Schema: "
        f"{len(FEATURE_COLUMNS)} columns "
        f"({len(MODEL_FEATURES)} inference predictors; no target labels)"
    )
    missing = {
        name: int(count)
        for name, count in selected.loc[:, MODEL_FEATURES].isna().sum().items()
        if count
    }
    print(f"Missing predictor values: {missing or 'none'}")
    print(
        f"Primary key: {', '.join(PRIMARY_KEY)} "
        "(unique and validated)"
    )
    print(
        f"Estimated payload: {payload_bytes:,} bytes UTF-8 CSV-equivalent "
        "(uncompressed)"
    )
    if not args.sample:
        batches = (len(selected) + FULL_BATCH_SIZE - 1) // FULL_BATCH_SIZE
        print(
            f"Full-sync quota budget: {quota_estimate:,} bytes "
            f"({FULL_QUOTA_SAFETY_FACTOR}x CSV estimate), {batches} materialization batches"
        )
    print("Secret ignore check: passed (.env is ignored and untracked)")

    if not args.sample and args.free_storage_remaining_bytes is None:
        print(
            "Free-plan quota: not yet verified; full execution requires the "
            "remaining-storage value and confirmation of available Free-plan compute."
        )
    elif args.full:
        print("Free-plan quota: supplied remaining-storage value covers the estimate")

    if args.preflight:
        try:
            feature_store = _connect_to_feature_store()
            group = _get_existing_feature_group(feature_store)
            print("Remote Feature Group schema verified.")
            count = len(_verify_existing_history(group, selected))
        except (SyncOperationError, FeatureDataError) as exc:
            print(f"Cloud preflight failed: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Cloud preflight failed ({type(exc).__name__}).", file=sys.stderr)
            return 1
        print(f"Remote offline row count: {count}; sample and all existing rows verified.")
        return 0

    if not args.execute:
        print("Execution: dry run only; no Hopsworks connection or remote writes.")
        if args.sample:
            print(
                "Planned remote operations: create or reuse the offline-only "
                "Feature Group; upsert at most 24 rows; wait; read back and verify."
            )
        else:
            print(
                "Planned remote operations: verify the existing offline-only "
                "Feature Group; deterministic upsert batches; checkpoint progress; "
                "read back and verify every batch."
            )
        return 0

    try:
        _execute_sync(
            selected,
            dataset_hash,
            full=args.full,
            progress_path=PROJECT_ROOT / "outputs" / "hopsworks_sync"
            / f"{FEATURE_GROUP_NAME}_v{FEATURE_GROUP_VERSION}.json",
        )
    except (SyncOperationError, FeatureDataError) as exc:
        print(f"Cloud sync failed: {exc}", file=sys.stderr)
        return 1
    print("Cloud operation completed; local source data was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
