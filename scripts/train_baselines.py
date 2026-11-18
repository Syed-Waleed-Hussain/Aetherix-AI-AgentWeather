"""Generate local Day 3 EDA and chronological baseline model evaluations."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from aetherix_weather.eda import generate_eda, load_and_validate_features
from aetherix_weather.modeling import (
    REGRESSION_TASKS,
    model_feature_columns,
    run_experiments,
    run_rolling_backtests,
)
from aetherix_weather.storage import (
    read_csv,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)

MODEL_CONFIG = {
    "random_seed": 42,
    "ridge_alphas": [0.1, 1.0, 10.0],
    "random_forest_n_estimators": 40,
    "random_forest_max_depth": 10,
    "random_forest_min_samples_leaf": 4,
    "random_forest_max_features": "sqrt",
    "random_forest_n_jobs": 1,
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump_joblib_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/day3"))
    parser.add_argument(
        "--backtest-folds",
        type=int,
        default=3,
        help="Number of expanding-window validation folds (default: 3).",
    )
    parser.add_argument(
        "--development-run",
        action="store_true",
        help="Evaluate representative horizons 1, 6, 12, and 24 hours only.",
    )
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        help="Run only rolling-origin backtesting, skipping EDA and final test training.",
    )
    parser.add_argument(
        "--backtest-horizons",
        nargs="+",
        type=int,
        metavar="H",
        help="Horizon hours to process in this run (for example 1 24).",
    )
    parser.add_argument(
        "--resume-backtest",
        action="store_true",
        help="Reuse only matching dataset/configuration backtest rows, atomically.",
    )
    parser.add_argument(
        "--chronological-horizons",
        nargs="+",
        type=int,
        metavar="H",
        help="Missing chronological horizons to evaluate (for example 2 3 4 5).",
    )
    parser.add_argument(
        "--resume-chronological",
        action="store_true",
        help="Reuse matching chronological horizons and evaluate only missing horizons.",
    )
    args = parser.parse_args(argv)
    if args.backtest_folds < 1:
        parser.error("--backtest-folds must be at least one.")
    try:
        features_path = args.data_dir / "processed" / "features.csv"
        frame, schema = load_and_validate_features(features_path)
        chronological_horizons = (
            tuple(args.chronological_horizons)
            if args.chronological_horizons
            else ((1, 6, 12, 24) if args.development_run else None)
        )
        rolling_horizons = (
            tuple(args.backtest_horizons)
            if args.backtest_horizons
            else ((1, 6, 12, 24) if args.development_run else (1, 6, 12, 24))
        )
        horizons = chronological_horizons
        if horizons:
            invalid = [h for h in horizons if h < 1 or h > 24]
            if invalid:
                parser.error(f"Unsupported horizons: {invalid}")
        invalid_rolling = [h for h in rolling_horizons if h < 1 or h > 24]
        if invalid_rolling:
            parser.error(f"Unsupported rolling horizons: {invalid_rolling}")
        raw_dir = args.data_dir / "raw"
        source_quality_report = _read_json(
            args.data_dir / "reports" / "quality_report.json"
        )
        forecast_manifest = _read_json(
            raw_dir / "historical_forecast_manifest.json"
        )
        if forecast_manifest:
            summary = forecast_manifest.get("report_summary", {})
            if summary.get("used_as_feature") is not False:
                raise ValueError(
                    "Historical forecast probabilities must remain excluded from "
                    "Day 3 model inputs."
                )
        if args.backtest_only:
            started_at = time.monotonic()
            existing_rows: list[dict[str, Any]] = []
            metadata_path = args.output_dir / "rolling_backtest_metadata.json"
            csv_path = args.output_dir / "rolling_backtest.csv"
            existing_metadata = _read_json(metadata_path)
            dataset_hash = _sha256_file(features_path)
            reusable = (
                args.resume_backtest
                and existing_metadata.get("dataset_sha256") == dataset_hash
                and existing_metadata.get("fold_count") == args.backtest_folds
                and existing_metadata.get("model_config") == MODEL_CONFIG
            )
            selected = rolling_horizons
            missing_horizons = list(selected)
            if reusable:
                existing_rows = read_csv(csv_path)
                existing_horizons = {
                    int(row["horizon_hours"])
                    for row in existing_rows
                    if row.get("horizon_hours")
                }
                missing_horizons = [
                    h for h in selected if h not in existing_horizons
                ]
                print(
                    f"[resume] Reusing validated horizons: "
                    f"{sorted(set(selected) & existing_horizons)}",
                    flush=True,
                )
            if missing_horizons:
                backtest = run_rolling_backtests(
                    frame,
                    args.output_dir,
                    fold_count=args.backtest_folds,
                    dataset_path=features_path,
                    horizons=tuple(missing_horizons),
                )
                new_rows = read_csv(Path(backtest["metrics_path"]))
                if reusable:
                    existing_rows = [
                        row
                        for row in existing_rows
                        if int(row["horizon_hours"]) not in set(missing_horizons)
                    ]
                    existing_rows.extend(new_rows)
                    write_csv_atomic(csv_path, existing_rows)
                    report = backtest["report_path"]
                    from aetherix_weather.modeling import _rolling_backtest_markdown
                    folds = backtest["folds"]
                    write_text = __import__(
                        "aetherix_weather.storage", fromlist=["write_text_atomic"]
                    ).write_text_atomic
                    write_text(report, _rolling_backtest_markdown(existing_rows, folds, pd.to_datetime(frame["timestamp_utc"], utc=True)))
            else:
                backtest = {
                    "metrics_path": str(csv_path),
                    "report_path": str(args.output_dir / "rolling_backtest.md"),
                }
                print("[resume] No missing horizons; all requested jobs already valid.", flush=True)
            elapsed = time.monotonic() - started_at
            metadata = _read_json(metadata_path)
            if not reusable or missing_horizons:
                metadata["backtest_runtime_seconds"] = round(elapsed, 3)
            if reusable:
                metadata["horizons_hours"] = sorted(set(selected).union(
                    int(row["horizon_hours"])
                    for row in read_csv(csv_path)
                    if row.get("horizon_hours")
                ))
            write_json_atomic(metadata_path, metadata)
            print(
                f"[backtest-only] Completed requested horizons {list(selected)} "
                f"in {elapsed:.1f}s",
                flush=True,
            )
            return 0
        if args.resume_chronological:
            started_at = time.monotonic()
            selected = (
                chronological_horizons
                if chronological_horizons is not None
                else tuple(range(1, 25))
            )
            invalid = [h for h in selected if h < 1 or h > 24]
            if invalid:
                parser.error(f"Unsupported chronological horizons: {invalid}")

            metrics_path = args.output_dir / "model_evaluation.csv"
            predictions_path = args.output_dir / "test_predictions.csv"
            selection_path = args.output_dir / "selected_models.json"
            joblib_path = args.output_dir / "selected_models.joblib"

            dataset_hash = _sha256_file(features_path)
            existing_metadata = _read_json(selection_path)
            training_metadata = existing_metadata.get("training_metadata", {})
            existing_split = existing_metadata.get("split", {})
            expected_features = model_feature_columns(frame)
            reusable = (
                training_metadata.get("dataset_sha256") == dataset_hash
                and training_metadata.get("input_features") == expected_features
                and training_metadata.get("model_config") == MODEL_CONFIG
                and existing_split.get("row_count") == len(frame)
                and existing_split.get("train_end_index_exclusive") == 12213
                and existing_split.get("validation_end_index_exclusive") == 14830
                and existing_split.get("train_fraction") == 0.7
                and existing_split.get("validation_fraction") == 0.15
            )

            existing_rows = read_csv(metrics_path) if reusable else []
            completed_horizons: set[int] = set()
            if reusable:
                expected_tasks = set(REGRESSION_TASKS) | {
                    "wind_direction_degrees",
                    "rain_occurred",
                }
                metadata_tasks: dict[int, set[str]] = {}
                for item in existing_metadata.get("selected_models", []):
                    metadata_tasks.setdefault(
                        int(item["horizon_hours"]), set()
                    ).add(str(item["metric"]))
                rows_by_horizon_split: dict[int, dict[str, set[str]]] = {}
                for row in existing_rows:
                    if row.get("horizon_hours") and row.get("split"):
                        horizon = int(row["horizon_hours"])
                        rows_by_horizon_split.setdefault(
                            horizon, {}
                        ).setdefault(str(row["split"]), set()).add(
                            str(row["metric"])
                        )
                completed_horizons = {
                    horizon
                    for horizon, splits in rows_by_horizon_split.items()
                    if expected_tasks <= metadata_tasks.get(horizon, set())
                    and all(
                        expected_tasks <= splits.get(split_name, set())
                        for split_name in ("validation", "test")
                    )
                }
            missing = [h for h in selected if h not in completed_horizons]
            print(
                "[resume-chronological] Reusing validated horizons: "
                + ", ".join(f"h{h:02}" for h in sorted(completed_horizons & set(selected))),
                flush=True,
            )

            if missing:
                print(
                    "[resume-chronological] Starting missing horizons: "
                    + ", ".join(f"h{h:02}" for h in missing),
                    flush=True,
                )
                existing_predictions = (
                    read_csv(predictions_path) if reusable else []
                )
                previous_estimators = (
                    joblib.load(joblib_path)
                    if reusable
                    and joblib_path.is_file()
                    and joblib_path.stat().st_size > 0
                    else {}
                )
                run_experiments(
                    frame,
                    features_path,
                    args.output_dir,
                    manifest_path=raw_dir / "manifest.json",
                    horizons=tuple(missing),
                )

                new_metrics = read_csv(metrics_path)
                existing_rows = [
                    row for row in existing_rows
                    if int(row["horizon_hours"]) not in set(missing)
                ]
                existing_rows.extend(new_metrics)
                write_csv_atomic(metrics_path, existing_rows)

                existing_predictions = [
                    row for row in existing_predictions
                    if int(row["forecast_horizon_hours"]) not in set(missing)
                ]
                new_predictions = read_csv(predictions_path)
                existing_predictions.extend(new_predictions)
                write_csv_atomic(predictions_path, existing_predictions)

                previous_selection = existing_metadata
                new_selection = _read_json(selection_path)
                missing_set = set(missing)
                previous_models = (
                    previous_selection.get("selected_models", [])
                    if reusable
                    else []
                )
                previous_models = [
                    row for row in previous_models
                    if int(row["horizon_hours"]) not in missing_set
                ]
                merged_models = previous_models + [
                    row for row in new_selection.get("selected_models", [])
                    if int(row["horizon_hours"]) in missing_set
                ]

                previous_rain = (
                    previous_selection.get("training_metadata", {}).get(
                        "rain_label_distribution_by_horizon", {}
                    )
                    if reusable
                    else {}
                )
                new_rain = new_selection.get("training_metadata", {}).get(
                    "rain_label_distribution_by_horizon", {}
                )
                merged_rain = dict(previous_rain)
                merged_rain.update(new_rain)

                merged_selection = dict(new_selection)
                merged_selection["selected_models"] = merged_models
                merged_training = dict(new_selection.get("training_metadata", {}))
                if reusable:
                    merged_training["rain_label_distribution_by_horizon"] = merged_rain
                    merged_training["evaluation_horizons_hours"] = sorted(
                        set(completed_horizons).union(missing)
                    )
                merged_selection["training_metadata"] = merged_training
                merged_selection["training_metadata"]["model_config"] = MODEL_CONFIG
                write_json_atomic(selection_path, merged_selection)

                from aetherix_weather.modeling import _evaluation_markdown
                write_text_atomic(
                    args.output_dir / "model_evaluation.md",
                    _evaluation_markdown(merged_models),
                )

                new_estimators = (
                    joblib.load(joblib_path)
                    if joblib_path.is_file()
                    else {}
                )
                # run_experiments already replaced the joblib file; keep new keys
                # and preserve keys outside the requested horizon set.
                merged_estimators = dict(previous_estimators)
                merged_estimators.update(new_estimators)
                _dump_joblib_atomic(joblib_path, merged_estimators)
            else:
                print(
                    "[resume-chronological] No missing chronological horizons.",
                    flush=True,
                )

            elapsed = time.monotonic() - started_at
            print(
                "[resume-chronological] "
                f"Completed requested horizons {[f'h{h:02}' for h in selected]} "
                f"in {elapsed:.1f}s",
                flush=True,
            )
            return 0
        eda_report = generate_eda(
            frame,
            args.output_dir / "eda",
            schema,
            forecast_manifest=forecast_manifest,
            source_quality_report=source_quality_report,
        )
        result = run_experiments(
            frame,
            features_path,
            args.output_dir,
            manifest_path=raw_dir / "manifest.json",
            horizons=horizons,
        )
        backtest = run_rolling_backtests(
            frame,
            args.output_dir,
            fold_count=args.backtest_folds,
            dataset_path=features_path,
            horizons=rolling_horizons,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Day 3 training failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Analyzed {eda_report['dataset']['row_count']} ordered hourly rows; "
        f"{result['input_feature_count']} non-target numeric input features."
    )
    print(
        "Chronological issue-time row ranges: "
        f"train {result['base_split_rows']['train']}, "
        f"validation {result['base_split_rows']['validation']}, "
        f"test {result['base_split_rows']['test']} before horizon purges."
    )
    if result["evaluation_horizons_hours"] != list(range(1, 25)):
        print(
            "Development horizons evaluated: "
            + ", ".join(map(str, result["evaluation_horizons_hours"]))
        )
    print(f"EDA report: {args.output_dir / 'eda' / 'eda_report.md'}")
    print(f"Model report: {args.output_dir / 'model_evaluation.md'}")
    print(f"Metrics: {args.output_dir / 'model_evaluation.csv'}")
    print(f"Test predictions: {args.output_dir / 'test_predictions.csv'}")
    print(f"Rolling backtest: {backtest['metrics_path']}")
    print(f"Rolling backtest report: {backtest['report_path']}")
    if result["model_artifact_path"]:
        print(f"Selected model artifact: {result['model_artifact_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
