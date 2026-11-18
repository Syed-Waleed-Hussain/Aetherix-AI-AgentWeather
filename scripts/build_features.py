"""Rebuild model-ready features and 24-hour targets from stored raw CSV files."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from aetherix_weather.dataset import (
    add_historical_forecast_information,
    build_feature_dataset,
)
from aetherix_weather.storage import (
    read_csv,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)
from scripts.backfill import _report_markdown


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-forward-fill-hours",
        type=int,
        default=0,
        help="Maximum causal fill duration for eligible weather fields (default: 0).",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    try:
        if args.max_forward_fill_hours < 0:
            parser.error("--max-forward-fill-hours must be zero or greater.")
        raw_dir = args.data_dir / "raw"
        manifest = _read_manifest(raw_dir / "manifest.json")
        weather_rows = read_csv(raw_dir / "weather.csv")
        air_quality_rows = read_csv(raw_dir / "air_quality.csv")
        if not weather_rows or not air_quality_rows:
            raise ValueError(
                "Both raw/weather.csv and raw/air_quality.csv must contain data; "
                "run python -m scripts.backfill first."
            )
        requested = manifest.get("requested_date_range", {})
        processed, report = build_feature_dataset(
            weather_rows,
            air_quality_rows,
            max_forward_fill_hours=args.max_forward_fill_hours,
            availability=manifest.get("availability", {}),
            requested_start_date=requested.get("start", ""),
            requested_end_date=requested.get("end", ""),
        )
        forecast_manifest_path = raw_dir / "historical_forecast_manifest.json"
        forecast_manifest = _read_manifest(forecast_manifest_path)
        add_historical_forecast_information(report, forecast_manifest)
        last_ingestion_range = manifest.get("last_ingestion_date_range")
        if last_ingestion_range:
            report["last_ingestion_date_range"] = last_ingestion_range
        current_feature_schema = report["feature_schema"]
        previous_feature_schema = manifest.get("feature_schema")
        if (
            previous_feature_schema is not None
            and previous_feature_schema != current_feature_schema
        ):
            report["quality_issues"].append(
                {
                    "severity": "warning",
                    "code": "feature_schema_changed",
                    "detail": "The generated feature columns changed from the previous manifest.",
                }
            )
        processed_dir = args.data_dir / "processed"
        reports_dir = args.data_dir / "reports"
        features_path = processed_dir / "features.csv"
        write_csv_atomic(features_path, processed)
        report["dataset_version_sha256"] = _sha256_file(features_path)
        write_json_atomic(reports_dir / "quality_report.json", report)
        write_text_atomic(
            reports_dir / "quality_report.md", _report_markdown(report)
        )
        if manifest:
            manifest["feature_schema"] = current_feature_schema
            manifest["returned_date_ranges_local"] = report[
                "returned_date_ranges_local"
            ]
            manifest["coverage_limitations"] = report["coverage_limitations"]
            manifest["source_variable_coverage"] = report[
                "source_variable_coverage"
            ]
            manifest["feature_dataset_sha256"] = report["dataset_version_sha256"]
            write_json_atomic(raw_dir / "manifest.json", manifest)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Feature build failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Built {len(processed)} feature rows at "
        f"{args.data_dir / 'processed' / 'features.csv'}."
    )
    print(f"Quality report: {args.data_dir / 'reports' / 'quality_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
