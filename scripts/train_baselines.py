"""Generate local Day 3 EDA and chronological baseline model evaluations."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from aetherix_weather.eda import generate_eda, load_and_validate_features
from aetherix_weather.modeling import run_experiments


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/day3"))
    args = parser.parse_args(argv)
    try:
        features_path = args.data_dir / "processed" / "features.csv"
        frame, schema = load_and_validate_features(features_path)
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
    print(f"EDA report: {args.output_dir / 'eda' / 'eda_report.md'}")
    print(f"Model report: {args.output_dir / 'model_evaluation.md'}")
    print(f"Metrics: {args.output_dir / 'model_evaluation.csv'}")
    print(f"Test predictions: {args.output_dir / 'test_predictions.csv'}")
    if result["model_artifact_path"]:
        print(f"Selected model artifact: {result['model_artifact_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
