"""Run a bounded SARIMAX evaluation against saved Day 3.5 results."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from aetherix_weather.eda import load_and_validate_features
from aetherix_weather.statistical import (
    DEFAULT_CONFIG,
    SUPPORTED_METRICS,
    SarimaxConfig,
    run_sarimax_experiment,
)
from aetherix_weather.storage import (
    read_csv,
    write_csv_atomic,
    write_text_atomic,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _comparison_rows(
    statistical_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, str]],
    metric: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    horizons = {int(row["horizon_hours"]) for row in statistical_rows}
    for stat_row in statistical_rows:
        output.append(
            {
                **stat_row,
                "comparison_sample_match": True,
            }
        )
    for row in baseline_rows:
        if (
            row.get("metric") != metric
            or not row.get("horizon_hours")
            or int(row["horizon_hours"]) not in horizons
        ):
            continue
        stat_row = next(
            item
            for item in statistical_rows
            if item["split"] == row["split"]
            and int(item["horizon_hours"]) == int(row["horizon_hours"])
        )
        output.append(
            {
                "split": row["split"],
                "metric": metric,
                "horizon_hours": int(row["horizon_hours"]),
                "model": row["model"],
                "model_version": "day3_5",
                "n": int(row["n"]),
                "mae": _optional_float(row.get("mae")),
                "rmse": _optional_float(row.get("rmse")),
                "r2": _optional_float(row.get("r2")),
                "comparison_sample_match": int(row["n"]) == int(stat_row["n"]),
            }
        )
    return sorted(
        output,
        key=lambda item: (
            item["split"],
            int(item["horizon_hours"]),
            item["model"],
        ),
    )


def _optional_float(value: str | None) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return None
    return float(value)


def _comparison_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Statistical model comparison",
        "",
        "Baseline metrics are copied from the verified Day 3.5 evaluation. "
        "Rows with `same sample = false` are not directly comparable because "
        "the run used a capped pilot subset.",
        "",
        "| Split | Horizon | Model | N | Same sample | MAE | RMSE | R2 |",
        "| --- | ---: | --- | ---: | :---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        r2 = "n/a" if row["r2"] is None else f"{row['r2']:.6f}"
        lines.append(
            f"| {row['split']} | h{int(row['horizon_hours']):02} | "
            f"{row['model']} | {row['n']} | "
            f"{'yes' if row['comparison_sample_match'] else 'no'} | "
            f"{row['mae']:.6f} | {row['rmse']:.6f} | {r2} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/day4a/pilot"))
    parser.add_argument(
        "--baseline-dir", type=Path, default=Path("artifacts/day3_5")
    )
    parser.add_argument(
        "--metric",
        choices=SUPPORTED_METRICS,
        default="temperature_c",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 6, 12, 24],
    )
    parser.add_argument(
        "--max-origins-per-split",
        type=int,
        default=168,
        help="Limit validation/test forecast origins for the initial pilot.",
    )
    parser.add_argument(
        "--full-periods",
        action="store_true",
        help="Evaluate every eligible validation and test origin.",
    )
    parser.add_argument(
        "--nonseasonal",
        action="store_true",
        help="Use a prespecified nonseasonal SARIMAX(1,0,0) instead of daily seasonal AR(1).",
    )
    args = parser.parse_args(argv)
    if any(horizon < 1 or horizon > 24 for horizon in args.horizons):
        parser.error("--horizons values must be between 1 and 24.")
    if args.max_origins_per_split < 1:
        parser.error("--max-origins-per-split must be positive.")
    started_at = time.monotonic()
    try:
        features_path = args.data_dir / "processed" / "features.csv"
        baseline_path = args.baseline_dir / "model_evaluation.csv"
        baseline_metadata_path = args.baseline_dir / "selected_models.json"
        frame, _ = load_and_validate_features(features_path)
        dataset_hash = hashlib.sha256(features_path.read_bytes()).hexdigest()
        baseline_metadata = _read_json(baseline_metadata_path)
        baseline_hash = baseline_metadata.get("training_metadata", {}).get(
            "dataset_sha256"
        )
        if dataset_hash != baseline_hash:
            raise ValueError(
                "Day 3.5 baseline artifacts do not match the current feature dataset."
            )
        if not baseline_path.is_file():
            raise ValueError(f"Baseline evaluation file is missing: {baseline_path}")

        result = run_sarimax_experiment(
            frame,
            features_path,
            args.output_dir,
            metric=args.metric,
            horizons=tuple(args.horizons),
            max_origins_per_split=(
                None if args.full_periods else args.max_origins_per_split
            ),
            config=(
                SarimaxConfig(
                    order=(1, 0, 0),
                    seasonal_order=(0, 0, 0, 0),
                    trend="c",
                    maxiter=40,
                )
                if args.nonseasonal
                else DEFAULT_CONFIG
            ),
        )
        comparison = _comparison_rows(
            result["metrics"], read_csv(baseline_path), args.metric
        )
        comparison_path = args.output_dir / "model_comparison.csv"
        write_csv_atomic(comparison_path, comparison)
        comparison_report_path = args.output_dir / "model_comparison.md"
        write_text_atomic(
            comparison_report_path, _comparison_markdown(comparison)
        )
        mismatches = sum(
            row["comparison_sample_match"] is False
            for row in comparison
            if row["model_version"] != "sarimax_day3_5"
        )
        print(
            f"[sarimax] Completed {args.metric} horizons {args.horizons}; "
            f"runtime={time.monotonic() - started_at:.1f}s; "
            f"baseline sample mismatches={mismatches}",
            flush=True,
        )
        print(f"Metrics: {result['metrics_path']}")
        print(f"Forecasts: {result['forecasts_path']}")
        print(f"Metadata: {result['metadata_path']}")
        print(f"Model comparison: {comparison_path}")
        print(f"Comparison report: {comparison_report_path}")
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Statistical forecasting failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
