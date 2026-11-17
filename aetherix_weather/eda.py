"""Exploratory analysis and local charts for the processed hourly dataset."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataset import RAIN_OCCURRENCE_THRESHOLD_MM
from .storage import write_json_atomic, write_text_atomic

CORE_METRICS = (
    "temperature_c",
    "relative_humidity_percent",
    "precipitation_mm",
    "wind_speed_kmh",
    "wind_direction_degrees",
    "us_aqi",
    "pm2_5_ug_m3",
    "uv_index",
)


def load_and_validate_features(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load actual processed rows and validate their timestamps and schema."""
    if not path.is_file():
        raise FileNotFoundError(f"Processed feature dataset not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Processed feature dataset is empty: {path}")
    required = {"timestamp_utc", "timestamp_local"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Processed dataset is missing columns: {sorted(missing)}")

    raw_utc = frame["timestamp_utc"].map(pd.Timestamp)
    if raw_utc.map(lambda value: value.tzinfo is None).any():
        raise ValueError("Every timestamp_utc value must include a timezone.")
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    if timestamps.duplicated().any():
        raise ValueError("Processed dataset contains duplicate UTC timestamps.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Processed dataset timestamps must be strictly increasing.")
    if len(timestamps) > 1 and not timestamps.diff().iloc[1:].eq(
        pd.Timedelta(1, unit="h")
    ).all():
        raise ValueError("Processed dataset must have a complete hourly timeline.")
    local_stamps = frame["timestamp_local"].map(pd.Timestamp)
    if local_stamps.map(lambda value: value.tzinfo is None).any():
        raise ValueError("Every timestamp_local value must include a timezone.")
    if not (
        pd.to_datetime(frame["timestamp_local"], utc=True, errors="raise")
        .reset_index(drop=True)
        .equals(timestamps.reset_index(drop=True))
    ):
        raise ValueError("UTC and local timestamps do not represent identical instants.")

    frame = frame.copy()
    frame["_timestamp_utc"] = timestamps
    frame["_timestamp_local"] = local_stamps
    schema = {
        "row_count": int(len(frame)),
        "column_count": int(
            sum(not column.startswith("_timestamp_") for column in frame.columns)
        ),
        "utc_start": timestamps.iloc[0].isoformat(),
        "utc_end": timestamps.iloc[-1].isoformat(),
        "timestamp_ordered": True,
        "duplicate_utc_timestamps": 0,
        "missing_hourly_periods": 0,
        "columns": {
            column: str(frame[column].dtype)
            for column in frame.columns
            if not column.startswith("_timestamp_")
        },
        "target_columns": [
            column for column in frame.columns if column.startswith("target_")
        ],
        "target_column_count": sum(
            column.startswith("target_") for column in frame.columns
        ),
        "missing_percent_by_column": {
            column: round(float(frame[column].isna().mean() * 100), 4)
            for column in frame.columns
            if not column.startswith("_timestamp_")
        },
        "source_metadata_counts": {
            column: {
                str(key): int(value)
                for key, value in frame[column].value_counts(dropna=False).items()
            }
            for column in (
                "weather_source",
                "air_quality_source",
                "uv_index_source",
            )
            if column in frame.columns
        },
    }
    return frame, schema


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _require_columns(frame: pd.DataFrame, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"Processed dataset is missing required metrics: {missing}")


def generate_eda(
    frame: pd.DataFrame,
    output_dir: Path,
    schema: dict[str, Any],
    forecast_manifest: dict[str, Any] | None = None,
    source_quality_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write charts, a machine-readable inspection, and a concise EDA report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _require_columns(
        frame,
        (
            "temperature_c",
            "relative_humidity_percent",
            "precipitation_mm",
            "wind_speed_kmh",
            "wind_direction_degrees",
            "us_aqi",
            "pm2_5_ug_m3",
            "uv_index",
        ),
    )
    local_hour = frame["_timestamp_local"].dt.hour
    rain = frame["precipitation_mm"].gt(RAIN_OCCURRENCE_THRESHOLD_MM).where(
        frame["precipitation_mm"].notna()
    )
    analysis = frame.copy()
    analysis["rain_occurred"] = rain
    analysis["hour_of_day_local"] = local_hour

    temperature_by_hour = analysis.groupby("hour_of_day_local")[
        "temperature_c"
    ].agg(["mean", "median"])
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(temperature_by_hour.index, temperature_by_hour["mean"], label="Mean")
    axis.plot(
        temperature_by_hour.index,
        temperature_by_hour["median"],
        label="Median",
        linestyle="--",
    )
    axis.set(title="Temperature by Karachi local hour", xlabel="Local hour", ylabel="°C")
    axis.set_xticks(range(0, 24, 2))
    axis.legend()
    _save_figure(figure, output_dir / "temperature_hourly_pattern.png")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    analysis["relative_humidity_percent"].dropna().plot.hist(
        bins=20, ax=axes[0], edgecolor="white"
    )
    axes[0].set(
        title="Relative humidity distribution",
        xlabel="Relative humidity (%)",
        ylabel="Hourly records",
    )
    humidity_by_hour = analysis.groupby("hour_of_day_local")[
        "relative_humidity_percent"
    ].median()
    axes[1].plot(humidity_by_hour.index, humidity_by_hour.values)
    axes[1].set(
        title="Median humidity by local hour",
        xlabel="Local hour",
        ylabel="Relative humidity (%)",
        xticks=range(0, 24, 3),
    )
    _save_figure(figure, output_dir / "humidity_distribution.png")

    daily_rain = analysis.set_index("_timestamp_local")["precipitation_mm"].resample(
        "D"
    ).sum(min_count=1)
    rain_rate = analysis.groupby("hour_of_day_local")["rain_occurred"].mean() * 100
    figure, axes = plt.subplots(2, 1, figsize=(11, 7))
    axes[0].bar(daily_rain.index, daily_rain.values, width=0.8)
    axes[0].set(title="Daily precipitation amount", ylabel="Precipitation (mm)")
    axes[1].bar(rain_rate.index, rain_rate.values)
    axes[1].set(
        title=f"Rain occurrence by local hour (precipitation > "
        f"{RAIN_OCCURRENCE_THRESHOLD_MM:g} mm)",
        xlabel="Local hour",
        ylabel="Hours with rain (%)",
        xticks=range(0, 24, 2),
    )
    _save_figure(figure, output_dir / "rainfall_and_occurrence.png")

    figure = plt.figure(figsize=(11, 5))
    grid = figure.add_gridspec(1, 2)
    speed_axis = figure.add_subplot(grid[0, 0])
    direction_axis = figure.add_subplot(grid[0, 1], projection="polar")
    speed_by_hour = analysis.groupby("hour_of_day_local")["wind_speed_kmh"].mean()
    speed_axis.plot(speed_by_hour.index, speed_by_hour.values)
    speed_axis.set(
        title="Mean wind speed by local hour",
        xlabel="Local hour",
        ylabel="Wind speed (km/h)",
        xticks=range(0, 24, 3),
    )
    directions = analysis["wind_direction_degrees"].dropna().to_numpy()
    bins = np.linspace(0, 2 * np.pi, 17)
    counts, edges = np.histogram(np.deg2rad(directions), bins=bins)
    direction_axis.set_theta_zero_location("N")
    direction_axis.set_theta_direction(-1)
    direction_axis.bar(
        edges[:-1],
        counts,
        width=np.diff(edges),
        align="edge",
        edgecolor="white",
    )
    direction_axis.set_title("Wind direction frequency\n(from north, clockwise)")
    _save_figure(figure, output_dir / "wind_speed_and_direction.png")

    daily_air = analysis.set_index("_timestamp_local")[
        ["us_aqi", "pm2_5_ug_m3"]
    ].resample("D").mean()
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    daily_air["us_aqi"].plot(ax=axes[0])
    axes[0].set(title="Daily mean US AQI", ylabel="US AQI")
    daily_air["pm2_5_ug_m3"].plot(ax=axes[1], color="tab:orange")
    axes[1].set(title="Daily mean PM2.5", ylabel="PM2.5 (µg/m³)", xlabel="Date")
    _save_figure(figure, output_dir / "air_quality_patterns.png")

    uv_by_hour = analysis.groupby("hour_of_day_local")["uv_index"].mean()
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(uv_by_hour.index, uv_by_hour.values, marker="o", markersize=3)
    axis.set(
        title="Mean modeled cloud-aware UV Index by local hour",
        xlabel="Karachi local hour",
        ylabel="UV Index",
        xticks=range(0, 24, 2),
    )
    _save_figure(figure, output_dir / "uv_hourly_pattern.png")

    missing_metrics = [
        column
        for column in (
            *CORE_METRICS,
            "precipitation_probability_percent",
            "weather_uv_index",
        )
        if column in frame.columns
    ]
    missing_percent = frame[missing_metrics].isna().mean().sort_values()
    correlation_metrics = [
        column
        for column in (
            "temperature_c",
            "relative_humidity_percent",
            "precipitation_mm",
            "wind_speed_kmh",
            "wind_direction_sin",
            "wind_direction_cos",
            "us_aqi",
            "pm2_5_ug_m3",
            "uv_index",
        )
        if column in frame.columns
    ]
    correlation_data = frame[correlation_metrics].copy()
    correlation_data["rain_occurred"] = rain
    correlation = correlation_data.corr(numeric_only=True)
    correlation_values = {
        column: {
            index: float(value) if pd.notna(value) else None
            for index, value in values.items()
        }
        for column, values in correlation.to_dict().items()
    }
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].barh(missing_percent.index, missing_percent.values * 100)
    axes[0].set(title="Missing values by core metric", xlabel="Missing (%)")
    image = axes[1].imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set_xticks(range(len(correlation.columns)), correlation.columns, rotation=75)
    axes[1].set_yticks(range(len(correlation.index)), correlation.index)
    axes[1].set_title("Core metric Pearson correlations")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    _save_figure(figure, output_dir / "missingness_and_correlations.png")

    behavior_metrics = (
        ("temperature_c", "Temperature (°C)"),
        ("relative_humidity_percent", "Humidity (%)"),
        ("precipitation_mm", "Precipitation (mm)"),
        ("wind_speed_kmh", "Wind speed (km/h)"),
        ("wind_direction_degrees", "Wind direction (°)"),
        ("us_aqi", "US AQI"),
        ("pm2_5_ug_m3", "PM2.5 (µg/m³)"),
        ("uv_index", "UV Index"),
    )
    figure, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
    hourly_stats: dict[str, dict[str, float | None]] = {}
    for axis, (metric, label) in zip(axes.flat, behavior_metrics):
        if metric == "wind_direction_degrees":
            radians = np.deg2rad(analysis[metric])
            grouped = pd.DataFrame(
                {
                    "sin": np.sin(radians),
                    "cos": np.cos(radians),
                    "hour": analysis["hour_of_day_local"],
                }
            ).groupby("hour")[
                ["sin", "cos"]
            ].mean()
            hourly = pd.Series(
                np.degrees(np.arctan2(grouped["sin"], grouped["cos"])) % 360,
                index=grouped.index,
            )
        else:
            hourly = analysis.groupby("hour_of_day_local")[metric].mean()
        axis.plot(hourly.index, hourly.values)
        axis.set(title=label, ylabel=label)
        hourly_stats[metric] = {
            str(int(hour)): (
                round(float(value), 6) if pd.notna(value) else None
            )
            for hour, value in hourly.items()
        }
    for axis in axes[-1, :]:
        axis.set_xlabel("Karachi local hour")
        axis.set_xticks(range(0, 24, 3))
    _save_figure(figure, output_dir / "time_of_day_behavior.png")

    rain_count = int(rain.sum(skipna=True))
    rain_valid = int(rain.notna().sum())
    report: dict[str, Any] = {
        "dataset": schema,
        "metric_summary": {
            metric: {
                "count": int(frame[metric].count()),
                "mean": float(frame[metric].mean()),
                "minimum": float(frame[metric].min()),
                "maximum": float(frame[metric].max()),
            }
            for metric in CORE_METRICS
            if metric in frame.columns and frame[metric].notna().any()
        },
        "rain_occurrence": {
            "definition": (
                f"precipitation_mm > {RAIN_OCCURRENCE_THRESHOLD_MM:g}; "
                "missing precipitation remains missing"
            ),
            "positive_hours": rain_count,
            "valid_hours": rain_valid,
            "positive_percent_of_valid_hours": (
                round(100 * rain_count / rain_valid, 4) if rain_valid else None
            ),
        },
        "hourly_mean_local": hourly_stats,
        "core_metric_correlations": correlation_values,
        "historical_forecast_information": forecast_manifest.get(
            "report_summary",
            {"status": "not_ingested", "used_as_feature": False},
        )
        if forecast_manifest
        else {"status": "not_ingested", "used_as_feature": False},
        "model_input_policy": (
            "The historical forecast probability CSV is not merged into features; "
            "its issue time and lead time are unavailable."
        ),
        "source_quality_report": {
            key: source_quality_report.get(key)
            for key in (
                "row_counts",
                "missing_hour_count",
                "duplicate_timestamps",
                "available_variables",
                "unavailable_variables",
                "quality_issues",
            )
            if source_quality_report is not None
        },
    }
    write_json_atomic(output_dir / "dataset_inspection.json", report)
    write_text_atomic(output_dir / "eda_report.md", _eda_markdown(report))
    return report


def _eda_markdown(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    rain = report["rain_occurrence"]
    lines = [
        "# AETHERIX AI Day 3 — Exploratory Data Analysis",
        "",
        f"- Processed rows: {dataset['row_count']}",
        f"- Processed columns: {dataset['column_count']}",
        f"- UTC coverage: {dataset['utc_start']} to {dataset['utc_end']}",
        f"- Target columns present in the source file: {dataset['target_column_count']} "
        "(excluded from every model input)",
        "- UTC timestamps are ordered, unique, timezone-aware, and hourly-contiguous.",
        "",
        "## Metric summaries",
        "",
        "| Metric | Count | Mean | Minimum | Maximum |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric, stats in report["metric_summary"].items():
        lines.append(
            f"| {metric} | {stats['count']} | {stats['mean']:.4f} | "
            f"{stats['minimum']:.4f} | {stats['maximum']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Rain occurrence",
            "",
            f"- Rule: `{rain['definition']}`.",
            f"- Positive hours: {rain['positive_hours']} of {rain['valid_hours']} "
            f"valid precipitation hours "
            f"({rain['positive_percent_of_valid_hours']}%).",
            "",
            "## Missingness and source metadata",
            "",
            "| Column | Missing (%) |",
            "| --- | ---: |",
        ]
    )
    for column, percent in sorted(
        dataset["missing_percent_by_column"].items(),
        key=lambda item: (-item[1], item[0]),
    )[:20]:
        lines.append(f"| {column} | {percent:.4f} |")
    lines.extend(
        [
            "",
            "Source counts and all column dtypes are in `dataset_inspection.json`.",
            "",
            "## Leakage and provenance",
            "",
            "- Every `target_*` column is excluded from model inputs.",
            "- The separately stored Historical Forecast API probability series is "
            "not used: per-record issue time and lead time are unavailable.",
            "- Weather and air-quality measurements are modeled/reanalysis data; "
            "rain occurrence is a reanalysis proxy, not station ground truth.",
            "",
            "## Charts",
            "",
            "Charts in this folder cover hourly temperature, humidity distribution, "
            "rainfall and occurrence, wind speed and direction, AQI and PM2.5, UV, "
            "missingness and correlations, and local time-of-day behavior.",
            "",
            "All statistics and charts are calculated from the local processed CSV; "
            "no observations are synthesized.",
        ]
    )
    return "\n".join(lines) + "\n"
