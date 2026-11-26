"""Read persisted graph outputs without importing model or cloud SDKs."""

from contextlib import closing
import json
from math import isfinite
from pathlib import Path
import sqlite3

import pandas as pd

METRICS = {
    "temperature_c": ("Temperature", "°C"),
    "relative_humidity_percent": ("Humidity", "%"),
    "wind_speed_kmh": ("Wind speed", "km/h"),
    "wind_direction_degrees": ("Wind direction", "°"),
    "precipitation_mm": ("Precipitation", "mm"),
    "rain_occurred": ("Rain probability", "%"),
    "uv_index": ("UV index", "UV index"),
    "us_aqi": ("Air quality", "US AQI"),
}
TIMEZONE = "Asia/Karachi"


def read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (ValueError, OSError):
        return {}


def _db_query(path: Path, query: str, parameters: tuple = ()) -> list:
    if not path.is_file():
        return []
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3)) as db:
        return db.execute(query, parameters).fetchall()


def run_history(path: Path) -> list[dict]:
    rows = _db_query(
        path,
        """SELECT request_id, json_extract(state_json, '$.mode'),
                  json_extract(state_json, '$.issue_utc'),
                  json_extract(state_json, '$.completed_at_utc'),
                  json_extract(state_json, '$.statuses.forecast')
           FROM agent_steps WHERE rowid IN (SELECT MAX(rowid) FROM agent_steps GROUP BY request_id)
           ORDER BY rowid DESC LIMIT 50""",
    )
    keys = ("request_id", "mode", "issue_utc", "completed_at_utc", "forecast_status")
    return [dict(zip(keys, row)) for row in rows]


def replay(path: Path, request_id: str) -> dict:
    rows = _db_query(
        path, "SELECT state_json FROM agent_steps WHERE request_id = ? ORDER BY step DESC LIMIT 1",
        (request_id,),
    )
    return json.loads(rows[0][0]) if rows else {}


def data_status(state: dict) -> str:
    if state.get("mode") == "synthetic":
        return "SYNTHETIC · demonstration fixture"
    if state.get("mode") == "local" and state.get("issue_utc") and state.get("predictions"):
        return "HISTORICAL · archived model forecast"
    return "UNAVAILABLE · forecast data missing"


def local_time(value: str | None) -> str:
    if not value:
        return "Unavailable"
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    return timestamp.tz_convert(TIMEZONE).strftime("%d %b %Y, %H:%M PKT") if pd.notna(timestamp) else "Unavailable"


def forecast_frame(state: dict, metric: str) -> pd.DataFrame:
    columns = ["horizon", "valid_at_pkt", "value", "model", "artifact", "schema"]
    if metric not in METRICS:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in state.get("predictions", []):
        value = row.get("value")
        if row.get("metric") != metric or type(value) not in (int, float) or not isfinite(value):
            continue
        stamp = pd.to_datetime(row.get("valid_at_utc"), utc=True, errors="coerce")
        if pd.isna(stamp):
            continue
        model = row.get("model", {})
        rows.append({
            "horizon": row["horizon"], "valid_at_pkt": stamp.tz_convert(TIMEZONE).tz_localize(None),
            "value": value * 100 if metric == "rain_occurred" else value,
            "model": model.get("name", "Unavailable"), "artifact": model.get("artifact", "Unavailable"),
            "schema": model.get("feature_schema_sha256", ""),
        })
    return pd.DataFrame(rows, columns=columns).sort_values("horizon")


def alert_rows(state: dict, audience: str) -> list[dict]:
    from aetherix_weather.agent_reporting import AUDIENCES, PRECAUTIONS

    return [
        {
            **row,
            "display_start": local_time(row.get("start_utc")),
            "display_end": local_time(row.get("end_utc_exclusive")),
            "precaution": PRECAUTIONS[row["type"]][AUDIENCES.index(audience)],
            "sources": ", ".join(sorted({item["name"] for item in row.get("models", [])})),
        }
        for row in state.get("alerts", []) if row.get("type") in PRECAUTIONS
    ]


def _table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, ValueError):
        return pd.DataFrame()


def comparisons(root: Path, family: str, horizon: int) -> pd.DataFrame:
    """Keep each trainer's explicitly matched test cohort in its own view."""
    columns = ["model", "n", "mae", "rmse", "r2"]
    if family == "SARIMAX":
        return matched_sarimax(root, horizon)
    table = _table(root / "day4b/model_comparison.csv")
    if not set([*columns, "horizon_hours", "same_sample"]).issubset(table):
        return pd.DataFrame(columns=columns)
    mask = table["horizon_hours"].eq(horizon) & table["same_sample"].astype(str).str.lower().eq("true")
    selected = table.loc[mask]
    if selected.empty or selected["n"].nunique(dropna=False) != 1:
        return pd.DataFrame(columns=columns)
    if any(
        key not in selected or selected[key].nunique(dropna=False) != 1
        or not isinstance(selected[key].iloc[0], str) or len(selected[key].iloc[0]) != 64
        for key in ("issue_timestamp_sha256", "target_timestamp_sha256")
    ):
        return pd.DataFrame(columns=columns)
    return selected[columns].reset_index(drop=True)


def matched_sarimax(root: Path, horizon: int) -> pd.DataFrame:
    """Score both models on the same verified issue/target pairs and observations."""
    columns = ["model", "n", "mae", "rmse", "r2"]
    empty = pd.DataFrame(columns=columns)
    keys = ["issue_time_utc", "target_time_utc"]

    def collect(path, hour_column, score_columns, require_test=False):
        fields = [*keys, "metric", hour_column, *score_columns]
        if require_test:
            fields.append("split")
        frames = []
        with pd.read_csv(path, usecols=fields, chunksize=40_000) as reader:
            for chunk in reader:
                keep = chunk["metric"].eq("temperature_c") & chunk[hour_column].eq(horizon)
                if require_test:
                    keep &= chunk["split"].eq("test")
                frames.append(chunk.loc[keep, [*keys, *score_columns]])
        return pd.concat(frames, ignore_index=True)

    try:
        classical = collect(
            root / "day3_5/test_predictions.csv", "forecast_horizon_hours",
            ["target_value", "selected_model", "prediction"],
        )
        statistical = collect(
            root / "day4a/temperature/statistical_forecasts.csv", "horizon_hours",
            ["actual", "model", "forecast"], require_test=True,
        )
        for table in (classical, statistical):
            for key in keys:
                table[key] = pd.to_datetime(table[key], utc=True, format="mixed")
            if table[keys].isna().any().any() or table.duplicated(keys).any():
                return empty
        joined = classical.merge(statistical, on=keys, validate="one_to_one")
    except (OSError, ValueError):
        return empty
    joined = joined.dropna(subset=["target_value", "actual", "prediction", "forecast"])
    if (
        len(joined) < 2
        or not joined[["target_value", "actual", "prediction", "forecast"]].map(isfinite).all().all()
        or joined["selected_model"].nunique() != 1 or joined["model"].nunique() != 1
        or (joined["target_value"] - joined["actual"]).abs().gt(1e-9).any()
    ):
        return empty
    truth = joined["actual"]
    total = (truth - truth.mean()).pow(2).sum()
    result = []
    for label, field in (("selected_model", "prediction"), ("model", "forecast")):
        error = joined[field] - truth
        squared = error.pow(2)
        result.append([joined[label].iloc[0], len(joined), error.abs().mean(),
                       squared.mean() ** 0.5, 1 - squared.sum() / total if total else None])
    return pd.DataFrame(result, columns=columns)


def champion_rows(root: Path, metric: str) -> pd.DataFrame:
    document = read_object(root / "day5c/champion_selection.json")
    if document.get("selection_split") != "validation":
        return pd.DataFrame()
    return pd.DataFrame([
        {"Hour": model["horizon_hours"], "Champion": model["model_name"],
         "Validation metric": model["metric"], "Score": model["validation_score"],
         "Decision": row["reason"]}
        for row in document.get("decisions", [])
        if (model := row.get("champion", {})).get("target") == metric
    ])
