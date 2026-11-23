"""Threshold alerts and a grounded report built entirely from graph state."""

from collections import defaultdict
from collections.abc import Callable
import math
import os

import pandas as pd
import requests

DEFAULT_THRESHOLDS = {
    "us_aqi": {"elevated": 101, "high": 151},
    "temperature_c": {"elevated": 38, "high": 42},
    "precipitation_mm": {"elevated": 7.5, "high": 15},
    "wind_speed_kmh": {"elevated": 40, "high": 60},
    "uv_index": {"elevated": 6, "high": 8},
}
AUDIENCES = ("general", "outdoor_workers", "sensitive_groups")
ALERT_TYPES = {
    "us_aqi": "high_aqi", "temperature_c": "extreme_heat",
    "precipitation_mm": "heavy_rain", "wind_speed_kmh": "strong_wind", "uv_index": "high_uv",
}
LABELS = {
    "temperature_c": "Temperature", "relative_humidity_percent": "Humidity",
    "wind_speed_kmh": "Wind speed", "wind_direction_degrees": "Wind direction",
    "precipitation_mm": "Hourly precipitation", "rain_occurred": "Rain probability",
    "uv_index": "UV index", "us_aqi": "Air quality",
}
PRECAUTIONS = {
    "high_aqi": (
        "Reduce prolonged strenuous outdoor activity.",
        "Move strenuous tasks indoors where practical and plan indoor breaks.",
        "Choose lighter outdoor activities or spend more time indoors.",
    ),
    "extreme_heat": (
        "Drink water and spend time in shade or a cooler place.",
        "Schedule strenuous tasks for cooler hours and take water and shade breaks.",
        "Avoid long sessions in direct sun and take cooling breaks.",
    ),
    "heavy_rain": (
        "Use another route if you encounter a flooded road.",
        "Pause work in exposed or flood-prone locations.",
        "Stay away from floodwater, storm drains, and fast-flowing water.",
    ),
    "strong_wind": (
        "Stay clear of trees, loose objects, and power lines.",
        "Secure loose materials and review outdoor work conditions before continuing.",
        "Move activities away from trees and unsecured structures.",
    ),
    "high_uv": (
        "Seek shade and use protective clothing and sunscreen.",
        "Use shade during breaks and wear protective clothing.",
        "Plan shaded activities and use sun protection.",
    ),
}
ALERT_LABELS = {
    "high_aqi": "air quality", "extreme_heat": "heat", "heavy_rain": "rainfall",
    "strong_wind": "wind", "high_uv": "UV",
}


def validated_thresholds(overrides: dict) -> dict:
    thresholds = {metric: levels.copy() for metric, levels in DEFAULT_THRESHOLDS.items()}
    for metric, levels in overrides.items():
        if metric not in thresholds or not isinstance(levels, dict) or set(levels) != {"elevated", "high"}:
            raise ValueError("Each override requires a supported alert metric and both severity levels.")
        if not all(type(value) in (int, float) and math.isfinite(value) for value in levels.values()):
            raise ValueError("Alert thresholds must be finite numbers.")
        if levels["elevated"] >= levels["high"]:
            raise ValueError("Elevated severity must have a lower threshold than high severity.")
        thresholds[metric] = levels
    return thresholds


def calculate_alerts(predictions: list[dict], thresholds: dict, audience: str) -> list[dict]:
    alerts = []
    audience_index = AUDIENCES.index(audience)
    by_metric = defaultdict(list)
    for row in predictions:
        if row["metric"] in thresholds and math.isfinite(row["value"]):
            by_metric[row["metric"]].append(row)
    for metric, rows in by_metric.items():
        levels = thresholds[metric]
        kind = ALERT_TYPES[metric]
        for row in sorted(rows, key=lambda item: item["horizon"]):
            value = row["value"]
            if value < levels["elevated"]:
                continue
            severity = "high" if value >= levels["high"] else "elevated"
            start = row["valid_at_utc"]
            end = (pd.Timestamp(start) + pd.Timedelta(1, unit="h")).isoformat()
            if (
                alerts and alerts[-1]["metric"] == metric and alerts[-1]["severity"] == severity
                and alerts[-1]["end_utc_exclusive"] == start
            ):
                alert = alerts[-1]
                alert["end_utc_exclusive"] = end
                alert["peak"] = max(alert["peak"], value)
                if row["model"] not in alert["models"]:
                    alert["models"].append(row["model"])
                continue
            threshold = levels[severity]
            alerts.append({
                "type": kind, "severity": severity, "start_utc": start,
                "end_utc_exclusive": end, "metric": metric, "threshold": threshold,
                "unit": row["unit"], "peak": value, "audience": audience,
                "explanation": f"{LABELS[metric]} meets the {threshold:g} {row['unit']} alert threshold.",
                "precaution": PRECAUTIONS[kind][audience_index], "models": [row["model"]],
            })
    return sorted(alerts, key=lambda alert: (alert["start_utc"], alert["type"]))


class ProviderError(Exception):
    """Fixed category for a provider error, safe for the decision log."""


def groq_report_layout(has_alerts: bool) -> str:
    key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("GROQ_MODEL")
    if not key or not model:
        raise ProviderError("groq_configuration_missing")
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model, "max_completion_tokens": 12,
                "messages": [{"role": "user", "content": (
                    "Choose the report section order. Reply exactly alerts_first or conditions_first. "
                    f"Active forecast alerts exist: {has_alerts}."
                )}],
            },
            timeout=(3, 5),
            allow_redirects=False,
        )
        if response.status_code == 429:
            raise ProviderError("groq_rate_limited")
        if response.status_code != 200:
            raise ProviderError("groq_http_error")
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.RequestException:
        raise ProviderError("groq_network_error") from None
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        raise ProviderError("groq_invalid_response") from None


def build_report(
    state: dict, provider: Callable[[bool], str] | None = None
) -> tuple[dict, str | None]:
    forecasts, alerts = state["predictions"], state["alerts"]
    layout = "alerts_first" if alerts else "conditions_first"
    provider_status, failure = "template", None
    if provider is not None and forecasts:
        try:
            chosen = provider(bool(alerts))
            if chosen not in ("alerts_first", "conditions_first"):
                raise ProviderError("groq_invalid_response")
            layout, provider_status = chosen, "groq_layout"
        except Exception as exc:
            allowed = {"groq_configuration_missing", "groq_rate_limited", "groq_http_error",
                       "groq_network_error", "groq_invalid_response"}
            failure = str(exc) if isinstance(exc, ProviderError) and str(exc) in allowed else "groq_failure"
            provider_status = "template_fallback"

    mode = ("Synthetic demonstration" if state["mode"] == "synthetic"
            else "Forecast from a local historical snapshot")
    header = f"{state['city'].title()} — {mode}"
    if not forecasts:
        return {"text": f"{header}\nForecast unavailable. Alert coverage is incomplete.",
                "provider": provider_status, "metrics": [], "alert_count": len(alerts)}, failure
    grouped = defaultdict(list)
    for row in forecasts:
        grouped[row["metric"]].append(row)
    metrics, conditions = [], []
    for name, rows in grouped.items():
        low = min(rows, key=lambda row: row["value"])
        high = max(rows, key=lambda row: row["value"])
        if name == "wind_direction_degrees":
            current = rows[0]
            metrics.append({"metric": name, "first": current})
            conditions.append(f"{LABELS[name]} starts at {current['value']:.2f}° at {current['valid_at_utc']}.")
            continue
        metrics.append({"metric": name, "minimum": low["value"], "maximum": high["value"],
                        "peak_at_utc": high["valid_at_utc"], "unit": high["unit"]})
        conditions.append(
            f"{LABELS[name]} {low['value']:.2f}–{high['value']:.2f} {high['unit']}. "
            f"Peak at {high['valid_at_utc']}."
        )
    gaps = len(state["coverage_gaps"])
    highlights = [
        f"{item['severity'].title()} {ALERT_LABELS[item['type']]} alert "
        f"from {item['start_utc']} until {item['end_utc_exclusive']} UTC. {item['precaution']}"
        for item in sorted(alerts, key=lambda item: (item["severity"] != "high", item["start_utc"]))
    ] or ["No alert threshold was reached by the available predictions."]
    if gaps:
        highlights.append(f"{gaps} forecast values are unavailable. Alerts cover available predictions only.")
    parts = [highlights, conditions] if layout == "alerts_first" else [conditions, highlights]
    report = {
        "text": "\n".join([header, f"Issue {state['issue_utc']} | {state['horizon']} hours", *parts[0], *parts[1]]),
        "metrics": metrics, "alert_count": len(alerts), "provider": provider_status,
    }
    return report, failure
