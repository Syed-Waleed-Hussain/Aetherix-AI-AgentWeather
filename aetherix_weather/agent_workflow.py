"""Run and audit the four weather agents in one local LangGraph StateGraph."""

from collections.abc import Callable
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Literal, TypedDict
from uuid import uuid4

import pandas as pd
from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

from .agent_drift import _feature_drift, drift_result, monitor_local
from .agent_forecast import InputError, LocalForecastSource, prediction, utc_hour
from .agent_reporting import AUDIENCES, build_report, calculate_alerts, validated_thresholds

NODES = ("forecast", "alert", "weather_report", "drift_detection")


class AgentState(TypedDict):
    request_id: str
    city: str
    mode: Literal["local", "synthetic"]
    requested_issue: str | None
    issue_utc: str | None
    horizon: int
    audience: str
    weather_aqi_inputs: dict
    model_provenance: dict
    predictions: list[dict]
    coverage_gaps: list[dict]
    forecast_fallbacks: list[dict]
    alerts: list[dict]
    report: dict
    drift: dict
    statuses: dict[str, str]
    errors: list[dict]
    decisions: list[dict]
    started_at_utc: str
    completed_at_utc: str | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DecisionStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=5)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS agent_steps ("
                "request_id TEXT NOT NULL, step INTEGER NOT NULL, agent TEXT NOT NULL, "
                "state_json TEXT NOT NULL, PRIMARY KEY(request_id, step))"
            )

    def append(self, agent: str, state: AgentState) -> None:
        payload = json.dumps(state, ensure_ascii=False, allow_nan=False)
        with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
            db.execute(
                "INSERT INTO agent_steps VALUES (?, ?, ?, ?)",
                (state["request_id"], len(state["decisions"]), agent, payload),
            )

    def replay(self, request_id: str) -> list[dict]:
        with closing(sqlite3.connect(self.path, timeout=5)) as db:
            records = db.execute(
                "SELECT state_json FROM agent_steps WHERE request_id = ? ORDER BY step",
                (request_id,),
            ).fetchall()
        return [json.loads(record[0]) for record in records]


def _synthetic_forecast(requested: str | None, horizon: int) -> dict:
    issue = utc_hour(requested or "2026-01-01T00:00:00+00:00")
    inputs = {
        "temperature_c": 37.0, "relative_humidity_percent": 43.0,
        "wind_speed_kmh": 30.0, "wind_direction_degrees": 270.0,
        "precipitation_mm": 0.0, "rain_occurred": 0.1, "uv_index": 5.0, "us_aqi": 100.0,
    }
    rows = []
    for hour in range(1, horizon + 1):
        values = inputs | (
            {"temperature_c": 42.0, "uv_index": 8.0, "us_aqi": 151.0,
             "wind_speed_kmh": 60.0, "precipitation_mm": 15.0, "rain_occurred": 0.8}
            if 5 <= hour <= 7 else {}
        )
        rows.extend(
            prediction(metric, hour, issue, value, {"name": "synthetic_fixture"})
            for metric, value in values.items()
        )
    return {
        "issue_utc": issue.isoformat(), "weather_aqi_inputs": inputs, "predictions": rows,
        "model_provenance": {"mode": "synthetic", "source": "fixed scenario", "model_inference": False},
        "coverage_gaps": [], "forecast_fallbacks": [],
    }


def build_graph(
    source: LocalForecastSource,
    store: DecisionStore,
    thresholds: dict | None = None,
    provider: Callable[[bool], str] | None = None,
):
    rules = validated_thresholds(thresholds or {})

    def forecast(state: AgentState) -> tuple[dict, str, str, str | None]:
        load = _synthetic_forecast if state["mode"] == "synthetic" else source.forecast
        patch = load(state["requested_issue"], state["horizon"])
        predictions, gaps = len(patch["predictions"]), len(patch["coverage_gaps"])
        fallbacks = len(patch["forecast_fallbacks"])
        status = "failed" if not predictions else ("partial" if gaps or fallbacks else "ok")
        detail = f"{predictions} predictions, {gaps} coverage gaps, {fallbacks} seasonal fallbacks."
        return patch, status, detail, "Forecast coverage is unavailable." if not predictions else None

    def alert(state: AgentState) -> tuple[dict, str, str, str | None]:
        if not state["predictions"]:
            return {}, "skipped", "No validated predictions were available.", None
        alerts = calculate_alerts(state["predictions"], rules, state["audience"])
        return {"alerts": alerts}, "ok", f"Found {len(alerts)} alert windows.", None

    def report(state: AgentState) -> tuple[dict, str, str, str | None]:
        summary, failure = build_report(state, provider)
        degraded = failure is not None or not state["predictions"] or state["statuses"]["alert"] == "failed"
        if state["statuses"]["alert"] == "failed":
            summary["text"] += "\nAlert evaluation failed for this run."
        return {"report": summary}, "degraded" if degraded else "ok", summary["provider"], failure

    def drift(state: AgentState) -> tuple[dict, str, str, str | None]:
        if state["mode"] == "synthetic":
            reference = pd.DataFrame({"temperature_c": [26.0] * 48})
            recent = pd.DataFrame({"temperature_c": [42.0] * 40 + [None] * 8})
            result = drift_result(_feature_drift(reference, recent), [], state["issue_utc"])
        else:
            result = monitor_local(source, state["requested_issue"])
        detail = result["status"]
        if result["retraining_review_requested"]:
            detail += "; manual retraining review requested"
        return {"drift": result}, "ok", detail, None

    def audited(name: str, node: Callable):
        def run(state: AgentState) -> dict:
            started = _now()
            try:
                patch, status, detail, error = node(state)
            except Exception as exc:
                patch, status, detail = {}, "failed", "Agent could not complete."
                error = str(exc) if isinstance(exc, InputError) else type(exc).__name__
            ended = _now()
            update = {
                **patch,
                "statuses": {**state["statuses"], name: status},
                "errors": state["errors"] + ([{"agent": name, "reason": error}] if error else []),
                "decisions": state["decisions"] + [{
                    "agent": name, "status": status, "decision": detail,
                    "started_at_utc": started, "finished_at_utc": ended,
                }],
            }
            if name == NODES[-1]:
                update["completed_at_utc"] = ended
            store.append(name, state | update)
            return update
        return run

    builder = StateGraph(AgentState)
    for name, node in zip(NODES, (forecast, alert, report, drift)):
        builder.add_node(name, audited(name, node))
    for previous, following in zip((START, *NODES), (*NODES, END)):
        builder.add_edge(previous, following)
    return builder.compile()


def run_workflow(
    root: Path, store: DecisionStore, *, city: str = "karachi", horizon: int = 24,
    issue: str | None = None, synthetic: bool = False, audience: str = "general",
    thresholds: dict | None = None, provider: Callable[[bool], str] | None = None,
) -> AgentState:
    if str(city).lower() != "karachi":
        raise InputError("The validated model bundle supports Karachi only.")
    if not 1 <= horizon <= 24:
        raise InputError("Forecast horizon must be between 1 and 24 hours.")
    if audience not in AUDIENCES:
        raise InputError("Unsupported alert audience.")
    # Store normalized arguments rather than untrusted raw input strings.
    issue = utc_hour(issue).isoformat() if issue is not None else None
    state: AgentState = {
        "request_id": str(uuid4()), "city": "karachi", "mode": "synthetic" if synthetic else "local",
        "requested_issue": issue, "issue_utc": None, "horizon": horizon, "audience": audience,
        "weather_aqi_inputs": {}, "model_provenance": {}, "predictions": [], "coverage_gaps": [],
        "forecast_fallbacks": [], "alerts": [], "report": {}, "drift": {}, "statuses": {},
        "errors": [], "decisions": [], "started_at_utc": _now(), "completed_at_utc": None,
    }
    if synthetic:
        provider = None
    graph = build_graph(LocalForecastSource(root), store, thresholds, provider)
    with tracing_context(enabled=False):
        return graph.invoke(state, config={"recursion_limit": 8})
