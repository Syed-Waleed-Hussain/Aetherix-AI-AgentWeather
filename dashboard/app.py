"""Aetherix AI weather dashboard."""

from html import escape
from importlib.util import find_spec
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import Lock

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.charts import draw_forecast
from dashboard.data import (
    METRICS, alert_rows, champion_rows, comparisons, data_status, forecast_frame,
    local_time, read_object, replay, run_history,
)

PAGES = ("Overview", "24-Hour Forecast", "Weather Alerts", "AI Weather Report",
         "Model Performance", "MLOps & Agent Monitoring")
DB = Path(os.environ.get("AETHERIX_AGENT_DB", str(ROOT / "outputs/agents/decisions.sqlite3")))
ARTIFACTS = ROOT / "artifacts"
AUDIENCES = {"General public": "general", "Outdoor workers": "outdoor_workers",
             "Children and sensitive groups": "sensitive_groups"}


@st.cache_resource(max_entries=1)
def runtime(_root: Path, version: tuple):
    from aetherix_weather.agent_forecast import LocalForecastSource

    return LocalForecastSource(_root), Lock()


@st.cache_data(max_entries=24, ttl=300, show_spinner=False)
def evaluation_table(root: Path, family: str, hour: int):
    return comparisons(root, family, hour)


def status_table(state):
    return pd.DataFrame([
        {"Agent": key.replace("_", " ").title(), "Status": value}
        for key, value in state.get("statuses", {}).items()
    ])


def saved_state(history: list[dict], selected: str | None) -> dict:
    if selected:
        return replay(DB, selected)
    configured = os.environ.get("AETHERIX_SNAPSHOT_PATH")
    if configured:
        return read_object(Path(configured))
    newest = next((row for row in history if row["mode"] == "local"
                   and row["forecast_status"] in ("ok", "partial") and row["completed_at_utc"]), None)
    return replay(DB, newest["request_id"]) if newest else read_object(ROOT / "outputs/agents/local.json")


def controls(history):
    st.sidebar.markdown('<div class="wordmark">AETHERIX AI</div>', unsafe_allow_html=True)
    page = st.sidebar.radio("Workspace", PAGES)
    runs = {row["request_id"]: row for row in history}
    chosen = st.sidebar.selectbox(
        "Saved run", [None, *runs],
        format_func=lambda key: (
            f"{local_time(runs[key]['issue_utc'])} · {key[:8]}" if key else "Latest historical run"
        ),
    )
    paths = [ROOT / entry for entry in (
        "data/processed/features.csv", "artifacts/day3_5/selected_models.json",
        "artifacts/day3_5/selected_models.joblib", "artifacts/day5c/champion_selection.json",
        "artifacts/day3_5/test_predictions.csv",
    )]
    local_enabled = all(p.is_file() for p in paths[:3]) and find_spec("langgraph") is not None
    with st.sidebar.expander("Run local forecast"):
        audience = st.selectbox("Run audience", list(AUDIENCES))
        groq = st.checkbox("Use free-tier Groq layout", disabled=not local_enabled)
        if st.button("Run 24-hour forecast", disabled=not local_enabled):
            try:
                from aetherix_weather.agent_reporting import groq_report_layout
                from aetherix_weather.agent_workflow import DecisionStore, run_workflow

                if groq:
                    # Reading root-level Streamlit secrets makes them available
                    # to the existing environment-based Groq integration.
                    try:
                        st.secrets.get("GROQ_MODEL")
                    except (FileNotFoundError, ValueError):
                        pass
                version = tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.is_file() else None for p in paths)
                source, lock = runtime(ROOT, version)
                if not lock.acquire(blocking=False):
                    st.info("A forecast is already running. Try again when it finishes.")
                else:
                    try:
                        with st.spinner("Running four weather agents"):
                            result = run_workflow(
                                ROOT, DecisionStore(DB), source=source, audience=AUDIENCES[audience],
                                provider=groq_report_layout if groq else None,
                            )
                    finally:
                        lock.release()
                    st.session_state["generated"] = result["request_id"]
                    st.rerun()
            except Exception:
                st.error("Local execution could not complete. Check the local artifacts and runtime.")
        if not local_enabled:
            st.caption("Local model files and LangGraph are required for inference.")
    return page, chosen or st.session_state.get("generated")


def overview(state):
    st.title("Karachi weather intelligence")
    st.caption("Asia/Karachi · Forecast issue " + local_time(state.get("issue_utc")))
    st.subheader("Latest available observation")
    st.caption("Saved issue-hour weather and modeled AQI from the workflow input.")
    inputs = state.get("weather_aqi_inputs", {})
    cols = st.columns(4)
    for column, key in zip(cols, ("temperature_c", "relative_humidity_percent", "wind_speed_kmh", "us_aqi")):
        label, unit = METRICS[key]
        value = inputs.get(key)
        column.metric(label, f"{value:.1f} {unit}" if isinstance(value, (int, float)) else "Unavailable")
    forecast = forecast_frame(state, "temperature_c")
    st.subheader("Temperature outlook")
    if forecast.empty:
        st.info("No temperature forecast is available for this run.")
    else:
        st.caption(f"{forecast['value'].min():.1f}–{forecast['value'].max():.1f} °C across {len(forecast)} hours")
        draw_forecast(forecast, "temperature_c")
    for alert in state.get("alerts", [])[:3]:
        st.warning(f"{alert['explanation']} {local_time(alert['start_utc'])}")
    st.subheader("Agent execution")
    table = status_table(state)
    st.dataframe(table, hide_index=True) if not table.empty else st.caption("No agent execution recorded.")


def forecast_page(state):
    st.title("24-hour forecast")
    metric = st.selectbox("Metric", list(METRICS), format_func=lambda key: METRICS[key][0],
                          key="forecast_metric")
    frame = forecast_frame(state, metric)
    st.caption(f"{len(frame)} of {state.get('horizon', 24)} forecast hours · {METRICS[metric][1]} · Karachi time")
    if frame.empty:
        st.info("No values are available for this metric in the selected run.")
        return
    draw_forecast(frame, metric)
    st.dataframe(frame, hide_index=True)
    if state.get("coverage_gaps"):
        with st.expander("Forecast coverage gaps"):
            st.dataframe(state["coverage_gaps"], hide_index=True)


def alerts_page(state):
    st.title("Weather alerts")
    audience = st.selectbox("Audience", list(AUDIENCES))
    alerts = alert_rows(state, AUDIENCES[audience])
    if not alerts:
        st.info("No alerts are recorded for the available forecasts." if state.get("predictions")
                else "Alert coverage is unavailable for this run.")
    for alert in alerts:
        with st.container(border=True):
            st.subheader(f"{alert['severity'].title()} · {METRICS[alert['metric']][0]}")
            st.caption(f"{alert['display_start']} → {alert['display_end']} (exclusive)")
            st.write(f"Peak {alert['peak']:g} {alert['unit']} · Threshold {alert['threshold']:g} {alert['unit']}")
            st.write(alert["precaution"])
            st.caption("Prediction sources · " + (alert["sources"] or "Unavailable"))


def report_page(state):
    st.title("AI weather report")
    report = state.get("report", {})
    names = {"template": "Deterministic template", "template_fallback": "Deterministic fallback",
             "groq_layout": "Groq layout with forecast-sourced facts"}
    st.caption(names.get(report.get("provider"), "Report provider unavailable"))
    text = report.get("text")
    if text:
        st.html('<div class="brief">' + "<br>".join(escape(line) for line in text.splitlines()) + "</div>")
        st.caption(f"{report.get('alert_count', 0)} forecast alert windows · Report timestamps are UTC")
    else:
        st.info("A saved report is required.")


def performance_page(state):
    st.title("Model performance")
    family = st.selectbox("Matched test cohort", ["SARIMAX", "LSTM"])
    hour = st.slider("Forecast horizon", 1, 24, 1)
    table = evaluation_table(ARTIFACTS, family, hour)
    st.caption("Temperature · classical ML versus " + family + " within this experiment's held-out cohort")
    if table.empty:
        st.info("No verified common-cohort results are available for this selection.")
    else:
        st.dataframe(table, hide_index=True)
    st.subheader("Validation champions")
    metric = st.selectbox("Champion target", list(METRICS), format_func=lambda key: METRICS[key][0])
    table = champion_rows(ARTIFACTS, metric)
    if table.empty:
        st.caption("The local champion manifest is unavailable.")
    else:
        st.dataframe(table, hide_index=True)
    lstm = read_object(ARTIFACTS / "day4b/lstm_metadata.json")
    loss = lstm.get("training", {}).get("validation_loss_history")
    if loss:
        with st.expander("LSTM training history"):
            st.line_chart(pd.DataFrame({"Epoch": range(1, len(loss) + 1), "Validation scaled MSE": loss}),
                          x="Epoch", y="Validation scaled MSE")
    with st.expander("Data lineage and provenance"):
        st.json(state.get("model_provenance", {}))


def monitoring_page(state, history):
    st.title("MLOps & agent monitoring")
    latest = next((row for row in history if row["mode"] == "local"
                   and row["forecast_status"] == "ok" and row["completed_at_utc"]), None)
    st.caption("Last successful local forecast · " + local_time(latest["completed_at_utc"] if latest else None))
    if history:
        st.dataframe(history, hide_index=True)
    st.subheader("Decision trail")
    if state.get("decisions"):
        st.dataframe(state["decisions"], hide_index=True)
    if state.get("errors"):
        st.warning("This run contains one or more agent errors. See the decision statuses.")
    st.subheader("Drift monitor")
    drift = state.get("drift", {})
    st.write(drift.get("status", "Unavailable").replace("_", " ").title())
    if drift:
        st.write(drift.get("recommended_action", "Review available evidence."))
        st.caption(f"{len(drift.get('affected_features', []))} feature flags · "
                   f"{len(drift.get('affected_models', []))} target/horizon error flags")
        with st.expander("Drift measurements and thresholds"):
            st.json(drift.get("thresholds", {}))
            st.dataframe(drift.get("features", []), hide_index=True)
            st.dataframe(drift.get("forecast_errors", []), hide_index=True)
    st.subheader("Hopsworks")
    st.markdown("**Feature Store** · 24 rows previously verified. Full sync pending.  \n"
                "**Model Registry** · Local package prepared. Cloud registration pending.")
    sdk = ROOT / ".venv-hopsworks" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if st.button("Check connectivity", disabled=not sdk.is_file()):
        try:
            result = subprocess.run(
                [str(sdk), "-m", "scripts.hopsworks_connection_check"], cwd=ROOT,
                capture_output=True, timeout=12, check=False,
            )
            st.session_state["hops_status"] = "Available" if result.returncode == 0 else "Check failed"
        except (OSError, subprocess.TimeoutExpired):
            st.session_state["hops_status"] = "Unavailable or timed out"
    st.caption("Connectivity · " + st.session_state.get("hops_status", "Unverified in this session"))


def main():
    st.set_page_config(page_title="Aetherix AI | Karachi Weather", layout="wide")
    st.html(Path(__file__).with_name("style.css"))
    try:
        history = run_history(DB)
    except (sqlite3.Error, ValueError):
        history = []
        st.warning("The local decision database could not be read.")
    page, selected = controls(history)
    try:
        state = saved_state(history, selected)
    except (sqlite3.Error, ValueError):
        state = {}
        st.warning("The selected run could not be read.")
    st.markdown(f'<div class="statusline">{data_status(state)}</div>', unsafe_allow_html=True)
    if page == PAGES[-1]:
        monitoring_page(state, history)
    else:
        pages = (overview, forecast_page, alerts_page, report_page, performance_page)
        pages[PAGES.index(page)](state)


if __name__ == "__main__":
    main()
