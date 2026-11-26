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
DEMO = Path(__file__).with_name("demo_snapshot.json")
LOCAL_INPUTS = tuple(ROOT / entry for entry in (
    "data/processed/features.csv", "artifacts/day3_5/selected_models.json",
    "artifacts/day3_5/selected_models.joblib", "artifacts/day5c/champion_selection.json",
    "artifacts/day3_5/test_predictions.csv",
))
AUDIENCES = {"General public": "general", "Outdoor workers": "outdoor_workers",
             "Children and sensitive groups": "sensitive_groups"}


@st.cache_resource(max_entries=1)
def runtime(_root: Path, version: tuple):
    from aetherix_weather.agent_forecast import LocalForecastSource

    return LocalForecastSource(_root), Lock()


@st.cache_data(max_entries=24, ttl=300, show_spinner=False)
def evaluation_table(root: Path, family: str, hour: int):
    return comparisons(root, family, hour)


@st.cache_data(show_spinner=False)
def bundled_demo():
    return read_object(DEMO)


def status_table(state):
    return pd.DataFrame([
        {"Agent": key.replace("_", " ").title(), "Status": value}
        for key, value in state.get("statuses", {}).items()
    ])


def saved_state(history: list[dict], selected: str | None, snapshot: bool) -> dict:
    if snapshot:
        configured = os.environ.get("AETHERIX_SNAPSHOT_PATH")
        return read_object(Path(configured)) if configured else bundled_demo()
    if selected:
        return replay(DB, selected)
    configured = os.environ.get("AETHERIX_SNAPSHOT_PATH")
    if configured:
        return read_object(Path(configured))
    newest = next((row for row in history if row["mode"] == "local"
                   and row["forecast_status"] in ("ok", "partial") and row["completed_at_utc"]), None)
    return replay(DB, newest["request_id"]) if newest else read_object(ROOT / "outputs/agents/local.json")


def controls(history, snapshot):
    st.sidebar.markdown('<div class="wordmark">AETHERIX AI</div>', unsafe_allow_html=True)
    page = st.sidebar.radio("Workspace", PAGES)
    if snapshot:
        st.sidebar.caption("Read-only demonstration. Local forecast runs and saved-run replay are disabled.")
        st.sidebar.button("Run 24-hour forecast", disabled=True)
        return page, None
    runs = {row["request_id"]: row for row in history}
    chosen = st.sidebar.selectbox(
        "Saved run", [None, *runs],
        format_func=lambda key: (
            f"{local_time(runs[key]['issue_utc'])} · {key[:8]}" if key else "Latest historical run"
        ),
    )
    local_enabled = all(p.is_file() for p in LOCAL_INPUTS[:3]) and find_spec("langgraph") is not None
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
                version = tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.is_file() else None
                                for p in LOCAL_INPUTS)
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
    st.subheader("Saved issue-hour inputs")
    st.caption("Weather and modeled AQI used by this forecast.")
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
    if table.empty:
        st.caption("No agent execution recorded.")
    else:
        st.dataframe(table, hide_index=True)


def forecast_page(state):
    st.title("24-hour forecast")
    metric = st.selectbox("Metric", list(METRICS), format_func=lambda key: METRICS[key][0],
                          key="forecast_metric")
    frame = forecast_frame(state, metric)
    st.caption("Forecast issue · " + local_time(state.get("issue_utc")))
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


def performance_page(state, snapshot):
    st.title("Model performance")
    family = st.selectbox("Matched test cohort", ["SARIMAX", "LSTM"])
    hour = st.slider("Forecast horizon", 1, 24, 1)
    table = (pd.DataFrame(state.get("evaluation", {}).get(family, {}).get(str(hour), []))
             if snapshot else evaluation_table(ARTIFACTS, family, hour))
    st.caption(f"Temperature · {family} matched holdout · MAE and RMSE in °C")
    if table.empty:
        st.info("No verified common-cohort results are available for this selection.")
    else:
        st.dataframe(table, hide_index=True)
    if snapshot:
        st.caption("Saved matched-cohort scores. Full validation and training artifacts are available in the local workspace.")
        with st.expander("Data lineage and provenance"):
            st.json(state.get("model_provenance", {}))
        return
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


def monitoring_page(state, history, snapshot):
    st.title("MLOps & agent monitoring")
    latest = next((row for row in history if row["mode"] == "local"
                   and row["forecast_status"] == "ok" and row["completed_at_utc"]), None)
    if snapshot:
        st.caption("Saved forecast execution · " + local_time(state.get("completed_at_utc")))
    else:
        st.caption("Last successful local forecast · " + local_time(latest["completed_at_utc"] if latest else None))
    if history:
        st.dataframe(history, hide_index=True)
    st.subheader("Decision trail")
    if state.get("decisions"):
        st.dataframe(state["decisions"], hide_index=True)
    elif snapshot:
        st.dataframe(status_table(state), hide_index=True)
        st.caption("Full decision history is available in the local SQLite workspace.")
    if state.get("errors"):
        st.warning("This run contains one or more agent errors. See the decision statuses.")
    st.subheader("Drift monitor")
    drift = state.get("drift", {})
    st.write(drift.get("status", "Unavailable").replace("_", " ").title())
    if drift:
        st.write(drift.get("recommended_action", "Review available evidence."))
        feature_count = drift.get("feature_flag_count", len(drift.get("affected_features", [])))
        error_count = drift.get("forecast_error_flag_count", len(drift.get("affected_models", [])))
        st.caption(f"{feature_count} feature flags · {error_count} target/horizon error flags")
        st.caption("Drift evaluated as of " + local_time(drift.get("evaluated_as_of_utc")))
        if not snapshot:
            with st.expander("Drift measurements and thresholds"):
                st.json(drift.get("thresholds", {}))
                st.dataframe(drift.get("features", []), hide_index=True)
                st.dataframe(drift.get("forecast_errors", []), hide_index=True)
    st.subheader("Hopsworks")
    st.markdown("**Feature Store** · 24 rows previously verified. Full sync pending.  \n"
                "**Model Registry** · Local package prepared. Cloud registration pending.")
    if snapshot:
        st.caption("Cloud connectivity checks are disabled in read-only mode.")
        return
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
    local_state = DB.is_file() or (ROOT / "outputs/agents/local.json").is_file() or LOCAL_INPUTS[0].is_file()
    snapshot = os.environ.get("AETHERIX_MODE") == "snapshot" or not local_state
    history = []
    if not snapshot:
        try:
            history = run_history(DB)
        except (sqlite3.Error, ValueError):
            st.warning("The local decision database could not be read.")
    page, selected = controls(history, snapshot)
    try:
        state = saved_state(history, selected, snapshot)
    except (sqlite3.Error, ValueError):
        state = {}
        st.warning("The selected run could not be read.")
    st.markdown(f'<div class="statusline">{data_status(state)}</div>', unsafe_allow_html=True)
    st.caption("Live observation · UNAVAILABLE")
    if snapshot:
        st.caption("Read-only snapshot · Karachi historical demonstration")
        if not state:
            st.warning("The forecast snapshot is missing or could not be read.")
    if page == PAGES[-1]:
        monitoring_page(state, history, snapshot)
    elif page == "Model Performance":
        performance_page(state, snapshot)
    else:
        pages = (overview, forecast_page, alerts_page, report_page)
        pages[PAGES.index(page)](state)


if __name__ == "__main__":
    main()
