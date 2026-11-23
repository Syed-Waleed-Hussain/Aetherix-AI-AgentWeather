"""Saved dashboard data and non-writing workflow contracts."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest
import requests
import yaml

from aetherix_weather.agent_workflow import DecisionStore, run_workflow
from dashboard.data import METRICS, alert_rows, comparisons, data_status, forecast_frame, replay, run_history
from scripts import mlops_check

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_reads_real_graph_state_and_every_metric(tmp_path):
    path = tmp_path / "decisions.sqlite3"
    assert run_history(path) == []
    assert not path.exists()
    state = run_workflow(tmp_path, DecisionStore(path), synthetic=True)
    assert run_history(path)[0]["request_id"] == state["request_id"]
    assert replay(path, state["request_id"]) == state
    assert replay(path, "' OR 1=1 --") == {}
    assert data_status(state) == "Synthetic demonstration"
    assert data_status({**state, "mode": "local"}) == "Historical forecast"
    assert data_status({}) == "Forecast unavailable"
    assert state["drift"]["status"] == "input_shift"
    for metric in METRICS:
        frame = forecast_frame(state, metric)
        assert len(frame) == 24
        assert list(frame["horizon"]) == list(range(1, 25))
    assert forecast_frame(state, "rain_occurred")["value"].max() == 80
    assert forecast_frame({}, "temperature_c").empty
    alerts = alert_rows(state, "outdoor_workers")
    assert len(alerts) == 5
    assert all("PKT" in alert["display_start"] and alert["sources"] for alert in alerts)
    assert alerts != alert_rows(state, "sensitive_groups")


def test_comparisons_reject_mismatched_lstm_cohorts(tmp_path):
    path = tmp_path / "day4b/model_comparison.csv"
    path.parent.mkdir()
    row = {
        "model": "lstm", "n": 20, "mae": 1, "rmse": 1.3, "r2": 0.8,
        "horizon_hours": 1, "same_sample": True, "issue_timestamp_sha256": "a" * 64,
        "target_timestamp_sha256": "b" * 64,
    }
    pd.DataFrame([row, {**row, "model": "ridge", "target_timestamp_sha256": "c" * 64}]).to_csv(path, index=False)
    assert comparisons(tmp_path, "LSTM", 1).empty
    pd.DataFrame([row, {**row, "model": "ridge"}]).to_csv(path, index=False)
    assert len(comparisons(tmp_path, "LSTM", 1)) == 2


def test_sarimax_comparison_scores_only_exact_timestamp_matches(tmp_path):
    classical = tmp_path / "day3_5/test_predictions.csv"
    statistical = tmp_path / "day4a/temperature/statistical_forecasts.csv"
    for file in (classical, statistical):
        file.parent.mkdir(parents=True)
    stars = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")

    def rows(offsets, statistical_run=False):
        for index in offsets:
            base = {"issue_time_utc": stars[index].isoformat(), "target_time_utc": stars[index + 1].isoformat(),
                    "metric": "temperature_c"}
            if statistical_run:
                yield base | {"horizon_hours": 1, "split": "test", "actual": index,
                              "model": "sarimax", "forecast": index}
            else:
                yield base | {"forecast_horizon_hours": 1, "target_value": index,
                              "selected_model": "ridge", "prediction": index + 1}

    pd.DataFrame(rows((0, 1, 2))).to_csv(classical, index=False)
    pd.DataFrame(rows((1, 2, 3), True)).to_csv(statistical, index=False)
    result = comparisons(tmp_path, "SARIMAX", 1).set_index("model")
    assert list(result["n"]) == [2, 2]
    assert result.loc["ridge", "mae"] == 1
    assert result.loc["sarimax", "r2"] == 1


def test_ingestion_uses_completed_utc_hours_and_rejects_duplicates(monkeypatch):
    now = datetime(2026, 1, 2, 12, 30, tzinfo=timezone.utc)
    start = now.replace(minute=0) - timedelta(hours=24)
    stamps = [(start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(24)]
    bounds = stamps[0], stamps[-1]
    calls = []

    def request(_session, url, params, timeout):
        calls.append(url)
        assert timeout <= 6
        assert params["timezone"] == "GMT"
        assert (params["start_hour"], params["end_hour"]) == bounds
        return {"timezone": "GMT", "hourly": {"time": stamps,
                **{name: [10.0] * 24 for name in params["hourly"].split(",")}}}

    monkeypatch.setattr(mlops_check, "request_json", request)
    result = mlops_check.ingestion_check(now)
    assert len(calls) == 2
    assert result["row_count"] == len({row["timestamp_utc"] for row in result["rows"]}) == 24
    assert result["persistent_write"] is False
    stamps[-1] = stamps[0]
    assert mlops_check.ingestion_check(now)["status"] == "provider_unavailable_or_invalid"


def test_provider_error_does_not_leak_credentials(monkeypatch):
    def fail(*_args):
        raise requests.ConnectionError("private-provider-token")

    monkeypatch.setattr(mlops_check, "fetch_recent", fail)
    state = mlops_check.ingestion_check(datetime.now(timezone.utc))
    assert state["row_count"] == 0
    assert "private-provider-token" not in json.dumps(state)


def test_training_gate_checks_compatible_local_evidence_without_training(tmp_path):
    assert mlops_check.training_gate(tmp_path)["decision"] == "blocked_missing_persistent_dataset"
    dataset = tmp_path / "data/processed/features.csv"
    data = b"temperature_c\n30\n"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(data)
    assert mlops_check.training_gate(tmp_path)["decision"] == "blocked_missing_compatible_validation_manifest"
    digest = hashlib.sha256(data).hexdigest()
    manifest = tmp_path / "artifacts/day5c/champion_selection.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"dataset_sha256": digest, "selection_split": "validation",
                                    "decisions": [{"champion": "preserved"}]}))
    path = tmp_path / "outputs/agents/decisions.sqlite3"
    DecisionStore(path)
    assert mlops_check.training_gate(tmp_path)["decision"] == "blocked_missing_compatible_drift_evidence"
    state = {"issue_utc": "2026-01-01T00:00:00Z", "completed_at_utc": "2026-01-02T00:00:00Z",
             "mode": "local", "model_provenance": {"dataset_sha256": digest},
             "drift": {"status": "forecast_error_increase", "retraining_review_requested": True}}
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO agent_steps VALUES ('run', 4, 'drift_detection', ?)", (json.dumps(state),))
    assert mlops_check.training_gate(tmp_path)["decision"] == "blocked_unverified_persistent_storage"
    ready = mlops_check.training_gate(tmp_path, persistent_storage_verified=True)
    assert ready["decision"] == "manual_retraining_review_required"
    assert not ready["training_executed"] and not ready["champion_changed"]


@pytest.mark.parametrize("file,command", [
    ("hourly-ingestion.yml", "ingestion"),
    ("daily-training-gate.yml", "training-gate"),
])
def test_workflow_yaml_read_only_and_bounded(file, command):
    document = yaml.load((ROOT / ".github/workflows" / file).read_text(), Loader=yaml.BaseLoader)
    assert set(document["on"]) == {"schedule", "workflow_dispatch"}
    assert document["permissions"] == {"contents": "read"}
    assert document["concurrency"]["cancel-in-progress"] == "false"
    job = next(iter(document["jobs"].values()))
    assert int(job["timeout-minutes"]) <= 5
    assert job["steps"][0]["with"]["persist-credentials"] == "false"
    assert any(step.get("run", "").startswith(f"python -m scripts.mlops_check {command}")
               for step in job["steps"])
    assert not any(word in json.dumps(job) for word in ("secrets.", "upload-artifact", "git push", "train_baselines"))


def test_streamlit_renders_all_sections_and_plotly_metrics(tmp_path, monkeypatch):
    import asyncio

    testing = pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import local_script_runner

    # AppTest needs an event-loop object for synchronous rendering. Avoid the
    # TCP socket pair used by the default Windows loop in restricted runners.
    monkeypatch.setattr(local_script_runner, "asyncio",
                        SimpleNamespace(new_event_loop=asyncio.BaseEventLoop))
    path = tmp_path / "decisions.sqlite3"
    state = run_workflow(tmp_path, DecisionStore(path), synthetic=True)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setenv("AETHERIX_SNAPSHOT_PATH", str(snapshot))
    monkeypatch.setenv("AETHERIX_AGENT_DB", str(path))
    app = testing.AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Karachi weather intelligence"
    for page in app.sidebar.radio[0].options[1:]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
        if page == "24-Hour Forecast":
            for metric in METRICS:
                app.selectbox(key="forecast_metric").set_value(metric).run()
                assert not app.exception
                plots = app.get("plotly_chart")
                assert len(plots) == 1
                spec = json.loads(plots[0].proto.spec)
                assert len(spec["data"][0]["x"]) == 24
