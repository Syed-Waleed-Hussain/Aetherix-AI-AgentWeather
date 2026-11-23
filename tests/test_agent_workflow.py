"""Day 6 regression coverage. No service calls or training runs."""

import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from aetherix_weather.agent_drift import _feature_drift, _forecast_error_checks, drift_result
from aetherix_weather.agent_forecast import InputError, LocalForecastSource, prediction, utc_hour
from aetherix_weather.agent_reporting import calculate_alerts, groq_report_layout, validated_thresholds
from aetherix_weather.agent_workflow import DecisionStore, NODES, build_graph, run_workflow
from aetherix_weather.model_registry import feature_schema_hash
from scripts import run_agents

ISSUE = pd.Timestamp("2026-01-01T00:00:00Z")


def test_real_graph_runs_four_nodes_and_persists_replay(tmp_path, monkeypatch):
    def unexpected_network(*_args, **_kwargs):
        pytest.fail("Network access is forbidden for offline execution")

    monkeypatch.setattr(requests.Session, "request", unexpected_network)
    store = DecisionStore(tmp_path / "decisions.sqlite3")
    graph = build_graph(LocalForecastSource(tmp_path), store)
    assert set(graph.get_graph().nodes) == {"__start__", "__end__", *NODES}
    first = run_workflow(tmp_path, store, synthetic=True)
    second = run_workflow(tmp_path, store, synthetic=True)
    assert first["request_id"] != second["request_id"]
    assert list(first["statuses"]) == list(NODES)
    assert len(first["predictions"]) == 192
    assert len(first["alerts"]) == 5
    assert "Synthetic demonstration" in first["report"]["text"]
    assert first["drift"]["status"] == "input_shift"
    assert first["completed_at_utc"]
    for row in first["predictions"]:
        assert pd.Timestamp(row["valid_at_utc"]) == ISSUE + pd.Timedelta(row["horizon"], unit="h")
    assert utc_hour("2026-01-01T05:30:00+05:30") == ISSUE
    with pytest.raises(InputError):
        utc_hour("2026-01-01T05:00:00+05:30")
    snapshots = store.replay(first["request_id"])
    assert [len(state["decisions"]) for state in snapshots] == [1, 2, 3, 4]
    assert all(not snapshot["completed_at_utc"] for snapshot in snapshots[:-1])
    assert snapshots[-1] == first
    assert snapshots[0]["alerts"] == []


def test_alert_boundaries_and_model_provenance():
    values = (37.999, 38, 38.01, 42, 42, 41.99, 37.9, 38)
    rows = [
        prediction("temperature_c", h, ISSUE, value, {"name": f"model{h}"})
        for h, value in enumerate(values, 1)
    ]
    alerts = calculate_alerts(rows, validated_thresholds({}), "outdoor_workers")
    assert [item["severity"] for item in alerts] == ["elevated", "high", "elevated", "elevated"]
    assert alerts[0]["start_utc"] == "2026-01-01T02:00:00+00:00"
    assert alerts[0]["end_utc_exclusive"] == "2026-01-01T04:00:00+00:00"
    assert [model["name"] for model in alerts[0]["models"]] == ["model2", "model3"]
    assert alerts[1]["threshold"] == 42
    assert "shade breaks" in alerts[1]["precaution"]
    custom = validated_thresholds({"temperature_c": {"elevated": 45, "high": 50}})
    assert calculate_alerts(rows, custom, "general") == []
    with pytest.raises(ValueError, match="finite"):
        validated_thresholds({"temperature_c": {"elevated": 45, "high": float("nan")}})


def _local_source(tmp_path: Path, *, fallback=True) -> LocalForecastSource:
    source = LocalForecastSource(tmp_path)
    columns = ["temperature_c"]
    source.dataset_hash = "verified-hash"
    source.frame = pd.DataFrame({
        "_timestamp_utc": pd.date_range(end=ISSUE, periods=24, freq="h"),
        "temperature_c": range(20, 44),
    })
    source.selections = {("temperature_c", 1): {
        "selected_model": "ridge_alpha_1", "feature_columns": columns,
        "seasonal_naive_validation_rows": int(fallback), "validation_rows": 1,
    }}
    source.manifest = {("temperature_c", 1): {
        "model_name": "ridge_alpha_1", "dataset_sha256": "verified-hash",
        "feature_schema_sha256": feature_schema_hash(columns), "metric": "mae",
    }}
    source.estimators = {}
    return source


@pytest.mark.parametrize("fallback", [True, False])
def test_local_model_failure_uses_only_a_validated_existing_fallback(tmp_path, fallback):
    source = _local_source(tmp_path, fallback=fallback)
    result = source.forecast(None, 1)
    if fallback:
        assert result["predictions"][0]["value"] == 20
        assert result["predictions"][0]["model"]["fallback_for"] == "ridge_alpha_1"
        assert result["forecast_fallbacks"][0]["reason"] == "KeyError"
    else:
        assert not result["predictions"]
        assert result["coverage_gaps"][0]["reason"] == "KeyError"


def test_incompatible_champion_is_rejected(tmp_path, monkeypatch):
    source = _local_source(tmp_path)
    source.manifest[("temperature_c", 1)]["feature_schema_sha256"] = "wrong"
    monkeypatch.setattr(source, "_forecast_value", lambda *_: pytest.fail("Inference attempted"))
    state = source.forecast(None, 1)
    assert not state["predictions"]
    assert "compatible" in state["coverage_gaps"][0]["reason"]


def test_failed_forecast_and_drift_record_safe_errors(tmp_path, monkeypatch):
    def fail(*_args):
        raise RuntimeError("secret-in-upstream-error")

    monkeypatch.setattr(LocalForecastSource, "forecast", fail)
    monkeypatch.setattr("aetherix_weather.agent_workflow.monitor_local", fail)
    store = DecisionStore(tmp_path / "decisions.sqlite3")
    state = run_workflow(tmp_path, store, horizon=1)
    assert state["statuses"] == {
        "forecast": "failed", "alert": "skipped", "weather_report": "degraded",
        "drift_detection": "failed",
    }
    assert "Forecast unavailable" in state["report"]["text"]
    assert [entry["reason"] for entry in state["errors"]] == ["RuntimeError", "RuntimeError"]
    assert "secret-in-upstream-error" not in json.dumps(store.replay(state["request_id"]))


@pytest.mark.parametrize("provider_result", ["conditions_first", "Fabricated 999°C", None])
def test_report_provider_can_only_choose_layout(tmp_path, monkeypatch, provider_result):
    source = _local_source(tmp_path)
    forecast = source.forecast
    monkeypatch.setattr(LocalForecastSource, "forecast", lambda *_: forecast(None, 1))
    monkeypatch.setattr("aetherix_weather.agent_workflow.monitor_local",
                        lambda *_: drift_result([], [], ISSUE.isoformat()))

    def provider(_has_alerts):
        if provider_result is None:
            raise requests.Timeout("provider-secret")
        return provider_result

    store = DecisionStore(tmp_path / "decisions.sqlite3")
    state = run_workflow(tmp_path, store, horizon=1, provider=provider)
    assert "20.00" in state["report"]["text"]
    logs = json.dumps(store.replay(state["request_id"]))
    assert all(word not in logs for word in ("provider-secret", "Fabricated 999"))
    if provider_result == "conditions_first":
        assert state["report"]["provider"] == "groq_layout"
    else:
        assert state["report"]["provider"] == "template_fallback"
        assert state["statuses"]["weather_report"] == "degraded"
    assert state["statuses"]["drift_detection"] == "ok"


def test_rate_limit_never_logs_provider_body(tmp_path, monkeypatch):
    source = _local_source(tmp_path)
    forecast = source.forecast
    monkeypatch.setattr(LocalForecastSource, "forecast", lambda *_: forecast(None, 1))
    monkeypatch.setattr("aetherix_weather.agent_workflow.monitor_local",
                        lambda *_: drift_result([], [], ISSUE.isoformat()))
    monkeypatch.setenv("GROQ_API_KEY", "test-secret-value")
    monkeypatch.setenv("GROQ_MODEL", "test-fixture")
    response = requests.Response()
    response.status_code = 429
    response._content = b"untrusted-provider-body"
    monkeypatch.setattr(requests, "post", lambda *_args, **_kwargs: response)
    store = DecisionStore(tmp_path / "decisions.sqlite3")
    state = run_workflow(tmp_path, store, horizon=1, provider=groq_report_layout)
    assert state["errors"] == [{"agent": "weather_report", "reason": "groq_rate_limited"}]
    logs = json.dumps(store.replay(state["request_id"]))
    assert "test-secret-value" not in logs
    assert "untrusted-provider-body" not in logs


def test_drift_separates_missingness_and_matured_circular_error():
    reference = pd.DataFrame({"temperature_c": [25.0] * 30})
    recent = pd.DataFrame({"temperature_c": [25.0] * 24 + [None] * 6})
    metrics = _feature_drift(reference, recent)
    assert metrics[0]["ks"] == 0
    assert metrics[0]["recent_missing_rate"] == pytest.approx(0.2)
    assert drift_result(metrics, [], ISSUE.isoformat())["status"] == "input_shift"
    assert not drift_result(metrics, [], ISSUE.isoformat())["retraining_review_requested"]

    times = pd.date_range(end=ISSUE, periods=25, freq="h")
    actual = pd.DataFrame({
        "_timestamp_utc": times, "wind_direction_degrees": [359.0] * 25,
        "wind_direction_degrees_was_missing": [False] * 24 + [True],
    })
    archive = pd.DataFrame({
        "target_time_utc": times, "prediction": [1.0] * 24 + [180.0],
        "metric": "wind_direction_degrees", "forecast_horizon_hours": 1,
        "selected_model": "ridge_alpha_1",
    })
    scores = _forecast_error_checks(archive, actual, {("wind_direction_degrees", 1): {
        "selected_model": "ridge_alpha_1", "validation_score": 1.0,
        "selection_metric": "circular_mae_degrees",
    }})
    assert scores[0]["observations"] == 24
    assert scores[0]["recent_holdout_error"] == 2
    assert drift_result(metrics, scores, ISSUE.isoformat())["retraining_review_requested"]


def test_cli_replays_saved_json_and_rejects_unsupported_input(tmp_path, capsys):
    db = tmp_path / "decisions.sqlite3"
    assert run_agents.main(["--synthetic", "--db", str(db), "--json", "--horizon", "1"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert run_agents.main(["--db", str(db), "--json", "--replay", first["request_id"]]) == 0
    assert json.loads(capsys.readouterr().out) == first
    assert run_agents.main(["--synthetic", "--db", str(db), "--output", str(db)]) == 2
    assert DecisionStore(db).replay(first["request_id"])[-1] == first
    with pytest.raises(InputError, match="Karachi"):
        run_workflow(tmp_path, DecisionStore(db), city="Mumbai", synthetic=True)
