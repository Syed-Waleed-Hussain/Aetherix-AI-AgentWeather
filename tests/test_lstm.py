from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from aetherix_weather.lstm import (
    FORECAST_HORIZON,
    SEQUENCE_LENGTH,
    TemperatureLSTM,
    build_sequences,
    chronological_sequence_origins,
    default_lstm_features,
    fit_training_scalers,
    set_reproducible_seed,
    transform_sequences,
)
from scripts.train_lstm import (
    comparison_rows_for_horizon,
    evaluate_split,
    feature_schema_hash,
    main,
    sha256,
)


def synthetic_frame(count: int = 120) -> pd.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=i) for i in range(count)]
    return pd.DataFrame({
        "timestamp_utc": [value.isoformat() for value in timestamps],
        "temperature_c": np.arange(count, dtype=np.float32),
        "feature_a": np.sin(np.arange(count)),
        "feature_b": np.arange(count, dtype=np.float32) % 7,
    })


def test_sequence_shape_and_24_horizons() -> None:
    frame = synthetic_frame()
    origins = np.arange(47, 96)
    sequences = build_sequences(
        frame, origins, ["feature_a", "feature_b"]
    )
    assert sequences.x.shape == (49, 48, 2)
    assert sequences.y.shape == (49, 24)
    assert sequences.target_times.shape == (49, 24)


def test_sequence_target_alignment_and_no_overlap() -> None:
    frame = synthetic_frame()
    sequences = build_sequences(
        frame, np.array([47, 48]), ["temperature_c", "feature_a"]
    )
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True)
    for row, origin in enumerate(sequences.origins):
        assert pd.Timestamp(sequences.target_times[row, 0]) == timestamps.iloc[origin + 1]
        assert pd.Timestamp(sequences.target_times[row, 23]) == timestamps.iloc[origin + 24]
        assert timestamps.iloc[origin] >= timestamps.iloc[origin - 47]
        for horizon in range(1, FORECAST_HORIZON + 1):
            assert pd.Timestamp(sequences.target_times[row, horizon - 1]) == (
                pd.Timestamp(timestamps.iloc[origin])
                + pd.Timedelta(horizon, unit="h")
            )


def test_future_leakage_and_partition_purging() -> None:
    origins = chronological_sequence_origins(1000, 700, 850)
    assert origins["train"][-1] + 24 < 700
    assert origins["validation"][-1] + 24 < 850
    assert origins["validation"][0] >= 700
    assert origins["test"][0] >= 850
    assert origins["train"][-1] < origins["validation"][0]


def test_training_only_scalers_and_missing_values() -> None:
    train_x = np.array([
        [[1.0, np.nan], [3.0, 2.0]],
        [[5.0, 4.0], [7.0, 6.0]],
    ], dtype=np.float32)
    train_y = np.array([[2.0, 3.0], [6.0, 7.0]], dtype=np.float32)
    imputer_x, scaler_x, imputer_y, scaler_y = fit_training_scalers(
        train_x, train_y
    )
    assert imputer_x.statistics_.tolist() == [4.0, 4.0]
    assert scaler_x.n_features_in_ == 2
    assert np.isfinite(scaler_y.mean_).all()


def test_lstm_output_and_deterministic_predictions() -> None:
    set_reproducible_seed(42)
    model = TemperatureLSTM(input_size=2)
    x = torch.randn(32, 48, 2)
    set_reproducible_seed(42)
    repeated = TemperatureLSTM(input_size=2)
    model.eval()
    repeated.eval()
    with torch.no_grad():
        first = model(x)
        second = repeated(x)
    assert first.shape == (32, 24)
    assert torch.equal(first, second)


def test_checkpoint_save_load_reproduces_prediction(tmp_path) -> None:
    set_reproducible_seed(42)
    model = TemperatureLSTM(input_size=2)
    x = torch.randn(4, 48, 2)
    model.eval()
    with torch.no_grad():
        expected = model(x)
    checkpoint = tmp_path / "lstm.pt"
    torch.save({
        "architecture": {"input_size": 2, "hidden_size": 32,
                         "num_layers": 1, "output_size": 24},
        "model_state_dict": model.state_dict(),
    }, checkpoint)

    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = TemperatureLSTM(
        input_size=loaded["architecture"]["input_size"]
    )
    restored.load_state_dict(loaded["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        actual = restored(x)
    assert torch.equal(expected, actual)


def test_inverse_scaling_returns_known_physical_temperature_values() -> None:
    physical = np.arange(24, dtype=np.float64).reshape(1, 24) + 18.5
    scaler = StandardScaler().fit(
        np.vstack([physical[0], physical[0] + 8.0, physical[0] - 4.0])
    )
    scaled = scaler.transform(physical).astype(np.float32)
    model_output = torch.from_numpy(scaled)

    class FixedModel(torch.nn.Module):
        def forward(self, _x):
            return model_output

    actual_targets = physical.astype(np.float32)
    predictions, returned_actuals = evaluate_split(
        FixedModel(),
        np.zeros((1, SEQUENCE_LENGTH, 1), dtype=np.float32),
        actual_targets,
        scaler,
    )
    np.testing.assert_allclose(predictions, physical, atol=1e-5)
    np.testing.assert_array_equal(returned_actuals, actual_targets)
    assert 18.0 < predictions.min() < predictions.max() < 43.0


def _write_evaluation_fixture(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "artifacts"
    processed = data_dir / "processed"
    processed.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    count = 300
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    index = np.arange(count)
    frame = pd.DataFrame({
        "timestamp_utc": [
            (start + timedelta(hours=int(i))).isoformat() for i in index
        ],
        "temperature_c": 25 + 4 * np.sin(index / 20),
        "relative_humidity_percent": 55 + index % 20,
        "precipitation_mm": index % 4,
        "wind_speed_kmh": 8 + index % 5,
        "uv_index": index % 10,
        "us_aqi": 60 + index % 15,
        "pm2_5_ug_m3": 20 + index % 9,
        "hour_sin": np.sin(index * 2 * np.pi / 24),
        "hour_cos": np.cos(index * 2 * np.pi / 24),
        "day_of_year_sin": np.sin(index * 2 * np.pi / 365),
        "day_of_year_cos": np.cos(index * 2 * np.pi / 365),
        "temperature_c_lag_24h": 25 + 4 * np.sin((index - 24) / 20),
        "temperature_c_lag_48h": 25 + 4 * np.sin((index - 48) / 20),
        "relative_humidity_percent_lag_24h": 55 + (index - 24) % 20,
        "wind_speed_kmh_lag_24h": 8 + (index - 24) % 5,
        "uv_index_lag_24h": (index - 24) % 10,
        "us_aqi_lag_24h": 60 + (index - 24) % 15,
    })
    data_path = processed / "features.csv"
    frame.to_csv(data_path, index=False)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    features = default_lstm_features(frame)
    train_end = int(count * 0.70)
    validation_end = int(count * 0.85)
    origins = chronological_sequence_origins(count, train_end, validation_end)
    train = build_sequences(frame, origins["train"], features)
    feature_imputer, feature_scaler, _, target_scaler = fit_training_scalers(
        train.x, train.y
    )
    model = TemperatureLSTM(len(features))
    architecture = {
        "input_size": len(features),
        "hidden_size": 32,
        "num_layers": 1,
        "output_size": FORECAST_HORIZON,
        "sequence_length": SEQUENCE_LENGTH,
    }
    boundaries = {
        "train_end": train_end,
        "validation_end": validation_end,
        "row_count": count,
        "train_issue_start": str(frame["timestamp_utc"].iloc[origins["train"][0]]),
        "train_issue_end": str(frame["timestamp_utc"].iloc[origins["train"][-1]]),
        "validation_issue_start": str(frame["timestamp_utc"].iloc[origins["validation"][0]]),
        "validation_issue_end": str(frame["timestamp_utc"].iloc[origins["validation"][-1]]),
        "test_issue_start": str(frame["timestamp_utc"].iloc[origins["test"][0]]),
        "test_issue_end": str(frame["timestamp_utc"].iloc[origins["test"][-1]]),
    }
    checkpoint_path = output_dir / "lstm_checkpoint.pt"
    torch.save({
        "architecture": architecture,
        "input_features": features,
        "dataset_sha256": sha256(data_path),
        "feature_schema_hash": feature_schema_hash(features),
        "boundaries": boundaries,
        "seed": 42,
        "python_version": "3.12.10",
        "torch_version": torch.__version__,
        "feature_scaler": {
            "mean": feature_scaler.mean_.tolist(),
            "scale": feature_scaler.scale_.tolist(),
            "imputer_median": feature_imputer.statistics_.tolist(),
        },
        "target_scaler": {
            "mean": target_scaler.mean_.astype(np.float32).tolist(),
            "scale": target_scaler.scale_.astype(np.float32).tolist(),
        },
        "model_state_dict": model.state_dict(),
        "training": {
            "epochs_completed": 1,
            "best_epoch": 1,
            "runtime_seconds": 1.0,
            "train_loss_history": [0.1],
            "validation_loss_history": [0.2],
            "final_train_loss": 0.1,
            "final_validation_loss": 0.2,
            "batch_size": 32,
            "learning_rate": 0.001,
            "patience": 3,
        },
    }, checkpoint_path)
    return data_dir, output_dir, checkpoint_path


def test_evaluate_only_preserves_checkpoint_and_reproduces_saved_predictions(
    tmp_path,
) -> None:
    data_dir, output_dir, checkpoint_path = _write_evaluation_fixture(tmp_path)
    before_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    assert main([
        "--evaluate-only",
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
    ]) == 0

    after_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert before_hash == after_hash
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data_path = data_dir / "processed" / "features.csv"
    assert checkpoint["dataset_sha256"] == sha256(data_path)
    assert checkpoint["feature_schema_hash"] == feature_schema_hash(
        checkpoint["input_features"]
    )
    assert len(checkpoint["model_state_dict"]) > 0

    frame = pd.read_csv(data_path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    features = checkpoint["input_features"]
    origins = chronological_sequence_origins(
        len(frame),
        checkpoint["boundaries"]["train_end"],
        checkpoint["boundaries"]["validation_end"],
    )
    train = build_sequences(frame, origins["train"], features)
    test = build_sequences(frame, origins["test"], features)
    feature_imputer, feature_scaler, _, target_scaler = fit_training_scalers(
        train.x, train.y
    )
    model = TemperatureLSTM(len(features))
    model.load_state_dict(checkpoint["model_state_dict"])
    restored_prediction, _ = evaluate_split(
        model,
        transform_sequences(test.x, feature_imputer, feature_scaler),
        test.y,
        target_scaler,
    )
    output = pd.read_csv(output_dir / "lstm_test_predictions.csv")
    np.testing.assert_allclose(
        output["predicted_temperature_c"].to_numpy(),
        restored_prediction.ravel(),
        rtol=0,
        atol=1e-6,
    )
    assert sorted(output["horizon_hours"].unique()) == list(
        range(1, FORECAST_HORIZON + 1)
    )
    issues = pd.to_datetime(output["issue_timestamp_utc"], utc=True)
    targets = pd.to_datetime(output["target_timestamp_utc"], utc=True)
    expected_targets = issues + pd.to_timedelta(output["horizon_hours"], unit="h")
    pd.testing.assert_series_equal(
        targets.reset_index(drop=True),
        expected_targets.reset_index(drop=True),
        check_names=False,
    )


def test_evaluate_only_rejects_dataset_hash_mismatch_without_checkpoint_write(
    tmp_path,
) -> None:
    data_dir, output_dir, checkpoint_path = _write_evaluation_fixture(tmp_path)
    data_path = data_dir / "processed" / "features.csv"
    frame = pd.read_csv(data_path)
    frame.loc[0, "temperature_c"] += 0.5
    frame.to_csv(data_path, index=False)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="dataset SHA-256"):
        main([
            "--evaluate-only",
            "--data-dir", str(data_dir),
            "--output-dir", str(output_dir),
        ])

    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == checkpoint_hash


def test_model_comparison_rejects_mixed_issue_populations() -> None:
    issue_times = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    target_times = issue_times + pd.Timedelta(1, unit="h")
    actual = np.array([20.0, 21.0, 22.0])
    baseline = pd.DataFrame({
        "metric": ["temperature_c"] * 3,
        "forecast_horizon_hours": [1] * 3,
        "issue_time_utc": issue_times,
        "target_time_utc": target_times,
        "target_value": [20.0, 21.0, 22.0],
        "selected_model": ["ridge", "ridge", "partial_model"],
        "prediction": [20.5, 21.5, 22.5],
        "seasonal_naive_prediction": [19.0, 20.0, 21.0],
    })
    rows, skipped = comparison_rows_for_horizon(
        baseline, issue_times, target_times, actual, 1
    )
    assert {row["model"] for row in rows} == {"seasonal_naive"}
    assert all(row["n"] == 3 and row["same_sample"] for row in rows)
    assert all(row["issue_timestamp_sha256"] for row in rows)
    assert "ridge:issue_timestamp_set_mismatch" in skipped


def test_cpu_dependency_and_gitignore_policies() -> None:
    root = Path(__file__).resolve().parents[1]
    cpu_requirements = (root / "requirements-day4b-cpu.txt").read_text(
        encoding="utf-8"
    )
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    ignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert "torch==2.6.0" in cpu_requirements
    assert "download.pytorch.org/whl/cpu" in workflow
    assert "requirements-day4b-cpu.txt" in workflow
    assert "cuda" not in workflow.lower()
    for rule in ("data/", "artifacts/", "*.pt", "*.pth", "*.joblib",
                 ".env", "secrets/", "*.secret", "*.credentials"):
        assert rule in ignore
