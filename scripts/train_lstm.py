"""Train and evaluate the bounded CPU temperature LSTM experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aetherix_weather.lstm import (
    FORECAST_HORIZON,
    SEQUENCE_LENGTH,
    TemperatureLSTM,
    build_sequences,
    chronological_sequence_origins,
    default_lstm_features,
    fit_training_scalers,
    inverse_transform_targets,
    set_reproducible_seed,
    transform_sequences,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_schema_hash(features: Iterable[str]) -> str:
    return hashlib.sha256(
        "\n".join(features).encode("utf-8")
    ).hexdigest()


def validate_checkpoint(
    checkpoint: dict,
    dataset_hash: str,
    features: list[str],
    architecture: dict,
    feature_scaler: object,
    feature_imputer: object,
    target_scaler: object,
    boundaries: dict,
) -> None:
    if checkpoint.get("dataset_sha256") != dataset_hash:
        raise ValueError("Checkpoint dataset SHA-256 does not match the current dataset.")
    if checkpoint.get("input_features") != features:
        raise ValueError("Checkpoint input features do not match the current schema.")
    if checkpoint.get("feature_schema_hash") != feature_schema_hash(features):
        raise ValueError("Checkpoint feature schema hash does not match.")
    if checkpoint.get("architecture") != architecture:
        raise ValueError("Checkpoint architecture does not match this experiment.")
    if checkpoint.get("boundaries") != boundaries:
        raise ValueError("Checkpoint chronological boundaries do not match.")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("Checkpoint is missing its model state dictionary.")
    saved_features = checkpoint.get("feature_scaler", {})
    saved_targets = checkpoint.get("target_scaler", {})
    pairs = (
        (saved_features.get("mean"), feature_scaler.mean_),
        (saved_features.get("scale"), feature_scaler.scale_),
        (saved_features.get("imputer_median"), feature_imputer.statistics_),
        (saved_targets.get("mean"), target_scaler.mean_),
        (saved_targets.get("scale"), target_scaler.scale_),
    )
    if any(
        values is None
        or not np.allclose(
            np.asarray(values), np.asarray(expected), rtol=0.0, atol=1e-6
        )
        for values, expected in pairs
    ):
        raise ValueError("Checkpoint preprocessing parameters do not match the dataset.")
    training = checkpoint.get("training")
    if not isinstance(training, dict) or not training.get("train_loss_history"):
        raise ValueError("Checkpoint is missing its training history.")


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def train_epochs(
    model: TemperatureLSTM,
    train_loader: DataLoader,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    epochs: int,
    patience: int,
    output_dir: Path,
    checkpoint_state: dict,
) -> tuple[list[float], list[float], int, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    train_history: list[float] = []
    validation_history: list[float] = []
    best_validation = float("inf")
    best_epoch = 0
    bad_epochs = 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        train_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predicted = model(batch_x)
            loss = loss_function(predicted, batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss detected.")
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            validation_predicted = model(validation_x)
            validation_loss = float(
                loss_function(validation_predicted, validation_y).item()
            )
        train_loss = float(np.mean(train_losses))
        train_history.append(train_loss)
        validation_history.append(validation_loss)
        elapsed_epoch = time.monotonic() - epoch_started
        print(
            f"[epoch] {epoch:02d}/{epochs} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} elapsed={elapsed_epoch:.1f}s",
            flush=True,
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            bad_epochs = 0
            checkpoint_state["model_state_dict"] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(
                    f"[early-stop] Patience reached at epoch {epoch}; "
                    f"best epoch {best_epoch}",
                    flush=True,
                )
                break
    torch.save(checkpoint_state, output_dir / "lstm_checkpoint.pt")
    return train_history, validation_history, best_epoch, time.monotonic() - started


def evaluate_split(
    model: TemperatureLSTM,
    x: np.ndarray,
    y: np.ndarray,
    target_scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        scaled_prediction = model(torch.from_numpy(x)).numpy()
    prediction = inverse_transform_targets(scaled_prediction, target_scaler)
    return prediction, y


def comparison_rows_for_horizon(
    baseline: pd.DataFrame,
    issue_times: pd.DatetimeIndex,
    target_times: pd.DatetimeIndex,
    actual: np.ndarray,
    horizon: int,
) -> tuple[list[dict], list[str]]:
    expected_issues = pd.DatetimeIndex(pd.to_datetime(issue_times, utc=True))
    expected_targets = pd.DatetimeIndex(pd.to_datetime(target_times, utc=True))
    expected = pd.DataFrame({
        "issue_time_utc": expected_issues,
        "target_time_utc": expected_targets,
        "target_value": actual,
    })
    expected_issue_set = set(expected_issues)
    expected_issue_hash = hashlib.sha256(
        "\n".join(sorted(value.isoformat() for value in expected_issue_set)).encode()
    ).hexdigest()
    expected_target_hash = hashlib.sha256(
        "\n".join(sorted(value.isoformat() for value in expected_targets)).encode()
    ).hexdigest()
    rows: list[dict] = []
    skipped: list[str] = []
    filtered = baseline[
        (baseline["metric"] == "temperature_c")
        & (baseline["forecast_horizon_hours"] == horizon)
    ].copy()
    filtered["issue_time_utc"] = pd.to_datetime(
        filtered["issue_time_utc"], utc=True, errors="raise"
    )
    filtered["target_time_utc"] = pd.to_datetime(
        filtered["target_time_utc"], utc=True, errors="raise"
    )

    def add_if_exact(model: str, predictions: pd.DataFrame, column: str) -> None:
        if predictions["issue_time_utc"].duplicated().any():
            skipped.append(f"{model}:duplicate_issue_timestamps")
            return
        indexed = predictions.set_index("issue_time_utc").sort_index()
        if not expected_issue_set.issubset(set(indexed.index)):
            skipped.append(f"{model}:issue_timestamp_set_mismatch")
            return
        indexed = indexed.reindex(expected_issues)
        actual_target_times = pd.DatetimeIndex(indexed["target_time_utc"])
        if not actual_target_times.equals(expected_targets):
            skipped.append(f"{model}:target_timestamp_mismatch")
            return
        baseline_actual = indexed["target_value"].to_numpy(dtype=float)
        if not np.allclose(baseline_actual, actual, rtol=0.0, atol=1e-5):
            skipped.append(f"{model}:actual_values_mismatch")
            return
        score = metrics(actual, indexed[column].to_numpy(dtype=float))
        rows.append({
            "horizon_hours": horizon,
            "model": model,
            "n": len(expected_issues),
            "same_sample": True,
            "issue_timestamp_sha256": expected_issue_hash,
            "target_timestamp_sha256": expected_target_hash,
            **score,
        })

    for model_name, model_rows in filtered.groupby("selected_model"):
        if model_name == "seasonal_naive":
            continue
        add_if_exact(
            str(model_name),
            model_rows,
            "prediction",
        )
    if (
        not filtered.empty
        and not filtered["seasonal_naive_prediction"].isna().all()
    ):
        add_if_exact("seasonal_naive", filtered, "seasonal_naive_prediction")
    return rows, skipped


def load_comparison_rows() -> pd.DataFrame:
    paths = [
        Path("artifacts/day3_5/model_evaluation.csv"),
        Path("artifacts/day4a/temperature/model_comparison.csv"),
    ]
    frames = []
    for path in paths:
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--pilot-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/day4b"))
    parser.add_argument("--skip-pilot", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args(argv)
    set_reproducible_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.data_dir / "processed" / "features.csv"
    frame = pd.read_csv(data_path)
    frame["timestamp_utc"] = pd.to_datetime(
        frame["timestamp_utc"], utc=True, errors="raise"
    )
    if not frame["timestamp_utc"].is_monotonic_increasing:
        raise ValueError("Processed dataset timestamps are not chronological.")
    features = default_lstm_features(frame)
    row_count = len(frame)
    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)
    origins = chronological_sequence_origins(
        row_count, train_end, validation_end
    )
    train = build_sequences(frame, origins["train"], features)
    validation = build_sequences(frame, origins["validation"], features)
    test = build_sequences(frame, origins["test"], features)
    if len(train.x) == 0 or len(validation.x) == 0 or len(test.x) == 0:
        raise RuntimeError("One or more LSTM partitions contain no valid sequences.")

    feature_imputer, feature_scaler, target_imputer, target_scaler = (
        fit_training_scalers(train.x, train.y)
    )
    train_x = transform_sequences(
        train.x, feature_imputer, feature_scaler
    )
    validation_x = transform_sequences(
        validation.x, feature_imputer, feature_scaler
    )
    test_x = transform_sequences(test.x, feature_imputer, feature_scaler)
    target_mean = target_scaler.mean_.astype(np.float32)
    target_scale = target_scaler.scale_.astype(np.float32)
    train_y = (train.y - target_mean) / target_scale
    validation_y = (validation.y - target_mean) / target_scale

    print(
        f"[preflight] rows={row_count} features={len(features)} "
        f"train_sequences={len(train.x)} val_sequences={len(validation.x)} "
        f"test_sequences={len(test.x)}",
        flush=True,
    )
    print(
        f"[shape] x={train_x.shape} y={train_y.shape} "
        f"test_x={test_x.shape}",
        flush=True,
    )
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x),
            torch.from_numpy(train_y.astype(np.float32)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_x_tensor = torch.from_numpy(validation_x)
    validation_y_tensor = torch.from_numpy(
        validation_y.astype(np.float32)
    )
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
        "row_count": row_count,
        "train_issue_start": str(frame["timestamp_utc"].iloc[origins["train"][0]]),
        "train_issue_end": str(frame["timestamp_utc"].iloc[origins["train"][-1]]),
        "validation_issue_start": str(frame["timestamp_utc"].iloc[origins["validation"][0]]),
        "validation_issue_end": str(frame["timestamp_utc"].iloc[origins["validation"][-1]]),
        "test_issue_start": str(frame["timestamp_utc"].iloc[origins["test"][0]]),
        "test_issue_end": str(frame["timestamp_utc"].iloc[origins["test"][-1]]),
    }
    checkpoint_state: dict = {
        "architecture": architecture,
        "input_features": features,
        "dataset_sha256": sha256(data_path),
        "feature_schema_hash": feature_schema_hash(features),
        "boundaries": boundaries,
        "seed": args.seed,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "feature_scaler": {
            "mean": feature_scaler.mean_.tolist(),
            "scale": feature_scaler.scale_.tolist(),
            "imputer_median": feature_imputer.statistics_.tolist(),
        },
        "target_scaler": {
            "mean": target_mean.tolist(),
            "scale": target_scale.tolist(),
        },
    }

    model = TemperatureLSTM(len(features), hidden_size=32, num_layers=1)
    if args.evaluate_only:
        checkpoint_path = args.output_dir / "lstm_checkpoint.pt"
        saved = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        validate_checkpoint(
            saved,
            checkpoint_state["dataset_sha256"],
            features,
            architecture,
            feature_scaler,
            feature_imputer,
            target_scaler,
            boundaries,
        )
        model.load_state_dict(saved["model_state_dict"])
        checkpoint_state = saved
        training = saved["training"]
        train_history = training["train_loss_history"]
        validation_history = training["validation_loss_history"]
        best_epoch = training["best_epoch"]
        runtime = training["runtime_seconds"]
    elif not args.skip_pilot:
        print("[pilot] Starting 2-epoch pilot.", flush=True)
        pilot_history = train_epochs(
            model,
            train_loader,
            validation_x_tensor,
            validation_y_tensor,
            args.pilot_epochs,
            args.patience,
            args.output_dir,
            checkpoint_state,
        )
        print(
            f"[pilot] completed epochs={len(pilot_history[0])} "
            f"elapsed={pilot_history[3]:.1f}s",
            flush=True,
        )
        if not all(
            np.isfinite(value)
            for value in pilot_history[0] + pilot_history[1]
        ):
            raise RuntimeError("Pilot produced non-finite losses.")
        if args.pilot_only:
            print("[pilot-only] Stopping before bounded full training.", flush=True)
            return 0
        set_reproducible_seed(args.seed)
        model = TemperatureLSTM(len(features), hidden_size=32, num_layers=1)

    if not args.evaluate_only:
        print("[full] Starting bounded training.", flush=True)
        train_history, validation_history, best_epoch, runtime = train_epochs(
            model,
            train_loader,
            validation_x_tensor,
            validation_y_tensor,
            args.epochs,
            args.patience,
            args.output_dir,
            checkpoint_state,
        )
        checkpoint = torch.load(
            args.output_dir / "lstm_checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    test_prediction, test_actual = evaluate_split(
        model,
        test_x,
        test.y,
        target_scaler,
    )

    rows = []
    test_issue_times = frame["timestamp_utc"].iloc[test.origins].to_numpy()
    for horizon in range(1, FORECAST_HORIZON + 1):
        score = metrics(test_actual[:, horizon - 1], test_prediction[:, horizon - 1])
        target_time = test.target_times[:, horizon - 1]
        rows.append({
            "split": "test",
            "metric": "temperature_c",
            "horizon_hours": horizon,
            "model": "lstm",
            "n": len(test_prediction),
            "issue_timestamp_utc": str(pd.Timestamp(test_issue_times[0])) if horizon == 1 else "",
            "target_timestamp_utc": str(pd.Timestamp(target_time[0])) if horizon == 1 else "",
            **score,
        })
    per_horizon = pd.DataFrame(rows)
    per_horizon.to_csv(
        args.output_dir / "lstm_per_horizon.csv", index=False
    )

    predictions = pd.DataFrame({
        "issue_timestamp_utc": np.repeat(test_issue_times, FORECAST_HORIZON),
        "horizon_hours": np.tile(np.arange(1, FORECAST_HORIZON + 1), len(test.origins)),
        "target_timestamp_utc": test.target_times.ravel(),
        "actual_temperature_c": test_actual.ravel(),
        "predicted_temperature_c": test_prediction.ravel(),
    })
    predictions.to_csv(
        args.output_dir / "lstm_test_predictions.csv", index=False
    )

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(train_history) + 1), train_history, label="train")
    plt.plot(range(1, len(validation_history) + 1), validation_history, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled MSE")
    plt.title("Temperature LSTM training and validation loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_dir / "training_loss.png", dpi=140)
    plt.close()

    horizon_plot = per_horizon.set_index("horizon_hours")
    horizon_plot["mae"].plot(marker="o", figsize=(9, 5), title="LSTM MAE by horizon")
    plt.xlabel("Forecast horizon (hours)")
    plt.ylabel("MAE (C)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_dir / "per_horizon_mae.png", dpi=140)
    plt.close()

    plt.figure(figsize=(10, 5))
    sample_count = min(500, len(test_actual))
    plt.plot(test_actual[:sample_count, 0], label="Actual h01", alpha=0.8)
    plt.plot(test_prediction[:sample_count, 0], label="Predicted h01", alpha=0.8)
    plt.title("Temperature actual versus predicted (first 500 test issues)")
    plt.xlabel("Test issue sequence")
    plt.ylabel("Temperature (C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "actual_vs_predicted_h01.png", dpi=140)
    plt.close()

    checkpoint_state["training"] = {
        "epochs_completed": len(train_history),
        "best_epoch": best_epoch,
        "runtime_seconds": round(runtime, 3),
        "train_loss_history": train_history,
        "validation_loss_history": validation_history,
        "final_train_loss": train_history[-1],
        "final_validation_loss": validation_history[-1],
        "batch_size": args.batch_size,
        "learning_rate": 0.001,
        "patience": args.patience,
    }
    checkpoint_state["evaluation"] = rows
    if not args.evaluate_only:
        torch.save(checkpoint_state, args.output_dir / "lstm_checkpoint.pt")
    metadata_state = dict(checkpoint_state)
    metadata_state.pop("model_state_dict", None)
    with (args.output_dir / "lstm_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata_state, handle, indent=2)

    comparison_rows = []
    test_prediction_path = Path("artifacts/day3_5/test_predictions.csv")
    baseline_predictions = (
        pd.read_csv(test_prediction_path)
        if test_prediction_path.is_file()
        else pd.DataFrame()
    )
    if not baseline_predictions.empty:
        baseline_predictions["issue_time_utc"] = pd.to_datetime(
            baseline_predictions["issue_time_utc"], utc=True
        )
    issue_times = pd.DatetimeIndex(pd.to_datetime(test_issue_times, utc=True))
    for _, lstm_row in per_horizon.iterrows():
        horizon = int(lstm_row["horizon_hours"])
        expected_targets = pd.DatetimeIndex(test.target_times[:, horizon - 1])
        actual = test_actual[:, horizon - 1]
        matched_rows: list[dict] = []
        skipped_models: list[str] = []
        if not baseline_predictions.empty:
            baseline_rows, skipped_models = comparison_rows_for_horizon(
                baseline_predictions,
                issue_times,
                expected_targets,
                actual,
                horizon,
            )
            matched_rows.extend(baseline_rows)
        statistical_path = Path(
            "artifacts/day4a/temperature/statistical_forecasts.csv"
        )
        if statistical_path.is_file():
            statistical = pd.read_csv(statistical_path)
            statistical = statistical[
                (statistical["split"] == "test")
                & (statistical["metric"] == "temperature_c")
                & (statistical["horizon_hours"] == horizon)
            ].rename(columns={
                "issue_time_utc": "issue_time_utc",
                "target_time_utc": "target_time_utc",
                "actual": "target_value",
                "forecast": "prediction",
                "model": "selected_model",
            })
            statistical["forecast_horizon_hours"] = statistical["horizon_hours"]
            statistical["seasonal_naive_prediction"] = np.nan
            sarimax_rows, sarimax_skipped = comparison_rows_for_horizon(
                statistical,
                issue_times,
                expected_targets,
                actual,
                horizon,
            )
            matched_rows.extend(sarimax_rows)
            skipped_models.extend(sarimax_skipped)
        for skipped in skipped_models:
            print(
                f"[comparison-skip] h{horizon:02d} {skipped}",
                flush=True,
            )
        comparison_rows.extend(matched_rows)
        issue_hash = hashlib.sha256(
            "\n".join(sorted(value.isoformat() for value in issue_times)).encode()
        ).hexdigest()
        target_hash = hashlib.sha256(
            "\n".join(sorted(value.isoformat() for value in expected_targets)).encode()
        ).hexdigest()
        comparison_rows.append({
            "horizon_hours": horizon,
            "model": "lstm",
            "n": int(lstm_row["n"]),
            "same_sample": True,
            "issue_timestamp_sha256": issue_hash,
            "target_timestamp_sha256": target_hash,
            "mae": float(lstm_row["mae"]),
            "rmse": float(lstm_row["rmse"]),
            "r2": float(lstm_row["r2"]),
        })
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame.to_csv(
        args.output_dir / "model_comparison.csv", index=False
    )
    print(
        f"[complete] epochs={len(train_history)} "
        f"runtime={runtime:.1f}s "
        f"test_samples={len(test_prediction)} "
        f"checkpoint={args.output_dir / 'lstm_checkpoint.pt'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
