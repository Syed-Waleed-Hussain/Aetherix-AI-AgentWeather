import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aetherix_weather.model_registry import (
    feature_schema_hash,
    prepare_temperature_lstm_metadata,
    select_champion,
)


def _write_registry_fixture(root: Path) -> tuple[Path, Path, Path]:
    dataset_path = root / "data" / "processed" / "features.csv"
    dataset_path.parent.mkdir(parents=True)
    pd.DataFrame({"temperature_c": np.arange(80, dtype=float)}).to_csv(
        dataset_path, index=False
    )
    metadata_path = root / "artifacts" / "day4b" / "lstm_metadata.json"
    metadata_path.parent.mkdir(parents=True)
    checkpoint_path = metadata_path.parent / "lstm_checkpoint.pt"
    checkpoint_path.write_bytes(b"fixture-checkpoint")
    features = ["temperature_c", "hour_sin"]
    metadata = {
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "feature_schema_hash": feature_schema_hash(features),
        "input_features": features,
        "architecture": {"output_size": 4},
        "boundaries": {
            "train_end": 40,
            "validation_end": 72,
            "row_count": 80,
            "validation_issue_start": "2024-01-02T16:00:00+00:00",
            "validation_issue_end": "2024-01-03T16:00:00+00:00",
        },
        "target_scaler": {"mean": [0.0] * 4, "scale": [1.0] * 4},
        "training": {
            "best_epoch": 2,
            "validation_loss_history": [0.2, 0.1],
            "epochs_completed": 3,
        },
        "evaluation": [
            {
                "split": "test",
                "horizon_hours": 1,
                "mae": 0.3,
                "rmse": 0.4,
                "r2": 0.8,
                "n": 10,
            },
            {
                "split": "validation",
                "horizon_hours": 1,
                "mae": 999,
                "rmse": 999,
                "r2": -1,
                "n": 10,
            },
        ],
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return dataset_path, metadata_path, checkpoint_path


def test_registry_metadata_hashes_artifact_and_selects_on_validation_only(
    tmp_path: Path,
) -> None:
    dataset, model_metadata, checkpoint = _write_registry_fixture(tmp_path)

    prepared = prepare_temperature_lstm_metadata(
        dataset, model_metadata, checkpoint
    )
    lstm, naive = prepared["candidates"]

    assert prepared["registered"] is False
    assert prepared["selection_metric"] == "scaled_mse"
    assert prepared["champion"]["model_name"] == "aetherix_temperature_lstm"
    assert lstm["model_family"] == "pytorch_lstm"
    assert lstm["target"] == "temperature_c"
    assert lstm["horizons_hours"] == [1, 2, 3, 4]
    assert lstm["feature_schema"] == ["temperature_c", "hour_sin"]
    assert lstm["artifact"]["sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert lstm["validation_metrics"]["scaled_mse"]["value"] == 0.1
    assert len(lstm["test_metrics"]) == 1
    assert naive["validation_metrics"]["scaled_mse"]["value"] > 0.1


def test_registry_preparation_rejects_stale_dataset_hash(tmp_path: Path) -> None:
    dataset, model_metadata, checkpoint = _write_registry_fixture(tmp_path)
    with dataset.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ValueError, match="dataset hash"):
        prepare_temperature_lstm_metadata(dataset, model_metadata, checkpoint)


def test_registry_preparation_rejects_inconsistent_best_epoch(tmp_path: Path) -> None:
    dataset, model_metadata, checkpoint = _write_registry_fixture(tmp_path)
    metadata = json.loads(model_metadata.read_text(encoding="utf-8"))
    metadata["training"]["best_epoch"] = 1
    model_metadata.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="best epoch"):
        prepare_temperature_lstm_metadata(dataset, model_metadata, checkpoint)


def test_champion_selection_requires_comparable_validation_candidates() -> None:
    common = {
        "target": "temperature_c",
        "horizons_hours": [1],
        "dataset_sha256": "same-dataset",
        "validation_period": {"start": "a", "end": "b"},
    }
    better = {
        **common,
        "model_name": "better",
        "validation_metrics": {
            "mae": {"value": 0.2, "direction": "minimize"}
        },
    }
    worse = {
        **common,
        "model_name": "worse",
        "validation_metrics": {
            "mae": {"value": 0.3, "direction": "minimize"}
        },
    }

    assert select_champion([worse, better]) is better

    incompatible = {**worse, "dataset_sha256": "other-dataset"}
    with pytest.raises(ValueError, match="share target"):
        select_champion([better, incompatible])
