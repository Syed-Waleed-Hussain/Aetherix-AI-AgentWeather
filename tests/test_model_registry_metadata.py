import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aetherix_weather import model_registry
from aetherix_weather.champions import decide_champion
from aetherix_weather.model_registry import (
    feature_schema_hash,
    prepare_temperature_lstm_metadata,
    select_champion,
)
from scripts import prepare_model_registry


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


@pytest.mark.parametrize(
    ("target", "metric"),
    [("temperature_c", "mae"), ("rain_occurred", "brier_score"),
     ("wind_direction_degrees", "circular_mae_degrees")],
)
def test_horizon_champion_uses_only_compatible_validation_evidence(target, metric):
    incumbent = {
        "model_name": "incumbent", "target": target, "horizon_hours": 1,
        "metric": metric, "dataset_sha256": "dataset", "feature_schema_sha256": "schema",
        "validation_observations": {"start": "a", "end": "b", "count": 10},
        "validation_score": 0.2,
    }
    better = {**incumbent, "model_name": "new", "validation_score": 0.1, "test_score": 99}
    for field, value in (
        ("target", "other"), ("horizon_hours", 2), ("dataset_sha256", "other"),
        ("feature_schema_sha256", "other"), ("validation_observations", {"count": 9}),
        ("metric", "scaled_mse"),
    ):
        decision = decide_champion(incumbent, [{**better, field: value}])
        assert decision["champion"] == incumbent
        assert field in decision["challengers"][0]["reason"]
    assert decide_champion(incumbent, [better])["champion"] == better
    assert decide_champion(incumbent, [{**better, "validation_score": 0.2}])["champion"] == incumbent


def test_registry_reuses_verified_version_and_aborts_if_metadata_changes(tmp_path, monkeypatch):
    checkpoint = tmp_path / "lstm_checkpoint.pt"
    checkpoint.write_bytes(b"temperature weights")
    metadata = {
        "model_name": "aetherix_temperature_lstm",
        "artifact": {"sha256": model_registry.sha256_file(checkpoint)},
        "validation_metrics": {"scaled_mse": {"value": 0.1}},
    }
    metadata_path = tmp_path / "registry_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    schema = {"input_schema": {}, "output_schema": {}}
    monkeypatch.setattr(
        model_registry, "_sdk_model_schema", lambda _: SimpleNamespace(to_dict=lambda: schema)
    )
    models = []
    writes = []

    def download(directory):
        for file in (checkpoint, metadata_path):
            shutil.copy(file, directory)
        return directory

    def create_model(name, *, description, metrics, model_schema):
        assert name == metadata["model_name"]

        def save(path):
            assert Path(path) == tmp_path
            version = SimpleNamespace(
                description=description, training_metrics=metrics, version=1,
                download=download, model_schema=model_schema.to_dict(),
            )
            models.append(version)
            writes.append(path)
            return version

        return SimpleNamespace(save=save)

    registry = SimpleNamespace(
        python=SimpleNamespace(create_model=create_model),
        get_models=lambda _name: models,
        get_model=lambda _name, *, version: models[version - 1],
    )
    sync = model_registry.sync_registry_candidate
    assert sync(registry, tmp_path, metadata, execute=False) == (None, False)
    assert sync(registry, tmp_path, metadata, execute=True) == (1, True)
    assert sync(registry, tmp_path, metadata, execute=True) == (1, False)
    metadata_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="read-back"):
        sync(registry, tmp_path, metadata, execute=True)
    assert len(writes) == 1


def test_registry_cli_local_fallback_and_sanitized_cloud_failure(tmp_path, monkeypatch, capsys):
    artifact = tmp_path / "day4b"
    artifact.mkdir()
    (artifact / "lstm_checkpoint.pt").write_bytes(b"local checkpoint")
    monkeypatch.setattr(prepare_model_registry, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(prepare_model_registry, "verify_secret_ignored", lambda _: None)
    monkeypatch.setattr(prepare_model_registry, "_verify_with_local_torch", lambda *_: None)
    monkeypatch.setattr(
        prepare_model_registry, "registration_metadata", lambda *_: {"artifact": {"sha256": "hash"}}
    )
    monkeypatch.setattr(
        prepare_model_registry, "build_champion_manifest", lambda *_: {"decisions": []}
    )
    with pytest.raises(SystemExit):
        prepare_model_registry.main(["--execute"])
    assert prepare_model_registry.main([]) == 0
    assert (tmp_path / "day5c" / "champion_selection.json").is_file()

    def unavailable():
        raise ConnectionError("secret-value")

    monkeypatch.setattr(
        prepare_model_registry, "_connect_to_project",
        lambda: SimpleNamespace(get_model_registry=unavailable),
    )
    assert prepare_model_registry.main(["--check-remote"]) == 1
    result = capsys.readouterr()
    assert "ConnectionError" in result.err
    assert "secret-value" not in (result.err + result.out)
