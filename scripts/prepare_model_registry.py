"""Verify the local LSTM and optionally synchronize it with Hopsworks Model Registry."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from aetherix_weather.champions import build_champion_manifest
from aetherix_weather.hopsworks_features import verify_secret_ignored
from aetherix_weather.model_registry import (
    registration_metadata,
    sync_registry_candidate,
    verify_temperature_checkpoint,
)
from scripts.sync_hopsworks_features import (
    SyncOperationError,
    _connect_to_project,
    _quiet_sdk_call,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def _verify_with_local_torch(checkpoint: Path, training: Path, digest: str) -> None:
    venv = PROJECT_ROOT / ".venv"
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if Path(sys.executable).resolve() == python.absolute():
        verify_temperature_checkpoint(checkpoint, training, digest)
        return
    result = subprocess.run(
        [
            str(python), "-c",
            "from pathlib import Path; import sys; "
            "from aetherix_weather.model_registry import verify_temperature_checkpoint; "
            "verify_temperature_checkpoint(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])",
            str(checkpoint), str(training), digest,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError("Checkpoint validation failed in the existing PyTorch environment.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-remote", action="store_true", help="Read-only duplicate check.")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-write", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=ARTIFACTS / "day5c" / "temperature_lstm_registry_metadata.json",
    )
    args = parser.parse_args(argv)
    if args.execute != args.confirm_write:
        parser.error("--execute and --confirm-write must be used together.")

    checkpoint = ARTIFACTS / "day4b" / "lstm_checkpoint.pt"
    training = ARTIFACTS / "day4b" / "lstm_metadata.json"
    dataset = PROJECT_ROOT / "data" / "processed" / "features.csv"
    manifest_path = ARTIFACTS / "day5c" / "champion_selection.json"
    try:
        verify_secret_ignored(PROJECT_ROOT)
        metadata = registration_metadata(dataset, training, checkpoint)
        with TemporaryDirectory() as directory:
            package = Path(directory)
            staged_checkpoint = package / "lstm_checkpoint.pt"
            shutil.copyfile(checkpoint, staged_checkpoint)
            digest = metadata["artifact"]["sha256"]
            _verify_with_local_torch(staged_checkpoint, training, digest)
            _write_json(package / "registry_metadata.json", metadata)
            _write_json(args.output, metadata)
            manifest = build_champion_manifest(
                ARTIFACTS / "day3_5" / "selected_models.json", ARTIFACTS / "day4a", metadata
            )
            _write_json(manifest_path, manifest)
            print(f"Verified local LSTM checkpoint, SHA-256: {digest}")
            print(f"Registry metadata: {args.output}")
            print(f"Champion manifest: {manifest_path} ({len(manifest['decisions'])} decisions)")
            if args.check_remote or args.execute:
                project = _connect_to_project()
                try:
                    registry = _quiet_sdk_call(project.get_model_registry)
                    version, uploaded = _quiet_sdk_call(
                        lambda: sync_registry_candidate(
                            registry, package, metadata, execute=args.execute
                        )
                    )
                except Exception as exc:
                    raise SyncOperationError(
                        f"Model Registry operation failed ({type(exc).__name__}). "
                        "Run --check-remote before retrying a write."
                    ) from None
                if version is None:
                    print("Model Registry: no equivalent version found.")
                else:
                    operation = "registered and verified" if uploaded else "existing version verified"
                    print(f"Model Registry: v{version} {operation}.")
            else:
                print("Model Registry: local preflight only.")
    except SyncOperationError as exc:
        print(f"Cloud preflight or registration failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Local registry preparation failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
