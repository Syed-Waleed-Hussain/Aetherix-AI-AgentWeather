"""Verify read-only access to the configured Hopsworks project."""

import logging
import os
from collections.abc import Callable
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from importlib.metadata import PackageNotFoundError, version
from io import StringIO
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _quiet_sdk_output() -> Iterator[None]:
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            yield
    finally:
        logging.disable(previous_disable_level)


def _sdk_version() -> str:
    try:
        return version("hopsworks")
    except PackageNotFoundError:
        return "not installed"


def _probe_service(
    name: str, accessor: Callable[[], object] | None
) -> tuple[str, bool]:
    if accessor is None:
        return f"{name}: unsupported by this SDK", True
    try:
        with _quiet_sdk_output():
            accessor()
    except Exception as exc:
        return f"{name}: unavailable ({type(exc).__name__})", False
    return f"{name}: available", True


def main() -> int:
    from dotenv import load_dotenv

    with _quiet_sdk_output():
        load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.environ.get("HOPSWORKS_API_KEY", "")
    project_name = os.environ.get("HOPSWORKS_PROJECT", "").strip()
    host = os.environ.get("HOPSWORKS_HOST", "").strip()
    configuration = {
        "HOPSWORKS_API_KEY": api_key.strip(),
        "HOPSWORKS_PROJECT": project_name,
        "HOPSWORKS_HOST": host,
    }
    missing = [name for name, value in configuration.items() if not value]

    print(f"Project: {project_name or 'not configured'}")
    print(f"SDK version: {_sdk_version()}")
    if missing:
        print(f"Connection status: FAILED (missing configuration: {', '.join(missing)})")
        print("Available services: not checked")
        return 1

    logging.disable(logging.CRITICAL)
    try:
        import hopsworks

        with _quiet_sdk_output():
            project = hopsworks.login(
                project=project_name,
                api_key_value=api_key,
                host=host,
                engine="python",
            )
            selected_project = project.name
    except Exception as exc:
        print(f"Connection status: FAILED (login: {type(exc).__name__})")
        print("Available services: not checked")
        return 1

    if selected_project != project_name:
        print("Connection status: FAILED (selected project does not match configuration)")
        print("Available services: not checked")
        return 1

    feature_store_status, feature_store_ok = _probe_service(
        "Feature Store", project.get_feature_store
    )
    model_registry_status, model_registry_ok = _probe_service(
        "Model Registry", getattr(project, "get_model_registry", None)
    )
    print("Connection status: CONNECTED (project verified)")
    print(f"Available services: {feature_store_status}; {model_registry_status}")
    return 0 if feature_store_ok and model_registry_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
