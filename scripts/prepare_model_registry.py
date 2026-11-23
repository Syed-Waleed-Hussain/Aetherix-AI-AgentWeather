"""Prepare local registry metadata for the verified Day 4B temperature LSTM."""

import argparse
import json
import sys
from pathlib import Path

from aetherix_weather.model_registry import prepare_temperature_lstm_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "day5b"
        / "temperature_lstm_registry_metadata.json",
        help="Local output path; no Hopsworks Model Registry calls are made.",
    )
    args = parser.parse_args(argv)
    try:
        metadata = prepare_temperature_lstm_metadata(
            PROJECT_ROOT / "data" / "processed" / "features.csv",
            PROJECT_ROOT / "artifacts" / "day4b" / "lstm_metadata.json",
            PROJECT_ROOT / "artifacts" / "day4b" / "lstm_checkpoint.pt",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Registry metadata preparation failed: {exc}", file=sys.stderr)
        return 1

    champion = metadata["champion"]
    print(f"Prepared local metadata: {args.output}")
    print(
        f"Validation champion candidate: {champion['model_name']} "
        f"({metadata['selection_metric']}={champion['validation_score']:.8f})"
    )
    print(
        "Model Registry: not contacted; model artifact was not uploaded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
