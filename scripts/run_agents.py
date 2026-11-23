"""Run or replay the local AgentWeather LangGraph workflow."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys
from uuid import UUID

from aetherix_weather.agent_forecast import InputError
from aetherix_weather.agent_reporting import AUDIENCES, groq_report_layout
from aetherix_weather.agent_workflow import DecisionStore, run_workflow
from aetherix_weather.storage import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "outputs/agents/decisions.sqlite3"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="karachi")
    parser.add_argument("--horizon", type=int, choices=range(1, 25), default=24)
    parser.add_argument("--issue", help="UTC-offset historical issue hour (default: latest local hour).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true", help="Offline fixture; no model inference.")
    mode.add_argument("--replay", metavar="REQUEST_ID", type=UUID, help="Replay persisted decisions without inference.")
    mode.add_argument("--groq-free", action="store_true", help="Opt in using an existing free-tier Groq key.")
    parser.add_argument("--audience", choices=AUDIENCES, default="general")
    parser.add_argument("--thresholds", type=Path, help="JSON file with alert threshold overrides.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--json", action="store_true", help="Print the final state as JSON.")
    parser.add_argument("--output", type=Path, help="Also save the final state as JSON.")
    args = parser.parse_args(argv)
    try:
        if args.output:
            output = args.output.resolve()
            protected = (ROOT / "data", ROOT / "artifacts")
            if (
                output == args.db.resolve() or output.name.lower().startswith(".env")
                or any(output.is_relative_to(folder.resolve()) for folder in protected)
            ):
                raise InputError("JSON output must be separate from project inputs and the decision database.")
        store = DecisionStore(args.db)
        if args.replay is not None:
            snapshots = store.replay(str(args.replay))
            if not snapshots:
                print("AgentWeather replay found no matching request.", file=sys.stderr)
                return 2
            state = snapshots[-1]
        else:
            thresholds = (json.loads(args.thresholds.read_text(encoding="utf-8"))
                          if args.thresholds else {})
            if not isinstance(thresholds, dict):
                raise InputError("Alert threshold overrides must be a JSON object.")
            state = run_workflow(
                ROOT, store, city=args.city, horizon=args.horizon, issue=args.issue,
                synthetic=args.synthetic, audience=args.audience, thresholds=thresholds,
                provider=groq_report_layout if args.groq_free else None,
            )
        if args.output:
            write_json_atomic(args.output, state)
        if args.json:
            print(json.dumps(state, ensure_ascii=True, allow_nan=False))
        else:
            print(state["report"].get("text", "Report unavailable."))
            print(f"Drift: {state['drift'].get('status', 'unavailable')}")
            print(f"Request ID: {state['request_id']}")
            print(f"Recorded steps: {len(state['decisions'])}; completed: {bool(state['completed_at_utc'])}")
        return int(not state["completed_at_utc"] or state["statuses"].get("forecast") == "failed")
    except InputError as exc:
        print(f"AgentWeather input error: {exc}", file=sys.stderr)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"AgentWeather run failed ({type(exc).__name__}).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
