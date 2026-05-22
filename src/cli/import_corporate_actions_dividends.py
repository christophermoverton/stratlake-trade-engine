from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.corporate_actions import import_dividend_events


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import local upstream corporate-action dividend files as StratLake event evidence. "
            "Consumes local Parquet/JSON artifacts only; no Alpaca calls, credentials, or network "
            "access are required. Dividend events are explicit event evidence, not adjusted price data."
        )
    )
    parser.add_argument("--source-data", required=True, help="Local upstream dividends.parquet path.")
    parser.add_argument("--source-metadata", required=True, help="Local upstream metadata.json path.")
    parser.add_argument("--output-root", required=True, help="Curated dividend event dataset root.")
    parser.add_argument("--artifact-root", default="artifacts/corporate_actions", help="Import artifact root.")
    parser.add_argument("--start", required=True, help="Inclusive ex_date window start, YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="Exclusive ex_date window end, YYYY-MM-DD.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Reject invalid or duplicate dividend event rows instead of advisory filtering.",
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="Run advisory mode: report invalid/duplicate rows and keep deterministic valid output.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    if args.strict and args.advisory:
        raise SystemExit("--strict and --advisory are mutually exclusive.")

    result = import_dividend_events(
        source_data_path=args.source_data,
        source_metadata_path=args.source_metadata,
        output_root=args.output_root,
        artifact_root=args.artifact_root,
        start=args.start,
        end=args.end,
        strict=not args.advisory,
    )
    payload = {
        "artifact_path": result.artifact_path,
        "output_root": result.output_root,
        "qa_status": _qa_status(Path(args.artifact_root) / result.run_id),
        "run_id": result.run_id,
        "written_row_count": result.written_row_count,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _qa_status(artifact_path: Path) -> str | None:
    qa_path = artifact_path / "qa_summary.json"
    if not qa_path.exists():
        return None
    return json.loads(qa_path.read_text(encoding="utf-8")).get("qa_status")


def main(argv: Sequence[str] | None = None) -> int:
    run_cli(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
