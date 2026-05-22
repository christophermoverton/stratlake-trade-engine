"""Optional M40 cross-repository dividend smoke workflow helper.

This helper validates the StratLake side of the manual cross-repo handoff:
local upstream dividend artifacts in, curated dividend event evidence and
catalog discovery out. It does not call Alpaca, shell out to
``fintech-ingest-corporate-actions``, read credentials, or require the upstream
package as a dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.catalog import CatalogQuery, build_catalog, query_catalog
from src.corporate_actions import import_dividend_events, load_dividend_events


def run_cross_repo_q1_dividend_smoke_workflow(
    *,
    source_data: str | Path,
    source_metadata: str | Path,
    output_root: str | Path,
    artifact_root: str | Path,
    start: str,
    end: str,
    symbol: str = "AAPL",
    strict: bool = True,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Import explicit local dividend artifacts and verify catalog discovery."""

    result = import_dividend_events(
        source_data_path=source_data,
        source_metadata_path=source_metadata,
        output_root=output_root,
        artifact_root=artifact_root,
        start=start,
        end=end,
        strict=strict,
    )
    loaded = load_dividend_events(output_root)
    qa_summary_path = Path(artifact_root) / result.run_id / "qa_summary.json"
    qa_summary = json.loads(qa_summary_path.read_text(encoding="utf-8"))

    catalog_artifacts_root = _catalog_artifacts_root(Path(artifact_root))
    records = build_catalog(catalog_artifacts_root, repo_root=repo_root)
    dividend_records = query_catalog(records, CatalogQuery(evidence_type="dividend_events"))
    matching_records = [record for record in dividend_records if record.run_id == result.run_id]

    normalized_symbol = symbol.strip().upper()
    symbol_rows = 0
    if "symbol" in loaded.columns:
        symbol_rows = int((loaded["symbol"].astype("string").str.upper() == normalized_symbol).sum())

    return {
        "artifact_path": result.artifact_path,
        "catalog_match_count": len(matching_records),
        "catalog_run_ids": sorted(record.run_id for record in dividend_records),
        "dataset_root": result.output_root,
        "end": result.end,
        "loaded_row_count": int(len(loaded)),
        "qa_status": qa_summary["qa_status"],
        "run_id": result.run_id,
        "start": result.start,
        "strict": strict,
        "symbol": normalized_symbol,
        "symbol_row_count": symbol_rows,
        "written_row_count": result.written_row_count,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the StratLake side of the optional M40 cross-repo Q1 dividend smoke workflow. "
            "Consumes explicit local upstream files only; it does not call Alpaca, read credentials, "
            "use network access, or shell out to fintech-ingest-corporate-actions."
        )
    )
    parser.add_argument("--source-data", required=True, help="Explicit local upstream dividends.parquet path.")
    parser.add_argument("--source-metadata", required=True, help="Explicit local upstream metadata.json path.")
    parser.add_argument("--output-root", required=True, help="Curated StratLake dividend event dataset root.")
    parser.add_argument("--artifact-root", required=True, help="Dividend import artifact root.")
    parser.add_argument("--start", default="2024-01-01", help="Inclusive ex_date window start, YYYY-MM-DD.")
    parser.add_argument("--end", default="2024-04-01", help="Exclusive ex_date window end, YYYY-MM-DD.")
    parser.add_argument("--symbol", default="AAPL", help="Expected smoke symbol for summary reporting.")
    parser.add_argument("--repo-root", default=".", help="Repository root used for catalog path normalization.")
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="Run importer advisory mode instead of strict duplicate/validation rejection.",
    )
    return parser.parse_args(argv)


def _catalog_artifacts_root(artifact_root: Path) -> Path:
    if artifact_root.name == "corporate_actions" and artifact_root.parent.name == "artifacts":
        return artifact_root.parent
    return artifact_root


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_cross_repo_q1_dividend_smoke_workflow(
        source_data=args.source_data,
        source_metadata=args.source_metadata,
        output_root=args.output_root,
        artifact_root=args.artifact_root,
        start=args.start,
        end=args.end,
        symbol=args.symbol,
        strict=not args.advisory,
        repo_root=args.repo_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
