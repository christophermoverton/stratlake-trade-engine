from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.catalog import (
    CatalogQuery,
    build_catalog,
    query_catalog,
    records_to_dicts,
    records_to_rows,
    related_records,
    summarize_catalog,
)
from src.catalog.models import CatalogRecord

_TABLE_COLUMNS = (
    "catalog_id",
    "run_id",
    "run_type",
    "status",
    "artifact_root",
    "strategy_name",
    "portfolio_name",
    "allocator_name",
    "alpha_model_name",
    "timeframe",
    "start_ts",
    "end_ts",
    "review_status",
    "promotion_status",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the read-only M29 unified research catalog."
    )
    parser.add_argument("--artifacts-root", default="artifacts", help="Artifact root to scan. Defaults to artifacts.")
    parser.add_argument("--repo-root", default=".", help="Repository root for relative catalog paths.")
    parser.add_argument("--run-type", action="append", dest="run_types", help="Run type filter. May be repeated.")
    parser.add_argument("--status", action="append", dest="statuses", help="Status filter. May be repeated.")
    parser.add_argument("--strategy-name", help="Exact strategy name filter.")
    parser.add_argument("--portfolio-name", help="Exact portfolio name filter.")
    parser.add_argument("--allocator-name", help="Exact allocator name filter.")
    parser.add_argument("--alpha-model-name", help="Exact alpha model name filter.")
    parser.add_argument("--regime-method", help="Exact regime method filter.")
    parser.add_argument("--campaign-id", help="Exact campaign id filter.")
    parser.add_argument("--scenario-id", help="Exact scenario id filter.")
    parser.add_argument("--min-metric", nargs=2, metavar=("NAME", "VALUE"), help="Metric lower-bound filter.")
    parser.add_argument("--max-metric", nargs=2, metavar=("NAME", "VALUE"), help="Metric upper-bound filter.")
    parser.add_argument("--metric-equals", nargs=2, metavar=("NAME", "VALUE"), help="Metric equality filter.")
    parser.add_argument("--start-ts", help="Minimum record start_ts using lexicographic ISO-like comparison.")
    parser.add_argument("--end-ts", help="Maximum record end_ts using lexicographic ISO-like comparison.")
    parser.add_argument("--include-templates", action="store_true", help="Include portfolio_template records.")
    parser.add_argument(
        "--exclude-unknown",
        action="store_true",
        help="Exclude records with run_type=unknown or status=unknown.",
    )
    parser.add_argument("--format", choices=("json", "table"), default="table", help="Output format.")
    parser.add_argument("--summary", action="store_true", help="Print summary counts for matching records.")
    parser.add_argument("--limit", type=int, help="Maximum number of records to print.")
    parser.add_argument("--related", help="Find related records by run_id or catalog_id.")
    parser.add_argument(
        "--direction",
        choices=("upstream", "downstream", "both"),
        default="both",
        help="Related-record direction. Defaults to both.",
    )
    parser.add_argument("--edge-type", action="append", dest="edge_types", help="Lineage edge type. May be repeated.")
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> list[dict[str, object]] | dict[str, object]:
    args = parse_args(argv)
    repo_root = Path(args.repo_root)
    artifacts_root = Path(args.artifacts_root)
    if not artifacts_root.is_absolute():
        artifacts_root = repo_root / artifacts_root

    query = CatalogQuery(
        run_types=tuple(args.run_types) if args.run_types else None,
        statuses=tuple(args.statuses) if args.statuses else None,
        strategy_name=args.strategy_name,
        portfolio_name=args.portfolio_name,
        allocator_name=args.allocator_name,
        alpha_model_name=args.alpha_model_name,
        regime_method=args.regime_method,
        campaign_id=args.campaign_id,
        scenario_id=args.scenario_id,
        min_metric=_metric_arg(args.min_metric),
        max_metric=_metric_arg(args.max_metric),
        metric_equals=_metric_arg(args.metric_equals),
        start_ts=args.start_ts,
        end_ts=args.end_ts,
        include_templates=args.include_templates,
        include_unknown=not args.exclude_unknown,
    )
    all_records = build_catalog(artifacts_root, repo_root=repo_root)
    records = query_catalog(all_records, query)

    if args.related:
        target = _find_record(records, args.related) or _find_record(all_records, args.related)
        if target is None:
            raise CatalogCliError(f"Related target not found: {args.related}")
        records = related_records(
            target,
            all_records,
            direction=args.direction,
            edge_types=args.edge_types,
            repo_root=repo_root,
        )

    if args.summary:
        payload = summarize_catalog(records)
        _print_json(payload)
        return payload

    if args.limit is not None:
        if args.limit < 0:
            raise CatalogCliError("--limit must be non-negative")
        records = records[: args.limit]

    if args.format == "json":
        payload = records_to_dicts(records)
        _print_json(payload)
        return payload

    payload = records_to_rows(records)
    _print_table(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_cli(argv)
    except CatalogCliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


class CatalogCliError(ValueError):
    """Expected user-facing query_catalog CLI error."""


def _metric_arg(raw: Sequence[str] | None) -> tuple[str, float] | None:
    if raw is None:
        return None
    name, value = raw
    try:
        return name, float(value)
    except ValueError as exc:
        raise CatalogCliError(f"Metric value must be numeric: {value}") from exc


def _find_record(records: Sequence[CatalogRecord], identifier: str) -> CatalogRecord | None:
    for record in records:
        if record.run_id == identifier or record.catalog_id == identifier:
            return record
    return None


def _print_json(payload: object) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2))


def _print_table(rows: Sequence[dict[str, object]]) -> None:
    print("\t".join(_TABLE_COLUMNS))
    for row in rows:
        print("\t".join(_format_cell(row.get(column)) for column in _TABLE_COLUMNS))


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
