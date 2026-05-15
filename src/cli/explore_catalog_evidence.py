from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from src.catalog import (
    CatalogQuery,
    build_catalog,
    build_evidence_explorer_view,
    render_evidence_json,
    render_evidence_markdown,
    render_evidence_table,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a local read-only M35 catalog evidence explorer view."
    )
    parser.add_argument("--artifacts-root", default="artifacts", help="Artifact root to scan. Defaults to artifacts.")
    parser.add_argument("--repo-root", default=".", help="Repository root for relative catalog paths.")
    parser.add_argument("--run-id", help="Selected run_id to review with related evidence.")
    parser.add_argument("--catalog-id", help="Selected catalog_id to review with related evidence.")
    parser.add_argument("--record-family", help="Exact M35 evidence record-family filter.")
    parser.add_argument("--robustness-status", help="Exact robustness_status filter.")
    parser.add_argument("--governance-status", help="Exact governance_status filter.")
    parser.add_argument(
        "--validation-readiness-present",
        type=_bool_arg,
        metavar="true|false",
        help="Filter by validation_readiness_present.",
    )
    parser.add_argument(
        "--release-validation-present",
        type=_bool_arg,
        metavar="true|false",
        help="Filter by release_validation_present.",
    )
    parser.add_argument(
        "--include-lineage",
        type=_bool_arg,
        default=True,
        metavar="true|false",
        help="Include evidence lineage edges. Defaults to true.",
    )
    parser.add_argument("--format", choices=("markdown", "json", "table"), default="markdown", help="Output format.")
    parser.add_argument("--output", help="Optional path for derived review output.")
    parser.add_argument("--limit", type=int, help="Maximum number of matching records to render.")
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    repo_root = Path(args.repo_root)
    artifacts_root = Path(args.artifacts_root)
    if not artifacts_root.is_absolute():
        artifacts_root = repo_root / artifacts_root

    query = CatalogQuery(
        record_family=args.record_family,
        robustness_status=args.robustness_status,
        governance_status=args.governance_status,
        validation_readiness_present=args.validation_readiness_present,
        release_validation_present=args.release_validation_present,
    )
    records = build_catalog(artifacts_root, repo_root=repo_root)
    view = build_evidence_explorer_view(
        records,
        query=query,
        selected_run_id=args.run_id,
        selected_catalog_id=args.catalog_id,
        include_lineage=args.include_lineage,
        repo_root=repo_root,
        limit=args.limit,
    )
    rendered = _render(view, args.format)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8", newline="\n")
    else:
        print(rendered, end="")
    return view


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_cli(argv)
    except (CatalogExplorerCliError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


class CatalogExplorerCliError(ValueError):
    """Expected user-facing catalog evidence explorer CLI error."""


def _render(view: dict[str, object], output_format: str) -> str:
    if output_format == "json":
        return render_evidence_json(view)
    if output_format == "table":
        return render_evidence_table(view)
    return render_evidence_markdown(view)


def _bool_arg(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got: {raw}")


if __name__ == "__main__":
    raise SystemExit(main())
