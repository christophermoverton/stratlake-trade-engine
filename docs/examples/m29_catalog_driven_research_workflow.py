"""Notebook-friendly M29 catalog exploration workflow.

This example is intentionally read-only. It scans existing artifact metadata,
queries in-memory catalog records, inspects lineage relationships, validates
catalog integrity, and returns artifact paths for follow-up analysis without
running research workflows or writing files.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.catalog import (
    CatalogQuery,
    build_catalog,
    get_downstream_records,
    get_upstream_records,
    query_catalog,
    records_to_dicts,
    records_to_rows,
    validate_catalog,
)
from src.catalog.models import CatalogRecord


def load_catalog(
    artifacts_root: str = "artifacts",
    repo_root: str = ".",
) -> list[CatalogRecord]:
    """Build the unified catalog from existing artifacts."""
    return build_catalog(artifacts_root, repo_root=repo_root)


def find_completed_strategy_runs(records: Iterable[CatalogRecord]) -> list[CatalogRecord]:
    """Return completed strategy runs from the catalog."""
    return query_catalog(
        records,
        CatalogQuery(run_types=("strategy",), statuses=("completed",)),
    )


def find_candidate_portfolios(records: Iterable[CatalogRecord]) -> list[CatalogRecord]:
    """Return completed portfolio runs from the catalog."""
    return query_catalog(
        records,
        CatalogQuery(run_types=("portfolio",), statuses=("completed",)),
    )


def find_metric_filtered_records(records: Iterable[CatalogRecord]) -> tuple[str | None, list[CatalogRecord]]:
    """Demonstrate metric-based filtering when scalar metrics are available."""
    completed_records = query_catalog(
        records,
        CatalogQuery(
            run_types=("strategy", "portfolio", "alpha_evaluation"),
            statuses=("completed",),
        ),
    )
    metric_names = sorted(
        {
            name
            for record in completed_records
            for name, value in (record.metrics_summary or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    if not metric_names:
        return None, []

    metric_name = metric_names[0]
    numeric_values = sorted(
        float((record.metrics_summary or {})[metric_name])
        for record in completed_records
        if isinstance((record.metrics_summary or {}).get(metric_name), (int, float))
        and not isinstance((record.metrics_summary or {}).get(metric_name), bool)
    )
    threshold = numeric_values[0]
    matches = query_catalog(
        completed_records,
        CatalogQuery(min_metric=(metric_name, threshold)),
    )
    return metric_name, matches


def inspect_related_runs(
    record: CatalogRecord,
    records: Iterable[CatalogRecord],
    repo_root: str = ".",
) -> dict[str, list[dict[str, object]]]:
    """Return upstream and downstream catalog rows for one record."""
    record_list = list(records)
    upstream = get_upstream_records(record, record_list, repo_root=repo_root)
    downstream = get_downstream_records(record, record_list, repo_root=repo_root)
    return {
        "upstream": records_to_rows(upstream),
        "downstream": records_to_rows(downstream),
    }


def summarize_validation(
    records: Iterable[CatalogRecord],
    repo_root: str = ".",
) -> dict[str, object]:
    """Validate the catalog and return a deterministic summary."""
    report = validate_catalog(records, repo_root=repo_root)
    return {
        "total_records": report.total_records,
        "total_artifacts": report.total_artifacts,
        "error_count": report.error_count,
        "warning_count": report.warning_count,
        "by_code": report.summary.get("by_code", {}),
    }


def artifact_paths_for_follow_up(records: Iterable[CatalogRecord], limit: int = 3) -> list[str]:
    """Return relative artifact roots that can be loaded by later analysis cells."""
    paths = [
        row["artifact_root"]
        for row in records_to_rows(records)
        if isinstance(row.get("artifact_root"), str) and row["artifact_root"]
    ]
    return paths[:limit]


def _print_rows(label: str, records: Iterable[CatalogRecord], limit: int = 5) -> None:
    rows = records_to_rows(records)[:limit]
    print(f"{label}: {len(rows)} shown")
    if not rows:
        print("  no matching records")
        return
    for row in rows:
        print(
            "  "
            + " | ".join(
                str(row.get(key) or "")
                for key in ("run_id", "run_type", "status", "artifact_root")
            )
        )


def main() -> int:
    """Run a concise read-only catalog exploration from the repository root."""
    repo_root = "."
    artifacts_root = "artifacts"

    print("M29 Catalog-Driven Research Workflow")
    print("====================================")
    print(f"repo_root: {repo_root}")
    print(f"artifacts_root: {artifacts_root}")

    records = load_catalog(artifacts_root=artifacts_root, repo_root=repo_root)
    print(f"catalog_records: {len(records)}")
    if not Path(artifacts_root).exists():
        print("artifact_state: artifacts directory not found; continuing with an empty catalog")
    elif not records:
        print("artifact_state: no catalog records discovered")
    else:
        print("artifact_state: catalog records discovered")

    strategies = find_completed_strategy_runs(records)
    portfolios = find_candidate_portfolios(records)
    _print_rows("completed_strategy_runs", strategies)
    _print_rows("completed_portfolio_runs", portfolios)

    metric_name, metric_matches = find_metric_filtered_records(records)
    if metric_name is None:
        print("metric_filter: skipped; no scalar metrics available")
    else:
        print(f"metric_filter: {metric_name} matched {len(metric_matches)} records")

    lineage_target = next(iter(portfolios or strategies or records), None)
    if lineage_target is None:
        print("lineage: skipped; no records available")
    else:
        related = inspect_related_runs(lineage_target, records, repo_root=repo_root)
        print(
            "lineage: "
            f"target={lineage_target.run_id or lineage_target.catalog_id} "
            f"upstream={len(related['upstream'])} downstream={len(related['downstream'])}"
        )

    validation_summary = summarize_validation(records, repo_root=repo_root)
    print(
        "validation: "
        f"records={validation_summary['total_records']} "
        f"artifacts={validation_summary['total_artifacts']} "
        f"errors={validation_summary['error_count']} "
        f"warnings={validation_summary['warning_count']}"
    )

    reusable_paths = artifact_paths_for_follow_up(records)
    if reusable_paths:
        print("artifact_paths:")
        for path in reusable_paths:
            print(f"  {path}")
    else:
        print("artifact_paths: none available")

    sample_dicts = records_to_dicts(records[:1])
    print(f"records_to_dicts_sample_count: {len(sample_dicts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
