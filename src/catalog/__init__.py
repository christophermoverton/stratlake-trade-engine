"""M29 Unified Catalog read-only artifact indexer and query package.

Usage::

    from src.catalog import build_catalog, query_catalog

    records = build_catalog("artifacts")
    for record in query_catalog(records):
        print(record.run_id, record.status)
"""

from src.catalog.indexer import build_artifact_records, build_catalog, build_catalog_record
from src.catalog.lineage import build_catalog_lookup, build_lineage_edges, build_run_lookup
from src.catalog.models import ArtifactRecord, CatalogRecord, CatalogValidationStatus, LineageEdge
from src.catalog.query import (
    CatalogQuery,
    filter_catalog_records,
    get_downstream_records,
    get_upstream_records,
    query_catalog,
    records_to_dicts,
    records_to_rows,
    related_records,
    summarize_catalog,
)
from src.catalog.validation import (
    CatalogValidationIssue,
    CatalogValidationReport,
    validate_artifact_records,
    validate_catalog,
    validate_record,
)

__all__ = [
    "build_catalog",
    "build_artifact_records",
    "build_catalog_record",
    "build_lineage_edges",
    "build_run_lookup",
    "build_catalog_lookup",
    "CatalogQuery",
    "query_catalog",
    "filter_catalog_records",
    "summarize_catalog",
    "related_records",
    "get_upstream_records",
    "get_downstream_records",
    "records_to_dicts",
    "records_to_rows",
    "CatalogRecord",
    "ArtifactRecord",
    "LineageEdge",
    "CatalogValidationStatus",
    "CatalogValidationIssue",
    "CatalogValidationReport",
    "validate_catalog",
    "validate_record",
    "validate_artifact_records",
]
