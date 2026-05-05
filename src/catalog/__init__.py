"""M29 Unified Catalog — read-only artifact indexer package.

Usage::

    from src.catalog import build_catalog, build_lineage_edges, CatalogRecord, ArtifactRecord

    records = build_catalog("artifacts")
    for record in records:
        print(record.run_id, record.status)
"""

from src.catalog.indexer import build_catalog, build_artifact_records, build_catalog_record
from src.catalog.lineage import build_catalog_lookup, build_lineage_edges, build_run_lookup
from src.catalog.models import ArtifactRecord, CatalogRecord, CatalogValidationStatus, LineageEdge

__all__ = [
    "build_catalog",
    "build_artifact_records",
    "build_catalog_record",
    "build_lineage_edges",
    "build_run_lookup",
    "build_catalog_lookup",
    "CatalogRecord",
    "ArtifactRecord",
    "LineageEdge",
    "CatalogValidationStatus",
]
