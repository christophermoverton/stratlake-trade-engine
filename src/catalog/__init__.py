"""M29 Unified Catalog — read-only artifact indexer package.

Usage::

    from src.catalog import build_catalog, CatalogRecord, ArtifactRecord

    records = build_catalog("artifacts")
    for record in records:
        print(record.run_id, record.status)
"""

from src.catalog.indexer import build_catalog, build_artifact_records, build_catalog_record
from src.catalog.models import ArtifactRecord, CatalogRecord, CatalogValidationStatus

__all__ = [
    "build_catalog",
    "build_artifact_records",
    "build_catalog_record",
    "CatalogRecord",
    "ArtifactRecord",
    "CatalogValidationStatus",
]
