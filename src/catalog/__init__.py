"""M29 Unified Catalog read-only artifact indexer and query package.

Usage::

    from src.catalog import build_catalog, query_catalog

    records = build_catalog("artifacts")
    for record in query_catalog(records):
        print(record.run_id, record.status)
"""

from src.catalog.indexer import build_artifact_records, build_catalog, build_catalog_record
from src.catalog.canonicality import (
    CANONICALITY_SCHEMA_VERSION,
    RESOLVER_HINT,
    build_canonicality_envelope,
    canonical_authority_paths,
    canonicality_status,
    portable_path,
    validate_portable_repository_path,
)
from src.catalog.derived_index import (
    DEFAULT_DERIVED_INDEX_PATH,
    DerivedIndexError,
    DerivedIndexValidation,
    build_derived_index,
    load_catalog_records,
    load_catalog_records_with_source,
    validate_derived_index,
)
from src.catalog.load_source import (
    LOAD_SOURCE_SCHEMA_VERSION,
    CatalogLoadResult,
    build_load_source,
    derive_view_load_source,
)
from src.catalog.explorer import (
    build_evidence_explorer_view,
    render_evidence_json,
    render_evidence_markdown,
    render_evidence_table,
)
from src.catalog.lineage import build_catalog_lookup, build_lineage_edges, build_run_lookup
from src.catalog.lineage_export import (
    LineageExportError,
    export_lineage,
    export_lineage_openlineage,
    export_lineage_prov,
    validate_lineage_export,
)
from src.catalog.lineage_fingerprints import (
    build_dataset_lineage,
    build_feature_lineage,
    dataset_schema_fingerprint,
    feature_columns_fingerprint,
    portable_dataset_path,
    stable_json_fingerprint,
)
from src.catalog.models import ArtifactRecord, CatalogRecord, CatalogValidationStatus, LineageEdge
from src.catalog.notebook import (
    build_notebook_evidence_view,
    evidence_for_run,
    evidence_lineage_rows,
    find_governance_evidence,
    find_release_evidence,
    find_robustness_evidence,
    find_validation_evidence,
    render_notebook_json,
    render_notebook_markdown,
    render_notebook_table,
    summarize_evidence_for_run,
)
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
from src.catalog.resolver import (
    CanonicalRecordResolution,
    ResolvedSource,
    resolve_canonical_record,
    resolve_canonical_record_by_id,
    resolve_canonical_sources,
)
from src.catalog.validation import (
    CatalogValidationIssue,
    CatalogValidationReport,
    validate_artifact_records,
    validate_catalog,
    validate_record,
)
from src.catalog.workflows import (
    build_evidence_view_for_workflow,
    build_lineage_export_for_workflow,
    load_catalog_for_workflow,
    resolve_workflow_roots,
)

__all__ = [
    "build_catalog",
    "build_canonicality_envelope",
    "build_derived_index",
    "build_evidence_explorer_view",
    "build_artifact_records",
    "build_catalog_record",
    "load_catalog_records",
    "load_catalog_records_with_source",
    "validate_derived_index",
    "build_lineage_edges",
    "export_lineage",
    "export_lineage_openlineage",
    "export_lineage_prov",
    "stable_json_fingerprint",
    "feature_columns_fingerprint",
    "dataset_schema_fingerprint",
    "portable_dataset_path",
    "build_dataset_lineage",
    "build_feature_lineage",
    "build_run_lookup",
    "build_catalog_lookup",
    "CatalogQuery",
    "build_notebook_evidence_view",
    "load_catalog_for_workflow",
    "build_lineage_export_for_workflow",
    "build_evidence_view_for_workflow",
    "resolve_workflow_roots",
    "evidence_for_run",
    "evidence_lineage_rows",
    "find_governance_evidence",
    "find_release_evidence",
    "find_robustness_evidence",
    "find_validation_evidence",
    "query_catalog",
    "filter_catalog_records",
    "summarize_catalog",
    "related_records",
    "get_upstream_records",
    "get_downstream_records",
    "records_to_dicts",
    "records_to_rows",
    "render_evidence_json",
    "render_evidence_markdown",
    "render_evidence_table",
    "render_notebook_json",
    "render_notebook_markdown",
    "render_notebook_table",
    "summarize_evidence_for_run",
    "CatalogRecord",
    "ArtifactRecord",
    "LineageEdge",
    "LineageExportError",
    "CatalogValidationStatus",
    "CatalogValidationIssue",
    "CatalogValidationReport",
    "DerivedIndexError",
    "DerivedIndexValidation",
    "CatalogLoadResult",
    "build_load_source",
    "derive_view_load_source",
    "LOAD_SOURCE_SCHEMA_VERSION",
    "DEFAULT_DERIVED_INDEX_PATH",
    "CanonicalRecordResolution",
    "ResolvedSource",
    "resolve_canonical_record",
    "resolve_canonical_record_by_id",
    "resolve_canonical_sources",
    "validate_catalog",
    "validate_record",
    "validate_artifact_records",
    "validate_lineage_export",
    "canonical_authority_paths",
    "canonicality_status",
    "portable_path",
    "validate_portable_repository_path",
    "CANONICALITY_SCHEMA_VERSION",
    "RESOLVER_HINT",
]
