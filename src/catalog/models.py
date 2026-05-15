from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CatalogValidationStatus:
    """Validation status for a catalog record."""

    catalog_status: str
    marker_status: str
    manifest_status: str
    artifact_status: str
    qa_status: str | None
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_status": self.catalog_status,
            "marker_status": self.marker_status,
            "manifest_status": self.manifest_status,
            "artifact_status": self.artifact_status,
            "qa_status": self.qa_status,
            "validation_errors": list(self.validation_errors),
            "validation_warnings": list(self.validation_warnings),
        }


@dataclass
class ArtifactRecord:
    """A single artifact file discovered under a catalog record's artifact root."""

    artifact_id: str
    catalog_id: str
    run_id: str | None
    artifact_type: str
    path: str
    relative_path: str
    filename: str
    extension: str
    declared_in_manifest: bool
    exists: bool
    size_bytes: int | None
    modified_time: str | None
    checksum_optional: str | None
    schema_hint: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "catalog_id": self.catalog_id,
            "run_id": self.run_id,
            "artifact_type": self.artifact_type,
            "path": self.path,
            "relative_path": self.relative_path,
            "filename": self.filename,
            "extension": self.extension,
            "declared_in_manifest": self.declared_in_manifest,
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "modified_time": self.modified_time,
            "checksum_optional": self.checksum_optional,
            "schema_hint": self.schema_hint,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LineageEdge:
    """A deterministic, read-only relationship derived from catalog metadata."""

    edge_id: str
    source_catalog_id: str | None
    target_catalog_id: str | None
    source_run_id: str | None
    target_run_id: str | None
    edge_type: str
    relationship_source: str
    relationship_path: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_catalog_id": self.source_catalog_id,
            "target_catalog_id": self.target_catalog_id,
            "source_run_id": self.source_run_id,
            "target_run_id": self.target_run_id,
            "edge_type": self.edge_type,
            "relationship_source": self.relationship_source,
            "relationship_path": self.relationship_path,
            "metadata": dict(self.metadata),
        }


@dataclass
class CatalogRecord:
    """A normalized in-memory catalog record for one artifact root."""

    catalog_id: str
    run_id: str | None
    run_type: str
    status: str
    artifact_root: str
    source_registry_path: str | None
    source_manifest_path: str | None
    source_marker_path: str | None
    created_at: str | None
    timeframe: str | None
    start_ts: str | None
    end_ts: str | None
    strategy_name: str | None
    portfolio_name: str | None
    allocator_name: str | None
    alpha_model_name: str | None
    regime_method: str | None
    campaign_id: str | None
    scenario_id: str | None
    metrics_summary: dict[str, Any] | None
    qa_status: str | None
    review_status: str | None
    promotion_status: str | None
    record_family: str | None = None
    robustness_status: str | None = None
    wfe_status: str | None = None
    sample_size_status: str | None = None
    trade_count_status: str | None = None
    sensitivity_status: str | None = None
    fragility_status: str | None = None
    multiple_testing_status: str | None = None
    temporal_validation_status: str | None = None
    governance_status: str | None = None
    promotion_review_status: str | None = None
    validation_readiness_present: bool = False
    release_validation_present: bool = False
    tags: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    validation: CatalogValidationStatus = field(
        default_factory=lambda: CatalogValidationStatus(
            catalog_status="unknown",
            marker_status="unknown",
            manifest_status="unknown",
            artifact_status="unknown",
            qa_status=None,
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "run_id": self.run_id,
            "run_type": self.run_type,
            "status": self.status,
            "artifact_root": self.artifact_root,
            "source_registry_path": self.source_registry_path,
            "source_manifest_path": self.source_manifest_path,
            "source_marker_path": self.source_marker_path,
            "created_at": self.created_at,
            "timeframe": self.timeframe,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "strategy_name": self.strategy_name,
            "portfolio_name": self.portfolio_name,
            "allocator_name": self.allocator_name,
            "alpha_model_name": self.alpha_model_name,
            "regime_method": self.regime_method,
            "campaign_id": self.campaign_id,
            "scenario_id": self.scenario_id,
            "metrics_summary": dict(self.metrics_summary) if self.metrics_summary is not None else None,
            "qa_status": self.qa_status,
            "review_status": self.review_status,
            "promotion_status": self.promotion_status,
            "record_family": self.record_family,
            "robustness_status": self.robustness_status,
            "wfe_status": self.wfe_status,
            "sample_size_status": self.sample_size_status,
            "trade_count_status": self.trade_count_status,
            "sensitivity_status": self.sensitivity_status,
            "fragility_status": self.fragility_status,
            "multiple_testing_status": self.multiple_testing_status,
            "temporal_validation_status": self.temporal_validation_status,
            "governance_status": self.governance_status,
            "promotion_review_status": self.promotion_review_status,
            "validation_readiness_present": self.validation_readiness_present,
            "release_validation_present": self.release_validation_present,
            "tags": list(self.tags),
            "source_files": list(self.source_files),
            "metadata": dict(self.metadata),
            "validation": self.validation.to_dict(),
        }
