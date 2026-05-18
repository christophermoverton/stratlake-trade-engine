"""Static contract helpers for derived evidence review packs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable, Literal

from src.catalog.canonicality import (
    build_canonicality_envelope,
    canonical_authority_paths,
    portable_path,
    validate_portable_repository_path,
)
from src.catalog.derived_index import IndexMode, load_catalog_records_with_source
from src.catalog.lineage import build_lineage_edges
from src.catalog.lineage_export import export_lineage
from src.catalog.lineage_fingerprints import stable_json_fingerprint
from src.catalog.load_source import build_load_source, derive_view_load_source
from src.catalog.models import CatalogRecord
from src.catalog.query import records_to_dicts, related_records
from src.catalog.resolver import resolve_canonical_record, resolve_canonical_sources
from src.catalog.workflows import resolve_workflow_roots

REVIEW_PACK_SCHEMA_VERSION = "review_pack.v1"
DEFAULT_REVIEW_PACK_ROOT = "artifacts/_derived/evidence_review"
REQUIRED_REVIEW_PACK_FILES = (
    "manifest.json",
    "review_request.json",
    "review_summary.json",
    "catalog_health_diagnostics.json",
    "validation.json",
    "selected_record.json",
    "related_records.json",
    "resolver_resolution.json",
    "evidence_index.json",
    "artifact_inventory.csv",
    "report.md",
)
OPTIONAL_REVIEW_PACK_FILES = (
    "selected_lineage.openlineage.json",
    "selected_lineage.prov.json",
    "report.html",
)
EvidenceReviewLineageFormat = Literal["openlineage", "prov", "both"]


class EvidenceReviewError(ValueError):
    """Raised when a selected-run evidence review cannot be built deterministically."""


DiagnosticStatus = Literal["PASS", "WARN", "FAIL", "NA"]


def review_pack_root(review_id: str) -> str:
    """Return the deterministic repository-relative output root for one pack."""

    normalized_review_id = portable_path(review_id)
    validate_portable_repository_path(normalized_review_id)
    if "/" in normalized_review_id:
        raise ValueError("Review IDs must be single portable path segments.")
    return f"{DEFAULT_REVIEW_PACK_ROOT}/{normalized_review_id}"


def build_review_pack_metadata(
    *,
    authority_paths: Iterable[str | Path],
    fingerprint_payload: Any | None = None,
) -> dict[str, Any]:
    """Build shared non-authoritative metadata for review-pack payloads."""

    metadata = build_canonicality_envelope(
        derived_class="review_pack",
        authority_paths=authority_paths,
        fingerprint_payload=fingerprint_payload,
    )
    metadata["load_source"] = build_load_source(loaded_from="review_pack")
    return metadata


def build_evidence_review_for_workflow(
    artifacts_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    index_path: str | Path | None = None,
    index_mode: IndexMode = "direct",
    selected_run_id: str | None = None,
    selected_catalog_id: str | None = None,
    review_id: str | None = None,
    resolve_related: bool = False,
    lineage_format: EvidenceReviewLineageFormat = "both",
) -> dict[str, Any]:
    """Build one deterministic resolver-backed review model without writing files."""

    if lineage_format not in {"openlineage", "prov", "both"}:
        raise EvidenceReviewError(f"Unsupported lineage format: {lineage_format}")
    if selected_run_id is None and selected_catalog_id is None:
        raise EvidenceReviewError("selected_run_id or selected_catalog_id is required")

    resolved_artifacts, resolved_repo = resolve_workflow_roots(artifacts_root, repo_root=repo_root)
    load_result = load_catalog_records_with_source(
        resolved_artifacts,
        repo_root=resolved_repo,
        index_path=index_path,
        mode=index_mode,
    )
    selected = _select_subject(
        load_result.records,
        selected_run_id=selected_run_id,
        selected_catalog_id=selected_catalog_id,
    )
    related = related_records(selected, load_result.records, repo_root=resolved_repo)
    selected_resolution = resolve_canonical_record(
        selected,
        artifacts_root=resolved_artifacts,
        repo_root=resolved_repo,
    )
    related_resolutions = (
        resolve_canonical_sources(
            related,
            artifacts_root=resolved_artifacts,
            repo_root=resolved_repo,
        )
        if resolve_related
        else []
    )
    lineage_edges = build_lineage_edges(load_result.records, repo_root=resolved_repo)
    lineage_exports = _build_lineage_exports(
        load_result.records,
        lineage_edges,
        selected_run_id=selected.run_id,
        lineage_format=lineage_format,
        load_source=derive_view_load_source(load_result.load_source, loaded_from="lineage_export"),
    )
    resolved_review_id = review_id or _deterministic_review_id(
        selected_run_id=selected_run_id,
        selected_catalog_id=selected_catalog_id,
        index_mode=index_mode,
        index_path=load_result.load_source.get("index_path"),
        lineage_format=lineage_format,
        resolve_related=resolve_related,
    )
    metadata = build_canonicality_envelope(
        derived_class="review_pack",
        authority_root=_authority_root(resolved_artifacts, resolved_repo),
        authority_paths=canonical_authority_paths(load_result.records),
        fingerprint_payload={
            "review_id": resolved_review_id,
            "selected_catalog_id": selected.catalog_id,
            "selected_run_id": selected.run_id,
        },
    )
    review_load_source = derive_view_load_source(load_result.load_source, loaded_from="review_pack")
    warnings = sorted(
        {
            *selected_resolution.warnings,
            *(warning for resolution in related_resolutions for warning in resolution.warnings),
        }
    )
    source_fingerprints = {
        selected.catalog_id: selected_resolution.source_fingerprint,
        **{
            resolution.record.catalog_id: resolution.source_fingerprint
            for resolution in related_resolutions
        },
    }
    review_model = {
            "schema_version": REVIEW_PACK_SCHEMA_VERSION,
            "review_id": resolved_review_id,
            "review_root": review_pack_root(resolved_review_id),
            "review_request": {
                "schema_version": "review_request.v1",
                "review_id": resolved_review_id,
                "selected_run_id": selected.run_id,
                "selected_catalog_id": selected_catalog_id,
                "index_mode": index_mode,
                "resolve_related": resolve_related,
                "lineage_format": lineage_format,
            },
            "selected_record": selected.to_dict(),
            "related_records": records_to_dicts(related),
            "resolver_resolution": selected_resolution.to_dict(),
            "related_resolver_resolutions": [resolution.to_dict() for resolution in related_resolutions],
            "canonical_sources": list(selected_resolution.source_paths),
            "source_fingerprints": source_fingerprints,
            "warning_summary": {
                "count": len(warnings),
                "warnings": warnings,
                "missing_source_count": len(selected_resolution.missing_sources),
            },
            "load_source_summary": {
                "requested_mode": load_result.load_source.get("requested_mode"),
                "resolved_mode": load_result.load_source.get("resolved_mode"),
                "loaded_from": load_result.load_source.get("loaded_from"),
                "index_path": load_result.load_source.get("index_path"),
                "index_validated": load_result.load_source.get("index_validated"),
            },
            "lineage_summary": _lineage_summary(lineage_exports),
            "selected_lineage": lineage_exports,
            **metadata,
            "load_source": review_load_source,
        }
    review_model["catalog_health_diagnostics"] = build_catalog_health_diagnostics(
        review_model,
        records=load_result.records,
        repo_root=resolved_repo,
    )
    return _json_safe(review_model)


def build_catalog_health_diagnostics(
    review_model: dict[str, Any] | None = None,
    *,
    records: Sequence[CatalogRecord] | None = None,
    selected_record: CatalogRecord | Mapping[str, Any] | None = None,
    resolver_resolution: Mapping[str, Any] | None = None,
    related_records: Sequence[Any] | None = None,
    lineage_summary: Mapping[str, Any] | None = None,
    load_source_summary: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build deterministic advisory health diagnostics without mutating artifacts."""

    model = review_model or {}
    review_id = str(model.get("review_id") or "ad_hoc")
    selected_payload = selected_record or model.get("selected_record")
    resolver_payload = resolver_resolution or model.get("resolver_resolution") or {}
    related_payload = list(related_records or model.get("related_records") or [])
    lineage_payload = dict(lineage_summary or model.get("lineage_summary") or {})
    load_summary = dict(load_source_summary or model.get("load_source_summary") or {})
    canonicality = model.get("canonicality") if isinstance(model.get("canonicality"), Mapping) else {}
    load_source = model.get("load_source") if isinstance(model.get("load_source"), Mapping) else {}
    findings: list[dict[str, Any]] = []

    selected_dict = _record_payload(selected_payload)
    if selected_dict is None:
        findings.append(
            _finding(
                "selected_record_found",
                "FAIL",
                "selected_record",
                "selection",
                "Selected record is missing from the review model.",
                remediation="Rebuild the review model with one exact selected run or catalog id.",
            )
        )
    else:
        findings.append(
            _finding(
                "selected_record_found",
                "PASS",
                "selected_record",
                "selection",
                "Selected record is present.",
            )
        )
        findings.append(_selected_identity_finding(selected_dict))

    findings.extend(_resolver_findings(resolver_payload))
    findings.extend(_canonicality_findings(canonicality))
    findings.extend(_load_source_findings(load_source, load_summary))
    findings.extend(_path_findings(model, selected_dict))
    findings.extend(_lineage_findings(lineage_payload, selected_dict, related_payload))
    findings.extend(_coverage_findings(selected_dict))
    findings.extend(_catalog_validation_findings(selected_dict, records, repo_root))
    findings = sorted(findings, key=lambda finding: finding["finding_id"])
    summary = _diagnostic_summary(findings, selected_dict)
    metadata = build_review_pack_metadata(
        authority_paths=_portable_authority_paths(model.get("canonicality", {}).get("authority_paths", [])),
        fingerprint_payload={"review_id": review_id, "diagnostics": [item["finding_id"] for item in findings]},
    )
    return _json_safe(
        {
            "schema_version": "catalog_health_diagnostics.v1",
            "review_id": review_id,
            "summary": summary,
            "findings": findings,
            "warnings": [finding for finding in findings if finding["status"] in {"WARN", "FAIL"}],
            **metadata,
        }
    )


def _select_subject(
    records: list[CatalogRecord],
    *,
    selected_run_id: str | None,
    selected_catalog_id: str | None,
) -> CatalogRecord:
    matches = [
        record
        for record in records
        if (selected_run_id is None or record.run_id == selected_run_id)
        and (selected_catalog_id is None or record.catalog_id == selected_catalog_id)
    ]
    if not matches:
        raise EvidenceReviewError("Selected catalog record not found.")
    if len(matches) > 1:
        raise EvidenceReviewError("Selected catalog record is ambiguous.")
    return matches[0]


def _record_payload(record: CatalogRecord | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if isinstance(record, CatalogRecord):
        return record.to_dict()
    if isinstance(record, Mapping):
        return dict(record)
    return None


def _selected_identity_finding(record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("catalog_id") and record.get("run_id"):
        return _finding(
            "selected_record_identity",
            "PASS",
            "selected_record",
            "selection",
            "Selected record has catalog and run identity.",
        )
    return _finding(
        "selected_record_identity",
        "WARN",
        "selected_record",
        "selection",
        "Selected record is missing catalog or run identity.",
        remediation="Prefer artifact-backed records with explicit catalog_id and run_id.",
    )


def _resolver_findings(resolution: Mapping[str, Any]) -> list[dict[str, Any]]:
    status = resolution.get("resolution_status")
    if status == "resolved":
        resolver_status = "PASS"
        message = "Selected record canonical sources resolved."
    elif status == "partial":
        resolver_status = "WARN"
        message = "Selected record canonical sources resolved only partially."
    elif status == "unresolved":
        resolver_status = "FAIL"
        message = "Selected record canonical sources did not resolve."
    else:
        resolver_status = "NA"
        message = "Resolver status is unavailable."
    findings = [
        _finding(
            "resolver_status",
            resolver_status,
            "selected_record",
            "resolver",
            message,
            paths=resolution.get("missing_sources", []),
            remediation="Reopen or repair declared canonical sources before relying on this review context."
            if resolver_status in {"WARN", "FAIL"}
            else None,
        )
    ]
    missing_sources = list(resolution.get("missing_sources") or [])
    findings.append(
        _finding(
            "missing_canonical_sources",
            "WARN" if missing_sources else ("NA" if resolver_status == "NA" else "PASS"),
            "selected_record",
            "resolver",
            f"{len(missing_sources)} declared canonical source path(s) are missing."
            if missing_sources
            else "No missing canonical source paths reported.",
            paths=missing_sources,
        )
    )
    fingerprint = resolution.get("source_fingerprint")
    findings.append(
        _finding(
            "source_fingerprint_present",
            "PASS" if fingerprint else ("NA" if resolver_status == "NA" else "WARN"),
            "selected_record",
            "resolver",
            "Canonical source fingerprint is present."
            if fingerprint
            else "Canonical source fingerprint is unavailable.",
        )
    )
    warnings = list(resolution.get("warnings") or [])
    findings.append(
        _finding(
            "resolver_warnings",
            "WARN" if warnings else ("NA" if resolver_status == "NA" else "PASS"),
            "selected_record",
            "resolver",
            "Resolver warnings were reported." if warnings else "No resolver warnings reported.",
            remediation="Inspect resolver warnings and canonical paths before decision-sensitive use."
            if warnings
            else None,
        )
    )
    return findings


def _canonicality_findings(canonicality: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _finding(
            "canonicality_envelope",
            "PASS" if canonicality.get("schema_version") == "canonicality.v1" else "WARN",
            "review_model",
            "canonicality",
            "Canonicality Envelope v1 is present."
            if canonicality.get("schema_version") == "canonicality.v1"
            else "Canonicality Envelope v1 is missing or incompatible.",
        ),
        _finding(
            "canonicality_semantics",
            "PASS"
            if canonicality.get("derived_class") == "review_pack"
            and canonicality.get("non_authoritative") is True
            and canonicality.get("write_back_forbidden") is True
            and canonicality.get("rebuildable") is True
            else "FAIL",
            "review_model",
            "canonicality",
            "Review model preserves derived non-authoritative semantics."
            if canonicality.get("derived_class") == "review_pack"
            and canonicality.get("non_authoritative") is True
            and canonicality.get("write_back_forbidden") is True
            and canonicality.get("rebuildable") is True
            else "Review model canonicality semantics are unsafe for review-pack use.",
        ),
    ]


def _load_source_findings(
    load_source: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    findings = [
        _finding(
            "load_source_envelope",
            "PASS"
            if load_source.get("schema_version") == "load_source.v1"
            and load_source.get("loaded_from") == "review_pack"
            else "FAIL",
            "review_model",
            "load_source",
            "load_source.v1 identifies a review-pack derived surface."
            if load_source.get("schema_version") == "load_source.v1"
            and load_source.get("loaded_from") == "review_pack"
            else "Review model load-source metadata is missing or incompatible.",
        )
    ]
    requested = summary.get("requested_mode")
    resolved = summary.get("resolved_mode")
    findings.append(
        _finding(
            "load_mode_summary",
            "PASS" if requested and resolved else "WARN",
            "review_model",
            "load_source",
            "Requested and resolved catalog load modes are recorded."
            if requested and resolved
            else "Requested or resolved catalog load mode is missing.",
        )
    )
    index_validated = summary.get("index_validated")
    if resolved == "index":
        status: DiagnosticStatus = "PASS" if index_validated is True else "FAIL"
        message = (
            "Derived index was validated before use."
            if index_validated is True
            else "Index-backed review lacks validated-index evidence."
        )
    else:
        status = "NA"
        message = "Derived-index validation is not applicable to direct-scan review."
    findings.append(_finding("derived_index_validation", status, "review_model", "load_source", message))
    return findings


def _path_findings(model: Mapping[str, Any], selected_record: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    path_values = _collect_path_values(model, selected_record)
    issues: list[tuple[str, str]] = []
    derived_authority_paths: list[str] = []
    for path in path_values:
        issue = _path_issue(path)
        if issue is not None:
            issues.append((path, issue))
        if path.startswith("artifacts/_derived/"):
            derived_authority_paths.append(path)
    findings = [
        _finding(
            "portable_paths",
            "PASS" if not issues else "FAIL",
            "review_model",
            "path_portability",
            "All reviewed paths are repository-relative POSIX paths."
            if not issues
            else "Non-portable or unsafe paths were found.",
            paths=[path for path, _ in sorted(issues)],
            remediation="Use repository-relative POSIX paths only." if issues else None,
        ),
        _finding(
            "derived_authority_leakage",
            "FAIL" if derived_authority_paths else "PASS",
            "review_model",
            "derived_namespace",
            "Canonical authority paths do not point into _derived."
            if not derived_authority_paths
            else "Canonical authority paths point into _derived review/read-model space.",
            paths=sorted(derived_authority_paths),
        ),
    ]
    review_root = model.get("review_root")
    findings.append(
        _finding(
            "review_root_namespace",
            "PASS"
            if isinstance(review_root, str)
            and review_root.startswith("artifacts/_derived/evidence_review/")
            else "FAIL",
            "review_model",
            "derived_namespace",
            "Review root remains under artifacts/_derived/evidence_review/."
            if isinstance(review_root, str)
            and review_root.startswith("artifacts/_derived/evidence_review/")
            else "Review root is outside the approved derived evidence-review namespace.",
            paths=[review_root] if isinstance(review_root, str) else [],
        )
    )
    return findings


def _lineage_findings(
    lineage_summary: Mapping[str, Any],
    selected_record: Mapping[str, Any] | None,
    related_records_payload: Sequence[Any],
) -> list[dict[str, Any]]:
    if not lineage_summary:
        return [
            _finding(
                "selected_lineage_summary",
                "NA",
                "selected_record",
                "lineage",
                "Selected lineage summary is unavailable.",
            )
        ]
    edge_count = lineage_summary.get("selected_edge_count")
    findings = [
        _finding(
            "selected_lineage_summary",
            "PASS" if isinstance(edge_count, int) else "WARN",
            "selected_record",
            "lineage",
            "Selected lineage summary is present."
            if isinstance(edge_count, int)
            else "Selected lineage summary is incomplete.",
        )
    ]
    selected_run_type = selected_record.get("run_type") if selected_record else None
    if isinstance(edge_count, int) and edge_count > 0:
        findings.append(
            _finding(
                "selected_lineage_coverage",
                "PASS",
                "selected_record",
                "lineage",
                "Selected record has explicit one-hop lineage coverage.",
            )
        )
    elif selected_run_type in {"strategy", "portfolio", "alpha_evaluation"}:
        findings.append(
            _finding(
                "selected_lineage_coverage",
                "WARN",
                "selected_record",
                "lineage",
                "No explicit one-hop lineage edges were found for a lineage-capable record.",
            )
        )
    else:
        findings.append(
            _finding(
                "selected_lineage_coverage",
                "NA",
                "selected_record",
                "lineage",
                "Explicit one-hop lineage coverage is not applicable for this record type.",
            )
        )
    findings.append(
        _finding(
            "related_records_summary",
            "PASS" if related_records_payload else "NA",
            "related_records",
            "lineage",
            "One-hop related records are present."
            if related_records_payload
            else "No one-hop related records were reported.",
        )
    )
    return findings


def _coverage_findings(selected_record: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    metadata = selected_record.get("metadata", {}) if selected_record else {}
    findings = [
        _finding(
            "dataset_lineage_metadata",
            "PASS" if isinstance(metadata, Mapping) and "dataset_lineage" in metadata else "NA",
            "selected_record",
            "lineage",
            "Dataset lineage metadata is present."
            if isinstance(metadata, Mapping) and "dataset_lineage" in metadata
            else "Dataset lineage metadata is not present.",
        ),
        _finding(
            "feature_lineage_metadata",
            "PASS" if isinstance(metadata, Mapping) and "feature_lineage" in metadata else "NA",
            "selected_record",
            "lineage",
            "Feature lineage metadata is present."
            if isinstance(metadata, Mapping) and "feature_lineage" in metadata
            else "Feature lineage metadata is not present.",
        ),
    ]
    governance_present = bool(selected_record and selected_record.get("governance_status"))
    release_present = bool(selected_record and selected_record.get("release_validation_present"))
    findings.extend(
        [
            _finding(
                "governance_evidence_presence",
                "PASS" if governance_present else "NA",
                "selected_record",
                "governance",
                "Governance evidence is present."
                if governance_present
                else "Governance evidence is not present for this selected record.",
            ),
            _finding(
                "release_validation_presence",
                "PASS" if release_present else "NA",
                "selected_record",
                "release_validation",
                "Release-validation evidence is present."
                if release_present
                else "Release-validation evidence is not present for this selected record.",
            ),
        ]
    )
    return findings


def _catalog_validation_findings(
    selected_record: Mapping[str, Any] | None,
    records: Sequence[CatalogRecord] | None,
    repo_root: str | Path | None,
) -> list[dict[str, Any]]:
    if selected_record is None or records is None or repo_root is None:
        return []
    matched = [record for record in records if record.catalog_id == selected_record.get("catalog_id")]
    if not matched:
        return []
    from src.catalog.validation import validate_record

    issues = validate_record(matched[0], repo_root=repo_root)
    return [
        _finding(
            "catalog_validation_selected_record",
            "PASS" if not issues else "WARN",
            "selected_record",
            "catalog_validation",
            "Selected record passed catalog validation."
            if not issues
            else "Selected record has catalog validation findings.",
            paths=[issue.path for issue in issues if issue.path],
        )
    ]


def _collect_path_values(model: Mapping[str, Any], selected_record: Mapping[str, Any] | None) -> list[str]:
    paths: set[str] = set()
    canonicality = model.get("canonicality")
    if isinstance(canonicality, Mapping):
        for path in canonicality.get("authority_paths", []) or []:
            if isinstance(path, str):
                paths.add(path)
    for path in model.get("canonical_sources", []) or []:
        if isinstance(path, str):
            paths.add(path)
    if isinstance(selected_record, Mapping):
        for key in ("artifact_root", "source_registry_path", "source_manifest_path", "source_marker_path"):
            value = selected_record.get(key)
            if isinstance(value, str):
                paths.add(value)
        for path in selected_record.get("source_files", []) or []:
            if isinstance(path, str):
                paths.add(path)
    return sorted(paths)


def _path_issue(path: str) -> str | None:
    if "\\" in path:
        return "backslash"
    if path.startswith("/"):
        return "absolute"
    if "://" in path:
        return "uri"
    parts = PurePosixPath(path).parts
    if any(part == ".." for part in parts):
        return "parent_traversal"
    if parts and len(parts[0]) == 2 and parts[0][1] == ":":
        return "drive_absolute"
    return None


def _portable_authority_paths(paths: Iterable[Any]) -> list[str]:
    portable: list[str] = []
    for path in paths:
        if not isinstance(path, str):
            continue
        normalized = portable_path(path)
        try:
            validate_portable_repository_path(normalized)
        except ValueError:
            continue
        portable.append(normalized)
    return sorted(set(portable))


def _finding(
    check_id: str,
    status: DiagnosticStatus,
    scope: str,
    category: str,
    message: str,
    *,
    paths: Sequence[str] | None = None,
    remediation: str | None = None,
) -> dict[str, Any]:
    payload = {
        "finding_id": f"{check_id}:{scope}",
        "check_id": check_id,
        "severity": status,
        "status": status,
        "scope": scope,
        "category": category,
        "message": message,
        "paths": sorted(path for path in (paths or []) if isinstance(path, str)),
    }
    if remediation is not None:
        payload["remediation"] = remediation
    return payload


def _diagnostic_summary(
    findings: Sequence[Mapping[str, Any]],
    selected_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    counts_by_status = Counter(str(finding["status"]) for finding in findings)
    counts_by_category = Counter(str(finding["category"]) for finding in findings)
    counts_by_scope = Counter(str(finding["scope"]) for finding in findings)
    overall_status = "FAIL" if counts_by_status["FAIL"] else "WARN" if counts_by_status["WARN"] else "PASS"
    return {
        "overall_status": overall_status,
        "finding_count": len(findings),
        "counts_by_status": {status: counts_by_status[status] for status in ("PASS", "WARN", "FAIL", "NA")},
        "counts_by_category": {key: counts_by_category[key] for key in sorted(counts_by_category)},
        "counts_by_scope": {key: counts_by_scope[key] for key in sorted(counts_by_scope)},
        "selected_catalog_id": selected_record.get("catalog_id") if selected_record else None,
        "selected_run_id": selected_record.get("run_id") if selected_record else None,
    }


def _build_lineage_exports(
    records: list[CatalogRecord],
    edges: list[Any],
    *,
    selected_run_id: str | None,
    lineage_format: EvidenceReviewLineageFormat,
    load_source: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if selected_run_id is None:
        return {}
    formats = ("openlineage", "prov") if lineage_format == "both" else (lineage_format,)
    return {
        export_format: export_lineage(
            records,
            edges,
            format=export_format,
            selected_run_id=selected_run_id,
            load_source=load_source,
        )
        for export_format in formats
    }


def _lineage_summary(lineage_exports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not lineage_exports:
        return {
            "formats": [],
            "selected_record_count": 0,
            "selected_edge_count": 0,
            "related_run_ids": [],
        }
    first = lineage_exports[sorted(lineage_exports)[0]]
    nodes = first.get("nodes") or first.get("entities") or []
    related_run_ids = sorted(
        {
            node_metadata.get("run_id")
            for node in nodes
            if isinstance((node_metadata := node.get("facets") or node.get("attributes")), dict)
            and node_metadata.get("run_id")
            and node_metadata.get("run_id") != first.get("selected_run_id")
        }
    )
    return {
        "formats": sorted(lineage_exports),
        "selected_record_count": first.get("record_count", 0),
        "selected_edge_count": first.get("edge_count", 0),
        "related_run_ids": related_run_ids,
    }


def _deterministic_review_id(
    *,
    selected_run_id: str | None,
    selected_catalog_id: str | None,
    index_mode: str,
    index_path: str | Path | None,
    lineage_format: str,
    resolve_related: bool,
) -> str:
    normalized_index_path = portable_path(index_path) if index_path is not None else None
    if normalized_index_path is not None:
        validate_portable_repository_path(normalized_index_path)
    digest = stable_json_fingerprint(
        {
            "selected_run_id": selected_run_id,
            "selected_catalog_id": selected_catalog_id,
            "index_mode": index_mode,
            "index_path": normalized_index_path,
            "lineage_format": lineage_format,
            "resolve_related": resolve_related,
        }
    )
    return f"review_{digest[:16]}"


def _authority_root(artifacts_root: Path, repo_root: Path) -> str:
    if artifacts_root.is_relative_to(repo_root):
        return artifacts_root.relative_to(repo_root).as_posix()
    return portable_path(artifacts_root.name)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value
