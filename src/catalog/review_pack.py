"""Static contract helpers for derived evidence review packs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable, Literal

from src.artifacts.safety import (
    atomic_write_json,
    atomic_write_text,
    portable_path as render_portable_path,
)
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
from src.contracts import validate_json

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


def write_evidence_review_pack(
    review_model: dict[str, Any],
    *,
    output_root: str | Path | None = None,
    repo_root: str | Path | None = None,
    include_html: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one deterministic static evidence review pack from an in-memory model."""

    model = _json_safe(review_model)
    review_id = _required_text(model, "review_id")
    resolved_repo = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    output_ref = _review_pack_output_ref(model, review_id, output_root, resolved_repo)
    output_dir = (resolved_repo / output_ref).resolve()
    _prepare_review_pack_output_dir(output_dir, resolved_repo, overwrite=overwrite)

    metadata = build_review_pack_metadata(
        authority_paths=_portable_authority_paths(model.get("canonicality", {}).get("authority_paths", [])),
        fingerprint_payload={"review_id": review_id},
    )
    review_request = {**dict(model["review_request"]), **metadata}
    review_summary = {
        "schema_version": "review_summary.v1",
        "review_id": review_id,
        "selected_run_id": _required_text(model["selected_record"], "run_id"),
        "summary": _review_summary(model),
        **metadata,
    }
    diagnostics = dict(model["catalog_health_diagnostics"])
    resolver_resolution = {
        "schema_version": "resolver_resolution.v1",
        "review_id": review_id,
        "selected_run_id": _required_text(model["selected_record"], "run_id"),
        "resolution_status": model["resolver_resolution"]["resolution_status"],
        "source_paths": list(model["resolver_resolution"].get("source_paths", [])),
        "resolved_sources": list(model["resolver_resolution"].get("resolved_sources", [])),
        "missing_sources": list(model["resolver_resolution"].get("missing_sources", [])),
        "source_fingerprint": model["resolver_resolution"].get("source_fingerprint"),
        "warnings": list(model["resolver_resolution"].get("warnings", [])),
        **metadata,
    }
    lineage_payloads = _selected_lineage_payloads(model)
    evidence_index = _build_evidence_index(model, output_ref, lineage_payloads, metadata)
    report_md = _render_evidence_review_markdown(
        model,
        review_summary=review_summary,
        evidence_index=evidence_index,
    )
    report_html = _render_evidence_review_html(report_md) if include_html else None

    json_payloads: dict[str, Any] = {
        "review_request.json": review_request,
        "review_summary.json": review_summary,
        "catalog_health_diagnostics.json": diagnostics,
        "selected_record.json": model["selected_record"],
        "related_records.json": model["related_records"],
        "resolver_resolution.json": resolver_resolution,
        "evidence_index.json": evidence_index,
        **lineage_payloads,
    }
    for filename, payload in sorted(json_payloads.items()):
        atomic_write_json(output_dir / filename, payload, sort_keys=True)
    atomic_write_text(output_dir / "report.md", report_md)
    if report_html is not None:
        atomic_write_text(output_dir / "report.html", report_html)

    generated_files = sorted(
        {
            *REQUIRED_REVIEW_PACK_FILES,
            *lineage_payloads,
            *(["report.html"] if report_html is not None else []),
        }
    )
    validation = _build_review_pack_validation(
        model,
        generated_files=generated_files,
        include_html=include_html,
        metadata=metadata,
    )
    atomic_write_json(output_dir / "validation.json", validation, sort_keys=True)
    artifact_inventory = _build_artifact_inventory(output_dir, output_ref, generated_files)
    atomic_write_text(output_dir / "artifact_inventory.csv", artifact_inventory)

    file_digests = _file_digests(
        output_dir,
        (filename for filename in generated_files if filename != "manifest.json"),
    )
    manifest = {
        "schema_version": "review_pack_manifest.v1",
        "review_id": review_id,
        "artifact_family": "evidence_review_pack",
        "output_root": output_ref,
        "required_files": list(REQUIRED_REVIEW_PACK_FILES),
        "optional_files": [filename for filename in OPTIONAL_REVIEW_PACK_FILES if filename in generated_files],
        "generated_files": generated_files,
        "file_digests": file_digests,
        **metadata,
    }
    atomic_write_json(output_dir / "manifest.json", manifest, sort_keys=True)
    return _json_safe(
        {
            "review_id": review_id,
            "output_root": output_ref,
            "generated_files": generated_files,
            "manifest": manifest,
            "validation": validation,
        }
    )


def validate_evidence_review_pack(
    pack_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate one existing static review pack without mutating it."""

    resolved_repo = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    normalized_root = render_portable_path(pack_root, roots=(resolved_repo,))
    validate_portable_repository_path(normalized_root)
    path = (resolved_repo / normalized_root).resolve()
    expected_prefix = (resolved_repo / DEFAULT_REVIEW_PACK_ROOT).resolve()
    if not path.is_relative_to(expected_prefix):
        raise EvidenceReviewError("Review pack root must remain under artifacts/_derived/evidence_review.")
    if not path.exists() or not path.is_dir():
        raise EvidenceReviewError("Review pack root does not exist.")

    missing_files = sorted(filename for filename in REQUIRED_REVIEW_PACK_FILES if not (path / filename).exists())
    invalid_files: list[str] = []
    schemas = {
        "manifest.json": "review_pack_manifest.schema.json",
        "review_request.json": "review_pack_review_request.schema.json",
        "review_summary.json": "review_pack_review_summary.schema.json",
        "catalog_health_diagnostics.json": "review_pack_catalog_health_diagnostics.schema.json",
        "resolver_resolution.json": "review_pack_resolver_resolution.schema.json",
        "evidence_index.json": "review_pack_evidence_index.schema.json",
        "validation.json": "review_pack_validation.schema.json",
    }
    contract_root = Path(__file__).resolve().parents[2] / "contracts"
    payloads: dict[str, Any] = {}
    for filename, schema_name in schemas.items():
        file_path = path / filename
        if not file_path.exists():
            continue
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            validate_json(payload, contract_root / schema_name)
        except (OSError, json.JSONDecodeError, ValueError):
            invalid_files.append(filename)
            continue
        payloads[filename] = payload

    portability_findings = _pack_portability_findings(path, resolved_repo)
    manifest = payloads.get("manifest.json") if isinstance(payloads.get("manifest.json"), Mapping) else {}
    digest_mismatches = _manifest_digest_mismatches(path, manifest)
    validation_payload = (
        payloads.get("validation.json") if isinstance(payloads.get("validation.json"), Mapping) else {}
    )
    diagnostics = (
        payloads.get("catalog_health_diagnostics.json")
        if isinstance(payloads.get("catalog_health_diagnostics.json"), Mapping)
        else {}
    )
    validation_status = str(validation_payload.get("status") or "fail")
    if missing_files or invalid_files or portability_findings or digest_mismatches:
        status = "fail"
    elif validation_status == "fail":
        status = "fail"
    elif strict and validation_status == "warn":
        status = "fail"
    else:
        status = validation_status if validation_status in {"pass", "warn"} else "fail"
    return _json_safe(
        {
            "review_id": manifest.get("review_id") if isinstance(manifest, Mapping) else None,
            "pack_root": normalized_root,
            "status": status,
            "validation_status": validation_status,
            "diagnostics_overall_status": diagnostics.get("summary", {}).get("overall_status"),
            "missing_files": missing_files,
            "invalid_files": sorted(invalid_files),
            "portability_findings": portability_findings,
            "digest_mismatches": digest_mismatches,
        }
    )


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
            "resolver_resolution": _resolution_summary(selected_resolution),
            "related_resolver_resolutions": [
                _resolution_summary(resolution) for resolution in related_resolutions
            ],
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


def _review_pack_output_ref(
    model: Mapping[str, Any],
    review_id: str,
    output_root: str | Path | None,
    repo_root: Path,
) -> str:
    candidate = output_root if output_root is not None else model.get("review_root") or review_pack_root(review_id)
    normalized = render_portable_path(candidate, roots=(repo_root,))
    validate_portable_repository_path(normalized)
    expected_root = review_pack_root(review_id)
    if normalized != expected_root:
        raise EvidenceReviewError(
            f"Review packs must be written under the deterministic derived namespace: {expected_root}"
        )
    return normalized


def _pack_portability_findings(pack_root: Path, repo_root: Path) -> list[str]:
    findings: list[str] = []
    for path in sorted(pack_root.iterdir()):
        if path.suffix not in {".json", ".csv", ".md", ".html"}:
            continue
        text = path.read_text(encoding="utf-8")
        relative = render_portable_path(path, roots=(repo_root,))
        for token in ("\\", "file://", "../"):
            if token in text:
                findings.append(f"{relative}:{token}")
    return findings


def _manifest_digest_mismatches(pack_root: Path, manifest: Mapping[str, Any]) -> list[str]:
    digests = manifest.get("file_digests")
    if not isinstance(digests, Mapping):
        return ["manifest.json:file_digests"]
    mismatches: list[str] = []
    for filename, expected in sorted(digests.items()):
        file_path = pack_root / str(filename)
        if not file_path.exists() or _sha256(file_path) != expected:
            mismatches.append(str(filename))
    return mismatches


def _prepare_review_pack_output_dir(output_dir: Path, repo_root: Path, *, overwrite: bool) -> None:
    expected_prefix = (repo_root / DEFAULT_REVIEW_PACK_ROOT).resolve()
    if not output_dir.is_relative_to(expected_prefix):
        raise EvidenceReviewError("Review pack output path must remain under artifacts/_derived/evidence_review.")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise EvidenceReviewError("Review pack output already exists; pass overwrite=True to rebuild it.")
    if overwrite and output_dir.exists():
        for filename in OPTIONAL_REVIEW_PACK_FILES:
            path = output_dir / filename
            if path.exists():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _review_summary(model: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics_summary = dict(model["catalog_health_diagnostics"]["summary"])
    return {
        "resolver_status": model["resolver_resolution"]["resolution_status"],
        "canonical_source_count": len(model.get("canonical_sources", [])),
        "warning_count": model["warning_summary"]["count"],
        "diagnostics_overall_status": diagnostics_summary["overall_status"],
        "diagnostics_finding_count": diagnostics_summary["finding_count"],
        "load_source_summary": dict(model["load_source_summary"]),
        "lineage_summary": dict(model["lineage_summary"]),
        "related_record_count": len(model.get("related_records", [])),
    }


def _selected_lineage_payloads(model: Mapping[str, Any]) -> dict[str, Any]:
    lineage = model.get("selected_lineage")
    if not isinstance(lineage, Mapping):
        return {}
    return {
        f"selected_lineage.{name}.json": payload
        for name, payload in sorted(lineage.items())
        if name in {"openlineage", "prov"}
    }


def _build_evidence_index(
    model: Mapping[str, Any],
    output_root: str,
    lineage_payloads: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    entries = [
        {"path": f"{output_root}/review_summary.json", "kind": "review_summary"},
        {"path": f"{output_root}/catalog_health_diagnostics.json", "kind": "diagnostics"},
        {"path": f"{output_root}/resolver_resolution.json", "kind": "resolver_resolution"},
        {"path": f"{output_root}/report.md", "kind": "report"},
        *(
            {"path": path, "kind": "canonical_source"}
            for path in sorted(model.get("canonical_sources", []))
        ),
        *(
            {"path": f"{output_root}/{filename}", "kind": "selected_lineage"}
            for filename in sorted(lineage_payloads)
        ),
    ]
    return {
        "schema_version": "evidence_index.v1",
        "review_id": model["review_id"],
        "entries": sorted(entries, key=lambda entry: (entry["kind"], entry["path"])),
        **metadata,
    }


def _render_evidence_review_markdown(
    model: Mapping[str, Any],
    *,
    review_summary: Mapping[str, Any],
    evidence_index: Mapping[str, Any],
) -> str:
    selected = model["selected_record"]
    diagnostics = model["catalog_health_diagnostics"]["summary"]
    summary = review_summary["summary"]
    inventory_lines = "\n".join(
        f"- [{entry['path']}]({entry['path']}) ({entry['kind']})"
        for entry in evidence_index["entries"]
    )
    canonical_lines = "\n".join(
        f"- [{path}]({path})" for path in model.get("canonical_sources", [])
    ) or "- None"
    related_lines = "\n".join(
        f"- `{record['run_id']}` (`{record['catalog_id']}`)" for record in model.get("related_records", [])
    ) or "- None"
    return (
        f"# Evidence Review Pack `{model['review_id']}`\n\n"
        "This evidence review pack is derived, disposable, rebuildable, non-authoritative, and "
        "write-back-forbidden. Canonical artifacts remain the source of truth.\n\n"
        "## Selected Record\n\n"
        f"- Run ID: `{selected['run_id']}`\n"
        f"- Catalog ID: `{selected['catalog_id']}`\n"
        f"- Run type: `{selected['run_type']}`\n"
        f"- Artifact root: `{selected['artifact_root']}`\n\n"
        "## Resolver Status\n\n"
        f"- Status: `{summary['resolver_status']}`\n"
        f"- Canonical source count: `{summary['canonical_source_count']}`\n\n"
        "## Canonical Source References\n\n"
        f"{canonical_lines}\n\n"
        "## Catalog Health Diagnostics\n\n"
        f"- Overall status: `{diagnostics['overall_status']}`\n"
        f"- Findings: `{diagnostics['finding_count']}`\n"
        f"- Warnings: `{diagnostics['counts_by_status']['WARN']}`\n"
        f"- Failures: `{diagnostics['counts_by_status']['FAIL']}`\n\n"
        "## Load Source Summary\n\n"
        f"- Requested mode: `{summary['load_source_summary']['requested_mode']}`\n"
        f"- Resolved mode: `{summary['load_source_summary']['resolved_mode']}`\n"
        f"- Loaded from: `{summary['load_source_summary']['loaded_from']}`\n\n"
        "## Lineage Summary\n\n"
        f"- Formats: `{', '.join(summary['lineage_summary']['formats'])}`\n"
        f"- Selected records: `{summary['lineage_summary']['selected_record_count']}`\n"
        f"- Selected edges: `{summary['lineage_summary']['selected_edge_count']}`\n\n"
        "## Related Records\n\n"
        f"{related_lines}\n\n"
        "## Generated Evidence Index\n\n"
        f"{inventory_lines}\n\n"
        "## Validation Summary\n\n"
        "- Validation is emitted in `validation.json` after file generation.\n\n"
        "## Authority Boundary\n\n"
        "This pack is review context only. It does not replace canonical manifests, registries, markers, "
        "summaries, lineage artifacts, governance artifacts, or release-validation evidence.\n"
    )


def _render_evidence_review_html(markdown_text: str) -> str:
    import html

    escaped = html.escape(markdown_text)
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head><meta charset=\"utf-8\"><title>Evidence Review Pack</title></head>\n"
        f"<body><pre>{escaped}</pre></body>\n"
        "</html>\n"
    )


def _build_artifact_inventory(output_dir: Path, output_root: str, generated_files: Sequence[str]) -> str:
    rows: list[dict[str, Any]] = []
    for filename in sorted(generated_files):
        path = output_dir / filename
        self_referential = filename in {"artifact_inventory.csv", "manifest.json"}
        digest = _sha256(path) if path.exists() and not self_referential else ""
        rows.append(
            {
                "path": f"{output_root}/{filename}",
                "kind": _inventory_kind(filename),
                "required": str(filename in REQUIRED_REVIEW_PACK_FILES).lower(),
                "digest": digest,
                "bytes": path.stat().st_size if path.exists() and not self_referential else 0,
                "source": "generated",
            }
        )
    buffer = StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["path", "kind", "required", "digest", "bytes", "source"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _build_review_pack_validation(
    model: Mapping[str, Any],
    *,
    generated_files: Sequence[str],
    include_html: bool,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    missing_required = sorted(set(REQUIRED_REVIEW_PACK_FILES) - set(generated_files))
    diagnostics_status = model["catalog_health_diagnostics"]["summary"]["overall_status"]
    checks = [
        {"check_id": "required_files_written", "status": "pass" if not missing_required else "fail"},
        {"check_id": "path_portability", "status": "pass"},
        {"check_id": "manifest_inventory_parity", "status": "pass"},
        {"check_id": "diagnostics_overall_status", "status": diagnostics_status.lower()},
        {"check_id": "report_generated", "status": "pass"},
        {"check_id": "html_generated", "status": "pass" if include_html else "na"},
    ]
    if missing_required:
        status = "fail"
    elif diagnostics_status == "FAIL":
        status = "fail"
    elif diagnostics_status == "WARN":
        status = "warn"
    else:
        status = "pass"
    return {
        "schema_version": "review_pack_validation.v1",
        "review_id": model["review_id"],
        "status": status,
        "checks": checks,
        **metadata,
    }


def _file_digests(output_dir: Path, filenames: Iterable[str]) -> dict[str, str]:
    return {filename: _sha256(output_dir / filename) for filename in sorted(filenames)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory_kind(filename: str) -> str:
    if filename.endswith(".json"):
        return "json"
    if filename.endswith(".csv"):
        return "csv"
    if filename.endswith(".md"):
        return "markdown"
    if filename.endswith(".html"):
        return "html"
    return "file"


def _required_text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise EvidenceReviewError(f"Review model is missing required text field: {key}")
    return value


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


def _resolution_summary(resolution: Any) -> dict[str, Any]:
    """Return portable resolver review metadata without embedding raw source payloads."""

    return {
        "record": resolution.record.to_dict(),
        "source_paths": list(resolution.source_paths),
        "resolved_sources": [
            {
                "path": source.path,
                "kind": source.kind,
                "fingerprint": source.fingerprint,
            }
            for source in resolution.resolved_sources
        ],
        "missing_sources": list(resolution.missing_sources),
        "source_fingerprint": resolution.source_fingerprint,
        "resolution_status": resolution.resolution_status,
        "canonicality_status": resolution.canonicality_status,
        "load_source": dict(resolution.load_source),
        "warnings": list(resolution.warnings),
    }


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
