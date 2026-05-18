"""Read-only resolver APIs for reopening canonical catalog source files."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from src.catalog.canonicality import canonicality_status, portable_path, validate_portable_repository_path
from src.catalog.indexer import build_catalog
from src.catalog.load_source import build_load_source
from src.catalog.models import CatalogRecord, CatalogValidationStatus

ResolutionStatus = Literal["resolved", "partial", "unresolved"]


@dataclass(frozen=True)
class ResolvedSource:
    """One reopened canonical source file."""

    path: str
    kind: str
    fingerprint: str
    content: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "fingerprint": self.fingerprint,
            "content": self.content,
        }


@dataclass(frozen=True)
class CanonicalRecordResolution:
    """Deterministic read-only result of resolving one catalog record."""

    record: CatalogRecord
    source_paths: list[str]
    resolved_sources: list[ResolvedSource]
    missing_sources: list[str]
    source_fingerprint: str | None
    resolution_status: ResolutionStatus
    canonicality_status: str
    load_source: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "source_paths": list(self.source_paths),
            "resolved_sources": [source.to_dict() for source in self.resolved_sources],
            "missing_sources": list(self.missing_sources),
            "source_fingerprint": self.source_fingerprint,
            "resolution_status": self.resolution_status,
            "canonicality_status": self.canonicality_status,
            "load_source": dict(self.load_source),
            "warnings": list(self.warnings),
        }


def resolve_canonical_record(
    record_or_row: CatalogRecord | Mapping[str, Any],
    *,
    artifacts_root: str | Path,
    repo_root: str | Path | None = None,
) -> CanonicalRecordResolution:
    """Reopen declared canonical source files for one record or serialized record."""

    record, envelope = _coerce_record(record_or_row)
    resolved_artifacts = Path(artifacts_root).resolve()
    resolved_repo = Path(repo_root).resolve() if repo_root is not None else resolved_artifacts.parent
    source_paths = _declared_source_paths(record)
    resolved_sources: list[ResolvedSource] = []
    missing_sources: list[str] = []
    warnings: list[str] = []

    for source_path in source_paths:
        normalized = portable_path(source_path)
        try:
            validate_portable_repository_path(normalized)
        except ValueError:
            missing_sources.append(normalized)
            warnings.append(f"non_portable_source_path:{normalized}")
            continue
        absolute_path = (resolved_repo / normalized).resolve()
        if not _is_within_root(absolute_path, resolved_repo):
            missing_sources.append(normalized)
            warnings.append(f"source_path_outside_repo:{normalized}")
            continue
        if not _is_within_root(absolute_path, resolved_artifacts):
            missing_sources.append(normalized)
            warnings.append(f"source_path_outside_artifacts_root:{normalized}")
            continue
        if not absolute_path.exists() or not absolute_path.is_file():
            missing_sources.append(normalized)
            warnings.append(f"missing_source:{normalized}")
            continue
        payload = absolute_path.read_bytes()
        resolved_sources.append(
            ResolvedSource(
                path=normalized,
                kind=_source_kind(normalized, record),
                fingerprint=hashlib.sha256(payload).hexdigest(),
                content=_decode_source(absolute_path, payload),
            )
        )

    if not source_paths or not resolved_sources:
        status: ResolutionStatus = "unresolved"
    elif missing_sources:
        status = "partial"
    else:
        status = "resolved"
    source_fingerprint = _sources_fingerprint(resolved_sources) if resolved_sources else None
    return CanonicalRecordResolution(
        record=record,
        source_paths=source_paths,
        resolved_sources=sorted(resolved_sources, key=lambda source: source.path),
        missing_sources=sorted(missing_sources),
        source_fingerprint=source_fingerprint,
        resolution_status=status,
        canonicality_status=_resolution_canonicality_status(record_or_row, envelope),
        load_source=_resolution_load_source(envelope),
        warnings=sorted(warnings),
    )


def resolve_canonical_sources(
    records_or_rows: list[CatalogRecord | Mapping[str, Any]],
    *,
    artifacts_root: str | Path,
    repo_root: str | Path | None = None,
) -> list[CanonicalRecordResolution]:
    """Resolve several records deterministically."""

    return [
        resolve_canonical_record(record, artifacts_root=artifacts_root, repo_root=repo_root)
        for record in records_or_rows
    ]


def resolve_canonical_record_by_id(
    *,
    artifacts_root: str | Path,
    repo_root: str | Path | None = None,
    catalog_id: str | None = None,
    run_id: str | None = None,
) -> CanonicalRecordResolution:
    """Direct-scan lookup helper that resolves one matching canonical record."""

    if catalog_id is None and run_id is None:
        raise ValueError("catalog_id or run_id is required")
    records = build_catalog(artifacts_root, repo_root=repo_root)
    for record in records:
        if catalog_id is not None and record.catalog_id == catalog_id:
            return resolve_canonical_record(record, artifacts_root=artifacts_root, repo_root=repo_root)
        if run_id is not None and record.run_id == run_id:
            return resolve_canonical_record(record, artifacts_root=artifacts_root, repo_root=repo_root)
    raise ValueError("Canonical catalog record not found.")


def _coerce_record(record_or_row: CatalogRecord | Mapping[str, Any]) -> tuple[CatalogRecord, Mapping[str, Any] | None]:
    if isinstance(record_or_row, CatalogRecord):
        return record_or_row, None
    payload = record_or_row.get("record") if isinstance(record_or_row.get("record"), Mapping) else record_or_row
    if not {"catalog_id", "run_id", "run_type", "status", "artifact_root"}.issubset(payload):
        raise ValueError("Serialized catalog record is missing required fields.")
    validation = payload.get("validation")
    if isinstance(validation, Mapping):
        validation_status = CatalogValidationStatus(**validation)
    else:
        validation_status = CatalogValidationStatus(
            catalog_status="unknown",
            marker_status="unknown",
            manifest_status="unknown",
            artifact_status="unknown",
            qa_status=None,
        )
    fields = CatalogRecord.__dataclass_fields__
    values: dict[str, Any] = {}
    for key, field_info in fields.items():
        if key == "validation":
            continue
        if key in payload:
            values[key] = payload[key]
        elif field_info.default is not MISSING:
            values[key] = field_info.default
        elif field_info.default_factory is not MISSING:  # type: ignore[comparison-overlap]
            values[key] = field_info.default_factory()
        else:
            values[key] = None
    return CatalogRecord(**values, validation=validation_status), record_or_row


def _declared_source_paths(record: CatalogRecord) -> list[str]:
    values = {
        path
        for path in (
            record.source_registry_path,
            record.source_manifest_path,
            record.source_marker_path,
            *record.source_files,
        )
        if path
    }
    return sorted(portable_path(path) for path in values)


def _decode_source(path: Path, payload: bytes) -> Any:
    if path.suffix == ".json":
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
    if path.suffix == ".jsonl":
        try:
            return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
    return {"size_bytes": len(payload)}


def _source_kind(path: str, record: CatalogRecord) -> str:
    if path == record.source_registry_path:
        return "registry"
    if path == record.source_manifest_path:
        return "manifest"
    if path == record.source_marker_path:
        return "marker"
    return Path(path).name


def _sources_fingerprint(sources: list[ResolvedSource]) -> str:
    digest = hashlib.sha256()
    for source in sorted(sources, key=lambda item: item.path):
        digest.update(source.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.fingerprint.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolution_canonicality_status(
    record_or_row: CatalogRecord | Mapping[str, Any],
    envelope: Mapping[str, Any] | None,
) -> str:
    if isinstance(record_or_row, CatalogRecord):
        return "not_applicable"
    return canonicality_status(dict(envelope or record_or_row))


def _resolution_load_source(envelope: Mapping[str, Any] | None) -> dict[str, Any]:
    if envelope is not None and isinstance(envelope.get("load_source"), Mapping):
        return dict(envelope["load_source"])
    return build_load_source(loaded_from="direct_scan", requested_mode="direct", resolved_mode="direct")


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
