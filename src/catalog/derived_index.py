"""Optional disposable SQLite index over canonical catalog records."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Literal

from src.catalog.canonicality import (
    build_canonicality_envelope,
    canonical_authority_paths,
    canonicality_status,
)
from src.catalog.indexer import build_catalog
from src.catalog.models import CatalogRecord, CatalogValidationStatus

DERIVED_INDEX_SCHEMA_VERSION = 1
CATALOG_RECORD_SCHEMA_VERSION = 1
INDEX_KIND = "catalog_derived_index"
BUILDER_VERSION = "m36_issue405_v1"
IndexMode = Literal["direct", "index", "auto"]


class DerivedIndexError(ValueError):
    """Raised when a derived catalog index is missing, stale, or incompatible."""


@dataclass(frozen=True)
class DerivedIndexValidation:
    metadata: dict[str, Any]
    records: list[CatalogRecord]


def build_derived_index(
    artifacts_root: str | Path,
    output_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build an atomic disposable SQLite index from canonical artifact scans."""

    resolved_artifacts = Path(artifacts_root).resolve()
    resolved_repo = Path(repo_root).resolve() if repo_root is not None else resolved_artifacts.parent
    records = build_catalog(resolved_artifacts, repo_root=resolved_repo)
    metadata = _build_metadata(records, resolved_artifacts=resolved_artifacts, resolved_repo=resolved_repo)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        _write_index(temp_path, metadata, records)
        temp_path.replace(target)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return metadata


def validate_derived_index(
    index_path: str | Path,
    *,
    artifacts_root: str | Path,
    repo_root: str | Path | None = None,
    check_source_fingerprint: bool = True,
) -> DerivedIndexValidation:
    """Validate an index and return its decoded records when it is safe to use."""

    path = Path(index_path)
    if not path.exists():
        raise DerivedIndexError(f"Derived catalog index not found: {path}")

    resolved_artifacts = Path(artifacts_root).resolve()
    resolved_repo = Path(repo_root).resolve() if repo_root is not None else resolved_artifacts.parent

    try:
        with sqlite3.connect(path) as connection:
            _validate_tables(connection)
            metadata = _read_metadata(connection)
            records = _read_records(connection)
    except sqlite3.DatabaseError as exc:
        raise DerivedIndexError(f"Derived catalog index is unreadable; rebuild required: {path}") from exc

    _validate_metadata(metadata, resolved_artifacts=resolved_artifacts, resolved_repo=resolved_repo)
    metadata = {**metadata, "canonicality_status": canonicality_status(metadata)}
    _validate_internal_counts(metadata, records)
    if check_source_fingerprint:
        current_records = build_catalog(resolved_artifacts, repo_root=resolved_repo)
        current_fingerprint = _records_fingerprint(current_records)
        if metadata["source_fingerprint"] != current_fingerprint:
            raise DerivedIndexError("Derived catalog index is stale; rebuild required.")
    return DerivedIndexValidation(metadata=metadata, records=records)


def load_catalog_records(
    artifacts_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    index_path: str | Path | None = None,
    mode: IndexMode = "direct",
) -> list[CatalogRecord]:
    """Load catalog records by direct scan, index, or safe auto fallback."""

    resolved_artifacts = Path(artifacts_root)
    resolved_repo = Path(repo_root) if repo_root is not None else None
    if mode == "direct":
        return build_catalog(resolved_artifacts, repo_root=resolved_repo)
    if mode not in {"index", "auto"}:
        raise ValueError("mode must be 'direct', 'index', or 'auto'")
    if index_path is None:
        if mode == "index":
            raise DerivedIndexError("Index mode requires an explicit index path.")
        return build_catalog(resolved_artifacts, repo_root=resolved_repo)

    path = Path(index_path)
    if not path.exists() and mode == "auto":
        return build_catalog(resolved_artifacts, repo_root=resolved_repo)
    return validate_derived_index(
        path,
        artifacts_root=resolved_artifacts,
        repo_root=resolved_repo,
    ).records


def _build_metadata(
    records: list[CatalogRecord],
    *,
    resolved_artifacts: Path,
    resolved_repo: Path,
) -> dict[str, Any]:
    source_fingerprint = _records_fingerprint(records)
    metadata = {
        "schema_version": DERIVED_INDEX_SCHEMA_VERSION,
        "index_kind": INDEX_KIND,
        "source_artifact_root": _portable_path(resolved_artifacts, resolved_repo),
        "repo_root": ".",
        "record_count": len(records),
        "record_family_counts": _family_counts(records),
        "created_at_utc": None,
        "source_fingerprint": source_fingerprint,
        "catalog_record_schema_version": CATALOG_RECORD_SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "is_derived": True,
        "canonical_source": "artifacts",
    }
    metadata.update(
        build_canonicality_envelope(
            derived_class="sqlite_read_model",
            authority_root=_portable_path(resolved_artifacts, resolved_repo),
            authority_paths=canonical_authority_paths(records),
            authority_fingerprint=source_fingerprint,
        )
    )
    metadata["canonicality_status"] = canonicality_status(metadata)
    return metadata


def _write_index(path: Path, metadata: dict[str, Any], records: list[CatalogRecord]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)")
        connection.execute(
            """
            CREATE TABLE catalog_records (
                catalog_id TEXT PRIMARY KEY,
                run_id TEXT,
                run_type TEXT NOT NULL,
                status TEXT NOT NULL,
                record_family TEXT,
                artifact_root TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX idx_catalog_records_run_type ON catalog_records(run_type)")
        connection.execute("CREATE INDEX idx_catalog_records_record_family ON catalog_records(record_family)")
        connection.executemany(
            "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
            [(key, json.dumps(metadata[key], sort_keys=True)) for key in sorted(metadata)],
        )
        connection.executemany(
            """
            INSERT INTO catalog_records(
                catalog_id, run_id, run_type, status, record_family, artifact_root, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.catalog_id,
                    record.run_id,
                    record.run_type,
                    record.status,
                    record.record_family,
                    record.artifact_root,
                    json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")),
                )
                for record in records
            ],
        )
        connection.commit()
    finally:
        connection.close()


def _validate_tables(connection: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        )
    }
    if not {"metadata", "catalog_records"}.issubset(tables):
        raise DerivedIndexError("Derived catalog index is missing required tables; rebuild required.")
    columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(catalog_records)")
    }
    required = {"catalog_id", "run_id", "run_type", "status", "record_family", "artifact_root", "payload_json"}
    if not required.issubset(columns):
        raise DerivedIndexError("Derived catalog index is missing required columns; rebuild required.")


def _read_metadata(connection: sqlite3.Connection) -> dict[str, Any]:
    return {
        key: json.loads(value)
        for key, value in connection.execute("SELECT key, value_json FROM metadata ORDER BY key")
    }


def _read_records(connection: sqlite3.Connection) -> list[CatalogRecord]:
    rows = connection.execute(
        "SELECT payload_json FROM catalog_records ORDER BY run_type, COALESCE(run_id, ''), catalog_id, artifact_root"
    )
    return [_record_from_dict(json.loads(row[0])) for row in rows]


def _validate_metadata(metadata: dict[str, Any], *, resolved_artifacts: Path, resolved_repo: Path) -> None:
    if metadata.get("schema_version") != DERIVED_INDEX_SCHEMA_VERSION:
        raise DerivedIndexError("Derived catalog index schema is incompatible; rebuild required.")
    if metadata.get("index_kind") != INDEX_KIND or metadata.get("is_derived") is not True:
        raise DerivedIndexError("Index metadata does not describe a derived catalog index.")
    if metadata.get("canonical_source") != "artifacts":
        raise DerivedIndexError("Derived catalog index canonical source is invalid.")
    expected_root = _portable_path(resolved_artifacts, resolved_repo)
    if metadata.get("source_artifact_root") != expected_root:
        raise DerivedIndexError("Derived catalog index artifact root does not match; rebuild required.")


def _validate_internal_counts(metadata: dict[str, Any], records: list[CatalogRecord]) -> None:
    if metadata.get("record_count") != len(records):
        raise DerivedIndexError("Derived catalog index record count is inconsistent; rebuild required.")
    if metadata.get("record_family_counts") != _family_counts(records):
        raise DerivedIndexError("Derived catalog index family counts are inconsistent; rebuild required.")


def _record_from_dict(payload: dict[str, Any]) -> CatalogRecord:
    validation = payload["validation"]
    return CatalogRecord(
        **{
            key: value
            for key, value in payload.items()
            if key != "validation"
        },
        validation=CatalogValidationStatus(**validation),
    )


def _records_fingerprint(records: list[CatalogRecord]) -> str:
    serialized = json.dumps([record.to_dict() for record in records], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _family_counts(records: list[CatalogRecord]) -> dict[str, int]:
    counts = Counter(record.record_family for record in records if record.record_family)
    return {key: counts[key] for key in sorted(counts)}


def _portable_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.name
