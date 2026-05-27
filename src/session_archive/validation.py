from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tarfile
from typing import Any, Mapping

from src.artifacts.safety import atomic_write_text
from src.session_archive.manifest import (
    SessionArchiveBoundaries,
    SessionArchiveError,
    SessionArchiveLogicalGroup,
    SessionArchiveManifest,
    SessionArchiveShard,
    validate_session_archive_manifest,
)

SESSION_ARCHIVE_VALIDATION_REPORT_SCHEMA_VERSION = 1
SESSION_ARCHIVE_INSPECTION_REPORT_SCHEMA_VERSION = 1
EXPECTED_SIDECARS = ("archive_index.json", "checksums.json", "restore_plan.json")
OPTIONAL_LOGICAL_GROUPS = frozenset(group.value for group in SessionArchiveLogicalGroup)
_URL_LIKE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_WINDOWS_DRIVE_PREFIX_PATTERN = re.compile(r"^[A-Za-z]:")


class SessionArchiveIssueCode:
    MISSING_MANIFEST = "missing_manifest"
    MALFORMED_MANIFEST_JSON = "malformed_manifest_json"
    UNSUPPORTED_SCHEMA_VERSION = "unsupported_schema_version"
    MISSING_REQUIRED_MANIFEST_FIELDS = "missing_required_manifest_fields"
    MISSING_SHARD_INDEX = "missing_shard_index"
    MALFORMED_SHARD_INDEX = "malformed_shard_index"
    MISSING_REQUIRED_SHARD = "missing_required_shard"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    MALFORMED_SHARD_METADATA = "malformed_shard_metadata"
    ARCHIVE_INDEX_INCONSISTENCY = "archive_index_inconsistency"
    UNSAFE_ARCHIVE_ENTRY = "unsafe_archive_entry"
    UNSAFE_RESTORE_PATH = "unsafe_restore_path"
    MISSING_OPTIONAL_LOGICAL_GROUP = "missing_optional_logical_group"
    OPTIONAL_DUCKDB_SNAPSHOT_MISSING = "optional_duckdb_snapshot_missing"
    OPTIONAL_DUCKDB_METADATA_WARNING = "optional_duckdb_metadata_warning"
    EMPTY_ARCHIVE_PACK = "empty_archive_pack"
    UNKNOWN_LOGICAL_GROUP = "unknown_logical_group"
    NON_PORTABLE_PATH = "non_portable_path"
    REPORT_WRITE_FAILURE = "report_write_failure"


@dataclass(frozen=True)
class SessionArchiveValidationIssue:
    code: str
    severity: str
    message: str
    path: str | None = None
    context: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.context:
            data["context"] = _stable_jsonable(self.context)
        if self.path is not None:
            data["path"] = self.path
        return data


@dataclass(frozen=True)
class SessionArchiveValidationResult:
    archive_root: Path
    status: str
    archive_id: str | None
    manifest: SessionArchiveManifest | None
    issues: tuple[SessionArchiveValidationIssue, ...]
    checksum_status: str
    inspected_entries: tuple[dict[str, Any], ...]
    report: dict[str, Any]

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


@dataclass(frozen=True)
class SessionArchiveInspectionSummary:
    archive_id: str | None
    schema_version: int | None
    created_at_utc: str | None
    session_id: str | None
    source_runtime_profile: str | None
    source_profile_path: str | None
    logical_groups_included: tuple[str, ...]
    missing_optional_groups: tuple[str, ...]
    shard_count: int
    shards: tuple[dict[str, Any], ...]
    estimated_restored_file_count: int
    estimated_restored_total_size: int
    restore_roots: Mapping[str, str]
    boundary_status: Mapping[str, bool]
    duckdb_snapshot_status: str
    portability_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_id": self.archive_id,
            "boundary_status": dict(self.boundary_status),
            "created_at_utc": self.created_at_utc,
            "duckdb_snapshot_status": self.duckdb_snapshot_status,
            "estimated_restored_file_count": self.estimated_restored_file_count,
            "estimated_restored_total_size": self.estimated_restored_total_size,
            "logical_groups_included": list(self.logical_groups_included),
            "missing_optional_groups": list(self.missing_optional_groups),
            "portability_status": self.portability_status,
            "restore_roots": dict(sorted(self.restore_roots.items())),
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "shard_count": self.shard_count,
            "shards": [_stable_jsonable(shard) for shard in self.shards],
            "source_profile_path": self.source_profile_path,
            "source_runtime_profile": self.source_runtime_profile,
        }


@dataclass(frozen=True)
class SessionArchiveInspectionResult:
    archive_root: Path
    status: str
    summary: SessionArchiveInspectionSummary
    issues: tuple[SessionArchiveValidationIssue, ...]
    report: dict[str, Any]


def validate_session_archive(
    archive_root: str | Path,
    *,
    verify_checksums: bool = True,
) -> SessionArchiveValidationResult:
    root = Path(archive_root).resolve()
    issues: list[SessionArchiveValidationIssue] = []
    manifest_payload = _load_manifest_payload(root, issues)
    manifest = _validate_manifest_payload(manifest_payload, issues)
    sidecars = _load_sidecars(root, issues)
    inspected_entries: tuple[dict[str, Any], ...] = ()
    checksum_status = "not_requested"

    if manifest is not None:
        _validate_sidecar_archive_ids(manifest.archive_id, sidecars, issues)
        checksum_status = "passed" if verify_checksums else "not_requested"
        inspected_entries = _validate_shards(
            root,
            manifest,
            sidecars.get("checksums.json"),
            verify_checksums,
            issues,
        )
        _validate_archive_index(
            manifest, sidecars.get("archive_index.json"), inspected_entries, issues
        )
        _validate_restore_sidecar(manifest, sidecars.get("restore_plan.json"), issues)
        _add_optional_group_warnings(manifest, issues)
        _add_duckdb_warnings(manifest, issues)
    elif root.exists() and not any(root.iterdir()):
        _issue(
            issues, SessionArchiveIssueCode.EMPTY_ARCHIVE_PACK, "error", "Archive pack is empty."
        )

    status = _status(issues)
    report = _validation_report(
        archive_root=root,
        archive_id=manifest.archive_id if manifest is not None else None,
        status=status,
        verify_checksums=verify_checksums,
        checksum_status="failed"
        if _has_code(issues, SessionArchiveIssueCode.CHECKSUM_MISMATCH)
        else checksum_status,
        issues=tuple(issues),
        manifest=manifest,
        inspected_entries=inspected_entries,
    )
    return SessionArchiveValidationResult(
        archive_root=root,
        status=status,
        archive_id=manifest.archive_id if manifest is not None else None,
        manifest=manifest,
        issues=tuple(issues),
        checksum_status=report["checksum_status"],
        inspected_entries=inspected_entries,
        report=report,
    )


def inspect_session_archive(
    archive_root: str | Path,
    *,
    verify_checksums: bool = True,
) -> SessionArchiveInspectionResult:
    validation = validate_session_archive(archive_root, verify_checksums=verify_checksums)
    summary = _inspection_summary(validation)
    report = {
        "archive_root": validation.archive_root.name,
        "boundaries": SessionArchiveBoundaries().to_dict(),
        "issues": [issue.to_dict() for issue in validation.issues],
        "schema_version": SESSION_ARCHIVE_INSPECTION_REPORT_SCHEMA_VERSION,
        "status": validation.status,
        "summary": summary.to_dict(),
    }
    return SessionArchiveInspectionResult(
        archive_root=validation.archive_root,
        status=validation.status,
        summary=summary,
        issues=validation.issues,
        report=report,
    )


def write_session_archive_validation_report(
    archive_root: str | Path,
    output_path: str | Path | None = None,
    *,
    verify_checksums: bool = True,
) -> Path:
    result = validate_session_archive(archive_root, verify_checksums=verify_checksums)
    path = (
        Path(output_path)
        if output_path is not None
        else result.archive_root / "validation_report.json"
    )
    try:
        return atomic_write_text(path, _deterministic_json(result.report))
    except OSError as exc:
        raise SessionArchiveError("Session archive validation report write failure.") from exc


def write_session_archive_inspection_report(
    archive_root: str | Path,
    output_path: str | Path | None = None,
    *,
    verify_checksums: bool = True,
) -> Path:
    result = inspect_session_archive(archive_root, verify_checksums=verify_checksums)
    path = (
        Path(output_path)
        if output_path is not None
        else result.archive_root / "inspection_report.json"
    )
    try:
        return atomic_write_text(path, _deterministic_json(result.report))
    except OSError as exc:
        raise SessionArchiveError("Session archive inspection report write failure.") from exc


def _load_manifest_payload(
    archive_root: Path,
    issues: list[SessionArchiveValidationIssue],
) -> Mapping[str, Any] | None:
    manifest_path = archive_root / "manifest.json"
    if not manifest_path.is_file():
        _issue(
            issues,
            SessionArchiveIssueCode.MISSING_MANIFEST,
            "error",
            "Session archive manifest.json is missing.",
            "manifest.json",
        )
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _issue(
            issues,
            SessionArchiveIssueCode.MALFORMED_MANIFEST_JSON,
            "error",
            "Session archive manifest.json is not valid JSON.",
            "manifest.json",
        )
        return None
    if not isinstance(payload, Mapping):
        _issue(
            issues,
            SessionArchiveIssueCode.MISSING_REQUIRED_MANIFEST_FIELDS,
            "error",
            "Session archive manifest.json must be a JSON object.",
            "manifest.json",
        )
        return None
    return payload


def _validate_manifest_payload(
    payload: Mapping[str, Any] | None,
    issues: list[SessionArchiveValidationIssue],
) -> SessionArchiveManifest | None:
    if payload is None:
        return None
    try:
        return validate_session_archive_manifest(payload)
    except SessionArchiveError as exc:
        message = str(exc)
        code = _manifest_error_code(message)
        _issue(issues, code, "error", message, "manifest.json")
    return None


def _manifest_error_code(message: str) -> str:
    lowered = message.lower()
    if "unsupported" in lowered and "schema" in lowered:
        return SessionArchiveIssueCode.UNSUPPORTED_SCHEMA_VERSION
    if "logical_group" in lowered:
        return SessionArchiveIssueCode.UNKNOWN_LOGICAL_GROUP
    if "path" in lowered or "uri" in lowered or "url" in lowered or "windows" in lowered:
        return SessionArchiveIssueCode.NON_PORTABLE_PATH
    if "shard" in lowered:
        return SessionArchiveIssueCode.MALFORMED_SHARD_METADATA
    return SessionArchiveIssueCode.MISSING_REQUIRED_MANIFEST_FIELDS


def _load_sidecars(
    archive_root: Path,
    issues: list[SessionArchiveValidationIssue],
) -> dict[str, Mapping[str, Any] | None]:
    sidecars: dict[str, Mapping[str, Any] | None] = {}
    for name in EXPECTED_SIDECARS:
        path = archive_root / name
        if not path.is_file():
            severity = "error" if name == "archive_index.json" else "warning"
            code = (
                SessionArchiveIssueCode.MISSING_SHARD_INDEX
                if name == "archive_index.json"
                else SessionArchiveIssueCode.MALFORMED_SHARD_INDEX
            )
            _issue(issues, code, severity, f"Session archive sidecar is missing: {name}.", name)
            sidecars[name] = None
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            _issue(
                issues,
                SessionArchiveIssueCode.MALFORMED_SHARD_INDEX,
                "error",
                f"Session archive sidecar is not valid JSON: {name}.",
                name,
            )
            sidecars[name] = None
            continue
        if not isinstance(payload, Mapping):
            _issue(
                issues,
                SessionArchiveIssueCode.MALFORMED_SHARD_INDEX,
                "error",
                f"Session archive sidecar must be a JSON object: {name}.",
                name,
            )
            sidecars[name] = None
            continue
        sidecars[name] = payload
    return sidecars


def _validate_sidecar_archive_ids(
    archive_id: str,
    sidecars: Mapping[str, Mapping[str, Any] | None],
    issues: list[SessionArchiveValidationIssue],
) -> None:
    for name, payload in sidecars.items():
        if payload is not None and payload.get("archive_id") not in {None, archive_id}:
            _issue(
                issues,
                SessionArchiveIssueCode.ARCHIVE_INDEX_INCONSISTENCY,
                "error",
                f"Session archive sidecar archive_id mismatch: {name}.",
                name,
            )


def _validate_shards(
    archive_root: Path,
    manifest: SessionArchiveManifest,
    checksums: Mapping[str, Any] | None,
    verify_checksums: bool,
    issues: list[SessionArchiveValidationIssue],
) -> tuple[dict[str, Any], ...]:
    entries: list[dict[str, Any]] = []
    if not manifest.shards:
        _issue(
            issues, SessionArchiveIssueCode.EMPTY_ARCHIVE_PACK, "error", "Manifest has no shards."
        )
        return ()
    sidecar_checksums = _shard_checksums(checksums, issues)
    for shard in sorted(manifest.shards, key=lambda item: item.sort_key()):
        if shard.archive_format != "tar" or shard.compression != "none":
            _issue(
                issues,
                SessionArchiveIssueCode.MALFORMED_SHARD_METADATA,
                "error",
                "Session archive validation supports only archive_format='tar' and compression='none'.",
                shard.shard_path,
            )
            continue
        shard_path = archive_root / "shards" / shard.shard_name
        if not shard_path.is_file():
            _issue(
                issues,
                SessionArchiveIssueCode.MISSING_REQUIRED_SHARD,
                "error",
                f"Session archive required shard is missing: {shard.shard_name}.",
                shard.shard_path,
            )
            continue
        if verify_checksums:
            expected = sidecar_checksums.get(shard.shard_name, shard.checksum)
            digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
            if digest != expected:
                _issue(
                    issues,
                    SessionArchiveIssueCode.CHECKSUM_MISMATCH,
                    "error",
                    f"Session archive shard checksum mismatch: {shard.shard_name}.",
                    shard.shard_path,
                )
        entries.extend(_inspect_shard_entries(shard_path, shard, issues))
    return tuple(sorted(entries, key=lambda item: (item["shard_name"], item["member_path"])))


def _inspect_shard_entries(
    shard_path: Path,
    shard: SessionArchiveShard,
    issues: list[SessionArchiveValidationIssue],
) -> tuple[dict[str, Any], ...]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with tarfile.open(shard_path, mode="r") as archive:
            for member in archive.getmembers():
                member_name = member.name
                if not member.isfile():
                    _issue(
                        issues,
                        SessionArchiveIssueCode.UNSAFE_ARCHIVE_ENTRY,
                        "error",
                        f"Shard {shard.shard_name} contains unsupported non-regular member.",
                        member_name,
                    )
                    continue
                try:
                    member_path = _portable_member_path(member_name)
                except SessionArchiveError as exc:
                    _issue(
                        issues,
                        SessionArchiveIssueCode.UNSAFE_ARCHIVE_ENTRY,
                        "error",
                        str(exc),
                        member_name,
                    )
                    continue
                if member_path in seen:
                    _issue(
                        issues,
                        SessionArchiveIssueCode.UNSAFE_RESTORE_PATH,
                        "error",
                        f"Shard {shard.shard_name} contains duplicate member path: {member_path}.",
                        member_path,
                    )
                    continue
                seen.add(member_path)
                entries.append(
                    {
                        "logical_group": _group_value(shard.logical_group),
                        "member_path": member_path,
                        "shard_name": shard.shard_name,
                        "size_bytes": member.size,
                    }
                )
    except tarfile.TarError:
        _issue(
            issues,
            SessionArchiveIssueCode.MALFORMED_SHARD_METADATA,
            "error",
            f"Session archive shard is not a readable tar: {shard.shard_name}.",
            shard.shard_path,
        )
    return tuple(entries)


def _validate_archive_index(
    manifest: SessionArchiveManifest,
    archive_index: Mapping[str, Any] | None,
    inspected_entries: tuple[dict[str, Any], ...],
    issues: list[SessionArchiveValidationIssue],
) -> None:
    if archive_index is None:
        return
    shards = archive_index.get("shards")
    inventory = archive_index.get("file_inventory")
    groups = archive_index.get("logical_groups")
    if (
        not isinstance(shards, list)
        or not isinstance(inventory, list)
        or not isinstance(groups, Mapping)
    ):
        _issue(
            issues,
            SessionArchiveIssueCode.MALFORMED_SHARD_INDEX,
            "error",
            "archive_index.json must contain shards, file_inventory, and logical_groups.",
            "archive_index.json",
        )
        return
    manifest_shards = {shard.shard_name for shard in manifest.shards}
    index_shards = {item.get("shard_name") for item in shards if isinstance(item, Mapping)}
    if manifest_shards != index_shards:
        _issue(
            issues,
            SessionArchiveIssueCode.ARCHIVE_INDEX_INCONSISTENCY,
            "error",
            "archive_index.json shard list does not match manifest shards.",
            "archive_index.json",
        )
    if len(inventory) != len(inspected_entries):
        _issue(
            issues,
            SessionArchiveIssueCode.ARCHIVE_INDEX_INCONSISTENCY,
            "error",
            "archive_index.json file inventory count does not match shard contents.",
            "archive_index.json",
            {"expected": len(inventory), "actual": len(inspected_entries)},
        )
    inspected_by_group: dict[str, int] = {}
    inspected_size_by_group: dict[str, int] = {}
    for entry in inspected_entries:
        group = str(entry["logical_group"])
        inspected_by_group[group] = inspected_by_group.get(group, 0) + 1
        inspected_size_by_group[group] = inspected_size_by_group.get(group, 0) + int(
            entry["size_bytes"]
        )
    for group, payload in groups.items():
        if group not in {item.value for item in SessionArchiveLogicalGroup}:
            _issue(
                issues,
                SessionArchiveIssueCode.UNKNOWN_LOGICAL_GROUP,
                "error",
                f"archive_index.json contains unknown logical group: {group}.",
                "archive_index.json",
            )
            continue
        if not isinstance(payload, Mapping):
            _issue(
                issues,
                SessionArchiveIssueCode.MALFORMED_SHARD_INDEX,
                "error",
                f"archive_index.json logical group metadata is malformed: {group}.",
                "archive_index.json",
            )
            continue
        if payload.get("file_count") != inspected_by_group.get(str(group), 0):
            _issue(
                issues,
                SessionArchiveIssueCode.ARCHIVE_INDEX_INCONSISTENCY,
                "error",
                f"archive_index.json file_count mismatch for logical group: {group}.",
                "archive_index.json",
            )
        if payload.get("size_bytes") != inspected_size_by_group.get(str(group), 0):
            _issue(
                issues,
                SessionArchiveIssueCode.ARCHIVE_INDEX_INCONSISTENCY,
                "error",
                f"archive_index.json size_bytes mismatch for logical group: {group}.",
                "archive_index.json",
            )


def _validate_restore_sidecar(
    manifest: SessionArchiveManifest,
    restore_sidecar: Mapping[str, Any] | None,
    issues: list[SessionArchiveValidationIssue],
) -> None:
    if restore_sidecar is None:
        return
    roots = restore_sidecar.get("target_relative_roots")
    if roots is None:
        return
    if not isinstance(roots, Mapping):
        _issue(
            issues,
            SessionArchiveIssueCode.UNSAFE_RESTORE_PATH,
            "error",
            "restore_plan.json target_relative_roots must be a mapping.",
            "restore_plan.json",
        )
        return
    for group, path in roots.items():
        if group not in {item.value for item in SessionArchiveLogicalGroup}:
            _issue(
                issues,
                SessionArchiveIssueCode.UNKNOWN_LOGICAL_GROUP,
                "error",
                f"restore_plan.json contains unknown logical group: {group}.",
                "restore_plan.json",
            )
        try:
            _portable_member_path(str(path))
        except SessionArchiveError as exc:
            _issue(
                issues,
                SessionArchiveIssueCode.UNSAFE_RESTORE_PATH,
                "error",
                str(exc),
                "restore_plan.json",
            )
    if dict(sorted(roots.items())) != dict(sorted(manifest.restore.target_relative_roots.items())):
        _issue(
            issues,
            SessionArchiveIssueCode.ARCHIVE_INDEX_INCONSISTENCY,
            "error",
            "restore_plan.json target roots do not match manifest restore expectations.",
            "restore_plan.json",
        )


def _add_optional_group_warnings(
    manifest: SessionArchiveManifest,
    issues: list[SessionArchiveValidationIssue],
) -> None:
    included = {_group_value(group) for group in manifest.included_groups}
    for group in sorted(OPTIONAL_LOGICAL_GROUPS - included):
        _issue(
            issues,
            SessionArchiveIssueCode.MISSING_OPTIONAL_LOGICAL_GROUP,
            "warning",
            f"Optional logical group is not included in this archive pack: {group}.",
            context={"logical_group": group},
        )


def _add_duckdb_warnings(
    manifest: SessionArchiveManifest,
    issues: list[SessionArchiveValidationIssue],
) -> None:
    snapshot = manifest.duckdb_snapshot
    if snapshot is None:
        _issue(
            issues,
            SessionArchiveIssueCode.OPTIONAL_DUCKDB_SNAPSHOT_MISSING,
            "warning",
            "Optional DuckDB snapshot metadata is not present.",
        )
        return
    if snapshot.source_path == ":memory:":
        _issue(
            issues,
            SessionArchiveIssueCode.OPTIONAL_DUCKDB_METADATA_WARNING,
            "warning",
            "DuckDB source is :memory: metadata only; no snapshot file is expected.",
        )
    elif (
        snapshot.included
        and not snapshot.snapshot_path
        and not any(
            _group_value(group) == SessionArchiveLogicalGroup.DUCKDB_SNAPSHOT.value
            for group in manifest.included_groups
        )
    ):
        _issue(
            issues,
            SessionArchiveIssueCode.OPTIONAL_DUCKDB_METADATA_WARNING,
            "warning",
            "DuckDB metadata says a snapshot is included but no duckdb_snapshot group is present.",
        )


def _shard_checksums(
    checksums: Mapping[str, Any] | None,
    issues: list[SessionArchiveValidationIssue],
) -> dict[str, str]:
    values: dict[str, str] = {}
    if checksums is None:
        return values
    shard_rows = checksums.get("shards", [])
    if not isinstance(shard_rows, list):
        _issue(
            issues,
            SessionArchiveIssueCode.MALFORMED_SHARD_INDEX,
            "error",
            "checksums.json shards must be a list.",
            "checksums.json",
        )
        return values
    for row in shard_rows:
        if not isinstance(row, Mapping):
            _issue(
                issues,
                SessionArchiveIssueCode.MALFORMED_SHARD_INDEX,
                "error",
                "checksums.json shard entries must be mappings.",
                "checksums.json",
            )
            continue
        name = row.get("shard_name")
        checksum = row.get("checksum")
        if isinstance(name, str) and isinstance(checksum, str):
            values[name] = checksum
    return values


def _portable_member_path(value: str) -> str:
    text = value.strip()
    if not text or "\\" in text or text.startswith("~") or _URL_LIKE_PATTERN.match(text):
        raise SessionArchiveError("Session archive path must be repository-relative POSIX.")
    if _WINDOWS_DRIVE_PREFIX_PATTERN.match(text):
        raise SessionArchiveError("Session archive path must not use a Windows drive path.")
    path = PurePosixPath(text)
    first_part = path.parts[0] if path.parts else ""
    if (
        path.is_absolute()
        or PureWindowsPath(text).is_absolute()
        or (len(first_part) == 2 and first_part[1] == ":")
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != text
    ):
        raise SessionArchiveError(
            "Session archive path must be normalized repository-relative POSIX."
        )
    return path.as_posix()


def _inspection_summary(
    validation: SessionArchiveValidationResult,
) -> SessionArchiveInspectionSummary:
    manifest = validation.manifest
    if manifest is None:
        return SessionArchiveInspectionSummary(
            archive_id=None,
            schema_version=None,
            created_at_utc=None,
            session_id=None,
            source_runtime_profile=None,
            source_profile_path=None,
            logical_groups_included=(),
            missing_optional_groups=(),
            shard_count=0,
            shards=(),
            estimated_restored_file_count=0,
            estimated_restored_total_size=0,
            restore_roots={},
            boundary_status=SessionArchiveBoundaries().to_dict(),
            duckdb_snapshot_status="unknown",
            portability_status="failed",
        )
    included = tuple(sorted(_group_value(group) for group in manifest.included_groups))
    missing = tuple(sorted(OPTIONAL_LOGICAL_GROUPS - set(included)))
    return SessionArchiveInspectionSummary(
        archive_id=manifest.archive_id,
        schema_version=manifest.schema_version,
        created_at_utc=manifest.created_at_utc,
        session_id=manifest.session_id,
        source_runtime_profile=manifest.source_runtime_profile,
        source_profile_path=manifest.source_profile_path,
        logical_groups_included=included,
        missing_optional_groups=missing,
        shard_count=len(manifest.shards),
        shards=tuple(
            {
                "archive_format": shard.archive_format,
                "compression": shard.compression,
                "file_count": shard.file_count,
                "logical_group": _group_value(shard.logical_group),
                "shard_name": shard.shard_name,
                "size_bytes": shard.size_bytes,
            }
            for shard in sorted(manifest.shards, key=lambda item: item.sort_key())
        ),
        estimated_restored_file_count=len(validation.inspected_entries),
        estimated_restored_total_size=sum(
            int(entry["size_bytes"]) for entry in validation.inspected_entries
        ),
        restore_roots=manifest.restore.target_relative_roots,
        boundary_status=manifest.boundaries.to_dict(),
        duckdb_snapshot_status=_duckdb_status(manifest),
        portability_status="failed"
        if any(issue.severity == "error" for issue in validation.issues)
        else "portable",
    )


def _duckdb_status(manifest: SessionArchiveManifest) -> str:
    if manifest.duckdb_snapshot is None:
        return "not_present"
    if manifest.duckdb_snapshot.source_path == ":memory:":
        return "memory_metadata_only"
    if manifest.duckdb_snapshot.included:
        return "file_snapshot_metadata_present"
    return "metadata_present_not_included"


def _validation_report(
    *,
    archive_root: Path,
    archive_id: str | None,
    status: str,
    verify_checksums: bool,
    checksum_status: str,
    issues: tuple[SessionArchiveValidationIssue, ...],
    manifest: SessionArchiveManifest | None,
    inspected_entries: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    return {
        "archive_id": archive_id,
        "archive_root": archive_root.name,
        "boundaries": SessionArchiveBoundaries().to_dict(),
        "checksum_status": checksum_status,
        "estimated_restored_file_count": len(inspected_entries),
        "estimated_restored_total_size": sum(
            int(entry["size_bytes"]) for entry in inspected_entries
        ),
        "included_groups": []
        if manifest is None
        else sorted(_group_value(group) for group in manifest.included_groups),
        "issues": [issue.to_dict() for issue in issues],
        "manifest_metadata": {} if manifest is None else _stable_jsonable(manifest.metadata),
        "schema_version": SESSION_ARCHIVE_VALIDATION_REPORT_SCHEMA_VERSION,
        "status": status,
        "verify_checksums": verify_checksums,
    }


def _issue(
    issues: list[SessionArchiveValidationIssue],
    code: str,
    severity: str,
    message: str,
    path: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> None:
    issues.append(
        SessionArchiveValidationIssue(
            code=code,
            severity=severity,
            message=message,
            path=path,
            context=context,
        )
    )


def _status(issues: list[SessionArchiveValidationIssue]) -> str:
    if any(issue.severity == "error" for issue in issues):
        return "failed"
    if any(issue.severity == "warning" for issue in issues):
        return "warning"
    return "passed"


def _has_code(issues: list[SessionArchiveValidationIssue], code: str) -> bool:
    return any(issue.code == code for issue in issues)


def _group_value(value: SessionArchiveLogicalGroup | str) -> str:
    if isinstance(value, SessionArchiveLogicalGroup):
        return value.value
    return str(value)


def _stable_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stable_jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_stable_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_stable_jsonable(item) for item in value]
    return value


def _deterministic_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
