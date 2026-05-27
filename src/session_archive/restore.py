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
    SessionArchiveManifest,
    SessionArchiveShard,
    validate_session_archive_manifest,
)

SUPPORTED_RESTORE_OVERWRITE_POLICIES = frozenset(
    {"fail_if_exists", "skip_existing", "replace_existing"}
)
RESTORE_REPORT_SCHEMA_VERSION = 1
_URL_LIKE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_WINDOWS_DRIVE_PREFIX_PATTERN = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True)
class SessionArchiveRestoreRequest:
    archive_root: str | Path
    target_root: str | Path
    overwrite_policy: str = "fail_if_exists"
    verify_checksums: bool = True
    write_report: bool = True
    report_root: str | Path | None = None


@dataclass(frozen=True)
class SessionArchiveRestoreEntry:
    shard_name: str
    member_path: str
    target_path: str
    size_bytes: int
    checksum_algorithm: str | None = None
    checksum: str | None = None
    action: str = "restore"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "action": self.action,
            "member_path": self.member_path,
            "shard_name": self.shard_name,
            "size_bytes": self.size_bytes,
            "target_path": self.target_path,
        }
        if self.checksum_algorithm is not None:
            data["checksum_algorithm"] = self.checksum_algorithm
        if self.checksum is not None:
            data["checksum"] = self.checksum
        return data


@dataclass(frozen=True)
class SessionArchiveRestorePlan:
    archive_id: str
    archive_root: Path
    target_root: Path
    overwrite_policy: str
    verify_checksums: bool
    manifest: SessionArchiveManifest
    shards: tuple[SessionArchiveShard, ...]
    restore_entries: tuple[SessionArchiveRestoreEntry, ...]
    skipped_entries: tuple[SessionArchiveRestoreEntry, ...]
    warnings: tuple[str, ...]
    checksum_status: str
    report: dict[str, Any]


@dataclass(frozen=True)
class SessionArchiveRestoreResult:
    plan: SessionArchiveRestorePlan
    restored_paths: tuple[Path, ...]
    skipped_paths: tuple[Path, ...]
    report_path: Path | None


def build_session_archive_restore_plan(
    request: SessionArchiveRestoreRequest,
) -> SessionArchiveRestorePlan:
    archive_root = Path(request.archive_root).resolve()
    target_root = Path(request.target_root).resolve()
    overwrite_policy = _overwrite_policy(request.overwrite_policy)
    manifest = _load_manifest(archive_root)
    archive_index = _load_optional_json(archive_root / "archive_index.json")
    checksums = _load_optional_json(archive_root / "checksums.json")
    restore_sidecar = _load_optional_json(archive_root / "restore_plan.json")
    _validate_sidecar_archive_id(manifest.archive_id, archive_index, "archive_index.json")
    _validate_sidecar_archive_id(manifest.archive_id, checksums, "checksums.json")
    _validate_sidecar_archive_id(manifest.archive_id, restore_sidecar, "restore_plan.json")

    checksum_status = "not_requested"
    if request.verify_checksums:
        _verify_shard_checksums(archive_root, manifest, checksums)
        checksum_status = "passed"

    file_checksums = _file_checksums(checksums)
    restore_entries: list[SessionArchiveRestoreEntry] = []
    skipped_entries: list[SessionArchiveRestoreEntry] = []
    warnings: list[str] = []
    for shard in sorted(manifest.shards, key=lambda item: item.sort_key()):
        _validate_supported_shard(shard)
        shard_path = _shard_path(archive_root, shard)
        if not shard_path.is_file():
            raise SessionArchiveError(
                f"Session archive required shard is missing: {shard.shard_name}."
            )
        for member in _inspect_tar_members(shard_path, shard.shard_name):
            target_path = (target_root / member.member_path).resolve()
            _ensure_under_root(target_path, target_root, "target_path")
            checksum = file_checksums.get(member.member_path, {})
            entry = SessionArchiveRestoreEntry(
                shard_name=shard.shard_name,
                member_path=member.member_path,
                target_path=member.member_path,
                size_bytes=member.size_bytes,
                checksum_algorithm=checksum.get("checksum_algorithm"),
                checksum=checksum.get("checksum"),
                action="restore",
            )
            if target_path.exists():
                if overwrite_policy == "fail_if_exists":
                    raise SessionArchiveError(
                        "Session archive restore target file exists under fail_if_exists: "
                        f"{member.member_path}."
                    )
                if overwrite_policy == "skip_existing":
                    skipped_entries.append(
                        SessionArchiveRestoreEntry(**{**entry.to_dict(), "action": "skip_existing"})
                    )
                    continue
                if target_path.is_dir():
                    raise SessionArchiveError(
                        "Session archive restore target is a directory and cannot be replaced: "
                        f"{member.member_path}."
                    )
            restore_entries.append(entry)

    report = _restore_report(
        archive_id=manifest.archive_id,
        archive_root=archive_root,
        target_root=target_root,
        overwrite_policy=overwrite_policy,
        verify_checksums=request.verify_checksums,
        checksum_status=checksum_status,
        restored_entries=tuple(restore_entries),
        skipped_entries=tuple(skipped_entries),
        warnings=tuple(warnings),
        manifest=manifest,
    )
    return SessionArchiveRestorePlan(
        archive_id=manifest.archive_id,
        archive_root=archive_root,
        target_root=target_root,
        overwrite_policy=overwrite_policy,
        verify_checksums=request.verify_checksums,
        manifest=manifest,
        shards=tuple(sorted(manifest.shards, key=lambda item: item.sort_key())),
        restore_entries=tuple(restore_entries),
        skipped_entries=tuple(skipped_entries),
        warnings=tuple(warnings),
        checksum_status=checksum_status,
        report=report,
    )


def restore_session_archive_pack(
    request: SessionArchiveRestoreRequest,
) -> SessionArchiveRestoreResult:
    plan = build_session_archive_restore_plan(request)
    restored_paths: list[Path] = []
    skipped_paths = [plan.target_root / entry.target_path for entry in plan.skipped_entries]
    entries_by_shard: dict[str, list[SessionArchiveRestoreEntry]] = {}
    for entry in plan.restore_entries:
        entries_by_shard.setdefault(entry.shard_name, []).append(entry)

    for shard in plan.shards:
        shard_entries = entries_by_shard.get(shard.shard_name, [])
        if not shard_entries:
            continue
        wanted = {entry.member_path: entry for entry in shard_entries}
        with tarfile.open(_shard_path(plan.archive_root, shard), mode="r") as archive:
            for member in archive.getmembers():
                if member.name not in wanted:
                    continue
                _validate_regular_member(member, shard.shard_name)
                data = archive.extractfile(member)
                if data is None:
                    raise SessionArchiveError(
                        f"Session archive member cannot be read: {member.name}."
                    )
                content = data.read()
                entry = wanted[member.name]
                if entry.checksum:
                    digest = hashlib.sha256(content).hexdigest()
                    if digest != entry.checksum:
                        raise SessionArchiveError(
                            f"Session archive restored file checksum mismatch: {member.name}."
                        )
                target_path = (plan.target_root / entry.target_path).resolve()
                _ensure_under_root(target_path, plan.target_root, "target_path")
                _atomic_write_bytes(target_path, content)
                restored_paths.append(target_path)

    report_path = None
    if request.write_report:
        report_root = _report_root(request, plan)
        report_path = report_root / "restore_report.json"
        atomic_write_text(report_path, _deterministic_json(plan.report))
    return SessionArchiveRestoreResult(
        plan=plan,
        restored_paths=tuple(restored_paths),
        skipped_paths=tuple(skipped_paths),
        report_path=report_path,
    )


def _load_manifest(archive_root: Path) -> SessionArchiveManifest:
    manifest_path = archive_root / "manifest.json"
    if not manifest_path.is_file():
        raise SessionArchiveError("Session archive manifest.json is missing.")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SessionArchiveError("Session archive manifest.json is not valid JSON.") from exc
    return validate_session_archive_manifest(payload)


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SessionArchiveError(
            f"Session archive sidecar is not valid JSON: {path.name}."
        ) from exc
    if not isinstance(payload, dict):
        raise SessionArchiveError(f"Session archive sidecar must be a JSON object: {path.name}.")
    return payload


def _validate_sidecar_archive_id(
    archive_id: str,
    payload: Mapping[str, Any] | None,
    name: str,
) -> None:
    if payload is not None and payload.get("archive_id") not in {None, archive_id}:
        raise SessionArchiveError(f"Session archive sidecar archive_id mismatch: {name}.")


def _verify_shard_checksums(
    archive_root: Path,
    manifest: SessionArchiveManifest,
    checksums: Mapping[str, Any] | None,
) -> None:
    sidecar_checksums = _shard_checksums(checksums)
    for shard in sorted(manifest.shards, key=lambda item: item.sort_key()):
        shard_path = _shard_path(archive_root, shard)
        if not shard_path.is_file():
            raise SessionArchiveError(
                f"Session archive required shard is missing: {shard.shard_name}."
            )
        expected = sidecar_checksums.get(shard.shard_name, shard.checksum)
        digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        if digest != expected:
            raise SessionArchiveError(
                f"Session archive shard checksum mismatch: {shard.shard_name}."
            )


def _shard_checksums(checksums: Mapping[str, Any] | None) -> dict[str, str]:
    values: dict[str, str] = {}
    if checksums is None:
        return values
    for item in checksums.get("shards", []):
        if isinstance(item, Mapping) and isinstance(item.get("shard_name"), str):
            checksum = item.get("checksum")
            if isinstance(checksum, str):
                values[item["shard_name"]] = checksum
    return values


def _file_checksums(checksums: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    values: dict[str, dict[str, str]] = {}
    if checksums is None:
        return values
    for item in checksums.get("files", []):
        if not isinstance(item, Mapping) or not isinstance(item.get("archive_member_path"), str):
            continue
        checksum = item.get("checksum")
        algorithm = item.get("checksum_algorithm")
        if isinstance(checksum, str) and isinstance(algorithm, str):
            values[item["archive_member_path"]] = {
                "checksum": checksum,
                "checksum_algorithm": algorithm,
            }
    return values


@dataclass(frozen=True)
class _TarMember:
    member_path: str
    size_bytes: int


def _inspect_tar_members(shard_path: Path, shard_name: str) -> tuple[_TarMember, ...]:
    members: list[_TarMember] = []
    try:
        with tarfile.open(shard_path, mode="r") as archive:
            for member in archive.getmembers():
                _validate_regular_member(member, shard_name)
                member_path = _portable_member_path(member.name)
                members.append(_TarMember(member_path=member_path, size_bytes=member.size))
    except tarfile.TarError as exc:
        raise SessionArchiveError(
            f"Session archive shard is not a readable tar: {shard_name}."
        ) from exc
    return tuple(sorted(members, key=lambda item: item.member_path))


def _validate_regular_member(member: tarfile.TarInfo, shard_name: str) -> None:
    if not member.isfile():
        raise SessionArchiveError(
            f"Session archive shard {shard_name} contains unsupported non-regular member: "
            f"{member.name}."
        )
    _portable_member_path(member.name)


def _portable_member_path(value: str) -> str:
    text = value.strip()
    if not text or "\\" in text or text.startswith("~") or _URL_LIKE_PATTERN.match(text):
        raise SessionArchiveError("Session archive member path must be repository-relative POSIX.")
    if _WINDOWS_DRIVE_PREFIX_PATTERN.match(text):
        raise SessionArchiveError("Session archive member path must not use a Windows drive path.")
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
            "Session archive member path must be normalized repository-relative POSIX."
        )
    return path.as_posix()


def _validate_supported_shard(shard: SessionArchiveShard) -> None:
    if shard.archive_format != "tar" or shard.compression != "none":
        raise SessionArchiveError(
            "Session archive restore supports only archive_format='tar' and compression='none'."
        )


def _shard_path(archive_root: Path, shard: SessionArchiveShard) -> Path:
    return archive_root / "shards" / shard.shard_name


def _overwrite_policy(value: str) -> str:
    if value not in SUPPORTED_RESTORE_OVERWRITE_POLICIES:
        raise SessionArchiveError(
            "Session archive restore overwrite_policy must be one of "
            f"{sorted(SUPPORTED_RESTORE_OVERWRITE_POLICIES)}."
        )
    return value


def _restore_report(
    *,
    archive_id: str,
    archive_root: Path,
    target_root: Path,
    overwrite_policy: str,
    verify_checksums: bool,
    checksum_status: str,
    restored_entries: tuple[SessionArchiveRestoreEntry, ...],
    skipped_entries: tuple[SessionArchiveRestoreEntry, ...],
    warnings: tuple[str, ...],
    manifest: SessionArchiveManifest,
) -> dict[str, Any]:
    return {
        "archive_id": archive_id,
        "boundaries": SessionArchiveBoundaries().to_dict(),
        "checksum_status": checksum_status,
        "manifest_metadata": dict(manifest.metadata),
        "overwrite_policy": overwrite_policy,
        "restored_entries": [entry.to_dict() for entry in restored_entries],
        "schema_version": RESTORE_REPORT_SCHEMA_VERSION,
        "skipped_entries": [entry.to_dict() for entry in skipped_entries],
        "source_archive_root": archive_root.name,
        "target_root": target_root.name,
        "verify_checksums": verify_checksums,
        "warnings": list(warnings),
    }


def _report_root(request: SessionArchiveRestoreRequest, plan: SessionArchiveRestorePlan) -> Path:
    if request.report_root is not None:
        root = Path(request.report_root).resolve()
        _ensure_under_root(root, plan.target_root, "report_root")
        root.mkdir(parents=True, exist_ok=True)
        return root
    root = plan.target_root / "artifacts" / "_derived" / "session_archives" / plan.archive_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ensure_under_root(path: Path, root: Path, field_name: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SessionArchiveError(
            f"Session archive restore field '{field_name}' must stay under target root."
        ) from exc


def _deterministic_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _atomic_write_bytes(path: Path, data: bytes) -> Path:
    if path.exists() and path.is_dir():
        raise SessionArchiveError("Session archive restore target is a directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_bytes(data)
    temp_path.replace(path)
    return path
