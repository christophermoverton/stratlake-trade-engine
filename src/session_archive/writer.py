from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tarfile
from typing import Any, Mapping, Sequence

from src.artifacts.safety import atomic_write_text
from src.session_archive.manifest import (
    SESSION_ARCHIVE_MANIFEST_SCHEMA_VERSION,
    SessionArchiveBoundaries,
    SessionArchiveDuckDBSnapshot,
    SessionArchiveError,
    SessionArchiveLogicalGroup,
    SessionArchiveManifest,
    SessionArchiveRestoreExpectations,
    SessionArchiveShard,
    manifest_to_deterministic_json,
    validate_session_archive_manifest,
)

DEFAULT_ARCHIVE_ROOT = "artifacts/_derived/session_archives"
DEFAULT_MAX_SHARD_SIZE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_ENTRIES_PER_SHARD = 1000
DEFAULT_INCLUDE_PATHS = {
    SessionArchiveLogicalGroup.FEATURES: ("data/curated",),
    SessionArchiveLogicalGroup.ARTIFACTS: ("artifacts",),
    SessionArchiveLogicalGroup.CONFIGS: ("configs",),
}
DEFAULT_EXCLUDE_PATTERNS = (
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    ".DS_Store",
    "*.tmp",
    "*.temp",
    "artifacts/_derived/session_archives",
    "artifacts/_derived/session_archives/*",
)
_URL_LIKE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_WINDOWS_DRIVE_PREFIX_PATTERN = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True)
class SessionArchiveIncludePolicy:
    include_groups: tuple[SessionArchiveLogicalGroup | str, ...] = (
        SessionArchiveLogicalGroup.FEATURES,
        SessionArchiveLogicalGroup.ARTIFACTS,
        SessionArchiveLogicalGroup.CONFIGS,
    )
    include_paths: Mapping[SessionArchiveLogicalGroup | str, Sequence[str]] = field(
        default_factory=dict
    )
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS


@dataclass(frozen=True)
class SessionArchiveWriteRequest:
    archive_id: str
    repository_root: str | Path
    output_root: str | Path = DEFAULT_ARCHIVE_ROOT
    include_policy: SessionArchiveIncludePolicy = field(default_factory=SessionArchiveIncludePolicy)
    max_shard_size_bytes: int = DEFAULT_MAX_SHARD_SIZE_BYTES
    max_entries_per_shard: int = DEFAULT_MAX_ENTRIES_PER_SHARD
    session_id: str | None = None
    source_runtime_profile: str | None = None
    source_profile_path: str | None = None
    duckdb_snapshot_source_path: str | None = None
    duckdb_snapshot_description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionArchiveShardEntry:
    logical_group: SessionArchiveLogicalGroup
    source_file_path: Path
    source_path: str
    archive_member_path: str
    size_bytes: int
    checksum_algorithm: str
    checksum: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_member_path": self.archive_member_path,
            "checksum": self.checksum,
            "checksum_algorithm": self.checksum_algorithm,
            "logical_group": self.logical_group.value,
            "size_bytes": self.size_bytes,
            "source_path": self.source_path,
        }


@dataclass(frozen=True)
class SessionArchiveShardPlan:
    logical_group: SessionArchiveLogicalGroup
    shard_index: int
    shard_name: str
    shard_path: str
    entries: tuple[SessionArchiveShardEntry, ...]
    size_bytes: int
    checksum_algorithm: str
    checksum: str
    archive_format: str = "tar"
    compression: str = "none"

    @property
    def file_count(self) -> int:
        return len(self.entries)

    def to_manifest_shard(self) -> SessionArchiveShard:
        return SessionArchiveShard(
            shard_name=self.shard_name,
            shard_path=self.shard_path,
            logical_group=self.logical_group,
            shard_index=self.shard_index,
            file_count=self.file_count,
            size_bytes=self.size_bytes,
            checksum_algorithm=self.checksum_algorithm,
            checksum=self.checksum,
            archive_format=self.archive_format,
            compression=self.compression,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_format": self.archive_format,
            "checksum": self.checksum,
            "checksum_algorithm": self.checksum_algorithm,
            "compression": self.compression,
            "entries": [entry.to_dict() for entry in self.entries],
            "file_count": self.file_count,
            "logical_group": self.logical_group.value,
            "shard_index": self.shard_index,
            "shard_name": self.shard_name,
            "shard_path": self.shard_path,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class SessionArchivePlan:
    request: SessionArchiveWriteRequest
    repository_root: Path
    archive_root: Path
    entries: tuple[SessionArchiveShardEntry, ...]
    shards: tuple[SessionArchiveShardPlan, ...]
    manifest: SessionArchiveManifest
    archive_index: dict[str, Any]
    checksums: dict[str, Any]
    restore_plan: dict[str, Any]


@dataclass(frozen=True)
class SessionArchiveWriteResult:
    archive_root: Path
    manifest_path: Path
    archive_index_path: Path
    checksums_path: Path
    restore_plan_path: Path
    shard_paths: tuple[Path, ...]
    plan: SessionArchivePlan


def build_session_archive_plan(request: SessionArchiveWriteRequest) -> SessionArchivePlan:
    repository_root = Path(request.repository_root).resolve()
    if not repository_root.exists() or not repository_root.is_dir():
        raise SessionArchiveError("Session archive repository_root must be an existing directory.")
    _safe_archive_id(request.archive_id)
    output_root = _resolve_repository_relative_path(
        request.output_root,
        repository_root=repository_root,
        field_name="output_root",
    )
    archive_root = output_root / request.archive_id
    archive_root_relative = _relative_to_repository(archive_root, repository_root, "archive_root")
    include_groups = _include_groups(request.include_policy.include_groups)
    if not include_groups:
        raise SessionArchiveError("Session archive include_groups must not be empty.")
    if request.max_shard_size_bytes <= 0:
        raise SessionArchiveError("Session archive max_shard_size_bytes must be positive.")
    if request.max_entries_per_shard <= 0:
        raise SessionArchiveError("Session archive max_entries_per_shard must be positive.")

    include_paths = _include_paths(request.include_policy.include_paths)
    entries = _collect_entries(
        repository_root=repository_root,
        archive_root=archive_root,
        include_groups=include_groups,
        include_paths=include_paths,
        exclude_patterns=request.include_policy.exclude_patterns,
    )
    duckdb_entry = _duckdb_entry(request, repository_root, include_groups)
    if duckdb_entry is not None:
        entries = tuple(
            sorted(
                (*entries, duckdb_entry),
                key=lambda entry: (entry.logical_group.value, entry.source_path),
            )
        )
    if not entries:
        raise SessionArchiveError("Session archive plan must include at least one file.")

    grouped_entries = {
        group: tuple(entry for entry in entries if entry.logical_group is group)
        for group in include_groups
        if any(entry.logical_group is group for entry in entries)
    }
    shards = tuple(
        shard
        for group in include_groups
        for shard in _plan_group_shards(
            group=group,
            entries=grouped_entries.get(group, ()),
            archive_root_relative=archive_root_relative,
            max_shard_size_bytes=request.max_shard_size_bytes,
            max_entries_per_shard=request.max_entries_per_shard,
        )
    )
    if not shards:
        raise SessionArchiveError("Session archive plan must include at least one shard.")

    source_roots = _source_roots(
        include_groups,
        include_paths,
        request.duckdb_snapshot_source_path,
    )
    restore = SessionArchiveRestoreExpectations(
        target_relative_roots=source_roots,
        overwrite_policy="fail_if_exists",
        compatibility={
            "archive_format": "tar",
            "compression": "none",
            "minimum_manifest_schema_version": SESSION_ARCHIVE_MANIFEST_SCHEMA_VERSION,
        },
    )
    duckdb_snapshot = _duckdb_snapshot(request, repository_root)
    manifest = SessionArchiveManifest(
        schema_version=SESSION_ARCHIVE_MANIFEST_SCHEMA_VERSION,
        archive_id=request.archive_id,
        session_id=request.session_id,
        source_runtime_profile=request.source_runtime_profile,
        source_profile_path=_optional_portable_path(
            request.source_profile_path, "source_profile_path"
        ),
        source_roots=source_roots,
        included_groups=tuple(group for group in grouped_entries),
        shards=tuple(shard.to_manifest_shard() for shard in shards),
        restore=restore,
        boundaries=SessionArchiveBoundaries(),
        duckdb_snapshot=duckdb_snapshot,
        metadata={
            "writer": "session_archive.writer",
            "artifact_role": "derived_transport_snapshot",
            **dict(request.metadata),
        },
    )
    validate_session_archive_manifest(manifest)
    archive_index = _archive_index(request.archive_id, grouped_entries, shards)
    checksums = _checksums(request.archive_id, entries, shards)
    restore_plan = _restore_plan(request.archive_id, restore)
    return SessionArchivePlan(
        request=request,
        repository_root=repository_root,
        archive_root=archive_root,
        entries=entries,
        shards=shards,
        manifest=manifest,
        archive_index=archive_index,
        checksums=checksums,
        restore_plan=restore_plan,
    )


def write_session_archive_pack(request: SessionArchiveWriteRequest) -> SessionArchiveWriteResult:
    plan = build_session_archive_plan(request)
    shards_root = plan.archive_root / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []
    for shard in plan.shards:
        shard_path = plan.repository_root / shard.shard_path
        _ensure_under_root(shard_path, plan.archive_root, "shard_path")
        _atomic_write_bytes(shard_path, _tar_bytes(shard.entries))
        shard_paths.append(shard_path)

    manifest_path = plan.archive_root / "manifest.json"
    archive_index_path = plan.archive_root / "archive_index.json"
    checksums_path = plan.archive_root / "checksums.json"
    restore_plan_path = plan.archive_root / "restore_plan.json"
    atomic_write_text(manifest_path, manifest_to_deterministic_json(plan.manifest))
    atomic_write_text(archive_index_path, _deterministic_json(plan.archive_index))
    atomic_write_text(checksums_path, _deterministic_json(plan.checksums))
    atomic_write_text(restore_plan_path, _deterministic_json(plan.restore_plan))
    return SessionArchiveWriteResult(
        archive_root=plan.archive_root,
        manifest_path=manifest_path,
        archive_index_path=archive_index_path,
        checksums_path=checksums_path,
        restore_plan_path=restore_plan_path,
        shard_paths=tuple(shard_paths),
        plan=plan,
    )


def _collect_entries(
    *,
    repository_root: Path,
    archive_root: Path,
    include_groups: tuple[SessionArchiveLogicalGroup, ...],
    include_paths: Mapping[SessionArchiveLogicalGroup, tuple[str, ...]],
    exclude_patterns: tuple[str, ...],
) -> tuple[SessionArchiveShardEntry, ...]:
    entries: list[SessionArchiveShardEntry] = []
    seen: set[str] = set()
    for group in include_groups:
        if group is SessionArchiveLogicalGroup.DUCKDB_SNAPSHOT:
            continue
        roots = include_paths.get(group, DEFAULT_INCLUDE_PATHS.get(group, ()))
        for root_text in roots:
            root = _resolve_repository_relative_path(
                root_text,
                repository_root=repository_root,
                field_name=f"include_paths.{group.value}",
            )
            if not root.exists():
                continue
            candidates = [root] if root.is_file() else sorted(root.rglob("*"), key=_path_sort_key)
            for candidate in candidates:
                if candidate.is_dir():
                    continue
                resolved = candidate.resolve()
                _ensure_under_root(resolved, repository_root, "source_path")
                if _is_relative_to(resolved, archive_root):
                    continue
                relative = _relative_to_repository(candidate, repository_root, "source_path")
                if _is_excluded(relative, exclude_patterns):
                    continue
                if candidate.is_symlink():
                    _ensure_under_root(resolved, repository_root, "source_path")
                if not candidate.is_file():
                    continue
                if relative in seen:
                    continue
                seen.add(relative)
                content = candidate.read_bytes()
                entries.append(
                    SessionArchiveShardEntry(
                        logical_group=group,
                        source_file_path=candidate,
                        source_path=relative,
                        archive_member_path=relative,
                        size_bytes=len(content),
                        checksum_algorithm="sha256",
                        checksum=hashlib.sha256(content).hexdigest(),
                    )
                )
    return tuple(sorted(entries, key=lambda entry: (entry.logical_group.value, entry.source_path)))


def _plan_group_shards(
    *,
    group: SessionArchiveLogicalGroup,
    entries: tuple[SessionArchiveShardEntry, ...],
    archive_root_relative: str,
    max_shard_size_bytes: int,
    max_entries_per_shard: int,
) -> tuple[SessionArchiveShardPlan, ...]:
    if not entries:
        return ()
    buckets: list[list[SessionArchiveShardEntry]] = [[]]
    current_size = 0
    for entry in entries:
        current_bucket = buckets[-1]
        would_exceed_size = (
            current_bucket and current_size + entry.size_bytes > max_shard_size_bytes
        )
        would_exceed_count = len(current_bucket) >= max_entries_per_shard
        if would_exceed_size or would_exceed_count:
            buckets.append([])
            current_size = 0
            current_bucket = buckets[-1]
        current_bucket.append(entry)
        current_size += entry.size_bytes

    plans: list[SessionArchiveShardPlan] = []
    for index, bucket in enumerate(buckets):
        shard_name = f"{group.value}__{index:03d}.tar"
        shard_path = f"{archive_root_relative}/shards/{shard_name}"
        data = _tar_bytes(tuple(bucket))
        plans.append(
            SessionArchiveShardPlan(
                logical_group=group,
                shard_index=index,
                shard_name=shard_name,
                shard_path=shard_path,
                entries=tuple(bucket),
                size_bytes=len(data),
                checksum_algorithm="sha256",
                checksum=hashlib.sha256(data).hexdigest(),
            )
        )
    return tuple(plans)


def _tar_bytes(entries: tuple[SessionArchiveShardEntry, ...]) -> bytes:
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for entry in sorted(entries, key=lambda item: item.archive_member_path):
            data = entry.source_file_path.read_bytes()
            info = tarfile.TarInfo(entry.archive_member_path)
            info.size = len(data)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o644
            archive.addfile(info, BytesIO(data))
    return buffer.getvalue()


def _archive_index(
    archive_id: str,
    grouped_entries: Mapping[SessionArchiveLogicalGroup, tuple[SessionArchiveShardEntry, ...]],
    shards: tuple[SessionArchiveShardPlan, ...],
) -> dict[str, Any]:
    shard_by_entry = {
        entry.source_path: shard.shard_name for shard in shards for entry in shard.entries
    }
    return _derived_payload(
        archive_id,
        {
            "logical_groups": {
                group.value: {
                    "file_count": len(entries),
                    "included_source_paths": [entry.source_path for entry in entries],
                    "size_bytes": sum(entry.size_bytes for entry in entries),
                }
                for group, entries in sorted(
                    grouped_entries.items(), key=lambda item: item[0].value
                )
            },
            "file_inventory": [
                {**entry.to_dict(), "shard_name": shard_by_entry[entry.source_path]}
                for entries in grouped_entries.values()
                for entry in entries
            ],
            "shards": [shard.to_dict() for shard in shards],
        },
    )


def _checksums(
    archive_id: str,
    entries: tuple[SessionArchiveShardEntry, ...],
    shards: tuple[SessionArchiveShardPlan, ...],
) -> dict[str, Any]:
    return _derived_payload(
        archive_id,
        {
            "checksum_algorithm": "sha256",
            "files": [entry.to_dict() for entry in entries],
            "shards": [
                {
                    "checksum": shard.checksum,
                    "checksum_algorithm": shard.checksum_algorithm,
                    "shard_name": shard.shard_name,
                    "shard_path": shard.shard_path,
                    "size_bytes": shard.size_bytes,
                }
                for shard in shards
            ],
        },
    )


def _restore_plan(
    archive_id: str,
    restore: SessionArchiveRestoreExpectations,
) -> dict[str, Any]:
    return _derived_payload(
        archive_id,
        {
            "compatibility": restore.to_dict()["compatibility"],
            "overwrite_policy": restore.overwrite_policy,
            "target_relative_roots": dict(sorted(restore.target_relative_roots.items())),
        },
    )


def _derived_payload(archive_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "archive_id": archive_id,
        "boundaries": SessionArchiveBoundaries().to_dict(),
        "schema_version": 1,
        **dict(payload),
    }


def _source_roots(
    include_groups: tuple[SessionArchiveLogicalGroup, ...],
    include_paths: Mapping[SessionArchiveLogicalGroup, tuple[str, ...]],
    duckdb_snapshot_source_path: str | None = None,
) -> dict[str, str]:
    roots: dict[str, str] = {}
    for group in include_groups:
        if group is SessionArchiveLogicalGroup.DUCKDB_SNAPSHOT:
            continue
        paths = include_paths.get(group, DEFAULT_INCLUDE_PATHS.get(group, ()))
        if paths:
            roots[group.value] = paths[0]
    if duckdb_snapshot_source_path and duckdb_snapshot_source_path != ":memory:":
        roots[SessionArchiveLogicalGroup.DUCKDB_SNAPSHOT.value] = duckdb_snapshot_source_path
    return roots


def _duckdb_snapshot(
    request: SessionArchiveWriteRequest,
    repository_root: Path,
) -> SessionArchiveDuckDBSnapshot | None:
    if request.duckdb_snapshot_source_path is None:
        return None
    source_path = request.duckdb_snapshot_source_path
    if source_path != ":memory:":
        source = _resolve_repository_relative_path(
            source_path,
            repository_root=repository_root,
            field_name="duckdb_snapshot_source_path",
        )
        source_path = _relative_to_repository(
            source, repository_root, "duckdb_snapshot_source_path"
        )
    return SessionArchiveDuckDBSnapshot(
        included=source_path != ":memory:",
        source_path=source_path,
        description=request.duckdb_snapshot_description,
    )


def _duckdb_entry(
    request: SessionArchiveWriteRequest,
    repository_root: Path,
    include_groups: tuple[SessionArchiveLogicalGroup, ...],
) -> SessionArchiveShardEntry | None:
    if SessionArchiveLogicalGroup.DUCKDB_SNAPSHOT not in include_groups:
        return None
    if request.duckdb_snapshot_source_path in {None, ":memory:"}:
        return None
    source = _resolve_repository_relative_path(
        request.duckdb_snapshot_source_path,
        repository_root=repository_root,
        field_name="duckdb_snapshot_source_path",
    )
    if not source.exists() or not source.is_file():
        raise SessionArchiveError(
            "Session archive DuckDB snapshot source must be an existing file."
        )
    relative = _relative_to_repository(source, repository_root, "duckdb_snapshot_source_path")
    content = source.read_bytes()
    return SessionArchiveShardEntry(
        logical_group=SessionArchiveLogicalGroup.DUCKDB_SNAPSHOT,
        source_file_path=source,
        source_path=relative,
        archive_member_path=relative,
        size_bytes=len(content),
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(content).hexdigest(),
    )


def _include_groups(
    values: Sequence[SessionArchiveLogicalGroup | str],
) -> tuple[SessionArchiveLogicalGroup, ...]:
    groups = tuple(_logical_group(value) for value in values)
    if len({group.value for group in groups}) != len(groups):
        raise SessionArchiveError("Session archive include_groups must not contain duplicates.")
    return tuple(sorted(groups, key=lambda group: group.value))


def _include_paths(
    values: Mapping[SessionArchiveLogicalGroup | str, Sequence[str]],
) -> dict[SessionArchiveLogicalGroup, tuple[str, ...]]:
    return {
        _logical_group(group): tuple(
            _portable_path(path, f"include_paths.{group}") for path in paths
        )
        for group, paths in values.items()
    }


def _logical_group(value: SessionArchiveLogicalGroup | str) -> SessionArchiveLogicalGroup:
    if isinstance(value, SessionArchiveLogicalGroup):
        return value
    try:
        return SessionArchiveLogicalGroup(value)
    except ValueError as exc:
        raise SessionArchiveError(f"Unsupported session archive logical group: {value!r}.") from exc


def _resolve_repository_relative_path(
    value: str | Path,
    *,
    repository_root: Path,
    field_name: str,
) -> Path:
    relative = _portable_path(value, field_name)
    path = (repository_root / relative).resolve()
    _ensure_under_root(path, repository_root, field_name)
    return path


def _portable_path(value: str | Path, field_name: str) -> str:
    if isinstance(value, Path):
        text = value.as_posix()
    else:
        text = str(value).strip()
    if not text:
        raise SessionArchiveError(f"Session archive field '{field_name}' must be non-empty.")
    if "\\" in text:
        raise SessionArchiveError(
            f"Session archive field '{field_name}' must use repository-relative POSIX paths."
        )
    if text.startswith("~") or _URL_LIKE_PATTERN.match(text):
        raise SessionArchiveError(
            f"Session archive field '{field_name}' must use repository-relative POSIX paths."
        )
    if _WINDOWS_DRIVE_PREFIX_PATTERN.match(text):
        raise SessionArchiveError(
            f"Session archive field '{field_name}' must not use a Windows drive path."
        )
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
            f"Session archive field '{field_name}' must use normalized repository-relative paths."
        )
    return path.as_posix()


def _safe_archive_id(value: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."}:
        raise SessionArchiveError("Session archive archive_id must be a safe path segment.")
    if "/" in text or "\\" in text or _WINDOWS_DRIVE_PREFIX_PATTERN.match(text):
        raise SessionArchiveError("Session archive archive_id must be a safe path segment.")
    return text


def _optional_portable_path(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _portable_path(value, field_name)


def _relative_to_repository(path: Path, repository_root: Path, field_name: str) -> str:
    try:
        return path.resolve().relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise SessionArchiveError(
            f"Session archive field '{field_name}' must stay under repository_root."
        ) from exc


def _ensure_under_root(path: Path, root: Path, field_name: str) -> None:
    if not _is_relative_to(path.resolve(), root.resolve()):
        raise SessionArchiveError(
            f"Session archive field '{field_name}' must stay under the configured root."
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_excluded(relative_path: str, patterns: tuple[str, ...]) -> bool:
    parts = PurePosixPath(relative_path).parts
    for pattern in patterns:
        normalized = pattern.replace("\\", "/")
        if any(fnmatch.fnmatch(part, normalized) for part in parts):
            return True
        if fnmatch.fnmatch(relative_path, normalized):
            return True
        if relative_path.startswith(normalized.rstrip("/") + "/"):
            return True
    return False


def _path_sort_key(path: Path) -> str:
    return path.as_posix()


def _deterministic_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _atomic_write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_bytes(data)
    temp_path.replace(path)
    return path
