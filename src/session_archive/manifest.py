from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Mapping

from src.artifacts.safety import atomic_write_text

SESSION_ARCHIVE_MANIFEST_SCHEMA_VERSION = 1
SUPPORTED_ARCHIVE_FORMATS = frozenset({"tar", "zip", "directory"})
SUPPORTED_COMPRESSIONS = frozenset({"none", "gzip", "zstd"})
SUPPORTED_CHECKSUM_ALGORITHMS = frozenset({"sha256"})
SUPPORTED_OVERWRITE_POLICIES = frozenset({"fail_if_exists", "skip_existing", "overwrite_allowed"})
_SECRET_KEY_PATTERN = re.compile(
    r"(secret|token|credential|password|passwd|api[_-]?key|access[_-]?key|private[_-]?key)",
    re.IGNORECASE,
)
_URL_LIKE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_WINDOWS_DRIVE_PREFIX_PATTERN = re.compile(r"^[A-Za-z]:")


class SessionArchiveError(ValueError):
    """Raised when a session archive manifest violates the portable contract."""


class SessionArchiveLogicalGroup(str, Enum):
    FEATURES = "features"
    ARTIFACTS = "artifacts"
    CONFIGS = "configs"
    DUCKDB_SNAPSHOT = "duckdb_snapshot"


@dataclass(frozen=True)
class SessionArchiveBoundaries:
    derived: bool = True
    disposable: bool = True
    transport_only: bool = True
    authoritative: bool = False
    canonical_storage: bool = False
    requires_network: bool = False
    requires_credentials: bool = False
    requires_live_market_data: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "SessionArchiveBoundaries":
        if payload is None:
            return cls()
        _require_mapping(payload, "boundaries")
        allowed = set(cls().__dict__)
        _reject_unknown_keys(payload, allowed, "boundaries")
        values = {
            key: _required_bool(payload.get(key), f"boundaries.{key}")
            for key in allowed
            if key in payload
        }
        boundaries = cls(**values)
        boundaries.validate()
        return boundaries

    def validate(self) -> None:
        expected = {
            "derived": True,
            "disposable": True,
            "transport_only": True,
            "authoritative": False,
            "canonical_storage": False,
            "requires_network": False,
            "requires_credentials": False,
            "requires_live_market_data": False,
        }
        for key, expected_value in expected.items():
            value = getattr(self, key)
            if not isinstance(value, bool):
                raise SessionArchiveError(f"Manifest field 'boundaries.{key}' must be a boolean.")
            if value is not expected_value:
                raise SessionArchiveError(
                    f"Manifest field 'boundaries.{key}' must be {expected_value!r}."
                )

    def to_dict(self) -> dict[str, bool]:
        self.validate()
        return {
            "authoritative": self.authoritative,
            "canonical_storage": self.canonical_storage,
            "derived": self.derived,
            "disposable": self.disposable,
            "requires_credentials": self.requires_credentials,
            "requires_live_market_data": self.requires_live_market_data,
            "requires_network": self.requires_network,
            "transport_only": self.transport_only,
        }


@dataclass(frozen=True)
class SessionArchiveShard:
    shard_name: str
    shard_path: str
    logical_group: SessionArchiveLogicalGroup | str
    shard_index: int
    file_count: int
    size_bytes: int
    checksum_algorithm: str
    checksum: str
    archive_format: str
    compression: str = "none"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SessionArchiveShard":
        _require_mapping(payload, "shard")
        required = {
            "archive_format",
            "checksum",
            "checksum_algorithm",
            "file_count",
            "logical_group",
            "shard_index",
            "shard_name",
            "shard_path",
            "size_bytes",
        }
        _reject_unknown_keys(payload, required | {"compression"}, "shard")
        missing = sorted(required - set(payload))
        if missing:
            raise SessionArchiveError(f"Manifest shard is missing required field(s): {missing}.")
        return cls(
            shard_name=_safe_shard_name(payload["shard_name"], "shard.shard_name"),
            shard_path=_portable_repository_path(payload["shard_path"], "shard.shard_path"),
            logical_group=_logical_group(payload["logical_group"], "shard.logical_group"),
            shard_index=_non_negative_int(payload["shard_index"], "shard.shard_index"),
            file_count=_non_negative_int(payload["file_count"], "shard.file_count"),
            size_bytes=_non_negative_int(payload["size_bytes"], "shard.size_bytes"),
            checksum_algorithm=_required_string(
                payload["checksum_algorithm"], "shard.checksum_algorithm"
            ),
            checksum=_required_string(payload["checksum"], "shard.checksum"),
            archive_format=_required_string(payload["archive_format"], "shard.archive_format"),
            compression=_required_string(payload.get("compression", "none"), "shard.compression"),
        )

    def validate(self) -> None:
        _safe_shard_name(self.shard_name, "shard.shard_name")
        _portable_repository_path(self.shard_path, "shard.shard_path")
        _logical_group(self.logical_group, "shard.logical_group")
        _non_negative_int(self.shard_index, "shard.shard_index")
        _non_negative_int(self.file_count, "shard.file_count")
        _non_negative_int(self.size_bytes, "shard.size_bytes")
        if self.checksum_algorithm not in SUPPORTED_CHECKSUM_ALGORITHMS:
            raise SessionArchiveError(
                "Manifest field 'shard.checksum_algorithm' must be one of "
                f"{sorted(SUPPORTED_CHECKSUM_ALGORITHMS)}."
            )
        checksum = _required_string(self.checksum, "shard.checksum")
        if self.checksum_algorithm == "sha256" and not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise SessionArchiveError(
                "Manifest field 'shard.checksum' must be a SHA-256 hex digest."
            )
        if self.archive_format not in SUPPORTED_ARCHIVE_FORMATS:
            raise SessionArchiveError(
                f"Manifest field 'shard.archive_format' must be one of {sorted(SUPPORTED_ARCHIVE_FORMATS)}."
            )
        if self.compression not in SUPPORTED_COMPRESSIONS:
            raise SessionArchiveError(
                f"Manifest field 'shard.compression' must be one of {sorted(SUPPORTED_COMPRESSIONS)}."
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "archive_format": self.archive_format,
            "checksum": self.checksum,
            "checksum_algorithm": self.checksum_algorithm,
            "compression": self.compression,
            "file_count": self.file_count,
            "logical_group": _logical_group(self.logical_group, "shard.logical_group").value,
            "shard_index": self.shard_index,
            "shard_name": self.shard_name,
            "shard_path": self.shard_path,
            "size_bytes": self.size_bytes,
        }

    def sort_key(self) -> tuple[str, int, str, str]:
        group = _logical_group(self.logical_group, "shard.logical_group").value
        return (group, self.shard_index, self.shard_name, self.shard_path)


@dataclass(frozen=True)
class SessionArchiveDuckDBSnapshot:
    included: bool = False
    source_path: str | None = None
    snapshot_path: str | None = None
    description: str | None = None

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any] | None
    ) -> "SessionArchiveDuckDBSnapshot | None":
        if payload is None:
            return None
        _require_mapping(payload, "duckdb_snapshot")
        _reject_unknown_keys(
            payload, {"included", "source_path", "snapshot_path", "description"}, "duckdb_snapshot"
        )
        return cls(
            included=_required_bool(payload.get("included", False), "duckdb_snapshot.included"),
            source_path=_optional_duckdb_path(
                payload.get("source_path"), "duckdb_snapshot.source_path"
            ),
            snapshot_path=_optional_portable_path(
                payload.get("snapshot_path"), "duckdb_snapshot.snapshot_path"
            ),
            description=_optional_string(payload.get("description"), "duckdb_snapshot.description"),
        )

    def validate(self) -> None:
        _required_bool(self.included, "duckdb_snapshot.included")
        _optional_duckdb_path(self.source_path, "duckdb_snapshot.source_path")
        _optional_portable_path(self.snapshot_path, "duckdb_snapshot.snapshot_path")
        _optional_string(self.description, "duckdb_snapshot.description")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data: dict[str, Any] = {"included": self.included}
        if self.description is not None:
            data["description"] = self.description
        if self.snapshot_path is not None:
            data["snapshot_path"] = self.snapshot_path
        if self.source_path is not None:
            data["source_path"] = self.source_path
        return data


@dataclass(frozen=True)
class SessionArchiveRestoreExpectations:
    target_relative_roots: Mapping[str, str]
    overwrite_policy: str = "fail_if_exists"
    compatibility: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SessionArchiveRestoreExpectations":
        _require_mapping(payload, "restore")
        _reject_unknown_keys(
            payload, {"target_relative_roots", "overwrite_policy", "compatibility"}, "restore"
        )
        if "target_relative_roots" not in payload:
            raise SessionArchiveError("Manifest field 'restore.target_relative_roots' is required.")
        return cls(
            target_relative_roots=_path_mapping(
                payload["target_relative_roots"], "restore.target_relative_roots"
            ),
            overwrite_policy=_required_string(
                payload.get("overwrite_policy", "fail_if_exists"), "restore.overwrite_policy"
            ),
            compatibility=_safe_metadata_mapping(
                payload.get("compatibility", {}), "restore.compatibility"
            ),
        )

    def validate(self) -> None:
        _path_mapping(self.target_relative_roots, "restore.target_relative_roots")
        if self.overwrite_policy not in SUPPORTED_OVERWRITE_POLICIES:
            raise SessionArchiveError(
                "Manifest field 'restore.overwrite_policy' must be one of "
                f"{sorted(SUPPORTED_OVERWRITE_POLICIES)}."
            )
        _safe_metadata_mapping(self.compatibility, "restore.compatibility")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "compatibility": _stable_jsonable(self.compatibility),
            "overwrite_policy": self.overwrite_policy,
            "target_relative_roots": {
                key: self.target_relative_roots[key] for key in sorted(self.target_relative_roots)
            },
        }


@dataclass(frozen=True)
class SessionArchiveManifest:
    schema_version: int
    archive_id: str
    source_roots: Mapping[str, str]
    included_groups: tuple[SessionArchiveLogicalGroup | str, ...]
    shards: tuple[SessionArchiveShard, ...]
    restore: SessionArchiveRestoreExpectations
    boundaries: SessionArchiveBoundaries = field(default_factory=SessionArchiveBoundaries)
    session_id: str | None = None
    created_at_utc: str | None = None
    source_runtime_profile: str | None = None
    source_profile_path: str | None = None
    duckdb_snapshot: SessionArchiveDuckDBSnapshot | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SessionArchiveManifest":
        _require_mapping(payload, "manifest")
        allowed = {
            "archive_id",
            "boundaries",
            "created_at_utc",
            "duckdb_snapshot",
            "included_groups",
            "metadata",
            "restore",
            "schema_version",
            "session_id",
            "shards",
            "source_profile_path",
            "source_roots",
            "source_runtime_profile",
        }
        _reject_unknown_keys(payload, allowed, "manifest")
        required = {
            "archive_id",
            "boundaries",
            "included_groups",
            "restore",
            "schema_version",
            "shards",
            "source_roots",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise SessionArchiveError(f"Manifest is missing required field(s): {missing}.")
        shards_payload = payload["shards"]
        if not isinstance(shards_payload, list | tuple):
            raise SessionArchiveError("Manifest field 'shards' must be a list.")
        included_payload = payload["included_groups"]
        if not isinstance(included_payload, list | tuple):
            raise SessionArchiveError("Manifest field 'included_groups' must be a list.")
        return cls(
            schema_version=_schema_version(payload["schema_version"]),
            archive_id=_required_string(payload["archive_id"], "archive_id"),
            source_roots=_path_mapping(payload["source_roots"], "source_roots"),
            included_groups=tuple(
                _logical_group(value, "included_groups") for value in included_payload
            ),
            shards=tuple(SessionArchiveShard.from_mapping(shard) for shard in shards_payload),
            restore=SessionArchiveRestoreExpectations.from_mapping(payload["restore"]),
            boundaries=SessionArchiveBoundaries.from_mapping(payload["boundaries"]),
            session_id=_optional_string(payload.get("session_id"), "session_id"),
            created_at_utc=_optional_string(payload.get("created_at_utc"), "created_at_utc"),
            source_runtime_profile=_optional_string(
                payload.get("source_runtime_profile"), "source_runtime_profile"
            ),
            source_profile_path=_optional_portable_path(
                payload.get("source_profile_path"), "source_profile_path"
            ),
            duckdb_snapshot=SessionArchiveDuckDBSnapshot.from_mapping(
                payload.get("duckdb_snapshot")
            ),
            metadata=_safe_metadata_mapping(payload.get("metadata", {}), "metadata"),
        )

    def validate(self) -> None:
        _schema_version(self.schema_version)
        _required_string(self.archive_id, "archive_id")
        _optional_string(self.session_id, "session_id")
        _optional_string(self.created_at_utc, "created_at_utc")
        _optional_string(self.source_runtime_profile, "source_runtime_profile")
        _optional_portable_path(self.source_profile_path, "source_profile_path")
        _path_mapping(self.source_roots, "source_roots")
        if not self.included_groups:
            raise SessionArchiveError("Manifest field 'included_groups' must not be empty.")
        included = tuple(_logical_group(value, "included_groups") for value in self.included_groups)
        included_values = [group.value for group in included]
        if len(included_values) != len(set(included_values)):
            raise SessionArchiveError(
                "Manifest field 'included_groups' must not contain duplicates."
            )
        if not self.shards:
            raise SessionArchiveError("Manifest field 'shards' must not be empty.")
        shard_groups = set()
        seen_indices: set[tuple[str, int]] = set()
        seen_paths: set[str] = set()
        for shard in self.shards:
            shard.validate()
            group = _logical_group(shard.logical_group, "shard.logical_group")
            shard_groups.add(group)
            group_index = (group.value, shard.shard_index)
            if group_index in seen_indices:
                raise SessionArchiveError(
                    "Manifest shards must not repeat logical_group/shard_index pairs."
                )
            seen_indices.add(group_index)
            if shard.shard_path in seen_paths:
                raise SessionArchiveError("Manifest shards must not repeat shard_path values.")
            seen_paths.add(shard.shard_path)
        missing_shard_groups = sorted(
            group.value for group in included if group not in shard_groups
        )
        if missing_shard_groups:
            raise SessionArchiveError(
                "Manifest included_groups must have at least one matching shard: "
                f"{missing_shard_groups}."
            )
        self.restore.validate()
        self.boundaries.validate()
        if self.duckdb_snapshot is not None:
            self.duckdb_snapshot.validate()
        _safe_metadata_mapping(self.metadata, "metadata")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data: dict[str, Any] = {
            "archive_id": self.archive_id,
            "boundaries": self.boundaries.to_dict(),
            "included_groups": sorted(
                {_logical_group(value, "included_groups").value for value in self.included_groups}
            ),
            "metadata": _stable_jsonable(self.metadata),
            "restore": self.restore.to_dict(),
            "schema_version": self.schema_version,
            "shards": [
                shard.to_dict() for shard in sorted(self.shards, key=lambda shard: shard.sort_key())
            ],
            "source_roots": {key: self.source_roots[key] for key in sorted(self.source_roots)},
        }
        if self.created_at_utc is not None:
            data["created_at_utc"] = self.created_at_utc
        if self.duckdb_snapshot is not None:
            data["duckdb_snapshot"] = self.duckdb_snapshot.to_dict()
        if self.session_id is not None:
            data["session_id"] = self.session_id
        if self.source_profile_path is not None:
            data["source_profile_path"] = self.source_profile_path
        if self.source_runtime_profile is not None:
            data["source_runtime_profile"] = self.source_runtime_profile
        return data


def validate_session_archive_manifest(
    manifest: SessionArchiveManifest | Mapping[str, Any],
) -> SessionArchiveManifest:
    resolved = (
        manifest
        if isinstance(manifest, SessionArchiveManifest)
        else SessionArchiveManifest.from_mapping(manifest)
    )
    resolved.validate()
    return resolved


def manifest_to_deterministic_json(manifest: SessionArchiveManifest | Mapping[str, Any]) -> str:
    resolved = validate_session_archive_manifest(manifest)
    return json.dumps(resolved.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_session_archive_manifest(
    path: str | Path,
    manifest: SessionArchiveManifest | Mapping[str, Any],
) -> Path:
    return atomic_write_text(path, manifest_to_deterministic_json(manifest))


def _schema_version(value: Any) -> int:
    if value != SESSION_ARCHIVE_MANIFEST_SCHEMA_VERSION:
        raise SessionArchiveError(
            "Unsupported session archive manifest schema_version "
            f"{value!r}; expected {SESSION_ARCHIVE_MANIFEST_SCHEMA_VERSION}."
        )
    return SESSION_ARCHIVE_MANIFEST_SCHEMA_VERSION


def _logical_group(value: Any, field_name: str) -> SessionArchiveLogicalGroup:
    if isinstance(value, SessionArchiveLogicalGroup):
        return value
    try:
        return SessionArchiveLogicalGroup(_required_string(value, field_name))
    except ValueError as exc:
        raise SessionArchiveError(
            f"Manifest field '{field_name}' must be one of "
            f"{[group.value for group in SessionArchiveLogicalGroup]}."
        ) from exc


def _path_mapping(value: Any, field_name: str) -> dict[str, str]:
    _require_mapping(value, field_name)
    if not value:
        raise SessionArchiveError(f"Manifest field '{field_name}' must not be empty.")
    resolved: dict[str, str] = {}
    for key, path in value.items():
        key_text = _safe_metadata_key(key, f"{field_name} key")
        resolved[key_text] = _portable_repository_path(path, f"{field_name}.{key_text}")
    return resolved


def _portable_repository_path(value: Any, field_name: str) -> str:
    text = _required_string(value, field_name)
    if text == ":memory:":
        raise SessionArchiveError(
            f"Manifest field '{field_name}' must be a repository-relative path."
        )
    return _portable_path_text(text, field_name)


def _optional_portable_path(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _portable_repository_path(value, field_name)


def _optional_duckdb_path(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    text = _required_string(value, field_name)
    if text == ":memory:":
        return text
    return _portable_path_text(text, field_name)


def _portable_path_text(text: str, field_name: str) -> str:
    if "\\" in text:
        raise SessionArchiveError(
            f"Manifest field '{field_name}' must use POSIX-style '/' separators."
        )
    if _URL_LIKE_PATTERN.match(text):
        raise SessionArchiveError(f"Manifest field '{field_name}' must not be a URI or URL.")
    if text.startswith("~"):
        raise SessionArchiveError(f"Manifest field '{field_name}' must not use a home shortcut.")
    if _WINDOWS_DRIVE_PREFIX_PATTERN.match(text):
        raise SessionArchiveError(
            f"Manifest field '{field_name}' must be a normalized repository-relative path "
            "and must not use a Windows drive path."
        )
    posix_path = PurePosixPath(text)
    windows_path = PureWindowsPath(text)
    first_part = posix_path.parts[0] if posix_path.parts else ""
    invalid = (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or (len(first_part) == 2 and first_part[1] == ":")
        or any(part in {"", ".", ".."} for part in posix_path.parts)
        or posix_path.as_posix() != text
    )
    if invalid:
        raise SessionArchiveError(
            f"Manifest field '{field_name}' must be a normalized repository-relative path."
        )
    return posix_path.as_posix()


def _safe_shard_name(value: Any, field_name: str) -> str:
    text = _required_string(value, field_name)
    if text in {".", ".."}:
        raise SessionArchiveError(f"Manifest field '{field_name}' must be a safe shard filename.")
    if "/" in text or "\\" in text:
        raise SessionArchiveError(
            f"Manifest field '{field_name}' must be a safe shard filename without path separators."
        )
    if _WINDOWS_DRIVE_PREFIX_PATTERN.match(text):
        raise SessionArchiveError(
            f"Manifest field '{field_name}' must be a safe shard filename without a Windows drive prefix."
        )
    return text


def _safe_metadata_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    _require_mapping(value, field_name)
    resolved: dict[str, Any] = {}
    for key, item in value.items():
        key_text = _safe_metadata_key(key, f"{field_name} key")
        resolved[key_text] = _safe_metadata_value(item, f"{field_name}.{key_text}")
    return resolved


def _safe_metadata_value(value: Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        return _safe_metadata_mapping(value, field_name)
    if isinstance(value, list | tuple):
        return [_safe_metadata_value(item, field_name) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise SessionArchiveError(f"Manifest field '{field_name}' contains a non-finite float.")
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise SessionArchiveError(f"Manifest field '{field_name}' contains unsupported metadata value.")


def _safe_metadata_key(value: Any, field_name: str) -> str:
    key = _required_string(value, field_name)
    if _SECRET_KEY_PATTERN.search(key):
        raise SessionArchiveError(f"Manifest field '{field_name}' contains a secret-like key name.")
    return key


def _stable_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stable_jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_stable_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_stable_jsonable(item) for item in value]
    return value


def _require_mapping(value: Any, field_name: str) -> None:
    if not isinstance(value, Mapping):
        raise SessionArchiveError(f"Manifest field '{field_name}' must be a mapping.")


def _reject_unknown_keys(payload: Mapping[str, Any], allowed: set[str], field_name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        for key in unknown:
            _safe_metadata_key(key, f"{field_name} key")
        raise SessionArchiveError(
            f"Manifest field '{field_name}' contains unsupported keys: {unknown}."
        )


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise SessionArchiveError(f"Manifest field '{field_name}' must be a non-empty string.")
    text = value.strip()
    if not text:
        raise SessionArchiveError(f"Manifest field '{field_name}' must be a non-empty string.")
    return text


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field_name)


def _required_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise SessionArchiveError(f"Manifest field '{field_name}' must be a boolean.")
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SessionArchiveError(f"Manifest field '{field_name}' must be a non-negative integer.")
    return value
