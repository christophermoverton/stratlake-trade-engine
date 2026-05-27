from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    write_session_archive_manifest,
)


SHA256_A = "a" * 64
SHA256_B = "b" * 64


def _shard(
    *,
    name: str = "features-000.tar.zst",
    path: str = "archives/session-a/shards/features-000.tar.zst",
    group: str = "features",
    index: int = 0,
) -> dict[str, object]:
    return {
        "shard_name": name,
        "shard_path": path,
        "logical_group": group,
        "shard_index": index,
        "file_count": 3,
        "size_bytes": 2048,
        "checksum_algorithm": "sha256",
        "checksum": SHA256_A if index == 0 else SHA256_B,
        "archive_format": "tar",
        "compression": "zstd",
    }


def _manifest_payload() -> dict[str, object]:
    return {
        "schema_version": SESSION_ARCHIVE_MANIFEST_SCHEMA_VERSION,
        "archive_id": "session-a",
        "session_id": "notebook-session-a",
        "created_at_utc": "2026-05-27T00:00:00Z",
        "source_runtime_profile": "notebook",
        "source_profile_path": "configs/profiles/notebook.yml",
        "source_roots": {
            "artifacts": "artifacts",
            "configs": "configs",
            "features": "data/curated",
        },
        "included_groups": ["features", "artifacts", "configs"],
        "shards": [
            _shard(
                name="configs-000.tar.zst",
                path="archives/session-a/shards/configs-000.tar.zst",
                group="configs",
                index=0,
            ),
            _shard(
                name="artifacts-000.tar.zst",
                path="archives/session-a/shards/artifacts-000.tar.zst",
                group="artifacts",
                index=0,
            ),
            _shard(),
        ],
        "restore": {
            "target_relative_roots": {
                "artifacts": "artifacts",
                "configs": "configs",
                "features": "data/curated",
            },
            "overwrite_policy": "fail_if_exists",
            "compatibility": {"minimum_manifest_schema_version": 1},
        },
        "boundaries": {
            "derived": True,
            "disposable": True,
            "transport_only": True,
            "authoritative": False,
            "canonical_storage": False,
            "requires_network": False,
            "requires_credentials": False,
            "requires_live_market_data": False,
        },
        "metadata": {"issue": "465"},
    }


def test_valid_manifest_construction_preserves_boundary_contract() -> None:
    manifest = SessionArchiveManifest.from_mapping(_manifest_payload())

    assert manifest.archive_id == "session-a"
    assert manifest.boundaries.derived is True
    assert manifest.boundaries.disposable is True
    assert manifest.boundaries.transport_only is True
    assert manifest.boundaries.authoritative is False
    assert manifest.boundaries.canonical_storage is False
    assert manifest.boundaries.requires_network is False
    assert manifest.boundaries.requires_credentials is False
    assert manifest.boundaries.requires_live_market_data is False


def test_dataclass_manifest_construction_is_valid() -> None:
    manifest = SessionArchiveManifest(
        schema_version=1,
        archive_id="manual-session",
        source_roots={"features": "data/curated"},
        included_groups=(SessionArchiveLogicalGroup.FEATURES,),
        shards=(
            SessionArchiveShard(
                shard_name="features-000.tar",
                shard_path="archives/manual-session/shards/features-000.tar",
                logical_group=SessionArchiveLogicalGroup.FEATURES,
                shard_index=0,
                file_count=1,
                size_bytes=12,
                checksum_algorithm="sha256",
                checksum=SHA256_A,
                archive_format="tar",
            ),
        ),
        restore=SessionArchiveRestoreExpectations(
            target_relative_roots={"features": "data/curated"}
        ),
        boundaries=SessionArchiveBoundaries(),
    )

    assert validate_session_archive_manifest(manifest) is manifest


def test_deterministic_json_serialization_and_byte_stability() -> None:
    first = manifest_to_deterministic_json(_manifest_payload())
    second = manifest_to_deterministic_json(_manifest_payload())

    assert first == second
    assert first.endswith("\n")
    assert json.loads(first)["schema_version"] == 1
    assert "C:\\Users" not in first


def test_stable_shard_ordering() -> None:
    payload_a = _manifest_payload()
    payload_b = _manifest_payload()
    payload_b["shards"] = list(reversed(payload_a["shards"]))  # type: ignore[index]

    assert manifest_to_deterministic_json(payload_a) == manifest_to_deterministic_json(payload_b)

    serialized = json.loads(manifest_to_deterministic_json(payload_a))
    assert [shard["logical_group"] for shard in serialized["shards"]] == [
        "artifacts",
        "configs",
        "features",
    ]


def test_repository_relative_posix_paths_are_allowed() -> None:
    payload = _manifest_payload()
    payload["source_roots"] = {"features": "data/curated/features_daily"}

    manifest = validate_session_archive_manifest(payload)

    assert manifest.source_roots["features"] == "data/curated/features_daily"


@pytest.mark.parametrize(
    "bad_path",
    [
        "/home/example/data",
        "C:/Users/example/data",
        "C:\\Users\\example\\data",
        "~/data",
        "file://data/curated",
        "../data/curated",
        "https://example.test/archive",
        "data/./curated",
        "data//curated",
    ],
)
def test_rejects_non_portable_paths(bad_path: str) -> None:
    payload = _manifest_payload()
    payload["source_roots"] = {"features": bad_path}

    with pytest.raises(SessionArchiveError, match="repository-relative|URI|URL|home|POSIX"):
        validate_session_archive_manifest(payload)


def test_rejects_malformed_shard_metadata() -> None:
    payload = _manifest_payload()
    shard = dict(_shard())
    shard["file_count"] = -1
    payload["shards"] = [shard]

    with pytest.raises(SessionArchiveError, match="file_count"):
        validate_session_archive_manifest(payload)

    shard = dict(_shard())
    shard.pop("checksum")
    payload["shards"] = [shard]

    with pytest.raises(SessionArchiveError, match="missing required"):
        validate_session_archive_manifest(payload)


def test_rejects_unsupported_schema_versions() -> None:
    payload = _manifest_payload()
    payload["schema_version"] = 999

    with pytest.raises(SessionArchiveError, match="Unsupported"):
        validate_session_archive_manifest(payload)


def test_rejects_secret_like_fields() -> None:
    payload = _manifest_payload()
    payload["metadata"] = {"api_key": "not-allowed"}

    with pytest.raises(SessionArchiveError, match="secret-like"):
        validate_session_archive_manifest(payload)


def test_optional_duckdb_snapshot_metadata_without_requiring_snapshot() -> None:
    without_snapshot = validate_session_archive_manifest(_manifest_payload())
    assert without_snapshot.duckdb_snapshot is None

    payload = _manifest_payload()
    payload["duckdb_snapshot"] = {
        "included": False,
        "source_path": ":memory:",
        "description": "In-memory DuckDB context only; no snapshot shard included.",
    }

    with_snapshot = validate_session_archive_manifest(payload)

    assert isinstance(with_snapshot.duckdb_snapshot, SessionArchiveDuckDBSnapshot)
    assert with_snapshot.duckdb_snapshot.source_path == ":memory:"


def test_rejects_canonical_boundary_claims() -> None:
    payload = _manifest_payload()
    boundaries = dict(payload["boundaries"])  # type: ignore[arg-type]
    boundaries["authoritative"] = True
    payload["boundaries"] = boundaries

    with pytest.raises(SessionArchiveError, match="authoritative"):
        validate_session_archive_manifest(payload)


def test_write_session_archive_manifest_uses_deterministic_json(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"

    write_session_archive_manifest(output, _manifest_payload())

    assert output.read_text(encoding="utf-8") == manifest_to_deterministic_json(_manifest_payload())
