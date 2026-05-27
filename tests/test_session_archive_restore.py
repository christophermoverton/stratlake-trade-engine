from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import tarfile

import pytest

from src.session_archive.manifest import SessionArchiveError, SessionArchiveLogicalGroup
from src.session_archive.restore import (
    SessionArchiveRestoreRequest,
    build_session_archive_restore_plan,
    restore_session_archive_pack,
)
from src.session_archive.writer import (
    SessionArchiveIncludePolicy,
    SessionArchiveWriteRequest,
    write_session_archive_pack,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _repo(root: Path) -> Path:
    _write(root / "data/curated/features_daily/AAPL.parquet", "feature-a\n")
    _write(root / "data/curated/features_daily/MSFT.parquet", "feature-m\n")
    _write(root / "artifacts/strategies/run_a/manifest.json", '{"run_id":"run_a"}\n')
    _write(root / "artifacts/strategies/run_a/metrics.json", '{"sharpe":1.0}\n')
    _write(root / "configs/profiles/notebook.yml", "schema_version: 1\nprofile: notebook\n")
    _write(root / "configs/strategies.yml", "strategies: []\n")
    return root


def _archive(root: Path) -> Path:
    result = write_session_archive_pack(
        SessionArchiveWriteRequest(
            archive_id="session-a",
            repository_root=root,
            include_policy=SessionArchiveIncludePolicy(
                include_groups=(
                    SessionArchiveLogicalGroup.FEATURES,
                    SessionArchiveLogicalGroup.ARTIFACTS,
                    SessionArchiveLogicalGroup.CONFIGS,
                ),
                include_paths={
                    SessionArchiveLogicalGroup.FEATURES: ("data/curated/features_daily",),
                    SessionArchiveLogicalGroup.ARTIFACTS: ("artifacts/strategies",),
                    SessionArchiveLogicalGroup.CONFIGS: ("configs",),
                },
            ),
        )
    )
    return result.archive_root


def _restore_request(
    archive_root: Path, target_root: Path, **kwargs: object
) -> SessionArchiveRestoreRequest:
    return SessionArchiveRestoreRequest(
        archive_root=archive_root,
        target_root=target_root,
        **kwargs,
    )


def test_valid_archive_restore_recreates_repository_layout(tmp_path: Path) -> None:
    source = _repo(tmp_path / "source")
    archive_root = _archive(source)
    target = tmp_path / "target"

    result = restore_session_archive_pack(_restore_request(archive_root, target))

    assert (target / "data/curated/features_daily/AAPL.parquet").read_text(
        encoding="utf-8"
    ) == "feature-a\n"
    assert (target / "artifacts/strategies/run_a/metrics.json").read_text(
        encoding="utf-8"
    ) == '{"sharpe":1.0}\n'
    assert (target / "configs/strategies.yml").read_text(encoding="utf-8") == "strategies: []\n"
    assert result.report_path is not None and result.report_path.is_file()


def test_dry_run_restore_planning_writes_no_files(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    target = tmp_path / "target"

    plan = build_session_archive_restore_plan(_restore_request(archive_root, target))

    assert plan.restore_entries
    assert not (target / "configs/strategies.yml").exists()
    assert not (
        target / "artifacts/_derived/session_archives/session-a/restore_report.json"
    ).exists()


def test_missing_manifest_fails(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    (archive_root / "manifest.json").unlink()

    with pytest.raises(SessionArchiveError, match="manifest"):
        build_session_archive_restore_plan(_restore_request(archive_root, tmp_path / "target"))


def test_missing_shard_fails_before_extraction(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    first_shard = next((archive_root / "shards").glob("*.tar"))
    first_shard.unlink()
    target = tmp_path / "target"

    with pytest.raises(SessionArchiveError, match="missing"):
        restore_session_archive_pack(_restore_request(archive_root, target))

    assert not (target / "configs").exists()


def test_shard_checksum_mismatch_fails_before_extraction(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    first_shard = next((archive_root / "shards").glob("*.tar"))
    first_shard.write_bytes(first_shard.read_bytes() + b"tamper")
    target = tmp_path / "target"

    with pytest.raises(SessionArchiveError, match="checksum"):
        restore_session_archive_pack(_restore_request(archive_root, target))

    assert not (target / "configs").exists()


@pytest.mark.parametrize(
    "member_name",
    [
        "../outside.txt",
        "/absolute.txt",
        "C:relative/path.txt",
        "C:/absolute/path.txt",
        "file://bad.txt",
        "https://example.test/bad.txt",
        "~/bad.txt",
    ],
)
def test_unsafe_tar_member_path_is_rejected(tmp_path: Path, member_name: str) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    _replace_first_shard(archive_root, [(member_name, b"bad")])

    with pytest.raises(SessionArchiveError, match="member path|Windows drive"):
        build_session_archive_restore_plan(
            _restore_request(archive_root, tmp_path / "target", verify_checksums=False)
        )


def test_non_regular_tar_member_is_rejected(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    shard = next((archive_root / "shards").glob("*.tar"))
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        info = tarfile.TarInfo("configs/link.txt")
        info.type = tarfile.SYMTYPE
        info.linkname = "configs/strategies.yml"
        archive.addfile(info)
    shard.write_bytes(buffer.getvalue())

    with pytest.raises(SessionArchiveError, match="non-regular"):
        build_session_archive_restore_plan(
            _restore_request(archive_root, tmp_path / "target", verify_checksums=False)
        )


def test_fail_if_exists_policy_fails_before_extraction(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    target = tmp_path / "target"
    _write(target / "configs/strategies.yml", "existing\n")

    with pytest.raises(SessionArchiveError, match="exists"):
        restore_session_archive_pack(_restore_request(archive_root, target))

    assert not (target / "data/curated/features_daily/AAPL.parquet").exists()


def test_skip_existing_policy_skips_existing_and_restores_other_files(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    target = tmp_path / "target"
    _write(target / "configs/strategies.yml", "existing\n")

    result = restore_session_archive_pack(
        _restore_request(archive_root, target, overwrite_policy="skip_existing")
    )

    assert (target / "configs/strategies.yml").read_text(encoding="utf-8") == "existing\n"
    assert (target / "data/curated/features_daily/AAPL.parquet").exists()
    assert result.skipped_paths == (target / "configs/strategies.yml",)


def test_replace_existing_policy_replaces_existing_file(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    target = tmp_path / "target"
    _write(target / "configs/strategies.yml", "existing\n")

    restore_session_archive_pack(
        _restore_request(archive_root, target, overwrite_policy="replace_existing")
    )

    assert (target / "configs/strategies.yml").read_text(encoding="utf-8") == "strategies: []\n"


def test_restore_report_is_deterministic_and_portable(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    target = tmp_path / "target"

    first = build_session_archive_restore_plan(_restore_request(archive_root, target)).report
    second = build_session_archive_restore_plan(_restore_request(archive_root, target)).report
    text = json.dumps(first, indent=2, sort_keys=True)

    assert first == second
    assert str(tmp_path) not in text
    assert tmp_path.as_posix() not in text
    assert first["checksum_status"] == "passed"
    assert first["boundaries"]["transport_only"] is True


def test_round_trip_writer_to_restore_preserves_expected_contents(tmp_path: Path) -> None:
    source = _repo(tmp_path / "source")
    archive_root = _archive(source)
    target = tmp_path / "target"

    restore_session_archive_pack(_restore_request(archive_root, target))

    for relative in [
        "data/curated/features_daily/MSFT.parquet",
        "artifacts/strategies/run_a/manifest.json",
        "configs/profiles/notebook.yml",
    ]:
        assert (target / relative).read_text(encoding="utf-8") == (source / relative).read_text(
            encoding="utf-8"
        )


def test_duckdb_file_backed_snapshot_restore(tmp_path: Path) -> None:
    source = _repo(tmp_path / "source")
    _write(source / "artifacts/session.duckdb", "duckdb bytes\n")
    archive = write_session_archive_pack(
        SessionArchiveWriteRequest(
            archive_id="session-a",
            repository_root=source,
            include_policy=SessionArchiveIncludePolicy(
                include_groups=(
                    SessionArchiveLogicalGroup.CONFIGS,
                    SessionArchiveLogicalGroup.DUCKDB_SNAPSHOT,
                ),
                include_paths={SessionArchiveLogicalGroup.CONFIGS: ("configs",)},
            ),
            duckdb_snapshot_source_path="artifacts/session.duckdb",
        )
    )
    target = tmp_path / "target"

    restore_session_archive_pack(_restore_request(archive.archive_root, target))

    assert (target / "artifacts/session.duckdb").read_text(encoding="utf-8") == "duckdb bytes\n"


def test_duckdb_memory_metadata_does_not_require_or_restore_snapshot_file(tmp_path: Path) -> None:
    source = _repo(tmp_path / "source")
    archive = write_session_archive_pack(
        SessionArchiveWriteRequest(
            archive_id="session-a",
            repository_root=source,
            include_policy=SessionArchiveIncludePolicy(
                include_groups=(SessionArchiveLogicalGroup.CONFIGS,),
                include_paths={SessionArchiveLogicalGroup.CONFIGS: ("configs",)},
            ),
            duckdb_snapshot_source_path=":memory:",
        )
    )
    target = tmp_path / "target"

    restore_session_archive_pack(_restore_request(archive.archive_root, target))

    assert (target / "configs/strategies.yml").exists()
    assert not (target / "artifacts/session.duckdb").exists()


def _replace_first_shard(archive_root: Path, members: list[tuple[str, bytes]]) -> None:
    shard = next((archive_root / "shards").glob("*.tar"))
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            archive.addfile(info, BytesIO(data))
    shard.write_bytes(buffer.getvalue())
    _rewrite_shard_checksum(archive_root, shard)


def _rewrite_shard_checksum(archive_root: Path, shard: Path) -> None:
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    checksums_path = archive_root / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    for row in checksums["shards"]:
        if row["shard_name"] == shard.name:
            row["checksum"] = digest
    checksums_path.write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    manifest_path = archive_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["shards"]:
        if row["shard_name"] == shard.name:
            row["checksum"] = digest
            row["size_bytes"] = shard.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
