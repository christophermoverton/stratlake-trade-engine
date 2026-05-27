from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import tarfile

import pytest

from src.session_archive.manifest import SessionArchiveError, SessionArchiveLogicalGroup
from src.session_archive.validation import (
    SessionArchiveIssueCode,
    inspect_session_archive,
    validate_session_archive,
    write_session_archive_inspection_report,
    write_session_archive_validation_report,
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


def _archive(root: Path, *, duckdb_source: str | None = None) -> Path:
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
            duckdb_snapshot_source_path=duckdb_source,
        )
    )
    return result.archive_root


def _issue_codes(result: object) -> set[str]:
    return {issue.code for issue in result.issues}  # type: ignore[attr-defined]


def test_valid_archive_validation_inspection_and_reports_are_deterministic(
    tmp_path: Path,
) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))

    validation = validate_session_archive(archive_root)
    inspection = inspect_session_archive(archive_root)
    first_validation_report = write_session_archive_validation_report(archive_root)
    validation_text = first_validation_report.read_text(encoding="utf-8")
    write_session_archive_validation_report(archive_root)
    second_validation_text = first_validation_report.read_text(encoding="utf-8")
    first_inspection_report = write_session_archive_inspection_report(archive_root)
    inspection_text = first_inspection_report.read_text(encoding="utf-8")
    write_session_archive_inspection_report(archive_root)

    assert validation.passed is True
    assert validation.archive_id == "session-a"
    assert validation.checksum_status == "passed"
    assert validation.status == "warning"
    assert validation_text == second_validation_text
    assert inspection.status == "warning"
    assert inspection.summary.archive_id == "session-a"
    assert inspection.summary.shard_count == 3
    assert inspection.summary.estimated_restored_file_count == 6
    assert inspection.summary.portability_status == "portable"
    assert first_inspection_report.read_text(encoding="utf-8") == inspection_text
    assert str(tmp_path) not in validation_text + inspection_text
    assert tmp_path.as_posix() not in validation_text + inspection_text


def test_reports_can_write_to_derived_output_root(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    output_root = tmp_path / "reports" / "artifacts"

    validation_path = write_session_archive_validation_report(archive_root, output_root=output_root)
    inspection_path = write_session_archive_inspection_report(archive_root, output_root=output_root)
    validation_text = validation_path.read_text(encoding="utf-8")
    inspection_text = inspection_path.read_text(encoding="utf-8")

    assert validation_path == (
        output_root / "_derived/session_archives/session-a/validation_report.json"
    )
    assert inspection_path == (
        output_root / "_derived/session_archives/session-a/inspection_report.json"
    )
    write_session_archive_validation_report(archive_root, output_root=output_root)
    write_session_archive_inspection_report(archive_root, output_root=output_root)
    assert validation_path.read_text(encoding="utf-8") == validation_text
    assert inspection_path.read_text(encoding="utf-8") == inspection_text
    assert str(tmp_path) not in validation_text + inspection_text
    assert tmp_path.as_posix() not in validation_text + inspection_text


def test_reports_preserve_explicit_output_path(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    validation_path = tmp_path / "explicit" / "validation.json"
    inspection_path = tmp_path / "explicit" / "inspection.json"

    assert (
        write_session_archive_validation_report(archive_root, output_path=validation_path)
        == validation_path
    )
    assert (
        write_session_archive_inspection_report(archive_root, output_path=inspection_path)
        == inspection_path
    )
    assert validation_path.is_file()
    assert inspection_path.is_file()


def test_report_output_path_and_output_root_conflict_fails(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))

    with pytest.raises(SessionArchiveError, match="output_path and output_root"):
        write_session_archive_validation_report(
            archive_root,
            output_path=tmp_path / "validation.json",
            output_root=tmp_path / "reports",
        )

    with pytest.raises(SessionArchiveError, match="output_path and output_root"):
        write_session_archive_inspection_report(
            archive_root,
            output_path=tmp_path / "inspection.json",
            output_root=tmp_path / "reports",
        )


def test_derived_report_output_root_uses_unknown_archive_for_invalid_pack(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "bad-archive"
    archive_root.mkdir()

    path = write_session_archive_validation_report(
        archive_root,
        output_root=tmp_path / "reports" / "artifacts",
    )

    assert path == (
        tmp_path
        / "reports"
        / "artifacts"
        / "_derived/session_archives/unknown_archive/validation_report.json"
    )
    assert SessionArchiveIssueCode.MISSING_MANIFEST in {
        item["code"] for item in json.loads(path.read_text(encoding="utf-8"))["issues"]
    }


def test_missing_manifest_fails_clearly(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    (archive_root / "manifest.json").unlink()

    result = validate_session_archive(archive_root)

    assert result.status == "failed"
    assert SessionArchiveIssueCode.MISSING_MANIFEST in _issue_codes(result)


def test_malformed_manifest_json_fails_clearly(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    (archive_root / "manifest.json").write_text("{bad", encoding="utf-8")

    result = validate_session_archive(archive_root)

    assert result.status == "failed"
    assert SessionArchiveIssueCode.MALFORMED_MANIFEST_JSON in _issue_codes(result)


def test_unsupported_schema_version_fails_clearly(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    manifest_path = archive_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 999
    _write_json(manifest_path, manifest)

    result = validate_session_archive(archive_root)

    assert result.status == "failed"
    assert SessionArchiveIssueCode.UNSUPPORTED_SCHEMA_VERSION in _issue_codes(result)


def test_missing_required_shard_fails_clearly(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    next((archive_root / "shards").glob("*.tar")).unlink()

    result = validate_session_archive(archive_root)

    assert result.status == "failed"
    assert SessionArchiveIssueCode.MISSING_REQUIRED_SHARD in _issue_codes(result)


def test_missing_shard_index_fails_clearly(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    (archive_root / "archive_index.json").unlink()

    result = validate_session_archive(archive_root)

    assert result.status == "failed"
    assert SessionArchiveIssueCode.MISSING_SHARD_INDEX in _issue_codes(result)


def test_missing_checksums_sidecar_fails_clearly(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    (archive_root / "checksums.json").unlink()

    result = validate_session_archive(archive_root)

    assert result.status == "failed"
    assert SessionArchiveIssueCode.MISSING_CHECKSUMS in _issue_codes(result)


def test_missing_restore_plan_sidecar_fails_clearly(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    (archive_root / "restore_plan.json").unlink()

    result = validate_session_archive(archive_root)

    assert result.status == "failed"
    assert SessionArchiveIssueCode.MISSING_RESTORE_PLAN in _issue_codes(result)


def test_malformed_shard_index_fails_clearly(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    _write(archive_root / "archive_index.json", "{bad")

    result = validate_session_archive(archive_root)

    assert result.status == "failed"
    assert SessionArchiveIssueCode.MALFORMED_SHARD_INDEX in _issue_codes(result)


def test_checksum_mismatch_fails_clearly(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    shard = next((archive_root / "shards").glob("*.tar"))
    shard.write_bytes(shard.read_bytes() + b"tamper")

    result = validate_session_archive(archive_root)

    assert result.status == "failed"
    assert SessionArchiveIssueCode.CHECKSUM_MISMATCH in _issue_codes(result)


def test_checksum_validation_streams_without_path_read_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))

    def fail_read_bytes(path: Path) -> bytes:
        raise AssertionError(f"read_bytes should not be used for checksum validation: {path}")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    result = validate_session_archive(archive_root)

    assert result.passed is True
    assert result.checksum_status == "passed"


def test_malformed_shard_metadata_fails_clearly(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    manifest_path = archive_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][0]["checksum"] = "bad"
    _write_json(manifest_path, manifest)

    result = validate_session_archive(archive_root)

    assert result.status == "failed"
    assert SessionArchiveIssueCode.MALFORMED_SHARD_METADATA in _issue_codes(result)


def test_archive_index_inconsistency_fails_clearly(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    index_path = archive_root / "archive_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["logical_groups"]["configs"]["file_count"] = 999
    _write_json(index_path, index)

    result = validate_session_archive(archive_root)

    assert result.status == "failed"
    assert SessionArchiveIssueCode.ARCHIVE_INDEX_INCONSISTENCY in _issue_codes(result)


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
def test_unsafe_archive_entries_fail_without_extraction(
    tmp_path: Path,
    member_name: str,
) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    _replace_first_shard(archive_root, [(member_name, b"bad\n")])

    result = validate_session_archive(archive_root, verify_checksums=False)

    assert result.status == "failed"
    assert SessionArchiveIssueCode.UNSAFE_ARCHIVE_ENTRY in _issue_codes(result)
    assert not (tmp_path / "outside.txt").exists()


def test_non_regular_archive_entry_fails_without_extraction(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    shard = next((archive_root / "shards").glob("*.tar"))
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        info = tarfile.TarInfo("configs/link.txt")
        info.type = tarfile.SYMTYPE
        info.linkname = "configs/strategies.yml"
        archive.addfile(info)
    shard.write_bytes(buffer.getvalue())
    _rewrite_shard_checksum(archive_root, shard)

    result = validate_session_archive(archive_root, verify_checksums=False)

    assert result.status == "failed"
    assert SessionArchiveIssueCode.UNSAFE_ARCHIVE_ENTRY in _issue_codes(result)


def test_restore_plan_unsafe_target_root_fails(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    restore_plan_path = archive_root / "restore_plan.json"
    restore_plan = json.loads(restore_plan_path.read_text(encoding="utf-8"))
    restore_plan["target_relative_roots"]["configs"] = "../configs"
    _write_json(restore_plan_path, restore_plan)

    result = validate_session_archive(archive_root)

    assert result.status == "failed"
    assert SessionArchiveIssueCode.UNSAFE_RESTORE_PATH in _issue_codes(result)


def test_optional_duckdb_memory_metadata_warns_without_requiring_snapshot(
    tmp_path: Path,
) -> None:
    archive_root = _archive(_repo(tmp_path / "source"), duckdb_source=":memory:")

    result = validate_session_archive(archive_root)
    inspection = inspect_session_archive(archive_root)

    assert result.passed is True
    assert SessionArchiveIssueCode.OPTIONAL_DUCKDB_METADATA_WARNING in _issue_codes(result)
    assert inspection.summary.duckdb_snapshot_status == "memory_metadata_only"


def test_missing_optional_duckdb_snapshot_warns_not_fails(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))

    result = validate_session_archive(archive_root)

    assert result.passed is True
    assert SessionArchiveIssueCode.OPTIONAL_DUCKDB_SNAPSHOT_MISSING in _issue_codes(result)


def test_inspection_does_not_mutate_target_paths_or_extract_contents(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    target = tmp_path / "target"

    result = inspect_session_archive(archive_root)

    assert result.summary.estimated_restored_file_count == 6
    assert not target.exists()
    assert not (archive_root / "restore_report.json").exists()


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
    _write_json(checksums_path, checksums)
    manifest_path = archive_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["shards"]:
        if row["shard_name"] == shard.name:
            row["checksum"] = digest
            row["size_bytes"] = shard.stat().st_size
    _write_json(manifest_path, manifest)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
