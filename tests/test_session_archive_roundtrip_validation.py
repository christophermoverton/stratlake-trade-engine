from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import tarfile

import pytest

from src.cli.session_archive import main as session_archive_cli
from src.session_archive import (
    SessionArchiveError,
    SessionArchiveIncludePolicy,
    SessionArchiveIssueCode,
    SessionArchiveLogicalGroup,
    SessionArchiveRestoreRequest,
    SessionArchiveWriteRequest,
    build_session_archive_restore_plan,
    inspect_session_archive,
    restore_session_archive_pack,
    validate_session_archive,
    validate_session_archive_manifest,
    write_session_archive_inspection_report,
    write_session_archive_pack,
    write_session_archive_validation_report,
)


EXPECTED_RELATIVE_FILES = (
    "data/curated/features_daily/symbol=AAPL/year=2025/part-000.parquet",
    "data/curated/features_daily/symbol=MSFT/year=2025/part-000.parquet",
    "artifacts/strategies/run_a/manifest.json",
    "artifacts/strategies/run_a/metrics.json",
    "artifacts/strategies/run_a/equity_curve.csv",
    "artifacts/alpha/run_alpha_a/manifest.json",
    "artifacts/alpha/run_alpha_a/alpha_metrics.json",
    "configs/profiles/notebook.yml",
    "configs/strategies/demo_strategy.yml",
    "configs/tickers_demo.txt",
    "data/duckdb/session.duckdb",
)


def test_session_archive_api_roundtrip_is_deterministic_and_portable(
    tmp_path: Path,
) -> None:
    source = _synthetic_repo(tmp_path / "source")
    before = _relative_digest_map(source, EXPECTED_RELATIVE_FILES)

    result = write_session_archive_pack(_write_request(source))
    archive_root = result.archive_root

    assert archive_root == source / "artifacts/_derived/session_archives/roundtrip-session"
    assert (archive_root / "manifest.json").is_file()
    assert (archive_root / "archive_index.json").is_file()
    assert (archive_root / "checksums.json").is_file()
    assert (archive_root / "restore_plan.json").is_file()
    assert (archive_root / "shards").is_dir()
    validate_session_archive_manifest(_read_json(archive_root / "manifest.json"))

    validation = validate_session_archive(archive_root)
    inspection = inspect_session_archive(archive_root)
    assert validation.passed is True
    assert not _issues(validation, "error")
    assert inspection.summary.archive_id == "roundtrip-session"
    assert inspection.summary.shard_count > 0
    assert inspection.summary.estimated_restored_file_count == len(EXPECTED_RELATIVE_FILES)
    assert inspection.summary.portability_status == "portable"

    report_root = tmp_path / "reports" / "artifacts"
    validation_report = write_session_archive_validation_report(
        archive_root, output_root=report_root
    )
    inspection_report = write_session_archive_inspection_report(
        archive_root, output_root=report_root
    )
    validation_text = validation_report.read_text(encoding="utf-8")
    inspection_text = inspection_report.read_text(encoding="utf-8")
    write_session_archive_validation_report(archive_root, output_root=report_root)
    write_session_archive_inspection_report(archive_root, output_root=report_root)
    assert validation_report.read_text(encoding="utf-8") == validation_text
    assert inspection_report.read_text(encoding="utf-8") == inspection_text

    _assert_no_machine_paths(validation_text + inspection_text, tmp_path)
    validation_payload = json.loads(validation_text)
    inspection_payload = json.loads(inspection_text)
    assert validation_payload["boundaries"]["derived"] is True
    assert validation_payload["boundaries"]["authoritative"] is False
    assert inspection_payload["boundaries"]["canonical_storage"] is False

    _assert_archive_members_are_portable(archive_root)

    target = tmp_path / "restored"
    restore_request = SessionArchiveRestoreRequest(
        archive_root=archive_root,
        target_root=target,
        overwrite_policy="fail_if_exists",
    )
    first_plan = build_session_archive_restore_plan(restore_request)
    second_plan = build_session_archive_restore_plan(restore_request)
    assert first_plan.report == second_plan.report
    assert len(first_plan.restore_entries) == len(EXPECTED_RELATIVE_FILES)
    assert first_plan.report["boundaries"]["transport_only"] is True

    restore = restore_session_archive_pack(restore_request)
    assert len(restore.restored_paths) == len(EXPECTED_RELATIVE_FILES)
    assert restore.report_path is not None and restore.report_path.is_file()
    restore_text = restore.report_path.read_text(encoding="utf-8")
    _assert_no_machine_paths(restore_text, tmp_path)
    assert json.loads(restore_text)["boundaries"]["canonical_storage"] is False

    _assert_tree_matches(source, target, EXPECTED_RELATIVE_FILES)
    assert _relative_digest_map(source, EXPECTED_RELATIVE_FILES) == before
    assert not (target / "shards").exists()
    assert all((target / relative).is_file() for relative in EXPECTED_RELATIVE_FILES)


def test_session_archive_cli_roundtrip_lifecycle(tmp_path: Path) -> None:
    source = _synthetic_repo(tmp_path / "source")
    archive_root = source / "artifacts/_derived/session_archives/cli-session"
    target = tmp_path / "cli-restored"

    assert (
        session_archive_cli(
            [
                "pack",
                "--repository-root",
                str(source),
                "--archive-id",
                "cli-session",
                "--include-group",
                "features",
                "--include-group",
                "artifacts",
                "--include-group",
                "configs",
                "--include-group",
                "duckdb_snapshot",
                "--include-path",
                "features=data/curated/features_daily",
                "--include-path",
                "artifacts=artifacts",
                "--include-path",
                "configs=configs",
                "--duckdb-snapshot-source-path",
                "data/duckdb/session.duckdb",
                "--max-entries-per-shard",
                "3",
            ]
        )
        == 0
    )
    assert session_archive_cli(["validate", "--archive-root", str(archive_root)]) == 0
    assert session_archive_cli(["inspect", "--archive-root", str(archive_root)]) == 0
    assert (
        session_archive_cli(
            [
                "restore",
                "--archive-root",
                str(archive_root),
                "--target-root",
                str(target),
                "--dry-run",
            ]
        )
        == 0
    )
    assert not (target / EXPECTED_RELATIVE_FILES[0]).exists()
    assert (
        session_archive_cli(
            [
                "restore",
                "--archive-root",
                str(archive_root),
                "--target-root",
                str(target),
            ]
        )
        == 0
    )
    _assert_tree_matches(source, target, EXPECTED_RELATIVE_FILES)


def test_roundtrip_validation_reports_missing_shard_and_restore_fails_before_write(
    tmp_path: Path,
) -> None:
    archive_root = write_session_archive_pack(
        _write_request(_synthetic_repo(tmp_path / "source"))
    ).archive_root
    shard = next((archive_root / "shards").glob("*.tar"))
    shard.unlink()
    target = tmp_path / "target"

    validation = validate_session_archive(archive_root)

    assert validation.status == "failed"
    assert SessionArchiveIssueCode.MISSING_REQUIRED_SHARD in _issue_codes(validation)
    with pytest.raises(SessionArchiveError, match="missing"):
        restore_session_archive_pack(
            SessionArchiveRestoreRequest(archive_root=archive_root, target_root=target)
        )
    assert not target.exists()


def test_roundtrip_validation_reports_checksum_mismatch_and_cli_fails(
    tmp_path: Path,
) -> None:
    archive_root = write_session_archive_pack(
        _write_request(_synthetic_repo(tmp_path / "source"))
    ).archive_root
    shard = next((archive_root / "shards").glob("*.tar"))
    shard.write_bytes(shard.read_bytes() + b"tamper")

    validation = validate_session_archive(archive_root)

    assert validation.status == "failed"
    assert SessionArchiveIssueCode.CHECKSUM_MISMATCH in _issue_codes(validation)
    assert session_archive_cli(["validate", "--archive-root", str(archive_root)]) == 1


def test_roundtrip_validation_rejects_unsafe_member_before_restore(
    tmp_path: Path,
) -> None:
    archive_root = write_session_archive_pack(
        _write_request(_synthetic_repo(tmp_path / "source"))
    ).archive_root
    _replace_first_shard(archive_root, [("../outside.txt", b"bad\n")])
    target = tmp_path / "target"

    validation = validate_session_archive(archive_root)

    assert validation.status == "failed"
    assert SessionArchiveIssueCode.UNSAFE_ARCHIVE_ENTRY in _issue_codes(validation)
    with pytest.raises(SessionArchiveError, match="member path"):
        build_session_archive_restore_plan(
            SessionArchiveRestoreRequest(archive_root=archive_root, target_root=target)
        )
    assert not (tmp_path / "outside.txt").exists()
    assert not target.exists()


def _synthetic_repo(root: Path) -> Path:
    _write_text(
        root / "data/curated/features_daily/symbol=AAPL/year=2025/part-000.parquet",
        "feature,aapl,2025\n",
    )
    _write_text(
        root / "data/curated/features_daily/symbol=MSFT/year=2025/part-000.parquet",
        "feature,msft,2025\n",
    )
    _write_json(
        root / "artifacts/strategies/run_a/manifest.json", {"run_id": "run_a", "strategy": "demo"}
    )
    _write_json(root / "artifacts/strategies/run_a/metrics.json", {"sharpe": 1.25, "trades": 2})
    _write_text(
        root / "artifacts/strategies/run_a/equity_curve.csv", "date,equity\n2025-01-02,1.0\n"
    )
    _write_json(root / "artifacts/alpha/run_alpha_a/manifest.json", {"alpha_id": "alpha_a"})
    _write_json(root / "artifacts/alpha/run_alpha_a/alpha_metrics.json", {"ic": 0.03})
    _write_text(root / "configs/profiles/notebook.yml", "schema_version: 1\nprofile: notebook\n")
    _write_text(root / "configs/strategies/demo_strategy.yml", "strategy: demo\n")
    _write_text(root / "configs/tickers_demo.txt", "AAPL\nMSFT\n")
    _write_text(root / "data/duckdb/session.duckdb", "duckdb snapshot bytes\n")
    return root


def _write_request(root: Path, archive_id: str = "roundtrip-session") -> SessionArchiveWriteRequest:
    return SessionArchiveWriteRequest(
        archive_id=archive_id,
        repository_root=root,
        include_policy=SessionArchiveIncludePolicy(
            include_groups=(
                SessionArchiveLogicalGroup.FEATURES,
                SessionArchiveLogicalGroup.ARTIFACTS,
                SessionArchiveLogicalGroup.CONFIGS,
                SessionArchiveLogicalGroup.DUCKDB_SNAPSHOT,
            ),
            include_paths={
                SessionArchiveLogicalGroup.FEATURES: ("data/curated/features_daily",),
                SessionArchiveLogicalGroup.ARTIFACTS: ("artifacts",),
                SessionArchiveLogicalGroup.CONFIGS: ("configs",),
            },
        ),
        max_entries_per_shard=3,
        max_shard_size_bytes=1024,
        session_id="notebook-roundtrip",
        source_runtime_profile="notebook",
        source_profile_path="configs/profiles/notebook.yml",
        duckdb_snapshot_source_path="data/duckdb/session.duckdb",
    )


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _write_json(path: Path, payload: object) -> Path:
    return _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_digest_map(root: Path, relative_paths: tuple[str, ...]) -> dict[str, str]:
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in sorted(relative_paths)
    }


def _assert_tree_matches(
    expected_root: Path, actual_root: Path, relative_paths: tuple[str, ...]
) -> None:
    for relative in sorted(relative_paths):
        assert (actual_root / relative).read_bytes() == (expected_root / relative).read_bytes()


def _assert_no_machine_paths(text: str, tmp_path: Path) -> None:
    assert str(tmp_path) not in text
    assert tmp_path.as_posix() not in text
    assert "C:/" not in text
    assert "C:\\" not in text
    assert "/Users/" not in text
    assert "/home/" not in text


def _assert_archive_members_are_portable(archive_root: Path) -> None:
    for shard in sorted((archive_root / "shards").glob("*.tar")):
        with tarfile.open(shard, mode="r") as archive:
            names = [member.name for member in archive.getmembers()]
        assert names == sorted(names)
        for name in names:
            path = PurePosixPath(name)
            assert not path.is_absolute()
            assert "\\" not in name
            assert ".." not in path.parts


def _issues(result: object, severity: str) -> tuple[object, ...]:
    return tuple(issue for issue in result.issues if issue.severity == severity)  # type: ignore[attr-defined]


def _issue_codes(result: object) -> set[str]:
    return {issue.code for issue in result.issues}  # type: ignore[attr-defined]


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
    for sidecar_name in ("checksums.json", "manifest.json"):
        payload = json.loads((archive_root / sidecar_name).read_text(encoding="utf-8"))
        rows = payload["shards"]
        for row in rows:
            if row["shard_name"] == shard.name:
                row["checksum"] = digest
                row["size_bytes"] = shard.stat().st_size
        _write_json(archive_root / sidecar_name, payload)
