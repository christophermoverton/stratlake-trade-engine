from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from src.cli import session_archive_restore_bootstrap as restore_bootstrap
from src.session_archive import (
    SessionArchiveError,
    SessionArchiveIncludePolicy,
    SessionArchiveLogicalGroup,
    SessionArchiveWriteRequest,
    write_session_archive_pack,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _repo(root: Path) -> Path:
    _write(root / "data/curated/features_daily/AAPL.parquet", "feature-a\n")
    _write(root / "artifacts/strategies/run_a/manifest.json", '{"run_id":"run_a"}\n')
    _write(root / "configs/strategies.yml", "strategies: []\n")
    _write(root / "configs/profiles/notebook.yml", "schema_version: 1\nprofile: notebook\n")
    return root


def _archive(root: Path, archive_id: str = "archive-a") -> Path:
    result = write_session_archive_pack(
        SessionArchiveWriteRequest(
            archive_id=archive_id,
            repository_root=root,
            include_policy=SessionArchiveIncludePolicy(
                include_groups=(
                    SessionArchiveLogicalGroup.FEATURES,
                    SessionArchiveLogicalGroup.ARTIFACTS,
                    SessionArchiveLogicalGroup.CONFIGS,
                )
            ),
        )
    )
    return result.archive_root


def _argv(archive_root: Path, target_root: Path, *extra: str) -> list[str]:
    return [
        "--archive-root",
        str(archive_root),
        "--target-root",
        str(target_root),
        *extra,
    ]


def _restored_feature(target_root: Path) -> Path:
    return target_root / "data/curated/features_daily/AAPL.parquet"


def test_help_documents_boundaries(capsys: pytest.CaptureFixture[str]) -> None:
    assert restore_bootstrap.main(["--help"]) == 0

    output = capsys.readouterr().out

    assert "derived, disposable, transport-only" in output
    assert "not canonical storage" in output
    assert "not canonical evidence" in output
    assert "not a registry" in output


def test_dry_run_writes_no_restored_files_or_bootstrap_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    target_root = tmp_path / "target"

    code = restore_bootstrap.main(
        _argv(
            archive_root,
            target_root,
            "--validate-before-restore",
            "--inspect-before-restore",
            "--dry-run",
        )
    )

    output = capsys.readouterr().out

    assert code == 0
    assert "Dry run: True" in output
    assert not _restored_feature(target_root).exists()
    assert not (
        target_root / "artifacts/_derived/session_archives/archive-a/restore_bootstrap_report.json"
    ).exists()


def test_dry_run_json_is_deterministic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    target_root = tmp_path / "target"
    argv = _argv(
        archive_root,
        target_root,
        "--validate-before-restore",
        "--inspect-before-restore",
        "--dry-run",
        "--json",
    )

    assert restore_bootstrap.main(argv) == 0
    first = capsys.readouterr().out
    assert restore_bootstrap.main(argv) == 0
    second = capsys.readouterr().out

    payload = json.loads(first)

    assert first == second
    assert payload["dry_run"] is True
    assert payload["validation_status"] in {"passed", "warning"}
    assert payload["inspection_status"] in {"passed", "warning"}
    assert payload["bootstrap_report_path"] is None


def test_successful_local_archive_restore(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    target_root = tmp_path / "target"

    code = restore_bootstrap.main(
        _argv(
            archive_root,
            target_root,
            "--validate-before-restore",
            "--inspect-before-restore",
        )
    )

    report_path = (
        target_root / "artifacts/_derived/session_archives/archive-a/restore_bootstrap_report.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert code == 0
    assert _restored_feature(target_root).read_text(encoding="utf-8") == "feature-a\n"
    assert payload["boundaries"]["derived"] is True
    assert payload["boundaries"]["canonical_storage"] is False
    assert payload["status"] == "restored"


def test_successful_mounted_drive_style_path_restore(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    drive_archive_root = tmp_path / "mounted_drive/stratlake_archives/archive-a"
    shutil.copytree(archive_root, drive_archive_root)
    target_root = tmp_path / "target"

    code = restore_bootstrap.main(
        _argv(
            drive_archive_root,
            target_root,
            "--validate-before-restore",
            "--inspect-before-restore",
        )
    )

    assert code == 0
    assert _restored_feature(target_root).read_text(encoding="utf-8") == "feature-a\n"


def test_validate_before_restore_prevents_restore_when_validation_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    (archive_root / "checksums.json").unlink()
    target_root = tmp_path / "target"

    code = restore_bootstrap.main(_argv(archive_root, target_root, "--validate-before-restore"))

    captured = capsys.readouterr()

    assert code == 1
    assert "missing_checksums" in captured.out
    assert not _restored_feature(target_root).exists()


def test_validation_failure_json_includes_safe_archive_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    (archive_root / "checksums.json").unlink()
    target_root = tmp_path / "target"

    code = restore_bootstrap.main(
        _argv(archive_root, target_root, "--validate-before-restore", "--json")
    )

    payload = json.loads(capsys.readouterr().out)

    assert code == 1
    assert payload["archive_id"] == "archive-a"
    assert payload["status"] == "failed"
    assert any("missing_checksums" in error for error in payload["errors"])
    assert not _restored_feature(target_root).exists()


def test_inspect_before_restore_prevents_restore_when_inspection_has_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    (archive_root / "restore_plan.json").unlink()
    target_root = tmp_path / "target"

    code = restore_bootstrap.main(_argv(archive_root, target_root, "--inspect-before-restore"))

    captured = capsys.readouterr()

    assert code == 1
    assert "missing_restore_plan" in captured.out
    assert not _restored_feature(target_root).exists()


def test_inspection_failure_json_includes_safe_archive_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    (archive_root / "restore_plan.json").unlink()
    target_root = tmp_path / "target"

    code = restore_bootstrap.main(
        _argv(archive_root, target_root, "--inspect-before-restore", "--json")
    )

    payload = json.loads(capsys.readouterr().out)

    assert code == 1
    assert payload["archive_id"] == "archive-a"
    assert payload["status"] == "failed"
    assert any("missing_restore_plan" in error for error in payload["errors"])
    assert not _restored_feature(target_root).exists()


def test_validation_failure_json_uses_null_archive_id_for_missing_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    (archive_root / "manifest.json").unlink()
    target_root = tmp_path / "target"

    code = restore_bootstrap.main(
        _argv(archive_root, target_root, "--validate-before-restore", "--json")
    )

    payload = json.loads(capsys.readouterr().out)

    assert code == 1
    assert payload["archive_id"] is None
    assert payload["status"] == "failed"
    assert any("missing_manifest" in error for error in payload["errors"])


def test_safe_archive_id_returns_none_for_invalid_manifest(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _write(archive_root / "manifest.json", "{bad json\n")

    assert restore_bootstrap._safe_archive_id(archive_root) is None


def test_warning_only_inspection_does_not_block_restore(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    target_root = tmp_path / "target"

    code = restore_bootstrap.main(_argv(archive_root, target_root, "--inspect-before-restore"))

    assert code == 0
    assert _restored_feature(target_root).is_file()


def test_verify_checksums_defaults_to_true(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, bool] = {}

    def _plan(request: object) -> object:
        captured["verify_checksums"] = request.verify_checksums
        return SimpleNamespace(
            archive_id="archive-a",
            archive_root=Path("archive-a"),
            checksum_status="passed",
            overwrite_policy="fail_if_exists",
            restore_entries=(),
            skipped_entries=(),
        )

    monkeypatch.setattr(restore_bootstrap, "build_session_archive_restore_plan", _plan)
    monkeypatch.setattr(restore_bootstrap, "_emit", lambda *_args, **_kwargs: None)

    assert (
        restore_bootstrap.main(
            ["--archive-root", "archive-a", "--target-root", "target", "--dry-run"]
        )
        == 0
    )
    assert captured["verify_checksums"] is True


def test_no_verify_checksums_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, bool] = {}

    def _plan(request: object) -> object:
        captured["verify_checksums"] = request.verify_checksums
        return SimpleNamespace(
            archive_id="archive-a",
            archive_root=Path("archive-a"),
            checksum_status="not_requested",
            overwrite_policy="fail_if_exists",
            restore_entries=(),
            skipped_entries=(),
        )

    monkeypatch.setattr(restore_bootstrap, "build_session_archive_restore_plan", _plan)
    monkeypatch.setattr(restore_bootstrap, "_emit", lambda *_args, **_kwargs: None)

    assert (
        restore_bootstrap.main(
            [
                "--archive-root",
                "archive-a",
                "--target-root",
                "target",
                "--no-verify-checksums",
                "--dry-run",
            ]
        )
        == 0
    )
    assert captured["verify_checksums"] is False


def test_fail_if_exists_preserves_existing_files_by_default(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    target_root = tmp_path / "target"
    _write(_restored_feature(target_root), "local\n")

    code = restore_bootstrap.main(_argv(archive_root, target_root))

    assert code == 2
    assert _restored_feature(target_root).read_text(encoding="utf-8") == "local\n"


def test_json_planning_error_emits_structured_failure_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    target_root = tmp_path / "target"

    def _plan(_request: object) -> object:
        raise SessionArchiveError("planned failure")

    monkeypatch.setattr(restore_bootstrap, "build_session_archive_restore_plan", _plan)

    code = restore_bootstrap.main(_argv(archive_root, target_root, "--dry-run", "--json"))

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 2
    assert captured.err == ""
    assert payload["archive_id"] == "archive-a"
    assert payload["archive_root"] == archive_root.resolve().as_posix()
    assert payload["target_root"] == target_root.resolve().as_posix()
    assert payload["overwrite_policy"] == "fail_if_exists"
    assert payload["restore_overwrite_policy"] == "fail_if_exists"
    assert payload["verify_checksums"] is True
    assert payload["dry_run"] is True
    assert payload["status"] == "failed"
    assert payload["errors"] == ["planned failure"]
    assert payload["exit_code"] == 2
    assert payload["boundaries"]["transport_only"] is True
    assert payload["boundaries"]["canonical_storage"] is False
    assert payload["bootstrap_report_path"] is None
    assert not (
        target_root / "artifacts/_derived/session_archives/archive-a/restore_bootstrap_report.json"
    ).exists()


def test_non_json_planning_error_preserves_stderr_behavior(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    target_root = tmp_path / "target"

    def _plan(_request: object) -> object:
        raise SessionArchiveError("planned failure")

    monkeypatch.setattr(restore_bootstrap, "build_session_archive_restore_plan", _plan)

    code = restore_bootstrap.main(_argv(archive_root, target_root, "--dry-run"))

    captured = capsys.readouterr()

    assert code == 2
    assert captured.out == ""
    assert "error: planned failure" in captured.err


def test_json_planning_error_payload_is_deterministic(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    target_root = tmp_path / "target"

    def _plan(_request: object) -> object:
        raise SessionArchiveError("planned failure")

    monkeypatch.setattr(restore_bootstrap, "build_session_archive_restore_plan", _plan)
    argv = _argv(archive_root, target_root, "--dry-run", "--json")

    assert restore_bootstrap.main(argv) == 2
    first = capsys.readouterr().out
    assert restore_bootstrap.main(argv) == 2
    second = capsys.readouterr().out

    assert first == second


def test_json_restore_error_emits_structured_failure_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    target_root = tmp_path / "target"

    def _restore(_request: object) -> object:
        raise SessionArchiveError("restore failure")

    monkeypatch.setattr(restore_bootstrap, "restore_session_archive_pack", _restore)

    code = restore_bootstrap.main(
        _argv(
            archive_root,
            target_root,
            "--overwrite-policy",
            "overwrite_allowed",
            "--no-verify-checksums",
            "--json",
        )
    )

    payload = json.loads(capsys.readouterr().out)

    assert code == 2
    assert payload["archive_id"] == "archive-a"
    assert payload["errors"] == ["restore failure"]
    assert payload["overwrite_policy"] == "overwrite_allowed"
    assert payload["restore_overwrite_policy"] == "replace_existing"
    assert payload["verify_checksums"] is False
    assert payload["checksum_status"] == "not_requested"
    assert payload["dry_run"] is False


def test_skip_existing_follows_restore_semantics(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    target_root = tmp_path / "target"
    _write(_restored_feature(target_root), "local\n")

    code = restore_bootstrap.main(
        _argv(archive_root, target_root, "--overwrite-policy", "skip_existing")
    )

    assert code == 0
    assert _restored_feature(target_root).read_text(encoding="utf-8") == "local\n"
    assert (target_root / "configs/strategies.yml").is_file()


def test_overwrite_allowed_follows_replace_existing_restore_semantics(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    target_root = tmp_path / "target"
    _write(_restored_feature(target_root), "local\n")

    code = restore_bootstrap.main(
        _argv(archive_root, target_root, "--overwrite-policy", "overwrite_allowed")
    )

    assert code == 0
    assert _restored_feature(target_root).read_text(encoding="utf-8") == "feature-a\n"


def test_report_root_controls_restore_and_bootstrap_report_location(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    target_root = tmp_path / "target"
    report_root = target_root / "artifacts/_derived/custom_restore_reports"

    code = restore_bootstrap.main(
        _argv(archive_root, target_root, "--report-root", str(report_root))
    )

    assert code == 0
    assert (report_root / "restore_report.json").is_file()
    assert (report_root / "restore_bootstrap_report.json").is_file()


def test_bootstrap_report_json_is_deterministic(tmp_path: Path) -> None:
    archive_root = _archive(_repo(tmp_path / "repo"))
    target_root = tmp_path / "target"

    assert restore_bootstrap.main(_argv(archive_root, target_root)) == 0
    report_path = (
        target_root / "artifacts/_derived/session_archives/archive-a/restore_bootstrap_report.json"
    )
    first = report_path.read_text(encoding="utf-8")
    payload = json.loads(first)
    rewritten = restore_bootstrap._write_restore_bootstrap_report(target_root, payload)
    second = rewritten.read_text(encoding="utf-8")

    assert payload["boundaries"]["transport_only"] is True
    assert payload["boundaries"]["authoritative"] is False
    assert payload["overwrite_policy"] == "fail_if_exists"
    assert first == second


def test_module_has_no_google_or_oauth_dependencies() -> None:
    text = Path(restore_bootstrap.__file__).read_text(encoding="utf-8")
    lowered = text.lower()

    assert "google." not in lowered
    assert "oauth" not in lowered
