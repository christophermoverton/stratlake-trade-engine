from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cli.session_archive import main
from src.session_archive.manifest import SessionArchiveLogicalGroup
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


def test_top_level_help_documents_archive_boundaries(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--help"]) == 0

    output = capsys.readouterr().out

    assert "derived, disposable, transport-only" in output
    assert "not canonical storage" in output


@pytest.mark.parametrize("command", ["pack", "restore", "validate", "inspect"])
def test_subcommand_help_documents_archive_boundaries(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([command, "--help"]) == 0

    output = capsys.readouterr().out

    assert "derived, disposable, transport-only" in output
    assert "not canonical storage" in output


def test_pack_command_creates_archive_and_supports_groups(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path / "repo")

    code = main(
        [
            "pack",
            "--repository-root",
            str(root),
            "--archive-id",
            "session-a",
            "--include-group",
            "features",
            "--include-path",
            "features=data/curated/features_daily",
        ]
    )

    output = capsys.readouterr().out

    assert code == 0
    assert "Session archive pack created" in output
    assert "Groups: features" in output
    assert (root / "artifacts/_derived/session_archives/session-a/manifest.json").is_file()


def test_pack_dry_run_does_not_write_archive_pack(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path / "repo")

    code = main(
        [
            "pack",
            "--repository-root",
            str(root),
            "--archive-id",
            "session-a",
            "--include-group",
            "configs",
            "--include-path",
            "configs=configs",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out

    assert code == 0
    assert "dry-run complete" in output
    assert not (root / "artifacts/_derived/session_archives/session-a").exists()


def test_restore_command_restores_archive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    target = tmp_path / "target"

    code = main(
        [
            "restore",
            "--archive-root",
            str(archive_root),
            "--target-root",
            str(target),
            "--overwrite-policy",
            "fail_if_exists",
        ]
    )

    output = capsys.readouterr().out

    assert code == 0
    assert "Session archive restore complete" in output
    assert (target / "configs/strategies.yml").read_text(encoding="utf-8") == "strategies: []\n"


def test_restore_dry_run_does_not_extract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    target = tmp_path / "target"

    code = main(
        [
            "restore",
            "--archive-root",
            str(archive_root),
            "--target-root",
            str(target),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out

    assert code == 0
    assert "restore dry-run complete" in output
    assert not (target / "configs/strategies.yml").exists()


def test_restore_invalid_archive_returns_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_root = tmp_path / "bad"
    archive_root.mkdir()

    code = main(
        [
            "restore",
            "--archive-root",
            str(archive_root),
            "--target-root",
            str(tmp_path / "target"),
        ]
    )

    captured = capsys.readouterr()

    assert code == 2
    assert "error:" in captured.err


def test_validate_command_writes_output_root_and_returns_zero_for_warnings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    output_root = tmp_path / "reports" / "artifacts"

    code = main(
        [
            "validate",
            "--archive-root",
            str(archive_root),
            "--output-root",
            str(output_root),
        ]
    )

    output = capsys.readouterr().out
    report_path = output_root / "_derived/session_archives/session-a/validation_report.json"

    assert code == 0
    assert "Session archive validation: warning" in output
    assert report_path.is_file()


def test_validate_command_supports_output_path_and_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    output_path = tmp_path / "reports" / "validation.json"

    code = main(
        [
            "validate",
            "--archive-root",
            str(archive_root),
            "--output-path",
            str(output_path),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["archive_id"] == "session-a"
    assert output_path.is_file()


def test_validate_invalid_archive_returns_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    (archive_root / "manifest.json").unlink()

    code = main(["validate", "--archive-root", str(archive_root)])

    output = capsys.readouterr().out

    assert code == 1
    assert "missing_manifest" in output


def test_inspect_command_prints_summary_and_writes_output_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_root = _archive(_repo(tmp_path / "source"))
    output_root = tmp_path / "reports" / "artifacts"

    code = main(
        [
            "inspect",
            "--archive-root",
            str(archive_root),
            "--output-root",
            str(output_root),
        ]
    )

    output = capsys.readouterr().out
    report_path = output_root / "_derived/session_archives/session-a/inspection_report.json"

    assert code == 0
    assert "Session archive inspection: warning" in output
    assert "Groups: artifacts, configs, features" in output
    assert report_path.is_file()


def test_inspect_command_supports_json_and_invalid_archive_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_root = tmp_path / "bad"
    archive_root.mkdir()

    code = main(["inspect", "--archive-root", str(archive_root), "--json"])

    payload = json.loads(capsys.readouterr().out)

    assert code == 1
    assert payload["status"] == "failed"
    assert payload["issues"][0]["code"] == "missing_manifest"
