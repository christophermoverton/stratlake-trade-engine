from __future__ import annotations

import json
from pathlib import Path

from src.cli.session_export import main as export_main
from src.cli.session_export import run_cli as run_export_cli
from src.cli.session_import import main as import_main
from src.cli.session_import import run_cli as run_import_cli
from src.session import create_notebook_project_session, write_session_files


def _make_session_roots(tmp_path: Path) -> tuple[Path, Path]:
    local_root = tmp_path / "local-stratlake"
    drive_root = tmp_path / "drive-root"
    write_session_files(
        create_notebook_project_session(
            project_root=local_root,
            project_name="demo",
            drive_root=drive_root,
            drive_persistence_enabled=True,
        )
    )
    return local_root, drive_root


def _write(path: Path, text: str = "content\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_export_cli_returns_useful_summary_and_exit_code(
    tmp_path: Path,
    capsys,
) -> None:
    local_root, drive_root = _make_session_roots(tmp_path)
    _write(local_root / "configs" / "alpha.yml")

    exit_code = export_main(
        [
            "--root",
            str(local_root),
            "--drive-root",
            str(drive_root),
            "--include-configs",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "StratLake session export:" in captured.out
    assert "copied:" in captured.out
    assert (drive_root / "configs" / "alpha.yml").is_file()


def test_export_cli_dry_run_writes_no_files_without_manifest(tmp_path: Path) -> None:
    local_root, drive_root = _make_session_roots(tmp_path)
    _write(local_root / "configs" / "alpha.yml")

    summary = run_export_cli(
        [
            "--root",
            str(local_root),
            "--drive-root",
            str(drive_root),
            "--include-configs",
            "--dry-run",
        ]
    )

    assert summary["dry_run"] is True
    assert summary["manifest_path"] is None
    assert not (drive_root / "configs" / "alpha.yml").exists()


def test_export_cli_dry_run_manifest_is_marked_dry_run(tmp_path: Path) -> None:
    local_root, drive_root = _make_session_roots(tmp_path)
    _write(local_root / "configs" / "alpha.yml")

    summary = run_export_cli(
        [
            "--root",
            str(local_root),
            "--drive-root",
            str(drive_root),
            "--include-configs",
            "--dry-run",
            "--write-manifest",
            "--operation-id",
            "dry-run",
        ]
    )

    manifest_path = Path(summary["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    assert not (drive_root / "configs" / "alpha.yml").exists()


def test_import_cli_does_not_overwrite_without_force(tmp_path: Path) -> None:
    local_root, drive_root = _make_session_roots(tmp_path)
    _write(local_root / "configs" / "alpha.yml", "local\n")
    _write(drive_root / "configs" / "alpha.yml", "drive\n")

    exit_code = import_main(
        [
            "--root",
            str(local_root),
            "--drive-root",
            str(drive_root),
            "--include-configs",
        ]
    )

    assert exit_code == 0
    assert (local_root / "configs" / "alpha.yml").read_text(encoding="utf-8") == "local\n"


def test_import_cli_force_overwrites(tmp_path: Path) -> None:
    local_root, drive_root = _make_session_roots(tmp_path)
    _write(local_root / "configs" / "alpha.yml", "local\n")
    _write(drive_root / "configs" / "alpha.yml", "drive\n")

    summary = run_import_cli(
        [
            "--root",
            str(local_root),
            "--drive-root",
            str(drive_root),
            "--include-configs",
            "--force",
        ]
    )

    assert summary["overwritten_count"] >= 1
    assert (local_root / "configs" / "alpha.yml").read_text(encoding="utf-8") == "drive\n"


def test_cli_requires_session_or_reports_clear_error(tmp_path: Path) -> None:
    exit_code = export_main(["--root", str(tmp_path), "--drive-root", str(tmp_path / "drive")])

    assert exit_code == 2
