from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.cli.init_session import main, run_cli
from src.cli.init_notebook_workspace import _resolve_resource_root


def test_init_session_first_run_creates_workspace_and_session_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_root = tmp_path / "stratlake-demo"

    summary = run_cli(
        [
            "--root",
            str(project_root),
            "--project-name",
            "demo",
            "--marketlake-root",
            "./fintech/data/curated",
        ]
    )
    captured = capsys.readouterr()

    assert (project_root / "notebooks").is_dir()
    assert (project_root / "configs" / "session.yml").is_file()
    assert (project_root / "docs" / "notebook_integration.md").is_file()
    assert (project_root / ".stratlake" / "session.json").is_file()
    assert (project_root / ".stratlake" / "path_resolution.json").is_file()
    assert summary["session_path"] == (project_root / ".stratlake" / "session.json").as_posix()
    assert "Initialized StratLake notebook session at:" in captured.out
    assert "Template status:" in captured.out
    assert "Next: open notebooks from this project root" in captured.out


def test_init_session_writes_required_session_json_fields(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"

    run_cli(["--root", str(project_root), "--project-name", "demo"])

    session_json = json.loads(
        (project_root / ".stratlake" / "session.json").read_text(encoding="utf-8")
    )
    resolution_json = json.loads(
        (project_root / ".stratlake" / "path_resolution.json").read_text(encoding="utf-8")
    )
    assert session_json["schema_version"] == 1
    assert session_json["project_name"] == "demo"
    assert session_json["project_root"]["path"] == "."
    assert session_json["configs_root"]["path"] == "configs"
    assert session_json["artifacts_root"]["path"] == "artifacts"
    assert session_json["features_root"]["path"] == "data/curated"
    assert session_json["drive_persistence"] == {
        "enabled": False,
        "mode": "metadata_only",
    }
    assert resolution_json["paths"]["project_root"]["source"] == "explicit_root"
    assert resolution_json["paths"]["configs_root"]["base"] == project_root.resolve().as_posix()


def test_repeated_run_without_force_preserves_user_files_and_fails_clearly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_root = tmp_path / "stratlake-demo"
    run_cli(["--root", str(project_root), "--project-name", "demo"])
    user_config = project_root / "configs" / "session.yml"
    user_config.write_text("user-owned: true\n", encoding="utf-8")

    exit_code = main(["--root", str(project_root), "--project-name", "demo"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Session metadata already exists" in captured.err
    assert user_config.read_text(encoding="utf-8") == "user-owned: true\n"


def test_force_refreshes_known_templates_and_session_metadata_only(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"
    run_cli(["--root", str(project_root), "--project-name", "demo"])
    known_template = project_root / "configs" / "session.yml"
    unrelated_config = project_root / "configs" / "custom_user_config.yml"
    artifact = project_root / "artifacts" / "canonical.txt"
    known_template.write_text("user-edited-template: true\n", encoding="utf-8")
    unrelated_config.write_text("do-not-touch: true\n", encoding="utf-8")
    artifact.write_text("canonical artifact\n", encoding="utf-8")

    summary = run_cli(
        [
            "--root",
            str(project_root),
            "--project-name",
            "demo-force",
            "--force",
        ]
    )

    assert "configs/session.yml" in summary["bootstrap"]["overwritten"]
    assert "user-edited-template" not in known_template.read_text(encoding="utf-8")
    assert unrelated_config.read_text(encoding="utf-8") == "do-not-touch: true\n"
    assert artifact.read_text(encoding="utf-8") == "canonical artifact\n"
    session_json = json.loads(
        (project_root / ".stratlake" / "session.json").read_text(encoding="utf-8")
    )
    assert session_json["project_name"] == "demo-force"


def test_external_marketlake_root_is_classified_as_external(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"
    marketlake_root = tmp_path / "external-marketlake" / "data" / "curated"

    run_cli(
        [
            "--root",
            str(project_root),
            "--marketlake-root",
            str(marketlake_root),
        ]
    )

    session_json = json.loads(
        (project_root / ".stratlake" / "session.json").read_text(encoding="utf-8")
    )
    assert session_json["marketlake_root"]["path"] == marketlake_root.resolve().as_posix()
    assert session_json["marketlake_root"]["kind"] == "external_absolute"
    assert session_json["marketlake_root"]["source"] == "explicit_marketlake_root"


def test_drive_persistence_records_metadata_without_drive_sync(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"
    drive_root = tmp_path / "drive" / "MyDrive" / "stratlake-demo"

    run_cli(
        [
            "--root",
            str(project_root),
            "--drive-root",
            str(drive_root),
            "--enable-drive-persistence",
        ]
    )

    session_json = json.loads(
        (project_root / ".stratlake" / "session.json").read_text(encoding="utf-8")
    )
    assert session_json["drive_root"]["path"] == drive_root.resolve().as_posix()
    assert session_json["drive_root"]["kind"] == "external_absolute"
    assert session_json["drive_persistence"] == {
        "enabled": True,
        "mode": "metadata_only",
    }
    assert not drive_root.exists()


def test_enable_drive_persistence_requires_drive_root(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"

    with pytest.raises(SystemExit) as exc_info:
        run_cli(["--root", str(project_root), "--enable-drive-persistence"])

    assert exc_info.value.code == 2
    assert not project_root.exists()


def test_init_session_uses_explicit_root_from_any_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "stratlake-demo"
    notebook_cwd = tmp_path / "notebook-cwd"
    notebook_cwd.mkdir()
    monkeypatch.chdir(notebook_cwd)

    run_cli(["--root", str(project_root), "--project-name", "demo"])

    assert (project_root / ".stratlake" / "session.json").is_file()
    assert not (notebook_cwd / ".stratlake").exists()


def test_init_session_does_not_mutate_env_files_or_process_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "stratlake-demo"
    env_path = project_root / ".env"
    project_root.mkdir()
    env_path.write_text("SENTINEL=1\n", encoding="utf-8")
    before_environ = dict(os.environ)
    monkeypatch.setenv("STRATLAKE_TEST_SENTINEL", "keep")

    run_cli(["--root", str(project_root), "--project-name", "demo"])

    assert env_path.read_text(encoding="utf-8") == "SENTINEL=1\n"
    assert os.environ["STRATLAKE_TEST_SENTINEL"] == "keep"
    after_environ = dict(os.environ)
    after_environ.pop("STRATLAKE_TEST_SENTINEL", None)
    assert after_environ == before_environ


def test_init_session_does_not_mutate_package_resource_templates(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"
    resource = _resolve_resource_root().joinpath("configs").joinpath("session.yml")
    before = resource.read_text(encoding="utf-8")

    run_cli(["--root", str(project_root), "--project-name", "demo", "--force"])

    assert resource.read_text(encoding="utf-8") == before


def test_unsafe_session_metadata_write_outside_root_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "stratlake-demo"
    monkeypatch.setattr("src.cli.init_session.SESSION_DIR_NAME", "../outside")

    exit_code = main(["--root", str(project_root)])

    assert exit_code == 2
    assert not project_root.exists()
