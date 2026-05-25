from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.session import create_notebook_project_session, write_session_files


def test_session_contract_creation_under_explicit_root(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"

    session = create_notebook_project_session(
        project_name="stratlake-demo",
        project_root=project_root,
        notebook_cwd=tmp_path / "notebook-cwd",
    )

    assert session.schema_version == 1
    assert session.project_name == "stratlake-demo"
    assert session.project_root.path == "."
    assert session.project_root.source.value == "explicit_root"
    assert session.configs_root.path == "configs"
    assert session.artifacts_root.path == "artifacts"
    assert session.features_root.path == "data/curated"
    assert session.marketlake_root.path == "data/curated"
    assert session.notebook_cwd.kind.value == "external_absolute"


def test_write_session_files_creates_required_json_files(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"
    session = create_notebook_project_session(
        project_name="stratlake-demo",
        project_root=project_root,
        notebook_cwd=tmp_path / "notebooks",
        drive_root=tmp_path / "drive" / "stratlake-demo",
    )

    result = write_session_files(session)

    assert result.session_path == project_root / ".stratlake" / "session.json"
    assert result.path_resolution_path == project_root / ".stratlake" / "path_resolution.json"
    session_json = json.loads(result.session_path.read_text(encoding="utf-8"))
    assert session_json["schema_version"] == 1
    assert session_json["project_name"] == "stratlake-demo"
    assert session_json["project_root"] == {
        "path": ".",
        "kind": "project_internal",
        "source": "explicit_root",
    }
    assert session_json["configs_root"]["path"] == "configs"
    assert session_json["drive_root"]["kind"] == "external_absolute"


def test_write_session_files_preserves_existing_user_owned_files_by_default(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "stratlake-demo"
    metadata_dir = project_root / ".stratlake"
    metadata_dir.mkdir(parents=True)
    session_path = metadata_dir / "session.json"
    session_path.write_text('{"user_owned": true}\n', encoding="utf-8")

    session = create_notebook_project_session(project_root=project_root)

    with pytest.raises(FileExistsError, match="Refusing to overwrite existing session metadata"):
        write_session_files(session)

    assert session_path.read_text(encoding="utf-8") == '{"user_owned": true}\n'


def test_write_session_files_does_not_create_or_mutate_canonical_artifacts(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "stratlake-demo"
    artifacts_root = project_root / "artifacts"
    artifacts_root.mkdir(parents=True)
    sentinel = artifacts_root / "sentinel.txt"
    sentinel.write_text("canonical artifact placeholder\n", encoding="utf-8")

    session = create_notebook_project_session(project_root=project_root)
    write_session_files(session)

    assert sentinel.read_text(encoding="utf-8") == "canonical artifact placeholder\n"
    assert not (artifacts_root / "session.json").exists()
    assert sorted(path.name for path in project_root.iterdir()) == [".stratlake", "artifacts"]


def test_notebook_cwd_does_not_affect_output_once_explicit_root_is_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "stratlake-demo"
    first_cwd = tmp_path / "first-cwd"
    second_cwd = tmp_path / "second-cwd"
    first_cwd.mkdir()
    second_cwd.mkdir()

    monkeypatch.chdir(first_cwd)
    first = create_notebook_project_session(
        project_name="stratlake-demo",
        project_root=project_root,
        notebook_cwd=first_cwd,
    ).to_dict()
    monkeypatch.chdir(second_cwd)
    second = create_notebook_project_session(
        project_name="stratlake-demo",
        project_root=project_root,
        notebook_cwd=first_cwd,
    ).to_dict()

    assert first == second


def test_session_writer_rejects_metadata_paths_outside_selected_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "real-root"
    session = create_notebook_project_session(project_root=project_root)
    monkeypatch.setattr("src.session.io.SESSION_DIR_NAME", "../outside")

    with pytest.raises(ValueError, match="outside project root"):
        write_session_files(session)
