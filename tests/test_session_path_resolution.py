from __future__ import annotations

import json
from pathlib import Path

from src.session import create_notebook_project_session, write_session_files


def test_project_internal_paths_serialize_as_posix_relative_paths(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"

    session = create_notebook_project_session(
        project_root=project_root,
        configs_root=Path("configs") / "profiles",
        artifacts_root=Path("artifacts") / "notebooks" / "run_001",
        features_root=Path("data") / "curated",
    )

    assert session.configs_root.path == "configs/profiles"
    assert session.artifacts_root.path == "artifacts/notebooks/run_001"
    assert session.features_root.path == "data/curated"
    assert "\\" not in session.artifacts_root.path
    assert session.configs_root.kind.value == "project_internal"


def test_external_absolute_paths_are_marked_external(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"
    external_marketlake_root = tmp_path / "external-marketlake" / "curated"
    external_drive_root = tmp_path / "drive" / "MyDrive" / "stratlake-demo"

    session = create_notebook_project_session(
        project_root=project_root,
        marketlake_root=external_marketlake_root,
        drive_root=external_drive_root,
    )

    assert session.marketlake_root.path == external_marketlake_root.resolve().as_posix()
    assert session.marketlake_root.kind.value == "external_absolute"
    assert session.marketlake_root.source.value == "explicit_marketlake_root"
    assert session.drive_root is not None
    assert session.drive_root.path == external_drive_root.resolve().as_posix()
    assert session.drive_root.kind.value == "external_absolute"
    assert session.drive_root.source.value == "explicit_drive_root"


def test_project_relative_paths_that_escape_root_keep_their_relative_text(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "workspace" / "stratlake-demo"

    session = create_notebook_project_session(
        project_root=project_root,
        marketlake_root="../fintech/data/curated",
    )

    assert session.marketlake_root.path == "../fintech/data/curated"
    assert session.marketlake_root.kind.value == "external_or_project_relative"
    assert session.marketlake_root.resolved_path.endswith("/workspace/fintech/data/curated")


def test_path_resolution_report_contains_readable_provenance(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"
    notebook_cwd = tmp_path / "cloud-notebook-root"
    external_marketlake_root = tmp_path / "marketlake" / "data" / "curated"

    session = create_notebook_project_session(
        project_name="stratlake-demo",
        project_root=project_root,
        notebook_cwd=notebook_cwd,
        marketlake_root=external_marketlake_root,
    )
    result = write_session_files(session)

    report = json.loads(result.path_resolution_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["project_name"] == "stratlake-demo"
    assert report["paths"]["notebook_cwd"]["source"] == "current_working_directory"
    assert report["paths"]["notebook_cwd"]["resolved_path"] == notebook_cwd.resolve().as_posix()
    assert report["paths"]["project_root"]["path"] == "."
    assert report["paths"]["configs_root"]["base"] == project_root.resolve().as_posix()
    assert report["paths"]["marketlake_root"]["kind"] == "external_absolute"
    assert report["paths"]["marketlake_root"]["input_path"] == str(external_marketlake_root)
