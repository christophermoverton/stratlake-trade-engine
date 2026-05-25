from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.session import (
    create_notebook_project_session,
    find_session_root,
    load_session,
    resolve_session_paths,
    write_path_resolution_report,
    write_session_files,
)


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


def test_find_session_root_finds_session_from_nested_directory(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"
    nested = project_root / "notebooks" / "experiments"
    nested.mkdir(parents=True)
    write_session_files(create_notebook_project_session(project_root=project_root))

    assert find_session_root(nested) == project_root.resolve()


def test_find_session_root_fails_clearly_when_no_session_exists(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No StratLake session root found"):
        find_session_root(tmp_path)


def test_load_session_loads_valid_session_written_by_writer(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"
    write_session_files(
        create_notebook_project_session(
            project_root=project_root,
            project_name="demo",
            marketlake_root="../marketlake/data/curated",
        )
    )

    session = load_session(project_root / "notebooks")

    assert session.project_name == "demo"
    assert session.project_root.resolved_path == project_root.resolve().as_posix()
    assert session.marketlake_root.path == "../marketlake/data/curated"
    assert session.marketlake_root.kind.value == "external_or_project_relative"


def test_load_session_fails_for_missing_session_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No StratLake session root found"):
        load_session(tmp_path)


def test_load_session_fails_for_invalid_json(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"
    session_path = project_root / ".stratlake" / "session.json"
    session_path.parent.mkdir(parents=True)
    session_path.write_text("{not-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid StratLake session JSON"):
        load_session(project_root)


def test_load_session_fails_for_unsupported_schema_version(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"
    write_session_files(create_notebook_project_session(project_root=project_root))
    session_path = project_root / ".stratlake" / "session.json"
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    session_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported StratLake session schema version"):
        load_session(project_root)


def test_resolve_session_paths_uses_explicit_root_without_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "stratlake-demo"
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    write_session_files(create_notebook_project_session(project_root=project_root))
    monkeypatch.chdir(other_cwd)

    paths = resolve_session_paths(project_root)

    assert paths["project_root"].resolved_path == project_root.resolve().as_posix()
    assert paths["configs_root"].resolved_path == (project_root / "configs").resolve().as_posix()
    assert not (other_cwd / ".stratlake").exists()


def test_explicit_overrides_win_over_session_metadata(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"
    external_marketlake = tmp_path / "external" / "curated"
    write_session_files(create_notebook_project_session(project_root=project_root))

    paths = resolve_session_paths(
        project_root,
        overrides={
            "marketlake_root": external_marketlake,
            "artifacts_root": "artifacts/session-demo",
        },
    )

    assert paths["marketlake_root"].path == external_marketlake.resolve().as_posix()
    assert paths["marketlake_root"].kind.value == "external_absolute"
    assert paths["marketlake_root"].source.value == "explicit_override"
    assert paths["artifacts_root"].path == "artifacts/session-demo"
    assert paths["artifacts_root"].source.value == "explicit_override"


def test_environment_fallback_records_provenance_for_partial_session_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "stratlake-demo"
    session = create_notebook_project_session(project_root=project_root)
    partial_payload = session.to_dict()
    partial_payload["project_root"] = {
        **partial_payload["project_root"],
        "resolved_path": project_root.resolve().as_posix(),
    }
    partial_payload["marketlake_root"] = {
        "path": "",
        "kind": "project_internal",
        "source": "default",
    }
    external_marketlake = tmp_path / "env-marketlake" / "curated"
    before = dict(os.environ)
    monkeypatch.setenv("MARKETLAKE_ROOT", str(external_marketlake))

    paths = resolve_session_paths(partial_payload)

    assert paths["marketlake_root"].path == external_marketlake.resolve().as_posix()
    assert paths["marketlake_root"].source.value == "environment_variable"
    assert paths["marketlake_root"].input_path == f"MARKETLAKE_ROOT={external_marketlake}"
    after = dict(os.environ)
    after.pop("MARKETLAKE_ROOT", None)
    assert after == before


def test_resolve_session_paths_preserves_portable_and_external_classification(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "workspace" / "stratlake-demo"
    write_session_files(
        create_notebook_project_session(
            project_root=project_root,
            configs_root="configs/profiles",
            marketlake_root="../marketlake/data/curated",
            drive_root=tmp_path / "drive" / "MyDrive" / "stratlake-demo",
        )
    )

    paths = resolve_session_paths(project_root)

    assert paths["configs_root"].path == "configs/profiles"
    assert "\\" not in paths["configs_root"].path
    assert paths["configs_root"].kind.value == "project_internal"
    assert paths["marketlake_root"].path == "../marketlake/data/curated"
    assert paths["marketlake_root"].kind.value == "external_or_project_relative"
    assert paths["drive_root"].kind.value == "external_absolute"


def test_write_path_resolution_report_writes_deterministic_json(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"
    write_session_files(create_notebook_project_session(project_root=project_root))

    report_path = write_path_resolution_report(
        project_root,
        overrides={"artifacts_root": "artifacts/notebook-demo"},
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_path == project_root / ".stratlake" / "path_resolution.json"
    assert report["paths"]["artifacts_root"]["path"] == "artifacts/notebook-demo"
    assert report["paths"]["artifacts_root"]["source"] == "explicit_override"
    assert report_path.read_text(encoding="utf-8").endswith("\n")


def test_write_path_resolution_report_refuses_unsafe_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "stratlake-demo"
    write_session_files(create_notebook_project_session(project_root=project_root))
    session = load_session(project_root)
    monkeypatch.setattr("src.session.paths.SESSION_DIR_NAME", "../outside")

    with pytest.raises(ValueError, match="outside project root"):
        write_path_resolution_report(session)


def test_path_helpers_do_not_mutate_canonical_artifacts(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"
    artifact = project_root / "artifacts" / "canonical.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("canonical artifact\n", encoding="utf-8")
    write_session_files(create_notebook_project_session(project_root=project_root))

    resolve_session_paths(project_root)
    write_path_resolution_report(project_root)

    assert artifact.read_text(encoding="utf-8") == "canonical artifact\n"
