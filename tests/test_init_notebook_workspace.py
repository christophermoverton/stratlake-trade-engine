from __future__ import annotations

from pathlib import Path
import pytest

from src.cli.init_notebook_workspace import (
    _destination_path,
    _resolve_resource_root,
    initialize_notebook_workspace,
    run_cli,
)


def test_initialize_notebook_workspace_creates_expected_layout(tmp_path: Path) -> None:
    workspace_root = tmp_path / "notebook-workspace"

    summary = initialize_notebook_workspace(workspace_root)

    for directory in ("notebooks", "configs", "docs", "contracts", "artifacts"):
        assert (workspace_root / directory).is_dir()

    assert (workspace_root / "configs" / "features.yml").is_file()
    assert (workspace_root / "configs" / "profiles" / "notebook.yml").is_file()
    assert (workspace_root / "docs" / "notebook_integration.md").is_file()
    assert (workspace_root / "docs" / "examples" / "notebook_execution_api_examples.py").is_file()

    assert summary["workspace_preexisting"] is False
    assert len(summary["copied"]) > 0


def test_initialize_notebook_workspace_does_not_overwrite_by_default(tmp_path: Path) -> None:
    workspace_root = tmp_path / "notebook-workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)

    config_path = workspace_root / "configs" / "features.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("sentinel\n", encoding="utf-8")

    summary = initialize_notebook_workspace(workspace_root)

    assert config_path.read_text(encoding="utf-8") == "sentinel\n"
    assert "configs/features.yml" in summary["skipped"]


def test_initialize_notebook_workspace_force_overwrites_templates(tmp_path: Path) -> None:
    workspace_root = tmp_path / "notebook-workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)

    config_path = workspace_root / "configs" / "features.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("sentinel\n", encoding="utf-8")

    summary = initialize_notebook_workspace(workspace_root, force=True)

    source_text = _resolve_resource_root().joinpath("configs").joinpath("features.yml").read_text(
        encoding="utf-8"
    )
    assert config_path.read_text(encoding="utf-8") == source_text
    assert "configs/features.yml" in summary["overwritten"]


def test_initialize_notebook_workspace_raises_for_non_file_starter_destination(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "notebook-workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)

    conflicting_destination = workspace_root / "configs" / "features.yml"
    conflicting_destination.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError) as exc_info:
        initialize_notebook_workspace(workspace_root)

    message = str(exc_info.value)
    assert "starter destination exists and is not a file" in message.lower()
    assert conflicting_destination.as_posix() in message


def test_run_cli_prints_concise_summary(tmp_path: Path, capsys) -> None:
    workspace_root = tmp_path / "notebook-workspace"

    run_cli(["--root", str(workspace_root)])
    captured = capsys.readouterr()

    assert "Initialized StratLake notebook workspace at:" in captured.out
    assert "Template status:" in captured.out


def test_package_resources_are_discoverable() -> None:
    resource_root = _resolve_resource_root()

    assert resource_root.joinpath("configs").joinpath("features.yml").is_file()
    assert resource_root.joinpath("configs").joinpath("profiles").joinpath("notebook.yml").is_file()
    assert resource_root.joinpath("docs").joinpath("notebook_integration.md").is_file()
    assert (
        resource_root.joinpath("docs").joinpath("examples").joinpath("notebook_execution_api_examples.py").is_file()
    )


def test_destination_path_rejects_writes_outside_workspace_root(tmp_path: Path) -> None:
    workspace_root = tmp_path.resolve()

    with pytest.raises(ValueError):
        _destination_path(workspace_root, "../outside.txt")
