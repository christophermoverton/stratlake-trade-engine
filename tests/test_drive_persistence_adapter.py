from __future__ import annotations

import json
from pathlib import Path

from src.session import create_notebook_project_session, write_session_files
from src.session.drive_adapter import (
    export_session_to_drive,
    import_session_from_drive,
    plan_session_copy,
)


def _make_session_roots(tmp_path: Path) -> tuple[Path, Path]:
    local_root = tmp_path / "local-stratlake"
    drive_root = tmp_path / "mounted-drive" / "stratlake-demo"
    session = create_notebook_project_session(
        project_root=local_root,
        project_name="demo",
        drive_root=drive_root,
        drive_persistence_enabled=True,
    )
    write_session_files(session)
    return local_root, drive_root


def _write(path: Path, text: str = "content\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_export_dry_run_creates_plan_and_writes_no_copied_files(tmp_path: Path) -> None:
    local_root, drive_root = _make_session_roots(tmp_path)
    _write(local_root / "configs" / "alpha.yml", "alpha: 1\n")

    result = export_session_to_drive(
        root=local_root,
        include_categories=("configs",),
        dry_run=True,
    )

    planned = [item for item in result.plan.items if item.category == "configs"]
    assert planned[0].status == "would_copy"
    assert result.copied_count == 0
    assert result.manifest_path is None
    assert not (drive_root / "configs" / "alpha.yml").exists()


def test_export_copies_selected_configs_only_when_included(tmp_path: Path) -> None:
    local_root, drive_root = _make_session_roots(tmp_path)
    _write(local_root / "configs" / "alpha.yml")
    _write(local_root / "docs" / "guide.md")

    export_session_to_drive(root=local_root, include_categories=("configs",))

    assert (drive_root / "configs" / "alpha.yml").is_file()
    assert not (drive_root / "docs" / "guide.md").exists()


def test_export_copies_selected_artifacts_only_when_included(tmp_path: Path) -> None:
    local_root, drive_root = _make_session_roots(tmp_path)
    _write(local_root / "artifacts" / "run" / "summary.json", "{}\n")
    _write(local_root / "configs" / "alpha.yml")

    export_session_to_drive(root=local_root, include_categories=("artifacts",))

    assert (drive_root / "artifacts" / "run" / "summary.json").is_file()
    assert not (drive_root / "configs" / "alpha.yml").exists()


def test_export_feature_and_market_data_require_explicit_categories(tmp_path: Path) -> None:
    local_root, drive_root = _make_session_roots(tmp_path)
    _write(local_root / "data" / "curated" / "features.parquet", "feature-data\n")

    export_session_to_drive(root=local_root, include_categories=("configs",))
    assert not (drive_root / "data" / "curated" / "features.parquet").exists()

    export_session_to_drive(root=local_root, include_categories=("features",), force=True)
    assert (drive_root / "data" / "curated" / "features.parquet").is_file()

    market_target = drive_root / "data" / "curated" / "market.parquet"
    _write(local_root / "data" / "curated" / "market.parquet", "market-data\n")
    market_target.unlink(missing_ok=True)
    export_session_to_drive(root=local_root, include_categories=("market_data",), force=True)
    assert market_target.is_file()


def test_safe_default_excludes_sensitive_and_cache_files(tmp_path: Path) -> None:
    local_root, drive_root = _make_session_roots(tmp_path)
    _write(local_root / "configs" / "alpha.yml")
    _write(local_root / "configs" / ".env", "TOKEN=secret\n")
    _write(local_root / "configs" / "api_key.txt", "secret\n")
    _write(local_root / "configs" / "credentials.json", "{}\n")
    _write(local_root / "configs" / ".ipynb_checkpoints" / "alpha.yml")
    _write(local_root / "configs" / "__pycache__" / "cached.pyc", "bytecode\n")
    _write(local_root / "configs" / "scratch.tmp", "tmp\n")

    export_session_to_drive(root=local_root, include_categories=("configs",))

    assert (drive_root / "configs" / "alpha.yml").is_file()
    assert not (drive_root / "configs" / ".env").exists()
    assert not (drive_root / "configs" / "api_key.txt").exists()
    assert not (drive_root / "configs" / "credentials.json").exists()
    assert not (drive_root / "configs" / ".ipynb_checkpoints").exists()
    assert not (drive_root / "configs" / "__pycache__").exists()
    assert not (drive_root / "configs" / "scratch.tmp").exists()


def test_manifest_includes_deterministic_order_and_file_metadata(tmp_path: Path) -> None:
    local_root, _drive_root = _make_session_roots(tmp_path)
    _write(local_root / "configs" / "b.yml", "b\n")
    _write(local_root / "configs" / "a.yml", "a\n")

    result = export_session_to_drive(
        root=local_root,
        include_categories=("configs",),
        operation_id="deterministic-test",
    )

    assert result.manifest_path is not None
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    item_paths = [item["relative_path"] for item in manifest["items"] if item["category"] == "configs"]
    assert item_paths == ["a.yml", "b.yml"]
    item = next(item for item in manifest["items"] if item["relative_path"] == "a.yml")
    assert item["source_path"]
    assert item["destination_path"]
    assert item["category"] == "configs"
    assert item["size_bytes"] == len((local_root / "configs" / "a.yml").read_bytes())
    assert len(item["sha256"]) == 64
    assert item["status"] == "copied"
    assert manifest["authoritative"] is False


def test_drive_manifest_uses_resolved_artifacts_root(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace"
    custom_artifacts = tmp_path / "custom-artifacts"
    drive_root = tmp_path / "mounted-drive"
    session = create_notebook_project_session(
        project_root=project_root,
        artifacts_root=custom_artifacts,
        drive_root=drive_root,
    )
    write_session_files(session)
    _write(project_root / "configs" / "demo.yml", "demo: true\n")

    result = export_session_to_drive(
        root=project_root,
        include_categories=("configs",),
        operation_id="custom-artifacts",
    )

    expected_manifest = (
        custom_artifacts
        / "_derived"
        / "notebook_sessions"
        / "export_custom-artifacts"
        / "drive_sync_manifest.json"
    ).resolve()
    unexpected_project_manifest = (
        project_root
        / "artifacts"
        / "_derived"
        / "notebook_sessions"
        / "export_custom-artifacts"
        / "drive_sync_manifest.json"
    )

    assert result.manifest_path == expected_manifest
    assert expected_manifest.is_file()
    assert not unexpected_project_manifest.exists()
    manifest = json.loads(expected_manifest.read_text(encoding="utf-8"))
    assert manifest["artifacts_root"] == custom_artifacts.resolve().as_posix()
    assert manifest["authoritative"] is False


def test_import_does_not_overwrite_by_default_and_force_overwrites(tmp_path: Path) -> None:
    local_root, drive_root = _make_session_roots(tmp_path)
    _write(local_root / "configs" / "alpha.yml", "local\n")
    _write(drive_root / "configs" / "alpha.yml", "drive\n")

    result = import_session_from_drive(root=local_root, include_categories=("configs",))
    assert result.skipped_count >= 1
    assert (local_root / "configs" / "alpha.yml").read_text(encoding="utf-8") == "local\n"

    result = import_session_from_drive(root=local_root, include_categories=("configs",), force=True)
    assert result.overwritten_count >= 1
    assert (local_root / "configs" / "alpha.yml").read_text(encoding="utf-8") == "drive\n"


def test_adapter_uses_generic_mounted_filesystem_paths(tmp_path: Path) -> None:
    local_root, drive_root = _make_session_roots(tmp_path)
    generic_mount = tmp_path / "generic-mounted-filesystem"
    _write(local_root / "configs" / "alpha.yml")

    export_session_to_drive(
        root=local_root,
        drive_root=generic_mount,
        include_categories=("configs",),
    )

    assert (generic_mount / "configs" / "alpha.yml").is_file()
    assert drive_root != generic_mount


def test_plan_session_copy_is_deterministic(tmp_path: Path) -> None:
    local_root, _drive_root = _make_session_roots(tmp_path)
    _write(local_root / "configs" / "b.yml")
    _write(local_root / "configs" / "a.yml")

    first = plan_session_copy(
        operation="export",
        root=local_root,
        include_categories=("configs",),
        dry_run=True,
    )
    second = plan_session_copy(
        operation="export",
        root=local_root,
        include_categories=("configs",),
        dry_run=True,
    )

    assert [item.to_dict() for item in first.items] == [item.to_dict() for item in second.items]
