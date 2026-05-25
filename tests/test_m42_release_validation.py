from __future__ import annotations

import json
import os
from pathlib import Path

from src.cli.init_session import run_cli as run_init_session
from src.cli.session_export import run_cli as run_export
from src.cli.session_import import run_cli as run_import
from src.session import load_session, resolve_session_paths


def test_m42_full_notebook_session_lifecycle_is_deterministic_and_ci_safe(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "stratlake"
    drive_root = tmp_path / "mounted-drive" / "stratlake-demo"
    marketlake_root = tmp_path / "external-marketlake" / "data" / "curated"
    before_environ = dict(os.environ)

    init_summary = run_init_session(
        [
            "--root",
            str(project_root),
            "--project-name",
            "stratlake-demo",
            "--marketlake-root",
            str(marketlake_root),
            "--drive-root",
            str(drive_root),
            "--enable-drive-persistence",
        ]
    )

    assert (project_root / "configs" / "session.yml").is_file()
    assert (project_root / "docs" / "colab_project_sessions.md").is_file()
    assert Path(init_summary["session_path"]).is_file()
    assert Path(init_summary["path_resolution_path"]).is_file()

    session_json = json.loads((project_root / ".stratlake" / "session.json").read_text())
    path_resolution = json.loads(
        (project_root / ".stratlake" / "path_resolution.json").read_text()
    )
    assert session_json["schema_version"] == 1
    assert session_json["project_root"]["path"] == "."
    assert session_json["configs_root"]["path"] == "configs"
    assert session_json["artifacts_root"]["path"] == "artifacts"
    assert session_json["features_root"]["path"] == "data/curated"
    assert session_json["marketlake_root"]["kind"] == "external_absolute"
    assert session_json["drive_root"]["kind"] == "external_absolute"
    assert path_resolution["paths"]["marketlake_root"]["source"] == "explicit_marketlake_root"

    session = load_session(project_root)
    paths = resolve_session_paths(session)
    assert Path(paths["configs_root"].resolved_path) == project_root / "configs"
    assert Path(paths["marketlake_root"].resolved_path) == marketlake_root
    assert Path(paths["drive_root"].resolved_path) == drive_root

    _write(project_root / "configs" / "alpha.yml", "alpha: 1\n")
    _write(project_root / "configs" / ".env", "TOKEN=secret\n")
    _write(project_root / "configs" / "api_key.txt", "secret\n")
    _write(project_root / "artifacts" / "run" / "summary.json", "{}\n")
    _write(project_root / "data" / "curated" / "features_daily" / "part-000.csv", "feature\n")
    _write(marketlake_root / "raw_market" / "part-000.csv", "market\n")

    dry_run = run_export(
        [
            "--root",
            str(project_root),
            "--drive-root",
            str(drive_root),
            "--include-configs",
            "--include-artifacts",
            "--dry-run",
        ]
    )
    assert dry_run["dry_run"] is True
    assert dry_run["copied_count"] == 0
    assert dry_run["manifest_path"] is None
    assert not (drive_root / "configs" / "alpha.yml").exists()

    export_without_data = run_export(
        [
            "--root",
            str(project_root),
            "--drive-root",
            str(drive_root),
            "--include-configs",
            "--include-artifacts",
            "--operation-id",
            "lifecycle",
        ]
    )
    assert export_without_data["copied_count"] >= 3
    assert (drive_root / "configs" / "alpha.yml").is_file()
    assert (drive_root / "artifacts" / "run" / "summary.json").is_file()
    assert not (drive_root / "configs" / ".env").exists()
    assert not (drive_root / "configs" / "api_key.txt").exists()
    assert not (drive_root / "data" / "curated" / "features_daily" / "part-000.csv").exists()
    assert not (drive_root / "market_data" / "raw_market" / "part-000.csv").exists()

    export_with_data = run_export(
        [
            "--root",
            str(project_root),
            "--drive-root",
            str(drive_root),
            "--include-features",
            "--include-market-data",
            "--force",
            "--operation-id",
            "lifecycle-data",
        ]
    )
    assert export_with_data["copied_count"] >= 2
    assert (drive_root / "data" / "curated" / "features_daily" / "part-000.csv").is_file()
    assert (drive_root / "market_data" / "raw_market" / "part-000.csv").is_file()

    manifest = json.loads(Path(export_with_data["manifest_path"]).read_text())
    assert manifest["schema_version"] == 1
    assert manifest["authoritative"] is False
    assert manifest["operation"] == "export"
    assert manifest["include_categories"] == ["features", "market_data", "session_metadata"]
    manifest_order = [(item["category"], item["relative_path"]) for item in manifest["items"]]
    assert manifest_order == sorted(manifest_order)

    local_config = project_root / "configs" / "alpha.yml"
    local_config.write_text("local edit\n", encoding="utf-8")
    no_force_import = run_import(
        [
            "--root",
            str(project_root),
            "--drive-root",
            str(drive_root),
            "--include-configs",
        ]
    )
    assert no_force_import["skipped_count"] >= 1
    assert local_config.read_text(encoding="utf-8") == "local edit\n"

    force_import = run_import(
        [
            "--root",
            str(project_root),
            "--drive-root",
            str(drive_root),
            "--include-configs",
            "--force",
            "--operation-id",
            "restore",
        ]
    )
    assert force_import["overwritten_count"] >= 1
    assert local_config.read_text(encoding="utf-8") == "alpha: 1\n"

    after_environ = dict(os.environ)
    assert after_environ == before_environ


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
