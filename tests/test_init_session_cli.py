from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

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

    run_cli(
        [
            "--root",
            str(project_root),
            "--project-name",
            "demo",
            "--notebook-configs",
        ]
    )

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


def test_notebook_configs_generate_expected_bundle_files(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"
    marketlake_root = tmp_path / "external-marketlake" / "data" / "curated"
    drive_root = tmp_path / "drive" / "MyDrive" / "stratlake-demo"

    summary = run_cli(
        [
            "--root",
            str(project_root),
            "--project-name",
            "stratlake-demo",
            "--marketlake-root",
            str(marketlake_root),
            "--drive-root",
            str(drive_root),
            "--notebook-configs",
        ]
    )

    paths_path = project_root / "configs" / "paths.yml"
    universe_path = project_root / "configs" / "universe.yml"
    tickers_path = project_root / "configs" / "tickers_sample.txt"

    assert paths_path.is_file()
    assert universe_path.is_file()
    assert tickers_path.is_file()

    paths_payload = yaml.safe_load(paths_path.read_text(encoding="utf-8"))
    universe_payload = yaml.safe_load(universe_path.read_text(encoding="utf-8"))
    tickers_lines = tickers_path.read_text(encoding="utf-8").splitlines()

    assert paths_payload["project_root"] == "."
    assert paths_payload["configs_root"] == "configs"
    assert paths_payload["marketlake_root"] == marketlake_root.resolve().as_posix()
    assert paths_payload["drive_root"] == drive_root.resolve().as_posix()
    assert paths_payload["path_kinds"]["marketlake_root"] == "external_absolute"
    assert paths_payload["path_kinds"]["drive_root"] == "external_absolute"

    assert universe_payload["tickers_file"] == "configs/tickers_sample.txt"
    assert universe_payload["timeframe"] == "1D"
    assert tickers_lines == sorted(tickers_lines)

    assert summary["notebook_configs"]["requested"] is True
    assert summary["notebook_configs"]["force"] is False
    assert summary["notebook_configs"]["generated"] == [
        "configs/paths.yml",
        "configs/universe.yml",
        "configs/tickers_sample.txt",
    ]


def test_notebook_configs_preserve_existing_user_files_by_default(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"
    configs_root = project_root / "configs"
    configs_root.mkdir(parents=True)
    sentinel_paths = "marketlake_root: keep/me\n"
    sentinel_universe = "tickers_file: configs/custom.txt\n"
    sentinel_tickers = "CUSTOM\n"
    (configs_root / "paths.yml").write_text(sentinel_paths, encoding="utf-8")
    (configs_root / "universe.yml").write_text(sentinel_universe, encoding="utf-8")
    (configs_root / "tickers_sample.txt").write_text(sentinel_tickers, encoding="utf-8")

    summary = run_cli(
        [
            "--root",
            str(project_root),
            "--project-name",
            "demo",
            "--notebook-configs",
        ]
    )

    assert (configs_root / "paths.yml").read_text(encoding="utf-8") == sentinel_paths
    assert (configs_root / "universe.yml").read_text(encoding="utf-8") == sentinel_universe
    assert (configs_root / "tickers_sample.txt").read_text(encoding="utf-8") == sentinel_tickers
    assert summary["notebook_configs"]["generated"] == []
    assert summary["notebook_configs"]["overwritten"] == []
    assert summary["notebook_configs"]["skipped"] == [
        "configs/paths.yml",
        "configs/universe.yml",
        "configs/tickers_sample.txt",
    ]


def test_notebook_configs_force_overwrites_only_bundle_targets(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"
    run_cli(["--root", str(project_root), "--project-name", "demo"])

    paths = project_root / "configs" / "paths.yml"
    universe = project_root / "configs" / "universe.yml"
    tickers = project_root / "configs" / "tickers_sample.txt"
    unrelated = project_root / "configs" / "custom_user_config.yml"

    paths.write_text("marketlake_root: stale\n", encoding="utf-8")
    universe.write_text("tickers_file: stale.txt\n", encoding="utf-8")
    tickers.write_text("STALE\n", encoding="utf-8")
    unrelated.write_text("do-not-touch: true\n", encoding="utf-8")

    summary = run_cli(
        [
            "--root",
            str(project_root),
            "--project-name",
            "demo",
            "--notebook-configs",
            "--force-notebook-configs",
            "--force",
        ]
    )

    assert "stale" not in paths.read_text(encoding="utf-8")
    assert "stale.txt" not in universe.read_text(encoding="utf-8")
    assert tickers.read_text(encoding="utf-8") != "STALE\n"
    assert unrelated.read_text(encoding="utf-8") == "do-not-touch: true\n"
    assert summary["notebook_configs"]["overwritten"] == [
        "configs/paths.yml",
        "configs/universe.yml",
        "configs/tickers_sample.txt",
    ]


def test_notebook_configs_are_deterministic_for_same_inputs(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"

    run_cli(
        [
            "--root",
            str(project_root),
            "--project-name",
            "demo",
            "--notebook-configs",
        ]
    )

    first_paths = (project_root / "configs" / "paths.yml").read_text(encoding="utf-8")
    first_universe = (project_root / "configs" / "universe.yml").read_text(encoding="utf-8")
    first_tickers = (project_root / "configs" / "tickers_sample.txt").read_text(encoding="utf-8")

    run_cli(
        [
            "--root",
            str(project_root),
            "--project-name",
            "demo",
            "--notebook-configs",
            "--force-notebook-configs",
            "--force",
        ]
    )

    assert (project_root / "configs" / "paths.yml").read_text(encoding="utf-8") == first_paths
    assert (project_root / "configs" / "universe.yml").read_text(encoding="utf-8") == first_universe
    assert (project_root / "configs" / "tickers_sample.txt").read_text(encoding="utf-8") == first_tickers


def test_force_notebook_configs_requires_notebook_configs(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"

    with pytest.raises(SystemExit) as exc_info:
        run_cli(["--root", str(project_root), "--force-notebook-configs"])

    assert exc_info.value.code == 2


def test_path_resolution_metadata_records_notebook_bundle_status(tmp_path: Path) -> None:
    project_root = tmp_path / "stratlake-demo"

    run_cli(
        [
            "--root",
            str(project_root),
            "--project-name",
            "demo",
            "--notebook-configs",
        ]
    )

    resolution_json = json.loads(
        (project_root / ".stratlake" / "path_resolution.json").read_text(encoding="utf-8")
    )
    bundle = resolution_json["notebook_config_bundle"]

    assert bundle["requested"] is True
    assert bundle["force"] is False
    assert bundle["config_dir"] == "configs"
    assert bundle["generated"] == [
        "configs/paths.yml",
        "configs/universe.yml",
        "configs/tickers_sample.txt",
    ]
    assert bundle["session_root"] == project_root.resolve().as_posix()
    assert bundle["marketlake_root"]["kind"] in {
        "project_internal",
        "external_absolute",
        "external_or_project_relative",
    }
