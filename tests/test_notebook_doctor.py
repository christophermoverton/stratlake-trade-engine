from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cli.notebook_doctor import main, run_cli
from src.validation.notebook_doctor import run_notebook_doctor


def test_notebook_doctor_api_passes_with_full_read_only_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _create_workspace(tmp_path / "workspace")
    marketlake_root = tmp_path / "marketlake"
    (marketlake_root / "bars_daily").mkdir(parents=True)
    (marketlake_root / "bars_1m").mkdir(parents=True)

    archive_root = tmp_path / "session_archives" / "archive-001"
    archive_root.mkdir(parents=True)
    for marker in ("manifest.json", "archive_index.json", "checksums.json", "restore_plan.json"):
        (archive_root / marker).write_text("{}\n", encoding="utf-8")

    archive_destination = tmp_path / "archive_destination"
    archive_destination.mkdir(parents=True)

    monkeypatch.setenv("ALPACA_API_KEY", "set")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "set")

    first = run_notebook_doctor(
        root=workspace,
        marketlake_root=marketlake_root,
        archive_root=archive_root,
        archive_destination_root=archive_destination,
        check_configs=True,
        check_universe=True,
        check_archives=True,
        check_secrets=True,
        check_drive=False,
    )
    second = run_notebook_doctor(
        root=workspace,
        marketlake_root=marketlake_root,
        archive_root=archive_root,
        archive_destination_root=archive_destination,
        check_configs=True,
        check_universe=True,
        check_archives=True,
        check_secrets=True,
        check_drive=False,
    )

    payload = first.to_dict()
    assert payload["status"] == "pass"
    assert payload["read_only"] is True
    assert payload == second.to_dict()


def test_notebook_doctor_cli_json_emits_deterministic_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = _create_workspace(tmp_path / "workspace")

    report = run_cli(
        [
            "--root",
            str(workspace),
            "--check-configs",
            "--check-universe",
            "--no-check-marketlake",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    parsed = json.loads(captured.out)
    assert parsed == report
    assert report["status"] in {"pass", "warn"}
    assert "notebook_doctor_status:" in captured.err


def test_notebook_doctor_main_returns_nonzero_on_fail(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing-root"

    exit_code = main(["--root", str(missing_root), "--json"])

    assert exit_code == 1


def test_notebook_doctor_is_read_only_and_does_not_mutate_workspace(tmp_path: Path) -> None:
    workspace = _create_workspace(tmp_path / "workspace")
    before = _snapshot_files(workspace)

    report = run_notebook_doctor(
        root=workspace,
        check_configs=True,
        check_universe=True,
        check_marketlake=False,
    )

    after = _snapshot_files(workspace)
    assert report.read_only is True
    assert before == after


def test_notebook_doctor_redacts_secret_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _create_workspace(tmp_path / "workspace")
    monkeypatch.setenv("BROKER_TOKEN", "super-secret-value")

    report = run_notebook_doctor(
        root=workspace,
        check_marketlake=False,
        check_secrets=True,
        secret_names=["BROKER_TOKEN"],
    ).to_dict()

    serialized = json.dumps(report, sort_keys=True)
    assert "super-secret-value" not in serialized
    assert "BROKER_TOKEN" in serialized
    assert "SET" in serialized


def test_notebook_doctor_fails_when_archive_root_overlaps_target(tmp_path: Path) -> None:
    workspace = _create_workspace(tmp_path / "workspace")
    archive_root = workspace / "artifacts" / "_derived" / "archive"
    archive_root.mkdir(parents=True)
    for marker in ("manifest.json", "archive_index.json", "checksums.json", "restore_plan.json"):
        (archive_root / marker).write_text("{}\n", encoding="utf-8")

    report = run_notebook_doctor(
        root=workspace,
        archive_root=archive_root,
        check_archives=True,
        check_marketlake=False,
    ).to_dict()

    assert report["status"] == "fail"
    assert any(
        check["name"] == "archive_root_not_under_target_root" and check["status"] == "fail"
        for check in report["checks"]
    )


def _create_workspace(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "artifacts").mkdir()
    (root / "data").mkdir()
    (root / ".stratlake").mkdir()

    (root / "configs" / "paths.yml").write_text(
        "marketlake_root: data/curated\nartifacts_root: artifacts\n",
        encoding="utf-8",
    )
    (root / "configs" / "universe.yml").write_text(
        "symbols:\n  - AAPL\n  - MSFT\n",
        encoding="utf-8",
    )
    (root / "configs" / "strategies.yml").write_text("strategies: []\n", encoding="utf-8")
    (root / "configs" / "evaluation.yml").write_text("evaluation: {}\n", encoding="utf-8")
    (root / "configs" / "session.yml").write_text("session: {}\n", encoding="utf-8")

    (root / ".stratlake" / "session.json").write_text(
        json.dumps({"schema_version": 1, "project_name": "demo"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / ".stratlake" / "path_resolution.json").write_text(
        json.dumps({"schema_version": 1, "paths": {}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def _snapshot_files(root: Path) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            snapshot[path.relative_to(root).as_posix()] = path.read_bytes()
    return snapshot
