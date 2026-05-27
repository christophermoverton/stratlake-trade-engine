from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cli import session_archive_bootstrap as bootstrap


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _repo(root: Path) -> Path:
    _write(root / "data/curated/features_daily/AAPL.parquet", "feature-a\n")
    _write(root / "artifacts/strategies/run_a/manifest.json", '{"run_id":"run_a"}\n')
    _write(root / "configs/strategies.yml", "strategies: []\n")
    _write(root / "configs/profiles/notebook.yml", "schema_version: 1\nprofile: notebook\n")
    return root


def test_help_documents_boundaries(capsys: pytest.CaptureFixture[str]) -> None:
    assert bootstrap.main(["--help"]) == 0

    output = capsys.readouterr().out

    assert "derived, disposable, transport-only" in output
    assert "not canonical storage" in output


def test_include_flags_map_to_logical_groups(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path / "repo")

    code = bootstrap.main(
        [
            "--root",
            str(root),
            "--archive-id",
            "archive-a",
            "--include-configs",
            "--dry-run",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["included_logical_groups"] == ["configs"]


def test_dry_run_writes_no_archive_copy_or_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path / "repo")
    drive = tmp_path / "drive"

    code = bootstrap.main(
        [
            "--root",
            str(root),
            "--archive-id",
            "archive-a",
            "--drive-root",
            str(drive),
            "--include-features",
            "--include-artifacts",
            "--include-configs",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out

    assert code == 0
    assert "Copy status: not_started_dry_run" in output
    assert not (root / "artifacts/_derived/session_archives/archive-a").exists()
    assert not drive.exists()
    assert not (root / "artifacts/_derived/session_archives/archive-a/bootstrap_report.json").exists()


def test_local_only_bootstrap_creates_archive_and_report(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path / "repo")

    code = bootstrap.main(
        [
            "--root",
            str(root),
            "--archive-id",
            "archive-a",
            "--include-features",
            "--include-artifacts",
            "--include-configs",
        ]
    )

    assert code == 0
    assert (root / "artifacts/_derived/session_archives/archive-a/manifest.json").is_file()
    report_path = root / "artifacts/_derived/session_archives/archive-a/bootstrap_report.json"
    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["boundaries"]["transport_only"] is True
    assert payload["boundaries"]["canonical_storage"] is False


def test_drive_copy_creates_destination_archive(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    drive = tmp_path / "drive"

    code = bootstrap.main(
        [
            "--root",
            str(root),
            "--archive-id",
            "archive-a",
            "--drive-root",
            str(drive),
            "--include-features",
            "--include-artifacts",
            "--include-configs",
        ]
    )

    assert code == 0
    assert (drive / "archive-a/manifest.json").is_file()


def test_fail_if_exists_rejects_existing_destination(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path / "repo")
    drive_archive_root = tmp_path / "drive" / "archive-a"
    _write(drive_archive_root / "existing.txt", "keep\n")

    code = bootstrap.main(
        [
            "--root",
            str(root),
            "--archive-id",
            "archive-a",
            "--drive-root",
            str(tmp_path / "drive"),
            "--include-features",
            "--include-artifacts",
            "--include-configs",
            "--copy-policy",
            "fail_if_exists",
        ]
    )

    captured = capsys.readouterr()

    assert code == 2
    assert "fail_if_exists" in captured.err


def test_skip_existing_preserves_destination_files(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    drive_archive_root = tmp_path / "drive" / "archive-a"
    _write(drive_archive_root / "manifest.json", "preserve\n")

    code = bootstrap.main(
        [
            "--root",
            str(root),
            "--archive-id",
            "archive-a",
            "--drive-root",
            str(tmp_path / "drive"),
            "--include-features",
            "--include-artifacts",
            "--include-configs",
            "--copy-policy",
            "skip_existing",
        ]
    )

    assert code == 0
    assert (drive_archive_root / "manifest.json").read_text(encoding="utf-8") == "preserve\n"
    assert (drive_archive_root / "checksums.json").is_file()


def test_overwrite_allowed_replaces_destination_contents(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    drive_archive_root = tmp_path / "drive" / "archive-a"
    _write(drive_archive_root / "stale.json", "stale\n")

    code = bootstrap.main(
        [
            "--root",
            str(root),
            "--archive-id",
            "archive-a",
            "--drive-root",
            str(tmp_path / "drive"),
            "--include-features",
            "--include-artifacts",
            "--include-configs",
            "--copy-policy",
            "overwrite_allowed",
        ]
    )

    assert code == 0
    assert not (drive_archive_root / "stale.json").exists()
    assert (drive_archive_root / "manifest.json").is_file()


def test_validate_and_inspect_after_copy_use_copied_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path / "repo")
    drive = tmp_path / "drive"
    called: dict[str, str] = {}

    def _validate(archive_root: str | Path, *, verify_checksums: bool = True) -> object:
        called["validate"] = Path(archive_root).resolve().as_posix()
        return SimpleNamespace(status="passed", issues=(), passed=True)

    def _inspect(archive_root: str | Path, *, verify_checksums: bool = True) -> object:
        called["inspect"] = Path(archive_root).resolve().as_posix()
        return SimpleNamespace(status="passed", issues=())

    monkeypatch.setattr(bootstrap, "validate_session_archive", _validate)
    monkeypatch.setattr(bootstrap, "inspect_session_archive", _inspect)

    code = bootstrap.main(
        [
            "--root",
            str(root),
            "--archive-id",
            "archive-a",
            "--drive-root",
            str(drive),
            "--include-features",
            "--include-artifacts",
            "--include-configs",
            "--validate-after-copy",
            "--inspect-after-copy",
        ]
    )

    expected = (drive / "archive-a").resolve().as_posix()

    assert code == 0
    assert called["validate"] == expected
    assert called["inspect"] == expected


def test_validation_failure_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path / "repo")

    issue = SimpleNamespace(severity="error", code="checksum_mismatch", message="bad")

    def _validate(archive_root: str | Path, *, verify_checksums: bool = True) -> object:
        return SimpleNamespace(status="failed", issues=(issue,), passed=False)

    monkeypatch.setattr(bootstrap, "validate_session_archive", _validate)

    code = bootstrap.main(
        [
            "--root",
            str(root),
            "--archive-id",
            "archive-a",
            "--include-features",
            "--include-artifacts",
            "--include-configs",
            "--validate-after-copy",
        ]
    )

    assert code == 1


def test_json_output_is_deterministic_for_dry_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path / "repo")
    argv = [
        "--root",
        str(root),
        "--archive-id",
        "archive-a",
        "--include-features",
        "--include-artifacts",
        "--include-configs",
        "--dry-run",
        "--json",
    ]

    assert bootstrap.main(argv) == 0
    first = capsys.readouterr().out
    assert bootstrap.main(argv) == 0
    second = capsys.readouterr().out

    assert first == second


def test_bootstrap_report_is_deterministic(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")

    assert (
        bootstrap.main(
            [
                "--root",
                str(root),
                "--archive-id",
                "archive-a",
                "--include-features",
                "--include-artifacts",
                "--include-configs",
            ]
        )
        == 0
    )
    report_path = root / "artifacts/_derived/session_archives/archive-a/bootstrap_report.json"
    first = report_path.read_text(encoding="utf-8")

    payload = json.loads(first)
    rewritten = bootstrap._write_bootstrap_report(root, "archive-a", payload)
    second = rewritten.read_text(encoding="utf-8")

    assert payload["boundaries"]["derived"] is True
    assert payload["boundaries"]["authoritative"] is False
    assert first == second


def test_cli_delegates_to_existing_archive_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"build": 0, "write": 0}

    def _build(request: object) -> object:
        calls["build"] += 1
        entry = SimpleNamespace(logical_group=SimpleNamespace(value="configs"), size_bytes=1)
        return SimpleNamespace(
            manifest=SimpleNamespace(archive_id="archive-a"),
            archive_root=Path("repo") / "artifacts/_derived/session_archives/archive-a",
            entries=(entry,),
            shards=(SimpleNamespace(),),
        )

    def _write(request: object) -> object:
        calls["write"] += 1
        entry = SimpleNamespace(logical_group=SimpleNamespace(value="configs"), size_bytes=1)
        plan = SimpleNamespace(
            manifest=SimpleNamespace(archive_id="archive-a"),
            entries=(entry,),
            shards=(SimpleNamespace(),),
        )
        return SimpleNamespace(
            archive_root=Path("repo") / "artifacts/_derived/session_archives/archive-a",
            plan=plan,
        )

    monkeypatch.setattr(bootstrap, "build_session_archive_plan", _build)
    monkeypatch.setattr(bootstrap, "write_session_archive_pack", _write)
    monkeypatch.setattr(bootstrap, "_write_bootstrap_report", lambda *_args, **_kwargs: Path("report.json"))
    monkeypatch.setattr(bootstrap, "_emit", lambda *_args, **_kwargs: None)

    bootstrap.run_cli(["--root", ".", "--archive-id", "archive-a", "--dry-run"])
    bootstrap.run_cli(["--root", ".", "--archive-id", "archive-a"])

    assert calls["build"] == 1
    assert calls["write"] == 1


def test_module_has_no_google_or_oauth_dependencies() -> None:
    text = Path(bootstrap.__file__).read_text(encoding="utf-8")
    lowered = text.lower()

    assert "google." not in lowered
    assert "oauth" not in lowered
