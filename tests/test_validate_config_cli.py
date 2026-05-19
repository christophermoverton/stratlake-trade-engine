from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.cli.validate_config import main, run_cli, validate_config_from_args, parse_args
from src.config.profiles import load_runtime_profile


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPO_ROOT / "configs" / "profiles"


def test_validate_config_cli_succeeds_for_valid_starter_profiles(capsys: pytest.CaptureFixture[str]) -> None:
    for profile in ("ci", "local", "notebook", "pipeline"):
        report = run_cli(["--profile", profile])
        captured = capsys.readouterr()

        assert report["status"] == "passed"
        assert report["validated"] is True
        assert report["authoritative"] is False
        assert report["profile"]["name"] == profile
        assert report["profile"]["path"] == f"configs/profiles/{profile}.yml"
        assert report["findings"] == []
        assert "resolved_config" in report
        assert "provenance" in report
        assert "config_validation_status: passed" in captured.err
        assert json.loads(captured.out)["status"] == "passed"


def test_validate_config_cli_supports_default_no_profile(capsys: pytest.CaptureFixture[str]) -> None:
    report = run_cli([])
    captured = capsys.readouterr()

    assert report["status"] == "passed"
    assert report["profile"] == {"name": None, "path": None}
    assert report["resolved_config"]["settings"]["artifacts_root"] == "artifacts"
    assert json.loads(captured.out)["profile"] == {"name": None, "path": None}


def test_validate_config_cli_supports_explicit_profile_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "custom.yml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profile": "ci",
                "settings": {"artifacts_root": "artifacts/custom"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = run_cli(["--profile-path", str(profile_path)])
    captured = capsys.readouterr()
    stdout_report = json.loads(captured.out)

    assert report["status"] == "passed"
    assert report["profile"] == {"name": "ci", "path": f"<external>/{profile_path.name}"}
    assert report["resolved_config"]["settings"]["artifacts_root"] == "artifacts/custom"
    assert stdout_report == report
    assert str(tmp_path) not in captured.out


def test_validate_config_cli_fails_on_invalid_profile_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "invalid.yml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profile": "local",
                "settings": {"artifacts_root": "C:/Users/example/artifacts"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        run_cli(["--profile-path", str(profile_path)])
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert exc_info.value.code == 1
    assert report["status"] == "failed"
    assert report["validated"] is False
    assert report["authoritative"] is False
    assert report["profile"]["path"] == f"<external>/{profile_path.name}"
    assert report["findings"][0]["severity"] == "error"
    assert "repository-relative" in report["findings"][0]["message"]
    assert str(tmp_path) not in captured.out


def test_validate_config_cli_detects_unknown_keys(tmp_path: Path) -> None:
    profile_path = tmp_path / "unknown.yml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profile": "local",
                "secret_token": "not-allowed",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    args = parse_args(["--profile-path", str(profile_path)])
    report = validate_config_from_args(args)

    assert report["status"] == "failed"
    assert "unsupported keys" in report["findings"][0]["message"]
    assert "secret_token" in report["findings"][0]["message"]


def test_validate_config_cli_sanitizes_missing_external_profile_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yml"

    report = validate_config_from_args(parse_args(["--profile-path", str(missing_path)]))

    assert report["status"] == "failed"
    assert report["profile"]["path"] == "<external>/missing.yml"
    assert str(tmp_path) not in json.dumps(report, sort_keys=True)
    assert "<external>/missing.yml" in report["findings"][0]["message"]


def test_validate_config_cli_detects_non_portable_path_forms(tmp_path: Path) -> None:
    invalid_values = [
        "file://artifacts/report.json",
        "../artifacts",
        "artifacts\\validation",
    ]
    for index, invalid_path in enumerate(invalid_values):
        profile_path = tmp_path / f"invalid_path_{index}.yml"
        profile_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "profile": "local",
                    "settings": {"artifacts_root": invalid_path},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        report = validate_config_from_args(parse_args(["--profile-path", str(profile_path)]))

        assert report["status"] == "failed"
        assert report["findings"][0]["severity"] == "error"
        assert any(
            fragment in report["findings"][0]["message"]
            for fragment in ("URI", "normalized", "POSIX-style")
        )


def test_validate_config_cli_output_is_deterministic(capsys: pytest.CaptureFixture[str]) -> None:
    first = run_cli(["--profile", "ci"])
    first_output = capsys.readouterr().out
    second = run_cli(["--profile", "ci"])
    second_output = capsys.readouterr().out

    assert first == second
    assert first_output == second_output
    assert json.loads(first_output) == first


def test_validate_config_cli_api_parity_for_profile_failure(tmp_path: Path) -> None:
    profile_path = tmp_path / "bad.yml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profile": "local",
                "workflow_configs": {"unknown_config": "configs/missing.yml"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = validate_config_from_args(parse_args(["--profile-path", str(profile_path)]))

    with pytest.raises(Exception) as exc_info:
        load_runtime_profile(str(profile_path))
    assert report["status"] == "failed"
    assert report["findings"][0]["message"] == str(exc_info.value)


def test_validate_config_cli_writes_output_only_when_requested(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "artifacts" / "_derived" / "config_validation" / "ci_validation.json"

    run_cli(["--profile", "ci"])
    capsys.readouterr()
    assert not output_path.exists()

    report = run_cli(["--profile", "ci", "--output", str(output_path)])
    captured = capsys.readouterr()
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.exists()
    assert written == report
    assert written["authoritative"] is False
    assert written["status"] == "passed"
    assert captured.out == ""
    assert output_path.read_text(encoding="utf-8") == json.dumps(written, indent=2, sort_keys=True) + "\n"


def test_validate_config_cli_does_not_mutate_canonical_profiles_or_execute_workflows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _snapshot_profile_files()

    def fail_workflow(*args, **kwargs):
        raise AssertionError("validate_config must not execute workflows")

    monkeypatch.setattr("src.execution.run_strategy", fail_workflow, raising=False)
    monkeypatch.setattr("src.execution.run_pipeline", fail_workflow, raising=False)
    exit_code = main(["--profile", "ci"])

    after = _snapshot_profile_files()
    assert exit_code == 0
    assert after == before


def _snapshot_profile_files() -> dict[str, str]:
    return {
        path.relative_to(REPO_ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(PROFILE_DIR.glob("*.yml"))
    }
