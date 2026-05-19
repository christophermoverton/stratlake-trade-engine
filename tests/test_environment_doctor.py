from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.cli.stratlake_doctor import main, run_cli
from src.config.doctor import run_environment_doctor, write_environment_doctor_report


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPO_ROOT / "configs" / "profiles"


def test_environment_doctor_api_passes_for_ci_profile() -> None:
    report = run_environment_doctor("ci")
    payload = report.to_json_dict()

    assert payload["status"] == "passed"
    assert payload["run_type"] == "environment_doctor"
    assert payload["authoritative"] is False
    assert payload["profile"] == {"name": "ci", "path": "configs/profiles/ci.yml"}
    assert payload["workflows_executed"] is False
    assert payload["artifact_boundaries"]["direct_scan"] is True
    assert payload["artifact_boundaries"]["derived_outputs_authoritative"] is False
    assert payload["artifact_boundaries"]["mutates_canonical_artifacts"] is False
    assert payload["resolved_config"]["boundaries"]["requires_network"] is False
    assert payload["resolved_config"]["boundaries"]["requires_credentials"] is False
    assert payload["resolved_config"]["boundaries"]["requires_live_market_data"] is False
    assert payload["finding_counts"]["fail"] == 0
    assert _check(payload, "profile_resolves")["status"] == "pass"


def test_environment_doctor_cli_passes_for_ci_profile(capsys: pytest.CaptureFixture[str]) -> None:
    report = run_cli(["--profile", "ci"])
    captured = capsys.readouterr()

    assert report["status"] == "passed"
    assert report["finding_counts"]["fail"] == 0
    assert json.loads(captured.out) == report
    assert "environment_doctor_status: passed" in captured.err


def test_environment_doctor_works_for_all_starter_profiles() -> None:
    for profile in ("local", "notebook", "pipeline"):
        payload = run_environment_doctor(profile).to_json_dict()

        assert payload["status"] == "passed"
        assert payload["profile"]["name"] == profile
        assert payload["finding_counts"]["fail"] == 0
        assert payload["resolved_config"]["boundaries"]["requires_network"] is False
        assert payload["resolved_config"]["boundaries"]["requires_credentials"] is False
        assert payload["resolved_config"]["boundaries"]["requires_live_market_data"] is False


def test_environment_doctor_invalid_profile_produces_failed_finding_and_nonzero_exit(
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
    payload = json.loads(captured.out)

    assert exc_info.value.code == 1
    assert payload["status"] == "failed"
    assert payload["finding_counts"]["fail"] == 1
    assert _check(payload, "profile_resolves")["status"] == "fail"
    assert "repository-relative" in _check(payload, "profile_resolves")["message"]
    assert payload["profile"]["path"] == "<external>/invalid.yml"
    assert str(tmp_path) not in captured.out


def test_environment_doctor_missing_data_roots_are_skipped_not_failed(tmp_path: Path) -> None:
    profile_path = tmp_path / "missing_roots.yml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profile": "local",
                "settings": {
                    "marketlake_root": "data/not_present_curated",
                    "features_root": "data/not_present_features",
                    "artifacts_root": "artifacts/not_present",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    payload = run_environment_doctor(profile_path=profile_path).to_json_dict()

    assert payload["status"] == "passed"
    assert _check(payload, "optional_root:features_root")["status"] == "skipped"
    assert _check(payload, "optional_root:marketlake_root")["status"] == "skipped"
    assert payload["finding_counts"]["fail"] == 0


def test_environment_doctor_output_is_deterministic() -> None:
    first = run_environment_doctor("ci").to_json()
    second = run_environment_doctor("ci").to_json()

    assert first == second
    assert json.loads(first) == run_environment_doctor("ci").to_json_dict()


def test_environment_doctor_writes_output_only_when_requested(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "artifacts" / "_derived" / "environment_readiness" / "ci_doctor.json"

    run_cli(["--profile", "ci"])
    capsys.readouterr()
    assert not output_path.exists()

    report = run_cli(["--profile", "ci", "--output", str(output_path)])
    captured = capsys.readouterr()
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert captured.out == ""
    assert output_path.exists()
    assert written == report
    assert written["authoritative"] is False
    assert written["status"] == "passed"
    assert written["output_path"] == "<external>/ci_doctor.json"
    assert output_path.read_text(encoding="utf-8") == json.dumps(written, indent=2, sort_keys=True) + "\n"


def test_environment_doctor_external_paths_are_sanitized(tmp_path: Path) -> None:
    profile_path = tmp_path / "external_profile.yml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profile": "ci",
                "settings": {"artifacts_root": "artifacts/external"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    payload = run_environment_doctor(profile_path=profile_path).to_json_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["profile"] == {"name": "ci", "path": "<external>/external_profile.yml"}
    assert str(tmp_path) not in serialized


def test_environment_doctor_warns_for_output_outside_derived() -> None:
    payload = run_environment_doctor("ci", output_path="artifacts/doctor.json").to_json_dict()

    check = _check(payload, "output_path_recommendation")
    assert check["status"] == "warning"
    assert "artifacts/_derived/environment_readiness" in check["message"]


def test_environment_doctor_does_not_execute_workflows_or_mutate_canonical_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _snapshot_canonical_files()

    def fail_workflow(*args, **kwargs):
        raise AssertionError("environment doctor must not execute workflows")

    monkeypatch.setattr("src.execution.run_strategy", fail_workflow, raising=False)
    monkeypatch.setattr("src.execution.run_pipeline", fail_workflow, raising=False)
    monkeypatch.setattr("src.execution.run_benchmark_pack", fail_workflow, raising=False)
    exit_code = main(["--profile", "ci"])

    after = _snapshot_canonical_files()
    assert exit_code == 0
    assert after == before


def test_environment_doctor_report_writer_is_deterministic_and_advisory(tmp_path: Path) -> None:
    output_path = tmp_path / "doctor.json"
    report = run_environment_doctor("ci", output_path=output_path)

    written_path = write_environment_doctor_report(report, output_path)
    first = written_path.read_text(encoding="utf-8")
    write_environment_doctor_report(report, output_path)
    second = written_path.read_text(encoding="utf-8")
    payload = json.loads(first)

    assert first == second
    assert payload["authoritative"] is False
    assert payload["workflows_executed"] is False
    assert str(tmp_path) not in first


def _check(payload: dict[str, object], name: str) -> dict[str, object]:
    checks = payload["checks"]
    assert isinstance(checks, list)
    matches = [check for check in checks if check["name"] == name]
    assert len(matches) == 1
    return matches[0]


def _snapshot_canonical_files() -> dict[str, str]:
    paths = [
        *sorted(PROFILE_DIR.glob("*.yml")),
        REPO_ROOT / "configs" / "execution.yml",
        REPO_ROOT / "configs" / "sanity.yml",
        REPO_ROOT / "configs" / "review.yml",
    ]
    return {
        path.relative_to(REPO_ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in paths
    }
