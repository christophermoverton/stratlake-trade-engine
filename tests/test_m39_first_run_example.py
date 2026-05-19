from __future__ import annotations

import json
from pathlib import Path

import yaml

from docs.examples.m39_first_run_configuration_profile_example import (
    REPORT_FILENAMES,
    main,
    run_m39_first_run_configuration_profile_example,
)
from src.config.profiles import load_runtime_profile


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPO_ROOT / "configs" / "profiles"


def test_m39_first_run_example_runs_in_ci_without_external_dependencies(tmp_path: Path) -> None:
    output_root = tmp_path / "m39_first_run_configuration_profile_example"

    summary = run_m39_first_run_configuration_profile_example(output_root=output_root)

    assert summary["status"] == "passed"
    assert summary["profile"] == "ci"
    assert summary["workflow"] == "strategy"
    assert summary["authoritative"] is False
    assert summary["safety"]["workflows_executed"] is False
    assert summary["safety"]["canonical_artifacts_mutated"] is False
    assert summary["safety"]["requires_network"] is False
    assert summary["safety"]["requires_credentials"] is False
    assert summary["safety"]["requires_live_market_data"] is False
    assert summary["artifact_boundaries"]["direct_scan"] is True
    assert summary["artifact_boundaries"]["derived_outputs_authoritative"] is False
    assert summary["artifact_boundaries"]["mutates_canonical_artifacts"] is False

    for filename in REPORT_FILENAMES.values():
        assert (output_root / filename).exists()


def test_m39_first_run_example_outputs_are_deterministic_and_portable(tmp_path: Path) -> None:
    output_root = tmp_path / "m39_first_run_configuration_profile_example"

    first = run_m39_first_run_configuration_profile_example(output_root=output_root)
    first_files = _read_generated_files(output_root)
    second = run_m39_first_run_configuration_profile_example(
        output_root=output_root,
        reset_output=False,
    )
    second_files = _read_generated_files(output_root)

    assert first == second
    assert first_files == second_files
    serialized = json.dumps(second_files, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "file://" not in serialized
    assert "C:/" not in serialized
    assert "\\\\" not in serialized


def test_m39_first_run_reports_remain_advisory_and_non_authoritative(tmp_path: Path) -> None:
    output_root = tmp_path / "m39_first_run_configuration_profile_example"
    run_m39_first_run_configuration_profile_example(output_root=output_root)

    validation = _read_json(output_root / REPORT_FILENAMES["validation"])
    doctor = _read_json(output_root / REPORT_FILENAMES["doctor"])
    explain = _read_json(output_root / REPORT_FILENAMES["explain"])
    synthetic_probe = _read_json(output_root / REPORT_FILENAMES["synthetic_probe"])

    assert validation["status"] == "passed"
    assert validation["authoritative"] is False
    assert validation["resolved_config"]["boundaries"]["requires_live_market_data"] is False
    assert doctor["status"] == "passed"
    assert doctor["authoritative"] is False
    assert doctor["workflows_executed"] is False
    assert doctor["finding_counts"]["fail"] == 0
    assert explain["status"] == "passed"
    assert explain["authoritative"] is False
    assert explain["safety"]["workflows_executed"] is False
    assert explain["path_summary"]["expected_artifact_roots"]["evidence_review"] == (
        "artifacts/_derived/evidence_review"
    )
    assert synthetic_probe["status"] == "passed"
    assert synthetic_probe["authoritative"] is False
    assert synthetic_probe["safety"]["workflows_executed"] is False


def test_m39_first_run_cli_entrypoint_writes_requested_output(
    tmp_path: Path,
    capsys,
) -> None:
    output_root = tmp_path / "m39_first_run_configuration_profile_example"

    exit_code = main(["--output-root", str(output_root)])
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["status"] == "passed"
    assert _read_json(output_root / REPORT_FILENAMES["summary"]) == summary


def test_m39_first_run_does_not_execute_workflows_or_mutate_profiles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    before = _snapshot_profile_files()

    def fail_workflow(*args, **kwargs):
        raise AssertionError("M39 first-run example must not execute workflows")

    monkeypatch.setattr("src.execution.run_strategy", fail_workflow, raising=False)
    monkeypatch.setattr("src.execution.run_alpha", fail_workflow, raising=False)
    monkeypatch.setattr("src.execution.run_portfolio", fail_workflow, raising=False)
    monkeypatch.setattr("src.execution.run_pipeline", fail_workflow, raising=False)
    monkeypatch.setattr("src.execution.run_research_campaign", fail_workflow, raising=False)

    output_root = tmp_path / "m39_first_run_configuration_profile_example"
    summary = run_m39_first_run_configuration_profile_example(output_root=output_root)

    after = _snapshot_profile_files()
    assert summary["status"] == "passed"
    assert after == before


def test_m39_starter_profiles_are_ci_safe_and_portable() -> None:
    for profile_path in sorted(PROFILE_DIR.glob("*.yml")):
        profile = load_runtime_profile(str(profile_path))
        payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        serialized = json.dumps(payload, sort_keys=True)

        assert profile.boundaries["direct_scan"] is True
        assert profile.boundaries["derived_outputs_authoritative"] is False
        assert profile.boundaries["mutates_canonical_artifacts"] is False
        assert profile.boundaries["requires_network"] is False
        assert profile.boundaries["requires_credentials"] is False
        assert profile.boundaries["requires_live_market_data"] is False
        assert "file://" not in serialized
        assert "C:/" not in serialized
        assert "\\\\" not in serialized


def _read_generated_files(output_root: Path) -> dict[str, dict[str, object]]:
    return {
        key: _read_json(output_root / filename)
        for key, filename in sorted(REPORT_FILENAMES.items())
    }


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot_profile_files() -> dict[str, str]:
    return {
        path.relative_to(REPO_ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(PROFILE_DIR.glob("*.yml"))
    }
