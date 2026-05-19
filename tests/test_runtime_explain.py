from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.cli.explain_config import main, run_cli
from src.config.explain import (
    SUPPORTED_EXPLAIN_WORKFLOWS,
    build_runtime_explain_report,
    write_runtime_explain_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPO_ROOT / "configs" / "profiles"


def test_runtime_explain_api_succeeds_for_ci_profile() -> None:
    report = build_runtime_explain_report("ci", workflow="strategy")
    payload = report.to_json_dict()

    assert payload["status"] == "passed"
    assert payload["run_type"] == "runtime_explain"
    assert payload["authoritative"] is False
    assert payload["workflow"] == "strategy"
    assert payload["profile"] == {
        "name": "ci",
        "path": "configs/profiles/ci.yml",
        "source": "profile_name",
    }
    assert payload["artifact_boundaries"]["direct_scan"] is True
    assert payload["artifact_boundaries"]["derived_outputs_authoritative"] is False
    assert payload["safety"]["workflows_executed"] is False
    assert payload["safety"]["canonical_artifacts_mutated"] is False
    assert payload["safety"]["requires_network"] is False
    assert payload["safety"]["requires_credentials"] is False
    assert payload["safety"]["requires_live_market_data"] is False
    assert payload["path_summary"]["artifacts_root"] == "artifacts/ci"
    assert payload["path_summary"]["expected_artifact_roots"]["strategy"] == "artifacts/ci/strategies"


def test_runtime_explain_cli_succeeds_for_ci_profile(capsys: pytest.CaptureFixture[str]) -> None:
    report = run_cli(["--profile", "ci", "--workflow", "strategy"])
    captured = capsys.readouterr()

    assert report["status"] == "passed"
    assert report["workflow"] == "strategy"
    assert json.loads(captured.out) == report
    assert "runtime_explain_status: passed" in captured.err


def test_runtime_explain_supports_explicit_profile_path(tmp_path: Path) -> None:
    profile_path = tmp_path / "custom.yml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profile": "local",
                "settings": {"artifacts_root": "artifacts/custom"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    payload = build_runtime_explain_report(
        profile_path=profile_path,
        workflow="portfolio",
    ).to_json_dict()

    assert payload["status"] == "passed"
    assert payload["profile"] == {
        "name": "local",
        "path": "<external>/custom.yml",
        "source": "profile_path",
    }
    assert payload["resolved_config"]["settings"]["artifacts_root"] == "artifacts/custom"
    assert str(tmp_path) not in json.dumps(payload, sort_keys=True)


def test_runtime_explain_output_is_deterministic() -> None:
    first = build_runtime_explain_report("ci", workflow="pipeline").to_json()
    second = build_runtime_explain_report("ci", workflow="pipeline").to_json()

    assert first == second
    assert json.loads(first) == build_runtime_explain_report("ci", workflow="pipeline").to_json_dict()


def test_runtime_explain_provenance_summary_matches_full_provenance() -> None:
    payload = build_runtime_explain_report("ci", workflow="generic").to_json_dict()
    provenance = payload["provenance"]
    source_counts = payload["provenance_summary"]["source_counts"]

    assert payload["provenance_summary"]["field_count"] == len(provenance)
    for source, count in source_counts.items():
        assert count == sum(1 for entry in provenance.values() if entry["source"] == source)
    assert payload["provenance_summary"]["highest_precedence_source"] == "profile"


def test_runtime_explain_includes_workflow_assumptions_for_each_subject() -> None:
    expected = {
        "generic": "no_workflow_selected",
        "strategy": "strategy_not_run",
        "alpha": "alpha_not_run",
        "portfolio": "portfolio_not_run",
        "pipeline": "pipeline_not_run",
        "campaign": "campaign_not_run",
        "evidence_review": "evidence_review_not_run",
    }
    assert set(expected) == SUPPORTED_EXPLAIN_WORKFLOWS

    for workflow, assumption_name in expected.items():
        payload = build_runtime_explain_report("ci", workflow=workflow).to_json_dict()
        assumption_names = {item["name"] for item in payload["workflow_assumptions"]}

        assert "configuration_only" in assumption_names
        assert "no_execution" in assumption_names
        assert assumption_name in assumption_names
        assert payload["workflow"] == workflow


def test_runtime_explain_invalid_profile_reports_failure_and_cli_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "bad.yml"
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

    payload = build_runtime_explain_report(profile_path=profile_path).to_json_dict()
    assert payload["status"] == "failed"
    assert payload["findings"][0]["severity"] == "error"
    assert "repository-relative" in payload["findings"][0]["message"]
    assert str(tmp_path) not in json.dumps(payload, sort_keys=True)

    with pytest.raises(SystemExit) as exc_info:
        run_cli(["--profile-path", str(profile_path)])
    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert json.loads(captured.out)["status"] == "failed"


def test_runtime_explain_writes_output_only_when_requested(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "artifacts" / "_derived" / "config_explain" / "ci_strategy.json"

    run_cli(["--profile", "ci", "--workflow", "strategy"])
    capsys.readouterr()
    assert not output_path.exists()

    report = run_cli(
        [
            "--profile",
            "ci",
            "--workflow",
            "strategy",
            "--output",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert captured.out == ""
    assert written == report
    assert written["authoritative"] is False
    assert written["safety"]["workflows_executed"] is False
    assert written["output_path"] == "<external>/ci_strategy.json"
    assert output_path.read_text(encoding="utf-8") == json.dumps(written, indent=2, sort_keys=True) + "\n"


def test_runtime_explain_report_writer_is_deterministic_and_advisory(tmp_path: Path) -> None:
    output_path = tmp_path / "explain.json"
    report = build_runtime_explain_report("ci", workflow="campaign", output_path=output_path)

    written_path = write_runtime_explain_report(report, output_path)
    first = written_path.read_text(encoding="utf-8")
    write_runtime_explain_report(report, output_path)
    second = written_path.read_text(encoding="utf-8")
    payload = json.loads(first)

    assert first == second
    assert payload["authoritative"] is False
    assert payload["safety"]["canonical_artifacts_mutated"] is False
    assert str(tmp_path) not in first


def test_runtime_explain_does_not_execute_workflows_or_mutate_canonical_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _snapshot_canonical_files()

    def fail_workflow(*args, **kwargs):
        raise AssertionError("runtime explain must not execute workflows")

    monkeypatch.setattr("src.execution.run_strategy", fail_workflow, raising=False)
    monkeypatch.setattr("src.execution.run_alpha", fail_workflow, raising=False)
    monkeypatch.setattr("src.execution.run_portfolio", fail_workflow, raising=False)
    monkeypatch.setattr("src.execution.run_pipeline", fail_workflow, raising=False)
    monkeypatch.setattr("src.execution.run_research_campaign", fail_workflow, raising=False)
    exit_code = main(["--profile", "ci", "--workflow", "strategy"])

    after = _snapshot_canonical_files()
    assert exit_code == 0
    assert after == before


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
