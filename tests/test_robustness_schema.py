from __future__ import annotations

import csv
import json
from pathlib import Path

from src.research.robustness import (
    FINDINGS_FILENAME,
    MANIFEST_FILENAME,
    MULTIPLE_TESTING_FILENAME,
    REPORT_FILENAME,
    SAMPLE_SIZE_FILENAME,
    SENSITIVITY_FILENAME,
    SUMMARY_FILENAME,
    WALK_FORWARD_EFFICIENCY_COLUMNS,
    WALK_FORWARD_EFFICIENCY_FILENAME,
    ArtifactReference,
    MultipleTestingSummary,
    RobustnessFinding,
    RobustnessReport,
    SampleSizeValidation,
    SensitivitySummaryRow,
    UpstreamReferences,
    WalkForwardEfficiencyRow,
    build_robustness_summary,
    write_robustness_report_bundle,
)

EXPECTED_ARTIFACTS = [
    SUMMARY_FILENAME,
    FINDINGS_FILENAME,
    WALK_FORWARD_EFFICIENCY_FILENAME,
    SAMPLE_SIZE_FILENAME,
    SENSITIVITY_FILENAME,
    MULTIPLE_TESTING_FILENAME,
    REPORT_FILENAME,
    MANIFEST_FILENAME,
]


def _sample_report(repo_root: Path) -> RobustnessReport:
    return RobustnessReport(
        report_id="m34_contract_smoke",
        workflow_type="strategy",
        run_id="run_001",
        source_run_id="source_001",
        robustness_status="needs_review",
        upstream_references=UpstreamReferences(
            strategy=(
                ArtifactReference(
                    repo_root / "artifacts" / "strategies" / "run_001" / "summary.json",
                    artifact_type="strategy_summary",
                    metadata={"local_path": repo_root / "artifacts" / "strategies" / "run_001"},
                ),
            ),
            governance=(
                ArtifactReference(
                    "artifacts/promotion_governance/report_a/manifest.json",
                    artifact_type="promotion_governance_manifest",
                ),
            ),
        ),
        findings=(
            RobustnessFinding(
                check_id="sample_size.min_trades",
                severity="needs-review",
                workflow_type="strategy",
                run_id="run_001",
                message="Trade count is below the future M34 review threshold.",
                details={"evidence_path": repo_root / "artifacts" / "strategies" / "run_001" / "trades.csv"},
            ),
            RobustnessFinding(
                check_id="contract.placeholder",
                severity="info",
                workflow_type="alpha",
                run_id="alpha_001",
                message="Placeholder contract row emitted without running statistical diagnostics.",
                details={},
            ),
        ),
        walk_forward_efficiency=(
            WalkForwardEfficiencyRow(
                workflow_type="strategy",
                run_id="run_001",
                split_id="split_001",
                train_period="2025-01-01/2025-03-31",
                test_period="2025-04-01/2025-04-30",
                status="not_evaluated",
                details={"source": repo_root / "artifacts" / "wf.csv"},
            ),
        ),
        sample_size_validation=(
            SampleSizeValidation(
                workflow_type="strategy",
                run_id="run_001",
                check_id="sample_size.min_trades",
                sample_count=120,
                trade_count=8,
                minimum_sample_count=252,
                minimum_trade_count=30,
                status="needs_review",
            ),
        ),
        sensitivity_summary=(
            SensitivitySummaryRow(
                workflow_type="strategy",
                run_id="run_001",
                scenario_id="lookback_plus_5",
                parameter="lookback",
                baseline_value=20,
                scenario_value=25,
                metric="sharpe_ratio",
                status="not_evaluated",
            ),
        ),
        multiple_testing_summary=(
            MultipleTestingSummary(
                workflow_type="campaign",
                run_id="campaign_001",
                family_id="candidate_sweep",
                trial_count=12,
                effective_trial_count=12,
                adjustment_method="not_evaluated",
                status="not_evaluated",
            ),
        ),
        checks_present=("sample_size.min_trades", "contract.placeholder"),
        checks_missing=("walk_forward_efficiency", "multiple_testing.adjustment"),
        created_at_utc="2026-01-01T00:00:00Z",
        metadata={"config_path": repo_root / "configs" / "robustness.yml"},
    )


def test_finding_serialization_normalizes_severity_and_paths(tmp_path: Path) -> None:
    finding = RobustnessFinding(
        check_id="wfe.placeholder",
        severity="needs review",
        workflow_type="strategy",
        run_id="run_a",
        message="WFE placeholder emitted.",
        details={"absolute": tmp_path / "artifacts" / "run_a" / "wfe.csv"},
    )

    payload = finding.to_dict(roots=(tmp_path,))

    assert payload["severity"] == "needs_review"
    assert payload["details"]["absolute"] == "artifacts/run_a/wfe.csv"


def test_summary_serialization_is_deterministic() -> None:
    report = RobustnessReport(
        report_id="summary_contract",
        workflow_type="strategy",
        run_id="run_a",
        findings=(
            RobustnessFinding("z_check", "warning", "strategy", "run_a", "Warning.", {}),
            RobustnessFinding("a_check", "reject", "portfolio", "run_b", "Reject.", {}),
        ),
        checks_present=("z_check", "a_check"),
        checks_missing=("future_check",),
    )

    first = build_robustness_summary(report, generated_artifacts=["b.csv", "a.json"])
    second = build_robustness_summary(report, generated_artifacts=["a.json", "b.csv"])

    assert first == second
    assert json.dumps(first, sort_keys=True).index("artifact_count") < json.dumps(first, sort_keys=True).index("checks_missing")
    assert first["finding_count_by_severity"]["reject"] == 1
    assert first["highest_severity"] == "reject"


def test_writer_generates_complete_deterministic_portable_bundle(tmp_path: Path) -> None:
    report = _sample_report(tmp_path)

    first = write_robustness_report_bundle(report, output_root=tmp_path / "artifacts" / "robustness")
    first_snapshot = {
        path.relative_to(first.output_dir).as_posix(): path.read_bytes()
        for path in sorted(first.output_dir.iterdir())
        if path.is_file()
    }
    second = write_robustness_report_bundle(report, output_root=tmp_path / "artifacts" / "robustness")
    second_snapshot = {
        path.relative_to(second.output_dir).as_posix(): path.read_bytes()
        for path in sorted(second.output_dir.iterdir())
        if path.is_file()
    }

    assert sorted(first_snapshot) == sorted(EXPECTED_ARTIFACTS)
    assert first_snapshot == second_snapshot

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_files"] == sorted(EXPECTED_ARTIFACTS)
    assert sorted(manifest["artifacts"]) == sorted(EXPECTED_ARTIFACTS)
    assert manifest["output_dir"] == "artifacts/robustness/m34_contract_smoke"
    assert manifest["source_run_references"] == ["alpha_001", "campaign_001", "run_001", "source_001"]
    assert not _contains_absolute_path(manifest)

    findings = json.loads(first.findings_path.read_text(encoding="utf-8"))
    assert findings["finding_count"] == 2
    assert findings["findings"][1]["severity"] == "needs_review"
    assert not _contains_absolute_path(findings)

    with first.walk_forward_efficiency_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        assert next(reader) == WALK_FORWARD_EFFICIENCY_COLUMNS

    summary = json.loads(first.summary_path.read_text(encoding="utf-8"))
    assert summary["artifact_count"] == len(EXPECTED_ARTIFACTS)
    assert summary["checks_missing"] == ["multiple_testing.adjustment", "walk_forward_efficiency"]


def test_empty_optional_references_degrade_gracefully(tmp_path: Path) -> None:
    report = RobustnessReport(report_id="minimal", workflow_type="campaign", run_id="campaign_001")

    result = write_robustness_report_bundle(report, output_root=tmp_path / "artifacts" / "robustness")

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    sample_size = json.loads(result.sample_size_validation_path.read_text(encoding="utf-8"))
    multiple_testing = json.loads(result.multiple_testing_summary_path.read_text(encoding="utf-8"))

    assert manifest["source_artifacts"] == []
    assert sample_size["checks"] == []
    assert multiple_testing["families"] == []
    assert result.walk_forward_efficiency_path.read_text(encoding="utf-8").splitlines()[0] == ",".join(WALK_FORWARD_EFFICIENCY_COLUMNS)


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        return "C:/Users/" in normalized or normalized.startswith("file://") or normalized.startswith("/Users/") or normalized.startswith("/home/")
    return False
