from __future__ import annotations

import csv
import json
from pathlib import Path

from src.research.governance import build_governance_outcome_rows, load_governance_artifacts, run_promotion_governance_report
from src.research.registry import append_registry_entry
from src.research.robustness import (
    MultipleTestingSummary,
    RobustnessFinding,
    RobustnessGovernanceContext,
    RobustnessReport,
    SampleSizeValidation,
    SensitivitySummaryRow,
    WalkForwardEfficiencyRow,
    attach_robustness_context_to_governance_rows,
    build_robustness_governance_context,
    load_robustness_governance_context,
    map_robustness_findings_to_reason_codes,
    write_robustness_report_bundle,
)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return path


def _complete_robustness_bundle(tmp_path: Path) -> Path:
    report = RobustnessReport(
        report_id="robustness_for_governance",
        workflow_type="strategy",
        run_id="strategy_001",
        robustness_status="needs_review",
        findings=(
            RobustnessFinding(
                check_id="walk_forward_efficiency.weak",
                severity="warning",
                workflow_type="strategy",
                run_id="strategy_001",
                message="Weak WFE.",
            ),
            RobustnessFinding(
                check_id="sample_size.minimum_total_trades",
                severity="needs_review",
                workflow_type="strategy",
                run_id="strategy_001",
                message="Thin trade sample.",
            ),
            RobustnessFinding(
                check_id="sensitivity.fragile",
                severity="needs_review",
                workflow_type="strategy",
                run_id="strategy_001",
                message="Fragile parameter optimum.",
            ),
            RobustnessFinding(
                check_id="multiple_testing.high_risk",
                severity="needs_review",
                workflow_type="strategy",
                run_id="strategy_001",
                message="Large search space.",
            ),
            RobustnessFinding(
                check_id="temporal_validation.embargo_violation",
                severity="blocked",
                workflow_type="strategy",
                run_id="strategy_001",
                message="Embargo violation.",
            ),
        ),
        walk_forward_efficiency=(
            WalkForwardEfficiencyRow(
                workflow_type="strategy",
                run_id="strategy_001",
                split_id="split_001",
                status="weak",
            ),
        ),
        sample_size_validation=(
            SampleSizeValidation(
                workflow_type="strategy",
                run_id="strategy_001",
                check_id="sample_size.minimum_total_trades",
                trade_count=5,
                minimum_trade_count=30,
                status="needs_review",
            ),
        ),
        sensitivity_summary=(
            SensitivitySummaryRow(
                workflow_type="strategy",
                run_id="strategy_001",
                scenario_id="lookback_plus",
                parameter="lookback",
                metric="sharpe_ratio",
                status="fragile",
            ),
        ),
        multiple_testing_summary=(
            MultipleTestingSummary(
                workflow_type="strategy",
                run_id="strategy_001",
                family_id="candidate_sweep",
                trial_count=250,
                effective_trial_count=250,
                adjustment_method="metadata_only",
                status="high_risk",
            ),
        ),
        checks_present=("walk_forward_efficiency", "sample_size", "sensitivity", "multiple_testing"),
    )
    result = write_robustness_report_bundle(report, output_root=tmp_path / "artifacts" / "robustness")
    _write_json(
        result.output_dir / "leakage_validation.json",
        {
            "schema_version": 1,
            "overall_status": "blocked",
            "split_count": 1,
            "finding_count": 1,
            "checks": [
                {
                    "check_id": "temporal_validation.embargo_violation",
                    "status": "blocked",
                    "split_id": "split_001",
                    "message": "Embargo violation.",
                    "details": {},
                }
            ],
        },
    )
    return result.summary_path


def test_governance_context_with_complete_robustness_report(tmp_path: Path) -> None:
    summary_path = _complete_robustness_bundle(tmp_path)

    context = load_robustness_governance_context(summary_path, workflow_type="strategy", run_id="strategy_001", roots=(tmp_path,))

    assert context.robustness_available is True
    assert context.robustness_status == "blocked"
    assert context.wfe_status == "weak"
    assert context.sample_size_status == "needs_review"
    assert context.sensitivity_status == "fragile"
    assert context.multiple_testing_status == "high_risk"
    assert context.temporal_validation_status == "blocked"
    assert context.robustness_finding_count == 5
    assert context.highest_robustness_severity == "blocked"
    assert context.robustness_reason_codes == (
        "fragile_parameter_optimum",
        "large_search_space_warning",
        "temporal_validation_embargo_violation",
        "thin_trade_sample",
        "weak_walk_forward_efficiency",
    )
    assert not _contains_absolute_path(context.to_dict())


def test_governance_context_with_missing_robustness_report() -> None:
    context = load_robustness_governance_context(None, workflow_type="strategy", run_id="missing")

    assert context.robustness_available is False
    assert context.robustness_status == "missing"
    assert context.robustness_missing_reason == "robustness_report_not_found"
    assert context.to_governance_fields()["robustness_available"] == "false"


def test_context_from_summary_only_and_findings_only() -> None:
    summary_context = build_robustness_governance_context(
        summary={
            "report_id": "summary_only",
            "finding_count": 0,
            "robustness_status_counts": {"needs_review": 1},
        },
        workflow_type="strategy",
        run_id="summary_only",
    )
    findings_context = build_robustness_governance_context(
        findings=[
            {
                "check_id": "multiple_testing.extreme_risk",
                "severity": "needs_review",
                "workflow_type": "strategy",
                "run_id": "findings_only",
                "message": "Extreme search space.",
                "details": {},
            }
        ],
        workflow_type="strategy",
        run_id="findings_only",
    )

    assert summary_context.robustness_status == "needs_review"
    assert summary_context.wfe_status == "unavailable"
    assert findings_context.multiple_testing_status == "extreme_risk"
    assert findings_context.robustness_reason_codes == ("extreme_search_space_warning",)


def test_reason_code_mapping_is_sorted_and_deduplicated() -> None:
    codes = map_robustness_findings_to_reason_codes(
        [
            {"check_id": "walk_forward_efficiency.broken", "severity": "needs_review"},
            {"check_id": "walk_forward_efficiency.broken", "severity": "needs_review"},
            {"check_id": "sample_size.missing_trade_count", "severity": "needs_review"},
            {"check_id": "unknown.check", "severity": "warning"},
        ]
    )

    assert codes == [
        "missing_trade_count_metadata",
        "negative_oos_transfer",
        "robustness_review_finding",
    ]


def test_each_major_finding_family_maps_to_reason_code() -> None:
    codes = map_robustness_findings_to_reason_codes(
        [
            {"check_id": "walk_forward_efficiency.weak", "severity": "warning"},
            {"check_id": "sample_size.minimum_oos_trades", "severity": "needs_review"},
            {"check_id": "sample_size.minimum_total_samples", "severity": "warning"},
            {"check_id": "sensitivity.mildly_sensitive", "severity": "warning"},
            {"check_id": "multiple_testing.missing_trial_count_metadata", "severity": "needs_review"},
            {"check_id": "temporal_validation.purged_interval_overlap", "severity": "blocked"},
            {"check_id": "temporal_validation.train_validation_overlap", "severity": "blocked"},
        ]
    )

    assert codes == [
        "insufficient_oos_trades",
        "missing_trial_count_metadata",
        "sensitive_parameter_region",
        "temporal_validation_leakage_risk",
        "temporal_validation_overlap",
        "thin_total_sample",
        "weak_walk_forward_efficiency",
    ]


def test_malformed_and_conflicting_metadata_degrades_gracefully(tmp_path: Path) -> None:
    report_dir = tmp_path / "artifacts" / "robustness" / "bad_report"
    _write_json(report_dir / "robustness_summary.json", {"report_id": "bad_report", "finding_count": 0})
    _write_json(
        report_dir / "robustness_findings.json",
        {
            "report_id": "different_report",
            "finding_count": 1,
            "findings": [
                {
                    "check_id": "sample_size.minimum_total_samples",
                    "severity": "warning",
                    "workflow_type": "strategy",
                    "run_id": "run_bad",
                    "message": "Thin sample.",
                    "details": {},
                }
            ],
        },
    )
    malformed_dir = tmp_path / "artifacts" / "robustness" / "malformed"
    malformed_dir.mkdir(parents=True)
    (malformed_dir / "robustness_summary.json").write_text("{not json", encoding="utf-8")

    context = load_robustness_governance_context(report_dir, workflow_type="strategy", run_id="run_bad", roots=(tmp_path,))
    malformed = load_robustness_governance_context(malformed_dir, workflow_type="strategy", run_id="malformed", roots=(tmp_path,))

    assert context.robustness_available is True
    assert context.details["metadata_conflicts"][0]["check_id"] == "robustness_finding_count_mismatch"
    assert context.robustness_reason_codes == ("thin_total_sample",)
    assert malformed.robustness_available is False
    assert malformed.robustness_missing_reason == "robustness_report_malformed"


def test_attach_robustness_context_does_not_alter_promotion_fields() -> None:
    rows = [
        {
            "workflow_type": "strategy",
            "run_id": "run_a",
            "promotion_status": "eligible",
            "decision_reason_codes": "severity_warn",
        }
    ]
    context = RobustnessGovernanceContext(
        workflow_type="strategy",
        run_id="run_a",
        robustness_status="blocked",
        robustness_available=True,
        robustness_reason_codes=("temporal_validation_overlap",),
    )

    [row] = attach_robustness_context_to_governance_rows(rows, {("strategy", "run_a"): context})

    assert row["promotion_status"] == "eligible"
    assert row["decision_reason_codes"] == "severity_warn"
    assert row["robustness_status"] == "blocked"
    assert row["robustness_reason_codes"] == "temporal_validation_overlap"


def test_governance_report_surfaces_robustness_context_without_changing_decision(tmp_path: Path) -> None:
    summary_path = _complete_robustness_bundle(tmp_path)
    artifact_root = tmp_path / "artifacts" / "strategies"
    registry_path = artifact_root / "registry.jsonl"
    run_dir = artifact_root / "strategy_001"
    promotion_summary = {
        "promotion_status": "eligible",
        "evaluation_status": "pass",
        "highest_severity": None,
        "decision_reason_codes": [],
        "gate_count": 1,
    }
    _write_json(run_dir / "manifest.json", {"run_id": "strategy_001", "promotion_gate_summary": promotion_summary})
    append_registry_entry(
        registry_path,
        {
            "run_id": "strategy_001",
            "run_type": "strategy",
            "artifact_path": run_dir.as_posix(),
            "promotion_status": "eligible",
            "review_status": "candidate",
            "promotion_gate_summary": promotion_summary,
            "robustness_report_path": summary_path.as_posix(),
        },
    )

    dataset = load_governance_artifacts(registry_path=registry_path, artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)
    result = run_promotion_governance_report(
        registry_path=registry_path,
        artifact_root=artifact_root,
        output_dir=tmp_path / "governance",
        report_id="robustness_governance",
    )

    assert rows[0]["promotion_status"] == "eligible"
    assert rows[0]["review_status"] == "candidate"
    assert rows[0]["robustness_available"] == "true"
    assert rows[0]["robustness_status"] == "blocked"
    assert "temporal_validation_embargo_violation" in rows[0]["robustness_reason_codes"]

    with result.outcome_matrix_path.open("r", encoding="utf-8", newline="") as handle:
        [row] = list(csv.DictReader(handle))
    assert row["promotion_status"] == "eligible"
    assert row["robustness_status"] == "blocked"
    assert row["robustness_report_path"].endswith("robustness_summary.json")

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["robustness_available_count"] == 1
    assert summary["robustness_status_counts"]["blocked"] == 1
    assert not _contains_absolute_path(json.loads(result.manifest_path.read_text(encoding="utf-8")))
    assert not _contains_absolute_path(row)


def test_output_ordering_remains_deterministic() -> None:
    rows = [
        {"workflow_type": "strategy", "run_id": "b"},
        {"workflow_type": "alpha", "run_id": "a"},
    ]
    attached = attach_robustness_context_to_governance_rows(rows, {})

    assert [(row["workflow_type"], row["run_id"]) for row in attached] == [("alpha", "a"), ("strategy", "b")]


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        return (
            "C:/Users/" in normalized
            or normalized.startswith("file://")
            or normalized.startswith("/Users/")
            or normalized.startswith("/home/")
        )
    return False
