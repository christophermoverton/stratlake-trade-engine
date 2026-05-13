from __future__ import annotations

import json
import math
from pathlib import Path

from src.research.robustness import (
    FINDINGS_FILENAME,
    MULTIPLE_TESTING_FILENAME,
    SUMMARY_FILENAME,
    MultipleTestingInput,
    MultipleTestingThresholds,
    RobustnessReport,
    build_multiple_testing_evidence,
    build_multiple_testing_findings,
    build_multiple_testing_summaries,
    classify_trial_count_risk,
    evaluate_multiple_testing_risk,
    write_robustness_report_bundle,
)


def _record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "workflow_type": "campaign",
        "run_id": "campaign_001",
        "source_run_id": "source_001",
        "family_id": "candidate_sweep",
        "candidate_count": 8,
        "tested_configuration_count": 8,
        "selected_rank": 1,
        "selection_metric": "sharpe_ratio",
        "selection_metric_value": 1.25,
        "trial_count_source": "campaign_manifest",
    }
    base.update(overrides)
    return base


def test_classify_trial_count_risk_levels() -> None:
    thresholds = MultipleTestingThresholds()

    assert classify_trial_count_risk(10, thresholds=thresholds) == "low_risk"
    assert classify_trial_count_risk(50, thresholds=thresholds) == "moderate_risk"
    assert classify_trial_count_risk(250, thresholds=thresholds) == "high_risk"
    assert classify_trial_count_risk(1000, thresholds=thresholds) == "extreme_risk"


def test_evaluate_low_risk_known_trial_count() -> None:
    result = evaluate_multiple_testing_risk(_record(candidate_count=5, tested_configuration_count=4))

    assert result.status == "low_risk"
    assert result.effective_trial_count == 5
    assert result.trial_counts["candidate_count"] == 5


def test_evaluate_moderate_high_and_extreme_trial_counts() -> None:
    moderate = evaluate_multiple_testing_risk(_record(candidate_count=25))
    high = evaluate_multiple_testing_risk(_record(candidate_count=250))
    extreme = evaluate_multiple_testing_risk(_record(candidate_count=5000))

    assert moderate.status == "moderate_risk"
    assert high.status == "high_risk"
    assert extreme.status == "extreme_risk"


def test_missing_trial_count_metadata_emits_missing_finding() -> None:
    records = [_record(candidate_count=None, tested_configuration_count=None)]

    summaries, findings = build_multiple_testing_evidence(records)

    assert summaries[0].status == "missing"
    assert summaries[0].effective_trial_count is None
    assert [finding.check_id for finding in findings] == [
        "multiple_testing.missing_trial_count_metadata"
    ]
    assert findings[0].severity == "needs_review"


def test_undefined_non_finite_and_negative_trial_count_metadata() -> None:
    non_finite = evaluate_multiple_testing_risk(_record(candidate_count=math.inf))
    negative = evaluate_multiple_testing_risk(_record(candidate_count=-1))
    fractional = evaluate_multiple_testing_risk(_record(candidate_count=2.5))

    assert non_finite.status == "undefined"
    assert negative.status == "undefined"
    assert fractional.status == "undefined"


def test_selected_rank_and_selection_metric_recorded_correctly() -> None:
    summaries = build_multiple_testing_summaries(
        [_record(candidate_count=12, selected_rank="2", selection_metric="mean_ic", selection_metric_value="0.04")]
    )

    details = summaries[0].details
    assert summaries[0].effective_trial_count == 12
    assert details["selected_rank"] == 2
    assert details["selection_metric"] == "mean_ic"
    assert details["selection_metric_value"] == 0.04
    assert details["trial_count_source"] == "campaign_manifest"


def test_selected_rank_warning_for_top_ranked_large_search_space() -> None:
    findings = build_multiple_testing_findings([_record(candidate_count=250, selected_rank=1)])

    assert "multiple_testing.high_risk" in {finding.check_id for finding in findings}
    rank_findings = [finding for finding in findings if finding.check_id == "multiple_testing.selected_rank_warning"]
    assert rank_findings
    assert rank_findings[0].details["rank_reason"] == "top_rank_selected_from_large_search_space"


def test_selected_rank_greater_than_candidate_count_needs_review() -> None:
    findings = build_multiple_testing_findings([_record(candidate_count=3, selected_rank=4)])

    assert [finding.check_id for finding in findings] == ["multiple_testing.selected_rank_warning"]
    assert findings[0].details["selected_rank_status"] == "undefined"
    assert findings[0].details["rank_reason"] == "invalid_selected_rank"


def test_missing_selection_metric_when_selected_rank_present() -> None:
    findings = build_multiple_testing_findings([_record(candidate_count=3, selected_rank=2, selection_metric="")])

    assert [finding.check_id for finding in findings] == ["multiple_testing.selected_rank_warning"]
    assert findings[0].details["rank_reason"] == "missing_selection_metric"


def test_missing_selected_rank_when_candidate_count_present() -> None:
    findings = build_multiple_testing_findings([_record(candidate_count=3, selected_rank=None)])

    assert [finding.check_id for finding in findings] == ["multiple_testing.selected_rank_warning"]
    assert findings[0].details["rank_reason"] == "missing_selected_rank"


def test_effective_trial_count_uses_max_explicit_count_field() -> None:
    result = evaluate_multiple_testing_risk(
        _record(
            candidate_count=20,
            tested_configuration_count=30,
            parameter_combination_count=15,
            scenario_count=80,
            factor_count=5,
            model_count=7,
            portfolio_count=3,
            campaign_count=2,
        )
    )

    assert result.effective_trial_count == 80
    assert result.status == "moderate_risk"


def test_summary_and_findings_are_deterministically_ordered() -> None:
    records = [
        _record(run_id="run_b", trial_count_source="registry", selection_metric="z_metric", candidate_count=250),
        _record(run_id="run_a", trial_count_source="campaign_manifest", selection_metric="a_metric", candidate_count=5),
    ]

    summaries = build_multiple_testing_summaries(records)
    findings = build_multiple_testing_findings(records, include_info=True)

    summary_order = [
        (row.workflow_type, row.run_id, row.details["trial_count_source"], row.details["selection_metric"])
        for row in summaries
    ]
    finding_order = [
        (finding.workflow_type, finding.run_id, finding.details["trial_count_source"], finding.check_id)
        for finding in findings
    ]
    assert summary_order == sorted(summary_order)
    assert finding_order == sorted(finding_order)


def test_dataclass_input_supported() -> None:
    summaries = build_multiple_testing_summaries(
        [
            MultipleTestingInput(
                workflow_type="strategy",
                run_id="run_dataclass",
                candidate_count=125,
                selected_rank=1,
                selection_metric="sharpe_ratio",
                trial_count_source="explicit_config",
            )
        ]
    )

    assert summaries[0].status == "high_risk"


def test_multiple_testing_evidence_writes_through_report_bundle(tmp_path: Path) -> None:
    rows, findings = build_multiple_testing_evidence(
        [
            _record(
                run_id="campaign_run",
                candidate_count=250,
                selected_rank=1,
                details={"source_path": tmp_path / "artifacts" / "campaign_run" / "manifest.json"},
            ),
            _record(run_id="missing_run", candidate_count=None, tested_configuration_count=None),
            _record(run_id="undefined_run", candidate_count=math.nan),
        ],
        include_info_findings=True,
    )
    report = RobustnessReport(
        report_id="multiple_testing_bundle",
        workflow_type="campaign",
        run_id="campaign_run",
        robustness_status="needs_review",
        findings=tuple(findings),
        multiple_testing_summary=tuple(rows),
        checks_present=("multiple_testing",),
    )

    result = write_robustness_report_bundle(report, output_root=tmp_path / "artifacts" / "robustness")

    multiple_testing = json.loads((result.output_dir / MULTIPLE_TESTING_FILENAME).read_text(encoding="utf-8"))
    assert multiple_testing["report_id"] == "multiple_testing_bundle"
    assert {family["status"] for family in multiple_testing["families"]} >= {
        "high_risk",
        "missing",
        "undefined",
    }
    assert not _contains_absolute_path(multiple_testing)
    assert "nan" not in json.dumps(multiple_testing).lower()
    assert "inf" not in json.dumps(multiple_testing).lower()

    findings_payload = json.loads((result.output_dir / FINDINGS_FILENAME).read_text(encoding="utf-8"))
    assert findings_payload["finding_count"] == len(findings)
    assert {finding["check_id"] for finding in findings_payload["findings"]} >= {
        "multiple_testing.high_risk",
        "multiple_testing.missing_trial_count_metadata",
        "multiple_testing.undefined_trial_count_metadata",
        "multiple_testing.selected_rank_warning",
    }
    assert not _contains_absolute_path(findings_payload)

    summary = json.loads((result.output_dir / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["checks_present"] == ["multiple_testing"]
    assert summary["robustness_status_counts"]["high_risk"] >= 1
    assert summary["robustness_status_counts"]["missing"] >= 1
    assert summary["robustness_status_counts"]["undefined"] >= 1


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
