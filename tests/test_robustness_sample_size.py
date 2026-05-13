from __future__ import annotations

import json
import math
from pathlib import Path

from src.research.robustness import (
    FINDINGS_FILENAME,
    SAMPLE_SIZE_FILENAME,
    SUMMARY_FILENAME,
    RobustnessReport,
    SampleSizeInput,
    SampleSizeThresholds,
    build_sample_size_evidence,
    build_sample_size_findings,
    build_sample_size_validations,
    evaluate_sample_size_guardrails,
    write_robustness_report_bundle,
)


def _sufficient_record() -> SampleSizeInput:
    return SampleSizeInput(
        workflow_type="strategy",
        run_id="run_sufficient",
        source_run_id="source_sufficient",
        sample_count=300,
        trade_count=45,
        oos_trade_count=12,
        split_trade_counts={"split_002": 7, "split_001": 6},
        unique_period_count=120,
        regime_trade_counts={"calm": 9, "stress": 8},
    )


def test_sample_size_guardrails_pass_for_sufficient_evidence() -> None:
    validations = evaluate_sample_size_guardrails(_sufficient_record())

    assert {validation.status for validation in validations} == {"pass"}
    assert {validation.check_id for validation in validations} >= {
        "sample_size.minimum_total_samples",
        "sample_size.minimum_total_trades",
        "sample_size.minimum_oos_trades",
        "sample_size.minimum_trades_per_split",
        "sample_size.minimum_unique_periods",
        "sample_size.minimum_regime_coverage",
        "sample_size.minimum_trades_per_regime",
    }


def test_sample_size_guardrails_flag_thin_total_samples_and_trades() -> None:
    validations = build_sample_size_validations(
        [
            SampleSizeInput(
                workflow_type="strategy",
                run_id="run_thin",
                sample_count=100,
                trade_count=8,
                oos_trade_count=11,
                unique_period_count=40,
            )
        ]
    )

    by_check = {validation.check_id: validation for validation in validations}
    assert by_check["sample_size.minimum_total_samples"].status == "needs_review"
    assert by_check["sample_size.minimum_total_samples"].minimum_sample_count == 252
    assert by_check["sample_size.minimum_total_trades"].status == "needs_review"
    assert by_check["sample_size.minimum_total_trades"].minimum_trade_count == 30

    findings = build_sample_size_findings([{"workflow_type": "strategy", "run_id": "run_thin", "sample_count": 100, "trade_count": 8, "oos_trade_count": 11, "unique_period_count": 40}])
    assert {finding.check_id for finding in findings} == {
        "sample_size.minimum_total_samples",
        "sample_size.minimum_total_trades",
    }
    assert {finding.severity for finding in findings} == {"warning", "needs_review"}


def test_sample_size_guardrails_flag_oos_split_and_unique_period_evidence() -> None:
    thresholds = SampleSizeThresholds(minimum_oos_trades=10, minimum_trades_per_split=5, minimum_unique_periods=30)
    validations = build_sample_size_validations(
        [
            SampleSizeInput(
                workflow_type="portfolio",
                run_id="portfolio_run",
                sample_count=300,
                trade_count=50,
                oos_trade_count=3,
                split_trade_counts={"split_b": 2, "split_a": 8},
                unique_period_count=12,
            )
        ],
        thresholds=thresholds,
    )

    failing = [(validation.check_id, validation.details.get("split_id"), validation.status) for validation in validations if validation.status != "pass"]

    assert failing == [
        ("sample_size.minimum_oos_trades", None, "needs_review"),
        ("sample_size.minimum_trades_per_split", "split_b", "needs_review"),
        ("sample_size.minimum_unique_periods", None, "needs_review"),
    ]


def test_sample_size_guardrails_emit_distinct_missing_metadata_checks() -> None:
    validations = build_sample_size_validations(
        [
            {
                "workflow_type": "alpha",
                "run_id": "alpha_run",
                "sample_count": 300,
                "trade_count": None,
                "oos_trade_count": None,
                "split_id": "split_001",
                "split_trade_count": None,
                "unique_period_count": 60,
            }
        ]
    )

    missing = [validation for validation in validations if validation.status == "missing"]
    assert [(validation.check_id, validation.details.get("split_id")) for validation in missing] == [
        ("sample_size.missing_oos_trade_count", None),
        ("sample_size.missing_trade_count", None),
        ("sample_size.missing_trade_count", "split_001"),
    ]

    findings = build_sample_size_findings(
        [
            {
                "workflow_type": "alpha",
                "run_id": "alpha_run",
                "sample_count": 300,
                "trade_count": None,
                "oos_trade_count": None,
                "unique_period_count": 60,
            }
        ]
    )
    assert {finding.severity for finding in findings} == {"needs_review"}


def test_regime_coverage_runs_only_when_metadata_is_available() -> None:
    without_regimes = build_sample_size_validations([_sufficient_record().__dict__ | {"regime_trade_counts": None}])
    assert all("regime" not in validation.check_id for validation in without_regimes)

    with_regimes = build_sample_size_validations(
        [
            SampleSizeInput(
                workflow_type="strategy",
                run_id="regime_thin",
                sample_count=300,
                trade_count=40,
                oos_trade_count=12,
                unique_period_count=80,
                regime_trade_counts={"calm": 8, "stress": 2},
            )
        ]
    )
    by_regime = {
        (validation.check_id, validation.details.get("regime_id")): validation
        for validation in with_regimes
        if "regime" in validation.check_id
    }
    assert by_regime[("sample_size.minimum_regime_coverage", None)].status == "pass"
    assert by_regime[("sample_size.minimum_trades_per_regime", "stress")].status == "needs_review"


def test_sample_size_validations_and_findings_are_deterministically_ordered() -> None:
    records = [
        SampleSizeInput("strategy", "run_b", sample_count=10, trade_count=2, oos_trade_count=1, unique_period_count=5),
        SampleSizeInput("strategy", "run_a", sample_count=300, trade_count=50, oos_trade_count=12, split_trade_counts={"split_2": 1, "split_1": 7}, unique_period_count=60),
    ]

    validations = build_sample_size_validations(records)
    findings = build_sample_size_findings(records)

    assert [(validation.run_id, validation.check_id, validation.details.get("split_id", "")) for validation in validations] == sorted(
        (validation.run_id, validation.check_id, validation.details.get("split_id", "")) for validation in validations
    )
    assert [(finding.run_id, finding.check_id, finding.details.get("split_id", "")) for finding in findings] == sorted(
        (finding.run_id, finding.check_id, finding.details.get("split_id", "")) for finding in findings
    )


def test_sample_size_evidence_writes_through_report_bundle(tmp_path: Path) -> None:
    validations, findings = build_sample_size_evidence(
        [
            SampleSizeInput(
                workflow_type="campaign",
                run_id="campaign_run",
                source_run_id="campaign_source",
                sample_count=100,
                trade_count=math.nan,
                oos_trade_count=4,
                split_trade_counts={"split_001": 2},
                unique_period_count=10,
                details={"source_path": tmp_path / "artifacts" / "campaign_run" / "metrics.json"},
            )
        ]
    )
    report = RobustnessReport(
        report_id="sample_size_bundle",
        workflow_type="campaign",
        run_id="campaign_run",
        robustness_status="needs_review",
        findings=tuple(findings),
        sample_size_validation=tuple(validations),
        checks_present=("sample_size",),
    )

    result = write_robustness_report_bundle(report, output_root=tmp_path / "artifacts" / "robustness")

    sample_payload = json.loads((result.output_dir / SAMPLE_SIZE_FILENAME).read_text(encoding="utf-8"))
    assert sample_payload["report_id"] == "sample_size_bundle"
    assert {check["check_id"] for check in sample_payload["checks"]} >= {
        "sample_size.minimum_total_samples",
        "sample_size.missing_trade_count",
        "sample_size.minimum_oos_trades",
        "sample_size.minimum_trades_per_split",
        "sample_size.minimum_unique_periods",
    }
    assert not _contains_absolute_path(sample_payload)
    assert "nan" not in json.dumps(sample_payload).lower()
    assert "inf" not in json.dumps(sample_payload).lower()

    findings_payload = json.loads((result.output_dir / FINDINGS_FILENAME).read_text(encoding="utf-8"))
    assert findings_payload["finding_count"] == len(findings)
    assert "sample_size.missing_trade_count" in {finding["check_id"] for finding in findings_payload["findings"]}

    summary = json.loads((result.output_dir / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["checks_present"] == ["sample_size"]
    assert summary["robustness_status_counts"]["needs_review"] >= 1
    assert summary["robustness_status_counts"]["missing"] >= 1


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        return "C:/Users/" in normalized or normalized.startswith("file://") or normalized.startswith("/Users/") or normalized.startswith("/home/")
    return False
