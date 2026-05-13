from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from src.research.robustness import (
    FINDINGS_FILENAME,
    SUMMARY_FILENAME,
    WALK_FORWARD_EFFICIENCY_COLUMNS,
    WALK_FORWARD_EFFICIENCY_FILENAME,
    RobustnessReport,
    WalkForwardEfficiencyInput,
    WalkForwardEfficiencyThresholds,
    build_walk_forward_efficiency_evidence,
    build_walk_forward_efficiency_findings,
    build_walk_forward_efficiency_rows,
    classify_walk_forward_efficiency,
    compute_walk_forward_efficiency,
    write_robustness_report_bundle,
)


def test_compute_walk_forward_efficiency_classifies_core_statuses() -> None:
    assert compute_walk_forward_efficiency(2.0, 1.8).wfe == 0.9
    assert compute_walk_forward_efficiency(2.0, 1.8).status == "robust"
    assert compute_walk_forward_efficiency(2.0, 1.2).status == "acceptable"
    assert compute_walk_forward_efficiency(2.0, 0.6).status == "weak"
    assert compute_walk_forward_efficiency(2.0, -0.2).status == "broken"
    assert classify_walk_forward_efficiency(0.75) == "robust"


def test_compute_walk_forward_efficiency_handles_unavailable_metrics() -> None:
    assert compute_walk_forward_efficiency(0.0, 1.0).status == "undefined"
    assert compute_walk_forward_efficiency(1e-14, 1.0).reason == "near_zero_in_sample_sharpe"
    assert compute_walk_forward_efficiency(-1.0, -0.5).status == "broken"
    assert compute_walk_forward_efficiency(None, 1.0).status == "missing"
    assert compute_walk_forward_efficiency(1.0, None).reason == "missing_sharpe_oos"
    assert compute_walk_forward_efficiency(math.inf, 1.0).status == "undefined"
    assert compute_walk_forward_efficiency(1.0, math.nan).reason == "non_finite_sharpe_oos"


def test_thresholds_are_configurable() -> None:
    thresholds = WalkForwardEfficiencyThresholds(robust_min=0.9, acceptable_min=0.6, weak_min=0.1)

    assert compute_walk_forward_efficiency(2.0, 1.6, thresholds=thresholds).status == "acceptable"
    assert compute_walk_forward_efficiency(2.0, 0.1, thresholds=thresholds).status == "broken"


def test_build_walk_forward_efficiency_rows_are_deterministic_and_schema_valid() -> None:
    records = [
        {
            "workflow_type": "strategy",
            "run_id": "run_b",
            "split_id": "split_002",
            "sharpe_is": 2.0,
            "sharpe_oos": 0.4,
            "trade_count": 3,
        },
        WalkForwardEfficiencyInput(
            workflow_type="strategy",
            run_id="run_a",
            split_id="split_001",
            sharpe_is=2.0,
            sharpe_oos=1.6,
            train_start="2025-01-01",
            train_end="2025-03-01",
            test_start="2025-03-01",
            test_end="2025-04-01",
            n_trades_is=None,
            n_trades_oos=None,
            source_run_id="source_a",
        ),
    ]

    rows = build_walk_forward_efficiency_rows(records)

    assert [(row.run_id, row.split_id) for row in rows] == [("run_a", "split_001"), ("run_b", "split_002")]
    assert rows[0].status == "robust"
    assert rows[0].train_period == "2025-01-01/2025-03-01"
    assert rows[0].details["source_run_id"] == "source_a"
    assert "n_trades_is" not in rows[0].details
    assert rows[1].status == "weak"
    assert rows[1].details["n_trades_oos"] == 3


def test_build_walk_forward_efficiency_findings_include_structured_details() -> None:
    findings = build_walk_forward_efficiency_findings(
        [
            WalkForwardEfficiencyInput("strategy", "run_a", "split_001", sharpe_is=2.0, sharpe_oos=1.6),
            WalkForwardEfficiencyInput("strategy", "run_a", "split_002", sharpe_is=2.0, sharpe_oos=0.2),
            WalkForwardEfficiencyInput("strategy", "run_a", "split_003", sharpe_is=0.0, sharpe_oos=1.0),
            WalkForwardEfficiencyInput("strategy", "run_a", "split_004", sharpe_is=None, sharpe_oos=1.0),
        ]
    )

    assert [finding.check_id for finding in findings] == [
        "walk_forward_efficiency.robust",
        "walk_forward_efficiency.weak",
        "walk_forward_efficiency.undefined",
        "walk_forward_efficiency.missing",
    ]
    assert [finding.severity for finding in findings] == ["info", "warning", "needs_review", "needs_review"]
    undefined = findings[2].to_dict()["details"]
    assert undefined["reason"] == "near_zero_in_sample_sharpe"
    assert undefined["thresholds"]["robust_min"] == 0.75


def test_wfe_rows_and_findings_write_through_report_bundle(tmp_path: Path) -> None:
    rows, findings = build_walk_forward_efficiency_evidence(
        [
            WalkForwardEfficiencyInput(
                workflow_type="strategy",
                run_id="run_a",
                split_id="split_001",
                sharpe_is=2.0,
                sharpe_oos=1.6,
                n_trades_is=10,
                n_trades_oos=4,
            ),
            WalkForwardEfficiencyInput(
                workflow_type="strategy",
                run_id="run_a",
                split_id="split_002",
                sharpe_is=2.0,
                sharpe_oos=-0.4,
            ),
        ]
    )
    report = RobustnessReport(
        report_id="wfe_bundle",
        workflow_type="strategy",
        run_id="run_a",
        robustness_status="needs_review",
        findings=tuple(findings),
        walk_forward_efficiency=tuple(rows),
        checks_present=("walk_forward_efficiency",),
        checks_missing=(),
    )

    result = write_robustness_report_bundle(report, output_root=tmp_path / "artifacts" / "robustness")

    with result.walk_forward_efficiency_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == WALK_FORWARD_EFFICIENCY_COLUMNS
        csv_rows = list(reader)
    assert [row["status"] for row in csv_rows] == ["robust", "broken"]
    assert csv_rows[0]["walk_forward_efficiency"] == "0.8"

    findings_payload = json.loads((result.output_dir / FINDINGS_FILENAME).read_text(encoding="utf-8"))
    assert findings_payload["finding_count"] == 2
    assert {finding["check_id"] for finding in findings_payload["findings"]} == {
        "walk_forward_efficiency.broken",
        "walk_forward_efficiency.robust",
    }

    summary = json.loads((result.output_dir / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["checks_present"] == ["walk_forward_efficiency"]
    assert summary["robustness_status_counts"]["robust"] == 1
    assert summary["robustness_status_counts"]["broken"] == 1

    text = (result.output_dir / WALK_FORWARD_EFFICIENCY_FILENAME).read_text(encoding="utf-8")
    assert "nan" not in text.lower()
    assert "inf" not in text.lower()
    assert not _contains_absolute_path(json.loads(result.manifest_path.read_text(encoding="utf-8")))


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        return "C:/Users/" in normalized or normalized.startswith("file://") or normalized.startswith("/Users/") or normalized.startswith("/home/")
    return False
