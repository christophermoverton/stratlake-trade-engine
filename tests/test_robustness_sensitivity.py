from __future__ import annotations

import json
import math
from pathlib import Path

from src.research.robustness import (
    FINDINGS_FILENAME,
    SUMMARY_FILENAME,
    RobustnessReport,
    SensitivityInput,
    SensitivityThresholds,
    build_sensitivity_evidence,
    build_sensitivity_findings,
    build_sensitivity_summary_rows,
    classify_fragility,
    evaluate_parameter_sensitivity,
    write_robustness_report_bundle,
)


def _record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "workflow_type": "strategy",
        "run_id": "run_001",
        "source_run_id": "source_001",
        "scenario_id": "lookback_plus_5",
        "parameter_name": "lookback",
        "base_value": 20,
        "perturbed_value": 25,
        "metric_name": "sharpe_ratio",
        "base_metric_value": 1.0,
        "perturbed_metric_value": 0.95,
        "higher_is_better": True,
        "perturbation_type": "absolute",
        "perturbation_size": 5,
    }
    base.update(overrides)
    return base


def test_classify_fragility_levels() -> None:
    thresholds = SensitivityThresholds()

    assert classify_fragility(deterioration=-0.1, relative_deterioration=-0.1, thresholds=thresholds) == "improved"
    assert classify_fragility(deterioration=0.01, relative_deterioration=0.01, thresholds=thresholds) == "stable"
    assert classify_fragility(deterioration=0.03, relative_deterioration=0.08, thresholds=thresholds) == "mildly_sensitive"
    assert classify_fragility(deterioration=0.4, relative_deterioration=0.3, thresholds=thresholds) == "fragile"


def test_evaluate_parameter_sensitivity_stable_case() -> None:
    result = evaluate_parameter_sensitivity(_record(perturbed_metric_value=0.99))

    assert result.status == "stable"
    assert result.absolute_delta == -0.01
    assert result.deterioration == 0.01
    assert result.relative_deterioration == 0.01


def test_evaluate_parameter_sensitivity_mildly_sensitive_case() -> None:
    result = evaluate_parameter_sensitivity(_record(perturbed_metric_value=0.90))

    assert result.status == "mildly_sensitive"
    assert result.deterioration == 0.1
    assert result.relative_deterioration == 0.1


def test_evaluate_parameter_sensitivity_fragile_case() -> None:
    result = evaluate_parameter_sensitivity(_record(perturbed_metric_value=0.6))

    assert result.status == "fragile"
    assert result.deterioration == 0.4
    assert result.relative_deterioration == 0.4


def test_evaluate_parameter_sensitivity_improved_case() -> None:
    result = evaluate_parameter_sensitivity(_record(perturbed_metric_value=1.3))

    assert result.status == "improved"
    assert result.deterioration == -0.3
    assert result.relative_deterioration == -0.3


def test_evaluate_parameter_sensitivity_missing_base_metric() -> None:
    result = evaluate_parameter_sensitivity(_record(base_metric_value=None))

    assert result.status == "missing"
    assert result.reason == "missing_base_metric_value"


def test_evaluate_parameter_sensitivity_missing_perturbed_metric() -> None:
    result = evaluate_parameter_sensitivity(_record(perturbed_metric_value=None))

    assert result.status == "missing"
    assert result.reason == "missing_perturbed_metric_value"


def test_evaluate_parameter_sensitivity_missing_parameter_metadata() -> None:
    result = evaluate_parameter_sensitivity(_record(parameter_name=""))

    assert result.status == "missing"
    assert result.reason == "missing_parameter_name"


def test_evaluate_parameter_sensitivity_non_finite_metric_values() -> None:
    result = evaluate_parameter_sensitivity(_record(base_metric_value=math.inf))
    second = evaluate_parameter_sensitivity(_record(perturbed_metric_value=math.nan))

    assert result.status == "undefined"
    assert result.reason == "non_finite_base_metric_value"
    assert second.status == "undefined"
    assert second.reason == "non_finite_perturbed_metric_value"


def test_evaluate_parameter_sensitivity_near_zero_base_metric() -> None:
    result = evaluate_parameter_sensitivity(_record(base_metric_value=1e-14, perturbed_metric_value=0.5))

    assert result.status == "undefined"
    assert result.reason == "near_zero_base_metric"
    assert result.relative_deterioration is None
    assert result.absolute_delta is not None


def test_evaluate_parameter_sensitivity_lower_is_better_metric() -> None:
    # For lower-is-better metrics, a lower perturbed value should be improved.
    improved = evaluate_parameter_sensitivity(
        _record(metric_name="max_drawdown", base_metric_value=0.20, perturbed_metric_value=0.15, higher_is_better=False)
    )
    fragile = evaluate_parameter_sensitivity(
        _record(metric_name="max_drawdown", base_metric_value=0.20, perturbed_metric_value=0.35, higher_is_better=False)
    )

    assert improved.status == "improved"
    assert improved.deterioration == -0.05
    assert fragile.status == "fragile"
    assert fragile.deterioration == 0.15


def test_parameter_distance_calculation_for_numeric_values() -> None:
    result = evaluate_parameter_sensitivity(_record(base_value=20, perturbed_value=24))

    assert result.parameter_distance == 4.0


def test_normalized_parameter_distance_calculation() -> None:
    result = evaluate_parameter_sensitivity(_record(base_value=20, perturbed_value=24))

    assert result.normalized_parameter_distance == 0.2


def test_categorical_perturbation_has_no_numeric_distance() -> None:
    result = evaluate_parameter_sensitivity(_record(base_value="ema", perturbed_value="sma", perturbation_type="categorical"))

    assert result.parameter_distance is None
    assert result.normalized_parameter_distance is None


def test_summary_rows_are_deterministically_ordered() -> None:
    rows = build_sensitivity_summary_rows(
        [
            _record(run_id="run_b", scenario_id="zeta", parameter_name="window", perturbed_value=3),
            _record(run_id="run_a", scenario_id="alpha", parameter_name="lookback", perturbed_value=2),
            _record(run_id="run_a", scenario_id="beta", parameter_name="lookback", perturbed_value=1),
        ]
    )

    ordering = [
        (row.workflow_type, row.run_id, row.parameter, row.scenario_id, row.metric, str(row.scenario_value))
        for row in rows
    ]
    assert ordering == sorted(ordering)


def test_findings_are_structured_and_status_mapped() -> None:
    findings = build_sensitivity_findings(
        [
            _record(scenario_id="fragile", perturbed_metric_value=0.60),
            _record(scenario_id="mild", perturbed_metric_value=0.90),
            _record(scenario_id="undefined", base_metric_value=1e-14, perturbed_metric_value=0.5),
            _record(scenario_id="missing", perturbed_metric_value=None),
        ],
        include_info=False,
    )

    assert [finding.check_id for finding in findings] == [
        "sensitivity.fragile",
        "sensitivity.mildly_sensitive",
        "sensitivity.missing",
        "sensitivity.undefined",
    ]
    fragile_details = findings[0].to_dict()["details"]
    assert fragile_details["parameter_name"] == "lookback"
    assert fragile_details["metric_name"] == "sharpe_ratio"
    assert fragile_details["thresholds"]["fragile_relative_delta_min"] == 0.25


def test_info_findings_optional_for_stable_and_improved() -> None:
    records = [
        _record(scenario_id="improved", perturbed_metric_value=1.1),
        _record(scenario_id="stable", perturbed_metric_value=0.99),
    ]

    without_info = build_sensitivity_findings(records, include_info=False)
    with_info = build_sensitivity_findings(records, include_info=True)

    assert without_info == []
    assert [finding.check_id for finding in with_info] == ["sensitivity.improved", "sensitivity.stable"]


def test_sensitivity_evidence_writes_through_report_bundle(tmp_path: Path) -> None:
    rows, findings = build_sensitivity_evidence(
        [
            _record(
                run_id="run_001",
                scenario_id="fragile",
                perturbed_metric_value=0.60,
                details={"source_path": tmp_path / "artifacts" / "source" / "metrics.json"},
            ),
            _record(run_id="run_001", scenario_id="undefined", base_metric_value=1e-14, perturbed_metric_value=0.3),
            _record(run_id="run_001", scenario_id="missing", perturbed_metric_value=None),
        ],
        include_info_findings=True,
    )
    report = RobustnessReport(
        report_id="sensitivity_bundle",
        workflow_type="strategy",
        run_id="run_001",
        robustness_status="needs_review",
        findings=tuple(findings),
        sensitivity_summary=tuple(rows),
        checks_present=("sensitivity",),
    )

    result = write_robustness_report_bundle(report, output_root=tmp_path / "artifacts" / "robustness")

    sensitivity_text = result.sensitivity_summary_path.read_text(encoding="utf-8")
    assert "status" in sensitivity_text.splitlines()[0]
    assert "nan" not in sensitivity_text.lower()
    assert "inf" not in sensitivity_text.lower()

    findings_payload = json.loads((result.output_dir / FINDINGS_FILENAME).read_text(encoding="utf-8"))
    assert findings_payload["finding_count"] == len(findings)
    assert {item["check_id"] for item in findings_payload["findings"]} >= {
        "sensitivity.fragile",
        "sensitivity.missing",
        "sensitivity.undefined",
    }
    assert not _contains_absolute_path(findings_payload)

    summary = json.loads((result.output_dir / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["checks_present"] == ["sensitivity"]
    assert summary["robustness_status_counts"]["fragile"] >= 1
    assert summary["robustness_status_counts"]["missing"] >= 1
    assert summary["robustness_status_counts"]["undefined"] >= 1

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert not _contains_absolute_path(manifest)


def test_sensitivity_input_dataclass_supported() -> None:
    rows = build_sensitivity_summary_rows(
        [
            SensitivityInput(
                workflow_type="strategy",
                run_id="run_100",
                source_run_id="source_100",
                parameter_name="lookback",
                base_value=20,
                perturbed_value=30,
                metric_name="sharpe_ratio",
                base_metric_value=1.0,
                perturbed_metric_value=0.7,
                higher_is_better=True,
                scenario_id="plus10",
                perturbation_type="relative",
                perturbation_size=0.5,
            )
        ]
    )

    assert rows[0].status == "fragile"


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
