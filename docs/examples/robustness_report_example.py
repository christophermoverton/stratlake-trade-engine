from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.robustness import (  # noqa: E402
    ArtifactReference,
    MultipleTestingThresholds,
    PurgedSplitConfig,
    RobustnessReport,
    SampleSizeThresholds,
    SensitivityThresholds,
    UpstreamReferences,
    WalkForwardEfficiencyThresholds,
    build_multiple_testing_evidence,
    build_purged_split_evidence,
    build_sample_size_evidence,
    build_sensitivity_evidence,
    build_walk_forward_efficiency_evidence,
    load_robustness_governance_context,
    write_purged_split_artifacts,
    write_robustness_report_bundle,
)


REPORT_ID = "robustness_report_example"
RUN_ID = "docs_m34_synthetic_strategy"
WORKFLOW_TYPE = "strategy"
OUTPUT_DIR = REPO_ROOT / "docs" / "examples" / "output" / REPORT_ID


def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    wfe_rows, wfe_findings = build_walk_forward_efficiency_evidence(
        [
            {
                "workflow_type": WORKFLOW_TYPE,
                "run_id": RUN_ID,
                "split_id": "split_001",
                "sharpe_is": 1.40,
                "sharpe_oos": 0.95,
                "train_start": "2025-01-01",
                "train_end": "2025-03-31",
                "test_start": "2025-04-01",
                "test_end": "2025-04-30",
                "n_trades_is": 36,
                "n_trades_oos": 8,
                "details": {"example_input": "synthetic_fixture"},
            },
            {
                "workflow_type": WORKFLOW_TYPE,
                "run_id": RUN_ID,
                "split_id": "split_002",
                "sharpe_is": 1.30,
                "sharpe_oos": 0.20,
                "train_start": "2025-02-01",
                "train_end": "2025-04-30",
                "test_start": "2025-05-01",
                "test_end": "2025-05-31",
                "n_trades_is": 34,
                "n_trades_oos": 5,
                "details": {"example_input": "synthetic_fixture"},
            },
        ],
        thresholds=WalkForwardEfficiencyThresholds(),
        include_info_findings=False,
    )

    sample_rows, sample_findings = build_sample_size_evidence(
        [
            {
                "workflow_type": WORKFLOW_TYPE,
                "run_id": RUN_ID,
                "sample_count": 180,
                "trade_count": 24,
                "oos_trade_count": 8,
                "split_trade_counts": {"split_001": 8, "split_002": 5},
                "unique_period_count": 28,
                "regime_trade_counts": {"risk_on": 18, "risk_off": 6},
                "details": {"example_input": "synthetic_fixture"},
            }
        ],
        thresholds=SampleSizeThresholds(),
    )

    sensitivity_rows, sensitivity_findings = build_sensitivity_evidence(
        [
            {
                "workflow_type": WORKFLOW_TYPE,
                "run_id": RUN_ID,
                "scenario_id": "lookback_plus_5",
                "parameter_name": "lookback_days",
                "base_value": 20,
                "perturbed_value": 25,
                "metric_name": "sharpe_ratio",
                "base_metric_value": 1.20,
                "perturbed_metric_value": 0.82,
                "higher_is_better": True,
                "perturbation_type": "local_numeric_grid",
                "perturbation_size": 5,
                "details": {"example_input": "synthetic_fixture"},
            }
        ],
        thresholds=SensitivityThresholds(),
    )

    multiple_testing_rows, multiple_testing_findings = build_multiple_testing_evidence(
        [
            {
                "workflow_type": WORKFLOW_TYPE,
                "run_id": RUN_ID,
                "family_id": "docs_example_selection",
                "candidate_count": 160,
                "tested_configuration_count": 160,
                "parameter_combination_count": 80,
                "scenario_count": 4,
                "model_count": 2,
                "selected_rank": 1,
                "selection_metric": "sharpe_ratio",
                "selection_metric_value": 1.20,
                "trial_count_source": "explicit_config",
                "details": {"example_input": "synthetic_fixture"},
            }
        ],
        thresholds=MultipleTestingThresholds(),
    )

    temporal_plan, temporal_findings = build_purged_split_evidence(
        _temporal_observations(),
        config=PurgedSplitConfig(
            n_splits=2,
            validation_window_size=2,
            embargo_window="1D",
            min_train_observations=2,
            min_validation_observations=2,
        ),
        workflow_type=WORKFLOW_TYPE,
        run_id=RUN_ID,
        include_pass_findings=False,
    )

    findings = tuple(
        sorted(
            [
                *wfe_findings,
                *sample_findings,
                *sensitivity_findings,
                *multiple_testing_findings,
                *temporal_findings,
            ],
            key=lambda finding: (finding.severity, finding.workflow_type, finding.run_id, finding.check_id),
        )
    )
    report = RobustnessReport(
        report_id=REPORT_ID,
        workflow_type=WORKFLOW_TYPE,
        run_id=RUN_ID,
        source_run_id="docs_m34_source_run",
        robustness_status="needs_review",
        upstream_references=UpstreamReferences(
            strategy=(
                ArtifactReference(
                    path="docs/examples/robustness_report_example.py",
                    artifact_type="example_script",
                    description="Synthetic M34 robustness example input.",
                ),
            )
        ),
        findings=findings,
        walk_forward_efficiency=tuple(wfe_rows),
        sample_size_validation=tuple(sample_rows),
        sensitivity_summary=tuple(sensitivity_rows),
        multiple_testing_summary=tuple(multiple_testing_rows),
        checks_present=(
            "walk_forward_efficiency",
            "sample_size",
            "sensitivity",
            "multiple_testing",
            "temporal_validation",
            "governance_context",
        ),
        checks_missing=(),
        created_at_utc="2026-01-01T00:00:00Z",
        metadata={
            "example": REPORT_ID,
            "fixture_type": "deterministic_synthetic",
            "advanced_methods": {
                "cpcv_supported": False,
                "dsr_supported": False,
                "haircut_supported": False,
                "pbo_supported": False,
            },
        },
    )

    report_result = write_robustness_report_bundle(report, output_root=OUTPUT_DIR)
    temporal_result = write_purged_split_artifacts(temporal_plan, output_root=report_result.output_dir)
    governance_context = load_robustness_governance_context(
        report_result.summary_path,
        workflow_type=WORKFLOW_TYPE,
        run_id=RUN_ID,
        source_run_id="docs_m34_source_run",
        roots=(REPO_ROOT,),
    )

    summary = {
        "artifact_count": 11,
        "governance_available": governance_context.robustness_available,
        "highest_robustness_severity": governance_context.highest_robustness_severity,
        "output_dir": report_result.output_dir.relative_to(REPO_ROOT).as_posix(),
        "reason_codes": governance_context.robustness_reason_codes,
        "robustness_status": governance_context.robustness_status,
        "temporal_validation_status": governance_context.temporal_validation_status,
        "wrote": sorted(
            [
                report_result.summary_path.name,
                report_result.findings_path.name,
                report_result.walk_forward_efficiency_path.name,
                report_result.sample_size_validation_path.name,
                report_result.sensitivity_summary_path.name,
                report_result.multiple_testing_summary_path.name,
                report_result.markdown_path.name,
                report_result.manifest_path.name,
                temporal_result.purged_split_plan_path.name,
                temporal_result.purged_split_summary_path.name,
                temporal_result.leakage_validation_path.name,
            ]
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _temporal_observations() -> list[dict[str, str]]:
    return [
        {
            "observation_id": f"obs_{index:03d}",
            "timestamp": f"2025-06-{index:02d}",
            "label_start": f"2025-06-{index:02d}",
            "label_end": f"2025-06-{index + 1:02d}",
        }
        for index in range(1, 7)
    ]


if __name__ == "__main__":
    main()
