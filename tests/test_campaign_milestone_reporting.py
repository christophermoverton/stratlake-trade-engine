from __future__ import annotations

import json
from pathlib import Path

from src.research.reporting import (
    MILESTONE_DECISION_LOG_FILENAME,
    MILESTONE_MARKDOWN_REPORT_FILENAME,
    MILESTONE_MANIFEST_FILENAME,
    MILESTONE_SUMMARY_FILENAME,
    generate_campaign_milestone_report,
    load_campaign_reporting_payloads,
    resolve_campaign_artifact_dir,
    validate_milestone_decision_log_payload,
    validate_milestone_report_payload,
)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    return path


def _campaign_artifact_fixture(tmp_path: Path) -> Path:
    campaign_dir = tmp_path / "campaign_artifacts" / "research_campaign_demo"
    review_dir = tmp_path / "reviews" / "research_review"
    candidate_review_dir = tmp_path / "reviews" / "candidate_review"
    candidate_selection_dir = tmp_path / "candidate_selection" / "candidate_selection_demo"
    portfolio_dir = tmp_path / "portfolios" / "portfolio_demo"
    comparison_dir = tmp_path / "comparisons"

    review_summary_path = _write_json(
        review_dir / "review_summary.json",
        {
            "review_id": "review_demo",
            "entry_count": 5,
            "counts_by_run_type": {
                "alpha_evaluation": 3,
                "portfolio": 1,
                "strategy": 1,
            },
        },
    )
    promotion_gates_path = _write_json(
        review_dir / "promotion_gates.json",
        {
            "evaluation_status": "passed",
            "promotion_status": "approved",
            "gate_count": 3,
            "passed_gate_count": 3,
            "failed_gate_count": 0,
            "missing_gate_count": 0,
        },
    )
    _write_json(review_dir / "manifest.json", {"review_id": "review_demo"})
    candidate_review_summary_path = _write_json(
        candidate_review_dir / "candidate_review_summary.json",
        {
            "candidate_selection_run_id": "candidate_selection_demo",
            "portfolio_run_id": "portfolio_demo",
            "total_candidates": 4,
            "selected_candidates": 2,
            "rejected_candidates": 2,
        },
    )
    _write_json(candidate_review_dir / "manifest.json", {"run_id": "candidate_review_demo"})
    _write_json(candidate_selection_dir / "selection_summary.json", {"selected_candidates": 2})
    _write_json(candidate_selection_dir / "manifest.json", {"run_id": "candidate_selection_demo"})
    _write_json(portfolio_dir / "manifest.json", {"run_id": "portfolio_demo"})
    alpha_comparison_path = _write_json(comparison_dir / "alpha" / "summary.json", {"comparison": "alpha"})
    strategy_comparison_path = _write_json(comparison_dir / "strategy" / "summary.json", {"comparison": "strategy"})
    preflight_summary_path = _write_json(tmp_path / "preflight_summary.json", {"status": "passed"})
    checkpoint_path = _write_json(
        campaign_dir / "checkpoint.json",
        {
            "run_type": "research_campaign_checkpoint",
            "schema_version": 2,
            "checkpoint_path": (campaign_dir / "checkpoint.json").as_posix(),
            "stage_states": {
                "preflight": "completed",
                "research": "completed",
                "comparison": "completed",
                "candidate_selection": "completed",
                "portfolio": "completed",
                "candidate_review": "completed",
                "review": "completed",
            },
            "stages": [
                {
                    "stage_name": "preflight",
                    "fingerprint_inputs": {
                        "config": {
                            "time_windows": {
                                "start": "2026-01-01",
                                "end": "2026-03-31",
                            }
                        }
                    },
                }
            ],
        },
    )

    campaign_summary = {
        "run_type": "research_campaign",
        "campaign_run_id": "research_campaign_demo",
        "status": "completed",
        "preflight_status": "passed",
        "stage_statuses": {
            "preflight": "completed",
            "research": "completed",
            "comparison": "completed",
            "candidate_selection": "completed",
            "portfolio": "completed",
            "candidate_review": "completed",
            "review": "completed",
        },
        "stage_state_counts": {
            "completed": 7,
            "failed": 0,
            "partial": 0,
            "pending": 0,
            "reused": 0,
            "skipped": 0,
        },
        "selected_run_ids": {
            "alpha_run_ids": [
                "alpha_demo_a",
                "alpha_demo_b",
            ],
            "strategy_run_ids": ["strategy_demo"],
            "candidate_selection_run_id": "candidate_selection_demo",
            "portfolio_run_id": "portfolio_demo",
            "review_id": "review_demo",
        },
        "targets": {
            "alpha_names": ["alpha_a", "alpha_b"],
            "strategy_names": ["strategy_demo"],
            "portfolio_names": ["portfolio_demo"],
            "candidate_selection_alpha_name": "alpha_a",
            "portfolio_name": "portfolio_demo",
        },
        "key_metrics": {
            "alpha_runs": [
                {"run_id": "alpha_demo_a", "ic_ir": 1.1},
                {"run_id": "alpha_demo_b", "ic_ir": 0.9},
            ],
            "strategy_runs": [{"run_id": "strategy_demo", "sharpe_ratio": 1.4}],
            "candidate_selection": {
                "run_id": "candidate_selection_demo",
                "selected_count": 2,
                "rejected_count": 2,
                "eligible_count": 4,
            },
            "portfolio": {
                "run_id": "portfolio_demo",
                "portfolio_name": "portfolio_demo",
                "sharpe_ratio": 1.7,
                "total_return": 0.14,
                "max_drawdown": -0.05,
            },
            "review": {
                "review_id": "review_demo",
                "entry_count": 5,
            },
        },
        "output_paths": {
            "campaign_artifact_dir": campaign_dir.as_posix(),
            "campaign_checkpoint": checkpoint_path.as_posix(),
            "preflight_summary": preflight_summary_path.as_posix(),
            "campaign_manifest": (campaign_dir / "manifest.json").as_posix(),
            "campaign_summary": (campaign_dir / "summary.json").as_posix(),
            "alpha_comparison_summary": alpha_comparison_path.as_posix(),
            "strategy_comparison_summary": strategy_comparison_path.as_posix(),
            "candidate_selection_summary": (candidate_selection_dir / "selection_summary.json").as_posix(),
            "candidate_selection_manifest": (candidate_selection_dir / "manifest.json").as_posix(),
            "portfolio_artifact_dir": portfolio_dir.as_posix(),
            "candidate_review_summary": candidate_review_summary_path.as_posix(),
            "candidate_review_manifest": (candidate_review_dir / "manifest.json").as_posix(),
            "review_summary": review_summary_path.as_posix(),
            "review_manifest": (review_dir / "manifest.json").as_posix(),
            "review_promotion_gates": promotion_gates_path.as_posix(),
        },
        "final_outcomes": {
            "failed_stage_names": [],
            "partial_stage_names": [],
            "resumable_stage_names": [],
            "review_promotion_gate_summary": json.loads(promotion_gates_path.read_text(encoding="utf-8")),
            "candidate_review_counts": json.loads(candidate_review_summary_path.read_text(encoding="utf-8")),
        },
        "checkpoint": json.loads(checkpoint_path.read_text(encoding="utf-8")),
        "stages": [
            {
                "stage_name": "candidate_selection",
                "state": "completed",
                "status": "completed",
                "state_reason": "Completed successfully.",
                "selected_run_ids": {"candidate_selection_run_id": "candidate_selection_demo"},
                "key_metrics": {"selected_count": 2},
                "outcomes": {},
                "output_paths": {
                    "summary_json": (candidate_selection_dir / "selection_summary.json").as_posix(),
                    "manifest_json": (candidate_selection_dir / "manifest.json").as_posix(),
                },
            },
            {
                "stage_name": "review",
                "state": "completed",
                "status": "completed",
                "state_reason": "Completed successfully.",
                "selected_run_ids": {"review_id": "review_demo"},
                "key_metrics": {"entry_count": 5},
                "outcomes": {},
                "output_paths": {
                    "summary_json": review_summary_path.as_posix(),
                    "promotion_gates_json": promotion_gates_path.as_posix(),
                },
            },
        ],
    }
    _write_json(campaign_dir / "summary.json", campaign_summary)
    _write_json(
        campaign_dir / "manifest.json",
        {
            "run_type": "research_campaign",
            "campaign_run_id": "research_campaign_demo",
            "summary_path": "summary.json",
            "checkpoint_path": "checkpoint.json",
            "artifact_files": [
                "campaign_config.json",
                "checkpoint.json",
                "manifest.json",
                "preflight_summary.json",
                "summary.json",
            ],
        },
    )
    return campaign_dir


def test_generate_campaign_milestone_report_builds_milestone_pack_from_completed_campaign_artifacts(
    tmp_path: Path,
) -> None:
    campaign_dir = _campaign_artifact_fixture(tmp_path)

    summary_path, decision_log_path, manifest_path = generate_campaign_milestone_report(campaign_dir)
    markdown_report_path = campaign_dir / "milestone_report" / MILESTONE_MARKDOWN_REPORT_FILENAME

    assert summary_path == campaign_dir / "milestone_report" / MILESTONE_SUMMARY_FILENAME
    assert decision_log_path == campaign_dir / "milestone_report" / MILESTONE_DECISION_LOG_FILENAME
    assert manifest_path == campaign_dir / "milestone_report" / MILESTONE_MANIFEST_FILENAME
    assert markdown_report_path.exists()

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    decision_log_payload = json.loads(decision_log_path.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    markdown_report = markdown_report_path.read_text(encoding="utf-8")

    validate_milestone_report_payload(summary_payload)
    validate_milestone_decision_log_payload(decision_log_payload)

    assert summary_payload["milestone_name"] == "research_campaign_demo"
    assert summary_payload["title"] == "Milestone Report for research_campaign_demo"
    assert summary_payload["report_markdown_path"] == "report.md"
    assert summary_payload["reporting_window"] == {
        "start": "2026-01-01",
        "end": "2026-03-31",
    }
    assert summary_payload["related_artifacts"]["campaign_summary"] == "../summary.json"
    assert summary_payload["related_artifacts"]["review_promotion_gates"] == "../../../reviews/research_review/promotion_gates.json"
    assert "alpha:alpha_a" in summary_payload["scope"]
    assert "portfolio:portfolio_demo" in summary_payload["scope"]
    assert any("Stage outcomes: completed=7" in finding for finding in summary_payload["key_findings"])
    assert any("promotion_status=approved" in finding for finding in summary_payload["key_findings"])
    assert summary_payload["decision_counts_by_status"] == {"accepted": 2}

    assert [row["decision_id"] for row in decision_log_payload["decisions"]] == [
        "campaign_execution",
        "review_promotion_outcome",
    ]
    assert decision_log_payload["decisions"][1]["status"] == "accepted"
    assert decision_log_payload["decisions"][0]["follow_up_actions"] == [
        "No additional campaign execution follow-up actions were required."
    ]
    assert decision_log_payload["decisions"][1]["follow_up_actions"] == [
        "Prepare the approved reviewed outputs for promotion handoff."
    ]
    assert decision_log_payload["decisions"][1]["source_artifacts"] == [
        {
            "artifact_id": "review_summary",
            "label": "review summary",
            "path": "../../../reviews/research_review/review_summary.json",
            "role": "source_artifact",
        },
        {
            "artifact_id": "review_manifest",
            "label": "review manifest",
            "path": "../../../reviews/research_review/manifest.json",
            "role": "source_artifact",
        },
        {
            "artifact_id": "review_promotion_gates",
            "label": "review promotion gates",
            "path": "../../../reviews/research_review/promotion_gates.json",
            "role": "source_artifact",
        },
    ]
    assert "## 2. Review and promotion outcome" in decision_log_payload["rendered"]["markdown"]
    assert manifest_payload["artifacts"]["summary.json"]["decision_count"] == 2
    assert manifest_payload["artifacts"]["report.md"]["path"] == "report.md"
    assert "## Campaign Scope" in markdown_report
    assert "- Alpha Runs: alpha_demo_a, alpha_demo_b" in markdown_report
    assert "- Portfolio: max_drawdown=-0.05; portfolio_name=portfolio_demo; run_id=portfolio_demo; sharpe_ratio=1.7; total_return=0.14" in markdown_report
    assert "- Promotion Gates: evaluation_status=passed; failed_gate_count=0; gate_count=3; missing_gate_count=0; passed_gate_count=3; promotion_status=approved" in markdown_report
    assert "## Risks" in markdown_report
    assert "- No immediate risks were detected." in markdown_report
    assert "## Next Steps" in markdown_report
    assert "- Promotion gates passed; the selected outputs are ready for promotion handoff." in markdown_report

    metadata = summary_payload["metadata"]
    assert metadata["selected_run_ids"]["portfolio_run_id"] == "portfolio_demo"
    assert metadata["key_metrics"]["portfolio"]["sharpe_ratio"] == 1.7
    assert metadata["promotion_gates"]["promotion_status"] == "approved"
    assert metadata["stage_outcomes"][0]["output_paths"]["manifest_json"] == "../../../candidate_selection/candidate_selection_demo/manifest.json"


def test_generate_campaign_milestone_report_is_deterministic_for_identical_campaign_inputs(tmp_path: Path) -> None:
    campaign_dir = _campaign_artifact_fixture(tmp_path)

    first_paths = generate_campaign_milestone_report(campaign_dir)
    first_report_bytes = (campaign_dir / "milestone_report" / MILESTONE_MARKDOWN_REPORT_FILENAME).read_bytes()
    first_bytes = tuple(path.read_bytes() for path in first_paths) + (first_report_bytes,)
    second_paths = generate_campaign_milestone_report(campaign_dir)
    second_report_bytes = (campaign_dir / "milestone_report" / MILESTONE_MARKDOWN_REPORT_FILENAME).read_bytes()
    second_bytes = tuple(path.read_bytes() for path in second_paths) + (second_report_bytes,)

    assert first_paths == second_paths
    assert first_bytes == second_bytes


def test_campaign_milestone_helpers_accept_campaign_summary_path(tmp_path: Path) -> None:
    campaign_dir = _campaign_artifact_fixture(tmp_path)
    summary_json = campaign_dir / "summary.json"

    assert resolve_campaign_artifact_dir(summary_json) == campaign_dir

    payloads = load_campaign_reporting_payloads(summary_json)

    assert payloads["campaign_summary"]["campaign_run_id"] == "research_campaign_demo"
    assert payloads["review_summary"]["review_id"] == "review_demo"
    assert payloads["promotion_gates"]["promotion_status"] == "approved"
