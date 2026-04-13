from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.reporting import (
    MILESTONE_DECISION_LOG_FILENAME,
    MILESTONE_MARKDOWN_REPORT_FILENAME,
    MILESTONE_MANIFEST_FILENAME,
    MILESTONE_SUMMARY_FILENAME,
    MilestoneArtifactValidationError,
    MilestoneDecisionEntry,
    MilestoneReport,
    MilestoneSourceArtifact,
    build_milestone_report_id,
    resolve_milestone_artifact_dir,
    validate_milestone_decision_log_payload,
    validate_milestone_report_payload,
    write_milestone_report_artifacts,
)


def _sample_report() -> MilestoneReport:
    return MilestoneReport(
        milestone_id="milestone_report_deadbeefcafe",
        milestone_name="Milestone 18",
        title="Milestone 18 Readiness Report",
        status="final",
        summary="The milestone is ready for review with two follow-up questions.",
        owner="research-platform",
        reporting_window={"start": "2026-04-01", "end": "2026-04-13"},
        scope=["campaign orchestration", "artifact contracts"],
        key_findings=[
            "Checkpoint reuse is deterministic across repeated runs.",
            "Stage-local manifests remain the source of truth for downstream inspection.",
        ],
        recommendations=[
            "Adopt the artifact contract for milestone case studies.",
            "Keep decision evidence paths relative to the report root.",
        ],
        open_questions=["Should markdown rendering remain optional for future report packs?"],
        related_artifacts={"campaign_summary": "artifacts/research_campaigns/demo/summary.json"},
        metadata={"issue": 203},
    )


def _sample_decisions() -> list[MilestoneDecisionEntry]:
    return [
        MilestoneDecisionEntry(
            decision_id="reuse_summary_contract",
            title="Reuse summary and manifest naming",
            status="accepted",
            summary="Milestone report packs should continue to use summary.json and manifest.json.",
            rationale="This keeps milestone reporting aligned with campaign and review artifacts.",
            impact="Consumers can reuse existing summary-first loading patterns.",
            owner="research-platform",
            category="artifact_contract",
            timestamp="2026-04-13T12:00:00Z",
            follow_up_actions=(
                "Document the shared naming contract for milestone packs.",
                "Keep summary.json as the default report entry point.",
            ),
            evidence_artifacts=("docs/milestone_16_campaign_workflow.md", "artifacts/research_campaigns/demo/manifest.json"),
            source_artifacts=(
                MilestoneSourceArtifact(
                    artifact_id="workflow_doc",
                    path="docs/milestone_16_campaign_workflow.md",
                    label="workflow doc",
                    role="documentation",
                ),
                MilestoneSourceArtifact(
                    artifact_id="campaign_manifest",
                    path="artifacts/research_campaigns/demo/manifest.json",
                    label="campaign manifest",
                    role="artifact",
                ),
            ),
            related_stage_names=("review", "campaign"),
            tags=("naming", "compatibility"),
            metadata={"schema_alignment": "existing_manifest_summary_patterns"},
        ),
        MilestoneDecisionEntry(
            decision_id="keep_decision_log_separate",
            title="Store decision log as a separate artifact",
            status="accepted",
            summary="Decision details should live in a dedicated JSON artifact.",
            rationale="This keeps the report summary compact while preserving auditable decision detail.",
            follow_up_actions=("Keep the dedicated decision log payload compact but complete.",),
            evidence_artifacts=("docs/backfilled_2026_q1_research_workflow.md",),
            source_artifacts=(
                MilestoneSourceArtifact(
                    artifact_id="backfill_workflow_doc",
                    path="docs/backfilled_2026_q1_research_workflow.md",
                    label="backfill workflow doc",
                    role="documentation",
                ),
            ),
            tags=("audit", "decision_log"),
        ),
    ]


def test_build_milestone_report_id_is_deterministic() -> None:
    first = build_milestone_report_id(
        milestone_name="Milestone 18",
        title="Milestone 18 Readiness Report",
        reporting_window={"start": "2026-04-01", "end": "2026-04-13"},
        scope=["campaign orchestration", "artifact contracts"],
    )
    second = build_milestone_report_id(
        milestone_name="Milestone 18",
        title="Milestone 18 Readiness Report",
        reporting_window={"end": "2026-04-13", "start": "2026-04-01"},
        scope=["campaign orchestration", "artifact contracts"],
    )

    assert first == second
    assert first.startswith("milestone_report_")


def test_resolve_milestone_artifact_dir_accepts_directory_or_summary_json(tmp_path: Path) -> None:
    assert resolve_milestone_artifact_dir(tmp_path / "artifact_pack") == tmp_path / "artifact_pack"
    assert (
        resolve_milestone_artifact_dir(tmp_path / "artifact_pack" / MILESTONE_SUMMARY_FILENAME)
        == tmp_path / "artifact_pack"
    )

    with pytest.raises(MilestoneArtifactValidationError, match="summary.json"):
        resolve_milestone_artifact_dir(tmp_path / "artifact_pack" / "custom_report.json")


def test_write_milestone_report_artifacts_writes_summary_decision_log_and_manifest(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "milestone_18"
    summary_path, decision_log_path, manifest_path = write_milestone_report_artifacts(
        report=_sample_report(),
        decisions=_sample_decisions(),
        output_path=artifact_dir,
    )
    markdown_report_path = artifact_dir / MILESTONE_MARKDOWN_REPORT_FILENAME

    assert summary_path == artifact_dir / MILESTONE_SUMMARY_FILENAME
    assert decision_log_path == artifact_dir / MILESTONE_DECISION_LOG_FILENAME
    assert manifest_path == artifact_dir / MILESTONE_MANIFEST_FILENAME
    assert markdown_report_path == artifact_dir / "report.md"

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    decision_log_payload = json.loads(decision_log_path.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    markdown_report = markdown_report_path.read_text(encoding="utf-8")

    validate_milestone_report_payload(summary_payload)
    validate_milestone_decision_log_payload(decision_log_payload)

    assert summary_payload["decision_log_path"] == MILESTONE_DECISION_LOG_FILENAME
    assert summary_payload["report_markdown_path"] == MILESTONE_MARKDOWN_REPORT_FILENAME
    assert summary_payload["decision_counts_by_status"] == {"accepted": 2}
    assert summary_payload["decision_ids"] == [
        "reuse_summary_contract",
        "keep_decision_log_separate",
    ]

    assert decision_log_payload["decision_count"] == 2
    assert [row["decision_id"] for row in decision_log_payload["decisions"]] == [
        "reuse_summary_contract",
        "keep_decision_log_separate",
    ]
    assert decision_log_payload["decisions"][0]["follow_up_actions"] == [
        "Document the shared naming contract for milestone packs.",
        "Keep summary.json as the default report entry point.",
    ]
    assert decision_log_payload["decisions"][0]["source_artifacts"] == [
        {
            "artifact_id": "workflow_doc",
            "label": "workflow doc",
            "path": "docs/milestone_16_campaign_workflow.md",
            "role": "documentation",
        },
        {
            "artifact_id": "campaign_manifest",
            "label": "campaign manifest",
            "path": "artifacts/research_campaigns/demo/manifest.json",
            "role": "artifact",
        },
    ]
    assert "# Milestone 18 Readiness Report" in decision_log_payload["rendered"]["markdown"]
    assert "Linked Source Artifacts:" in decision_log_payload["rendered"]["text"]
    assert "# Milestone 18 Readiness Report" in markdown_report
    assert "## Campaign Scope" in markdown_report
    assert "- Strategy Runs: No run selections were recorded." not in markdown_report
    assert "- Summary: The milestone is ready for review with two follow-up questions." in markdown_report
    assert "- Risks: No immediate risks were detected." not in markdown_report
    assert "## Decision Snapshot" in markdown_report
    assert "### 1. Reuse summary and manifest naming" in markdown_report

    assert manifest_payload["artifact_files"] == [
        "decision_log.json",
        "manifest.json",
        "report.md",
        "summary.json",
    ]
    assert manifest_payload["summary_path"] == MILESTONE_SUMMARY_FILENAME
    assert manifest_payload["decision_log_path"] == MILESTONE_DECISION_LOG_FILENAME
    assert manifest_payload["report_markdown_path"] == MILESTONE_MARKDOWN_REPORT_FILENAME
    assert manifest_payload["artifacts"]["summary.json"]["finding_count"] == 2
    assert manifest_payload["artifacts"]["decision_log.json"]["decision_count"] == 2
    assert manifest_payload["artifacts"]["report.md"]["path"] == "report.md"


def test_write_milestone_report_artifacts_rejects_duplicate_decision_ids_and_absolute_evidence_paths(
    tmp_path: Path,
) -> None:
    duplicate_decisions = [
        MilestoneDecisionEntry(
            decision_id="duplicate",
            title="Decision A",
            status="accepted",
            summary="A",
            rationale="A",
        ),
        MilestoneDecisionEntry(
            decision_id="duplicate",
            title="Decision B",
            status="accepted",
            summary="B",
            rationale="B",
        ),
    ]

    with pytest.raises(MilestoneArtifactValidationError, match="Duplicate decision_id"):
        write_milestone_report_artifacts(
            report=_sample_report(),
            decisions=duplicate_decisions,
            output_path=tmp_path / "duplicate_ids",
        )

    invalid_follow_up_actions = [
        MilestoneDecisionEntry(
            decision_id="invalid_follow_up",
            title="Invalid follow-up action",
            status="accepted",
            summary="A",
            rationale="A",
            follow_up_actions=("",),
        )
    ]

    with pytest.raises(MilestoneArtifactValidationError, match="follow_up_actions"):
        write_milestone_report_artifacts(
            report=_sample_report(),
            decisions=invalid_follow_up_actions,
            output_path=tmp_path / "invalid_follow_up_actions",
        )

    absolute_path_decisions = [
        MilestoneDecisionEntry(
            decision_id="absolute_evidence",
            title="Absolute evidence path",
            status="accepted",
            summary="A",
            rationale="A",
            evidence_artifacts=(str((tmp_path / "evidence.json").resolve()),),
        )
    ]

    with pytest.raises(MilestoneArtifactValidationError, match="must be relative"):
        write_milestone_report_artifacts(
            report=_sample_report(),
            decisions=absolute_path_decisions,
            output_path=tmp_path / "absolute_paths",
        )


def test_write_milestone_report_artifacts_supports_generation_controls(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "controlled_milestone"
    summary_path, decision_log_path, manifest_path = write_milestone_report_artifacts(
        report=_sample_report(),
        decisions=_sample_decisions(),
        output_path=artifact_dir,
        options={
            "include_markdown_report": False,
            "decision_log_render_formats": ["text"],
            "decision_categories": ["artifact_contract"],
            "sections": {
                "related_artifacts": False,
                "decision_snapshot": False,
            },
        },
    )

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    decision_log_payload = json.loads(decision_log_path.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary_payload["report_markdown_path"] is None
    assert summary_payload["decision_count"] == 1
    assert summary_payload["decision_ids"] == ["reuse_summary_contract"]
    assert decision_log_payload["decision_ids"] == ["reuse_summary_contract"]
    assert sorted(decision_log_payload["rendered"]) == ["text"]
    assert "markdown" not in decision_log_payload["rendered"]
    assert "report.md" not in manifest_payload["artifact_files"]
    assert "markdown" not in manifest_payload["artifact_groups"]
    assert not (artifact_dir / MILESTONE_MARKDOWN_REPORT_FILENAME).exists()
