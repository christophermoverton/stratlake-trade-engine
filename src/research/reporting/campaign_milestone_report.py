from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.research.reporting.milestone_artifacts import (
    MilestoneDecisionEntry,
    MilestoneReport,
    MilestoneSourceArtifact,
    build_milestone_report_id,
    resolve_milestone_generation_options,
    resolve_milestone_artifact_dir,
    write_milestone_report_artifacts,
)

DEFAULT_CAMPAIGN_MILESTONE_DIRNAME = "milestone_report"


def generate_campaign_milestone_report(
    campaign_artifact_path: Path | str,
    *,
    output_path: Path | str | None = None,
    milestone_name: str | None = None,
    title: str | None = None,
    owner: str | None = None,
    status: str = "final",
    options: Mapping[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    """Generate a milestone report pack from completed campaign artifacts."""

    campaign_artifact_dir = resolve_campaign_artifact_dir(campaign_artifact_path)
    payloads = load_campaign_reporting_payloads(campaign_artifact_dir)
    resolved_output_path = (
        campaign_artifact_dir / DEFAULT_CAMPAIGN_MILESTONE_DIRNAME
        if output_path is None
        else Path(output_path)
    )
    artifact_dir = resolve_milestone_artifact_dir(resolved_output_path)
    report, decisions = build_campaign_milestone_report_payloads(
        campaign_artifact_dir=campaign_artifact_dir,
        artifact_dir=artifact_dir,
        campaign_summary=payloads["campaign_summary"],
        campaign_manifest=payloads["campaign_manifest"],
        review_summary=payloads["review_summary"],
        promotion_gates=payloads["promotion_gates"],
        candidate_review_summary=payloads["candidate_review_summary"],
        milestone_name=milestone_name,
        title=title,
        owner=owner,
        status=status,
        options=options,
    )
    return write_milestone_report_artifacts(
        report=report,
        decisions=decisions,
        output_path=artifact_dir,
        options=options,
    )


def build_campaign_milestone_report_payloads(
    *,
    campaign_artifact_dir: Path,
    artifact_dir: Path,
    campaign_summary: Mapping[str, Any],
    campaign_manifest: Mapping[str, Any] | None,
    review_summary: Mapping[str, Any] | None,
    promotion_gates: Mapping[str, Any] | None,
    candidate_review_summary: Mapping[str, Any] | None,
    milestone_name: str | None,
    title: str | None,
    owner: str | None,
    status: str,
    options: Mapping[str, Any] | None,
) -> tuple[MilestoneReport, list[MilestoneDecisionEntry]]:
    resolved_options = resolve_milestone_generation_options(options)
    run_type = str(campaign_summary.get("run_type") or "")
    if run_type != "research_campaign":
        raise ValueError(
            "Campaign milestone reporting requires a research campaign summary.json artifact, "
            f"observed run_type={run_type!r}."
        )

    campaign_run_id = str(campaign_summary.get("campaign_run_id") or campaign_artifact_dir.name)
    resolved_milestone_name = milestone_name or campaign_run_id
    resolved_title = title or f"Milestone Report for {campaign_run_id}"
    reporting_window = _reporting_window(campaign_summary)
    scope = _campaign_scope(campaign_summary)
    related_artifacts = _related_artifacts(
        artifact_dir=artifact_dir,
        campaign_artifact_dir=campaign_artifact_dir,
        campaign_summary=campaign_summary,
        campaign_manifest=campaign_manifest,
    )

    report = MilestoneReport(
        milestone_id=build_milestone_report_id(
            milestone_name=resolved_milestone_name,
            title=resolved_title,
            reporting_window=reporting_window,
            scope=scope,
        ),
        milestone_name=resolved_milestone_name,
        title=resolved_title,
        status=status,
        summary=_campaign_summary_text(
            campaign_summary=campaign_summary,
            promotion_gates=promotion_gates,
            review_summary=review_summary,
            summary_options=resolved_options["summary"],
        ),
        owner=owner,
        reporting_window=reporting_window,
        scope=scope,
        key_findings=_key_findings(
            campaign_summary=campaign_summary,
            candidate_review_summary=candidate_review_summary,
            review_summary=review_summary,
            promotion_gates=promotion_gates,
        ),
        recommendations=_recommendations(
            campaign_summary=campaign_summary,
            promotion_gates=promotion_gates,
        ),
        open_questions=_open_questions(
            campaign_summary=campaign_summary,
            review_summary=review_summary,
            promotion_gates=promotion_gates,
        ),
        related_artifacts=related_artifacts,
        metadata=_metadata(
            artifact_dir=artifact_dir,
            campaign_summary=campaign_summary,
            campaign_manifest=campaign_manifest,
            candidate_review_summary=candidate_review_summary,
            review_summary=review_summary,
            promotion_gates=promotion_gates,
        ),
    )
    decisions = _decision_entries(
        campaign_summary=campaign_summary,
        related_artifacts=related_artifacts,
        review_summary=review_summary,
        promotion_gates=promotion_gates,
    )
    if resolved_options["decision_categories"]:
        allowed = set(resolved_options["decision_categories"])
        decisions = [decision for decision in decisions if decision.category in allowed]
    return report, decisions


def load_campaign_reporting_payloads(campaign_artifact_dir: Path | str) -> dict[str, Any]:
    """Load the campaign summary and referenced downstream review artifacts."""

    root = resolve_campaign_artifact_dir(campaign_artifact_dir)
    campaign_summary = _require_json(root / "summary.json")
    campaign_manifest = _load_json_if_exists(root / "manifest.json")

    output_paths = campaign_summary.get("output_paths")
    if not isinstance(output_paths, Mapping):
        output_paths = {}

    review_summary = _load_mapped_json(output_paths.get("review_summary"))
    promotion_gates = _load_mapped_json(output_paths.get("review_promotion_gates"))
    candidate_review_summary = _load_mapped_json(output_paths.get("candidate_review_summary"))
    return {
        "campaign_summary": campaign_summary,
        "campaign_manifest": campaign_manifest,
        "review_summary": review_summary,
        "promotion_gates": promotion_gates,
        "candidate_review_summary": candidate_review_summary,
    }


def resolve_campaign_artifact_dir(campaign_artifact_path: Path | str) -> Path:
    """Resolve a campaign artifact directory from a directory or summary/manifest path."""

    resolved = Path(campaign_artifact_path)
    if resolved.is_dir():
        return resolved
    if resolved.name in {"summary.json", "manifest.json", "checkpoint.json"}:
        return resolved.parent
    raise ValueError(
        "campaign_artifact_path must point to a campaign artifact directory or a "
        "summary.json/manifest.json/checkpoint.json file."
    )


def _campaign_summary_text(
    *,
    campaign_summary: Mapping[str, Any],
    promotion_gates: Mapping[str, Any] | None,
    review_summary: Mapping[str, Any] | None,
    summary_options: Mapping[str, Any],
) -> str:
    campaign_run_id = str(campaign_summary.get("campaign_run_id") or "unknown_campaign")
    stage_counts = campaign_summary.get("stage_state_counts")
    if not isinstance(stage_counts, Mapping):
        stage_counts = {}
    tracked_stage_count = int(sum(_safe_int(value) for value in stage_counts.values()))
    completed_like_count = _safe_int(stage_counts.get("completed")) + _safe_int(stage_counts.get("reused"))
    failed_like_count = _safe_int(stage_counts.get("failed")) + _safe_int(stage_counts.get("partial"))

    parts = [f"Campaign `{campaign_run_id}` finished with status `{campaign_summary.get('status', 'unknown')}`."]
    if bool(summary_options.get("include_stage_counts", True)):
        parts.append(f"{completed_like_count} of {tracked_stage_count} tracked stages completed or were reused.")
    if failed_like_count:
        parts.append(f"{failed_like_count} stage(s) require follow-up.")

    review_fragment = None
    if bool(summary_options.get("include_review_outcome", True)):
        review_fragment = _review_result_fragment(review_summary=review_summary, promotion_gates=promotion_gates)
    if review_fragment is not None:
        parts.append(review_fragment)
    return " ".join(parts)


def _campaign_scope(campaign_summary: Mapping[str, Any]) -> list[str]:
    targets = campaign_summary.get("targets")
    if not isinstance(targets, Mapping):
        return []

    scope: list[str] = []
    for prefix, key in (
        ("alpha", "alpha_names"),
        ("strategy", "strategy_names"),
        ("portfolio", "portfolio_names"),
    ):
        values = targets.get(key)
        if isinstance(values, list):
            scope.extend(f"{prefix}:{value}" for value in values if str(value).strip())

    candidate_selection_alpha = targets.get("candidate_selection_alpha_name")
    if candidate_selection_alpha is not None and str(candidate_selection_alpha).strip():
        scope.append(f"candidate_selection_alpha:{candidate_selection_alpha}")
    return sorted(scope)


def _key_findings(
    *,
    campaign_summary: Mapping[str, Any],
    candidate_review_summary: Mapping[str, Any] | None,
    review_summary: Mapping[str, Any] | None,
    promotion_gates: Mapping[str, Any] | None,
) -> list[str]:
    findings: list[str] = []
    selected_run_ids = campaign_summary.get("selected_run_ids")
    if isinstance(selected_run_ids, Mapping):
        alpha_run_ids = selected_run_ids.get("alpha_run_ids") if isinstance(selected_run_ids.get("alpha_run_ids"), list) else []
        strategy_run_ids = (
            selected_run_ids.get("strategy_run_ids")
            if isinstance(selected_run_ids.get("strategy_run_ids"), list)
            else []
        )
        findings.append(
            "Research scope resolved "
            f"{len(alpha_run_ids)} alpha run(s), {len(strategy_run_ids)} strategy run(s), "
            f"candidate_selection={selected_run_ids.get('candidate_selection_run_id') or 'NA'}, "
            f"portfolio={selected_run_ids.get('portfolio_run_id') or 'NA'}."
        )

    stage_counts = campaign_summary.get("stage_state_counts")
    if isinstance(stage_counts, Mapping):
        findings.append(
            "Stage outcomes: "
            + ", ".join(
                f"{name}={_safe_int(stage_counts[name])}"
                for name in sorted(stage_counts)
            )
            + "."
        )

    portfolio_metrics = _mapping(campaign_summary.get("key_metrics")).get("portfolio")
    if isinstance(portfolio_metrics, Mapping):
        metrics_text: list[str] = []
        for label, key in (("Sharpe", "sharpe_ratio"), ("Total Return", "total_return"), ("Max Drawdown", "max_drawdown")):
            value = portfolio_metrics.get(key)
            if value is None:
                continue
            metrics_text.append(f"{label}={value}")
        if metrics_text:
            findings.append("Portfolio metrics: " + ", ".join(metrics_text) + ".")

    if isinstance(candidate_review_summary, Mapping):
        findings.append(
            "Candidate review evaluated "
            f"{_safe_int(candidate_review_summary.get('total_candidates'))} candidate(s) with "
            f"{_safe_int(candidate_review_summary.get('selected_candidates'))} selected and "
            f"{_safe_int(candidate_review_summary.get('rejected_candidates'))} rejected."
        )
    elif isinstance(_mapping(campaign_summary.get("final_outcomes")).get("candidate_review_counts"), Mapping):
        counts = _mapping(campaign_summary.get("final_outcomes")).get("candidate_review_counts")
        findings.append(
            "Candidate review evaluated "
            f"{_safe_int(counts.get('total_candidates'))} candidate(s) with "
            f"{_safe_int(counts.get('selected_candidates'))} selected and "
            f"{_safe_int(counts.get('rejected_candidates'))} rejected."
        )

    review_fragment = _review_result_fragment(review_summary=review_summary, promotion_gates=promotion_gates)
    if review_fragment is not None:
        findings.append(review_fragment)

    return findings


def _recommendations(
    *,
    campaign_summary: Mapping[str, Any],
    promotion_gates: Mapping[str, Any] | None,
) -> list[str]:
    final_outcomes = _mapping(campaign_summary.get("final_outcomes"))
    failed_stage_names = final_outcomes.get("failed_stage_names")
    partial_stage_names = final_outcomes.get("partial_stage_names")
    resumable_stage_names = final_outcomes.get("resumable_stage_names")
    recommendations: list[str] = []

    if isinstance(failed_stage_names, list) and failed_stage_names:
        recommendations.append(
            "Resolve failed stages before treating the milestone as promotion-ready: "
            + ", ".join(sorted(str(name) for name in failed_stage_names))
            + "."
        )
    if isinstance(partial_stage_names, list) and partial_stage_names:
        recommendations.append(
            "Review partial stage outputs before downstream reuse: "
            + ", ".join(sorted(str(name) for name in partial_stage_names))
            + "."
        )
    if isinstance(resumable_stage_names, list) and resumable_stage_names:
        recommendations.append(
            "Preserve checkpoint continuity for resumable stages: "
            + ", ".join(sorted(str(name) for name in resumable_stage_names))
            + "."
        )

    promotion_status = None if promotion_gates is None else str(promotion_gates.get("promotion_status") or "").strip()
    evaluation_status = None if promotion_gates is None else str(promotion_gates.get("evaluation_status") or "").strip()
    if promotion_status == "approved":
        recommendations.append("Promotion gates passed; the selected outputs are ready for promotion handoff.")
    elif promotion_status in {"review_ready", "pending", "deferred"} or evaluation_status in {"warn", "pending"}:
        recommendations.append("Complete review follow-ups before promotion to keep milestone decisions auditable.")
    elif promotion_status in {"blocked", "rejected"} or evaluation_status in {"failed", "blocked"}:
        recommendations.append("Address blocked promotion gates before promoting the reviewed campaign outputs.")

    if not recommendations:
        recommendations.append("No blocking follow-up actions were detected in the completed campaign artifacts.")
    return recommendations


def _open_questions(
    *,
    campaign_summary: Mapping[str, Any],
    review_summary: Mapping[str, Any] | None,
    promotion_gates: Mapping[str, Any] | None,
) -> list[str]:
    questions: list[str] = []
    if review_summary is None:
        questions.append("Should a research review summary be attached to future campaign milestones?")
    if promotion_gates is None and _mapping(campaign_summary.get("selected_run_ids")).get("review_id") is not None:
        questions.append("Should promotion gates be persisted for reviewed campaigns that stop at review readiness?")
    return questions


def _related_artifacts(
    *,
    artifact_dir: Path,
    campaign_artifact_dir: Path,
    campaign_summary: Mapping[str, Any],
    campaign_manifest: Mapping[str, Any] | None,
) -> dict[str, str]:
    related: dict[str, str] = {}
    output_paths = campaign_summary.get("output_paths")
    if not isinstance(output_paths, Mapping):
        output_paths = {}

    path_candidates = {
        "campaign_summary": campaign_artifact_dir / "summary.json",
        "campaign_manifest": campaign_artifact_dir / "manifest.json",
        "campaign_checkpoint": campaign_artifact_dir / "checkpoint.json",
        "preflight_summary": output_paths.get("preflight_summary"),
        "alpha_comparison_summary": output_paths.get("alpha_comparison_summary"),
        "strategy_comparison_summary": output_paths.get("strategy_comparison_summary"),
        "candidate_selection_summary": output_paths.get("candidate_selection_summary"),
        "candidate_selection_manifest": output_paths.get("candidate_selection_manifest"),
        "candidate_review_summary": output_paths.get("candidate_review_summary"),
        "candidate_review_manifest": output_paths.get("candidate_review_manifest"),
        "review_summary": output_paths.get("review_summary"),
        "review_manifest": output_paths.get("review_manifest"),
        "review_promotion_gates": output_paths.get("review_promotion_gates"),
    }

    if isinstance(campaign_manifest, Mapping):
        summary_path = campaign_manifest.get("summary_path")
        if isinstance(summary_path, str) and summary_path.strip():
            path_candidates.setdefault("campaign_manifest_summary_path", campaign_artifact_dir / summary_path)

    for label in sorted(path_candidates):
        candidate = path_candidates[label]
        if candidate is None:
            continue
        resolved = Path(candidate)
        if not resolved.exists():
            continue
        related[label] = _relative_path(artifact_dir, resolved)
    return related


def _metadata(
    *,
    artifact_dir: Path,
    campaign_summary: Mapping[str, Any],
    campaign_manifest: Mapping[str, Any] | None,
    candidate_review_summary: Mapping[str, Any] | None,
    review_summary: Mapping[str, Any] | None,
    promotion_gates: Mapping[str, Any] | None,
) -> dict[str, Any]:
    stage_summaries = []
    stages = campaign_summary.get("stages")
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, Mapping):
                continue
            stage_summaries.append(
                {
                    "stage_name": stage.get("stage_name"),
                    "state": stage.get("state"),
                    "status": stage.get("status"),
                    "state_reason": stage.get("state_reason"),
                    "selected_run_ids": _mapping(stage.get("selected_run_ids")),
                    "key_metrics": _mapping(stage.get("key_metrics")),
                    "outcomes": _mapping(stage.get("outcomes")),
                    "output_paths": {
                        key: _relative_path(artifact_dir, Path(value))
                        for key, value in sorted(_mapping(stage.get("output_paths")).items())
                        if isinstance(value, str) and value.strip() and Path(value).exists()
                    },
                }
            )

    metadata: dict[str, Any] = {
        "source_run_type": campaign_summary.get("run_type"),
        "source_campaign_run_id": campaign_summary.get("campaign_run_id"),
        "source_campaign_status": campaign_summary.get("status"),
        "preflight_status": campaign_summary.get("preflight_status"),
        "stage_statuses": _mapping(campaign_summary.get("stage_statuses")),
        "stage_state_counts": _mapping(campaign_summary.get("stage_state_counts")),
        "selected_run_ids": _mapping(campaign_summary.get("selected_run_ids")),
        "key_metrics": _mapping(campaign_summary.get("key_metrics")),
        "final_outcomes": _mapping(campaign_summary.get("final_outcomes")),
        "stage_outcomes": stage_summaries,
    }
    if campaign_manifest is not None:
        metadata["campaign_manifest"] = dict(campaign_manifest)
    if candidate_review_summary is not None:
        metadata["candidate_review_summary"] = dict(candidate_review_summary)
    if review_summary is not None:
        metadata["review_summary"] = dict(review_summary)
    if promotion_gates is not None:
        metadata["promotion_gates"] = dict(promotion_gates)
    return metadata


def _decision_entries(
    *,
    campaign_summary: Mapping[str, Any],
    related_artifacts: Mapping[str, str],
    review_summary: Mapping[str, Any] | None,
    promotion_gates: Mapping[str, Any] | None,
) -> list[MilestoneDecisionEntry]:
    decisions = [
        MilestoneDecisionEntry(
            decision_id="campaign_execution",
            title="Campaign execution outcome",
            status=_campaign_execution_decision_status(campaign_summary),
            summary=_campaign_execution_decision_summary(campaign_summary),
            rationale=(
                "Stage state counts, selected run IDs, and checkpoint-backed campaign outputs "
                "provide the deterministic source of truth for milestone execution status."
            ),
            category="campaign_execution",
            follow_up_actions=tuple(_campaign_execution_follow_up_actions(campaign_summary)),
            evidence_artifacts=tuple(
                path
                for key, path in (
                    ("campaign_summary", related_artifacts.get("campaign_summary")),
                    ("campaign_manifest", related_artifacts.get("campaign_manifest")),
                    ("campaign_checkpoint", related_artifacts.get("campaign_checkpoint")),
                )
                if path is not None
            ),
            source_artifacts=tuple(
                _source_artifact(key=key, path=path)
                for key, path in (
                    ("campaign_summary", related_artifacts.get("campaign_summary")),
                    ("campaign_manifest", related_artifacts.get("campaign_manifest")),
                    ("campaign_checkpoint", related_artifacts.get("campaign_checkpoint")),
                )
                if path is not None
            ),
            related_stage_names=tuple(
                str(stage_name)
                for stage_name in sorted(_mapping(campaign_summary.get("stage_statuses")))
            ),
        )
    ]

    review_decision = _review_decision(
        campaign_summary=campaign_summary,
        related_artifacts=related_artifacts,
        review_summary=review_summary,
        promotion_gates=promotion_gates,
    )
    if review_decision is not None:
        decisions.append(review_decision)
    return decisions


def _campaign_execution_decision_status(campaign_summary: Mapping[str, Any]) -> str:
    status = str(campaign_summary.get("status") or "").strip()
    final_outcomes = _mapping(campaign_summary.get("final_outcomes"))
    failed_stage_names = final_outcomes.get("failed_stage_names")
    partial_stage_names = final_outcomes.get("partial_stage_names")
    if status == "failed" or (isinstance(failed_stage_names, list) and failed_stage_names):
        return "rejected"
    if isinstance(partial_stage_names, list) and partial_stage_names:
        return "deferred"
    return "accepted"


def _campaign_execution_decision_summary(campaign_summary: Mapping[str, Any]) -> str:
    stage_statuses = _mapping(campaign_summary.get("stage_statuses"))
    return (
        f"Campaign `{campaign_summary.get('campaign_run_id', 'unknown_campaign')}` recorded "
        f"{len(stage_statuses)} tracked stage outcomes with overall status `{campaign_summary.get('status', 'unknown')}`."
    )


def _review_decision(
    *,
    campaign_summary: Mapping[str, Any],
    related_artifacts: Mapping[str, str],
    review_summary: Mapping[str, Any] | None,
    promotion_gates: Mapping[str, Any] | None,
) -> MilestoneDecisionEntry | None:
    selected_run_ids = _mapping(campaign_summary.get("selected_run_ids"))
    review_id = selected_run_ids.get("review_id")
    if review_summary is None and promotion_gates is None and review_id is None:
        return None

    evaluation_status = None if promotion_gates is None else str(promotion_gates.get("evaluation_status") or "").strip()
    promotion_status = None if promotion_gates is None else str(promotion_gates.get("promotion_status") or "").strip()
    decision_status = "deferred"
    if promotion_status == "approved":
        decision_status = "accepted"
    elif promotion_status in {"blocked", "rejected"} or evaluation_status in {"failed", "blocked"}:
        decision_status = "rejected"

    summary_parts: list[str] = []
    if review_id is not None:
        summary_parts.append(f"Review `{review_id}`")
    if isinstance(review_summary, Mapping):
        summary_parts.append(f"covered {_safe_int(review_summary.get('entry_count'))} ranked entries")
        counts_by_type = review_summary.get("counts_by_run_type")
        if isinstance(counts_by_type, Mapping) and counts_by_type:
            summary_parts.append(
                "with counts_by_run_type="
                + ",".join(f"{key}:{_safe_int(counts_by_type[key])}" for key in sorted(counts_by_type))
            )
    if promotion_status:
        summary_parts.append(f"promotion_status={promotion_status}")
    if evaluation_status:
        summary_parts.append(f"evaluation_status={evaluation_status}")
    summary_text = " ".join(summary_parts) if summary_parts else "Review and promotion outputs were resolved from campaign artifacts."

    evidence_artifacts = [
        path
        for key, path in (
            ("review_summary", related_artifacts.get("review_summary")),
            ("review_manifest", related_artifacts.get("review_manifest")),
            ("review_promotion_gates", related_artifacts.get("review_promotion_gates")),
        )
        if path is not None
    ]
    return MilestoneDecisionEntry(
        decision_id="review_promotion_outcome",
        title="Review and promotion outcome",
        status=decision_status,
        summary=summary_text,
        rationale=(
            "Review summaries and promotion-gate artifacts capture the final promotion recommendation "
            "for campaign outputs without re-evaluating prior stages."
        ),
        category="review_promotion",
        follow_up_actions=tuple(_review_follow_up_actions(promotion_gates=promotion_gates)),
        evidence_artifacts=tuple(evidence_artifacts),
        source_artifacts=tuple(
            _source_artifact(key=key, path=path)
            for key, path in (
                ("review_summary", related_artifacts.get("review_summary")),
                ("review_manifest", related_artifacts.get("review_manifest")),
                ("review_promotion_gates", related_artifacts.get("review_promotion_gates")),
            )
            if path is not None
        ),
        related_stage_names=("review",),
    )


def _review_result_fragment(
    *,
    review_summary: Mapping[str, Any] | None,
    promotion_gates: Mapping[str, Any] | None,
) -> str | None:
    fragments: list[str] = []
    if isinstance(review_summary, Mapping):
        fragments.append(f"Review ranked {_safe_int(review_summary.get('entry_count'))} entries")
    if isinstance(promotion_gates, Mapping):
        evaluation_status = promotion_gates.get("evaluation_status")
        promotion_status = promotion_gates.get("promotion_status")
        gate_count = _safe_int(promotion_gates.get("gate_count"))
        fragments.append(
            f"promotion evaluation_status={evaluation_status or 'NA'}, "
            f"promotion_status={promotion_status or 'NA'}, gate_count={gate_count}"
        )
    if not fragments:
        return None
    return "Review outcome: " + "; ".join(fragments) + "."


def _campaign_execution_follow_up_actions(campaign_summary: Mapping[str, Any]) -> list[str]:
    final_outcomes = _mapping(campaign_summary.get("final_outcomes"))
    actions: list[str] = []
    actions.extend(
        _stage_follow_up_actions(
            stage_names=final_outcomes.get("failed_stage_names"),
            template="Resolve failed stage `{stage_name}` before closing the milestone.",
        )
    )
    actions.extend(
        _stage_follow_up_actions(
            stage_names=final_outcomes.get("partial_stage_names"),
            template="Review partial outputs for `{stage_name}` before downstream reuse.",
        )
    )
    actions.extend(
        _stage_follow_up_actions(
            stage_names=final_outcomes.get("resumable_stage_names"),
            template="Preserve checkpoint continuity for resumable stage `{stage_name}`.",
        )
    )
    if not actions:
        actions.append("No additional campaign execution follow-up actions were required.")
    return actions


def _review_follow_up_actions(*, promotion_gates: Mapping[str, Any] | None) -> list[str]:
    if promotion_gates is None:
        return ["Persist promotion-gate outputs for reviewed campaigns so milestone decisions remain auditable."]

    promotion_status = str(promotion_gates.get("promotion_status") or "").strip()
    evaluation_status = str(promotion_gates.get("evaluation_status") or "").strip()
    if promotion_status == "approved":
        return ["Prepare the approved reviewed outputs for promotion handoff."]
    if promotion_status in {"blocked", "rejected"} or evaluation_status in {"failed", "blocked"}:
        return ["Address blocked promotion gates before promoting the reviewed campaign outputs."]
    return ["Complete review and promotion follow-ups before promoting the reviewed campaign outputs."]


def _stage_follow_up_actions(*, stage_names: Any, template: str) -> list[str]:
    if not isinstance(stage_names, list):
        return []
    return [template.format(stage_name=str(stage_name)) for stage_name in sorted(stage_names)]


def _source_artifact(*, key: str, path: str) -> MilestoneSourceArtifact:
    return MilestoneSourceArtifact(
        artifact_id=key,
        label=key.replace("_", " "),
        path=path,
        role="source_artifact",
    )


def _reporting_window(campaign_summary: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = campaign_summary.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        stages = checkpoint.get("stages")
        if isinstance(stages, list):
            for stage in stages:
                if not isinstance(stage, Mapping):
                    continue
                fingerprint = _mapping(stage.get("fingerprint_inputs"))
                config = _mapping(fingerprint.get("config"))
                time_windows = _mapping(config.get("time_windows"))
                if time_windows:
                    start = time_windows.get("start") or time_windows.get("train_start")
                    end = time_windows.get("end") or time_windows.get("test_end") or time_windows.get("predict_end")
                    if start is not None or end is not None:
                        return {key: value for key, value in (("start", start), ("end", end)) if value is not None}
    return {}


def _load_mapped_json(path_value: Any) -> dict[str, Any] | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    return _load_json_if_exists(Path(path_value))


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _require_json(path: Path) -> dict[str, Any]:
    payload = _load_json_if_exists(path)
    if payload is None:
        raise FileNotFoundError(f"Required campaign artifact not found: {path}")
    return payload


def _relative_path(base_dir: Path, target_path: Path) -> str:
    return Path(os.path.relpath(target_path, start=base_dir)).as_posix()


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "DEFAULT_CAMPAIGN_MILESTONE_DIRNAME",
    "build_campaign_milestone_report_payloads",
    "generate_campaign_milestone_report",
    "load_campaign_reporting_payloads",
    "resolve_campaign_artifact_dir",
]
