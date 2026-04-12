from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

from src.cli import compare_alpha as compare_alpha_cli
from src.cli import compare_research as compare_research_cli
from src.cli import compare_strategies as compare_strategies_cli
from src.cli import review_candidate_selection as review_candidate_selection_cli
from src.cli import run_alpha as run_alpha_cli
from src.cli import run_candidate_selection as run_candidate_selection_cli
from src.cli import run_portfolio as run_portfolio_cli
from src.cli import run_strategy as run_strategy_cli
from src.config.research_campaign import (
    CampaignReusePolicyConfig,
    ResearchCampaignConfig,
    ResearchCampaignConfigError,
    load_research_campaign_config,
    resolve_research_campaign_config,
)
from src.data.load_features import FeaturePaths, SUPPORTED_FEATURE_DATASETS
from src.research.alpha.catalog import load_alphas_config
from src.research.alpha_eval.registry import alpha_evaluation_registry_path
from src.research.campaign_checkpoint import (
    CAMPAIGN_STAGE_ORDER,
    CampaignCheckpointError,
    build_campaign_checkpoint_payload,
    build_campaign_stage_checkpoint,
    validate_campaign_checkpoint_payload,
    write_campaign_checkpoint,
)
from src.research.candidate_selection.registry import candidate_selection_registry_path
from src.research.experiment_tracker import ARTIFACTS_ROOT as STRATEGY_ARTIFACTS_ROOT
from src.research.registry import canonicalize_value, default_registry_path, load_registry

PREFLIGHT_SUMMARY_FILENAME = "preflight_summary.json"
CAMPAIGN_CONFIG_FILENAME = "campaign_config.json"
CAMPAIGN_CHECKPOINT_FILENAME = "checkpoint.json"
CAMPAIGN_MANIFEST_FILENAME = "manifest.json"
CAMPAIGN_SUMMARY_FILENAME = "summary.json"


@dataclass(frozen=True)
class CampaignPreflightCheck:
    check_id: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CampaignPreflightResult:
    status: str
    summary_path: Path
    summary: dict[str, Any]
    checks: tuple[CampaignPreflightCheck, ...]


@dataclass(frozen=True)
class CampaignStageRecord:
    stage_name: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CampaignStageReuseDecision:
    stage_name: str
    action: str
    reason: str
    matched_checkpoint: bool
    checkpoint_state: str | None
    checkpoint_input_fingerprint: str | None
    fingerprint_match: bool
    invalidated_by_stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "stage_name": self.stage_name,
                "action": self.action,
                "reason": self.reason,
                "matched_checkpoint": self.matched_checkpoint,
                "checkpoint_state": self.checkpoint_state,
                "checkpoint_input_fingerprint": self.checkpoint_input_fingerprint,
                "fingerprint_match": self.fingerprint_match,
                "invalidated_by_stage": self.invalidated_by_stage,
            }
        )


def _stage_retry_payload(existing_stage: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if existing_stage is None:
        return None
    previous_state = str(existing_stage.get("state") or "").strip()
    if previous_state not in {"failed", "partial"}:
        return None
    payload = {
        "attempted": True,
        "previous_state": previous_state,
        "previous_state_reason": existing_stage.get("state_reason"),
    }
    details = existing_stage.get("details") or {}
    if isinstance(details, Mapping) and isinstance(details.get("failure"), Mapping):
        payload["previous_failure"] = canonicalize_value(dict(details["failure"]))
    return canonicalize_value(payload)


def _stage_details_with_retry_metadata(
    details: Mapping[str, Any] | None,
    *,
    existing_stage: Mapping[str, Any] | None,
    reuse_decision: CampaignStageReuseDecision | None = None,
) -> dict[str, Any]:
    payload = canonicalize_value(dict(details or {}))
    retry = _stage_retry_payload(existing_stage)
    if retry is not None:
        payload["retry"] = retry
    if reuse_decision is not None:
        payload["reuse_policy"] = reuse_decision.to_dict()
    return canonicalize_value(payload)


def _build_stage_failure_details(
    stage_name: str,
    exc: BaseException,
    *,
    details: Mapping[str, Any] | None,
    existing_stage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = _stage_details_with_retry_metadata(details, existing_stage=existing_stage)
    message = str(exc).strip()
    payload["failure"] = canonicalize_value(
        {
            "stage_name": stage_name,
            "exception_type": exc.__class__.__name__,
            "message": message or exc.__class__.__name__,
            "kind": "interrupted" if isinstance(exc, KeyboardInterrupt) else "exception",
            "retryable": True,
        }
    )
    return canonicalize_value(payload)


def _stage_execution_metadata(
    checkpoint_stage: Mapping[str, Any],
    *,
    stage_details: Mapping[str, Any] | None,
) -> dict[str, Any]:
    details = stage_details if isinstance(stage_details, Mapping) else {}
    retry_details = details.get("retry") if isinstance(details.get("retry"), Mapping) else None
    reuse_details = (
        details.get("reuse_policy") if isinstance(details.get("reuse_policy"), Mapping) else None
    )
    failure_details = details.get("failure") if isinstance(details.get("failure"), Mapping) else None
    state = str(checkpoint_stage.get("state") or "")
    state_reason = checkpoint_stage.get("state_reason")
    return canonicalize_value(
        {
            "state": state,
            "resume": {
                "resumable": bool(checkpoint_stage.get("resumable")),
                "terminal": bool(checkpoint_stage.get("terminal")),
                "checkpoint_state": state,
                "checkpoint_state_reason": state_reason,
            },
            "retry": {
                "attempted": bool(retry_details.get("attempted")) if retry_details is not None else False,
                "previous_state": None if retry_details is None else retry_details.get("previous_state"),
                "previous_state_reason": (
                    None if retry_details is None else retry_details.get("previous_state_reason")
                ),
                "previous_failure": None if retry_details is None else retry_details.get("previous_failure"),
            },
            "reuse": {
                "reused": state == "reused",
                "decision": None if reuse_details is None else canonicalize_value(dict(reuse_details)),
            },
            "skip": {
                "skipped": state == "skipped",
                "reason": state_reason if state == "skipped" else None,
            },
            "failure": None if failure_details is None else canonicalize_value(dict(failure_details)),
            "checkpoint": {
                "stage_name": checkpoint_stage.get("stage_name"),
                "state": state,
                "state_reason": state_reason,
                "source": checkpoint_stage.get("source"),
                "terminal": bool(checkpoint_stage.get("terminal")),
                "resumable": bool(checkpoint_stage.get("resumable")),
            },
            "fingerprint": {
                "input_fingerprint": checkpoint_stage.get("input_fingerprint"),
                "fingerprint_inputs": canonicalize_value(
                    dict(checkpoint_stage.get("fingerprint_inputs") or {})
                ),
            },
        }
    )


def _stage_execution_by_name(
    checkpoint_payload: Mapping[str, Any],
    *,
    stages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stage_details_by_name = {
        str(stage.get("stage_name")): stage.get("details")
        for stage in stages
    }
    return canonicalize_value(
        {
            str(checkpoint_stage.get("stage_name")): _stage_execution_metadata(
                checkpoint_stage,
                stage_details=stage_details_by_name.get(str(checkpoint_stage.get("stage_name"))),
            )
            for checkpoint_stage in checkpoint_payload.get("stages", [])
        }
    )


def _upsert_stage_record(
    records: Sequence[CampaignStageRecord],
    record: CampaignStageRecord,
) -> list[CampaignStageRecord]:
    updated = [existing for existing in records if existing.stage_name != record.stage_name]
    updated.append(record)
    return updated


@dataclass(frozen=True)
class CampaignArtifactReference:
    stage_name: str
    source: str
    run_id: str
    artifact_dir: Path
    registry_path: Path | None = None
    match_criteria: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchCampaignRunResult:
    config: ResearchCampaignConfig
    campaign_run_id: str
    campaign_artifact_dir: Path
    campaign_checkpoint_path: Path
    campaign_manifest_path: Path
    campaign_summary_path: Path
    preflight_summary_path: Path
    campaign_checkpoint: dict[str, Any]
    campaign_manifest: dict[str, Any]
    campaign_summary: dict[str, Any]
    preflight_summary: dict[str, Any]
    stage_records: tuple[CampaignStageRecord, ...]
    alpha_results: tuple[Any, ...]
    strategy_results: tuple[Any, ...]
    alpha_comparison_result: Any | None
    strategy_comparison_result: Any | None
    candidate_selection_result: Any | None
    portfolio_result: Any | None
    candidate_review_result: Any | None
    review_result: Any | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one deterministic research campaign from a unified YAML/JSON config: "
            "preflight, research, comparison, candidate selection, portfolio, and review."
        )
    )
    parser.add_argument(
        "--config",
        help=(
            "Optional research campaign config path. "
            "Defaults to configs/research_campaign.yml."
        ),
    )
    return parser.parse_args(argv)


def resolve_cli_config(config_path: str | Path | None = None) -> ResearchCampaignConfig:
    if config_path is None:
        return resolve_research_campaign_config()
    loaded = load_research_campaign_config(Path(config_path))
    return resolve_research_campaign_config(loaded.to_dict())


def run_cli(argv: Sequence[str] | None = None) -> ResearchCampaignRunResult:
    args = parse_args(argv)
    config = resolve_cli_config(args.config)
    result = run_research_campaign(config)
    print_summary(result)
    return result


def run_research_campaign(config: ResearchCampaignConfig) -> ResearchCampaignRunResult:
    records: list[CampaignStageRecord] = []
    campaign_run_id = _build_campaign_run_id(config)
    campaign_artifact_dir = _campaign_artifact_dir(config, campaign_run_id=campaign_run_id)
    _persist_campaign_config(campaign_artifact_dir, config)
    existing_checkpoint = _load_existing_campaign_checkpoint(
        campaign_artifact_dir=campaign_artifact_dir,
        campaign_run_id=campaign_run_id,
    )
    stage_input_fingerprints: dict[str, str | None] = {}
    candidate_selection_reference: CampaignArtifactReference | None = None
    portfolio_reference: CampaignArtifactReference | None = None
    alpha_results: list[Any] = []
    strategy_results: list[Any] = []
    alpha_comparison_result: Any | None = None
    strategy_comparison_result: Any | None = None
    candidate_selection_result: Any | None = None
    portfolio_result: Any | None = None
    candidate_review_result: Any | None = None
    review_result: Any | None = None
    preflight_result = CampaignPreflightResult(
        status="unknown",
        summary_path=campaign_artifact_dir / PREFLIGHT_SUMMARY_FILENAME,
        summary={"status": "unknown", "checks": [], "check_counts": {}},
        checks=(),
    )
    current_stage_name: str | None = None
    current_stage_details: dict[str, Any] | None = None
    current_existing_stage: Mapping[str, Any] | None = None
    error_already_persisted = False
    downstream_invalidated_by_stage: str | None = None

    try:
        preflight_fingerprint, _ = _compute_stage_input_fingerprint(
            stage_name="preflight",
            config=config,
            alpha_results=alpha_results,
            strategy_results=strategy_results,
            candidate_selection_result=candidate_selection_result,
            candidate_selection_reference=candidate_selection_reference,
            portfolio_result=portfolio_result,
            portfolio_reference=portfolio_reference,
            stage_input_fingerprints=stage_input_fingerprints,
        )
        existing_preflight_stage = _checkpoint_stage(existing_checkpoint, "preflight")
        preflight_reuse_decision = _stage_reuse_decision(
            stage_name="preflight",
            checkpoint_stage=existing_preflight_stage,
            input_fingerprint=preflight_fingerprint,
            reuse_policy=config.reuse_policy,
            invalidated_by_stage=downstream_invalidated_by_stage,
        )
        if preflight_reuse_decision.action == "reuse":
            preflight_result = _restore_preflight_result(
                campaign_artifact_dir=campaign_artifact_dir,
                checkpoint_stage=existing_preflight_stage,
            )
            records.append(
                _restore_stage_record(
                    existing_preflight_stage,
                    reuse_decision=preflight_reuse_decision,
                )
            )
        else:
            current_stage_name = "preflight"
            current_stage_details = {"reuse_policy": preflight_reuse_decision.to_dict()}
            current_existing_stage = existing_preflight_stage
            preflight_result = _run_preflight(
                config,
                campaign_run_id=campaign_run_id,
                campaign_artifact_dir=campaign_artifact_dir,
            )
            records.append(
                CampaignStageRecord(
                    stage_name="preflight",
                    status="completed" if preflight_result.status == "passed" else "failed",
                    details=_stage_details_with_retry_metadata(
                        preflight_result.summary,
                        existing_stage=existing_preflight_stage,
                        reuse_decision=preflight_reuse_decision,
                    ),
                )
            )
            current_stage_name = None
            current_stage_details = None
            current_existing_stage = None
        downstream_invalidated_by_stage = _update_downstream_invalidation(
            current_invalidated_by_stage=downstream_invalidated_by_stage,
            stage_name="preflight",
            stage_reuse_decision=preflight_reuse_decision,
            reuse_policy=config.reuse_policy,
        )
        stage_input_fingerprints["preflight"] = preflight_fingerprint
        if preflight_result.status != "passed":
            _write_campaign_artifacts(
                config=config,
                campaign_run_id=campaign_run_id,
                campaign_artifact_dir=campaign_artifact_dir,
                preflight_result=preflight_result,
                stage_records=records,
                alpha_results=alpha_results,
                strategy_results=strategy_results,
                alpha_comparison_result=alpha_comparison_result,
                strategy_comparison_result=strategy_comparison_result,
                candidate_selection_result=candidate_selection_result,
                candidate_selection_reference=candidate_selection_reference,
                portfolio_result=portfolio_result,
                portfolio_reference=portfolio_reference,
                candidate_review_result=candidate_review_result,
                review_result=review_result,
                status="failed",
            )
            error_already_persisted = True
            failures = [
                check.message
                for check in preflight_result.checks
                if check.status == "failed"
            ]
            raise ValueError(
                "Campaign preflight failed. "
                f"See {preflight_result.summary_path.as_posix()}. "
                + (" | ".join(failures) if failures else "One or more checks failed.")
            )

        research_fingerprint, _ = _compute_stage_input_fingerprint(
            stage_name="research",
            config=config,
            alpha_results=alpha_results,
            strategy_results=strategy_results,
            candidate_selection_result=candidate_selection_result,
            candidate_selection_reference=candidate_selection_reference,
            portfolio_result=portfolio_result,
            portfolio_reference=portfolio_reference,
            stage_input_fingerprints=stage_input_fingerprints,
        )
        existing_research_stage = _checkpoint_stage(existing_checkpoint, "research")
        research_reuse_decision = _stage_reuse_decision(
            stage_name="research",
            checkpoint_stage=existing_research_stage,
            input_fingerprint=research_fingerprint,
            reuse_policy=config.reuse_policy,
            invalidated_by_stage=downstream_invalidated_by_stage,
        )
        if research_reuse_decision.action == "reuse":
            alpha_results, strategy_results = _restore_research_results(existing_research_stage)
            records.append(
                _restore_stage_record(
                    existing_research_stage,
                    reuse_decision=research_reuse_decision,
                )
            )
        else:
            current_stage_name = "research"
            current_stage_details = {"reuse_policy": research_reuse_decision.to_dict()}
            current_existing_stage = existing_research_stage
            alpha_results = _run_alpha_research(config)
            strategy_results = _run_strategy_research(config)
            records.append(
                CampaignStageRecord(
                    stage_name="research",
                    status="completed",
                    details=_stage_details_with_retry_metadata(
                        {
                            "alpha_runs": len(alpha_results),
                            "strategy_runs": len(strategy_results),
                        },
                        existing_stage=existing_research_stage,
                        reuse_decision=research_reuse_decision,
                    ),
                )
            )
            current_stage_name = None
            current_stage_details = None
            current_existing_stage = None
        downstream_invalidated_by_stage = _update_downstream_invalidation(
            current_invalidated_by_stage=downstream_invalidated_by_stage,
            stage_name="research",
            stage_reuse_decision=research_reuse_decision,
            reuse_policy=config.reuse_policy,
        )
        stage_input_fingerprints["research"] = research_fingerprint

        comparison_fingerprint, _ = _compute_stage_input_fingerprint(
            stage_name="comparison",
            config=config,
            alpha_results=alpha_results,
            strategy_results=strategy_results,
            candidate_selection_result=candidate_selection_result,
            candidate_selection_reference=candidate_selection_reference,
            portfolio_result=portfolio_result,
            portfolio_reference=portfolio_reference,
            stage_input_fingerprints=stage_input_fingerprints,
        )
        existing_comparison_stage = _checkpoint_stage(existing_checkpoint, "comparison")
        comparison_reuse_decision = _stage_reuse_decision(
            stage_name="comparison",
            checkpoint_stage=existing_comparison_stage,
            input_fingerprint=comparison_fingerprint,
            reuse_policy=config.reuse_policy,
            invalidated_by_stage=downstream_invalidated_by_stage,
        )
        if config.comparison.enabled and comparison_reuse_decision.action == "reuse":
            alpha_comparison_result, strategy_comparison_result = _restore_comparison_results(existing_comparison_stage)
            records.append(
                _restore_stage_record(
                    existing_comparison_stage,
                    reuse_decision=comparison_reuse_decision,
                )
            )
        else:
            current_stage_name = "comparison"
            current_stage_details = {"reuse_policy": comparison_reuse_decision.to_dict()}
            current_existing_stage = existing_comparison_stage
            if config.comparison.enabled:
                alpha_comparison_result = _run_alpha_comparison(config)
                strategy_comparison_result = _run_strategy_comparison(config)
            records.append(
                CampaignStageRecord(
                    stage_name="comparison",
                    status="completed" if config.comparison.enabled else "skipped",
                    details=_stage_details_with_retry_metadata(
                        {
                            "alpha_comparison_id": None
                            if alpha_comparison_result is None
                            else str(alpha_comparison_result.comparison_id),
                            "strategy_comparison_id": None
                            if strategy_comparison_result is None
                            else str(strategy_comparison_result.comparison_id),
                        },
                        existing_stage=existing_comparison_stage,
                        reuse_decision=comparison_reuse_decision,
                    ),
                )
            )
            current_stage_name = None
            current_stage_details = None
            current_existing_stage = None
        downstream_invalidated_by_stage = _update_downstream_invalidation(
            current_invalidated_by_stage=downstream_invalidated_by_stage,
            stage_name="comparison",
            stage_reuse_decision=comparison_reuse_decision,
            reuse_policy=config.reuse_policy,
        )
        stage_input_fingerprints["comparison"] = comparison_fingerprint

        candidate_selection_fingerprint, _ = _compute_stage_input_fingerprint(
            stage_name="candidate_selection",
            config=config,
            alpha_results=alpha_results,
            strategy_results=strategy_results,
            candidate_selection_result=candidate_selection_result,
            candidate_selection_reference=candidate_selection_reference,
            portfolio_result=portfolio_result,
            portfolio_reference=portfolio_reference,
            stage_input_fingerprints=stage_input_fingerprints,
        )
        existing_candidate_selection_stage = _checkpoint_stage(existing_checkpoint, "candidate_selection")
        candidate_selection_reuse_decision = _stage_reuse_decision(
            stage_name="candidate_selection",
            checkpoint_stage=existing_candidate_selection_stage,
            input_fingerprint=candidate_selection_fingerprint,
            reuse_policy=config.reuse_policy,
            invalidated_by_stage=downstream_invalidated_by_stage,
        )
        if config.candidate_selection.enabled and candidate_selection_reuse_decision.action == "reuse":
            candidate_selection_result, candidate_selection_reference = _restore_candidate_selection_stage(
                existing_candidate_selection_stage
            )
            records.append(
                _restore_stage_record(
                    existing_candidate_selection_stage,
                    reuse_decision=candidate_selection_reuse_decision,
                )
            )
        else:
            current_stage_name = "candidate_selection"
            current_stage_details = {"reuse_policy": candidate_selection_reuse_decision.to_dict()}
            current_existing_stage = existing_candidate_selection_stage
            if config.candidate_selection.enabled:
                candidate_selection_result, candidate_selection_reference = _run_candidate_selection(config)
            records.append(
                CampaignStageRecord(
                    stage_name="candidate_selection",
                    status="completed" if candidate_selection_result is not None else "skipped",
                    details=_stage_details_with_retry_metadata(
                        {
                            "run_id": None
                            if candidate_selection_result is None
                            else str(candidate_selection_result.run_id),
                            "input_reference": (
                                None
                                if candidate_selection_reference is None
                                else _artifact_reference_payload(candidate_selection_reference)
                            ),
                        },
                        existing_stage=existing_candidate_selection_stage,
                        reuse_decision=candidate_selection_reuse_decision,
                    ),
                )
            )
            current_stage_name = None
            current_stage_details = None
            current_existing_stage = None
        downstream_invalidated_by_stage = _update_downstream_invalidation(
            current_invalidated_by_stage=downstream_invalidated_by_stage,
            stage_name="candidate_selection",
            stage_reuse_decision=candidate_selection_reuse_decision,
            reuse_policy=config.reuse_policy,
        )
        stage_input_fingerprints["candidate_selection"] = candidate_selection_fingerprint

        portfolio_fingerprint, _ = _compute_stage_input_fingerprint(
            stage_name="portfolio",
            config=config,
            alpha_results=alpha_results,
            strategy_results=strategy_results,
            candidate_selection_result=candidate_selection_result,
            candidate_selection_reference=candidate_selection_reference,
            portfolio_result=portfolio_result,
            portfolio_reference=portfolio_reference,
            stage_input_fingerprints=stage_input_fingerprints,
        )
        existing_portfolio_stage = _checkpoint_stage(existing_checkpoint, "portfolio")
        portfolio_reuse_decision = _stage_reuse_decision(
            stage_name="portfolio",
            checkpoint_stage=existing_portfolio_stage,
            input_fingerprint=portfolio_fingerprint,
            reuse_policy=config.reuse_policy,
            invalidated_by_stage=downstream_invalidated_by_stage,
        )
        if config.portfolio.enabled and portfolio_reuse_decision.action == "reuse":
            portfolio_result, portfolio_reference = _restore_portfolio_stage(existing_portfolio_stage)
            if portfolio_result is not None:
                if candidate_selection_result is not None:
                    portfolio_reference = CampaignArtifactReference(
                        stage_name="candidate_selection",
                        source="same_campaign",
                        run_id=str(candidate_selection_result.run_id),
                        artifact_dir=Path(candidate_selection_result.artifact_dir),
                        metadata={"consumer": "portfolio"},
                    )
                elif candidate_selection_reference is not None:
                    portfolio_reference = candidate_selection_reference
            records.append(
                _restore_stage_record(
                    existing_portfolio_stage,
                    reuse_decision=portfolio_reuse_decision,
                )
            )
        else:
            current_stage_name = "portfolio"
            current_stage_details = {
                "reuse_policy": portfolio_reuse_decision.to_dict(),
                "input_candidate_selection": (
                    None
                    if candidate_selection_reference is None
                    else _artifact_reference_payload(candidate_selection_reference)
                ),
            }
            current_existing_stage = existing_portfolio_stage
            if config.portfolio.enabled:
                portfolio_result, portfolio_reference, candidate_selection_reference = _run_portfolio(
                    config,
                    candidate_selection_result=candidate_selection_result,
                    candidate_selection_reference=candidate_selection_reference,
                )
            if candidate_selection_result is None and candidate_selection_reference is not None:
                records = _replace_stage_record_details(
                    records,
                    stage_name="candidate_selection",
                    details_update={
                        "input_reference": _artifact_reference_payload(candidate_selection_reference),
                    },
                )
            records.append(
                CampaignStageRecord(
                    stage_name="portfolio",
                    status="completed" if portfolio_result is not None else "skipped",
                    details=_stage_details_with_retry_metadata(
                        {
                            "run_id": None if portfolio_result is None else str(portfolio_result.run_id),
                            "input_candidate_selection": (
                                None
                                if candidate_selection_reference is None
                                else _artifact_reference_payload(candidate_selection_reference)
                            ),
                            "input_reference": (
                                None
                                if portfolio_reference is None
                                else _artifact_reference_payload(portfolio_reference)
                            ),
                        },
                        existing_stage=existing_portfolio_stage,
                        reuse_decision=portfolio_reuse_decision,
                    ),
                )
            )
            current_stage_name = None
            current_stage_details = None
            current_existing_stage = None
        downstream_invalidated_by_stage = _update_downstream_invalidation(
            current_invalidated_by_stage=downstream_invalidated_by_stage,
            stage_name="portfolio",
            stage_reuse_decision=portfolio_reuse_decision,
            reuse_policy=config.reuse_policy,
        )
        stage_input_fingerprints["portfolio"] = portfolio_fingerprint

        candidate_review_fingerprint, _ = _compute_stage_input_fingerprint(
            stage_name="candidate_review",
            config=config,
            alpha_results=alpha_results,
            strategy_results=strategy_results,
            candidate_selection_result=candidate_selection_result,
            candidate_selection_reference=candidate_selection_reference,
            portfolio_result=portfolio_result,
            portfolio_reference=portfolio_reference,
            stage_input_fingerprints=stage_input_fingerprints,
        )
        existing_candidate_review_stage = _checkpoint_stage(existing_checkpoint, "candidate_review")
        candidate_review_enabled = bool(
            config.candidate_selection.enabled and config.candidate_selection.execution.enable_review
        )
        candidate_review_reuse_decision = _stage_reuse_decision(
            stage_name="candidate_review",
            checkpoint_stage=existing_candidate_review_stage,
            input_fingerprint=candidate_review_fingerprint,
            reuse_policy=config.reuse_policy,
            invalidated_by_stage=downstream_invalidated_by_stage,
        )
        if candidate_review_enabled and candidate_review_reuse_decision.action == "reuse":
            candidate_review_result = _restore_candidate_review_result(existing_candidate_review_stage)
            records.append(
                _restore_stage_record(
                    existing_candidate_review_stage,
                    reuse_decision=candidate_review_reuse_decision,
                )
            )
        else:
            current_stage_name = "candidate_review"
            current_stage_details = {
                "reuse_policy": candidate_review_reuse_decision.to_dict(),
                "input_candidate_selection": (
                    None
                    if candidate_selection_reference is None
                    else _artifact_reference_payload(candidate_selection_reference)
                ),
                "input_portfolio": (
                    None
                    if portfolio_reference is None
                    else _artifact_reference_payload(portfolio_reference)
                ),
            }
            current_existing_stage = existing_candidate_review_stage
            if candidate_review_enabled:
                if portfolio_result is None and portfolio_reference is None:
                    portfolio_reference = _resolve_portfolio_reference(
                        config,
                        candidate_selection_reference=candidate_selection_reference,
                    )
                candidate_review_result = _run_candidate_review(
                    config,
                    candidate_selection_result=candidate_selection_result,
                    candidate_selection_reference=candidate_selection_reference,
                    portfolio_result=portfolio_result,
                    portfolio_reference=portfolio_reference,
                )
            if portfolio_result is None and portfolio_reference is not None:
                records = _replace_stage_record_details(
                    records,
                    stage_name="portfolio",
                    details_update={
                        "input_reference": _artifact_reference_payload(portfolio_reference),
                    },
                )
            records.append(
                CampaignStageRecord(
                    stage_name="candidate_review",
                    status="completed" if candidate_review_result is not None else "skipped",
                    details=_stage_details_with_retry_metadata(
                        {
                            "review_dir": None
                            if candidate_review_result is None
                            else str(candidate_review_result.review_dir),
                            "candidate_selection_run_id": None
                            if candidate_review_result is None
                            else _string_or_none(candidate_review_result, "candidate_selection_run_id"),
                            "portfolio_run_id": None
                            if candidate_review_result is None
                            else _string_or_none(candidate_review_result, "portfolio_run_id"),
                            "input_candidate_selection": (
                                None
                                if candidate_selection_reference is None
                                else _artifact_reference_payload(candidate_selection_reference)
                            ),
                            "input_portfolio": (
                                None
                                if portfolio_reference is None
                                else _artifact_reference_payload(portfolio_reference)
                            ),
                        },
                        existing_stage=existing_candidate_review_stage,
                        reuse_decision=candidate_review_reuse_decision,
                    ),
                )
            )
            current_stage_name = None
            current_stage_details = None
            current_existing_stage = None
        downstream_invalidated_by_stage = _update_downstream_invalidation(
            current_invalidated_by_stage=downstream_invalidated_by_stage,
            stage_name="candidate_review",
            stage_reuse_decision=candidate_review_reuse_decision,
            reuse_policy=config.reuse_policy,
        )
        stage_input_fingerprints["candidate_review"] = candidate_review_fingerprint

        review_fingerprint, _ = _compute_stage_input_fingerprint(
            stage_name="review",
            config=config,
            alpha_results=alpha_results,
            strategy_results=strategy_results,
            candidate_selection_result=candidate_selection_result,
            candidate_selection_reference=candidate_selection_reference,
            portfolio_result=portfolio_result,
            portfolio_reference=portfolio_reference,
            stage_input_fingerprints=stage_input_fingerprints,
        )
        existing_review_stage = _checkpoint_stage(existing_checkpoint, "review")
        review_reuse_decision = _stage_reuse_decision(
            stage_name="review",
            checkpoint_stage=existing_review_stage,
            input_fingerprint=review_fingerprint,
            reuse_policy=config.reuse_policy,
            invalidated_by_stage=downstream_invalidated_by_stage,
        )
        if review_reuse_decision.action == "reuse":
            review_result = _restore_review_result(existing_review_stage)
            records.append(
                _restore_stage_record(
                    existing_review_stage,
                    reuse_decision=review_reuse_decision,
                )
            )
        else:
            current_stage_name = "review"
            current_stage_details = {"reuse_policy": review_reuse_decision.to_dict()}
            current_existing_stage = existing_review_stage
            review_result = _run_research_review(config)
            records.append(
                CampaignStageRecord(
                    stage_name="review",
                    status="completed",
                    details=_stage_details_with_retry_metadata(
                        {
                            "candidate_review_dir": None
                            if candidate_review_result is None
                            else str(candidate_review_result.review_dir),
                            "review_id": str(review_result.review_id),
                        },
                        existing_stage=existing_review_stage,
                        reuse_decision=review_reuse_decision,
                    ),
                )
            )
            current_stage_name = None
            current_stage_details = None
            current_existing_stage = None
        downstream_invalidated_by_stage = _update_downstream_invalidation(
            current_invalidated_by_stage=downstream_invalidated_by_stage,
            stage_name="review",
            stage_reuse_decision=review_reuse_decision,
            reuse_policy=config.reuse_policy,
        )
        stage_input_fingerprints["review"] = review_fingerprint
    except KeyboardInterrupt as exc:
        if not error_already_persisted and current_stage_name is not None:
            records = _upsert_stage_record(
                records,
                CampaignStageRecord(
                    stage_name=current_stage_name,
                    status="partial",
                    details=_build_stage_failure_details(
                        current_stage_name,
                        exc,
                        details=current_stage_details,
                        existing_stage=current_existing_stage,
                    ),
                ),
            )
            _write_campaign_artifacts(
                config=config,
                campaign_run_id=campaign_run_id,
                campaign_artifact_dir=campaign_artifact_dir,
                preflight_result=preflight_result,
                stage_records=records,
                alpha_results=alpha_results,
                strategy_results=strategy_results,
                alpha_comparison_result=alpha_comparison_result,
                strategy_comparison_result=strategy_comparison_result,
                candidate_selection_result=candidate_selection_result,
                candidate_selection_reference=candidate_selection_reference,
                portfolio_result=portfolio_result,
                portfolio_reference=portfolio_reference,
                candidate_review_result=candidate_review_result,
                review_result=review_result,
                status="partial",
            )
        raise
    except Exception as exc:
        if not error_already_persisted and current_stage_name is not None:
            records = _upsert_stage_record(
                records,
                CampaignStageRecord(
                    stage_name=current_stage_name,
                    status="failed",
                    details=_build_stage_failure_details(
                        current_stage_name,
                        exc,
                        details=current_stage_details,
                        existing_stage=current_existing_stage,
                    ),
                ),
            )
            _write_campaign_artifacts(
                config=config,
                campaign_run_id=campaign_run_id,
                campaign_artifact_dir=campaign_artifact_dir,
                preflight_result=preflight_result,
                stage_records=records,
                alpha_results=alpha_results,
                strategy_results=strategy_results,
                alpha_comparison_result=alpha_comparison_result,
                strategy_comparison_result=strategy_comparison_result,
                candidate_selection_result=candidate_selection_result,
                candidate_selection_reference=candidate_selection_reference,
                portfolio_result=portfolio_result,
                portfolio_reference=portfolio_reference,
                candidate_review_result=candidate_review_result,
                review_result=review_result,
                status="failed",
            )
        raise

    checkpoint_path, manifest_path, summary_path, checkpoint_payload, manifest_payload, summary_payload = _write_campaign_artifacts(
        config=config,
        campaign_run_id=campaign_run_id,
        campaign_artifact_dir=campaign_artifact_dir,
        preflight_result=preflight_result,
        stage_records=records,
        alpha_results=alpha_results,
        strategy_results=strategy_results,
        alpha_comparison_result=alpha_comparison_result,
        strategy_comparison_result=strategy_comparison_result,
        candidate_selection_result=candidate_selection_result,
        candidate_selection_reference=candidate_selection_reference,
        portfolio_result=portfolio_result,
        portfolio_reference=portfolio_reference,
        candidate_review_result=candidate_review_result,
        review_result=review_result,
        status="completed",
    )

    return ResearchCampaignRunResult(
        config=config,
        campaign_run_id=campaign_run_id,
        campaign_artifact_dir=campaign_artifact_dir,
        campaign_checkpoint_path=checkpoint_path,
        campaign_manifest_path=manifest_path,
        campaign_summary_path=summary_path,
        preflight_summary_path=preflight_result.summary_path,
        campaign_checkpoint=checkpoint_payload,
        campaign_manifest=manifest_payload,
        campaign_summary=summary_payload,
        preflight_summary=preflight_result.summary,
        stage_records=tuple(records),
        alpha_results=tuple(alpha_results),
        strategy_results=tuple(strategy_results),
        alpha_comparison_result=alpha_comparison_result,
        strategy_comparison_result=strategy_comparison_result,
        candidate_selection_result=candidate_selection_result,
        portfolio_result=portfolio_result,
        candidate_review_result=candidate_review_result,
        review_result=review_result,
    )


def print_summary(result: ResearchCampaignRunResult) -> None:
    stage_statuses = dict(result.campaign_summary.get("stage_statuses", {}))
    stage_counts = _stage_state_counts(stage_statuses)
    print("Research Campaign Summary")
    print("-------------------------")
    print(
        f"Campaign: {result.campaign_run_id} | "
        f"status={result.campaign_summary.get('status', 'unknown')} | "
        f"dir={result.campaign_artifact_dir.as_posix()}"
    )
    print(
        "Preflight: "
        f"{result.preflight_summary.get('status', 'unknown')} "
        f"({len(result.campaign_summary.get('stages', []))} stages tracked) | "
        f"summary={result.preflight_summary_path.as_posix()}"
    )
    print(
        "Stage States: "
        + " | ".join(
            f"{state}={count}"
            for state, count in stage_counts.items()
            if count
        )
    )
    print(
        "Stage Details: "
        + " | ".join(
            f"{stage_name}={stage_statuses.get(stage_name, 'unknown')}"
            for stage_name in CAMPAIGN_STAGE_ORDER
        )
    )
    print(
        f"Research: alpha_runs={len(result.alpha_results)} | "
        f"strategy_runs={len(result.strategy_results)}"
    )

    alpha_comparison_id = (
        None
        if result.alpha_comparison_result is None
        else str(result.alpha_comparison_result.comparison_id)
    )
    strategy_comparison_id = (
        None
        if result.strategy_comparison_result is None
        else str(result.strategy_comparison_result.comparison_id)
    )
    print(
        "Comparison: "
        f"alpha={alpha_comparison_id or 'skipped'} | "
        f"strategy={strategy_comparison_id or 'skipped'}"
    )

    candidate_run_id = (
        None
        if result.candidate_selection_result is None
        else str(result.candidate_selection_result.run_id)
    )
    portfolio_run_id = (
        None if result.portfolio_result is None else str(result.portfolio_result.run_id)
    )
    print(
        "Selection/Portfolio: "
        f"candidate={candidate_run_id or 'skipped'} | "
        f"portfolio={portfolio_run_id or 'skipped'}"
    )

    candidate_review_dir = (
        None
        if result.candidate_review_result is None
        else str(result.candidate_review_result.review_dir)
    )
    print(
        "Review: "
        f"candidate_review={candidate_review_dir or 'skipped'} | "
        f"review_id={result.review_result.review_id}"
    )
    print(
        "Campaign Artifacts: "
        f"manifest={result.campaign_manifest_path.as_posix()} | "
        f"checkpoint={result.campaign_checkpoint_path.as_posix()} | "
        f"summary={result.campaign_summary_path.as_posix()}"
    )


def main() -> None:
    try:
        run_cli()
    except (ResearchCampaignConfigError, ValueError) as exc:
        print(_format_run_failure(exc), file=sys.stderr)
        raise SystemExit(1) from exc


def _format_run_failure(exc: Exception) -> str:
    message = str(exc).strip()
    if message.startswith("Run failed:"):
        return message
    return f"Run failed: {message}"


def _stage_state_counts(stage_statuses: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in stage_statuses.values():
        normalized = str(state or "unknown")
        counts[normalized] = counts.get(normalized, 0) + 1
    return {
        state: counts.get(state, 0)
        for state in ("completed", "reused", "failed", "partial", "skipped", "pending")
    }


def _artifact_reference_payload(reference: CampaignArtifactReference) -> dict[str, Any]:
    payload = {
        "stage_name": reference.stage_name,
        "source": reference.source,
        "run_id": reference.run_id,
        "artifact_dir": reference.artifact_dir.as_posix(),
        "registry_path": None if reference.registry_path is None else reference.registry_path.as_posix(),
        "match_criteria": canonicalize_value(reference.match_criteria),
        "metadata": canonicalize_value(reference.metadata),
    }
    return canonicalize_value(payload)


def _candidate_selection_registry_path_for_campaign(config: ResearchCampaignConfig) -> Path:
    if config.candidate_selection.output.registry_path is not None:
        return Path(config.candidate_selection.output.registry_path)
    return candidate_selection_registry_path(config.candidate_selection.output.path)


def _portfolio_registry_path(config: ResearchCampaignConfig) -> Path:
    return default_registry_path(Path(config.outputs.portfolio_artifacts_root))


def _candidate_selection_match_criteria(config: ResearchCampaignConfig) -> dict[str, Any]:
    candidate = config.candidate_selection
    return canonicalize_value(
        {
            "alpha_name": candidate.alpha_name,
            "dataset": candidate.dataset,
            "timeframe": candidate.timeframe,
            "evaluation_horizon": candidate.evaluation_horizon,
            "mapping_name": candidate.mapping_name,
        }
    )


def _portfolio_match_criteria(
    config: ResearchCampaignConfig,
    *,
    candidate_selection_reference: CampaignArtifactReference | None = None,
) -> dict[str, Any]:
    return canonicalize_value(
        {
            "portfolio_name": config.portfolio.portfolio_name,
            "timeframe": config.portfolio.timeframe,
            "candidate_selection_run_id": (
                None if candidate_selection_reference is None else candidate_selection_reference.run_id
            ),
        }
    )


def _validate_candidate_selection_resolution(config: ResearchCampaignConfig) -> tuple[bool, str]:
    try:
        reference = _resolve_candidate_selection_reference(config)
    except ValueError as exc:
        return False, str(exc)
    return (
        True,
        "Resolved candidate-selection registry input "
        f"'{reference.run_id}' from {reference.registry_path.as_posix()}."
        if reference.registry_path is not None
        else f"Resolved candidate-selection registry input '{reference.run_id}'.",
    )


def _validate_portfolio_resolution(config: ResearchCampaignConfig) -> tuple[bool, str]:
    try:
        reference = _resolve_portfolio_reference(config, candidate_selection_reference=None)
    except ValueError as exc:
        return False, str(exc)
    return (
        True,
        "Resolved portfolio registry input "
        f"'{reference.run_id}' from {reference.registry_path.as_posix()}."
        if reference.registry_path is not None
        else f"Resolved portfolio registry input '{reference.run_id}'.",
    )


def _can_resolve_candidate_selection_reference(config: ResearchCampaignConfig) -> bool:
    ok, _message = _validate_candidate_selection_resolution(config)
    return ok


def _can_resolve_portfolio_reference(config: ResearchCampaignConfig) -> bool:
    ok, _message = _validate_portfolio_resolution(config)
    return ok


def _resolve_candidate_selection_reference(config: ResearchCampaignConfig) -> CampaignArtifactReference:
    registry_path = _candidate_selection_registry_path_for_campaign(config)
    criteria = _candidate_selection_match_criteria(config)
    entries = _load_required_registry_entries(registry_path, run_type="candidate_selection")
    matches = _filter_candidate_selection_registry_entries(entries, criteria, registry_path=registry_path)
    selected = _select_single_registry_match(
        matches,
        stage_name="candidate_selection",
        registry_path=registry_path,
        criteria=criteria,
    )
    return CampaignArtifactReference(
        stage_name="candidate_selection",
        source="registry",
        run_id=str(selected["run_id"]),
        artifact_dir=Path(str(selected["artifact_dir"])),
        registry_path=registry_path,
        match_criteria=criteria,
        metadata=selected["metadata"],
    )


def _resolve_portfolio_reference(
    config: ResearchCampaignConfig,
    *,
    candidate_selection_reference: CampaignArtifactReference | None,
) -> CampaignArtifactReference:
    registry_path = _portfolio_registry_path(config)
    criteria = _portfolio_match_criteria(
        config,
        candidate_selection_reference=candidate_selection_reference,
    )
    entries = _load_required_registry_entries(registry_path, run_type="portfolio")
    matches = _filter_portfolio_registry_entries(entries, criteria, registry_path=registry_path)
    selected = _select_single_registry_match(
        matches,
        stage_name="portfolio",
        registry_path=registry_path,
        criteria=criteria,
    )
    return CampaignArtifactReference(
        stage_name="portfolio",
        source="registry",
        run_id=str(selected["run_id"]),
        artifact_dir=Path(str(selected["artifact_dir"])),
        registry_path=registry_path,
        match_criteria=criteria,
        metadata=selected["metadata"],
    )


def _load_required_registry_entries(registry_path: Path, *, run_type: str) -> list[dict[str, Any]]:
    try:
        entries = load_registry(registry_path)
    except Exception as exc:
        raise ValueError(f"Failed to load {run_type} registry {registry_path.as_posix()}: {exc}") from exc
    if not registry_path.exists():
        raise ValueError(f"{run_type} registry path does not exist: {registry_path.as_posix()}")
    filtered = [entry for entry in entries if str(entry.get("run_type") or "") == run_type]
    if not filtered:
        raise ValueError(f"No {run_type} entries found in registry: {registry_path.as_posix()}")
    return filtered


def _filter_candidate_selection_registry_entries(
    entries: Sequence[dict[str, Any]],
    criteria: Mapping[str, Any],
    *,
    registry_path: Path,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    incomplete_messages: list[str] = []
    for entry in entries:
        try:
            metadata = _candidate_selection_entry_metadata(entry)
        except ValueError as exc:
            incomplete_messages.append(str(exc))
            continue
        if not _entry_matches_candidate_selection_criteria(metadata, criteria):
            continue
        matches.append(
            {
                "run_id": str(entry.get("run_id")),
                "artifact_dir": metadata["artifact_dir"],
                "metadata": metadata,
            }
        )
    if not matches and incomplete_messages:
        formatted = " | ".join(sorted(set(incomplete_messages)))
        raise ValueError(
            "Candidate-selection registry state is incomplete for campaign chaining at "
            f"{registry_path.as_posix()}: {formatted}"
        )
    return matches


def _filter_portfolio_registry_entries(
    entries: Sequence[dict[str, Any]],
    criteria: Mapping[str, Any],
    *,
    registry_path: Path,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    incomplete_messages: list[str] = []
    for entry in entries:
        try:
            metadata = _portfolio_entry_metadata(entry)
        except ValueError as exc:
            incomplete_messages.append(str(exc))
            continue
        if not _entry_matches_portfolio_criteria(metadata, criteria):
            continue
        matches.append(
            {
                "run_id": str(entry.get("run_id")),
                "artifact_dir": metadata["artifact_dir"],
                "metadata": metadata,
            }
        )
    if not matches and incomplete_messages:
        formatted = " | ".join(sorted(set(incomplete_messages)))
        raise ValueError(
            "Portfolio registry state is incomplete for campaign chaining at "
            f"{registry_path.as_posix()}: {formatted}"
        )
    return matches


def _select_single_registry_match(
    matches: Sequence[dict[str, Any]],
    *,
    stage_name: str,
    registry_path: Path,
    criteria: Mapping[str, Any],
) -> dict[str, Any]:
    if not matches:
        raise ValueError(
            f"No {stage_name} registry entry matched campaign criteria "
            f"{serialize_criteria(criteria)} in {registry_path.as_posix()}."
        )
    ordered = sorted(
        matches,
        key=lambda match: (str(match["run_id"]), str(match["artifact_dir"])),
    )
    if len(ordered) > 1:
        run_ids = ", ".join(str(match["run_id"]) for match in ordered)
        raise ValueError(
            f"Ambiguous {stage_name} registry state for campaign criteria "
            f"{serialize_criteria(criteria)} in {registry_path.as_posix()}: {run_ids}"
        )
    return ordered[0]


def serialize_criteria(criteria: Mapping[str, Any]) -> str:
    items = [
        f"{key}={value}"
        for key, value in sorted(criteria.items())
        if value is not None
    ]
    return ", ".join(items) if items else "(no filters)"


def _candidate_selection_entry_metadata(entry: Mapping[str, Any]) -> dict[str, Any]:
    run_id = str(entry.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("candidate_selection entry missing run_id.")
    artifact_path = entry.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        raise ValueError(f"candidate_selection entry '{run_id}' missing artifact_path.")
    artifact_dir = Path(artifact_path)
    if not artifact_dir.exists():
        raise ValueError(
            f"candidate_selection entry '{run_id}' points to missing artifact dir {artifact_dir.as_posix()}."
        )

    manifest_path_text = entry.get("manifest_path")
    manifest_path = (
        Path(str(manifest_path_text))
        if isinstance(manifest_path_text, str) and manifest_path_text.strip()
        else artifact_dir / "manifest.json"
    )
    manifest = _read_json_mapping(
        manifest_path,
        missing_message=(
            f"candidate_selection entry '{run_id}' is incomplete: manifest missing at {manifest_path.as_posix()}."
        ),
        invalid_message=(
            f"candidate_selection entry '{run_id}' has invalid manifest at {manifest_path.as_posix()}."
        ),
    )
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    config_snapshot = manifest.get("config_snapshot")
    if not isinstance(config_snapshot, Mapping):
        config_snapshot = {}
    filters = config_snapshot.get("filters")
    if not isinstance(filters, Mapping):
        filters = {}

    mapping_names = provenance.get("mapping_names")
    if not isinstance(mapping_names, list):
        mapping_names = []
    normalized_mapping_names = sorted(
        {
            str(item).strip()
            for item in mapping_names
            if str(item).strip()
        }
    )
    return canonicalize_value(
        {
            "artifact_dir": artifact_dir.as_posix(),
            "manifest_path": manifest_path.as_posix(),
            "alpha_name": filters.get("alpha_name"),
            "dataset": entry.get("dataset", provenance.get("dataset")),
            "timeframe": entry.get("timeframe", provenance.get("timeframe")),
            "evaluation_horizon": entry.get(
                "evaluation_horizon",
                provenance.get("evaluation_horizon"),
            ),
            "mapping_names": normalized_mapping_names,
            "primary_metric": (
                entry.get("metadata", {}).get("primary_metric")
                if isinstance(entry.get("metadata"), Mapping)
                else None
            ),
            "upstream_alpha_run_ids": (
                provenance.get("upstream", {}).get("alpha_run_ids")
                if isinstance(provenance.get("upstream"), Mapping)
                else None
            ),
        }
    )


def _portfolio_entry_metadata(entry: Mapping[str, Any]) -> dict[str, Any]:
    run_id = str(entry.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("portfolio entry missing run_id.")
    artifact_path = entry.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        raise ValueError(f"portfolio entry '{run_id}' missing artifact_path.")
    artifact_dir = Path(artifact_path)
    if not artifact_dir.exists():
        raise ValueError(f"portfolio entry '{run_id}' points to missing artifact dir {artifact_dir.as_posix()}.")

    config = entry.get("config")
    if not isinstance(config, Mapping):
        config = {}
    candidate_provenance = config.get("candidate_selection_provenance")
    if not isinstance(candidate_provenance, Mapping):
        candidate_provenance = {}

    return canonicalize_value(
        {
            "artifact_dir": artifact_dir.as_posix(),
            "portfolio_name": entry.get("portfolio_name"),
            "timeframe": entry.get("timeframe"),
            "candidate_selection_run_id": candidate_provenance.get("run_id"),
            "component_run_ids": entry.get("component_run_ids"),
        }
    )


def _read_json_mapping(path: Path, *, missing_message: str, invalid_message: str) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(missing_message)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(invalid_message) from exc
    if not isinstance(payload, dict):
        raise ValueError(invalid_message)
    return payload


def _entry_matches_candidate_selection_criteria(
    metadata: Mapping[str, Any],
    criteria: Mapping[str, Any],
) -> bool:
    if criteria.get("alpha_name") is not None and metadata.get("alpha_name") != criteria.get("alpha_name"):
        return False
    if criteria.get("dataset") is not None and metadata.get("dataset") != criteria.get("dataset"):
        return False
    if criteria.get("timeframe") is not None and metadata.get("timeframe") != criteria.get("timeframe"):
        return False
    if criteria.get("evaluation_horizon") is not None and metadata.get("evaluation_horizon") != criteria.get("evaluation_horizon"):
        return False
    if criteria.get("mapping_name") is not None:
        mapping_names = metadata.get("mapping_names")
        if not isinstance(mapping_names, list) or criteria.get("mapping_name") not in mapping_names:
            return False
    return True


def _entry_matches_portfolio_criteria(
    metadata: Mapping[str, Any],
    criteria: Mapping[str, Any],
) -> bool:
    if criteria.get("portfolio_name") is not None and metadata.get("portfolio_name") != criteria.get("portfolio_name"):
        return False
    if criteria.get("timeframe") is not None and metadata.get("timeframe") != criteria.get("timeframe"):
        return False
    if (
        criteria.get("candidate_selection_run_id") is not None
        and metadata.get("candidate_selection_run_id") != criteria.get("candidate_selection_run_id")
    ):
        return False
    return True


def _run_preflight(
    config: ResearchCampaignConfig,
    *,
    campaign_run_id: str,
    campaign_artifact_dir: Path,
) -> CampaignPreflightResult:
    checks: list[CampaignPreflightCheck] = []
    alpha_catalog = {}
    strategy_catalog = {}
    portfolio_config_payload: dict[str, Any] = {}

    def add_check(
        check_id: str,
        ok: bool,
        message: str,
        **details: Any,
    ) -> None:
        checks.append(
            CampaignPreflightCheck(
                check_id=check_id,
                status="passed" if ok else "failed",
                message=message,
                details=_normalize_jsonable(details),
            )
        )

    alpha_names = config.targets.alpha_names
    strategy_names = config.targets.strategy_names

    add_check(
        "targets.present",
        bool(alpha_names or strategy_names),
        (
            "Campaign has at least one alpha or strategy target."
            if alpha_names or strategy_names
            else "Campaign requires at least one alpha or strategy target in targets.alpha_names or targets.strategy_names."
        ),
        alpha_target_count=len(alpha_names),
        strategy_target_count=len(strategy_names),
    )
    add_check(
        "candidate_selection.alpha_name",
        True,
        (
            "Candidate selection alpha_name filter resolved."
            if bool(config.candidate_selection.alpha_name)
            else "Candidate selection will evaluate the full campaign alpha universe for the configured dataset/timeframe filters."
        ),
        enabled=config.candidate_selection.enabled,
        alpha_name=config.candidate_selection.alpha_name,
    )
    add_check(
        "portfolio.portfolio_name",
        (not config.portfolio.enabled) or bool(config.portfolio.portfolio_name),
        (
            "Portfolio name resolved."
            if (not config.portfolio.enabled) or bool(config.portfolio.portfolio_name)
            else "Portfolio stage requires one resolved portfolio_name via portfolio.portfolio_name or a single targets.portfolio_names entry."
        ),
        enabled=config.portfolio.enabled,
        portfolio_name=config.portfolio.portfolio_name,
    )
    add_check(
        "portfolio.from_candidate_selection",
        (not config.portfolio.enabled)
        or (not config.portfolio.from_candidate_selection)
        or config.candidate_selection.enabled
        or _can_resolve_candidate_selection_reference(config),
        (
            "Portfolio candidate-selection dependency is satisfied."
            if (not config.portfolio.enabled)
            or (not config.portfolio.from_candidate_selection)
            or config.candidate_selection.enabled
            or _can_resolve_candidate_selection_reference(config)
            else (
                "portfolio.from_candidate_selection requires either candidate_selection.enabled "
                "or one unambiguous candidate-selection registry match."
            )
        ),
        portfolio_enabled=config.portfolio.enabled,
        from_candidate_selection=config.portfolio.from_candidate_selection,
        candidate_selection_enabled=config.candidate_selection.enabled,
    )
    add_check(
        "candidate_review.requires_portfolio",
        (not config.candidate_selection.enabled)
        or (not config.candidate_selection.execution.enable_review)
        or config.portfolio.enabled
        or _can_resolve_portfolio_reference(config),
        (
            "Candidate review dependency is satisfied."
            if (not config.candidate_selection.enabled)
            or (not config.candidate_selection.execution.enable_review)
            or config.portfolio.enabled
            or _can_resolve_portfolio_reference(config)
            else (
                "candidate_selection.execution.enable_review requires either portfolio.enabled "
                "or one unambiguous portfolio registry match."
            )
        ),
        candidate_selection_enabled=config.candidate_selection.enabled,
        enable_review=config.candidate_selection.execution.enable_review,
        portfolio_enabled=config.portfolio.enabled,
    )

    alpha_catalog_path = Path(config.targets.alpha_catalog_path)
    strategy_config_path = Path(config.targets.strategy_config_path)
    portfolio_config_path = Path(config.targets.portfolio_config_path)

    add_check(
        "paths.alpha_catalog",
        alpha_catalog_path.exists(),
        (
            f"targets.alpha_catalog_path exists: {alpha_catalog_path.as_posix()}"
            if alpha_catalog_path.exists()
            else f"targets.alpha_catalog_path does not exist: {alpha_catalog_path.as_posix()}"
        ),
        path=alpha_catalog_path,
    )
    if alpha_catalog_path.exists():
        try:
            alpha_catalog = load_alphas_config(alpha_catalog_path)
            add_check(
                "catalog.alpha.load",
                True,
                f"Loaded alpha catalog from {alpha_catalog_path.as_posix()}.",
                alpha_count=len(alpha_catalog),
            )
        except Exception as exc:
            add_check(
                "catalog.alpha.load",
                False,
                f"Failed to load alpha catalog: {exc}",
                path=alpha_catalog_path,
            )
    for alpha_name in alpha_names:
        add_check(
            f"catalog.alpha.target.{alpha_name}",
            alpha_name in alpha_catalog,
            (
                f"Resolved alpha target '{alpha_name}'."
                if alpha_name in alpha_catalog
                else f"Unknown alpha '{alpha_name}' in targets.alpha_names."
            ),
            alpha_name=alpha_name,
        )

    add_check(
        "paths.strategy_config",
        strategy_config_path.exists(),
        (
            f"targets.strategy_config_path exists: {strategy_config_path.as_posix()}"
            if strategy_config_path.exists()
            else f"targets.strategy_config_path does not exist: {strategy_config_path.as_posix()}"
        ),
        path=strategy_config_path,
    )
    if strategy_config_path.exists():
        try:
            strategy_catalog = run_strategy_cli.load_strategies_config(strategy_config_path)
            add_check(
                "catalog.strategy.load",
                True,
                f"Loaded strategy config from {strategy_config_path.as_posix()}.",
                strategy_count=len(strategy_catalog),
            )
        except Exception as exc:
            add_check(
                "catalog.strategy.load",
                False,
                f"Failed to load strategy config: {exc}",
                path=strategy_config_path,
            )
    for strategy_name in strategy_names:
        add_check(
            f"catalog.strategy.target.{strategy_name}",
            strategy_name in strategy_catalog,
            (
                f"Resolved strategy target '{strategy_name}'."
                if strategy_name in strategy_catalog
                else f"Unknown strategy '{strategy_name}' in targets.strategy_names."
            ),
            strategy_name=strategy_name,
        )

    require_portfolio_config = config.portfolio.enabled and not config.portfolio.from_candidate_selection
    add_check(
        "paths.portfolio_config",
        (not require_portfolio_config) or portfolio_config_path.exists(),
        (
            f"targets.portfolio_config_path exists: {portfolio_config_path.as_posix()}"
            if (not require_portfolio_config) or portfolio_config_path.exists()
            else f"targets.portfolio_config_path does not exist: {portfolio_config_path.as_posix()}"
        ),
        required=require_portfolio_config,
        path=portfolio_config_path,
    )
    if require_portfolio_config and portfolio_config_path.exists():
        try:
            portfolio_config_payload = run_portfolio_cli.load_portfolio_config(portfolio_config_path)
            add_check(
                "catalog.portfolio.load",
                True,
                f"Loaded portfolio config from {portfolio_config_path.as_posix()}.",
                path=portfolio_config_path,
            )
        except Exception as exc:
            add_check(
                "catalog.portfolio.load",
                False,
                f"Failed to load portfolio config: {exc}",
                path=portfolio_config_path,
            )
        if config.portfolio.portfolio_name is not None and portfolio_config_payload:
            try:
                run_portfolio_cli.resolve_portfolio_definition(
                    portfolio_config_payload,
                    portfolio_name=config.portfolio.portfolio_name,
                )
                add_check(
                    "catalog.portfolio.target",
                    True,
                    f"Resolved portfolio target '{config.portfolio.portfolio_name}'.",
                    portfolio_name=config.portfolio.portfolio_name,
                )
            except Exception as exc:
                add_check(
                    "catalog.portfolio.target",
                    False,
                    f"Failed to resolve portfolio target '{config.portfolio.portfolio_name}': {exc}",
                    portfolio_name=config.portfolio.portfolio_name,
                )

    feature_paths = FeaturePaths()
    dataset_consumers = _collect_required_datasets(
        config,
        alpha_catalog=alpha_catalog,
        strategy_catalog=strategy_catalog,
    )
    dataset_records: list[dict[str, Any]] = []
    for dataset_name, consumers in sorted(dataset_consumers.items()):
        parquet_count = 0
        dataset_root: Path | None = None
        dataset_ok = False
        message = ""
        try:
            dataset_root = feature_paths.dataset_root(dataset_name)
            dataset_ok = dataset_root.exists()
            if dataset_ok:
                parquet_count = sum(1 for _ in dataset_root.glob("**/*.parquet"))
                dataset_ok = parquet_count > 0
            message = (
                f"Dataset '{dataset_name}' is available."
                if dataset_ok
                else f"Dataset '{dataset_name}' has no parquet files under {dataset_root.as_posix()}."
            )
        except Exception as exc:
            message = f"Dataset '{dataset_name}' failed validation: {exc}"
        add_check(
            f"dataset.{dataset_name}",
            dataset_ok,
            message,
            dataset=dataset_name,
            consumers=consumers,
            dataset_root=None if dataset_root is None else dataset_root.as_posix(),
            parquet_file_count=parquet_count,
            features_root=feature_paths.root.as_posix(),
        )
        dataset_records.append(
            {
                "dataset": dataset_name,
                "consumers": list(consumers),
                "dataset_root": None if dataset_root is None else dataset_root.as_posix(),
                "parquet_file_count": parquet_count,
            }
        )

    artifact_roots = _campaign_artifact_roots(config)
    for label, root in artifact_roots.items():
        ok, message = _ensure_directory(root)
        add_check(
            f"artifacts.{label}",
            ok,
            message,
            path=root.as_posix(),
        )

    if config.candidate_selection.enabled and config.candidate_selection.execution.from_registry:
        registry_path = (
            Path(config.candidate_selection.output.registry_path)
            if config.candidate_selection.output.registry_path is not None
            else candidate_selection_registry_path(config.candidate_selection.output.path)
        )
        ok, message, entry_count = _validate_registry_path(registry_path)
        add_check(
            "registry.candidate_selection",
            ok,
            message,
            path=registry_path.as_posix(),
            entry_count=entry_count,
        )
        candidate_ok, candidate_message = _validate_candidate_selection_resolution(config)
        add_check(
            "registry.candidate_selection.resolution",
            candidate_ok,
            candidate_message,
            path=registry_path.as_posix(),
        )

    if config.portfolio.enabled and config.portfolio.from_candidate_selection and not config.candidate_selection.enabled:
        registry_path = (
            Path(config.candidate_selection.output.registry_path)
            if config.candidate_selection.output.registry_path is not None
            else candidate_selection_registry_path(config.candidate_selection.output.path)
        )
        candidate_ok, candidate_message = _validate_candidate_selection_resolution(config)
        add_check(
            "registry.candidate_selection.portfolio_input",
            candidate_ok,
            candidate_message,
            path=registry_path.as_posix(),
        )

    if config.portfolio.enabled and config.portfolio.from_registry and not strategy_names:
        strategy_registry_path = default_registry_path(STRATEGY_ARTIFACTS_ROOT)
        ok, message, entry_count = _validate_registry_path(strategy_registry_path)
        add_check(
            "registry.strategy",
            ok,
            message,
            path=strategy_registry_path.as_posix(),
            entry_count=entry_count,
        )

    if config.candidate_selection.enabled and config.candidate_selection.execution.enable_review and not config.portfolio.enabled:
        portfolio_registry_path = _portfolio_registry_path(config)
        portfolio_ok, portfolio_message = _validate_portfolio_resolution(config)
        add_check(
            "registry.portfolio.candidate_review",
            portfolio_ok,
            portfolio_message,
            path=portfolio_registry_path.as_posix(),
        )

    failed_checks = [check for check in checks if check.status == "failed"]
    summary = {
        "campaign_run_id": campaign_run_id,
        "status": "failed" if failed_checks else "passed",
        "campaign_artifact_dir": campaign_artifact_dir.as_posix(),
        "config_path": (campaign_artifact_dir / CAMPAIGN_CONFIG_FILENAME).as_posix(),
        "environment": {
            "cwd": Path.cwd().as_posix(),
            "features_root": feature_paths.root.as_posix(),
            "strategy_artifacts_root": STRATEGY_ARTIFACTS_ROOT.as_posix(),
            "alpha_registry_path": alpha_evaluation_registry_path(config.outputs.alpha_artifacts_root).as_posix(),
        },
        "targets": {
            "alpha_names": list(alpha_names),
            "strategy_names": list(strategy_names),
            "portfolio_names": list(config.targets.portfolio_names),
            "resolved_candidate_selection_alpha_name": config.candidate_selection.alpha_name,
            "resolved_portfolio_name": config.portfolio.portfolio_name,
        },
        "datasets": dataset_records,
        "artifact_roots": {
            key: value.as_posix()
            for key, value in artifact_roots.items()
        },
        "check_counts": {
            "total": len(checks),
            "passed": sum(1 for check in checks if check.status == "passed"),
            "failed": len(failed_checks),
        },
        "failed_checks": [check.check_id for check in failed_checks],
        "checks": [
            {
                "check_id": check.check_id,
                "status": check.status,
                "message": check.message,
                "details": check.details,
            }
            for check in checks
        ],
    }
    summary_path = campaign_artifact_dir / PREFLIGHT_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return CampaignPreflightResult(
        status=str(summary["status"]),
        summary_path=summary_path,
        summary=summary,
        checks=tuple(checks),
    )


def _run_alpha_research(config: ResearchCampaignConfig) -> list[Any]:
    results: list[Any] = []
    for alpha_name in config.targets.alpha_names:
        argv = ["--alpha-name", alpha_name, "--config", config.targets.alpha_catalog_path]
        if config.outputs.alpha_artifacts_root:
            argv.extend(["--artifacts-root", config.outputs.alpha_artifacts_root])
        if config.dataset_selection.dataset is not None:
            argv.extend(["--dataset", config.dataset_selection.dataset])
        if config.time_windows.start is not None:
            argv.extend(["--start", config.time_windows.start])
        if config.time_windows.end is not None:
            argv.extend(["--end", config.time_windows.end])
        if config.time_windows.train_start is not None:
            argv.extend(["--train-start", config.time_windows.train_start])
        if config.time_windows.train_end is not None:
            argv.extend(["--train-end", config.time_windows.train_end])
        if config.time_windows.predict_start is not None:
            argv.extend(["--predict-start", config.time_windows.predict_start])
        if config.time_windows.predict_end is not None:
            argv.extend(["--predict-end", config.time_windows.predict_end])
        if config.dataset_selection.tickers_path is not None:
            argv.extend(["--tickers", config.dataset_selection.tickers_path])
        results.append(run_alpha_cli.run_cli(argv))
    return results


def _run_strategy_research(config: ResearchCampaignConfig) -> list[Any]:
    results: list[Any] = []
    with _strategy_config_override(Path(config.targets.strategy_config_path)):
        for strategy_name in config.targets.strategy_names:
            argv = ["--strategy", strategy_name]
            if config.time_windows.start is not None:
                argv.extend(["--start", config.time_windows.start])
            if config.time_windows.end is not None:
                argv.extend(["--end", config.time_windows.end])
            results.append(run_strategy_cli.run_cli(argv))
    return results


def _run_alpha_comparison(config: ResearchCampaignConfig) -> Any | None:
    if not config.targets.alpha_names:
        return None

    argv = ["--from-registry", "--view", config.comparison.alpha_view, "--metric", config.comparison.alpha_metric]
    argv.extend(["--sleeve-metric", config.comparison.alpha_sleeve_metric])
    if len(config.targets.alpha_names) == 1:
        argv.extend(["--alpha-name", config.targets.alpha_names[0]])
    if config.dataset_selection.dataset is not None:
        argv.extend(["--dataset", config.dataset_selection.dataset])
    if config.dataset_selection.timeframe is not None:
        argv.extend(["--timeframe", config.dataset_selection.timeframe])
    if config.dataset_selection.evaluation_horizon is not None:
        argv.extend(
            [
                "--evaluation-horizon",
                str(config.dataset_selection.evaluation_horizon),
            ]
        )
    if config.dataset_selection.mapping_name is not None:
        argv.extend(["--mapping-name", config.dataset_selection.mapping_name])
    if config.outputs.alpha_artifacts_root:
        argv.extend(["--artifacts-root", config.outputs.alpha_artifacts_root])
    output_path = _comparison_output_path(config, stage_name="alpha")
    if output_path is not None:
        argv.extend(["--output-path", output_path.as_posix()])
    return compare_alpha_cli.run_cli(argv)


def _run_strategy_comparison(config: ResearchCampaignConfig) -> Any | None:
    if not config.targets.strategy_names:
        return None

    argv = [
        "--strategies",
        *config.targets.strategy_names,
        "--metric",
        config.comparison.strategy_metric,
    ]
    if config.comparison.top_k is not None:
        argv.extend(["--top-k", str(config.comparison.top_k)])
    if config.comparison.from_registry:
        argv.append("--from-registry")
    else:
        if config.time_windows.start is not None:
            argv.extend(["--start", config.time_windows.start])
        if config.time_windows.end is not None:
            argv.extend(["--end", config.time_windows.end])
    output_path = _comparison_output_path(config, stage_name="strategy")
    if output_path is not None:
        argv.extend(["--output-path", output_path.as_posix()])
    with _strategy_config_override(Path(config.targets.strategy_config_path)):
        return compare_strategies_cli.run_cli(argv)


def _run_candidate_selection(
    config: ResearchCampaignConfig,
) -> tuple[Any, CampaignArtifactReference | None]:
    candidate = config.candidate_selection
    argv = [
        "--artifacts-root",
        candidate.artifacts_root,
        "--metric",
        candidate.metric,
        "--output-path",
        candidate.output.path,
    ]
    if candidate.alpha_name is not None:
        argv.extend(["--alpha-name", candidate.alpha_name])
    if candidate.dataset is not None:
        argv.extend(["--dataset", candidate.dataset])
    if candidate.timeframe is not None:
        argv.extend(["--timeframe", candidate.timeframe])
    if candidate.evaluation_horizon is not None:
        argv.extend(["--evaluation-horizon", str(candidate.evaluation_horizon)])
    if candidate.mapping_name is not None:
        argv.extend(["--mapping-name", candidate.mapping_name])
    if candidate.max_candidates is not None:
        argv.extend(["--max-candidates", str(candidate.max_candidates)])
    if candidate.eligibility.min_mean_ic is not None:
        argv.extend(["--min-ic", str(candidate.eligibility.min_mean_ic)])
    if candidate.eligibility.min_mean_rank_ic is not None:
        argv.extend(["--min-rank-ic", str(candidate.eligibility.min_mean_rank_ic)])
    if candidate.eligibility.min_ic_ir is not None:
        argv.extend(["--min-ic-ir", str(candidate.eligibility.min_ic_ir)])
    if candidate.eligibility.min_rank_ic_ir is not None:
        argv.extend(["--min-rank-ic-ir", str(candidate.eligibility.min_rank_ic_ir)])
    if candidate.eligibility.min_history_length is not None:
        argv.extend(
            [
                "--min-history-length",
                str(candidate.eligibility.min_history_length),
            ]
        )
    if candidate.eligibility.min_coverage is not None:
        argv.extend(["--min-coverage", str(candidate.eligibility.min_coverage)])
    if candidate.redundancy.max_pairwise_correlation is not None:
        argv.extend(
            [
                "--max-pairwise-correlation",
                str(candidate.redundancy.max_pairwise_correlation),
            ]
        )
    if candidate.redundancy.min_overlap_observations is not None:
        argv.extend(
            [
                "--min-overlap-observations",
                str(candidate.redundancy.min_overlap_observations),
            ]
        )
    argv.extend(["--allocation-method", candidate.allocation.allocation_method])
    if candidate.allocation.max_weight_per_candidate is not None:
        argv.extend(
            [
                "--max-weight-per-candidate",
                str(candidate.allocation.max_weight_per_candidate),
            ]
        )
    if candidate.allocation.min_allocation_candidate_count is not None:
        argv.extend(
            [
                "--min-allocation-candidate-count",
                str(candidate.allocation.min_allocation_candidate_count),
            ]
        )
    if candidate.allocation.min_allocation_weight is not None:
        argv.extend(
            [
                "--min-allocation-weight",
                str(candidate.allocation.min_allocation_weight),
            ]
        )
    argv.extend(
        [
            "--allocation-weight-sum-tolerance",
            str(candidate.allocation.allocation_weight_sum_tolerance),
            "--allocation-rounding-decimals",
            str(candidate.allocation.allocation_rounding_decimals),
        ]
    )
    if candidate.execution.strict_mode:
        argv.append("--strict")
    if candidate.execution.skip_eligibility:
        argv.append("--skip-eligibility")
    if candidate.execution.skip_redundancy:
        argv.append("--skip-redundancy")
    if candidate.execution.skip_allocation:
        argv.append("--skip-allocation")
    reference = None
    if candidate.execution.from_registry:
        reference = _resolve_candidate_selection_reference(config)
        argv.extend(["--candidate-selection-path", reference.artifact_dir.as_posix()])
    if candidate.execution.register_run:
        argv.append("--register-run")
    if candidate.execution.no_markdown_review:
        argv.append("--no-markdown-review")
    if candidate.output.registry_path is not None:
        argv.extend(["--registry-path", candidate.output.registry_path])
    return run_candidate_selection_cli.run_cli(argv), reference


def _run_portfolio(
    config: ResearchCampaignConfig,
    *,
    candidate_selection_result: Any | None,
    candidate_selection_reference: CampaignArtifactReference | None,
) -> tuple[Any, CampaignArtifactReference | None, CampaignArtifactReference | None]:
    portfolio = config.portfolio
    argv = [
        "--portfolio-name",
        str(portfolio.portfolio_name),
        "--timeframe",
        str(portfolio.timeframe),
    ]
    portfolio_input_reference = None
    if portfolio.from_candidate_selection:
        if candidate_selection_result is None and candidate_selection_reference is None:
            candidate_selection_reference = _resolve_candidate_selection_reference(config)
        argv.extend(
            [
                "--from-candidate-selection",
                (
                    Path(candidate_selection_result.artifact_dir).as_posix()
                    if candidate_selection_result is not None
                    else candidate_selection_reference.artifact_dir.as_posix()
                ),
            ]
        )
        portfolio_input_reference = (
            candidate_selection_reference
            if candidate_selection_reference is not None
            else CampaignArtifactReference(
                stage_name="candidate_selection",
                source="same_campaign",
                run_id=str(candidate_selection_result.run_id),
                artifact_dir=Path(candidate_selection_result.artifact_dir),
                metadata={"consumer": "portfolio"},
            )
        )
    else:
        argv.extend(["--portfolio-config", config.targets.portfolio_config_path])
        if portfolio.from_registry:
            argv.append("--from-registry")
    if portfolio.evaluation_path is not None:
        argv.extend(["--evaluation", portfolio.evaluation_path])
    if portfolio.optimizer_method is not None:
        argv.extend(["--optimizer-method", portfolio.optimizer_method])
    if config.outputs.portfolio_artifacts_root:
        argv.extend(["--output-dir", config.outputs.portfolio_artifacts_root])
    return run_portfolio_cli.run_cli(argv), portfolio_input_reference, candidate_selection_reference


def _run_candidate_review(
    config: ResearchCampaignConfig,
    *,
    candidate_selection_result: Any | None,
    candidate_selection_reference: CampaignArtifactReference | None,
    portfolio_result: Any | None,
    portfolio_reference: CampaignArtifactReference | None,
) -> Any:
    resolved_candidate_dir = (
        None
        if candidate_selection_result is None
        else Path(candidate_selection_result.artifact_dir)
    )
    if resolved_candidate_dir is None and candidate_selection_reference is not None:
        resolved_candidate_dir = candidate_selection_reference.artifact_dir

    resolved_portfolio_dir = (
        None
        if portfolio_result is None
        else Path(portfolio_result.experiment_dir)
    )
    if resolved_portfolio_dir is None:
        if portfolio_reference is None:
            portfolio_reference = _resolve_portfolio_reference(
                config,
                candidate_selection_reference=candidate_selection_reference,
            )
        resolved_portfolio_dir = portfolio_reference.artifact_dir

    if resolved_candidate_dir is None or resolved_portfolio_dir is None:
        raise ValueError("Candidate review requires resolved candidate selection and portfolio artifacts.")

    argv = [
        "--candidate-selection-path",
        resolved_candidate_dir.as_posix(),
        "--portfolio-path",
        resolved_portfolio_dir.as_posix(),
    ]
    review_output_path = config.candidate_selection.output.review_output_path
    if review_output_path is not None:
        argv.extend(["--output-path", review_output_path])
    if config.candidate_selection.execution.no_markdown_review:
        argv.append("--no-markdown-report")
    return review_candidate_selection_cli.run_cli(argv)


def _run_research_review(config: ResearchCampaignConfig) -> Any:
    review = config.review
    argv = ["--from-registry"]
    if review.filters.run_types:
        argv.extend(["--run-types", *review.filters.run_types])
    if review.filters.timeframe is not None:
        argv.extend(["--timeframe", review.filters.timeframe])
    if review.filters.dataset is not None:
        argv.extend(["--dataset", review.filters.dataset])
    if review.filters.alpha_name is not None:
        argv.extend(["--alpha-name", review.filters.alpha_name])
    if review.filters.strategy_name is not None:
        argv.extend(["--strategy-name", review.filters.strategy_name])
    if review.filters.portfolio_name is not None:
        argv.extend(["--portfolio-name", review.filters.portfolio_name])
    if review.filters.top_k_per_type is not None:
        argv.extend(["--top-k", str(review.filters.top_k_per_type)])
    argv.extend(["--alpha-artifacts-root", config.outputs.alpha_artifacts_root])
    argv.extend(["--strategy-artifacts-root", str(STRATEGY_ARTIFACTS_ROOT)])
    argv.extend(["--portfolio-artifacts-root", config.outputs.portfolio_artifacts_root])
    argv.extend(
        [
            "--alpha-metric",
            review.ranking.alpha_evaluation_primary_metric,
            "--alpha-secondary-metric",
            review.ranking.alpha_evaluation_secondary_metric,
            "--strategy-metric",
            review.ranking.strategy_primary_metric,
            "--strategy-secondary-metric",
            review.ranking.strategy_secondary_metric,
            "--portfolio-metric",
            review.ranking.portfolio_primary_metric,
            "--portfolio-secondary-metric",
            review.ranking.portfolio_secondary_metric,
        ]
    )
    if review.output.path is not None:
        argv.extend(["--output-path", review.output.path])
    if not review.output.emit_plots:
        argv.append("--disable-plots")
    return compare_research_cli.run_cli(argv)


def _require_existing_path(path_text: str, *, field_name: str) -> None:
    path = Path(path_text)
    if not path.exists():
        raise ValueError(f"{field_name} does not exist: {path.as_posix()}")


def _comparison_output_path(
    config: ResearchCampaignConfig,
    *,
    stage_name: str,
) -> Path | None:
    if config.outputs.comparison_output_path is None:
        return None
    return Path(config.outputs.comparison_output_path) / stage_name


def _build_campaign_run_id(config: ResearchCampaignConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"research_campaign_{digest}"


def _campaign_artifact_dir(config: ResearchCampaignConfig, *, campaign_run_id: str) -> Path:
    return Path(config.outputs.campaign_artifacts_root) / campaign_run_id


def _persist_campaign_config(campaign_artifact_dir: Path, config: ResearchCampaignConfig) -> Path:
    campaign_artifact_dir.mkdir(parents=True, exist_ok=True)
    path = campaign_artifact_dir / CAMPAIGN_CONFIG_FILENAME
    path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_existing_campaign_checkpoint(
    *,
    campaign_artifact_dir: Path,
    campaign_run_id: str,
) -> dict[str, Any] | None:
    checkpoint_path = campaign_artifact_dir / CAMPAIGN_CHECKPOINT_FILENAME
    if not checkpoint_path.exists():
        return None
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to load existing campaign checkpoint {checkpoint_path.as_posix()}: {exc}") from exc
    try:
        validate_campaign_checkpoint_payload(payload)
    except CampaignCheckpointError as exc:
        raise ValueError(f"Invalid existing campaign checkpoint {checkpoint_path.as_posix()}: {exc}") from exc
    if str(payload.get("campaign_run_id")) != campaign_run_id:
        return None
    return canonicalize_value(dict(payload))


def _checkpoint_stage(
    checkpoint_payload: Mapping[str, Any] | None,
    stage_name: str,
) -> dict[str, Any] | None:
    if checkpoint_payload is None:
        return None
    for stage in checkpoint_payload.get("stages", []):
        if str(stage.get("stage_name")) == stage_name:
            return canonicalize_value(dict(stage))
    return None


def _can_reuse_checkpoint_stage(
    checkpoint_stage: Mapping[str, Any] | None,
    *,
    input_fingerprint: str,
) -> bool:
    if checkpoint_stage is None:
        return False
    state = str(checkpoint_stage.get("state") or "")
    if state not in {"completed", "reused"}:
        return False
    return str(checkpoint_stage.get("input_fingerprint") or "") == input_fingerprint


def _stage_reuse_decision(
    *,
    stage_name: str,
    checkpoint_stage: Mapping[str, Any] | None,
    input_fingerprint: str,
    reuse_policy: CampaignReusePolicyConfig,
    invalidated_by_stage: str | None,
) -> CampaignStageReuseDecision:
    matched_checkpoint = checkpoint_stage is not None
    checkpoint_state = None if checkpoint_stage is None else str(checkpoint_stage.get("state") or "") or None
    checkpoint_input_fingerprint = (
        None if checkpoint_stage is None else str(checkpoint_stage.get("input_fingerprint") or "") or None
    )
    fingerprint_match = checkpoint_input_fingerprint == input_fingerprint

    if invalidated_by_stage is not None:
        return CampaignStageReuseDecision(
            stage_name=stage_name,
            action="rerun",
            reason=f"Rerun required because upstream stage '{invalidated_by_stage}' invalidated downstream reuse.",
            matched_checkpoint=matched_checkpoint,
            checkpoint_state=checkpoint_state,
            checkpoint_input_fingerprint=checkpoint_input_fingerprint,
            fingerprint_match=fingerprint_match,
            invalidated_by_stage=invalidated_by_stage,
        )
    if stage_name in reuse_policy.force_rerun_stages:
        return CampaignStageReuseDecision(
            stage_name=stage_name,
            action="rerun",
            reason="Rerun required by reuse_policy.force_rerun_stages.",
            matched_checkpoint=matched_checkpoint,
            checkpoint_state=checkpoint_state,
            checkpoint_input_fingerprint=checkpoint_input_fingerprint,
            fingerprint_match=fingerprint_match,
        )
    if not reuse_policy.enable_checkpoint_reuse:
        return CampaignStageReuseDecision(
            stage_name=stage_name,
            action="rerun",
            reason="Checkpoint reuse disabled by reuse_policy.enable_checkpoint_reuse=false.",
            matched_checkpoint=matched_checkpoint,
            checkpoint_state=checkpoint_state,
            checkpoint_input_fingerprint=checkpoint_input_fingerprint,
            fingerprint_match=fingerprint_match,
        )
    if stage_name not in reuse_policy.reuse_prior_stages:
        return CampaignStageReuseDecision(
            stage_name=stage_name,
            action="rerun",
            reason="Stage not listed in reuse_policy.reuse_prior_stages.",
            matched_checkpoint=matched_checkpoint,
            checkpoint_state=checkpoint_state,
            checkpoint_input_fingerprint=checkpoint_input_fingerprint,
            fingerprint_match=fingerprint_match,
        )
    if checkpoint_stage is None:
        return CampaignStageReuseDecision(
            stage_name=stage_name,
            action="rerun",
            reason="No persisted checkpoint stage is available to reuse.",
            matched_checkpoint=False,
            checkpoint_state=None,
            checkpoint_input_fingerprint=None,
            fingerprint_match=False,
        )
    if checkpoint_state not in {"completed", "reused"}:
        return CampaignStageReuseDecision(
            stage_name=stage_name,
            action="rerun",
            reason=f"Checkpoint stage state '{checkpoint_state}' is not reusable.",
            matched_checkpoint=True,
            checkpoint_state=checkpoint_state,
            checkpoint_input_fingerprint=checkpoint_input_fingerprint,
            fingerprint_match=fingerprint_match,
        )
    if not fingerprint_match:
        return CampaignStageReuseDecision(
            stage_name=stage_name,
            action="rerun",
            reason="Checkpoint input fingerprint does not match the current effective stage inputs.",
            matched_checkpoint=True,
            checkpoint_state=checkpoint_state,
            checkpoint_input_fingerprint=checkpoint_input_fingerprint,
            fingerprint_match=False,
        )
    return CampaignStageReuseDecision(
        stage_name=stage_name,
        action="reuse",
        reason="Reused persisted stage outputs from a matching checkpoint.",
        matched_checkpoint=True,
        checkpoint_state=checkpoint_state,
        checkpoint_input_fingerprint=checkpoint_input_fingerprint,
        fingerprint_match=True,
    )


def _restore_stage_record(
    checkpoint_stage: Mapping[str, Any],
    *,
    reuse_decision: CampaignStageReuseDecision,
) -> CampaignStageRecord:
    return CampaignStageRecord(
        stage_name=str(checkpoint_stage["stage_name"]),
        status="reused",
        details=_stage_details_with_retry_metadata(
            checkpoint_stage.get("details") or {},
            existing_stage=None,
            reuse_decision=reuse_decision,
        ),
    )


def _update_downstream_invalidation(
    *,
    current_invalidated_by_stage: str | None,
    stage_name: str,
    stage_reuse_decision: CampaignStageReuseDecision,
    reuse_policy: CampaignReusePolicyConfig,
) -> str | None:
    if current_invalidated_by_stage is not None:
        return current_invalidated_by_stage
    if stage_reuse_decision.action != "rerun":
        return None
    if stage_name not in reuse_policy.invalidate_downstream_after_stages:
        return None
    return stage_name


def _compute_stage_input_fingerprint(
    *,
    stage_name: str,
    config: ResearchCampaignConfig,
    alpha_results: Sequence[Any],
    strategy_results: Sequence[Any],
    candidate_selection_result: Any | None,
    candidate_selection_reference: CampaignArtifactReference | None,
    portfolio_result: Any | None,
    portfolio_reference: CampaignArtifactReference | None,
    stage_input_fingerprints: Mapping[str, str | None],
) -> tuple[str, dict[str, Any]]:
    fingerprint_inputs = _campaign_stage_fingerprint_inputs(
        stage_name=stage_name,
        config=config,
        alpha_results=alpha_results,
        strategy_results=strategy_results,
        candidate_selection_result=candidate_selection_result,
        candidate_selection_reference=candidate_selection_reference,
        portfolio_result=portfolio_result,
        portfolio_reference=portfolio_reference,
        stage_input_fingerprints=stage_input_fingerprints,
    )
    return _fingerprint_mapping(fingerprint_inputs), fingerprint_inputs


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _artifact_reference_from_payload(payload: Any) -> CampaignArtifactReference | None:
    if not isinstance(payload, Mapping):
        return None
    artifact_dir = _optional_path(payload.get("artifact_dir"))
    if artifact_dir is None:
        return None
    registry_path = _optional_path(payload.get("registry_path"))
    return CampaignArtifactReference(
        stage_name=str(payload.get("stage_name") or ""),
        source=str(payload.get("source") or ""),
        run_id=str(payload.get("run_id") or ""),
        artifact_dir=artifact_dir,
        registry_path=registry_path,
        match_criteria=canonicalize_value(dict(payload.get("match_criteria") or {})),
        metadata=canonicalize_value(dict(payload.get("metadata") or {})),
    )


def _restore_preflight_result(
    *,
    campaign_artifact_dir: Path,
    checkpoint_stage: Mapping[str, Any],
) -> CampaignPreflightResult:
    summary_path = campaign_artifact_dir / PREFLIGHT_SUMMARY_FILENAME
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = canonicalize_value(dict(checkpoint_stage.get("details") or {}))
    checks = tuple(
        CampaignPreflightCheck(
            check_id=str(check.get("check_id") or ""),
            status=str(check.get("status") or ""),
            message=str(check.get("message") or ""),
            details=canonicalize_value(dict(check.get("details") or {})),
        )
        for check in summary.get("checks", [])
        if isinstance(check, Mapping)
    )
    return CampaignPreflightResult(
        status=str(summary.get("status") or "unknown"),
        summary_path=summary_path,
        summary=canonicalize_value(dict(summary)),
        checks=checks,
    )


def _restore_research_results(checkpoint_stage: Mapping[str, Any]) -> tuple[list[Any], list[Any]]:
    key_metrics = checkpoint_stage.get("key_metrics") or {}
    output_paths = checkpoint_stage.get("output_paths") or {}
    alpha_paths = list(output_paths.get("alpha_artifact_dirs") or [])
    strategy_paths = list(output_paths.get("strategy_artifact_dirs") or [])
    alpha_results = [
        SimpleNamespace(
            alpha_name=entry.get("alpha_name"),
            run_id=entry.get("run_id"),
            artifact_dir=_optional_path(alpha_paths[index]) if index < len(alpha_paths) else None,
            evaluation=SimpleNamespace(
                evaluation_result=SimpleNamespace(
                    summary={
                        "mean_ic": entry.get("mean_ic"),
                        "ic_ir": entry.get("ic_ir"),
                        "n_periods": entry.get("n_periods"),
                    }
                ),
                manifest={
                    "sleeve": {
                        "metric_summary": {
                            "sharpe_ratio": entry.get("sleeve_sharpe_ratio"),
                            "total_return": entry.get("sleeve_total_return"),
                        }
                    }
                },
            ),
        )
        for index, entry in enumerate(list(key_metrics.get("alpha_runs") or []))
        if isinstance(entry, Mapping)
    ]
    strategy_results = [
        SimpleNamespace(
            strategy_name=entry.get("strategy_name"),
            run_id=entry.get("run_id"),
            experiment_dir=_optional_path(strategy_paths[index]) if index < len(strategy_paths) else None,
            metrics={
                "cumulative_return": entry.get("cumulative_return"),
                "sharpe_ratio": entry.get("sharpe_ratio"),
                "max_drawdown": entry.get("max_drawdown"),
            },
        )
        for index, entry in enumerate(list(key_metrics.get("strategy_runs") or []))
        if isinstance(entry, Mapping)
    ]
    return alpha_results, strategy_results


def _restore_comparison_results(checkpoint_stage: Mapping[str, Any]) -> tuple[Any | None, Any | None]:
    selected_run_ids = checkpoint_stage.get("selected_run_ids") or {}
    output_paths = checkpoint_stage.get("output_paths") or {}
    alpha_result = None
    if selected_run_ids.get("alpha_comparison_id") is not None or output_paths.get("alpha_comparison_csv") is not None:
        alpha_result = SimpleNamespace(
            comparison_id=selected_run_ids.get("alpha_comparison_id"),
            csv_path=_optional_path(output_paths.get("alpha_comparison_csv")),
            json_path=_optional_path(output_paths.get("alpha_comparison_summary")),
        )
    strategy_result = None
    if (
        selected_run_ids.get("strategy_comparison_id") is not None
        or output_paths.get("strategy_comparison_csv") is not None
    ):
        strategy_result = SimpleNamespace(
            comparison_id=selected_run_ids.get("strategy_comparison_id"),
            csv_path=_optional_path(output_paths.get("strategy_comparison_csv")),
            json_path=_optional_path(output_paths.get("strategy_comparison_summary")),
        )
    return alpha_result, strategy_result


def _restore_candidate_selection_stage(checkpoint_stage: Mapping[str, Any]) -> tuple[Any | None, CampaignArtifactReference | None]:
    selected_run_ids = checkpoint_stage.get("selected_run_ids") or {}
    key_metrics = checkpoint_stage.get("key_metrics") or {}
    output_paths = checkpoint_stage.get("output_paths") or {}
    details = checkpoint_stage.get("details") or {}
    artifact_dir = _optional_path(output_paths.get("artifact_dir"))
    run_id = selected_run_ids.get("candidate_selection_run_id")
    if run_id is None or artifact_dir is None:
        return None, None
    result = SimpleNamespace(
        run_id=run_id,
        artifact_dir=artifact_dir,
        summary_json=_optional_path(output_paths.get("summary_json")),
        manifest_json=_optional_path(output_paths.get("manifest_json")),
        primary_metric=key_metrics.get("primary_metric"),
        universe_count=key_metrics.get("universe_count"),
        eligible_count=key_metrics.get("eligible_count"),
        selected_count=key_metrics.get("selected_count"),
        rejected_count=key_metrics.get("rejected_count"),
        pruned_by_redundancy=key_metrics.get("pruned_by_redundancy"),
    )
    return result, _artifact_reference_from_payload(details.get("input_reference"))


def _restore_portfolio_stage(checkpoint_stage: Mapping[str, Any]) -> tuple[Any | None, CampaignArtifactReference | None]:
    selected_run_ids = checkpoint_stage.get("selected_run_ids") or {}
    key_metrics = checkpoint_stage.get("key_metrics") or {}
    output_paths = checkpoint_stage.get("output_paths") or {}
    details = checkpoint_stage.get("details") or {}
    artifact_dir = _optional_path(output_paths.get("artifact_dir"))
    run_id = selected_run_ids.get("portfolio_run_id")
    if run_id is None or artifact_dir is None:
        return None, None
    result = SimpleNamespace(
        run_id=run_id,
        experiment_dir=artifact_dir,
        portfolio_name=key_metrics.get("portfolio_name"),
        component_count=key_metrics.get("component_count"),
        metrics={
            "total_return": key_metrics.get("total_return"),
            "sharpe_ratio": key_metrics.get("sharpe_ratio"),
            "max_drawdown": key_metrics.get("max_drawdown"),
        },
    )
    return result, _artifact_reference_from_payload(details.get("input_reference"))


def _restore_candidate_review_result(checkpoint_stage: Mapping[str, Any]) -> Any | None:
    selected_run_ids = checkpoint_stage.get("selected_run_ids") or {}
    output_paths = checkpoint_stage.get("output_paths") or {}
    outcomes = checkpoint_stage.get("outcomes") or {}
    review_dir = _optional_path(output_paths.get("review_dir"))
    if review_dir is None:
        return None
    return SimpleNamespace(
        review_dir=review_dir,
        candidate_review_summary_json=_optional_path(output_paths.get("summary_json")),
        manifest_json=_optional_path(output_paths.get("manifest_json")),
        candidate_selection_run_id=selected_run_ids.get("candidate_selection_run_id"),
        portfolio_run_id=selected_run_ids.get("portfolio_run_id"),
        total_candidates=outcomes.get("total_candidates"),
        selected_candidates=outcomes.get("selected_candidates"),
        rejected_candidates=outcomes.get("rejected_candidates"),
    )


def _restore_review_result(checkpoint_stage: Mapping[str, Any]) -> Any | None:
    selected_run_ids = checkpoint_stage.get("selected_run_ids") or {}
    key_metrics = checkpoint_stage.get("key_metrics") or {}
    output_paths = checkpoint_stage.get("output_paths") or {}
    review_id = selected_run_ids.get("review_id")
    if review_id is None:
        return None
    counts_by_run_type = key_metrics.get("counts_by_run_type") or {}
    entries = [
        SimpleNamespace(run_type=run_type)
        for run_type, count in sorted(counts_by_run_type.items())
        for _ in range(int(count or 0))
    ]
    return SimpleNamespace(
        review_id=review_id,
        csv_path=_optional_path(output_paths.get("leaderboard_csv")),
        json_path=_optional_path(output_paths.get("summary_json")),
        manifest_path=_optional_path(output_paths.get("manifest_json")),
        promotion_gate_path=_optional_path(output_paths.get("promotion_gates_json")),
        entries=entries,
    )


def _fingerprint_mapping(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(canonicalize_value(dict(payload)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _campaign_stage_fingerprint_inputs(
    *,
    stage_name: str,
    config: ResearchCampaignConfig,
    alpha_results: Sequence[Any],
    strategy_results: Sequence[Any],
    candidate_selection_result: Any | None,
    candidate_selection_reference: CampaignArtifactReference | None,
    portfolio_result: Any | None,
    portfolio_reference: CampaignArtifactReference | None,
    stage_input_fingerprints: Mapping[str, str | None],
) -> dict[str, Any]:
    if stage_name == "preflight":
        payload = {"config": config.to_dict()}
    elif stage_name == "research":
        payload = {
            "dataset_selection": config.dataset_selection.to_dict(),
            "time_windows": config.time_windows.to_dict(),
            "targets": {
                "alpha_names": list(config.targets.alpha_names),
                "strategy_names": list(config.targets.strategy_names),
                "alpha_catalog_path": config.targets.alpha_catalog_path,
                "strategy_config_path": config.targets.strategy_config_path,
            },
            "outputs": {
                "alpha_artifacts_root": config.outputs.alpha_artifacts_root,
            },
        }
    elif stage_name == "comparison":
        payload = {
            "comparison": config.comparison.to_dict(),
            "targets": {
                "alpha_names": list(config.targets.alpha_names),
                "strategy_names": list(config.targets.strategy_names),
            },
            "research_input_fingerprint": stage_input_fingerprints.get("research"),
        }
    elif stage_name == "candidate_selection":
        payload = {
            "candidate_selection": _candidate_selection_fingerprint_config(config),
            "research_input_fingerprint": stage_input_fingerprints.get("research"),
            "resolved_input_reference": _reference_fingerprint_payload(candidate_selection_reference),
            "same_campaign_alpha_run_ids": sorted(str(result.run_id) for result in alpha_results),
        }
    elif stage_name == "portfolio":
        payload = {
            "portfolio": config.portfolio.to_dict(),
            "candidate_selection_input_fingerprint": stage_input_fingerprints.get("candidate_selection"),
            "resolved_candidate_selection_reference": _reference_fingerprint_payload(candidate_selection_reference),
            "same_campaign_candidate_selection_run_id": _string_or_none(candidate_selection_result, "run_id"),
        }
    elif stage_name == "candidate_review":
        payload = {
            "candidate_review": {
                "enabled": bool(
                    config.candidate_selection.enabled and config.candidate_selection.execution.enable_review
                ),
                "no_markdown_review": config.candidate_selection.execution.no_markdown_review,
            },
            "candidate_selection_input_fingerprint": stage_input_fingerprints.get("candidate_selection"),
            "portfolio_input_fingerprint": stage_input_fingerprints.get("portfolio"),
            "resolved_candidate_selection_reference": _reference_fingerprint_payload(candidate_selection_reference),
            "resolved_portfolio_reference": _reference_fingerprint_payload(portfolio_reference),
            "same_campaign_candidate_selection_run_id": _string_or_none(candidate_selection_result, "run_id"),
            "same_campaign_portfolio_run_id": _string_or_none(portfolio_result, "run_id"),
        }
    elif stage_name == "review":
        payload = {
            "review": _review_fingerprint_config(config),
            "artifact_roots": {
                "alpha_artifacts_root": config.outputs.alpha_artifacts_root,
                "strategy_artifacts_root": str(STRATEGY_ARTIFACTS_ROOT),
                "portfolio_artifacts_root": config.outputs.portfolio_artifacts_root,
            },
            "comparison_input_fingerprint": stage_input_fingerprints.get("comparison"),
            "candidate_review_input_fingerprint": stage_input_fingerprints.get("candidate_review"),
        }
    else:
        payload = {"stage_name": stage_name}
    return canonicalize_value(payload)


def _candidate_selection_fingerprint_config(config: ResearchCampaignConfig) -> dict[str, Any]:
    payload = config.candidate_selection.to_dict()
    payload.pop("output", None)
    execution = dict(payload.get("execution", {}))
    for field_name in ("enable_review", "from_registry", "no_markdown_review", "register_run"):
        execution.pop(field_name, None)
    payload["execution"] = execution
    return canonicalize_value(payload)


def _review_fingerprint_config(config: ResearchCampaignConfig) -> dict[str, Any]:
    payload = config.review.to_dict()
    payload.pop("output", None)
    return canonicalize_value(payload)


def _reference_fingerprint_payload(reference: CampaignArtifactReference | None) -> dict[str, Any] | None:
    if reference is None:
        return None
    metadata = reference.metadata if isinstance(reference.metadata, Mapping) else {}
    return canonicalize_value(
        {
            "stage_name": reference.stage_name,
            "source": reference.source,
            "run_id": reference.run_id,
            "match_criteria": reference.match_criteria,
            "input_fingerprint": metadata.get("input_fingerprint"),
            "upstream_alpha_run_ids": metadata.get("upstream_alpha_run_ids"),
            "candidate_selection_run_id": metadata.get("candidate_selection_run_id"),
        }
    )


def _collect_required_datasets(
    config: ResearchCampaignConfig,
    *,
    alpha_catalog: dict[str, dict[str, Any]],
    strategy_catalog: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    dataset_consumers: dict[str, list[str]] = {}

    def register(dataset_name: str | None, consumer: str) -> None:
        if dataset_name is None:
            return
        normalized = str(dataset_name).strip()
        if not normalized:
            return
        dataset_consumers.setdefault(normalized, [])
        if consumer not in dataset_consumers[normalized]:
            dataset_consumers[normalized].append(consumer)

    for alpha_name in config.targets.alpha_names:
        alpha_config = alpha_catalog.get(alpha_name, {})
        register(
            config.dataset_selection.dataset or alpha_config.get("dataset"),
            f"alpha:{alpha_name}",
        )

    for strategy_name in config.targets.strategy_names:
        strategy_config = strategy_catalog.get(strategy_name, {})
        register(
            strategy_config.get("dataset"),
            f"strategy:{strategy_name}",
        )

    if config.dataset_selection.dataset and not config.targets.alpha_names:
        register(config.dataset_selection.dataset, "campaign.dataset_selection")

    return {
        dataset: consumers
        for dataset, consumers in dataset_consumers.items()
        if dataset in SUPPORTED_FEATURE_DATASETS or consumers
    }


def _campaign_artifact_roots(config: ResearchCampaignConfig) -> dict[str, Path]:
    roots: dict[str, Path] = {
        "campaign_artifacts_root": Path(config.outputs.campaign_artifacts_root),
        "alpha_artifacts_root": Path(config.outputs.alpha_artifacts_root),
        "candidate_selection_artifacts_root": Path(config.candidate_selection.artifacts_root),
        "candidate_selection_output_path": Path(config.candidate_selection.output.path),
        "portfolio_artifacts_root": Path(config.outputs.portfolio_artifacts_root),
    }
    if config.outputs.comparison_output_path is not None:
        roots["comparison_output_path"] = Path(config.outputs.comparison_output_path)
    if config.review.output.path is not None:
        roots["review_output_path"] = Path(config.review.output.path)
    if config.candidate_selection.output.review_output_path is not None:
        roots["candidate_review_output_path"] = Path(config.candidate_selection.output.review_output_path)
    return roots


def _ensure_directory(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"Failed to create or access directory {path.as_posix()}: {exc}"
    if not path.exists() or not path.is_dir():
        return False, f"Path is not a directory: {path.as_posix()}"
    return True, f"Directory is available: {path.as_posix()}"


def _validate_registry_path(path: Path) -> tuple[bool, str, int]:
    try:
        entries = load_registry(path)
    except Exception as exc:
        return False, f"Failed to load registry {path.as_posix()}: {exc}", 0
    if not path.exists():
        return False, f"Registry path does not exist: {path.as_posix()}", 0
    if not entries:
        return False, f"Registry path has no entries: {path.as_posix()}", 0
    return True, f"Registry is available: {path.as_posix()}", len(entries)


def _normalize_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _normalize_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_jsonable(item) for item in value]
    return value


def _replace_stage_record_details(
    records: list[CampaignStageRecord],
    *,
    stage_name: str,
    details_update: Mapping[str, Any],
) -> list[CampaignStageRecord]:
    updated = list(records)
    for index, record in enumerate(updated):
        if record.stage_name != stage_name:
            continue
        updated[index] = CampaignStageRecord(
            stage_name=record.stage_name,
            status=record.status,
            details=canonicalize_value({**record.details, **dict(details_update)}),
        )
        break
    return updated


def _write_campaign_artifacts(
    *,
    config: ResearchCampaignConfig,
    campaign_run_id: str,
    campaign_artifact_dir: Path,
    preflight_result: CampaignPreflightResult,
    stage_records: Sequence[CampaignStageRecord],
    alpha_results: Sequence[Any],
    strategy_results: Sequence[Any],
    alpha_comparison_result: Any | None,
    strategy_comparison_result: Any | None,
    candidate_selection_result: Any | None,
    candidate_selection_reference: CampaignArtifactReference | None,
    portfolio_result: Any | None,
    portfolio_reference: CampaignArtifactReference | None,
    candidate_review_result: Any | None,
    review_result: Any | None,
    status: str,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    checkpoint_payload = _build_campaign_checkpoint(
        config=config,
        campaign_run_id=campaign_run_id,
        campaign_artifact_dir=campaign_artifact_dir,
        preflight_result=preflight_result,
        stage_records=stage_records,
        alpha_results=alpha_results,
        strategy_results=strategy_results,
        alpha_comparison_result=alpha_comparison_result,
        strategy_comparison_result=strategy_comparison_result,
        candidate_selection_result=candidate_selection_result,
        candidate_selection_reference=candidate_selection_reference,
        portfolio_result=portfolio_result,
        portfolio_reference=portfolio_reference,
        candidate_review_result=candidate_review_result,
        review_result=review_result,
        status=status,
    )
    summary_payload = _build_campaign_summary(
        config=config,
        campaign_run_id=campaign_run_id,
        campaign_artifact_dir=campaign_artifact_dir,
        checkpoint_payload=checkpoint_payload,
        preflight_result=preflight_result,
        stage_records=stage_records,
        alpha_results=alpha_results,
        strategy_results=strategy_results,
        alpha_comparison_result=alpha_comparison_result,
        strategy_comparison_result=strategy_comparison_result,
        candidate_selection_result=candidate_selection_result,
        portfolio_result=portfolio_result,
        candidate_review_result=candidate_review_result,
        review_result=review_result,
        status=status,
    )
    manifest_payload = _build_campaign_manifest(
        config=config,
        campaign_run_id=campaign_run_id,
        campaign_artifact_dir=campaign_artifact_dir,
        checkpoint_payload=checkpoint_payload,
        summary_payload=summary_payload,
    )
    checkpoint_path = campaign_artifact_dir / CAMPAIGN_CHECKPOINT_FILENAME
    summary_path = campaign_artifact_dir / CAMPAIGN_SUMMARY_FILENAME
    manifest_path = campaign_artifact_dir / CAMPAIGN_MANIFEST_FILENAME
    write_campaign_checkpoint(checkpoint_path, checkpoint_payload)
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    return checkpoint_path, manifest_path, summary_path, checkpoint_payload, manifest_payload, summary_payload


def _build_campaign_checkpoint(
    *,
    config: ResearchCampaignConfig,
    campaign_run_id: str,
    campaign_artifact_dir: Path,
    preflight_result: CampaignPreflightResult,
    stage_records: Sequence[CampaignStageRecord],
    alpha_results: Sequence[Any],
    strategy_results: Sequence[Any],
    alpha_comparison_result: Any | None,
    strategy_comparison_result: Any | None,
    candidate_selection_result: Any | None,
    candidate_selection_reference: CampaignArtifactReference | None,
    portfolio_result: Any | None,
    portfolio_reference: CampaignArtifactReference | None,
    candidate_review_result: Any | None,
    review_result: Any | None,
    status: str,
) -> dict[str, Any]:
    record_by_stage = {record.stage_name: record for record in stage_records}
    stage_input_fingerprints: dict[str, str | None] = {}
    stages: list[dict[str, Any]] = []
    for stage_name in CAMPAIGN_STAGE_ORDER:
        stage_payload = _campaign_stage_checkpoint(
            stage_name=stage_name,
            record=record_by_stage.get(stage_name),
            config=config,
            preflight_result=preflight_result,
            alpha_results=alpha_results,
            strategy_results=strategy_results,
            alpha_comparison_result=alpha_comparison_result,
            strategy_comparison_result=strategy_comparison_result,
            candidate_selection_result=candidate_selection_result,
            candidate_selection_reference=candidate_selection_reference,
            portfolio_result=portfolio_result,
            portfolio_reference=portfolio_reference,
            candidate_review_result=candidate_review_result,
            review_result=review_result,
            stage_input_fingerprints=stage_input_fingerprints,
        )
        stages.append(stage_payload)
        stage_input_fingerprints[stage_name] = stage_payload.get("input_fingerprint")
    return build_campaign_checkpoint_payload(
        campaign_run_id=campaign_run_id,
        status=status,
        checkpoint_path=campaign_artifact_dir / CAMPAIGN_CHECKPOINT_FILENAME,
        stages=stages,
    )


def _build_campaign_summary(
    *,
    config: ResearchCampaignConfig,
    campaign_run_id: str,
    campaign_artifact_dir: Path,
    checkpoint_payload: dict[str, Any],
    preflight_result: CampaignPreflightResult,
    stage_records: Sequence[CampaignStageRecord],
    alpha_results: Sequence[Any],
    strategy_results: Sequence[Any],
    alpha_comparison_result: Any | None,
    strategy_comparison_result: Any | None,
    candidate_selection_result: Any | None,
    portfolio_result: Any | None,
    candidate_review_result: Any | None,
    review_result: Any | None,
    status: str,
) -> dict[str, Any]:
    selected_run_ids = {
        "alpha_run_ids": sorted(str(result.run_id) for result in alpha_results),
        "strategy_run_ids": sorted(str(result.run_id) for result in strategy_results),
        "candidate_selection_run_id": _string_or_none(candidate_selection_result, "run_id"),
        "portfolio_run_id": _string_or_none(portfolio_result, "run_id"),
        "review_id": _string_or_none(review_result, "review_id"),
    }
    output_paths = {
        "campaign_artifact_dir": campaign_artifact_dir.as_posix(),
        "campaign_config": (campaign_artifact_dir / CAMPAIGN_CONFIG_FILENAME).as_posix(),
        "campaign_checkpoint": (campaign_artifact_dir / CAMPAIGN_CHECKPOINT_FILENAME).as_posix(),
        "preflight_summary": preflight_result.summary_path.as_posix(),
        "campaign_manifest": (campaign_artifact_dir / CAMPAIGN_MANIFEST_FILENAME).as_posix(),
        "campaign_summary": (campaign_artifact_dir / CAMPAIGN_SUMMARY_FILENAME).as_posix(),
        "alpha_artifact_dirs": sorted(_path_or_none(result, "artifact_dir") for result in alpha_results),
        "strategy_artifact_dirs": sorted(_path_or_none(result, "experiment_dir") for result in strategy_results),
        "alpha_comparison_csv": _path_or_none(alpha_comparison_result, "csv_path"),
        "alpha_comparison_summary": _path_or_none(alpha_comparison_result, "json_path"),
        "strategy_comparison_csv": _path_or_none(strategy_comparison_result, "csv_path"),
        "strategy_comparison_summary": _path_or_none(strategy_comparison_result, "json_path"),
        "candidate_selection_artifact_dir": _path_or_none(candidate_selection_result, "artifact_dir"),
        "candidate_selection_summary": _path_or_none(candidate_selection_result, "summary_json"),
        "candidate_selection_manifest": _path_or_none(candidate_selection_result, "manifest_json"),
        "portfolio_artifact_dir": _path_or_none(portfolio_result, "experiment_dir"),
        "candidate_review_dir": _path_or_none(candidate_review_result, "review_dir"),
        "candidate_review_summary": _path_or_none(candidate_review_result, "candidate_review_summary_json"),
        "candidate_review_manifest": _path_or_none(candidate_review_result, "manifest_json"),
        "review_leaderboard_csv": _path_or_none(review_result, "csv_path"),
        "review_summary": _path_or_none(review_result, "json_path"),
        "review_manifest": _path_or_none(review_result, "manifest_path"),
        "review_promotion_gates": _path_or_none(review_result, "promotion_gate_path"),
    }
    output_paths["alpha_artifact_dirs"] = [path for path in output_paths["alpha_artifact_dirs"] if path is not None]
    output_paths["strategy_artifact_dirs"] = [path for path in output_paths["strategy_artifact_dirs"] if path is not None]

    stage_summaries = {
        record.stage_name: _campaign_stage_summary(
            record=record,
            alpha_results=alpha_results,
            strategy_results=strategy_results,
            alpha_comparison_result=alpha_comparison_result,
            strategy_comparison_result=strategy_comparison_result,
            candidate_selection_result=candidate_selection_result,
            portfolio_result=portfolio_result,
            candidate_review_result=candidate_review_result,
            review_result=review_result,
        )
        for record in stage_records
    }
    stages = []
    for checkpoint_stage in checkpoint_payload["stages"]:
        stage_name = str(checkpoint_stage["stage_name"])
        summary_stage = stage_summaries.get(stage_name)
        if summary_stage is None:
            summary_stage = _campaign_pending_stage_summary(stage_name=stage_name)
        execution_metadata = _stage_execution_metadata(
            checkpoint_stage,
            stage_details=summary_stage.get("details"),
        )
        stages.append(
            canonicalize_value(
                {
                    **summary_stage,
                    "state": checkpoint_stage["state"],
                    "state_reason": checkpoint_stage.get("state_reason"),
                    "source": checkpoint_stage.get("source"),
                    "terminal": checkpoint_stage["terminal"],
                    "resumable": checkpoint_stage["resumable"],
                    "execution_metadata": execution_metadata,
                }
            )
        )
    stage_state_counts = _stage_state_counts(checkpoint_payload["stage_states"])
    stage_execution = _stage_execution_by_name(checkpoint_payload, stages=stages)
    retry_stage_names = [
        stage["stage_name"]
        for stage in stages
        if isinstance(stage.get("details"), Mapping)
        and isinstance(stage["details"].get("retry"), Mapping)
        and bool(stage["details"]["retry"].get("attempted"))
    ]
    failed_stage_names = [
        stage["stage_name"]
        for stage in stages
        if str(stage.get("state")) == "failed"
    ]
    partial_stage_names = [
        stage["stage_name"]
        for stage in stages
        if str(stage.get("state")) == "partial"
    ]
    reused_stage_names = [
        stage["stage_name"]
        for stage in stages
        if str(stage.get("state")) == "reused"
    ]
    skipped_stage_names = [
        stage["stage_name"]
        for stage in stages
        if str(stage.get("state")) == "skipped"
    ]
    resumable_stage_names = [
        stage["stage_name"]
        for stage in stages
        if bool(stage.get("resumable"))
    ]
    failure_summaries = [
        canonicalize_value(
            {
                "stage_name": stage["stage_name"],
                "state": stage["state"],
                "state_reason": stage.get("state_reason"),
                "failure": stage["details"].get("failure"),
            }
        )
        for stage in stages
        if isinstance(stage.get("details"), Mapping)
        and isinstance(stage["details"].get("failure"), Mapping)
    ]

    final_outcomes = {
        "preflight_status": preflight_result.status,
        "review_promotion_status": _mapping_value(_campaign_review_promotion_summary(review_result), "promotion_status"),
        "review_promotion_gate_status": _mapping_value(_campaign_review_promotion_summary(review_result), "evaluation_status"),
        "review_promotion_gate_summary": _campaign_review_promotion_summary(review_result),
        "review_counts_by_run_type": _counts_by_run_type(list(getattr(review_result, "entries", []))),
        "stage_state_counts": stage_state_counts,
        "retry_stage_names": retry_stage_names,
        "failed_stage_names": failed_stage_names,
        "partial_stage_names": partial_stage_names,
        "reused_stage_names": reused_stage_names,
        "skipped_stage_names": skipped_stage_names,
        "resumable_stage_names": resumable_stage_names,
        "failures": failure_summaries,
        "candidate_review_counts": (
            None
            if candidate_review_result is None
            else {
                "total_candidates": _coerce_int(getattr(candidate_review_result, "total_candidates", None)),
                "selected_candidates": _coerce_int(getattr(candidate_review_result, "selected_candidates", None)),
                "rejected_candidates": _coerce_int(getattr(candidate_review_result, "rejected_candidates", None)),
            }
        ),
    }

    key_metrics = {
        "alpha_runs": [_alpha_key_metrics(result) for result in sorted(alpha_results, key=lambda item: str(item.run_id))],
        "strategy_runs": [_strategy_key_metrics(result) for result in sorted(strategy_results, key=lambda item: str(item.run_id))],
        "candidate_selection": _candidate_selection_key_metrics(candidate_selection_result),
        "portfolio": _portfolio_key_metrics(portfolio_result),
        "review": _review_key_metrics(review_result),
    }

    payload = {
        "run_type": "research_campaign",
        "campaign_run_id": campaign_run_id,
        "status": status,
        "preflight_status": preflight_result.status,
        "checkpoint_schema_version": checkpoint_payload["schema_version"],
        "checkpoint_path": checkpoint_payload["checkpoint_path"],
        "targets": canonicalize_value(
            {
                "alpha_names": list(config.targets.alpha_names),
                "strategy_names": list(config.targets.strategy_names),
                "portfolio_names": list(config.targets.portfolio_names),
                "candidate_selection_alpha_name": config.candidate_selection.alpha_name,
                "portfolio_name": config.portfolio.portfolio_name,
            }
        ),
        "stage_statuses": dict(checkpoint_payload["stage_states"]),
        "stage_state_counts": stage_state_counts,
        "stage_execution": stage_execution,
        "checkpoint": checkpoint_payload,
        "stages": stages,
        "selected_run_ids": selected_run_ids,
        "key_metrics": key_metrics,
        "output_paths": output_paths,
        "final_outcomes": final_outcomes,
    }
    return canonicalize_value(payload)


def _build_campaign_manifest(
    *,
    config: ResearchCampaignConfig,
    campaign_run_id: str,
    campaign_artifact_dir: Path,
    checkpoint_payload: dict[str, Any],
    summary_payload: dict[str, Any],
) -> dict[str, Any]:
    stage_execution = _stage_execution_by_name(
        checkpoint_payload,
        stages=summary_payload.get("stages", []),
    )
    artifact_files = sorted(
        [
            CAMPAIGN_CONFIG_FILENAME,
            CAMPAIGN_CHECKPOINT_FILENAME,
            PREFLIGHT_SUMMARY_FILENAME,
            CAMPAIGN_MANIFEST_FILENAME,
            CAMPAIGN_SUMMARY_FILENAME,
        ]
    )
    artifact_groups = {
        "campaign": artifact_files,
        "core": artifact_files,
        "checkpoint": [CAMPAIGN_CHECKPOINT_FILENAME],
        "preflight": [PREFLIGHT_SUMMARY_FILENAME],
        "summary": [CAMPAIGN_SUMMARY_FILENAME],
    }
    payload = {
        "run_type": "research_campaign",
        "campaign_run_id": campaign_run_id,
        "status": summary_payload["status"],
        "artifact_files": artifact_files,
        "artifact_groups": artifact_groups,
        "artifacts": {
            CAMPAIGN_CONFIG_FILENAME: {"path": CAMPAIGN_CONFIG_FILENAME},
            CAMPAIGN_CHECKPOINT_FILENAME: {
                "path": CAMPAIGN_CHECKPOINT_FILENAME,
                "schema_version": checkpoint_payload["schema_version"],
                "stage_count": len(checkpoint_payload["stages"]),
            },
            PREFLIGHT_SUMMARY_FILENAME: {"path": PREFLIGHT_SUMMARY_FILENAME},
            CAMPAIGN_MANIFEST_FILENAME: {"path": CAMPAIGN_MANIFEST_FILENAME},
            CAMPAIGN_SUMMARY_FILENAME: {
                "path": CAMPAIGN_SUMMARY_FILENAME,
                "stage_count": len(summary_payload["stages"]),
            },
        },
        "campaign_artifact_dir": campaign_artifact_dir.as_posix(),
        "checkpoint_path": CAMPAIGN_CHECKPOINT_FILENAME,
        "stage_statuses": dict(checkpoint_payload["stage_states"]),
        "stage_state_counts": _stage_state_counts(checkpoint_payload["stage_states"]),
        "stage_execution": stage_execution,
        "retry_stage_names": list(summary_payload.get("final_outcomes", {}).get("retry_stage_names", [])),
        "failed_stage_names": list(summary_payload.get("final_outcomes", {}).get("failed_stage_names", [])),
        "partial_stage_names": list(summary_payload.get("final_outcomes", {}).get("partial_stage_names", [])),
        "reused_stage_names": list(summary_payload.get("final_outcomes", {}).get("reused_stage_names", [])),
        "skipped_stage_names": list(summary_payload.get("final_outcomes", {}).get("skipped_stage_names", [])),
        "resumable_stage_names": list(summary_payload.get("final_outcomes", {}).get("resumable_stage_names", [])),
        "selected_run_ids": dict(summary_payload["selected_run_ids"]),
        "targets": canonicalize_value(
            {
                "alpha_names": list(config.targets.alpha_names),
                "strategy_names": list(config.targets.strategy_names),
                "portfolio_names": list(config.targets.portfolio_names),
            }
        ),
        "summary_path": CAMPAIGN_SUMMARY_FILENAME,
    }
    return canonicalize_value(payload)


def _campaign_stage_checkpoint(
    *,
    stage_name: str,
    record: CampaignStageRecord | None,
    config: ResearchCampaignConfig,
    preflight_result: CampaignPreflightResult,
    alpha_results: Sequence[Any],
    strategy_results: Sequence[Any],
    alpha_comparison_result: Any | None,
    strategy_comparison_result: Any | None,
    candidate_selection_result: Any | None,
    candidate_selection_reference: CampaignArtifactReference | None,
    portfolio_result: Any | None,
    portfolio_reference: CampaignArtifactReference | None,
    candidate_review_result: Any | None,
    review_result: Any | None,
    stage_input_fingerprints: Mapping[str, str | None],
) -> dict[str, Any]:
    fingerprint_inputs = _campaign_stage_fingerprint_inputs(
        stage_name=stage_name,
        config=config,
        alpha_results=alpha_results,
        strategy_results=strategy_results,
        candidate_selection_result=candidate_selection_result,
        candidate_selection_reference=candidate_selection_reference,
        portfolio_result=portfolio_result,
        portfolio_reference=portfolio_reference,
        stage_input_fingerprints=stage_input_fingerprints,
    )
    input_fingerprint = _fingerprint_mapping(fingerprint_inputs)
    if record is None:
        return build_campaign_stage_checkpoint(
            stage_name=stage_name,
            state="pending",
            state_reason=(
                "Waiting for an upstream stage to complete."
                if preflight_result.status == "passed"
                else "Blocked because campaign preflight failed."
            ),
            source="planner",
            input_fingerprint=input_fingerprint,
            fingerprint_inputs=fingerprint_inputs,
        )

    state = record.status
    source = "executed"
    state_reason = None
    failure_details = record.details.get("failure") if isinstance(record.details.get("failure"), Mapping) else None
    if record.status == "reused":
        source = "checkpoint"
        state_reason = "Reused persisted stage outputs from a matching checkpoint."
    elif record.status == "skipped":
        source = "config"
        state_reason = "Stage disabled or not applicable for this campaign."
    elif record.status == "failed":
        failure_message = None if failure_details is None else str(failure_details.get("message") or "").strip()
        state_reason = (
            f"Stage execution failed: {failure_message}"
            if failure_message
            else "Stage execution failed."
        )
    elif record.status == "partial":
        failure_message = None if failure_details is None else str(failure_details.get("message") or "").strip()
        state_reason = (
            f"Stage execution interrupted: {failure_message}"
            if failure_message
            else "Stage execution interrupted before completion."
        )

    if stage_name == "candidate_selection":
        input_reference = record.details.get("input_reference")
        if candidate_selection_result is None and isinstance(input_reference, Mapping):
            state = "reused"
            source = str(input_reference.get("source") or "registry")
            state_reason = "Resolved existing candidate-selection artifacts for downstream chaining."
    elif stage_name == "portfolio":
        input_reference = record.details.get("input_reference")
        if portfolio_result is None and isinstance(input_reference, Mapping):
            state = "reused"
            source = str(input_reference.get("source") or "registry")
            state_reason = "Resolved existing portfolio artifacts for downstream chaining."

    summary_payload = _campaign_stage_summary(
        record=record,
        alpha_results=alpha_results,
        strategy_results=strategy_results,
        alpha_comparison_result=alpha_comparison_result,
        strategy_comparison_result=strategy_comparison_result,
        candidate_selection_result=candidate_selection_result,
        portfolio_result=portfolio_result,
        candidate_review_result=candidate_review_result,
        review_result=review_result,
    )
    return build_campaign_stage_checkpoint(
        stage_name=stage_name,
        state=state,
        state_reason=state_reason,
        source=source,
        input_fingerprint=input_fingerprint,
        fingerprint_inputs=fingerprint_inputs,
        selected_run_ids=summary_payload["selected_run_ids"],
        key_metrics=summary_payload["key_metrics"],
        output_paths=summary_payload["output_paths"],
        outcomes=summary_payload["outcomes"],
        details=summary_payload["details"],
    )


def _campaign_pending_stage_summary(*, stage_name: str) -> dict[str, Any]:
    return canonicalize_value(
        {
            "stage_name": stage_name,
            "status": "pending",
            "selected_run_ids": {},
            "key_metrics": {},
            "output_paths": {},
            "outcomes": {},
            "details": {},
        }
    )


def _campaign_stage_summary(
    *,
    record: CampaignStageRecord,
    alpha_results: Sequence[Any],
    strategy_results: Sequence[Any],
    alpha_comparison_result: Any | None,
    strategy_comparison_result: Any | None,
    candidate_selection_result: Any | None,
    portfolio_result: Any | None,
    candidate_review_result: Any | None,
    review_result: Any | None,
) -> dict[str, Any]:
    selected_run_ids: dict[str, Any] = {}
    key_metrics: dict[str, Any] = {}
    output_paths: dict[str, Any] = {}
    outcomes: dict[str, Any] = {}

    if record.stage_name == "preflight":
        outcomes["failed_checks"] = list(record.details.get("failed_checks", []))
        output_paths["summary"] = PREFLIGHT_SUMMARY_FILENAME
    elif record.stage_name == "research":
        selected_run_ids = {
            "alpha_run_ids": sorted(str(result.run_id) for result in alpha_results),
            "strategy_run_ids": sorted(str(result.run_id) for result in strategy_results),
        }
        key_metrics = {
            "alpha_runs": [_alpha_key_metrics(result) for result in sorted(alpha_results, key=lambda item: str(item.run_id))],
            "strategy_runs": [_strategy_key_metrics(result) for result in sorted(strategy_results, key=lambda item: str(item.run_id))],
        }
        output_paths = {
            "alpha_artifact_dirs": [path for path in sorted(_path_or_none(result, "artifact_dir") for result in alpha_results) if path],
            "strategy_artifact_dirs": [path for path in sorted(_path_or_none(result, "experiment_dir") for result in strategy_results) if path],
        }
    elif record.stage_name == "comparison":
        selected_run_ids = {
            "alpha_comparison_id": _string_or_none(alpha_comparison_result, "comparison_id"),
            "strategy_comparison_id": _string_or_none(strategy_comparison_result, "comparison_id"),
        }
        output_paths = {
            "alpha_comparison_csv": _path_or_none(alpha_comparison_result, "csv_path"),
            "alpha_comparison_summary": _path_or_none(alpha_comparison_result, "json_path"),
            "strategy_comparison_csv": _path_or_none(strategy_comparison_result, "csv_path"),
            "strategy_comparison_summary": _path_or_none(strategy_comparison_result, "json_path"),
        }
    elif record.stage_name == "candidate_selection":
        selected_run_ids = {"candidate_selection_run_id": _string_or_none(candidate_selection_result, "run_id")}
        key_metrics = _candidate_selection_key_metrics(candidate_selection_result) or {}
        output_paths = {
            "artifact_dir": _path_or_none(candidate_selection_result, "artifact_dir"),
            "summary_json": _path_or_none(candidate_selection_result, "summary_json"),
            "manifest_json": _path_or_none(candidate_selection_result, "manifest_json"),
        }
    elif record.stage_name == "portfolio":
        selected_run_ids = {"portfolio_run_id": _string_or_none(portfolio_result, "run_id")}
        key_metrics = _portfolio_key_metrics(portfolio_result) or {}
        output_paths = {"artifact_dir": _path_or_none(portfolio_result, "experiment_dir")}
    elif record.stage_name == "candidate_review":
        selected_run_ids = {
            "candidate_selection_run_id": _string_or_none(candidate_review_result, "candidate_selection_run_id"),
            "portfolio_run_id": _string_or_none(candidate_review_result, "portfolio_run_id"),
        }
        outcomes = (
            {}
            if candidate_review_result is None
            else {
                "total_candidates": _coerce_int(getattr(candidate_review_result, "total_candidates", None)),
                "selected_candidates": _coerce_int(getattr(candidate_review_result, "selected_candidates", None)),
                "rejected_candidates": _coerce_int(getattr(candidate_review_result, "rejected_candidates", None)),
            }
        )
        output_paths = {
            "review_dir": _path_or_none(candidate_review_result, "review_dir"),
            "summary_json": _path_or_none(candidate_review_result, "candidate_review_summary_json"),
            "manifest_json": _path_or_none(candidate_review_result, "manifest_json"),
        }
    elif record.stage_name == "review":
        selected_run_ids = {"review_id": _string_or_none(review_result, "review_id")}
        key_metrics = _review_key_metrics(review_result) or {}
        outcomes = _campaign_review_promotion_summary(review_result) or {}
        output_paths = {
            "leaderboard_csv": _path_or_none(review_result, "csv_path"),
            "summary_json": _path_or_none(review_result, "json_path"),
            "manifest_json": _path_or_none(review_result, "manifest_path"),
            "promotion_gates_json": _path_or_none(review_result, "promotion_gate_path"),
        }

    payload = {
        "stage_name": record.stage_name,
        "status": record.status,
        "selected_run_ids": selected_run_ids,
        "key_metrics": key_metrics,
        "output_paths": output_paths,
        "outcomes": outcomes,
        "details": _normalize_jsonable(record.details),
    }
    return canonicalize_value(payload)


def _alpha_key_metrics(result: Any) -> dict[str, Any]:
    summary = getattr(getattr(result, "evaluation", None), "evaluation_result", None)
    summary_payload = getattr(summary, "summary", {}) if summary is not None else {}
    sleeve_summary = _mapping_value(getattr(result, "evaluation", None), "manifest")
    sleeve_metrics = sleeve_summary.get("sleeve", {}).get("metric_summary") if isinstance(sleeve_summary, dict) else {}
    return canonicalize_value(
        {
            "alpha_name": _string_or_none(result, "alpha_name"),
            "run_id": _string_or_none(result, "run_id"),
            "mean_ic": _coerce_float(summary_payload.get("mean_ic")),
            "ic_ir": _coerce_float(summary_payload.get("ic_ir")),
            "n_periods": _coerce_int(summary_payload.get("n_periods")),
            "sleeve_sharpe_ratio": _coerce_float((sleeve_metrics or {}).get("sharpe_ratio")),
            "sleeve_total_return": _coerce_float((sleeve_metrics or {}).get("total_return")),
        }
    )


def _strategy_key_metrics(result: Any) -> dict[str, Any]:
    metrics = getattr(result, "metrics", {})
    return canonicalize_value(
        {
            "strategy_name": _string_or_none(result, "strategy_name"),
            "run_id": _string_or_none(result, "run_id"),
            "cumulative_return": _coerce_float(metrics.get("cumulative_return")),
            "sharpe_ratio": _coerce_float(metrics.get("sharpe_ratio")),
            "max_drawdown": _coerce_float(metrics.get("max_drawdown")),
        }
    )


def _candidate_selection_key_metrics(result: Any | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return canonicalize_value(
        {
            "run_id": _string_or_none(result, "run_id"),
            "primary_metric": _string_or_none(result, "primary_metric"),
            "universe_count": _coerce_int(getattr(result, "universe_count", None)),
            "eligible_count": _coerce_int(getattr(result, "eligible_count", None)),
            "selected_count": _coerce_int(getattr(result, "selected_count", None)),
            "rejected_count": _coerce_int(getattr(result, "rejected_count", None)),
            "pruned_by_redundancy": _coerce_int(getattr(result, "pruned_by_redundancy", None)),
        }
    )


def _portfolio_key_metrics(result: Any | None) -> dict[str, Any] | None:
    if result is None:
        return None
    metrics = getattr(result, "metrics", {})
    return canonicalize_value(
        {
            "run_id": _string_or_none(result, "run_id"),
            "portfolio_name": _string_or_none(result, "portfolio_name"),
            "component_count": _coerce_int(getattr(result, "component_count", None)),
            "total_return": _coerce_float(metrics.get("total_return")),
            "sharpe_ratio": _coerce_float(metrics.get("sharpe_ratio")),
            "max_drawdown": _coerce_float(metrics.get("max_drawdown")),
        }
    )


def _review_key_metrics(result: Any | None) -> dict[str, Any] | None:
    if result is None:
        return None
    entries = list(getattr(result, "entries", []))
    return canonicalize_value(
        {
            "review_id": _string_or_none(result, "review_id"),
            "entry_count": len(entries),
            "counts_by_run_type": _counts_by_run_type(entries),
        }
    )


def _campaign_review_promotion_summary(result: Any | None) -> dict[str, Any] | None:
    if result is None:
        return None
    promotion_gate_path = getattr(result, "promotion_gate_path", None)
    if not isinstance(promotion_gate_path, Path) or not promotion_gate_path.exists():
        return None
    try:
        payload = json.loads(promotion_gate_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return canonicalize_value(
        {
            "evaluation_status": payload.get("evaluation_status"),
            "promotion_status": payload.get("promotion_status"),
            "gate_count": payload.get("gate_count"),
            "passed_gate_count": payload.get("passed_gate_count"),
            "failed_gate_count": payload.get("failed_gate_count"),
            "missing_gate_count": payload.get("missing_gate_count"),
        }
    )


def _counts_by_run_type(entries: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        run_type = getattr(entry, "run_type", None)
        if not isinstance(run_type, str):
            continue
        counts[run_type] = counts.get(run_type, 0) + 1
    return dict(sorted(counts.items()))


def _string_or_none(obj: Any | None, attribute: str) -> str | None:
    if obj is None:
        return None
    value = getattr(obj, attribute, None)
    if value is None:
        return None
    return str(value)


def _path_or_none(obj: Any | None, attribute: str) -> str | None:
    if obj is None:
        return None
    value = getattr(obj, attribute, None)
    if value is None:
        return None
    return Path(value).as_posix() if isinstance(value, Path) else str(value)


def _mapping_value(obj: Any | None, attribute: str) -> dict[str, Any] | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        value = obj.get(attribute)
    else:
        value = getattr(obj, attribute, None)
    return value if isinstance(value, dict) else value


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    return int(numeric)


@contextmanager
def _strategy_config_override(path: Path) -> Iterator[None]:
    default_path = Path(run_strategy_cli.STRATEGIES_CONFIG)
    if path == default_path:
        yield
        return

    original_const = run_strategy_cli.STRATEGIES_CONFIG
    original_loader = run_strategy_cli.load_strategies_config

    def _load_override(_: Path = path) -> dict[str, dict[str, Any]]:
        return original_loader(path)

    run_strategy_cli.STRATEGIES_CONFIG = path
    run_strategy_cli.load_strategies_config = _load_override
    try:
        yield
    finally:
        run_strategy_cli.STRATEGIES_CONFIG = original_const
        run_strategy_cli.load_strategies_config = original_loader


if __name__ == "__main__":
    main()
