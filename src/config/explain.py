from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.config.profiles import RuntimeProfileError
from src.config.resolution import (
    ConfigResolutionError,
    ConfigResolutionResult,
    resolve_runtime_profile_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_EXPLAIN_WORKFLOWS = frozenset(
    {"generic", "strategy", "alpha", "portfolio", "pipeline", "campaign", "evidence_review"}
)
_SOURCE_RANK = {"default": 0, "profile": 1, "environment": 2, "cli_override": 3}


@dataclass(frozen=True)
class WorkflowAssumption:
    name: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "message": self.message,
        }


@dataclass(frozen=True)
class RuntimeExplainReport:
    status: str
    workflow: str
    profile: dict[str, Any]
    resolved_config: dict[str, Any] | None
    provenance: dict[str, Any] | None
    provenance_summary: dict[str, Any] | None
    path_summary: dict[str, Any] | None
    workflow_assumptions: tuple[WorkflowAssumption, ...]
    artifact_boundaries: dict[str, Any] | None
    findings: tuple[dict[str, Any], ...] = ()
    output_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _canonicalize(
            {
                "status": self.status,
                "schema_version": 1,
                "run_type": "runtime_explain",
                "authoritative": False,
                "workflow": self.workflow,
                "profile": self.profile,
                "resolved_config": self.resolved_config,
                "provenance": self.provenance,
                "provenance_summary": self.provenance_summary,
                "path_summary": self.path_summary,
                "workflow_assumptions": [
                    assumption.to_dict() for assumption in self.workflow_assumptions
                ],
                "artifact_boundaries": self.artifact_boundaries,
                "safety": {
                    "workflows_executed": False,
                    "canonical_artifacts_mutated": False,
                    "authoritative": False,
                    "direct_scan": None
                    if self.artifact_boundaries is None
                    else self.artifact_boundaries.get("direct_scan"),
                    "derived_outputs_authoritative": None
                    if self.artifact_boundaries is None
                    else self.artifact_boundaries.get("derived_outputs_authoritative"),
                    "mutates_canonical_artifacts": None
                    if self.artifact_boundaries is None
                    else self.artifact_boundaries.get("mutates_canonical_artifacts"),
                    "requires_network": None
                    if self.resolved_config is None
                    else self.resolved_config["boundaries"].get("requires_network"),
                    "requires_credentials": None
                    if self.resolved_config is None
                    else self.resolved_config["boundaries"].get("requires_credentials"),
                    "requires_live_market_data": None
                    if self.resolved_config is None
                    else self.resolved_config["boundaries"].get("requires_live_market_data"),
                },
                "findings": list(self.findings),
                "output_path": self.output_path,
            }
        )

    def to_json_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), indent=2, sort_keys=True)


def build_runtime_explain_report(
    profile: str | None = None,
    *,
    profile_path: str | Path | None = None,
    workflow: str = "generic",
    output_path: str | Path | None = None,
) -> RuntimeExplainReport:
    """
    Explain resolved runtime context without executing workflows or mutating artifacts.
    """

    normalized_workflow = _normalize_workflow(workflow)
    profile_payload = _initial_profile_payload(profile, profile_path)
    try:
        result = resolve_runtime_profile_config(profile, profile_path=profile_path)
    except (RuntimeProfileError, ConfigResolutionError, FileNotFoundError, OSError, ValueError) as exc:
        return RuntimeExplainReport(
            status="failed",
            workflow=normalized_workflow,
            profile=profile_payload,
            resolved_config=None,
            provenance=None,
            provenance_summary=None,
            path_summary=None,
            workflow_assumptions=_workflow_assumptions(normalized_workflow),
            artifact_boundaries=None,
            findings=(
                {
                    "severity": "error",
                    "message": _safe_message(str(exc), profile_path),
                },
            ),
            output_path=None if output_path is None else _display_path(output_path),
        )

    resolved = result.to_json_dict()
    profile_payload = {
        "name": result.profile_name,
        "path": result.profile_path,
        "source": _profile_source(profile, profile_path, result),
    }
    return RuntimeExplainReport(
        status="passed",
        workflow=normalized_workflow,
        profile=profile_payload,
        resolved_config=resolved["config"],
        provenance=resolved["provenance"],
        provenance_summary=_provenance_summary(resolved["provenance"]),
        path_summary=_path_summary(resolved["config"], output_path),
        workflow_assumptions=_workflow_assumptions(normalized_workflow),
        artifact_boundaries=resolved["artifact_boundaries"],
        findings=(),
        output_path=None if output_path is None else _display_path(output_path),
    )


def write_runtime_explain_report(report: RuntimeExplainReport, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_json() + "\n", encoding="utf-8")
    return output


def _normalize_workflow(workflow: str) -> str:
    normalized = workflow.strip().lower()
    if normalized not in SUPPORTED_EXPLAIN_WORKFLOWS:
        supported = ", ".join(sorted(SUPPORTED_EXPLAIN_WORKFLOWS))
        raise ValueError(f"Unsupported explain workflow {workflow!r}. Supported values: {supported}.")
    return normalized


def _provenance_summary(provenance: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    source_counts = {source: 0 for source in _SOURCE_RANK}
    for entry in provenance.values():
        source = str(entry.get("source"))
        source_counts[source] = source_counts.get(source, 0) + 1
    used_sources = [source for source, count in source_counts.items() if count > 0]
    highest = max(used_sources, key=lambda source: _SOURCE_RANK.get(source, -1)) if used_sources else None
    return {
        "source_counts": {source: source_counts[source] for source in sorted(source_counts)},
        "highest_precedence_source": highest,
        "field_count": sum(source_counts.values()),
    }


def _path_summary(config: Mapping[str, Any], output_path: str | Path | None) -> dict[str, Any]:
    settings = dict(config["settings"])
    workflow_configs = dict(config["workflow_configs"])
    artifacts_root = settings.get("artifacts_root")
    return {
        "artifacts_root": artifacts_root,
        "features_root": settings.get("features_root"),
        "marketlake_root": settings.get("marketlake_root"),
        "workflow_configs": workflow_configs,
        "expected_artifact_roots": _expected_artifact_roots(artifacts_root),
        "requested_output_path": None if output_path is None else _display_path(output_path),
        "derived_output_recommendation": "artifacts/_derived/config_explain/",
    }


def _expected_artifact_roots(artifacts_root: Any) -> dict[str, Any]:
    if artifacts_root is None:
        return {
            "strategy": None,
            "alpha": None,
            "portfolio": None,
            "pipeline": None,
            "campaign": None,
            "evidence_review": None,
        }
    root = str(artifacts_root).rstrip("/")
    return {
        "strategy": f"{root}/strategies",
        "alpha": f"{root}/alpha",
        "portfolio": f"{root}/portfolios",
        "pipeline": f"{root}/pipelines",
        "campaign": f"{root}/research_campaigns",
        "evidence_review": "artifacts/_derived/evidence_review",
    }


def _workflow_assumptions(workflow: str) -> tuple[WorkflowAssumption, ...]:
    shared = (
        WorkflowAssumption(
            "configuration_only",
            "This report explains resolved configuration context only.",
        ),
        WorkflowAssumption(
            "no_execution",
            "No workflow code is executed by the explain helper.",
        ),
    )
    specific: dict[str, tuple[WorkflowAssumption, ...]] = {
        "generic": (
            WorkflowAssumption(
                "no_workflow_selected",
                "No workflow is selected; the report describes resolved runtime context.",
            ),
        ),
        "strategy": (
            WorkflowAssumption("strategy_not_run", "Signal generation and backtest are not run."),
            WorkflowAssumption(
                "strategy_artifacts_not_written",
                "Strategy artifact directories are described but not created.",
            ),
        ),
        "alpha": (
            WorkflowAssumption("alpha_not_run", "Model training and inference are not run."),
            WorkflowAssumption("alpha_artifacts_not_written", "Alpha artifacts are not written."),
        ),
        "portfolio": (
            WorkflowAssumption("portfolio_not_run", "Portfolio construction is not run."),
            WorkflowAssumption("portfolio_artifacts_not_written", "Portfolio artifacts are not written."),
        ),
        "pipeline": (
            WorkflowAssumption("pipeline_not_run", "Pipeline stages are not run."),
            WorkflowAssumption(
                "pipeline_configs_reported_only",
                "Pipeline config references are reported only.",
            ),
        ),
        "campaign": (
            WorkflowAssumption("campaign_not_run", "Research campaign stages are not run."),
            WorkflowAssumption(
                "campaign_state_not_created",
                "Campaign scenarios, checkpoints, and manifests are not created.",
            ),
        ),
        "evidence_review": (
            WorkflowAssumption("evidence_review_not_run", "Evidence review packs are not built."),
            WorkflowAssumption(
                "derived_review_outputs_not_created",
                "Derived evidence review outputs are not created.",
            ),
        ),
    }
    return tuple(sorted((*shared, *specific[workflow]), key=lambda item: item.name))


def _profile_source(
    profile: str | None,
    profile_path: str | Path | None,
    result: ConfigResolutionResult,
) -> str:
    if profile_path is not None:
        return "profile_path"
    if profile is not None:
        return "profile_name"
    if result.profile_name is not None:
        return "profile"
    return "default"


def _initial_profile_payload(profile: str | None, profile_path: str | Path | None) -> dict[str, Any]:
    return {
        "name": profile,
        "path": None if profile_path is None else _display_path(profile_path),
        "source": "profile_path" if profile_path is not None else ("profile_name" if profile else "default"),
    }


def _display_path(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"<external>/{candidate.name}"


def _safe_message(message: str, profile_path: str | Path | None) -> str:
    if profile_path is None:
        return message
    candidate = Path(profile_path)
    if not candidate.is_absolute():
        return message
    return message.replace(str(candidate), _display_path(candidate))


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]
    return value
