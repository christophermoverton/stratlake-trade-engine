from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from src.execution.result import ExecutionResult


def run_research_campaign(
    config: Mapping[str, Any] | Any | None = None,
    *,
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ExecutionResult:
    """Run one research campaign or scenario orchestration from Python.

    This wrapper preserves the CLI execution path by resolving campaign config
    through the shared campaign config helpers, then delegating to the existing
    campaign runner.
    """

    raw_result = _run_campaign(_resolve_campaign_config(config, config_path=config_path, overrides=overrides))
    return summarize_campaign_execution(raw_result)


def run_campaign(
    config: Mapping[str, Any] | Any | None = None,
    *,
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ExecutionResult:
    """Alias for :func:`run_research_campaign`."""

    return run_research_campaign(config, config_path=config_path, overrides=overrides)


def run_research_campaign_from_cli_args(args: Any) -> ExecutionResult:
    return run_research_campaign(config_path=getattr(args, "config", None))


def run_research_campaign_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_research_campaign import parse_args

    return run_research_campaign_from_cli_args(parse_args(argv))


def summarize_campaign_execution(raw_result: Any) -> ExecutionResult:
    from src.cli.run_research_campaign import (
        ResearchCampaignOrchestrationResult,
        ResearchCampaignRunResult,
    )

    if isinstance(raw_result, ResearchCampaignOrchestrationResult):
        return _summarize_orchestration_result(raw_result)
    if isinstance(raw_result, ResearchCampaignRunResult):
        return _summarize_single_campaign_result(raw_result)
    raise TypeError(f"Unsupported campaign result type: {raw_result.__class__.__name__}")


def _resolve_campaign_config(
    config: Mapping[str, Any] | Any | None,
    *,
    config_path: str | Path | None,
    overrides: Mapping[str, Any] | None,
) -> Any:
    from src.config.research_campaign import (
        ResearchCampaignConfig,
        load_research_campaign_config,
        resolve_research_campaign_config,
    )

    if isinstance(config, ResearchCampaignConfig):
        if config_path is not None or overrides is not None:
            return resolve_research_campaign_config(
                config.to_dict(),
                None if overrides is None else dict(overrides),
            )
        return config

    sources: list[Mapping[str, Any]] = []
    if config_path is not None:
        sources.append(load_research_campaign_config(Path(config_path)).to_dict())
    if config is not None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a research campaign mapping or ResearchCampaignConfig.")
        sources.append(dict(config))
    return resolve_research_campaign_config(
        *sources,
        cli_overrides=None if overrides is None else dict(overrides),
    )


def _run_campaign(config: Any) -> Any:
    from src.cli.run_research_campaign import run_research_campaign as run_campaign_impl

    return run_campaign_impl(config)


def _summarize_single_campaign_result(result: Any) -> ExecutionResult:
    summary = dict(result.campaign_summary)
    output_paths = _campaign_output_paths(result)
    stage_statuses = dict(summary.get("stage_statuses", {}))
    return ExecutionResult(
        workflow="research_campaign",
        run_id=result.campaign_run_id,
        name=result.campaign_run_id,
        artifact_dir=result.campaign_artifact_dir,
        manifest_path=result.campaign_manifest_path,
        output_paths=output_paths,
        metrics={
            "status": summary.get("status"),
            "stage_counts": _count_values(stage_statuses),
            "alpha_run_count": len(result.alpha_results),
            "strategy_run_count": len(result.strategy_results),
            "resumable_stage_count": len(summary.get("final_outcomes", {}).get("resumable_stage_names", [])),
            "reused_stage_count": sum(1 for state in stage_statuses.values() if state == "reused"),
        },
        extra={
            "workflow_type": "research_campaign",
            "stage_statuses": stage_statuses,
            "stage_records": [_stage_record_payload(record) for record in result.stage_records],
            "selected_run_ids": dict(summary.get("selected_run_ids", {})),
            "final_outcomes": dict(summary.get("final_outcomes", {})),
            "stage_execution": dict(summary.get("stage_execution", {})),
            "checkpoint_path": result.campaign_checkpoint_path.as_posix(),
            "preflight_status": result.preflight_summary.get("status"),
            "preflight_summary_path": result.preflight_summary_path.as_posix(),
        },
        raw_result=result,
    )


def _summarize_orchestration_result(result: Any) -> ExecutionResult:
    summary = dict(result.orchestration_summary)
    scenario_entries = list(summary.get("scenarios", []))
    scenario_status_counts = dict(summary.get("scenario_status_counts", {}))
    return ExecutionResult(
        workflow="research_campaign_orchestration",
        run_id=result.orchestration_run_id,
        name=result.orchestration_run_id,
        artifact_dir=result.orchestration_artifact_dir,
        manifest_path=result.orchestration_manifest_path,
        output_paths={
            "campaign_config_json": result.orchestration_artifact_dir / "campaign_config.json",
            "manifest_json": result.orchestration_manifest_path,
            "summary_json": result.orchestration_summary_path,
            "scenario_catalog_json": result.scenario_catalog_path,
            "scenario_matrix_csv": result.scenario_matrix_csv_path,
            "scenario_matrix_summary_json": result.scenario_matrix_summary_path,
            "expansion_preflight_json": result.expansion_preflight_path,
        },
        metrics={
            "status": summary.get("status"),
            "scenario_count": len(result.scenario_results),
            "scenario_status_counts": scenario_status_counts,
        },
        extra={
            "workflow_type": "research_campaign_orchestration",
            "scenario_count": len(result.scenario_results),
            "scenario_status_counts": scenario_status_counts,
            "scenarios": scenario_entries,
            "expansion_preflight": dict(result.expansion_preflight),
            "scenario_run_ids": [
                scenario_result.result.campaign_run_id
                for scenario_result in result.scenario_results
            ],
        },
        raw_result=result,
    )


def _campaign_output_paths(result: Any) -> dict[str, Path]:
    output_paths: dict[str, Path] = {
        "campaign_config_json": result.campaign_artifact_dir / "campaign_config.json",
        "checkpoint_json": result.campaign_checkpoint_path,
        "manifest_json": result.campaign_manifest_path,
        "summary_json": result.campaign_summary_path,
        "preflight_summary_json": result.preflight_summary_path,
    }
    optional_paths = {
        "milestone_report_summary_json": result.campaign_milestone_summary_path,
        "milestone_report_decision_log_json": result.campaign_milestone_decision_log_path,
        "milestone_report_manifest_json": result.campaign_milestone_manifest_path,
        "milestone_report_markdown": result.campaign_milestone_markdown_path,
    }
    output_paths.update(
        {
            name: path
            for name, path in optional_paths.items()
            if path is not None
        }
    )
    summary_output_paths = result.campaign_summary.get("output_paths", {})
    if isinstance(summary_output_paths, Mapping):
        for name, value in summary_output_paths.items():
            path = _optional_path(value)
            if path is not None:
                output_paths[str(name)] = path
    return output_paths


def _optional_path(value: Any) -> Path | None:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value)
    return None


def _stage_record_payload(record: Any) -> dict[str, Any]:
    return {
        "stage_name": record.stage_name,
        "status": record.status,
        "details": dict(record.details),
    }


def _count_values(values: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values.values():
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts
