from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.config.review import ReviewConfig, ReviewConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_CAMPAIGN_CONFIG = REPO_ROOT / "configs" / "research_campaign.yml"

_ROOT_SECTION_KEYS = frozenset(
    {
        "dataset_selection",
        "time_windows",
        "targets",
        "reuse_policy",
        "comparison",
        "candidate_selection",
        "portfolio",
        "review",
        "milestone_reporting",
        "outputs",
        "scenarios",
    }
)
_DATASET_SELECTION_KEYS = frozenset(
    {"dataset", "timeframe", "evaluation_horizon", "mapping_name", "tickers_path"}
)
_TIME_WINDOW_KEYS = frozenset(
    {"start", "end", "train_start", "train_end", "predict_start", "predict_end"}
)
_TARGET_KEYS = frozenset(
    {
        "alpha_names",
        "strategy_names",
        "portfolio_names",
        "alpha_catalog_path",
        "strategy_config_path",
        "portfolio_config_path",
    }
)
_REUSE_POLICY_KEYS = frozenset(
    {
        "enable_checkpoint_reuse",
        "reuse_prior_stages",
        "force_rerun_stages",
        "invalidate_downstream_after_stages",
    }
)
_COMPARISON_KEYS = frozenset(
    {
        "enabled",
        "from_registry",
        "top_k",
        "alpha_view",
        "alpha_metric",
        "alpha_sleeve_metric",
        "strategy_metric",
    }
)
_CANDIDATE_SELECTION_KEYS = frozenset(
    {
        "enabled",
        "artifacts_root",
        "alpha_name",
        "dataset",
        "timeframe",
        "evaluation_horizon",
        "mapping_name",
        "metric",
        "max_candidates",
        "eligibility",
        "redundancy",
        "allocation",
        "execution",
        "output",
    }
)
_ELIGIBILITY_KEYS = frozenset(
    {
        "min_ic",
        "min_mean_ic",
        "min_rank_ic",
        "min_mean_rank_ic",
        "min_ic_ir",
        "min_rank_ic_ir",
        "min_history_length",
        "min_coverage",
    }
)
_REDUNDANCY_KEYS = frozenset(
    {"max_correlation", "max_pairwise_correlation", "min_overlap_observations"}
)
_ALLOCATION_KEYS = frozenset(
    {
        "method",
        "allocation_method",
        "max_weight",
        "max_weight_per_candidate",
        "min_candidates",
        "min_allocation_candidate_count",
        "min_weight",
        "min_allocation_weight",
        "weight_sum_tolerance",
        "allocation_weight_sum_tolerance",
        "rounding_decimals",
        "allocation_rounding_decimals",
    }
)
_CANDIDATE_EXECUTION_KEYS = frozenset(
    {
        "strict_mode",
        "skip_eligibility",
        "skip_redundancy",
        "skip_allocation",
        "enable_review",
        "no_markdown_review",
        "register_run",
        "from_registry",
    }
)
_CANDIDATE_OUTPUT_KEYS = frozenset({"path", "registry_path", "review_output_path"})
_PORTFOLIO_KEYS = frozenset(
    {
        "enabled",
        "portfolio_name",
        "timeframe",
        "from_registry",
        "from_candidate_selection",
        "evaluation_path",
        "optimizer_method",
    }
)
_OUTPUT_KEYS = frozenset(
    {
        "alpha_artifacts_root",
        "candidate_selection_output_path",
        "candidate_selection_registry_path",
        "campaign_artifacts_root",
        "portfolio_artifacts_root",
        "comparison_output_path",
        "review_output_path",
    }
)
_SCENARIOS_KEYS = frozenset({"enabled", "matrix", "include", "max_scenarios", "max_values_per_axis"})
_SCENARIO_MATRIX_KEYS = frozenset({"name", "path", "values"})
_SCENARIOS_HARD_MAX = 1000
_SCENARIO_INCLUDE_KEYS = frozenset({"scenario_id", "description", "overrides"})
_MILESTONE_REPORTING_KEYS = frozenset({"enabled", "decision_categories", "output", "sections", "summary"})
_MILESTONE_REPORTING_OUTPUT_KEYS = frozenset({"include_markdown_report", "decision_log_render_formats"})
_MILESTONE_REPORTING_SECTION_KEYS = frozenset(
    {
        "campaign_scope",
        "selections",
        "key_findings",
        "key_metrics",
        "gate_outcomes",
        "risks",
        "next_steps",
        "open_questions",
        "decision_snapshot",
        "related_artifacts",
    }
)
_MILESTONE_REPORTING_SUMMARY_KEYS = frozenset({"include_stage_counts", "include_review_outcome"})
_SUPPORTED_DECISION_LOG_RENDER_FORMATS = frozenset({"markdown", "text"})
_SUPPORTED_ALPHA_VIEWS = frozenset({"forecast", "sleeve", "combined"})
_DEFAULT_CANDIDATE_SELECTION_OUTPUT_PATH = "artifacts/candidate_selection"
_CAMPAIGN_STAGE_NAMES = (
    "preflight",
    "research",
    "comparison",
    "candidate_selection",
    "portfolio",
    "candidate_review",
    "review",
)


class ResearchCampaignConfigError(ValueError):
    """Raised when unified research campaign configuration is malformed."""


@dataclass(frozen=True)
class DatasetSelectionConfig:
    dataset: str | None
    timeframe: str | None
    evaluation_horizon: int | None
    mapping_name: str | None
    tickers_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "timeframe": self.timeframe,
            "evaluation_horizon": self.evaluation_horizon,
            "mapping_name": self.mapping_name,
            "tickers_path": self.tickers_path,
        }


@dataclass(frozen=True)
class TimeWindowsConfig:
    start: str | None
    end: str | None
    train_start: str | None
    train_end: str | None
    predict_start: str | None
    predict_end: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "predict_start": self.predict_start,
            "predict_end": self.predict_end,
        }


@dataclass(frozen=True)
class CampaignTargetsConfig:
    alpha_names: tuple[str, ...]
    strategy_names: tuple[str, ...]
    portfolio_names: tuple[str, ...]
    alpha_catalog_path: str
    strategy_config_path: str
    portfolio_config_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_names": list(self.alpha_names),
            "strategy_names": list(self.strategy_names),
            "portfolio_names": list(self.portfolio_names),
            "alpha_catalog_path": self.alpha_catalog_path,
            "strategy_config_path": self.strategy_config_path,
            "portfolio_config_path": self.portfolio_config_path,
        }


@dataclass(frozen=True)
class CampaignComparisonConfig:
    enabled: bool
    from_registry: bool
    top_k: int | None
    alpha_view: str
    alpha_metric: str
    alpha_sleeve_metric: str
    strategy_metric: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "from_registry": self.from_registry,
            "top_k": self.top_k,
            "alpha_view": self.alpha_view,
            "alpha_metric": self.alpha_metric,
            "alpha_sleeve_metric": self.alpha_sleeve_metric,
            "strategy_metric": self.strategy_metric,
        }


@dataclass(frozen=True)
class CampaignReusePolicyConfig:
    enable_checkpoint_reuse: bool
    reuse_prior_stages: tuple[str, ...]
    force_rerun_stages: tuple[str, ...]
    invalidate_downstream_after_stages: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_checkpoint_reuse": self.enable_checkpoint_reuse,
            "reuse_prior_stages": list(self.reuse_prior_stages),
            "force_rerun_stages": list(self.force_rerun_stages),
            "invalidate_downstream_after_stages": list(self.invalidate_downstream_after_stages),
        }


@dataclass(frozen=True)
class CandidateEligibilityConfig:
    min_mean_ic: float | None
    min_mean_rank_ic: float | None
    min_ic_ir: float | None
    min_rank_ic_ir: float | None
    min_history_length: int | None
    min_coverage: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_mean_ic": self.min_mean_ic,
            "min_mean_rank_ic": self.min_mean_rank_ic,
            "min_ic_ir": self.min_ic_ir,
            "min_rank_ic_ir": self.min_rank_ic_ir,
            "min_history_length": self.min_history_length,
            "min_coverage": self.min_coverage,
        }


@dataclass(frozen=True)
class CandidateRedundancyConfig:
    max_pairwise_correlation: float | None
    min_overlap_observations: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_pairwise_correlation": self.max_pairwise_correlation,
            "min_overlap_observations": self.min_overlap_observations,
        }


@dataclass(frozen=True)
class CandidateAllocationConfig:
    allocation_method: str
    max_weight_per_candidate: float | None
    min_allocation_candidate_count: int | None
    min_allocation_weight: float | None
    allocation_weight_sum_tolerance: float
    allocation_rounding_decimals: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocation_method": self.allocation_method,
            "max_weight_per_candidate": self.max_weight_per_candidate,
            "min_allocation_candidate_count": self.min_allocation_candidate_count,
            "min_allocation_weight": self.min_allocation_weight,
            "allocation_weight_sum_tolerance": self.allocation_weight_sum_tolerance,
            "allocation_rounding_decimals": self.allocation_rounding_decimals,
        }


@dataclass(frozen=True)
class CandidateExecutionConfig:
    strict_mode: bool
    skip_eligibility: bool
    skip_redundancy: bool
    skip_allocation: bool
    enable_review: bool
    no_markdown_review: bool
    register_run: bool
    from_registry: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "strict_mode": self.strict_mode,
            "skip_eligibility": self.skip_eligibility,
            "skip_redundancy": self.skip_redundancy,
            "skip_allocation": self.skip_allocation,
            "enable_review": self.enable_review,
            "no_markdown_review": self.no_markdown_review,
            "register_run": self.register_run,
            "from_registry": self.from_registry,
        }


@dataclass(frozen=True)
class CandidateOutputConfig:
    path: str
    registry_path: str | None
    review_output_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "registry_path": self.registry_path,
            "review_output_path": self.review_output_path,
        }


@dataclass(frozen=True)
class CampaignCandidateSelectionConfig:
    enabled: bool
    artifacts_root: str
    alpha_name: str | None
    dataset: str | None
    timeframe: str | None
    evaluation_horizon: int | None
    mapping_name: str | None
    metric: str
    max_candidates: int | None
    eligibility: CandidateEligibilityConfig
    redundancy: CandidateRedundancyConfig
    allocation: CandidateAllocationConfig
    execution: CandidateExecutionConfig
    output: CandidateOutputConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "artifacts_root": self.artifacts_root,
            "alpha_name": self.alpha_name,
            "dataset": self.dataset,
            "timeframe": self.timeframe,
            "evaluation_horizon": self.evaluation_horizon,
            "mapping_name": self.mapping_name,
            "metric": self.metric,
            "max_candidates": self.max_candidates,
            "eligibility": self.eligibility.to_dict(),
            "redundancy": self.redundancy.to_dict(),
            "allocation": self.allocation.to_dict(),
            "execution": self.execution.to_dict(),
            "output": self.output.to_dict(),
        }


@dataclass(frozen=True)
class CampaignPortfolioConfig:
    enabled: bool
    portfolio_name: str | None
    timeframe: str | None
    from_registry: bool
    from_candidate_selection: bool
    evaluation_path: str | None
    optimizer_method: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "portfolio_name": self.portfolio_name,
            "timeframe": self.timeframe,
            "from_registry": self.from_registry,
            "from_candidate_selection": self.from_candidate_selection,
            "evaluation_path": self.evaluation_path,
            "optimizer_method": self.optimizer_method,
        }


@dataclass(frozen=True)
class CampaignOutputsConfig:
    alpha_artifacts_root: str
    candidate_selection_output_path: str
    candidate_selection_registry_path: str | None
    campaign_artifacts_root: str
    portfolio_artifacts_root: str
    comparison_output_path: str | None
    review_output_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_artifacts_root": self.alpha_artifacts_root,
            "candidate_selection_output_path": self.candidate_selection_output_path,
            "candidate_selection_registry_path": self.candidate_selection_registry_path,
            "campaign_artifacts_root": self.campaign_artifacts_root,
            "portfolio_artifacts_root": self.portfolio_artifacts_root,
            "comparison_output_path": self.comparison_output_path,
            "review_output_path": self.review_output_path,
        }


@dataclass(frozen=True)
class CampaignMilestoneReportingOutputConfig:
    include_markdown_report: bool
    decision_log_render_formats: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "include_markdown_report": self.include_markdown_report,
            "decision_log_render_formats": list(self.decision_log_render_formats),
        }


@dataclass(frozen=True)
class CampaignMilestoneReportingSectionsConfig:
    campaign_scope: bool
    selections: bool
    key_findings: bool
    key_metrics: bool
    gate_outcomes: bool
    risks: bool
    next_steps: bool
    open_questions: bool
    decision_snapshot: bool
    related_artifacts: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_scope": self.campaign_scope,
            "selections": self.selections,
            "key_findings": self.key_findings,
            "key_metrics": self.key_metrics,
            "gate_outcomes": self.gate_outcomes,
            "risks": self.risks,
            "next_steps": self.next_steps,
            "open_questions": self.open_questions,
            "decision_snapshot": self.decision_snapshot,
            "related_artifacts": self.related_artifacts,
        }


@dataclass(frozen=True)
class CampaignMilestoneReportingSummaryConfig:
    include_stage_counts: bool
    include_review_outcome: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "include_stage_counts": self.include_stage_counts,
            "include_review_outcome": self.include_review_outcome,
        }


@dataclass(frozen=True)
class CampaignMilestoneReportingConfig:
    enabled: bool
    decision_categories: tuple[str, ...]
    output: CampaignMilestoneReportingOutputConfig
    sections: CampaignMilestoneReportingSectionsConfig
    summary: CampaignMilestoneReportingSummaryConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "decision_categories": list(self.decision_categories),
            "output": self.output.to_dict(),
            "sections": self.sections.to_dict(),
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True)
class CampaignScenarioMatrixAxisConfig:
    name: str
    path: str
    values: tuple[Any, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "values": list(self.values),
        }


@dataclass(frozen=True)
class CampaignScenarioIncludeConfig:
    scenario_id: str
    description: str | None
    overrides: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "overrides": _canonicalize_value(dict(self.overrides)),
        }


@dataclass(frozen=True)
class CampaignScenariosConfig:
    enabled: bool
    matrix: tuple[CampaignScenarioMatrixAxisConfig, ...]
    include: tuple[CampaignScenarioIncludeConfig, ...]
    max_scenarios: int | None
    max_values_per_axis: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "matrix": [axis.to_dict() for axis in self.matrix],
            "include": [scenario.to_dict() for scenario in self.include],
            "max_scenarios": self.max_scenarios,
            "max_values_per_axis": self.max_values_per_axis,
        }


@dataclass(frozen=True)
class ResolvedResearchCampaignScenario:
    scenario_id: str
    description: str | None
    source: str
    sweep_values: dict[str, Any]
    config: "ResearchCampaignConfig"
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        effective_config = self.config.to_dict()
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "source": self.source,
            "sweep_values": _canonicalize_value(dict(self.sweep_values)),
            "fingerprint": self.fingerprint,
            "config": effective_config,
            "effective_config": effective_config,
        }


@dataclass(frozen=True)
class ResearchCampaignConfig:
    dataset_selection: DatasetSelectionConfig
    time_windows: TimeWindowsConfig
    targets: CampaignTargetsConfig
    reuse_policy: CampaignReusePolicyConfig
    comparison: CampaignComparisonConfig
    candidate_selection: CampaignCandidateSelectionConfig
    portfolio: CampaignPortfolioConfig
    review: ReviewConfig
    milestone_reporting: CampaignMilestoneReportingConfig
    outputs: CampaignOutputsConfig
    scenarios: CampaignScenariosConfig

    @classmethod
    def default(cls) -> "ResearchCampaignConfig":
        outputs = CampaignOutputsConfig(
            alpha_artifacts_root="artifacts/alpha",
            candidate_selection_output_path=_DEFAULT_CANDIDATE_SELECTION_OUTPUT_PATH,
            candidate_selection_registry_path=None,
            campaign_artifacts_root="artifacts/research_campaigns",
            portfolio_artifacts_root="artifacts/portfolios",
            comparison_output_path=None,
            review_output_path=None,
        )
        return cls(
            dataset_selection=DatasetSelectionConfig(
                dataset=None,
                timeframe=None,
                evaluation_horizon=None,
                mapping_name=None,
                tickers_path=None,
            ),
            time_windows=TimeWindowsConfig(
                start=None,
                end=None,
                train_start=None,
                train_end=None,
                predict_start=None,
                predict_end=None,
            ),
            targets=CampaignTargetsConfig(
                alpha_names=(),
                strategy_names=(),
                portfolio_names=(),
                alpha_catalog_path="configs/alphas.yml",
                strategy_config_path="configs/strategies.yml",
                portfolio_config_path="configs/portfolios.yml",
            ),
            reuse_policy=CampaignReusePolicyConfig(
                enable_checkpoint_reuse=True,
                reuse_prior_stages=_CAMPAIGN_STAGE_NAMES,
                force_rerun_stages=(),
                invalidate_downstream_after_stages=(),
            ),
            comparison=CampaignComparisonConfig(
                enabled=False,
                from_registry=True,
                top_k=None,
                alpha_view="combined",
                alpha_metric="ic_ir",
                alpha_sleeve_metric="sharpe_ratio",
                strategy_metric="sharpe_ratio",
            ),
            candidate_selection=CampaignCandidateSelectionConfig(
                enabled=False,
                artifacts_root="artifacts/alpha",
                alpha_name=None,
                dataset=None,
                timeframe=None,
                evaluation_horizon=None,
                mapping_name=None,
                metric="ic_ir",
                max_candidates=None,
                eligibility=CandidateEligibilityConfig(
                    min_mean_ic=None,
                    min_mean_rank_ic=None,
                    min_ic_ir=None,
                    min_rank_ic_ir=None,
                    min_history_length=None,
                    min_coverage=None,
                ),
                redundancy=CandidateRedundancyConfig(
                    max_pairwise_correlation=None,
                    min_overlap_observations=None,
                ),
                allocation=CandidateAllocationConfig(
                    allocation_method="equal_weight",
                    max_weight_per_candidate=None,
                    min_allocation_candidate_count=None,
                    min_allocation_weight=None,
                    allocation_weight_sum_tolerance=1e-12,
                    allocation_rounding_decimals=12,
                ),
                execution=CandidateExecutionConfig(
                    strict_mode=False,
                    skip_eligibility=False,
                    skip_redundancy=False,
                    skip_allocation=False,
                    enable_review=False,
                    no_markdown_review=False,
                    register_run=False,
                    from_registry=False,
                ),
                output=CandidateOutputConfig(
                    path=outputs.candidate_selection_output_path,
                    registry_path=outputs.candidate_selection_registry_path,
                    review_output_path=outputs.review_output_path,
                ),
            ),
            portfolio=CampaignPortfolioConfig(
                enabled=False,
                portfolio_name=None,
                timeframe=None,
                from_registry=False,
                from_candidate_selection=False,
                evaluation_path=None,
                optimizer_method=None,
            ),
            review=ReviewConfig.default(),
            milestone_reporting=CampaignMilestoneReportingConfig(
                enabled=True,
                decision_categories=(),
                output=CampaignMilestoneReportingOutputConfig(
                    include_markdown_report=True,
                    decision_log_render_formats=("markdown", "text"),
                ),
                sections=CampaignMilestoneReportingSectionsConfig(
                    campaign_scope=True,
                    selections=True,
                    key_findings=True,
                    key_metrics=True,
                    gate_outcomes=True,
                    risks=True,
                    next_steps=True,
                    open_questions=True,
                    decision_snapshot=True,
                    related_artifacts=True,
                ),
                summary=CampaignMilestoneReportingSummaryConfig(
                    include_stage_counts=True,
                    include_review_outcome=True,
                ),
            ),
            outputs=outputs,
            scenarios=CampaignScenariosConfig(
                enabled=False,
                matrix=(),
                include=(),
                max_scenarios=None,
                max_values_per_axis=None,
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        base: "ResearchCampaignConfig" | None = None,
    ) -> "ResearchCampaignConfig":
        if payload is None:
            return base or cls.default()
        if not isinstance(payload, Mapping):
            raise ResearchCampaignConfigError("Research campaign configuration must be a mapping.")

        seed = base or cls.default()
        container = _extract_root_container(payload)
        outputs = _resolve_outputs(container.get("outputs"), base=seed.outputs)
        dataset_selection = _resolve_dataset_selection(
            container.get("dataset_selection"), base=seed.dataset_selection
        )
        time_windows = _resolve_time_windows(container.get("time_windows"), base=seed.time_windows)
        targets = _resolve_targets(container.get("targets"), base=seed.targets)
        reuse_policy = _resolve_reuse_policy(container.get("reuse_policy"), base=seed.reuse_policy)
        comparison = _resolve_comparison(container.get("comparison"), base=seed.comparison)
        candidate_selection = _resolve_candidate_selection(
            container.get("candidate_selection"),
            base=seed.candidate_selection,
            shared_dataset=dataset_selection,
            shared_targets=targets,
            outputs=outputs,
        )
        portfolio = _resolve_portfolio(
            container.get("portfolio"),
            base=seed.portfolio,
            shared_dataset=dataset_selection,
            shared_targets=targets,
        )
        review = _resolve_review(
            container.get("review"),
            base=seed.review,
            shared_dataset=dataset_selection,
            shared_targets=targets,
            outputs=outputs,
        )
        milestone_reporting = _resolve_milestone_reporting(
            container.get("milestone_reporting"),
            base=seed.milestone_reporting,
        )
        scenarios = _resolve_scenarios(
            container.get("scenarios"),
            base=seed.scenarios,
        )
        resolved = cls(
            dataset_selection=dataset_selection,
            time_windows=time_windows,
            targets=targets,
            reuse_policy=reuse_policy,
            comparison=comparison,
            candidate_selection=candidate_selection,
            portfolio=portfolio,
            review=review,
            milestone_reporting=milestone_reporting,
            outputs=outputs,
            scenarios=scenarios,
        )
        if resolved.scenarios.enabled:
            build_research_campaign_scenario_catalog(resolved)
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return _canonicalize_value(
            {
                "dataset_selection": self.dataset_selection.to_dict(),
                "time_windows": self.time_windows.to_dict(),
                "targets": self.targets.to_dict(),
                "reuse_policy": self.reuse_policy.to_dict(),
                "comparison": self.comparison.to_dict(),
                "candidate_selection": self.candidate_selection.to_dict(),
                "portfolio": self.portfolio.to_dict(),
                "review": self.review.to_dict(),
                "milestone_reporting": self.milestone_reporting.to_dict(),
                "outputs": self.outputs.to_dict(),
                "scenarios": self.scenarios.to_dict(),
            }
        )


def load_research_campaign_config(path: Path = RESEARCH_CAMPAIGN_CONFIG) -> ResearchCampaignConfig:
    """Load repository campaign defaults from YAML/JSON, or return code defaults when absent."""

    if not path.exists():
        return ResearchCampaignConfig.default()

    with path.open("r", encoding="utf-8") as handle:
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(handle) or {}
        else:
            raise ResearchCampaignConfigError(
                f"Unsupported research campaign config format {path.suffix!r}. Use JSON, YAML, or YML."
            )
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError(
            "Research campaign config file must contain a top-level mapping."
        )
    return ResearchCampaignConfig.from_mapping(payload)


def resolve_research_campaign_config(
    *sources: Mapping[str, Any] | None,
    cli_overrides: Mapping[str, Any] | None = None,
    defaults_path: Path | None = None,
) -> ResearchCampaignConfig:
    """
    Resolve repository defaults, config values, and CLI overrides into one research campaign contract.

    Precedence is deterministic: repository defaults < config < CLI overrides.
    """

    resolved = load_research_campaign_config(
        RESEARCH_CAMPAIGN_CONFIG if defaults_path is None else defaults_path
    )
    for source in sources:
        if source is None:
            continue
        resolved = ResearchCampaignConfig.from_mapping(source, base=resolved)
    if cli_overrides is not None:
        resolved = ResearchCampaignConfig.from_mapping(cli_overrides, base=resolved)
    return resolved


def _extract_root_container(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if "research_campaign" in payload:
        container = payload.get("research_campaign")
        if not isinstance(container, Mapping):
            raise ResearchCampaignConfigError(
                "Research campaign field 'research_campaign' must be a mapping."
            )
    else:
        container = payload

    unknown_keys = sorted(set(container) - _ROOT_SECTION_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign configuration contains unsupported keys: {unknown_keys}."
        )
    return container


def _resolve_dataset_selection(
    payload: Any, *, base: DatasetSelectionConfig
) -> DatasetSelectionConfig:
    if payload is None:
        return base
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError(
            "Research campaign field 'dataset_selection' must be a mapping."
        )
    unknown_keys = sorted(set(payload) - _DATASET_SELECTION_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign field 'dataset_selection' contains unsupported keys: {unknown_keys}."
        )
    return DatasetSelectionConfig(
        dataset=_normalize_optional_string(
            payload.get("dataset"), field_name="dataset_selection.dataset", default=base.dataset
        ),
        timeframe=_normalize_optional_string(
            payload.get("timeframe"),
            field_name="dataset_selection.timeframe",
            default=base.timeframe,
        ),
        evaluation_horizon=_resolve_optional_positive_int(
            payload.get("evaluation_horizon"),
            field_name="dataset_selection.evaluation_horizon",
            default=base.evaluation_horizon,
        ),
        mapping_name=_normalize_optional_string(
            payload.get("mapping_name"),
            field_name="dataset_selection.mapping_name",
            default=base.mapping_name,
        ),
        tickers_path=_normalize_optional_path_string(
            payload.get("tickers_path"),
            field_name="dataset_selection.tickers_path",
            default=base.tickers_path,
        ),
    )


def _resolve_time_windows(payload: Any, *, base: TimeWindowsConfig) -> TimeWindowsConfig:
    if payload is None:
        return base
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError(
            "Research campaign field 'time_windows' must be a mapping."
        )
    unknown_keys = sorted(set(payload) - _TIME_WINDOW_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign field 'time_windows' contains unsupported keys: {unknown_keys}."
        )
    return TimeWindowsConfig(
        start=_normalize_optional_temporal_string(
            payload.get("start"), field_name="time_windows.start", default=base.start
        ),
        end=_normalize_optional_temporal_string(
            payload.get("end"), field_name="time_windows.end", default=base.end
        ),
        train_start=_normalize_optional_temporal_string(
            payload.get("train_start"),
            field_name="time_windows.train_start",
            default=base.train_start,
        ),
        train_end=_normalize_optional_temporal_string(
            payload.get("train_end"), field_name="time_windows.train_end", default=base.train_end
        ),
        predict_start=_normalize_optional_temporal_string(
            payload.get("predict_start"),
            field_name="time_windows.predict_start",
            default=base.predict_start,
        ),
        predict_end=_normalize_optional_temporal_string(
            payload.get("predict_end"),
            field_name="time_windows.predict_end",
            default=base.predict_end,
        ),
    )


def _resolve_targets(payload: Any, *, base: CampaignTargetsConfig) -> CampaignTargetsConfig:
    if payload is None:
        return base
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError("Research campaign field 'targets' must be a mapping.")
    unknown_keys = sorted(set(payload) - _TARGET_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign field 'targets' contains unsupported keys: {unknown_keys}."
        )
    return CampaignTargetsConfig(
        alpha_names=_normalize_string_collection(
            payload.get("alpha_names"), field_name="targets.alpha_names", default=base.alpha_names
        ),
        strategy_names=_normalize_string_collection(
            payload.get("strategy_names"),
            field_name="targets.strategy_names",
            default=base.strategy_names,
        ),
        portfolio_names=_normalize_string_collection(
            payload.get("portfolio_names"),
            field_name="targets.portfolio_names",
            default=base.portfolio_names,
        ),
        alpha_catalog_path=_normalize_required_path_string(
            payload.get("alpha_catalog_path", base.alpha_catalog_path),
            field_name="targets.alpha_catalog_path",
        ),
        strategy_config_path=_normalize_required_path_string(
            payload.get("strategy_config_path", base.strategy_config_path),
            field_name="targets.strategy_config_path",
        ),
        portfolio_config_path=_normalize_required_path_string(
            payload.get("portfolio_config_path", base.portfolio_config_path),
            field_name="targets.portfolio_config_path",
        ),
    )


def _resolve_comparison(
    payload: Any, *, base: CampaignComparisonConfig
) -> CampaignComparisonConfig:
    if payload is None:
        return base
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError("Research campaign field 'comparison' must be a mapping.")
    unknown_keys = sorted(set(payload) - _COMPARISON_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign field 'comparison' contains unsupported keys: {unknown_keys}."
        )
    return CampaignComparisonConfig(
        enabled=_resolve_bool(
            payload.get("enabled"), field_name="comparison.enabled", default=base.enabled
        ),
        from_registry=_resolve_bool(
            payload.get("from_registry"),
            field_name="comparison.from_registry",
            default=base.from_registry,
        ),
        top_k=_resolve_optional_positive_int(
            payload.get("top_k"), field_name="comparison.top_k", default=base.top_k
        ),
        alpha_view=_resolve_choice(
            payload.get("alpha_view", base.alpha_view),
            field_name="comparison.alpha_view",
            supported_values=_SUPPORTED_ALPHA_VIEWS,
        ),
        alpha_metric=_normalize_required_string(
            payload.get("alpha_metric", base.alpha_metric), field_name="comparison.alpha_metric"
        ),
        alpha_sleeve_metric=_normalize_required_string(
            payload.get("alpha_sleeve_metric", base.alpha_sleeve_metric),
            field_name="comparison.alpha_sleeve_metric",
        ),
        strategy_metric=_normalize_required_string(
            payload.get("strategy_metric", base.strategy_metric),
            field_name="comparison.strategy_metric",
        ),
    )


def _resolve_reuse_policy(
    payload: Any,
    *,
    base: CampaignReusePolicyConfig,
) -> CampaignReusePolicyConfig:
    if payload is None:
        return base
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError("Research campaign field 'reuse_policy' must be a mapping.")
    unknown_keys = sorted(set(payload) - _REUSE_POLICY_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign field 'reuse_policy' contains unsupported keys: {unknown_keys}."
        )
    return CampaignReusePolicyConfig(
        enable_checkpoint_reuse=_resolve_bool(
            payload.get("enable_checkpoint_reuse"),
            field_name="reuse_policy.enable_checkpoint_reuse",
            default=base.enable_checkpoint_reuse,
        ),
        reuse_prior_stages=_normalize_stage_name_collection(
            payload.get("reuse_prior_stages"),
            field_name="reuse_policy.reuse_prior_stages",
            default=base.reuse_prior_stages,
        ),
        force_rerun_stages=_normalize_stage_name_collection(
            payload.get("force_rerun_stages"),
            field_name="reuse_policy.force_rerun_stages",
            default=base.force_rerun_stages,
        ),
        invalidate_downstream_after_stages=_normalize_stage_name_collection(
            payload.get("invalidate_downstream_after_stages"),
            field_name="reuse_policy.invalidate_downstream_after_stages",
            default=base.invalidate_downstream_after_stages,
        ),
    )


def _resolve_candidate_selection(
    payload: Any,
    *,
    base: CampaignCandidateSelectionConfig,
    shared_dataset: DatasetSelectionConfig,
    shared_targets: CampaignTargetsConfig,
    outputs: CampaignOutputsConfig,
) -> CampaignCandidateSelectionConfig:
    inherited = CampaignCandidateSelectionConfig(
        **{
            **base.__dict__,
            "artifacts_root": (
                outputs.alpha_artifacts_root
                if base.artifacts_root == "artifacts/alpha"
                else base.artifacts_root
            ),
            "dataset": base.dataset if base.dataset is not None else shared_dataset.dataset,
            "timeframe": base.timeframe if base.timeframe is not None else shared_dataset.timeframe,
            "evaluation_horizon": base.evaluation_horizon
            if base.evaluation_horizon is not None
            else shared_dataset.evaluation_horizon,
            "mapping_name": base.mapping_name
            if base.mapping_name is not None
            else shared_dataset.mapping_name,
            "alpha_name": base.alpha_name
            if base.alpha_name is not None
            else _single_or_none(shared_targets.alpha_names),
            "output": CandidateOutputConfig(
                path=(
                    outputs.candidate_selection_output_path
                    if base.output.path == _DEFAULT_CANDIDATE_SELECTION_OUTPUT_PATH
                    else base.output.path
                ),
                registry_path=base.output.registry_path
                if base.output.registry_path is not None
                else outputs.candidate_selection_registry_path,
                review_output_path=base.output.review_output_path
                if base.output.review_output_path is not None
                else outputs.review_output_path,
            ),
        }
    )
    if payload is None:
        return inherited
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError(
            "Research campaign field 'candidate_selection' must be a mapping."
        )
    unknown_keys = sorted(set(payload) - _CANDIDATE_SELECTION_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign field 'candidate_selection' contains unsupported keys: {unknown_keys}."
        )

    return CampaignCandidateSelectionConfig(
        enabled=_resolve_bool(
            payload.get("enabled"),
            field_name="candidate_selection.enabled",
            default=inherited.enabled,
        ),
        artifacts_root=_normalize_required_path_string(
            payload.get("artifacts_root", inherited.artifacts_root),
            field_name="candidate_selection.artifacts_root",
        ),
        alpha_name=_normalize_optional_string(
            payload.get("alpha_name"),
            field_name="candidate_selection.alpha_name",
            default=inherited.alpha_name,
        ),
        dataset=_normalize_optional_string(
            payload.get("dataset"),
            field_name="candidate_selection.dataset",
            default=inherited.dataset,
        ),
        timeframe=_normalize_optional_string(
            payload.get("timeframe"),
            field_name="candidate_selection.timeframe",
            default=inherited.timeframe,
        ),
        evaluation_horizon=_resolve_optional_positive_int(
            payload.get("evaluation_horizon"),
            field_name="candidate_selection.evaluation_horizon",
            default=inherited.evaluation_horizon,
        ),
        mapping_name=_normalize_optional_string(
            payload.get("mapping_name"),
            field_name="candidate_selection.mapping_name",
            default=inherited.mapping_name,
        ),
        metric=_normalize_required_string(
            payload.get("metric", inherited.metric), field_name="candidate_selection.metric"
        ),
        max_candidates=_resolve_optional_positive_int(
            payload.get("max_candidates"),
            field_name="candidate_selection.max_candidates",
            default=inherited.max_candidates,
        ),
        eligibility=_resolve_candidate_eligibility(
            payload.get("eligibility"), base=inherited.eligibility
        ),
        redundancy=_resolve_candidate_redundancy(
            payload.get("redundancy"), base=inherited.redundancy
        ),
        allocation=_resolve_candidate_allocation(
            payload.get("allocation"), base=inherited.allocation
        ),
        execution=_resolve_candidate_execution(payload.get("execution"), base=inherited.execution),
        output=_resolve_candidate_output(
            payload.get("output"), base=inherited.output, outputs=outputs
        ),
    )


def _resolve_candidate_eligibility(
    payload: Any, *, base: CandidateEligibilityConfig
) -> CandidateEligibilityConfig:
    if payload is None:
        return base
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError(
            "Research campaign field 'candidate_selection.eligibility' must be a mapping."
        )
    unknown_keys = sorted(set(payload) - _ELIGIBILITY_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign field 'candidate_selection.eligibility' contains unsupported keys: {unknown_keys}."
        )
    return CandidateEligibilityConfig(
        min_mean_ic=_resolve_optional_float(
            payload.get("min_mean_ic", payload.get("min_ic")),
            field_name="candidate_selection.eligibility.min_mean_ic",
            default=base.min_mean_ic,
        ),
        min_mean_rank_ic=_resolve_optional_float(
            payload.get("min_mean_rank_ic", payload.get("min_rank_ic")),
            field_name="candidate_selection.eligibility.min_mean_rank_ic",
            default=base.min_mean_rank_ic,
        ),
        min_ic_ir=_resolve_optional_float(
            payload.get("min_ic_ir"),
            field_name="candidate_selection.eligibility.min_ic_ir",
            default=base.min_ic_ir,
        ),
        min_rank_ic_ir=_resolve_optional_float(
            payload.get("min_rank_ic_ir"),
            field_name="candidate_selection.eligibility.min_rank_ic_ir",
            default=base.min_rank_ic_ir,
        ),
        min_history_length=_resolve_optional_positive_int(
            payload.get("min_history_length"),
            field_name="candidate_selection.eligibility.min_history_length",
            default=base.min_history_length,
        ),
        min_coverage=_resolve_optional_probability(
            payload.get("min_coverage"),
            field_name="candidate_selection.eligibility.min_coverage",
            default=base.min_coverage,
        ),
    )


def _resolve_candidate_redundancy(
    payload: Any, *, base: CandidateRedundancyConfig
) -> CandidateRedundancyConfig:
    if payload is None:
        return base
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError(
            "Research campaign field 'candidate_selection.redundancy' must be a mapping."
        )
    unknown_keys = sorted(set(payload) - _REDUNDANCY_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign field 'candidate_selection.redundancy' contains unsupported keys: {unknown_keys}."
        )
    return CandidateRedundancyConfig(
        max_pairwise_correlation=_resolve_optional_probability(
            payload.get("max_pairwise_correlation", payload.get("max_correlation")),
            field_name="candidate_selection.redundancy.max_pairwise_correlation",
            default=base.max_pairwise_correlation,
        ),
        min_overlap_observations=_resolve_optional_positive_int(
            payload.get("min_overlap_observations"),
            field_name="candidate_selection.redundancy.min_overlap_observations",
            default=base.min_overlap_observations,
        ),
    )


def _resolve_candidate_allocation(
    payload: Any, *, base: CandidateAllocationConfig
) -> CandidateAllocationConfig:
    if payload is None:
        return base
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError(
            "Research campaign field 'candidate_selection.allocation' must be a mapping."
        )
    unknown_keys = sorted(set(payload) - _ALLOCATION_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign field 'candidate_selection.allocation' contains unsupported keys: {unknown_keys}."
        )
    return CandidateAllocationConfig(
        allocation_method=_normalize_required_string(
            payload.get("allocation_method", payload.get("method", base.allocation_method)),
            field_name="candidate_selection.allocation.allocation_method",
        ),
        max_weight_per_candidate=_resolve_optional_probability(
            payload.get("max_weight_per_candidate", payload.get("max_weight")),
            field_name="candidate_selection.allocation.max_weight_per_candidate",
            default=base.max_weight_per_candidate,
        ),
        min_allocation_candidate_count=_resolve_optional_positive_int(
            payload.get("min_allocation_candidate_count", payload.get("min_candidates")),
            field_name="candidate_selection.allocation.min_allocation_candidate_count",
            default=base.min_allocation_candidate_count,
        ),
        min_allocation_weight=_resolve_optional_probability(
            payload.get("min_allocation_weight", payload.get("min_weight")),
            field_name="candidate_selection.allocation.min_allocation_weight",
            default=base.min_allocation_weight,
        ),
        allocation_weight_sum_tolerance=_resolve_non_negative_float(
            payload.get(
                "allocation_weight_sum_tolerance",
                payload.get("weight_sum_tolerance", base.allocation_weight_sum_tolerance),
            ),
            field_name="candidate_selection.allocation.allocation_weight_sum_tolerance",
        ),
        allocation_rounding_decimals=_resolve_non_negative_int(
            payload.get(
                "allocation_rounding_decimals",
                payload.get("rounding_decimals", base.allocation_rounding_decimals),
            ),
            field_name="candidate_selection.allocation.allocation_rounding_decimals",
        ),
    )


def _resolve_candidate_execution(
    payload: Any, *, base: CandidateExecutionConfig
) -> CandidateExecutionConfig:
    if payload is None:
        return base
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError(
            "Research campaign field 'candidate_selection.execution' must be a mapping."
        )
    unknown_keys = sorted(set(payload) - _CANDIDATE_EXECUTION_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign field 'candidate_selection.execution' contains unsupported keys: {unknown_keys}."
        )
    return CandidateExecutionConfig(
        strict_mode=_resolve_bool(
            payload.get("strict_mode"),
            field_name="candidate_selection.execution.strict_mode",
            default=base.strict_mode,
        ),
        skip_eligibility=_resolve_bool(
            payload.get("skip_eligibility"),
            field_name="candidate_selection.execution.skip_eligibility",
            default=base.skip_eligibility,
        ),
        skip_redundancy=_resolve_bool(
            payload.get("skip_redundancy"),
            field_name="candidate_selection.execution.skip_redundancy",
            default=base.skip_redundancy,
        ),
        skip_allocation=_resolve_bool(
            payload.get("skip_allocation"),
            field_name="candidate_selection.execution.skip_allocation",
            default=base.skip_allocation,
        ),
        enable_review=_resolve_bool(
            payload.get("enable_review"),
            field_name="candidate_selection.execution.enable_review",
            default=base.enable_review,
        ),
        no_markdown_review=_resolve_bool(
            payload.get("no_markdown_review"),
            field_name="candidate_selection.execution.no_markdown_review",
            default=base.no_markdown_review,
        ),
        register_run=_resolve_bool(
            payload.get("register_run"),
            field_name="candidate_selection.execution.register_run",
            default=base.register_run,
        ),
        from_registry=_resolve_bool(
            payload.get("from_registry"),
            field_name="candidate_selection.execution.from_registry",
            default=base.from_registry,
        ),
    )


def _resolve_candidate_output(
    payload: Any, *, base: CandidateOutputConfig, outputs: CampaignOutputsConfig
) -> CandidateOutputConfig:
    if payload is None:
        return CandidateOutputConfig(
            path=(
                outputs.candidate_selection_output_path
                if base.path == _DEFAULT_CANDIDATE_SELECTION_OUTPUT_PATH
                else base.path
            ),
            registry_path=base.registry_path
            if base.registry_path is not None
            else outputs.candidate_selection_registry_path,
            review_output_path=base.review_output_path
            if base.review_output_path is not None
            else outputs.review_output_path,
        )
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError(
            "Research campaign field 'candidate_selection.output' must be a mapping."
        )
    unknown_keys = sorted(set(payload) - _CANDIDATE_OUTPUT_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign field 'candidate_selection.output' contains unsupported keys: {unknown_keys}."
        )
    return CandidateOutputConfig(
        path=_normalize_required_path_string(
            payload.get("path", base.path or outputs.candidate_selection_output_path),
            field_name="candidate_selection.output.path",
        ),
        registry_path=_normalize_optional_path_string(
            payload.get("registry_path"),
            field_name="candidate_selection.output.registry_path",
            default=base.registry_path
            if base.registry_path is not None
            else outputs.candidate_selection_registry_path,
        ),
        review_output_path=_normalize_optional_path_string(
            payload.get("review_output_path"),
            field_name="candidate_selection.output.review_output_path",
            default=base.review_output_path
            if base.review_output_path is not None
            else outputs.review_output_path,
        ),
    )


def _resolve_portfolio(
    payload: Any,
    *,
    base: CampaignPortfolioConfig,
    shared_dataset: DatasetSelectionConfig,
    shared_targets: CampaignTargetsConfig,
) -> CampaignPortfolioConfig:
    inherited = CampaignPortfolioConfig(
        **{
            **base.__dict__,
            "portfolio_name": base.portfolio_name
            if base.portfolio_name is not None
            else _single_or_none(shared_targets.portfolio_names),
            "timeframe": base.timeframe if base.timeframe is not None else shared_dataset.timeframe,
        }
    )
    if payload is None:
        return inherited
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError("Research campaign field 'portfolio' must be a mapping.")
    unknown_keys = sorted(set(payload) - _PORTFOLIO_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign field 'portfolio' contains unsupported keys: {unknown_keys}."
        )
    return CampaignPortfolioConfig(
        enabled=_resolve_bool(
            payload.get("enabled"), field_name="portfolio.enabled", default=inherited.enabled
        ),
        portfolio_name=_normalize_optional_string(
            payload.get("portfolio_name"),
            field_name="portfolio.portfolio_name",
            default=inherited.portfolio_name,
        ),
        timeframe=_normalize_optional_string(
            payload.get("timeframe"), field_name="portfolio.timeframe", default=inherited.timeframe
        ),
        from_registry=_resolve_bool(
            payload.get("from_registry"),
            field_name="portfolio.from_registry",
            default=inherited.from_registry,
        ),
        from_candidate_selection=_resolve_bool(
            payload.get("from_candidate_selection"),
            field_name="portfolio.from_candidate_selection",
            default=inherited.from_candidate_selection,
        ),
        evaluation_path=_normalize_optional_path_string(
            payload.get("evaluation_path"),
            field_name="portfolio.evaluation_path",
            default=inherited.evaluation_path,
        ),
        optimizer_method=_normalize_optional_string(
            payload.get("optimizer_method"),
            field_name="portfolio.optimizer_method",
            default=inherited.optimizer_method,
        ),
    )


def _resolve_review(
    payload: Any,
    *,
    base: ReviewConfig,
    shared_dataset: DatasetSelectionConfig,
    shared_targets: CampaignTargetsConfig,
    outputs: CampaignOutputsConfig,
) -> ReviewConfig:
    inherited_payload = {
        "filters": {
            "dataset": shared_dataset.dataset,
            "timeframe": shared_dataset.timeframe,
            "alpha_name": _single_or_none(shared_targets.alpha_names),
            "strategy_name": _single_or_none(shared_targets.strategy_names),
            "portfolio_name": _single_or_none(shared_targets.portfolio_names),
        },
        "output": {
            "path": outputs.review_output_path,
        },
    }
    try:
        resolved = ReviewConfig.from_mapping(inherited_payload, base=base)
        if payload is None:
            return resolved
        return ReviewConfig.from_mapping(payload, base=resolved)
    except ReviewConfigError as exc:
        raise ResearchCampaignConfigError(str(exc)) from exc


def _resolve_milestone_reporting(
    payload: Any,
    *,
    base: CampaignMilestoneReportingConfig,
) -> CampaignMilestoneReportingConfig:
    if payload is None:
        return base
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError("Research campaign field 'milestone_reporting' must be a mapping.")
    unknown_keys = sorted(set(payload) - _MILESTONE_REPORTING_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign field 'milestone_reporting' contains unsupported keys: {unknown_keys}."
        )
    return CampaignMilestoneReportingConfig(
        enabled=_resolve_bool(
            payload.get("enabled"),
            field_name="milestone_reporting.enabled",
            default=base.enabled,
        ),
        decision_categories=_normalize_string_collection(
            payload.get("decision_categories"),
            field_name="milestone_reporting.decision_categories",
            default=base.decision_categories,
        ),
        output=_resolve_milestone_reporting_output(
            payload.get("output"),
            base=base.output,
        ),
        sections=_resolve_milestone_reporting_sections(
            payload.get("sections"),
            base=base.sections,
        ),
        summary=_resolve_milestone_reporting_summary(
            payload.get("summary"),
            base=base.summary,
        ),
    )


def _resolve_milestone_reporting_output(
    payload: Any,
    *,
    base: CampaignMilestoneReportingOutputConfig,
) -> CampaignMilestoneReportingOutputConfig:
    if payload is None:
        return base
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError("Research campaign field 'milestone_reporting.output' must be a mapping.")
    unknown_keys = sorted(set(payload) - _MILESTONE_REPORTING_OUTPUT_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign field 'milestone_reporting.output' contains unsupported keys: {unknown_keys}."
        )
    render_formats = payload.get("decision_log_render_formats")
    if render_formats is None:
        resolved_formats = base.decision_log_render_formats
    else:
        resolved_formats = _normalize_string_collection(
            render_formats,
            field_name="milestone_reporting.output.decision_log_render_formats",
            default=base.decision_log_render_formats,
        )
        invalid = [value for value in resolved_formats if value not in _SUPPORTED_DECISION_LOG_RENDER_FORMATS]
        if invalid:
            raise ResearchCampaignConfigError(
                "Research campaign field 'milestone_reporting.output.decision_log_render_formats' "
                f"must contain only {sorted(_SUPPORTED_DECISION_LOG_RENDER_FORMATS)}."
            )
    return CampaignMilestoneReportingOutputConfig(
        include_markdown_report=_resolve_bool(
            payload.get("include_markdown_report"),
            field_name="milestone_reporting.output.include_markdown_report",
            default=base.include_markdown_report,
        ),
        decision_log_render_formats=resolved_formats,
    )


def _resolve_milestone_reporting_sections(
    payload: Any,
    *,
    base: CampaignMilestoneReportingSectionsConfig,
) -> CampaignMilestoneReportingSectionsConfig:
    if payload is None:
        return base
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError("Research campaign field 'milestone_reporting.sections' must be a mapping.")
    unknown_keys = sorted(set(payload) - _MILESTONE_REPORTING_SECTION_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign field 'milestone_reporting.sections' contains unsupported keys: {unknown_keys}."
        )
    return CampaignMilestoneReportingSectionsConfig(
        campaign_scope=_resolve_bool(
            payload.get("campaign_scope"),
            field_name="milestone_reporting.sections.campaign_scope",
            default=base.campaign_scope,
        ),
        selections=_resolve_bool(
            payload.get("selections"),
            field_name="milestone_reporting.sections.selections",
            default=base.selections,
        ),
        key_findings=_resolve_bool(
            payload.get("key_findings"),
            field_name="milestone_reporting.sections.key_findings",
            default=base.key_findings,
        ),
        key_metrics=_resolve_bool(
            payload.get("key_metrics"),
            field_name="milestone_reporting.sections.key_metrics",
            default=base.key_metrics,
        ),
        gate_outcomes=_resolve_bool(
            payload.get("gate_outcomes"),
            field_name="milestone_reporting.sections.gate_outcomes",
            default=base.gate_outcomes,
        ),
        risks=_resolve_bool(
            payload.get("risks"),
            field_name="milestone_reporting.sections.risks",
            default=base.risks,
        ),
        next_steps=_resolve_bool(
            payload.get("next_steps"),
            field_name="milestone_reporting.sections.next_steps",
            default=base.next_steps,
        ),
        open_questions=_resolve_bool(
            payload.get("open_questions"),
            field_name="milestone_reporting.sections.open_questions",
            default=base.open_questions,
        ),
        decision_snapshot=_resolve_bool(
            payload.get("decision_snapshot"),
            field_name="milestone_reporting.sections.decision_snapshot",
            default=base.decision_snapshot,
        ),
        related_artifacts=_resolve_bool(
            payload.get("related_artifacts"),
            field_name="milestone_reporting.sections.related_artifacts",
            default=base.related_artifacts,
        ),
    )


def _resolve_milestone_reporting_summary(
    payload: Any,
    *,
    base: CampaignMilestoneReportingSummaryConfig,
) -> CampaignMilestoneReportingSummaryConfig:
    if payload is None:
        return base
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError("Research campaign field 'milestone_reporting.summary' must be a mapping.")
    unknown_keys = sorted(set(payload) - _MILESTONE_REPORTING_SUMMARY_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign field 'milestone_reporting.summary' contains unsupported keys: {unknown_keys}."
        )
    return CampaignMilestoneReportingSummaryConfig(
        include_stage_counts=_resolve_bool(
            payload.get("include_stage_counts"),
            field_name="milestone_reporting.summary.include_stage_counts",
            default=base.include_stage_counts,
        ),
        include_review_outcome=_resolve_bool(
            payload.get("include_review_outcome"),
            field_name="milestone_reporting.summary.include_review_outcome",
            default=base.include_review_outcome,
        ),
    )


def _resolve_scenarios(payload: Any, *, base: CampaignScenariosConfig) -> CampaignScenariosConfig:
    if payload is None:
        return base
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError("Research campaign field 'scenarios' must be a mapping.")
    unknown_keys = sorted(set(payload) - _SCENARIOS_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign field 'scenarios' contains unsupported keys: {unknown_keys}."
        )

    enabled = _resolve_bool(
        payload.get("enabled"),
        field_name="scenarios.enabled",
        default=base.enabled,
    )
    raw_matrix = payload.get("matrix")
    matrix = base.matrix if raw_matrix is None else _resolve_scenario_matrix(raw_matrix)
    raw_include = payload.get("include")
    include = base.include if raw_include is None else _resolve_scenario_include(raw_include)
    max_scenarios = _resolve_optional_positive_int(
        payload.get("max_scenarios"),
        field_name="scenarios.max_scenarios",
        default=base.max_scenarios,
    )
    max_values_per_axis = _resolve_optional_positive_int(
        payload.get("max_values_per_axis"),
        field_name="scenarios.max_values_per_axis",
        default=base.max_values_per_axis,
    )

    if enabled and not matrix and not include:
        raise ResearchCampaignConfigError(
            "Research campaign field 'scenarios' must define at least one matrix axis or included scenario when enabled."
        )

    if max_values_per_axis is not None:
        for axis in matrix:
            if len(axis.values) > max_values_per_axis:
                raise ResearchCampaignConfigError(
                    f"scenarios.matrix axis '{axis.name}' has {len(axis.values)} values, "
                    f"which exceeds scenarios.max_values_per_axis ({max_values_per_axis}). "
                    "Reduce the number of values for this axis or raise the limit."
                )

    if enabled:
        matrix_count = 1
        for axis in matrix:
            matrix_count *= len(axis.values)
        total_count = (matrix_count if matrix else 0) + len(include)
        effective_max = (
            min(_SCENARIOS_HARD_MAX, max_scenarios)
            if max_scenarios is not None
            else _SCENARIOS_HARD_MAX
        )
        if total_count > effective_max:
            raise ResearchCampaignConfigError(
                f"scenarios would expand to {total_count} scenarios, which exceeds the "
                f"effective limit of {effective_max} "
                f"(hard_max={_SCENARIOS_HARD_MAX}, "
                f"configured_max={'none' if max_scenarios is None else max_scenarios}). "
                "Reduce the matrix dimensions, set scenarios.max_scenarios, or split "
                "into multiple campaigns."
            )

    return CampaignScenariosConfig(
        enabled=enabled,
        matrix=matrix,
        include=include,
        max_scenarios=max_scenarios,
        max_values_per_axis=max_values_per_axis,
    )


def _resolve_scenario_matrix(payload: Any) -> tuple[CampaignScenarioMatrixAxisConfig, ...]:
    if not isinstance(payload, list):
        raise ResearchCampaignConfigError("Research campaign field 'scenarios.matrix' must be a list.")

    axes: list[CampaignScenarioMatrixAxisConfig] = []
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    for index, entry in enumerate(payload):
        field_name = f"scenarios.matrix[{index}]"
        if not isinstance(entry, Mapping):
            raise ResearchCampaignConfigError(
                f"Research campaign field '{field_name}' must be a mapping."
            )
        unknown_keys = sorted(set(entry) - _SCENARIO_MATRIX_KEYS)
        if unknown_keys:
            raise ResearchCampaignConfigError(
                f"Research campaign field '{field_name}' contains unsupported keys: {unknown_keys}."
            )

        axis = CampaignScenarioMatrixAxisConfig(
            name=_normalize_required_string(entry.get("name"), field_name=f"{field_name}.name"),
            path=_normalize_scenario_override_path(
                entry.get("path"),
                field_name=f"{field_name}.path",
            ),
            values=_normalize_scenario_matrix_values(
                entry.get("values"),
                field_name=f"{field_name}.values",
            ),
        )
        if axis.name in seen_names:
            raise ResearchCampaignConfigError(
                f"Research campaign field 'scenarios.matrix' contains duplicate axis name '{axis.name}'."
            )
        if axis.path in seen_paths:
            raise ResearchCampaignConfigError(
                f"Research campaign field 'scenarios.matrix' contains duplicate override path '{axis.path}'."
            )
        seen_names.add(axis.name)
        seen_paths.add(axis.path)
        axes.append(axis)
    return tuple(axes)


def _resolve_scenario_include(payload: Any) -> tuple[CampaignScenarioIncludeConfig, ...]:
    if not isinstance(payload, list):
        raise ResearchCampaignConfigError("Research campaign field 'scenarios.include' must be a list.")

    scenarios: list[CampaignScenarioIncludeConfig] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(payload):
        field_name = f"scenarios.include[{index}]"
        if not isinstance(entry, Mapping):
            raise ResearchCampaignConfigError(
                f"Research campaign field '{field_name}' must be a mapping."
            )
        unknown_keys = sorted(set(entry) - _SCENARIO_INCLUDE_KEYS)
        if unknown_keys:
            raise ResearchCampaignConfigError(
                f"Research campaign field '{field_name}' contains unsupported keys: {unknown_keys}."
            )

        scenario_id = _normalize_scenario_id(
            entry.get("scenario_id"),
            field_name=f"{field_name}.scenario_id",
        )
        if scenario_id in seen_ids:
            raise ResearchCampaignConfigError(
                f"Research campaign field 'scenarios.include' contains duplicate scenario_id '{scenario_id}'."
            )
        seen_ids.add(scenario_id)

        raw_overrides = entry.get("overrides")
        if not isinstance(raw_overrides, Mapping):
            raise ResearchCampaignConfigError(
                f"Research campaign field '{field_name}.overrides' must be a mapping."
            )

        overrides = dict(_extract_root_container(raw_overrides))
        if "scenarios" in overrides:
            raise ResearchCampaignConfigError(
                f"Research campaign field '{field_name}.overrides' cannot override the 'scenarios' section."
            )

        scenarios.append(
            CampaignScenarioIncludeConfig(
                scenario_id=scenario_id,
                description=_normalize_optional_string(
                    entry.get("description"),
                    field_name=f"{field_name}.description",
                    default=None,
                ),
                overrides=_canonicalize_value(overrides),
            )
        )
    return tuple(scenarios)


def _resolve_outputs(payload: Any, *, base: CampaignOutputsConfig) -> CampaignOutputsConfig:
    if payload is None:
        return base
    if not isinstance(payload, Mapping):
        raise ResearchCampaignConfigError("Research campaign field 'outputs' must be a mapping.")
    unknown_keys = sorted(set(payload) - _OUTPUT_KEYS)
    if unknown_keys:
        raise ResearchCampaignConfigError(
            f"Research campaign field 'outputs' contains unsupported keys: {unknown_keys}."
        )
    return CampaignOutputsConfig(
        alpha_artifacts_root=_normalize_required_path_string(
            payload.get("alpha_artifacts_root", base.alpha_artifacts_root),
            field_name="outputs.alpha_artifacts_root",
        ),
        candidate_selection_output_path=_normalize_required_path_string(
            payload.get("candidate_selection_output_path", base.candidate_selection_output_path),
            field_name="outputs.candidate_selection_output_path",
        ),
        candidate_selection_registry_path=_normalize_optional_path_string(
            payload.get("candidate_selection_registry_path"),
            field_name="outputs.candidate_selection_registry_path",
            default=base.candidate_selection_registry_path,
        ),
        campaign_artifacts_root=_normalize_required_path_string(
            payload.get("campaign_artifacts_root", base.campaign_artifacts_root),
            field_name="outputs.campaign_artifacts_root",
        ),
        portfolio_artifacts_root=_normalize_required_path_string(
            payload.get("portfolio_artifacts_root", base.portfolio_artifacts_root),
            field_name="outputs.portfolio_artifacts_root",
        ),
        comparison_output_path=_normalize_optional_path_string(
            payload.get("comparison_output_path"),
            field_name="outputs.comparison_output_path",
            default=base.comparison_output_path,
        ),
        review_output_path=_normalize_optional_path_string(
            payload.get("review_output_path"),
            field_name="outputs.review_output_path",
            default=base.review_output_path,
        ),
    )


def _normalize_required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be a non-empty string."
        )
    normalized = value.strip()
    if not normalized:
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be a non-empty string."
        )
    return normalized


def _normalize_optional_string(value: Any, *, field_name: str, default: str | None) -> str | None:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be a string when provided."
        )
    normalized = value.strip()
    return normalized or None


def _normalize_optional_temporal_string(
    value: Any, *, field_name: str, default: str | None
) -> str | None:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        text = str(isoformat()).strip()
        if text:
            return text
    raise ResearchCampaignConfigError(
        f"Research campaign field '{field_name}' must be a string when provided."
    )


def _normalize_required_path_string(value: Any, *, field_name: str) -> str:
    return str(Path(_normalize_required_string(value, field_name=field_name))).replace("\\", "/")


def _normalize_scenario_id(value: Any, *, field_name: str) -> str:
    scenario_id = _normalize_required_string(value, field_name=field_name)
    normalized = _slugify_identifier(scenario_id)
    if not normalized:
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must contain at least one alphanumeric character."
        )
    return normalized


def _normalize_scenario_override_path(value: Any, *, field_name: str) -> str:
    path = _normalize_required_string(value, field_name=field_name)
    parts = [part.strip() for part in path.split(".")]
    if any(not part for part in parts):
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be a dotted path like 'comparison.top_k'."
        )
    if any(part == "scenarios" for part in parts):
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' cannot target the 'scenarios' section."
        )
    normalized_path = ".".join(parts)
    if normalized_path not in _supported_scenario_override_paths():
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must target a supported scalar campaign field."
        )
    return normalized_path


def _normalize_scenario_matrix_values(value: Any, *, field_name: str) -> tuple[Any, ...]:
    if not isinstance(value, list) or not value:
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be a non-empty list."
        )

    normalized: list[Any] = []
    seen_signatures: set[str] = set()
    for item in value:
        normalized_item = _normalize_scenario_matrix_value(item, field_name=field_name)
        signature = json.dumps(_canonicalize_value(normalized_item), sort_keys=True, separators=(",", ":"))
        if signature in seen_signatures:
            raise ResearchCampaignConfigError(
                f"Research campaign field '{field_name}' contains duplicate values."
            )
        seen_signatures.add(signature)
        normalized.append(normalized_item)
    return tuple(normalized)


def _normalize_scenario_matrix_value(value: Any, *, field_name: str) -> Any:
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ResearchCampaignConfigError(
                f"Research campaign field '{field_name}' must not contain empty strings."
            )
        return normalized
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value
    raise ResearchCampaignConfigError(
        f"Research campaign field '{field_name}' must contain only scalar string, boolean, or numeric values."
    )


def _normalize_optional_path_string(
    value: Any, *, field_name: str, default: str | None
) -> str | None:
    text = _normalize_optional_string(value, field_name=field_name, default=default)
    if text is None:
        return None
    return str(Path(text)).replace("\\", "/")


def _normalize_string_collection(
    value: Any, *, field_name: str, default: tuple[str, ...]
) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = value
    else:
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be a string or list of strings."
        )

    normalized: list[str] = []
    for item in raw_items:
        if not isinstance(item, str):
            raise ResearchCampaignConfigError(
                f"Research campaign field '{field_name}' must contain only strings."
            )
        text = item.strip()
        if not text:
            raise ResearchCampaignConfigError(
                f"Research campaign field '{field_name}' must not contain empty strings."
            )
        if text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _normalize_stage_name_collection(
    value: Any, *, field_name: str, default: tuple[str, ...]
) -> tuple[str, ...]:
    items = _normalize_string_collection(value, field_name=field_name, default=default)
    invalid = [item for item in items if item not in _CAMPAIGN_STAGE_NAMES]
    if invalid:
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must contain only supported stage names: {list(_CAMPAIGN_STAGE_NAMES)}."
        )
    return items


def _resolve_bool(value: Any, *, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be a boolean."
        )
    return value


def _resolve_optional_positive_int(
    value: Any, *, field_name: str, default: int | None
) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be a positive integer when provided."
        )
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be a positive integer when provided."
        ) from exc
    if integer <= 0:
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be a positive integer when provided."
        )
    return integer


def _resolve_non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be a non-negative integer."
        )
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be a non-negative integer."
        ) from exc
    if integer < 0:
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be a non-negative integer."
        )
    return integer


def _resolve_optional_float(value: Any, *, field_name: str, default: float | None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be a number when provided."
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be a number when provided."
        ) from exc


def _resolve_non_negative_float(value: Any, *, field_name: str) -> float:
    numeric = _resolve_optional_float(value, field_name=field_name, default=None)
    if numeric is None or numeric < 0.0:
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be a non-negative number."
        )
    return numeric


def _resolve_optional_probability(
    value: Any, *, field_name: str, default: float | None
) -> float | None:
    numeric = _resolve_optional_float(value, field_name=field_name, default=default)
    if numeric is None:
        return None
    if not (0.0 <= numeric <= 1.0):
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be within [0, 1] when provided."
        )
    return numeric


def _resolve_choice(value: Any, *, field_name: str, supported_values: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be one of {sorted(supported_values)}."
        )
    normalized = value.strip().lower()
    if normalized not in supported_values:
        raise ResearchCampaignConfigError(
            f"Research campaign field '{field_name}' must be one of {sorted(supported_values)}."
        )
    return normalized


def _single_or_none(values: tuple[str, ...]) -> str | None:
    return values[0] if len(values) == 1 else None


def compute_scenario_expansion_size(config: ResearchCampaignConfig) -> dict[str, Any]:
    """Return a dictionary describing the scenario expansion dimensions and limits.

    This is a lightweight read-only computation that does not materialize full
    resolved scenario configs.  Use it in preflight checks and operator tooling
    to inspect expansion size without triggering the full expansion.

    Returned keys:
      matrix_axis_count         - number of declared sweep axes
      per_axis_value_counts     - list of value counts per axis in declaration order
      matrix_combination_count  - cartesian product of per-axis value counts (0 when no matrix)
      include_count             - number of explicit included scenarios
      total_scenario_count      - matrix_combination_count + include_count
      effective_max_scenarios   - the operative limit (min of hard cap and configured cap)
      hard_max_scenarios        - the hard ceiling (_SCENARIOS_HARD_MAX)
      configured_max_scenarios  - scenarios.max_scenarios value (None when unconfigured)
      max_values_per_axis       - scenarios.max_values_per_axis value (None when unconfigured)
      exceeds_limit             - True when total_scenario_count > effective_max_scenarios
    """
    sc = config.scenarios
    per_axis_counts = [len(axis.values) for axis in sc.matrix]
    matrix_count = 1
    for n in per_axis_counts:
        matrix_count *= n
    if not sc.matrix:
        matrix_count = 0
    include_count = len(sc.include)
    total_count = matrix_count + include_count
    effective_max = (
        min(_SCENARIOS_HARD_MAX, sc.max_scenarios)
        if sc.max_scenarios is not None
        else _SCENARIOS_HARD_MAX
    )
    return {
        "matrix_axis_count": len(sc.matrix),
        "per_axis_value_counts": per_axis_counts,
        "matrix_combination_count": matrix_count,
        "include_count": include_count,
        "total_scenario_count": total_count,
        "effective_max_scenarios": effective_max,
        "hard_max_scenarios": _SCENARIOS_HARD_MAX,
        "configured_max_scenarios": sc.max_scenarios,
        "max_values_per_axis": sc.max_values_per_axis,
        "exceeds_limit": total_count > effective_max,
    }


def expand_research_campaign_scenarios(
    config: ResearchCampaignConfig,
) -> tuple[ResolvedResearchCampaignScenario, ...]:
    base_config = ResearchCampaignConfig.from_mapping(_config_payload_without_scenarios(config))
    scenarios_config = config.scenarios
    if not scenarios_config.enabled:
        return (
            ResolvedResearchCampaignScenario(
                scenario_id="default",
                description="Implicit single-scenario campaign configuration.",
                source="default",
                sweep_values={},
                config=base_config,
                fingerprint=_scenario_fingerprint(base_config),
            ),
        )

    # Safety-net: enforce limits even when config was constructed programmatically
    # (normal resolution via _resolve_scenarios already validated these, so this
    # guard only fires for configs created outside the standard resolver).
    size = compute_scenario_expansion_size(config)
    if size["exceeds_limit"]:
        raise ResearchCampaignConfigError(
            f"scenarios would expand to {size['total_scenario_count']} scenarios, which exceeds the "
            f"effective limit of {size['effective_max_scenarios']} "
            f"(hard_max={size['hard_max_scenarios']}, "
            f"configured_max={'none' if size['configured_max_scenarios'] is None else size['configured_max_scenarios']}). "
            "Reduce the matrix dimensions, set scenarios.max_scenarios, or split "
            "into multiple campaigns."
        )

    resolved: list[ResolvedResearchCampaignScenario] = []
    seen_ids: set[str] = set()
    matrix_overrides = _expand_scenario_matrix_overrides(scenarios_config.matrix)
    for index, (sweep_values, override_payload) in enumerate(matrix_overrides):
        scenario_id = _build_matrix_scenario_id(index, sweep_values)
        if scenario_id in seen_ids:
            raise ResearchCampaignConfigError(
                f"Research campaign scenarios resolve duplicate scenario_id '{scenario_id}'."
            )
        seen_ids.add(scenario_id)
        scenario_config = ResearchCampaignConfig.from_mapping(override_payload, base=base_config)
        resolved.append(
            ResolvedResearchCampaignScenario(
                scenario_id=scenario_id,
                description=None,
                source="matrix",
                sweep_values=dict(sweep_values),
                config=scenario_config,
                fingerprint=_scenario_fingerprint(scenario_config),
            )
        )

    for included in scenarios_config.include:
        if included.scenario_id in seen_ids:
            raise ResearchCampaignConfigError(
                f"Research campaign scenarios resolve duplicate scenario_id '{included.scenario_id}'."
            )
        seen_ids.add(included.scenario_id)
        scenario_config = ResearchCampaignConfig.from_mapping(included.overrides, base=base_config)
        resolved.append(
            ResolvedResearchCampaignScenario(
                scenario_id=included.scenario_id,
                description=included.description,
                source="include",
                sweep_values={},
                config=scenario_config,
                fingerprint=_scenario_fingerprint(scenario_config),
            )
        )
    return tuple(resolved)


def build_research_campaign_scenario_catalog(config: ResearchCampaignConfig) -> dict[str, Any]:
    base_config = ResearchCampaignConfig.from_mapping(_config_payload_without_scenarios(config))
    scenarios = expand_research_campaign_scenarios(config)
    expansion_size = compute_scenario_expansion_size(config)
    return _canonicalize_value(
        {
            "scenario_count": len(scenarios),
            "base_fingerprint": _scenario_fingerprint(base_config),
            "scenarios_enabled": config.scenarios.enabled,
            "expansion": expansion_size,
            "scenarios": [scenario.to_dict() for scenario in scenarios],
        }
    )


def _config_payload_without_scenarios(config: ResearchCampaignConfig) -> dict[str, Any]:
    payload = dict(config.to_dict())
    payload.pop("scenarios", None)
    return payload


def _expand_scenario_matrix_overrides(
    axes: tuple[CampaignScenarioMatrixAxisConfig, ...],
) -> tuple[tuple[dict[str, Any], dict[str, Any]], ...]:
    if not axes:
        return ()

    combinations: list[tuple[dict[str, Any], dict[str, Any]]] = [({}, {})]
    for axis in axes:
        next_combinations: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for sweep_values, override_payload in combinations:
            for value in axis.values:
                next_sweep_values = {**sweep_values, axis.name: value}
                next_override_payload = _merge_nested_mappings(
                    override_payload,
                    _nested_mapping_for_path(axis.path, value),
                )
                next_combinations.append((next_sweep_values, next_override_payload))
        combinations = next_combinations
    return tuple(combinations)


def _nested_mapping_for_path(path: str, value: Any) -> dict[str, Any]:
    nested: Any = value
    for key in reversed(path.split(".")):
        nested = {key: nested}
    return nested


def _merge_nested_mappings(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_nested_mappings(merged[key], value)
        elif isinstance(value, Mapping):
            merged[key] = _merge_nested_mappings({}, value)
        else:
            merged[key] = value
    return merged


def _build_matrix_scenario_id(index: int, sweep_values: Mapping[str, Any]) -> str:
    parts = [
        f"{_slugify_identifier(name)}_{_slugify_identifier(_scenario_value_label(value))}"
        for name, value in sweep_values.items()
    ]
    suffix = "__".join(part for part in parts if part)
    if suffix:
        return f"scenario_{index:04d}_{suffix}"
    return f"scenario_{index:04d}"


def _scenario_value_label(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return str(value)


def _slugify_identifier(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "_" for character in value.strip())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _scenario_fingerprint(config: ResearchCampaignConfig) -> str:
    payload = json.dumps(
        _config_payload_without_scenarios(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _supported_scenario_override_paths() -> frozenset[str]:
    payload = _config_payload_without_scenarios(ResearchCampaignConfig.default())
    supported: set[str] = set()
    _collect_scalar_paths(payload, prefix=(), output=supported)
    return frozenset(supported)


def _collect_scalar_paths(value: Any, *, prefix: tuple[str, ...], output: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _collect_scalar_paths(child, prefix=(*prefix, key), output=output)
        return
    if isinstance(value, list):
        return
    if prefix:
        output.add(".".join(prefix))


def _canonicalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _canonicalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize_value(item) for item in value]
    return value


__all__ = [
    "RESEARCH_CAMPAIGN_CONFIG",
    "ResolvedResearchCampaignScenario",
    "ResearchCampaignConfig",
    "ResearchCampaignConfigError",
    "build_research_campaign_scenario_catalog",
    "compute_scenario_expansion_size",
    "expand_research_campaign_scenarios",
    "load_research_campaign_config",
    "resolve_research_campaign_config",
]
