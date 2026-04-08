from __future__ import annotations

import json
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
        "comparison",
        "candidate_selection",
        "portfolio",
        "review",
        "outputs",
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
_SUPPORTED_ALPHA_VIEWS = frozenset({"forecast", "sleeve", "combined"})
_DEFAULT_CANDIDATE_SELECTION_OUTPUT_PATH = "artifacts/candidate_selection"


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
class ResearchCampaignConfig:
    dataset_selection: DatasetSelectionConfig
    time_windows: TimeWindowsConfig
    targets: CampaignTargetsConfig
    comparison: CampaignComparisonConfig
    candidate_selection: CampaignCandidateSelectionConfig
    portfolio: CampaignPortfolioConfig
    review: ReviewConfig
    outputs: CampaignOutputsConfig

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
            outputs=outputs,
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
        return cls(
            dataset_selection=dataset_selection,
            time_windows=time_windows,
            targets=targets,
            comparison=comparison,
            candidate_selection=candidate_selection,
            portfolio=portfolio,
            review=review,
            outputs=outputs,
        )

    def to_dict(self) -> dict[str, Any]:
        return _canonicalize_value(
            {
                "dataset_selection": self.dataset_selection.to_dict(),
                "time_windows": self.time_windows.to_dict(),
                "targets": self.targets.to_dict(),
                "comparison": self.comparison.to_dict(),
                "candidate_selection": self.candidate_selection.to_dict(),
                "portfolio": self.portfolio.to_dict(),
                "review": self.review.to_dict(),
                "outputs": self.outputs.to_dict(),
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


def _canonicalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _canonicalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize_value(item) for item in value]
    return value


__all__ = [
    "RESEARCH_CAMPAIGN_CONFIG",
    "ResearchCampaignConfig",
    "ResearchCampaignConfigError",
    "load_research_campaign_config",
    "resolve_research_campaign_config",
]
