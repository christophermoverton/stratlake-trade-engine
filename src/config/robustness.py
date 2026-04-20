from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.config.settings import load_yaml_config

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBUSTNESS_CONFIG = REPO_ROOT / "configs" / "robustness.yml"
SUPPORTED_STABILITY_MODES = {"disabled", "subperiods", "walk_forward"}
SUPPORTED_VALIDITY_RANKING_METHODS = {"adjusted_q_value", "deflated_sharpe_ratio", "none"}


def _require_non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _normalize_optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_string(value, field_name=field_name)


def _normalize_value_list(value: Any, *, field_name: str) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{field_name} must not be empty when provided.")
        return tuple(value)
    return (value,)


@dataclass(frozen=True)
class MetricThreshold:
    """Deterministic threshold rule used in robustness pass-rate summaries."""

    metric: str
    min_value: float | None = None
    max_value: float | None = None

    @classmethod
    def from_mapping(cls, metric: str, payload: Any) -> "MetricThreshold":
        normalized_metric = _require_non_empty_string(metric, field_name="metric")
        if isinstance(payload, int | float):
            return cls(metric=normalized_metric, min_value=float(payload))
        if not isinstance(payload, dict):
            raise ValueError(
                f"Robustness threshold for '{normalized_metric}' must be a number or a mapping with min/max."
            )

        min_value = payload.get("min")
        max_value = payload.get("max")
        if min_value is None and max_value is None:
            raise ValueError(
                f"Robustness threshold for '{normalized_metric}' must define at least one of 'min' or 'max'."
            )
        if min_value is not None and not isinstance(min_value, int | float):
            raise ValueError(f"Robustness threshold min for '{normalized_metric}' must be numeric.")
        if max_value is not None and not isinstance(max_value, int | float):
            raise ValueError(f"Robustness threshold max for '{normalized_metric}' must be numeric.")
        return cls(
            metric=normalized_metric,
            min_value=None if min_value is None else float(min_value),
            max_value=None if max_value is None else float(max_value),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "min": self.min_value,
            "max": self.max_value,
        }


@dataclass(frozen=True)
class SweepDefinition:
    """Legacy one-parameter sweep definition with explicit ordered values."""

    parameter: str
    values: tuple[Any, ...]

    @classmethod
    def from_mapping(cls, payload: Any) -> "SweepDefinition":
        if not isinstance(payload, dict):
            raise ValueError("Each robustness sweep entry must be a mapping.")

        parameter = _require_non_empty_string(payload.get("parameter"), field_name="sweep.parameter")
        values = payload.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Robustness sweep '{parameter}' must define a non-empty values list.")
        return cls(parameter=parameter, values=tuple(values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "values": list(self.values),
        }


@dataclass(frozen=True)
class StabilityConfig:
    """Optional time-slice stability analysis configuration."""

    mode: str = "disabled"
    periods: int | None = None
    evaluation_path: str | None = None

    @classmethod
    def from_mapping(cls, payload: Any) -> "StabilityConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("Robustness stability configuration must be a mapping.")

        mode = payload.get("mode", "disabled")
        if not isinstance(mode, str) or mode not in SUPPORTED_STABILITY_MODES:
            supported = ", ".join(sorted(SUPPORTED_STABILITY_MODES))
            raise ValueError(f"Robustness stability mode must be one of: {supported}.")

        periods = payload.get("periods")
        if periods is not None:
            if not isinstance(periods, int):
                raise ValueError("Robustness stability periods must be an integer when provided.")
            if periods < 2:
                raise ValueError("Robustness stability periods must be at least 2.")

        evaluation_path = payload.get("evaluation_path")
        if evaluation_path is not None and not isinstance(evaluation_path, str):
            raise ValueError("Robustness stability evaluation_path must be a string when provided.")

        if mode == "subperiods" and periods is None:
            raise ValueError("Robustness subperiod stability requires a periods value.")

        return cls(mode=mode, periods=periods, evaluation_path=evaluation_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "periods": self.periods,
            "evaluation_path": self.evaluation_path,
        }


@dataclass(frozen=True)
class SweepMatchRule:
    """Deterministic exact-match filter for excluding invalid combinations."""

    match: dict[str, Any]
    reason: str | None = None

    @classmethod
    def from_mapping(cls, payload: Any) -> "SweepMatchRule":
        if not isinstance(payload, dict):
            raise ValueError("Sweep filters must be mappings.")
        raw_match = payload.get("match", payload.get("where", payload))
        if not isinstance(raw_match, Mapping) or not raw_match:
            raise ValueError("Sweep filters must define a non-empty match mapping.")
        reason = payload.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("Sweep filter reason must be a string when provided.")
        return cls(match={str(key): raw_match[key] for key in sorted(raw_match)}, reason=reason)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"match": dict(self.match)}
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True)
class SweepBatchingConfig:
    """Deterministic batching controls for large sweep spaces."""

    batch_size: int | None = None
    batch_index: int = 0

    @classmethod
    def from_mapping(cls, payload: Any) -> "SweepBatchingConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("Robustness batching must be a mapping when provided.")
        batch_size = payload.get("batch_size")
        if batch_size is not None:
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("Robustness batching batch_size must be a positive integer.")
        batch_index = payload.get("batch_index", 0)
        if not isinstance(batch_index, int) or batch_index < 0:
            raise ValueError("Robustness batching batch_index must be a non-negative integer.")
        return cls(batch_size=batch_size, batch_index=batch_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "batch_index": self.batch_index,
        }


@dataclass(frozen=True)
class RankingConfig:
    """Configurable deterministic ranking policy."""

    primary_metric: str = "sharpe_ratio"
    higher_is_better: bool | None = None
    tie_breakers: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        payload: Any,
        *,
        fallback_metric: str = "sharpe_ratio",
        fallback_higher_is_better: bool | None = None,
    ) -> "RankingConfig":
        if payload is None:
            return cls(
                primary_metric=fallback_metric,
                higher_is_better=fallback_higher_is_better,
            )
        if not isinstance(payload, Mapping):
            raise ValueError("Robustness ranking configuration must be a mapping.")
        primary_metric = _require_non_empty_string(
            payload.get("primary_metric", fallback_metric),
            field_name="ranking.primary_metric",
        )
        higher_is_better = payload.get("higher_is_better", fallback_higher_is_better)
        if higher_is_better is not None and not isinstance(higher_is_better, bool):
            raise ValueError("Robustness ranking higher_is_better must be a boolean when provided.")
        raw_tie_breakers = payload.get("tie_breakers", [])
        if not isinstance(raw_tie_breakers, list):
            raise ValueError("Robustness ranking tie_breakers must be a list when provided.")
        tie_breakers = tuple(
            _require_non_empty_string(value, field_name="ranking.tie_breakers[]")
            for value in raw_tie_breakers
        )
        return cls(
            primary_metric=primary_metric,
            higher_is_better=higher_is_better,
            tie_breakers=tie_breakers,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_metric": self.primary_metric,
            "higher_is_better": self.higher_is_better,
            "tie_breakers": list(self.tie_breakers),
        }


@dataclass(frozen=True)
class StatisticalControlsConfig:
    """Research-control scaffolding for large search spaces."""

    deflated_sharpe_ratio: bool = False
    multiple_testing_awareness: bool = True
    overfitting_warning_search_space: int = 100
    primary_metric: str | None = None
    validity_ranking_method: str = "adjusted_q_value"
    fdr_alpha: float = 0.10
    min_splits_for_inference: int = 4
    low_neighbor_gap_threshold: float = 0.05
    low_threshold_pass_rate_threshold: float = 0.50
    rank_instability_correlation_threshold: float = 0.50
    dsr_min_observations: int = 30
    reality_check_placeholder: bool = False

    @classmethod
    def from_mapping(cls, payload: Any) -> "StatisticalControlsConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("Robustness statistical_controls must be a mapping when provided.")
        overfitting_warning_search_space = payload.get("overfitting_warning_search_space", 100)
        if not isinstance(overfitting_warning_search_space, int) or overfitting_warning_search_space <= 0:
            raise ValueError(
                "Robustness statistical_controls overfitting_warning_search_space must be a positive integer."
            )
        primary_metric = _normalize_optional_string(
            payload.get("primary_metric"),
            field_name="statistical_controls.primary_metric",
        )
        validity_ranking_method = payload.get("validity_ranking_method", "adjusted_q_value")
        if (
            not isinstance(validity_ranking_method, str)
            or validity_ranking_method not in SUPPORTED_VALIDITY_RANKING_METHODS
        ):
            supported = ", ".join(sorted(SUPPORTED_VALIDITY_RANKING_METHODS))
            raise ValueError(
                "Robustness statistical_controls validity_ranking_method must be one of: "
                f"{supported}."
            )
        fdr_alpha = payload.get("fdr_alpha", 0.10)
        if not isinstance(fdr_alpha, int | float) or not 0.0 < float(fdr_alpha) <= 1.0:
            raise ValueError("Robustness statistical_controls fdr_alpha must be in the interval (0, 1].")
        min_splits_for_inference = payload.get("min_splits_for_inference", 4)
        if not isinstance(min_splits_for_inference, int) or min_splits_for_inference < 2:
            raise ValueError(
                "Robustness statistical_controls min_splits_for_inference must be an integer >= 2."
            )
        low_neighbor_gap_threshold = payload.get("low_neighbor_gap_threshold", 0.05)
        if not isinstance(low_neighbor_gap_threshold, int | float) or float(low_neighbor_gap_threshold) < 0.0:
            raise ValueError(
                "Robustness statistical_controls low_neighbor_gap_threshold must be a non-negative number."
            )
        low_threshold_pass_rate_threshold = payload.get("low_threshold_pass_rate_threshold", 0.50)
        if (
            not isinstance(low_threshold_pass_rate_threshold, int | float)
            or not 0.0 <= float(low_threshold_pass_rate_threshold) <= 1.0
        ):
            raise ValueError(
                "Robustness statistical_controls low_threshold_pass_rate_threshold must be in [0, 1]."
            )
        rank_instability_correlation_threshold = payload.get(
            "rank_instability_correlation_threshold",
            0.50,
        )
        if (
            not isinstance(rank_instability_correlation_threshold, int | float)
            or not -1.0 <= float(rank_instability_correlation_threshold) <= 1.0
        ):
            raise ValueError(
                "Robustness statistical_controls rank_instability_correlation_threshold must be in [-1, 1]."
            )
        dsr_min_observations = payload.get("dsr_min_observations", 30)
        if not isinstance(dsr_min_observations, int) or dsr_min_observations < 2:
            raise ValueError(
                "Robustness statistical_controls dsr_min_observations must be an integer >= 2."
            )
        return cls(
            deflated_sharpe_ratio=bool(payload.get("deflated_sharpe_ratio", False)),
            multiple_testing_awareness=bool(payload.get("multiple_testing_awareness", True)),
            overfitting_warning_search_space=overfitting_warning_search_space,
            primary_metric=primary_metric,
            validity_ranking_method=validity_ranking_method,
            fdr_alpha=float(fdr_alpha),
            min_splits_for_inference=min_splits_for_inference,
            low_neighbor_gap_threshold=float(low_neighbor_gap_threshold),
            low_threshold_pass_rate_threshold=float(low_threshold_pass_rate_threshold),
            rank_instability_correlation_threshold=float(rank_instability_correlation_threshold),
            dsr_min_observations=dsr_min_observations,
            reality_check_placeholder=bool(payload.get("reality_check_placeholder", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "deflated_sharpe_ratio": self.deflated_sharpe_ratio,
            "multiple_testing_awareness": self.multiple_testing_awareness,
            "overfitting_warning_search_space": self.overfitting_warning_search_space,
            "primary_metric": self.primary_metric,
            "validity_ranking_method": self.validity_ranking_method,
            "fdr_alpha": self.fdr_alpha,
            "min_splits_for_inference": self.min_splits_for_inference,
            "low_neighbor_gap_threshold": self.low_neighbor_gap_threshold,
            "low_threshold_pass_rate_threshold": self.low_threshold_pass_rate_threshold,
            "rank_instability_correlation_threshold": self.rank_instability_correlation_threshold,
            "dsr_min_observations": self.dsr_min_observations,
            "reality_check_placeholder": self.reality_check_placeholder,
        }


@dataclass(frozen=True)
class SweepLayerConfig:
    """Generic deterministic sweep layer for strategy/signal/constructor/asymmetry."""

    names: tuple[str, ...] = ()
    version: str | None = None
    params: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    options: dict[str, tuple[Any, ...]] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        payload: Any,
        *,
        name_field: str | tuple[str, ...] = "name",
        version_field: str = "version",
    ) -> "SweepLayerConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("Sweep layer definitions must be mappings.")

        candidate_name_fields = (name_field,) if isinstance(name_field, str) else name_field
        raw_names = None
        for candidate in candidate_name_fields:
            if candidate in payload:
                raw_names = payload.get(candidate)
                break
        names = tuple(
            _require_non_empty_string(value, field_name="sweep layer name")
            for value in _normalize_value_list(raw_names, field_name="layer.names")
        )
        version = _normalize_optional_string(payload.get(version_field), field_name=f"layer.{version_field}")

        raw_params = payload.get("params", {})
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, Mapping):
            raise ValueError("Sweep layer params must be a mapping when provided.")
        params = {
            str(key): _normalize_value_list(raw_params[key], field_name=f"params.{key}")
            for key in sorted(raw_params)
        }

        excluded = {"params", version_field, *candidate_name_fields}
        options = {
            str(key): _normalize_value_list(payload[key], field_name=f"options.{key}")
            for key in sorted(payload)
            if key not in excluded
        }
        return cls(names=names, version=version, params=params, options=options)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.names:
            payload["name"] = list(self.names)
        if self.version is not None:
            payload["version"] = self.version
        if self.params:
            payload["params"] = {key: list(self.params[key]) for key in sorted(self.params)}
        for key in sorted(self.options):
            payload[key] = list(self.options[key])
        return payload


@dataclass(frozen=True)
class SignalSweepConfig:
    """Explicit signal layer configuration."""

    types: tuple[str, ...] = ()
    version: str | None = None
    params: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    options: dict[str, tuple[Any, ...]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Any) -> "SignalSweepConfig":
        layer = SweepLayerConfig.from_mapping(payload, name_field=("type", "signal_type"))
        return cls(
            types=layer.names,
            version=layer.version,
            params=layer.params,
            options=layer.options,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.types:
            payload["type"] = list(self.types)
        if self.version is not None:
            payload["version"] = self.version
        if self.params:
            payload["params"] = {key: list(self.params[key]) for key in sorted(self.params)}
        for key in sorted(self.options):
            payload[key] = list(self.options[key])
        return payload


@dataclass(frozen=True)
class EnsembleDefinitionConfig:
    """One declarative ensemble definition."""

    name: str
    members: tuple[str, ...]
    weighting_method: str = "equal_weight"
    weights: dict[str, float] = field(default_factory=dict)
    selection_rule: str | None = None

    @classmethod
    def from_mapping(cls, payload: Any, *, index: int) -> "EnsembleDefinitionConfig":
        if not isinstance(payload, Mapping):
            raise ValueError("Ensemble definitions must be mappings.")
        name = _require_non_empty_string(
            payload.get("name", f"ensemble_{index:04d}"),
            field_name="ensemble.name",
        )
        raw_members = payload.get("members")
        if not isinstance(raw_members, list) or len(raw_members) < 2:
            raise ValueError("Ensemble definitions must declare at least two members.")
        members = tuple(
            _require_non_empty_string(member, field_name="ensemble.members[]")
            for member in raw_members
        )
        weighting_method = _require_non_empty_string(
            payload.get("weighting_method", "equal_weight"),
            field_name="ensemble.weighting_method",
        )
        raw_weights = payload.get("weights", {})
        if raw_weights is None:
            raw_weights = {}
        if not isinstance(raw_weights, Mapping):
            raise ValueError("Ensemble weights must be a mapping when provided.")
        weights: dict[str, float] = {}
        for key in sorted(raw_weights):
            if not isinstance(raw_weights[key], int | float):
                raise ValueError("Ensemble weights must be numeric.")
            weights[str(key)] = float(raw_weights[key])
        selection_rule = _normalize_optional_string(payload.get("selection_rule"), field_name="ensemble.selection_rule")
        return cls(
            name=name,
            members=members,
            weighting_method=weighting_method,
            weights=weights,
            selection_rule=selection_rule,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "members": list(self.members),
            "weighting_method": self.weighting_method,
        }
        if self.weights:
            payload["weights"] = dict(self.weights)
        if self.selection_rule is not None:
            payload["selection_rule"] = self.selection_rule
        return payload


@dataclass(frozen=True)
class EnsembleSweepConfig:
    """First-pass bounded ensemble sweep support."""

    definitions: tuple[EnsembleDefinitionConfig, ...] = ()
    include_constituents: bool = True

    @classmethod
    def from_mapping(cls, payload: Any) -> "EnsembleSweepConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("Robustness ensemble configuration must be a mapping.")
        raw_definitions = payload.get("definitions", payload.get("configs", []))
        if raw_definitions is None:
            raw_definitions = []
        if not isinstance(raw_definitions, list):
            raise ValueError("Robustness ensemble definitions must be a list when provided.")
        definitions = tuple(
            EnsembleDefinitionConfig.from_mapping(definition, index=index)
            for index, definition in enumerate(raw_definitions)
        )
        include_constituents = payload.get("include_constituents", True)
        if not isinstance(include_constituents, bool):
            raise ValueError("Robustness ensemble include_constituents must be a boolean.")
        return cls(definitions=definitions, include_constituents=include_constituents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "definitions": [definition.to_dict() for definition in self.definitions],
            "include_constituents": self.include_constituents,
        }


@dataclass(frozen=True)
class ResearchSweepConfig:
    """Normalized multi-layer sweep definition."""

    strategy: SweepLayerConfig = field(default_factory=SweepLayerConfig)
    signal: SignalSweepConfig = field(default_factory=SignalSweepConfig)
    constructor: SweepLayerConfig = field(default_factory=SweepLayerConfig)
    asymmetry: SweepLayerConfig = field(default_factory=SweepLayerConfig)
    ensemble: EnsembleSweepConfig = field(default_factory=EnsembleSweepConfig)
    exclude: tuple[SweepMatchRule, ...] = ()
    group_by: tuple[str, ...] = ()
    batching: SweepBatchingConfig = field(default_factory=SweepBatchingConfig)

    @classmethod
    def from_mapping(cls, payload: Any) -> "ResearchSweepConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("Extended robustness sweep must be a mapping.")
        raw_exclude = payload.get("exclude", payload.get("filters", []))
        if raw_exclude is None:
            raw_exclude = []
        if not isinstance(raw_exclude, list):
            raise ValueError("Extended robustness sweep exclude filters must be a list when provided.")
        raw_group_by = payload.get("group_by", [])
        if raw_group_by is None:
            raw_group_by = []
        if not isinstance(raw_group_by, list):
            raise ValueError("Extended robustness sweep group_by must be a list when provided.")
        return cls(
            strategy=SweepLayerConfig.from_mapping(payload.get("strategy")),
            signal=SignalSweepConfig.from_mapping(payload.get("signal")),
            constructor=SweepLayerConfig.from_mapping(payload.get("constructor"), name_field=("name", "constructor_id")),
            asymmetry=SweepLayerConfig.from_mapping(payload.get("asymmetry"), name_field=("name",)),
            ensemble=EnsembleSweepConfig.from_mapping(payload.get("ensemble")),
            exclude=tuple(SweepMatchRule.from_mapping(rule) for rule in raw_exclude),
            group_by=tuple(
                _require_non_empty_string(value, field_name="sweep.group_by[]")
                for value in raw_group_by
            ),
            batching=SweepBatchingConfig.from_mapping(payload.get("batching")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.to_dict(),
            "signal": self.signal.to_dict(),
            "constructor": self.constructor.to_dict(),
            "asymmetry": self.asymmetry.to_dict(),
            "ensemble": self.ensemble.to_dict(),
            "exclude": [rule.to_dict() for rule in self.exclude],
            "group_by": list(self.group_by),
            "batching": self.batching.to_dict(),
        }

    @property
    def has_content(self) -> bool:
        return any(
            (
                self.strategy.names,
                self.strategy.params,
                self.strategy.options,
                self.signal.types,
                self.signal.params,
                self.signal.options,
                self.constructor.names,
                self.constructor.params,
                self.constructor.options,
                self.asymmetry.names,
                self.asymmetry.params,
                self.asymmetry.options,
                self.ensemble.definitions,
                self.exclude,
                self.group_by,
                self.batching.batch_size is not None,
            )
        )


@dataclass(frozen=True)
class RobustnessConfig:
    """Typed robustness-run configuration loaded from YAML."""

    strategy_name: str | None
    ranking_metric: str
    higher_is_better: bool | None
    sweep: tuple[SweepDefinition, ...]
    stability: StabilityConfig
    thresholds: tuple[MetricThreshold, ...]
    research_sweep: ResearchSweepConfig = field(default_factory=ResearchSweepConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    statistical_controls: StatisticalControlsConfig = field(default_factory=StatisticalControlsConfig)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "RobustnessConfig":
        strategy_name = payload.get("strategy_name")
        if strategy_name is not None:
            strategy_name = _require_non_empty_string(strategy_name, field_name="robustness.strategy_name")

        legacy_ranking_metric = _require_non_empty_string(
            payload.get("ranking_metric", "sharpe_ratio"),
            field_name="robustness.ranking_metric",
        )
        higher_is_better = payload.get("higher_is_better")
        if higher_is_better is not None and not isinstance(higher_is_better, bool):
            raise ValueError("Robustness higher_is_better must be a boolean when provided.")

        raw_sweep = payload.get("sweep")
        legacy_sweep: tuple[SweepDefinition, ...] = ()
        research_sweep = ResearchSweepConfig()
        if isinstance(raw_sweep, list):
            legacy_sweep = tuple(SweepDefinition.from_mapping(entry) for entry in raw_sweep)
            seen_parameters: set[str] = set()
            for definition in legacy_sweep:
                if definition.parameter in seen_parameters:
                    raise ValueError(f"Robustness sweep contains duplicate parameter '{definition.parameter}'.")
                seen_parameters.add(definition.parameter)
        elif isinstance(raw_sweep, Mapping):
            research_sweep = ResearchSweepConfig.from_mapping(raw_sweep)
        elif raw_sweep is not None:
            raise ValueError("Robustness sweep must be either a list or a mapping.")

        if not legacy_sweep and not research_sweep.has_content:
            raise ValueError("Robustness configuration must define a non-empty sweep.")

        stability = StabilityConfig.from_mapping(payload.get("stability"))

        raw_thresholds = payload.get("thresholds", {})
        if not isinstance(raw_thresholds, Mapping):
            raise ValueError("Robustness thresholds must be a mapping when provided.")
        thresholds = tuple(
            MetricThreshold.from_mapping(metric_name, raw_thresholds[metric_name])
            for metric_name in sorted(raw_thresholds)
        )

        ranking = RankingConfig.from_mapping(
            payload.get("ranking"),
            fallback_metric=legacy_ranking_metric,
            fallback_higher_is_better=higher_is_better,
        )
        statistical_controls = StatisticalControlsConfig.from_mapping(payload.get("statistical_controls"))

        return cls(
            strategy_name=strategy_name,
            ranking_metric=ranking.primary_metric,
            higher_is_better=ranking.higher_is_better,
            sweep=legacy_sweep,
            stability=stability,
            thresholds=thresholds,
            research_sweep=research_sweep,
            ranking=ranking,
            statistical_controls=statistical_controls,
        )

    @property
    def is_extended_sweep(self) -> bool:
        return self.research_sweep.has_content

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "strategy_name": self.strategy_name,
            "ranking_metric": self.ranking_metric,
            "higher_is_better": self.higher_is_better,
            "stability": self.stability.to_dict(),
            "thresholds": {
                threshold.metric: {
                    key: value
                    for key, value in (("min", threshold.min_value), ("max", threshold.max_value))
                    if value is not None
                }
                for threshold in self.thresholds
            },
            "ranking": self.ranking.to_dict(),
            "statistical_controls": self.statistical_controls.to_dict(),
        }
        if self.sweep:
            payload["sweep"] = [definition.to_dict() for definition in self.sweep]
        else:
            payload["sweep"] = self.research_sweep.to_dict()
        return payload

    def resolve_strategy_name(self, cli_strategy_name: str | None) -> str:
        if self.is_extended_sweep:
            configured_names = self.research_sweep.strategy.names
            if cli_strategy_name is not None:
                normalized_cli = _require_non_empty_string(cli_strategy_name, field_name="strategy_name")
                if configured_names and normalized_cli not in configured_names:
                    raise ValueError(
                        "Robustness strategy mismatch: CLI requested "
                        f"'{normalized_cli}' but config targets {list(configured_names)!r}."
                    )
                return normalized_cli
            if self.strategy_name is not None:
                return self.strategy_name
            if len(configured_names) == 1:
                return configured_names[0]
            raise ValueError(
                "Extended robustness runs with multiple strategies must resolve strategy selection from the sweep itself."
            )

        if cli_strategy_name and self.strategy_name and cli_strategy_name != self.strategy_name:
            raise ValueError(
                f"Robustness strategy mismatch: CLI requested '{cli_strategy_name}' but config targets "
                f"'{self.strategy_name}'."
            )
        resolved = cli_strategy_name or self.strategy_name
        if resolved is None:
            raise ValueError("Robustness runs require a strategy_name in the config or via --strategy.")
        return resolved


def load_robustness_config(path: Path = ROBUSTNESS_CONFIG) -> RobustnessConfig:
    """Load a robustness configuration from YAML."""

    payload = load_yaml_config(path)
    if not isinstance(payload, dict):
        raise ValueError("Robustness configuration file must contain a top-level mapping.")

    robustness_payload = payload.get("robustness", payload)
    if not isinstance(robustness_payload, dict):
        raise ValueError("Robustness configuration must define a 'robustness' mapping.")

    return RobustnessConfig.from_mapping(robustness_payload)
