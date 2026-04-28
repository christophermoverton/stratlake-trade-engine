from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.research.registry import canonicalize_value


class RegimeAwareCandidateSelectionConfigError(ValueError):
    """Raised when regime-aware candidate-selection config is malformed."""


_ROOT_KEYS = frozenset(
    {
        "selection_name",
        "source_review_pack",
        "source_candidate_universe",
        "regime_context",
        "selection_categories",
        "redundancy",
        "allocation_hints",
        "output_root",
    }
)
_UNIVERSE_KEYS = frozenset(
    {"alpha_registry", "strategy_registry", "portfolio_registry", "candidate_metrics_path"}
)
_REGIME_CONTEXT_KEYS = frozenset(
    {"regime_source", "allow_gmm_confidence_overlay", "min_regime_confidence", "transition_window_bars"}
)
_CATEGORY_KEYS = frozenset(
    {
        "global_performer",
        "regime_specialist",
        "transition_resilient",
        "defensive_fallback",
    }
)
_REDUNDANCY_KEYS = frozenset(
    {"enabled", "max_pairwise_correlation", "apply_within_category", "apply_across_categories"}
)
_ALLOCATION_KEYS = frozenset({"write_category_weight_hints", "default_category_budget"})
_BUDGET_KEYS = _CATEGORY_KEYS


@dataclass(frozen=True)
class SourceCandidateUniverseConfig:
    alpha_registry: str | None = None
    strategy_registry: str | None = None
    portfolio_registry: str | None = None
    candidate_metrics_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "alpha_registry": _path_or_none(self.alpha_registry),
                "strategy_registry": _path_or_none(self.strategy_registry),
                "portfolio_registry": _path_or_none(self.portfolio_registry),
                "candidate_metrics_path": _path_or_none(self.candidate_metrics_path),
            }
        )


@dataclass(frozen=True)
class RegimeContextConfig:
    regime_source: str = "calibrated_taxonomy"
    allow_gmm_confidence_overlay: bool = True
    min_regime_confidence: float = 0.55
    transition_window_bars: int = 5

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(self.__dict__)


@dataclass(frozen=True)
class GlobalPerformerConfig:
    enabled: bool = True
    max_candidates: int = 5
    min_global_score: float = 0.60

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(self.__dict__)


@dataclass(frozen=True)
class RegimeSpecialistConfig:
    enabled: bool = True
    max_candidates_per_regime: int = 3
    min_regime_score: float = 0.65
    min_regime_observations: int = 20

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(self.__dict__)


@dataclass(frozen=True)
class TransitionResilientConfig:
    enabled: bool = True
    max_candidates: int = 5
    max_transition_drawdown: float = -0.08
    min_transition_score: float = 0.55

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(self.__dict__)


@dataclass(frozen=True)
class DefensiveFallbackConfig:
    enabled: bool = True
    max_candidates: int = 3
    max_high_vol_drawdown: float = -0.12
    max_correlation_to_selected: float = 0.70

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(self.__dict__)


@dataclass(frozen=True)
class SelectionCategoriesConfig:
    global_performer: GlobalPerformerConfig = field(default_factory=GlobalPerformerConfig)
    regime_specialist: RegimeSpecialistConfig = field(default_factory=RegimeSpecialistConfig)
    transition_resilient: TransitionResilientConfig = field(default_factory=TransitionResilientConfig)
    defensive_fallback: DefensiveFallbackConfig = field(default_factory=DefensiveFallbackConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_performer": self.global_performer.to_dict(),
            "regime_specialist": self.regime_specialist.to_dict(),
            "transition_resilient": self.transition_resilient.to_dict(),
            "defensive_fallback": self.defensive_fallback.to_dict(),
        }


@dataclass(frozen=True)
class RedundancyConfig:
    enabled: bool = True
    max_pairwise_correlation: float = 0.85
    apply_within_category: bool = True
    apply_across_categories: bool = True

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(self.__dict__)


@dataclass(frozen=True)
class AllocationHintsConfig:
    write_category_weight_hints: bool = True
    default_category_budget: dict[str, float] = field(
        default_factory=lambda: {
            "global_performer": 0.45,
            "regime_specialist": 0.30,
            "transition_resilient": 0.15,
            "defensive_fallback": 0.10,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "write_category_weight_hints": self.write_category_weight_hints,
                "default_category_budget": self.default_category_budget,
            }
        )


@dataclass(frozen=True)
class RegimeAwareCandidateSelectionConfig:
    selection_name: str
    source_review_pack: str
    source_candidate_universe: SourceCandidateUniverseConfig
    regime_context: RegimeContextConfig = field(default_factory=RegimeContextConfig)
    selection_categories: SelectionCategoriesConfig = field(default_factory=SelectionCategoriesConfig)
    redundancy: RedundancyConfig = field(default_factory=RedundancyConfig)
    allocation_hints: AllocationHintsConfig = field(default_factory=AllocationHintsConfig)
    output_root: str = "artifacts/candidate_selection"

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "selection_name": self.selection_name,
                "source_review_pack": Path(self.source_review_pack).as_posix(),
                "source_candidate_universe": self.source_candidate_universe.to_dict(),
                "regime_context": self.regime_context.to_dict(),
                "selection_categories": self.selection_categories.to_dict(),
                "redundancy": self.redundancy.to_dict(),
                "allocation_hints": self.allocation_hints.to_dict(),
                "output_root": Path(self.output_root).as_posix(),
            }
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RegimeAwareCandidateSelectionConfig":
        unknown = sorted(set(payload) - _ROOT_KEYS)
        if unknown:
            raise RegimeAwareCandidateSelectionConfigError(
                f"Regime-aware candidate-selection config contains unsupported keys: {unknown}."
            )
        return cls(
            selection_name=_required_string(payload.get("selection_name"), "selection_name"),
            source_review_pack=_path_string(payload.get("source_review_pack"), "source_review_pack"),
            source_candidate_universe=_resolve_universe(payload.get("source_candidate_universe")),
            regime_context=_resolve_regime_context(payload.get("regime_context")),
            selection_categories=_resolve_categories(payload.get("selection_categories")),
            redundancy=_resolve_redundancy(payload.get("redundancy")),
            allocation_hints=_resolve_allocation(payload.get("allocation_hints")),
            output_root=_path_string(payload.get("output_root", "artifacts/candidate_selection"), "output_root"),
        )


def load_regime_aware_candidate_selection_config(path: Path) -> RegimeAwareCandidateSelectionConfig:
    if not path.exists():
        raise RegimeAwareCandidateSelectionConfigError(
            f"Regime-aware candidate-selection config does not exist: {path.as_posix()}"
        )
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            payload = json.load(handle)
        elif path.suffix.lower() in {".yml", ".yaml"}:
            payload = yaml.safe_load(handle) or {}
        else:
            raise RegimeAwareCandidateSelectionConfigError(
                f"Unsupported regime-aware candidate-selection config format {path.suffix!r}. Use JSON, YAML, or YML."
            )
    if not isinstance(payload, Mapping):
        raise RegimeAwareCandidateSelectionConfigError(
            "Regime-aware candidate-selection config must contain a top-level mapping."
        )
    return RegimeAwareCandidateSelectionConfig.from_mapping(payload)


def apply_regime_aware_candidate_selection_overrides(
    config: RegimeAwareCandidateSelectionConfig,
    *,
    source_review_pack: str | Path | None = None,
    candidate_metrics_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> RegimeAwareCandidateSelectionConfig:
    universe = config.source_candidate_universe
    if candidate_metrics_path is not None:
        universe = SourceCandidateUniverseConfig(
            alpha_registry=universe.alpha_registry,
            strategy_registry=universe.strategy_registry,
            portfolio_registry=universe.portfolio_registry,
            candidate_metrics_path=Path(candidate_metrics_path).as_posix(),
        )
    return RegimeAwareCandidateSelectionConfig(
        selection_name=config.selection_name,
        source_review_pack=Path(source_review_pack or config.source_review_pack).as_posix(),
        source_candidate_universe=universe,
        regime_context=config.regime_context,
        selection_categories=config.selection_categories,
        redundancy=config.redundancy,
        allocation_hints=config.allocation_hints,
        output_root=Path(output_root or config.output_root).as_posix(),
    )


def _resolve_universe(value: Any) -> SourceCandidateUniverseConfig:
    if not isinstance(value, Mapping):
        raise RegimeAwareCandidateSelectionConfigError(
            "Regime-aware candidate-selection field 'source_candidate_universe' must be a mapping."
        )
    _reject_unknown(value, _UNIVERSE_KEYS, "source_candidate_universe")
    config = SourceCandidateUniverseConfig(
        alpha_registry=_optional_path_string(value.get("alpha_registry"), "source_candidate_universe.alpha_registry"),
        strategy_registry=_optional_path_string(value.get("strategy_registry"), "source_candidate_universe.strategy_registry"),
        portfolio_registry=_optional_path_string(value.get("portfolio_registry"), "source_candidate_universe.portfolio_registry"),
        candidate_metrics_path=_optional_path_string(value.get("candidate_metrics_path"), "source_candidate_universe.candidate_metrics_path"),
    )
    if config.candidate_metrics_path is None and not any(
        (config.alpha_registry, config.strategy_registry, config.portfolio_registry)
    ):
        raise RegimeAwareCandidateSelectionConfigError(
            "Regime-aware candidate-selection requires candidate_metrics_path or at least one registry path."
        )
    return config


def _resolve_regime_context(value: Any) -> RegimeContextConfig:
    if value is None:
        return RegimeContextConfig()
    if not isinstance(value, Mapping):
        raise RegimeAwareCandidateSelectionConfigError("Field 'regime_context' must be a mapping.")
    _reject_unknown(value, _REGIME_CONTEXT_KEYS, "regime_context")
    return RegimeContextConfig(
        regime_source=_required_string(value.get("regime_source", "calibrated_taxonomy"), "regime_context.regime_source"),
        allow_gmm_confidence_overlay=_bool(value.get("allow_gmm_confidence_overlay", True), "regime_context.allow_gmm_confidence_overlay"),
        min_regime_confidence=_score(value.get("min_regime_confidence", 0.55), "regime_context.min_regime_confidence"),
        transition_window_bars=_positive_int(value.get("transition_window_bars", 5), "regime_context.transition_window_bars"),
    )


def _resolve_categories(value: Any) -> SelectionCategoriesConfig:
    if value is None:
        return SelectionCategoriesConfig()
    if not isinstance(value, Mapping):
        raise RegimeAwareCandidateSelectionConfigError("Field 'selection_categories' must be a mapping.")
    _reject_unknown(value, _CATEGORY_KEYS, "selection_categories")
    return SelectionCategoriesConfig(
        global_performer=_global_category(value.get("global_performer")),
        regime_specialist=_regime_category(value.get("regime_specialist")),
        transition_resilient=_transition_category(value.get("transition_resilient")),
        defensive_fallback=_defensive_category(value.get("defensive_fallback")),
    )


def _global_category(value: Any) -> GlobalPerformerConfig:
    data = _category_mapping(value, "selection_categories.global_performer")
    return GlobalPerformerConfig(
        enabled=_bool(data.get("enabled", True), "selection_categories.global_performer.enabled"),
        max_candidates=_nonnegative_int(data.get("max_candidates", 5), "selection_categories.global_performer.max_candidates"),
        min_global_score=_score(data.get("min_global_score", 0.60), "selection_categories.global_performer.min_global_score"),
    )


def _regime_category(value: Any) -> RegimeSpecialistConfig:
    data = _category_mapping(value, "selection_categories.regime_specialist")
    return RegimeSpecialistConfig(
        enabled=_bool(data.get("enabled", True), "selection_categories.regime_specialist.enabled"),
        max_candidates_per_regime=_nonnegative_int(data.get("max_candidates_per_regime", 3), "selection_categories.regime_specialist.max_candidates_per_regime"),
        min_regime_score=_score(data.get("min_regime_score", 0.65), "selection_categories.regime_specialist.min_regime_score"),
        min_regime_observations=_nonnegative_int(data.get("min_regime_observations", 20), "selection_categories.regime_specialist.min_regime_observations"),
    )


def _transition_category(value: Any) -> TransitionResilientConfig:
    data = _category_mapping(value, "selection_categories.transition_resilient")
    return TransitionResilientConfig(
        enabled=_bool(data.get("enabled", True), "selection_categories.transition_resilient.enabled"),
        max_candidates=_nonnegative_int(data.get("max_candidates", 5), "selection_categories.transition_resilient.max_candidates"),
        max_transition_drawdown=_float(data.get("max_transition_drawdown", -0.08), "selection_categories.transition_resilient.max_transition_drawdown"),
        min_transition_score=_score(data.get("min_transition_score", 0.55), "selection_categories.transition_resilient.min_transition_score"),
    )


def _defensive_category(value: Any) -> DefensiveFallbackConfig:
    data = _category_mapping(value, "selection_categories.defensive_fallback")
    return DefensiveFallbackConfig(
        enabled=_bool(data.get("enabled", True), "selection_categories.defensive_fallback.enabled"),
        max_candidates=_nonnegative_int(data.get("max_candidates", 3), "selection_categories.defensive_fallback.max_candidates"),
        max_high_vol_drawdown=_float(data.get("max_high_vol_drawdown", -0.12), "selection_categories.defensive_fallback.max_high_vol_drawdown"),
        max_correlation_to_selected=_score(data.get("max_correlation_to_selected", 0.70), "selection_categories.defensive_fallback.max_correlation_to_selected"),
    )


def _resolve_redundancy(value: Any) -> RedundancyConfig:
    if value is None:
        return RedundancyConfig()
    if not isinstance(value, Mapping):
        raise RegimeAwareCandidateSelectionConfigError("Field 'redundancy' must be a mapping.")
    _reject_unknown(value, _REDUNDANCY_KEYS, "redundancy")
    return RedundancyConfig(
        enabled=_bool(value.get("enabled", True), "redundancy.enabled"),
        max_pairwise_correlation=_score(value.get("max_pairwise_correlation", 0.85), "redundancy.max_pairwise_correlation"),
        apply_within_category=_bool(value.get("apply_within_category", True), "redundancy.apply_within_category"),
        apply_across_categories=_bool(value.get("apply_across_categories", True), "redundancy.apply_across_categories"),
    )


def _resolve_allocation(value: Any) -> AllocationHintsConfig:
    if value is None:
        return AllocationHintsConfig()
    if not isinstance(value, Mapping):
        raise RegimeAwareCandidateSelectionConfigError("Field 'allocation_hints' must be a mapping.")
    _reject_unknown(value, _ALLOCATION_KEYS, "allocation_hints")
    budgets = value.get("default_category_budget", AllocationHintsConfig().default_category_budget)
    if not isinstance(budgets, Mapping):
        raise RegimeAwareCandidateSelectionConfigError("Field 'allocation_hints.default_category_budget' must be a mapping.")
    _reject_unknown(budgets, _BUDGET_KEYS, "allocation_hints.default_category_budget")
    resolved = {key: _score(budgets.get(key, 0.0), f"allocation_hints.default_category_budget.{key}") for key in _BUDGET_KEYS}
    total = sum(resolved.values())
    if abs(total - 1.0) > 1e-9:
        raise RegimeAwareCandidateSelectionConfigError(
            "Field 'allocation_hints.default_category_budget' must sum to 1.0."
        )
    return AllocationHintsConfig(
        write_category_weight_hints=_bool(value.get("write_category_weight_hints", True), "allocation_hints.write_category_weight_hints"),
        default_category_budget=resolved,
    )


def _category_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RegimeAwareCandidateSelectionConfigError(f"Field '{field_name}' must be a mapping.")
    return value


def _reject_unknown(value: Mapping[str, Any], supported: frozenset[str], field_name: str) -> None:
    unknown = sorted(set(value) - supported)
    if unknown:
        raise RegimeAwareCandidateSelectionConfigError(f"Field '{field_name}' contains unsupported keys: {unknown}.")


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegimeAwareCandidateSelectionConfigError(f"Field '{field_name}' must be a non-empty string.")
    return value.strip()


def _path_string(value: Any, field_name: str) -> str:
    return Path(_required_string(value, field_name)).as_posix()


def _optional_path_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _path_string(value, field_name)


def _path_or_none(value: str | None) -> str | None:
    return None if value is None else Path(value).as_posix()


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise RegimeAwareCandidateSelectionConfigError(f"Field '{field_name}' must be boolean.")
    return value


def _float(value: Any, field_name: str) -> float:
    if not isinstance(value, int | float):
        raise RegimeAwareCandidateSelectionConfigError(f"Field '{field_name}' must be numeric.")
    return float(value)


def _score(value: Any, field_name: str) -> float:
    number = _float(value, field_name)
    if number < 0.0 or number > 1.0:
        raise RegimeAwareCandidateSelectionConfigError(f"Field '{field_name}' must be between 0.0 and 1.0.")
    return number


def _positive_int(value: Any, field_name: str) -> int:
    number = _nonnegative_int(value, field_name)
    if number <= 0:
        raise RegimeAwareCandidateSelectionConfigError(f"Field '{field_name}' must be positive.")
    return number


def _nonnegative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise RegimeAwareCandidateSelectionConfigError(f"Field '{field_name}' must be a non-negative integer.")
    return int(value)


__all__ = [
    "AllocationHintsConfig",
    "DefensiveFallbackConfig",
    "GlobalPerformerConfig",
    "RegimeAwareCandidateSelectionConfig",
    "RegimeAwareCandidateSelectionConfigError",
    "RegimeContextConfig",
    "RegimeSpecialistConfig",
    "RedundancyConfig",
    "SelectionCategoriesConfig",
    "SourceCandidateUniverseConfig",
    "TransitionResilientConfig",
    "apply_regime_aware_candidate_selection_overrides",
    "load_regime_aware_candidate_selection_config",
]
