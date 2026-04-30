from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.research.registry import canonicalize_value, serialize_canonical_json


class RegimePolicyStressTestConfigError(ValueError):
    """Raised when a regime policy stress-test configuration is malformed."""


_ROOT_KEYS = frozenset(
    {
        "stress_test_name",
        "source_review_pack",
        "source_policy_candidates",
        "regime_context",
        "scenarios",
        "metrics",
        "stress_gates",
        "output_root",
        "metadata",
        "source_candidate_selection",
        "market_simulation_stress",
    }
)
_MARKET_SIMULATION_STRESS_KEYS = frozenset(
    {
        "enabled",
        "mode",
        "simulation_metrics_dir",
        "config_path",
        "include_in_policy_stress_summary",
        "include_in_case_study_report",
    }
)
_SOURCE_POLICY_KEYS = frozenset(
    {
        "policy_metrics_path",
        "baseline_policy",
        "candidate_policy_names",
    }
)
_REGIME_CONTEXT_KEYS = frozenset(
    {
        "preferred_regime_source",
        "ml_overlay",
        "transition_window_bars",
        "confidence_floor",
        "entropy_ceiling",
    }
)
_METRIC_KEYS = frozenset(
    {
        "include_policy_turnover",
        "include_drawdown",
        "include_transition_windows",
        "include_fallback_usage",
        "include_adaptive_vs_static",
    }
)
_STRESS_GATE_KEYS = frozenset(
    {
        "max_policy_turnover",
        "max_stress_drawdown",
        "min_adaptive_vs_static_drawdown_delta",
        "max_fallback_activation_rate",
        "max_state_change_count",
    }
)
_SCENARIO_COMMON_KEYS = frozenset({"scenario_id", "name", "type", "stress_intensity"})
_SCENARIO_TYPE_KEYS: dict[str, frozenset[str]] = {
    "transition_shock": frozenset({"from_regime", "to_regime", "shock_start", "shock_length_bars"}),
    "regime_whipsaw": frozenset({"regimes", "cycle_length_bars", "cycles"}),
    "high_vol_persistence": frozenset({"target_regime", "persistence_length_bars"}),
    "classifier_uncertainty": frozenset({"confidence_multiplier", "entropy_multiplier"}),
    "taxonomy_ml_disagreement": frozenset({"disagreement_rate", "conflict_resolution_modes"}),
    "confidence_collapse": frozenset({"confidence_floor_override", "collapse_length_bars"}),
}


@dataclass(frozen=True)
class RegimePolicyStressCandidateSourceConfig:
    policy_metrics_path: str
    baseline_policy: str
    candidate_policy_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_metrics_path": Path(self.policy_metrics_path).as_posix(),
            "baseline_policy": self.baseline_policy,
            "candidate_policy_names": list(self.candidate_policy_names),
        }


@dataclass(frozen=True)
class RegimePolicyStressRegimeContextConfig:
    preferred_regime_source: str = "calibrated_taxonomy"
    ml_overlay: str | None = "gmm_classifier"
    transition_window_bars: int = 5
    confidence_floor: float = 0.55
    entropy_ceiling: float = 1.10

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(self.__dict__)


@dataclass(frozen=True)
class RegimePolicyStressMetricsConfig:
    include_policy_turnover: bool = True
    include_drawdown: bool = True
    include_transition_windows: bool = True
    include_fallback_usage: bool = True
    include_adaptive_vs_static: bool = True

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(self.__dict__)


@dataclass(frozen=True)
class RegimePolicyStressGatesConfig:
    max_policy_turnover: float | None = 0.60
    max_stress_drawdown: float | None = -0.18
    min_adaptive_vs_static_drawdown_delta: float | None = -0.03
    max_fallback_activation_rate: float | None = 0.75
    max_state_change_count: float | None = 40.0

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(self.__dict__)


@dataclass(frozen=True)
class RegimePolicyStressScenarioConfig:
    scenario_id: str
    scenario_name: str
    scenario_type: str
    stress_intensity: float
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "scenario_id": self.scenario_id,
            "name": self.scenario_name,
            "type": self.scenario_type,
            "stress_intensity": self.stress_intensity,
        }
        payload.update(self.parameters)
        return canonicalize_value(payload)


@dataclass(frozen=True)
class RegimePolicyMarketSimulationStressConfig:
    enabled: bool = False
    mode: str = "existing_artifacts"
    simulation_metrics_dir: str | None = None
    config_path: str | None = None
    include_in_policy_stress_summary: bool = True
    include_in_case_study_report: bool = True

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "enabled": self.enabled,
                "mode": self.mode,
                "simulation_metrics_dir": _path_or_none(self.simulation_metrics_dir),
                "config_path": _path_or_none(self.config_path),
                "include_in_policy_stress_summary": self.include_in_policy_stress_summary,
                "include_in_case_study_report": self.include_in_case_study_report,
            }
        )


@dataclass(frozen=True)
class RegimePolicyStressTestConfig:
    stress_test_name: str
    source_review_pack: str | None = None
    source_policy_candidates: RegimePolicyStressCandidateSourceConfig | None = None
    regime_context: RegimePolicyStressRegimeContextConfig = field(default_factory=RegimePolicyStressRegimeContextConfig)
    scenarios: tuple[RegimePolicyStressScenarioConfig, ...] = ()
    metrics: RegimePolicyStressMetricsConfig = field(default_factory=RegimePolicyStressMetricsConfig)
    stress_gates: RegimePolicyStressGatesConfig = field(default_factory=RegimePolicyStressGatesConfig)
    output_root: str = "artifacts/regime_stress_tests"
    metadata: dict[str, Any] = field(default_factory=dict)
    source_candidate_selection: str | None = None
    market_simulation_stress: RegimePolicyMarketSimulationStressConfig = field(
        default_factory=RegimePolicyMarketSimulationStressConfig
    )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "stress_test_name": self.stress_test_name,
            "source_review_pack": _path_or_none(self.source_review_pack),
            "source_policy_candidates": None
            if self.source_policy_candidates is None
            else self.source_policy_candidates.to_dict(),
            "regime_context": self.regime_context.to_dict(),
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "metrics": self.metrics.to_dict(),
            "stress_gates": self.stress_gates.to_dict(),
            "output_root": Path(self.output_root).as_posix(),
            "metadata": dict(self.metadata),
            "source_candidate_selection": _path_or_none(self.source_candidate_selection),
        }
        if self.market_simulation_stress.enabled:
            payload["market_simulation_stress"] = self.market_simulation_stress.to_dict()
        return canonicalize_value(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RegimePolicyStressTestConfig":
        unknown = sorted(set(payload) - _ROOT_KEYS)
        if unknown:
            raise RegimePolicyStressTestConfigError(
                f"Regime policy stress-test config contains unsupported keys: {unknown}."
            )
        source_review_pack = _optional_path_string(payload.get("source_review_pack"), "source_review_pack")
        source_policy_candidates = _resolve_source_policy_candidates(payload.get("source_policy_candidates"))
        if source_review_pack is None and source_policy_candidates is None:
            raise RegimePolicyStressTestConfigError(
                "Regime policy stress-test config requires source_review_pack or source_policy_candidates."
            )
        scenarios = _resolve_scenarios(payload.get("scenarios"))
        if not scenarios:
            raise RegimePolicyStressTestConfigError(
                "Regime policy stress-test config requires at least one scenario."
            )
        return cls(
            stress_test_name=_required_string(payload.get("stress_test_name"), "stress_test_name"),
            source_review_pack=source_review_pack,
            source_policy_candidates=source_policy_candidates,
            regime_context=_resolve_regime_context(payload.get("regime_context")),
            scenarios=scenarios,
            metrics=_resolve_metrics(payload.get("metrics")),
            stress_gates=_resolve_stress_gates(payload.get("stress_gates")),
            output_root=_path_string(payload.get("output_root", "artifacts/regime_stress_tests"), "output_root"),
            metadata=_resolve_optional_mapping(payload.get("metadata"), field_name="metadata"),
            source_candidate_selection=_optional_path_string(
                payload.get("source_candidate_selection"), "source_candidate_selection"
            ),
            market_simulation_stress=_resolve_market_simulation_stress(
                payload.get("market_simulation_stress")
            ),
        )


def load_regime_policy_stress_test_config(path: Path) -> RegimePolicyStressTestConfig:
    if not path.exists():
        raise RegimePolicyStressTestConfigError(
            f"Regime policy stress-test config does not exist: {path.as_posix()}"
        )
    with path.open("r", encoding="utf-8") as handle:
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {".yml", ".yaml"}:
            payload = yaml.safe_load(handle) or {}
        else:
            raise RegimePolicyStressTestConfigError(
                f"Unsupported regime policy stress-test config format {path.suffix!r}. Use JSON, YAML, or YML."
            )
    if not isinstance(payload, Mapping):
        raise RegimePolicyStressTestConfigError(
            "Regime policy stress-test config must contain a top-level mapping."
        )
    return RegimePolicyStressTestConfig.from_mapping(payload)


def apply_regime_policy_stress_test_overrides(
    config: RegimePolicyStressTestConfig,
    *,
    source_review_pack: str | Path | None = None,
    policy_metrics_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> RegimePolicyStressTestConfig:
    source_policy_candidates = config.source_policy_candidates
    if policy_metrics_path is not None:
        if source_policy_candidates is None:
            raise RegimePolicyStressTestConfigError(
                "Cannot override policy_metrics_path when source_policy_candidates is not configured."
            )
        source_policy_candidates = RegimePolicyStressCandidateSourceConfig(
            policy_metrics_path=Path(policy_metrics_path).as_posix(),
            baseline_policy=source_policy_candidates.baseline_policy,
            candidate_policy_names=source_policy_candidates.candidate_policy_names,
        )
    return RegimePolicyStressTestConfig(
        stress_test_name=config.stress_test_name,
        source_review_pack=Path(source_review_pack).as_posix() if source_review_pack is not None else config.source_review_pack,
        source_policy_candidates=source_policy_candidates,
        regime_context=config.regime_context,
        scenarios=config.scenarios,
        metrics=config.metrics,
        stress_gates=config.stress_gates,
        output_root=Path(output_root).as_posix() if output_root is not None else config.output_root,
        metadata=config.metadata,
        source_candidate_selection=config.source_candidate_selection,
        market_simulation_stress=config.market_simulation_stress,
    )


def deterministic_scenario_id(scenario_name: str, scenario_type: str, parameters: Mapping[str, Any]) -> str:
    payload = {
        "name": scenario_name,
        "type": scenario_type,
        "parameters": canonicalize_value(dict(parameters)),
    }
    digest = hashlib.sha256(serialize_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"{_slugify(scenario_name)}_{digest}"


def _resolve_source_policy_candidates(value: Any) -> RegimePolicyStressCandidateSourceConfig | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RegimePolicyStressTestConfigError(
            "Regime policy stress-test field 'source_policy_candidates' must be a mapping."
        )
    unknown = sorted(set(value) - _SOURCE_POLICY_KEYS)
    if unknown:
        raise RegimePolicyStressTestConfigError(
            "Regime policy stress-test field 'source_policy_candidates' contains unsupported keys: "
            f"{unknown}."
        )
    names = _optional_string_sequence(value.get("candidate_policy_names"), "source_policy_candidates.candidate_policy_names")
    return RegimePolicyStressCandidateSourceConfig(
        policy_metrics_path=_path_string(value.get("policy_metrics_path"), "source_policy_candidates.policy_metrics_path"),
        baseline_policy=_required_string(value.get("baseline_policy"), "source_policy_candidates.baseline_policy"),
        candidate_policy_names=names,
    )


def _resolve_regime_context(value: Any) -> RegimePolicyStressRegimeContextConfig:
    if value is None:
        return RegimePolicyStressRegimeContextConfig()
    if not isinstance(value, Mapping):
        raise RegimePolicyStressTestConfigError("Regime policy stress-test field 'regime_context' must be a mapping.")
    unknown = sorted(set(value) - _REGIME_CONTEXT_KEYS)
    if unknown:
        raise RegimePolicyStressTestConfigError(
            f"Regime policy stress-test field 'regime_context' contains unsupported keys: {unknown}."
        )
    return RegimePolicyStressRegimeContextConfig(
        preferred_regime_source=_required_string(
            value.get("preferred_regime_source", "calibrated_taxonomy"),
            "regime_context.preferred_regime_source",
        ),
        ml_overlay=_optional_string(value.get("ml_overlay", "gmm_classifier"), "regime_context.ml_overlay"),
        transition_window_bars=_positive_int(value.get("transition_window_bars", 5), "regime_context.transition_window_bars"),
        confidence_floor=_score(value.get("confidence_floor", 0.55), "regime_context.confidence_floor"),
        entropy_ceiling=_nonnegative_float(value.get("entropy_ceiling", 1.10), "regime_context.entropy_ceiling"),
    )


def _resolve_metrics(value: Any) -> RegimePolicyStressMetricsConfig:
    if value is None:
        return RegimePolicyStressMetricsConfig()
    if not isinstance(value, Mapping):
        raise RegimePolicyStressTestConfigError("Regime policy stress-test field 'metrics' must be a mapping.")
    unknown = sorted(set(value) - _METRIC_KEYS)
    if unknown:
        raise RegimePolicyStressTestConfigError(
            f"Regime policy stress-test field 'metrics' contains unsupported keys: {unknown}."
        )
    return RegimePolicyStressMetricsConfig(
        include_policy_turnover=_bool(value.get("include_policy_turnover", True), "metrics.include_policy_turnover"),
        include_drawdown=_bool(value.get("include_drawdown", True), "metrics.include_drawdown"),
        include_transition_windows=_bool(value.get("include_transition_windows", True), "metrics.include_transition_windows"),
        include_fallback_usage=_bool(value.get("include_fallback_usage", True), "metrics.include_fallback_usage"),
        include_adaptive_vs_static=_bool(value.get("include_adaptive_vs_static", True), "metrics.include_adaptive_vs_static"),
    )


def _resolve_stress_gates(value: Any) -> RegimePolicyStressGatesConfig:
    if value is None:
        return RegimePolicyStressGatesConfig()
    if not isinstance(value, Mapping):
        raise RegimePolicyStressTestConfigError("Regime policy stress-test field 'stress_gates' must be a mapping.")
    unknown = sorted(set(value) - _STRESS_GATE_KEYS)
    if unknown:
        raise RegimePolicyStressTestConfigError(
            f"Regime policy stress-test field 'stress_gates' contains unsupported keys: {unknown}."
        )

    max_policy_turnover = _optional_score(value.get("max_policy_turnover"), "stress_gates.max_policy_turnover")
    max_stress_drawdown = _optional_float(value.get("max_stress_drawdown"), "stress_gates.max_stress_drawdown")
    min_dd_delta = _optional_float(
        value.get("min_adaptive_vs_static_drawdown_delta"),
        "stress_gates.min_adaptive_vs_static_drawdown_delta",
    )
    max_fallback_rate = _optional_score(
        value.get("max_fallback_activation_rate"),
        "stress_gates.max_fallback_activation_rate",
    )
    max_state_changes = _optional_nonnegative_float(
        value.get("max_state_change_count"),
        "stress_gates.max_state_change_count",
    )

    if max_stress_drawdown is not None and max_stress_drawdown > 0.0:
        raise RegimePolicyStressTestConfigError(
            "Regime policy stress-test field 'stress_gates.max_stress_drawdown' must be less than or equal to 0."
        )

    return RegimePolicyStressGatesConfig(
        max_policy_turnover=max_policy_turnover,
        max_stress_drawdown=max_stress_drawdown,
        min_adaptive_vs_static_drawdown_delta=min_dd_delta,
        max_fallback_activation_rate=max_fallback_rate,
        max_state_change_count=max_state_changes,
    )


def _resolve_scenarios(value: Any) -> tuple[RegimePolicyStressScenarioConfig, ...]:
    if not isinstance(value, list):
        raise RegimePolicyStressTestConfigError("Regime policy stress-test field 'scenarios' must be a sequence.")
    scenarios: list[RegimePolicyStressScenarioConfig] = []
    names_seen: set[str] = set()
    ids_seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise RegimePolicyStressTestConfigError(
                f"Regime policy stress-test scenario at index {index} must be a mapping."
            )
        scenario_type = _required_string(item.get("type"), f"scenarios[{index}].type")
        if scenario_type not in _SCENARIO_TYPE_KEYS:
            expected = ", ".join(sorted(_SCENARIO_TYPE_KEYS))
            raise RegimePolicyStressTestConfigError(
                f"Regime policy stress-test field 'scenarios[{index}].type' must be one of: {expected}."
            )
        supported = _SCENARIO_COMMON_KEYS | _SCENARIO_TYPE_KEYS[scenario_type]
        unknown = sorted(set(item) - supported)
        if unknown:
            raise RegimePolicyStressTestConfigError(
                f"Regime policy stress-test scenario {index} contains unsupported keys: {unknown}."
            )
        scenario_name = _required_string(item.get("name"), f"scenarios[{index}].name")
        if scenario_name in names_seen:
            raise RegimePolicyStressTestConfigError(
                f"Regime policy stress-test scenario names must be unique. Duplicate: {scenario_name!r}."
            )
        names_seen.add(scenario_name)
        params = _normalize_scenario_params(item, scenario_type=scenario_type, index=index)
        scenario_id = deterministic_scenario_id(scenario_name, scenario_type, params)
        provided_scenario_id = _optional_string(item.get("scenario_id"), f"scenarios[{index}].scenario_id")
        if provided_scenario_id is not None and provided_scenario_id != scenario_id:
            raise RegimePolicyStressTestConfigError(
                f"Regime policy stress-test field 'scenarios[{index}].scenario_id' does not match deterministic scenario id {scenario_id!r}."
            )
        if scenario_id in ids_seen:
            raise RegimePolicyStressTestConfigError(
                "Regime policy stress-test produced duplicate scenario IDs; adjust scenario definitions to avoid collisions."
            )
        ids_seen.add(scenario_id)
        scenarios.append(
            RegimePolicyStressScenarioConfig(
                scenario_id=scenario_id,
                scenario_name=scenario_name,
                scenario_type=scenario_type,
                stress_intensity=_nonnegative_float(item.get("stress_intensity", 1.0), f"scenarios[{index}].stress_intensity"),
                parameters=params,
            )
        )
    return tuple(scenarios)


def _resolve_market_simulation_stress(value: Any) -> RegimePolicyMarketSimulationStressConfig:
    if value is None:
        return RegimePolicyMarketSimulationStressConfig()
    if not isinstance(value, Mapping):
        raise RegimePolicyStressTestConfigError(
            "Regime policy stress-test field 'market_simulation_stress' must be a mapping."
        )
    unknown = sorted(set(value) - _MARKET_SIMULATION_STRESS_KEYS)
    if unknown:
        raise RegimePolicyStressTestConfigError(
            "Regime policy stress-test field 'market_simulation_stress' contains unsupported keys: "
            f"{unknown}."
        )
    enabled = _bool(value.get("enabled", False), "market_simulation_stress.enabled")
    mode = _required_string(value.get("mode", "existing_artifacts"), "market_simulation_stress.mode")
    if mode not in {"existing_artifacts", "run_config"}:
        raise RegimePolicyStressTestConfigError(
            "Regime policy stress-test field 'market_simulation_stress.mode' must be "
            "'existing_artifacts' or 'run_config'."
        )
    simulation_metrics_dir = _optional_path_string(
        value.get("simulation_metrics_dir"), "market_simulation_stress.simulation_metrics_dir"
    )
    config_path = _optional_path_string(value.get("config_path"), "market_simulation_stress.config_path")
    if enabled and mode == "existing_artifacts" and simulation_metrics_dir is None:
        raise RegimePolicyStressTestConfigError(
            "Enabled market_simulation_stress existing_artifacts mode requires simulation_metrics_dir."
        )
    if enabled and mode == "run_config" and config_path is None:
        raise RegimePolicyStressTestConfigError(
            "Enabled market_simulation_stress run_config mode requires config_path."
        )
    return RegimePolicyMarketSimulationStressConfig(
        enabled=enabled,
        mode=mode,
        simulation_metrics_dir=simulation_metrics_dir,
        config_path=config_path,
        include_in_policy_stress_summary=_bool(
            value.get("include_in_policy_stress_summary", True),
            "market_simulation_stress.include_in_policy_stress_summary",
        ),
        include_in_case_study_report=_bool(
            value.get("include_in_case_study_report", True),
            "market_simulation_stress.include_in_case_study_report",
        ),
    )


def _normalize_scenario_params(payload: Mapping[str, Any], *, scenario_type: str, index: int) -> dict[str, Any]:
    prefix = f"scenarios[{index}]"
    if scenario_type == "transition_shock":
        return {
            "from_regime": _required_string(payload.get("from_regime"), f"{prefix}.from_regime"),
            "to_regime": _required_string(payload.get("to_regime"), f"{prefix}.to_regime"),
            "shock_start": _required_string(payload.get("shock_start"), f"{prefix}.shock_start"),
            "shock_length_bars": _positive_int(payload.get("shock_length_bars"), f"{prefix}.shock_length_bars"),
        }
    if scenario_type == "regime_whipsaw":
        regimes = _string_sequence(payload.get("regimes"), f"{prefix}.regimes")
        if len(regimes) < 2:
            raise RegimePolicyStressTestConfigError(
                f"Regime policy stress-test field '{prefix}.regimes' must include at least two regimes."
            )
        return {
            "regimes": list(regimes),
            "cycle_length_bars": _positive_int(payload.get("cycle_length_bars"), f"{prefix}.cycle_length_bars"),
            "cycles": _positive_int(payload.get("cycles"), f"{prefix}.cycles"),
        }
    if scenario_type == "high_vol_persistence":
        return {
            "target_regime": _required_string(payload.get("target_regime"), f"{prefix}.target_regime"),
            "persistence_length_bars": _positive_int(payload.get("persistence_length_bars"), f"{prefix}.persistence_length_bars"),
        }
    if scenario_type == "classifier_uncertainty":
        return {
            "confidence_multiplier": _nonnegative_float(payload.get("confidence_multiplier"), f"{prefix}.confidence_multiplier"),
            "entropy_multiplier": _nonnegative_float(payload.get("entropy_multiplier"), f"{prefix}.entropy_multiplier"),
        }
    if scenario_type == "taxonomy_ml_disagreement":
        modes = _string_sequence(payload.get("conflict_resolution_modes"), f"{prefix}.conflict_resolution_modes")
        return {
            "disagreement_rate": _score(payload.get("disagreement_rate"), f"{prefix}.disagreement_rate"),
            "conflict_resolution_modes": list(sorted(set(modes))),
        }
    if scenario_type == "confidence_collapse":
        return {
            "confidence_floor_override": _score(payload.get("confidence_floor_override"), f"{prefix}.confidence_floor_override"),
            "collapse_length_bars": _positive_int(payload.get("collapse_length_bars"), f"{prefix}.collapse_length_bars"),
        }
    raise RegimePolicyStressTestConfigError(f"Unsupported scenario type {scenario_type!r}.")


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise RegimePolicyStressTestConfigError(
            f"Regime policy stress-test field '{field_name}' must be a non-empty string."
        )
    normalized = value.strip()
    if not normalized:
        raise RegimePolicyStressTestConfigError(
            f"Regime policy stress-test field '{field_name}' must be a non-empty string."
        )
    return normalized


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field_name)


def _optional_string_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    return _string_sequence(value, field_name)


def _string_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise RegimePolicyStressTestConfigError(
            f"Regime policy stress-test field '{field_name}' must be a sequence."
        )
    return tuple(_required_string(item, field_name) for item in value)


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise RegimePolicyStressTestConfigError(
            f"Regime policy stress-test field '{field_name}' must be boolean."
        )
    return value


def _float(value: Any, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise RegimePolicyStressTestConfigError(
            f"Regime policy stress-test field '{field_name}' must be numeric."
        )
    return float(value)


def _score(value: Any, field_name: str) -> float:
    number = _float(value, field_name)
    if number < 0.0 or number > 1.0:
        raise RegimePolicyStressTestConfigError(
            f"Regime policy stress-test field '{field_name}' must be between 0 and 1."
        )
    return number


def _optional_score(value: Any, field_name: str) -> float | None:
    return None if value is None else _score(value, field_name)


def _nonnegative_float(value: Any, field_name: str) -> float:
    number = _float(value, field_name)
    if number < 0.0:
        raise RegimePolicyStressTestConfigError(
            f"Regime policy stress-test field '{field_name}' must be non-negative."
        )
    return number


def _optional_nonnegative_float(value: Any, field_name: str) -> float | None:
    return None if value is None else _nonnegative_float(value, field_name)


def _optional_float(value: Any, field_name: str) -> float | None:
    return None if value is None else _float(value, field_name)


def _positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RegimePolicyStressTestConfigError(
            f"Regime policy stress-test field '{field_name}' must be a positive integer."
        )
    return int(value)


def _path_string(value: Any, field_name: str) -> str:
    return Path(_required_string(value, field_name)).as_posix()


def _optional_path_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _path_string(value, field_name)


def _resolve_optional_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RegimePolicyStressTestConfigError(
            f"Regime policy stress-test field '{field_name}' must be a mapping when provided."
        )
    return canonicalize_value(dict(value))


def _path_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    return Path(value).as_posix()


def _slugify(value: str) -> str:
    chars = [character.lower() if character.isalnum() else "_" for character in value.strip()]
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "scenario"


__all__ = [
    "RegimePolicyStressCandidateSourceConfig",
    "RegimePolicyStressGatesConfig",
    "RegimePolicyStressMetricsConfig",
    "RegimePolicyMarketSimulationStressConfig",
    "RegimePolicyStressRegimeContextConfig",
    "RegimePolicyStressScenarioConfig",
    "RegimePolicyStressTestConfig",
    "RegimePolicyStressTestConfigError",
    "apply_regime_policy_stress_test_overrides",
    "deterministic_scenario_id",
    "load_regime_policy_stress_test_config",
]
